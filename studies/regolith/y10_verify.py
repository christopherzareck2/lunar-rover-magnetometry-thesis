"""Y10 step 1: verify that a +-1.2 km regolith slab with a 4 x 4 m hole equals the plug (-M_reg) formulation,
and that the engine's prism kernel has the right sign/magnitude (independent point-dipole sum).
Read-only use of the thesis repo engine. Run: nice python y10_verify.py
"""
import os, sys, json
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "2")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
sys.path.insert(0, os.path.join(REPO, "analysis"))
import a09_systematics as A9                      # noqa (chdirs into engine/)
from inversion_helpers import prism_cell_operator

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "y10_verify.json")
ZG = -0.5          # ground surface in the engine frame (sensor z = 0 is 0.5 m above ground)
MU0 = 4e-7 * np.pi


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


res = {}
rng = np.random.default_rng(20261013)
for dx in (0.0, 0.5):
    sx = A9.two_loops(dx=dx)
    for h in (0.5, 1.0, 3.0, 5.0):
        z0, z1 = ZG - h, ZG
        Mreg = 0.04 * 300e-9 / MU0 * (lambda v: v / np.linalg.norm(v))(rng.normal(size=3))   # chi 0.04 in 300 nT, random dir
        # plug formulation
        Gh = prism_cell_operator((-2, 2, -2, 2, z0, z1), sx)
        plug = Gh @ (-Mreg)
        # +-1.2 km slab with the hole: the 8 prisms of a 3 x 3 grid, centre excluded
        L = 1200.0
        edges = [-L, -2.0, 2.0, L]
        slab_hole = np.zeros(plug.size)
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                slab_hole += prism_cell_operator((edges[i], edges[i + 1], edges[j], edges[j + 1], z0, z1), sx) @ Mreg
        # whole slab (no hole)
        slab = prism_cell_operator((-L, L, -L, L, z0, z1), sx) @ Mreg
        key = f"dx{dx}_h{h}"
        res[key] = {
            "Mreg_A_per_m": [float(v) for v in Mreg], "Mreg_over_Mref": float(np.linalg.norm(Mreg) / 2.4e-3),
            "rms_plug_nT": rms(plug), "rms_slab_with_hole_nT": rms(slab_hole),
            "rms_difference_nT": rms(slab_hole - plug), "rms_full_slab_nT": rms(slab),
            "max_abs_difference_nT": float(np.max(np.abs(slab_hole - plug))),
            "relative_difference": rms(slab_hole - plug) / rms(plug),
        }
        print(key, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in res[key].items() if k != "Mreg_A_per_m"}, flush=True)

# independent kernel check: point-dipole sum over a fine grid for the h = 1 m plug (sensor subset)
sx = A9.two_loops(dx=0.0)[::40]
h = 1.0
Mv = np.array([1.0, 0.5, -0.8]) * 1e-3
Gh = prism_cell_operator((-2, 2, -2, 2, ZG - h, ZG), sx)
eng = Gh @ Mv
n = 40; dz = 0.05; nzc = int(h / dz)
xs = (np.arange(n) + 0.5) * (4.0 / n) - 2.0
zs = ZG - h + (np.arange(nzc) + 0.5) * dz
X, Y, Z = np.meshgrid(xs, xs, zs, indexing="ij")
P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
vol = (4.0 / n) ** 2 * dz
m = Mv * vol                                        # A m^2 per cell
dip = np.zeros((len(sx), 3))
for k, s in enumerate(sx):
    r = s - P; d = np.linalg.norm(r, axis=1, keepdims=True); u = r / d
    B = 1e-7 * (3 * (u @ m)[:, None] * u - m) / d ** 3 * 1e9      # nT
    dip[k] = B.sum(0)
res["kernel_check_point_dipole_h1"] = {"rms_engine_nT": rms(eng), "rms_dipole_sum_nT": rms(dip.reshape(-1)),
                                       "rms_difference_nT": rms(eng - dip.reshape(-1)),
                                       "relative_difference": rms(eng - dip.reshape(-1)) / rms(eng)}
print(res["kernel_check_point_dipole_h1"], flush=True)
json.dump(res, open(OUT, "w"), indent=1)

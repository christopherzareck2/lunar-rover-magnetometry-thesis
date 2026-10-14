"""a03 - Three-layer wall: is the middle flow's poor single-traverse recovery a direction effect or a
distance effect?

Same wall, traverse, noise and solver as Table 4's single ground traverse (engine/three_layer_wall.py,
arm A): flows x -50..50, y 0..50, z 0-5/5-10/10-15 m; traverse y = -1 m, z = 0.5 m, x = -120..+120 m,
1 cm sampling; sigma_B = 0.50 nT; K = 3 closed-form (undamped) weighted least squares, as in the first
columns of Table 4 (3.7/28.1/33.4 deg for the baseline case).
N = 500 inversions per case, identical noise draws across cases. Result: the middle flow's error (~30 deg)
does not depend on its magnetization direction; it is set by its distance from the traverse.

Cases (unit directions, |M| = 2.4e-3 A/m for every flow):
  T0  baseline           L1 -z, L2 +y, L3 +z
  T1  middle along x     L1 -z, L2 +x, L3 +z
  T2  middle vertical    L1 -z, L2 -z, L3 +z
  T3  swap L1/L2 dirs    L1 +y, L2 -z, L3 +z   (puts the horizontal-into-wall vector on the nearest flow)

Run:  python analysis/a03_three_layer_direction_vs_distance.py
Out:  results/analysis/a03_three_layer_direction_vs_distance.json
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "engine")); os.chdir(os.path.join(ROOT, "engine"))
from inversion_helpers import three_layer_G, ls_operator, dir_err_deg, seed_for, M_NRM_MIN, SIGMA_B

N = 500
U = {"+x": [1, 0, 0], "+y": [0, 1, 0], "-z": [0, 0, -1], "+z": [0, 0, 1]}
CASES = {"T0_baseline": ("-z", "+y", "+z"), "T1_middle_along_x": ("-z", "+x", "+z"),
         "T2_middle_vertical": ("-z", "-z", "+z"), "T3_swap_L1_L2": ("+y", "-z", "+z")}


def to9(layers):
    L = np.asarray(layers, float)
    return np.concatenate([L[:, 0], L[:, 1], L[:, 2]])


def per_layer(m9):
    return np.stack([m9[[i, 3 + i, 6 + i]] for i in range(3)])


def main():
    t0 = time.time()
    G, sxyz = three_layer_G(-1.0, 0.5, spacing=0.01)
    K = ls_operator(G)
    rng = np.random.default_rng(seed_for("a03:noise"))
    noise = rng.normal(0.0, SIGMA_B, size=(N, G.shape[0]))
    out = {"meta": dict(script="analysis/a03_three_layer_direction_vs_distance.py", N=N, sigma_B_nT=SIGMA_B,
                        n_stations=int(len(sxyz)), solver="closed-form weighted LS, K=3"), "cases": {}}
    for name, dirs in CASES.items():
        Ml = M_NRM_MIN * np.array([U[d] for d in dirs])
        m9 = to9(Ml)
        d = G @ m9
        Mrec = (d[None, :] + noise) @ K.T
        res = {}
        for i in range(3):
            th = np.array([dir_err_deg(per_layer(r)[i], Ml[i]) for r in Mrec])
            dm = 100 * (np.linalg.norm(np.array([per_layer(r)[i] for r in Mrec]), axis=1) - M_NRM_MIN) / M_NRM_MIN
            sig = G[:, [i, 3 + i, 6 + i]] @ Ml[i]
            res[f"layer_{i+1}"] = dict(direction=dirs[i], dir_err_mean=float(th.mean()), dir_err_std=float(th.std()),
                                       dM_signed_mean=float(dm.mean()), dM_abs_median=float(np.median(np.abs(dm))),
                                       signal_rms_nT=float(np.sqrt(np.mean(sig ** 2))))
        out["cases"][name] = res
        print(name, " | ".join(f"L{i+1} {res[f'layer_{i+1}']['direction']}: {res[f'layer_{i+1}']['dir_err_mean']:.1f}+-{res[f'layer_{i+1}']['dir_err_std']:.1f} deg "
                               f"(sig {res[f'layer_{i+1}']['signal_rms_nT']:.3f} nT)" for i in range(3)), flush=True)
    out["meta"]["wall_s"] = time.time() - t0
    os.makedirs(os.path.join(ROOT, "results", "analysis"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "analysis", "a03_three_layer_direction_vs_distance.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

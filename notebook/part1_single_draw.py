"""The single seeded noise draw shown in Part 1 of the notebook, stored in results/notebook/part1_single_draw.json.

With RERUN = False the notebook reads this file instead of drawing the noise and inverting it. The function
`compute()` repeats the notebook's setup (4 x 4 x 1 m prism, drive-around at 1 m, 0.5 nT per axis, seed 0) so the
stored numbers can be regenerated and checked:

    python notebook/part1_single_draw.py            # print the recomputed values
    python notebook/part1_single_draw.py --write    # overwrite the stored file
    python notebook/part1_single_draw.py --check    # exit 1 if the stored file differs (used by tests/)

Takes a few seconds; no other randomness is involved.
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORED = os.path.join(ROOT, "results", "notebook", "part1_single_draw.json")


def compute():
    import discretize
    from simpeg import maps
    from simpeg.potential_fields import magnetics as mag

    M_true = 2.4e-3 * np.array([0.0, 0.0, -1.0])        # A/m
    sigma_B = 0.5                                       # nT per axis
    z = 0.5
    t = np.linspace(-3.0, 3.0, 600)
    c = np.zeros_like(t)
    loop = np.vstack([np.c_[t, c - 3], np.c_[c + 3, t], np.c_[-t, c + 3], np.c_[c - 3, -t][:-1]])
    xyz = np.c_[loop, np.full(len(loop), z)]
    mesh = discretize.TensorMesh([[4.0], [4.0], [1.0]], origin=[-2.0, -2.0, 0.0])
    rx = mag.receivers.Point(xyz, components=["bx", "by", "bz"])
    src = mag.sources.UniformBackgroundField(receiver_list=[rx], amplitude=4e-7 * np.pi * 1e9,
                                             inclination=90.0, declination=0.0)
    sim = mag.simulation.Simulation3DIntegral(mesh, survey=mag.survey.Survey(src), chiMap=maps.IdentityMap(nP=3),
                                              model_type="vector", store_sensitivities="ram")
    G = sim.G
    d_obs = G @ M_true + np.random.default_rng(0).normal(0.0, sigma_B, G.shape[0])
    M_rec = np.linalg.solve(G.T @ G, G.T @ d_obs)
    cos = M_rec @ M_true / (np.linalg.norm(M_rec) * np.linalg.norm(M_true))
    return {
        "description": "Part 1 of notebook/reproduce_paper.ipynb: one noise draw, np.random.default_rng(0), "
                       "drive-around at 1 m, sigma_B = 0.5 nT per axis, true M = (0, 0, -2.4e-3) A/m",
        "n_measurements": int(G.shape[0] // 3),
        "M_rec_A_per_m": [float(v) for v in M_rec],
        "dir_err_deg": float(np.degrees(np.arccos(np.clip(cos, -1, 1)))),
        "dM_abs_pct": float(100 * abs(np.linalg.norm(M_rec) - np.linalg.norm(M_true)) / np.linalg.norm(M_true)),
    }


def check(new, stored, rtol=None):
    if rtol is None:  # 1e-6 here; looser where LUNARMAG_RTOL is set (CI on Linux)
        rtol = max(1e-6, float(os.environ.get("LUNARMAG_RTOL", "0")))
    ok = new["n_measurements"] == stored["n_measurements"]
    ok &= np.allclose(new["M_rec_A_per_m"], stored["M_rec_A_per_m"], rtol=rtol, atol=1e-12)
    ok &= np.isclose(new["dir_err_deg"], stored["dir_err_deg"], rtol=rtol)
    ok &= np.isclose(new["dM_abs_pct"], stored["dM_abs_pct"], rtol=rtol)
    return bool(ok)


if __name__ == "__main__":
    res = compute()
    if "--write" in sys.argv:
        os.makedirs(os.path.dirname(STORED), exist_ok=True)
        with open(STORED, "w") as f:
            json.dump(res, f, indent=1)
            f.write("\n")
        print("wrote", os.path.relpath(STORED, ROOT))
    elif "--check" in sys.argv:
        stored = json.load(open(STORED))
        ok = check(res, stored)
        print("part1 single draw:", "matches stored" if ok else f"DIFFERS: {res} vs {stored}")
        sys.exit(0 if ok else 1)
    print(json.dumps(res, indent=1))

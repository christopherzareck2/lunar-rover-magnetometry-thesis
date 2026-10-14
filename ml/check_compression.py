"""How much of the burial and pose signal lies outside the feature span Q (a check on the compression).

For random (b, pose) drawn from the training prior and random unit m, report the fraction of the nuisance-induced
change of the outcrop field, (G(b, xi) - G_0) m, and of the full outcrop field G(b, xi) m, that lies outside span(Q),
in nT rms and relative terms. The white-noise level per datum is 0.50 nT.

Run:  python ml/check_compression.py      Out: ml/results/compression_check.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    sv = C.Survey(0.5); rng = np.random.default_rng(C.seed_for("ml:compression_check"))
    rows = {"burial_only": [], "pose_only": [], "both": []}
    for case in rows:
        for _ in range(300):
            u = rng.normal(size=3); u /= np.linalg.norm(u); m = 2.0 * u
            b = rng.uniform(0, 1) if case != "pose_only" else 0.0
            if case == "burial_only":
                G = C.operator(C.buried_bounds(b), sv.sx)
            else:
                az, el = rng.uniform(-3, 3, 2); T = rng.uniform(-0.15, 0.15, 3)
                G = C.posed_operator(C.buried_bounds(b), sv.sx, C.rot_az_el(az, el), T)
            f = G @ m; df = f - sv.G @ m
            out_f = f - sv.Q @ (sv.Q.T @ f); out_df = df - sv.Q @ (sv.Q.T @ df)
            rows[case].append({"rms_outside_nT": float(np.sqrt(np.mean(out_f ** 2))),
                               "outside_energy_over_noise": float(out_f @ out_f / C.SIGMA_B ** 2),
                               "frac_change_outside": float(np.linalg.norm(out_df) / max(np.linalg.norm(df), 1e-300))})
    res = {}
    for case, r in rows.items():
        a = {k: np.array([x[k] for x in r]) for k in r[0]}
        res[case] = {k: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(v.max())} for k, v in a.items()}
        print(case, {k: f"med {v['median']:.3g} p95 {v['p95']:.3g}" for k, v in res[case].items()})
    json.dump(res, open(os.path.join(C.RESULTS, "compression_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

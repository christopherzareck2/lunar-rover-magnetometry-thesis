"""Simulator check: the thesis estimator (11 columns: magnetization, offsets, gradient) on data from this package's
simulator, one systematic at a time at full tolerance with the loops 0.5 m off centre, against the stored thesis
results a09 'Z_single_*' (results/analysis/a09_systematics.json, n = 80 per direction there).

One difference is deliberate: a09 puts the true body at the nominal pose and mis-poses the inversion model, while this
package mis-poses the true body and keeps the model nominal (the estimator's model is fixed, which an amortized
estimator needs). The two are statistically close but not identical, so the pose row is compared loosely.

Run:  python ml/replicate_thesis.py      Out: ml/results/replication_vs_a09.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

N = 400
BASE = dict(C.PRIOR_NONE, fixed=True)
CASES = {"none": {}, "burial": dict(bmax=1.0), "pose": dict(pose_deg=3.0, pose_m=0.15),
         "drift": dict(drift_max=0.03), "tilt": dict(tilt_max=0.05, site_B0=100.0, site_R=(1e6, 1e6)),
         "site": dict(site_B0=100.0, site_R=(530.0, 530.0))}


def main():
    sv = C.Survey(0.5)
    ref = json.load(open(os.path.join(C.ROOT, "results", "analysis", "a09_systematics.json")))
    out = {}
    for cn, kw in CASES.items():
        pr = dict(BASE); pr.update(kw)
        row = {}
        for dn, u in C.DIRS3.items():
            rng = np.random.default_rng(C.seed_for(f"ml:replicate:{cn}:{dn}"))
            o = C.simulate(sv, rng, N, pr, m=np.tile(u, (N, 1)), profile=False)
            e = C.dir_err(o["est_LS11"], o["M"])
            t = ref[f"Z_single_{cn}"][dn]["all_sources"]["dir_err_deg"]
            row[dn] = {"ml_sim": C.summary(e), "a09": {"mean": t["mean"], "p95": t["p95"], "N": t["N"]},
                       "diff_in_sem": float((e.mean() - t["mean"]) / np.hypot(e.std() / np.sqrt(N), t["std"] / np.sqrt(t["N"])))}
        out[cn] = row
        print(f"{cn:7s} " + "  ".join(f"{dn}: {row[dn]['ml_sim']['mean']:.2f} vs {row[dn]['a09']['mean']:.2f} "
                                     f"({row[dn]['diff_in_sem']:+.1f} se)" for dn in C.DIRS3), flush=True)
    os.makedirs(C.RESULTS, exist_ok=True)
    json.dump(out, open(os.path.join(C.RESULTS, "replication_vs_a09.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

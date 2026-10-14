"""Does driving each loop twice, once in each direction, reduce the slow-drift error itself? (Y6-14 follow-up)

The thesis says the reversed design "separates slow drift from the outcrop's field". This check splits the error of the
two designs (one-way two-loop survey; each loop driven both ways) into the drift-only part (noise-free except drift)
and the total with 0.5 nT white noise, at the same drift rms about the survey mean (0.01 nT, L = 100 m, as a09 parts d
and r), loops centred, no other error source. 500 draws; the drift and noise realizations are drawn for each design's
own path (the reversed path is twice as long, and its drift is renormalized to the same survey rms, as in a09).

Model: ``_budget.OutcropModel`` (a09 physics). Seeds: study:driftrev:<design>:<k>.

Run:  python studies/drift_reversal_check.py      (about 1 min)
Out:  studies/results/drift_reversal_check.json
"""
from __future__ import annotations

import time

import numpy as np

import _common as C
import _budget as B

N = 500


def main():
    t0 = time.time()
    out = {"meta": C.provenance(__file__, {"draws": N, "drift_rms_nT": 0.01, "dx_m": 0.0,
                                           "seeds": "study:driftrev:<design>:<k>"}), "designs": {}}
    for design, rev in (("one_way", False), ("both_ways", True)):
        res = {}
        for drift in (0.0, 0.01):
            m = B.OutcropModel(0.0, rev, False, 1e6, 100.0, 0.0, 0.0, 0.0, drift, C.DIRS6)
            Td, Tt = [], []
            for k in range(N):
                r = C.rng_for(f"study:driftrev:{design}:{k}")
                o = m.evaluate(b=0.0, az_deg=0.0, el_deg=0.0, trans=np.zeros(3), site_az=0.0, site_m=np.array([0, 0, 1.0]),
                               tilt_axis=0.0, eta=0.0, drift_seed=int(r.integers(2 ** 62)), noise_seed=int(r.integers(2 ** 62)))
                Td.append([C.dir_err_deg(o["Mdrift"][i], m.MT[i]) for i in range(6)])
                Tt.append([C.dir_err_deg(o["Mrec"][i], m.MT[i]) for i in range(6)])
            Td, Tt = np.array(Td), np.array(Tt)
            res[f"drift_{drift:g}nT"] = {dn: {"drift_only_dir_err": C.mean_se(Td[:, i]), "total_dir_err": C.mean_se(Tt[:, i])}
                                         for i, dn in enumerate(C.DIRS6)}
        out["designs"][design] = res
        print(design, " | ".join(f"{dn}: drift-only {res['drift_0.01nT'][dn]['drift_only_dir_err']['mean']:.2f}, "
                                 f"total {res['drift_0.01nT'][dn]['total_dir_err']['mean']:.2f} (white only {res['drift_0nT'][dn]['total_dir_err']['mean']:.2f})"
                                 for dn in C.DIRS6), flush=True)
    out["meta"]["wall_s"] = time.time() - t0
    print(C.save("drift_reversal_check.json", out))


if __name__ == "__main__":
    main()

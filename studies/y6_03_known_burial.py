"""Y6-03: "burial known" rows rerun with known NONZERO burial depths.

The released "burial known" rows (a09 parts q/c, h and j: REQ_c, REQF_c, REQF_chi5_c) pass bmax = 0, so the true body
and the model body are both the exposed prism; they show the b = 0 case only. Here the true body extends b metres
below the surface with b ~ U(0, bmax) drawn per draw, and the inversion operator is built from the same buried prism at
the imaged (mis-posed) pose, i.e. the depth is known exactly ("modeled exactly in G"). Everything else is the released
configuration: loops 0.5 m off centre, requirement-set tolerances (pose +-1.5 deg / +-0.075 m, site residual ~0.02 nT,
drift 0.01 nT, tilt 0.05 nT, chi known to a factor of 1.4), six magnetization directions, 120 draws per direction.

bmax = 1.0 m is the headline case (the burial range of the full-tolerance budget); bmax = 0.5 m is added for comparison
with the b <= 0.5 m rows, and bmax = 0 with the same new seed labels is a same-code control for the released b = 0 rows
(it separates the effect of the depth from Monte Carlo scatter).

Modified copy: ``_budget.budget_vectors(..., known_burial=True)`` (see its header). New seed labels
"study:y6-03:<key>:b<bmax>:<dir>", so the draws are independent of the released rows.

Run:  python studies/y6_03_known_burial.py      (about 3 min)
Out:  studies/results/y6_03_known_burial.json
"""
from __future__ import annotations

import time

import numpy as np

import _common as C
import _budget as B

L14 = np.log(1.4)
BASE = dict(dx=0.5, R_site=530.0, pose=(1.5, 0.075), drift_rms=0.01, tilt_nT=0.05, chi_err=L14, n=120, dirs=C.DIRS6)
ROWS = [  # (thesis row, released key, extra arguments)
    ("Requirement set, burial known", "REQ_c_dx0.5_b0", dict(chi=2e-3)),
    ("Mitigated set, burial known", "REQF_c_dx0.5_b0", dict(chi=2e-3, fit_tilt=True, reverse=True)),
    ("Mitigated set, chi = 5e-3, burial known", "REQF_chi5_c_dx0.5_b0", dict(chi=5e-3, fit_tilt=True, reverse=True)),
]


def summarize(V):
    d = {}
    for dn, v in V.items():
        mt = v["mt"]
        a_tot = np.array([C.dir_err_deg(m, mt) for m in v["Mrec"]])
        a_nf = np.array([C.dir_err_deg(m, mt) for m in v["Mnf"]])
        dm = 100 * (np.linalg.norm(v["Mrec"], axis=1) - np.linalg.norm(mt)) / np.linalg.norm(mt)
        edges = np.linspace(0, max(v["b"].max(), 1e-9), 5)
        by_b = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (v["b"] >= lo) & (v["b"] <= hi)
            if sel.sum() > 2:
                by_b.append({"b_lo": float(lo), "b_hi": float(hi), "n": int(sel.sum()),
                             "dir_err_mean": float(a_tot[sel].mean()), "dir_err_se": float(a_tot[sel].std(ddof=1) / np.sqrt(sel.sum()))})
        d[dn] = {"dir_err": C.mean_se(a_tot), "bias_mean_nf_angle": C.mean_se(a_nf),
                 "angle_of_mean_nf": float(C.dir_err_deg(v["Mnf"].mean(0), mt)), "dM_signed_pct": C.mean_se(dm),
                 "b_mean_m": float(v["b"].mean()), "dir_err_by_b_quartile": by_b,
                 "corr_dir_err_vs_b": float(np.corrcoef(a_tot, v["b"])[0, 1]) if v["b"].std() > 0 else None}
    rng = lambda f: [float(min(f(x) for x in d.values())), float(max(f(x) for x in d.values()))]
    return {"directions": d,
            "table": {"theta_mean_range": rng(lambda x: x["dir_err"]["mean"]),
                      "theta_mean_se_range": rng(lambda x: x["dir_err"]["se"]),
                      "theta_p95_range": rng(lambda x: x["dir_err"]["p95"]),
                      "bias_range": rng(lambda x: x["bias_mean_nf_angle"]["mean"]),
                      "dM_signed_range": rng(lambda x: x["dM_signed_pct"]["mean"]),
                      "n_dirs_mean_above_5deg": int(sum(x["dir_err"]["mean"] > 5 for x in d.values()))}}


def stored_table(s):
    d = {k: v for k, v in s.items() if k != "config"}
    rng = lambda f: [float(min(f(x) for x in d.values())), float(max(f(x) for x in d.values()))]
    se = lambda x: x["all_sources"]["dir_err_deg"]["std"] / np.sqrt(x["all_sources"]["dir_err_deg"]["N"])
    return {"theta_mean_range": rng(lambda x: x["all_sources"]["dir_err_deg"]["mean"]),
            "theta_mean_se_range": rng(se),
            "theta_p95_range": rng(lambda x: x["all_sources"]["dir_err_deg"]["p95"]),
            "bias_range": rng(lambda x: x["noise_free_bias"]["dir_err_deg"]["mean"]),
            "dM_signed_range": rng(lambda x: x["all_sources"]["dM_signed_pct"]["mean"]),
            "n_dirs_mean_above_5deg": int(sum(x["all_sources"]["dir_err_deg"]["mean"] > 5 for x in d.values()))}


def main():
    t0 = time.time()
    st = C.stored("results/analysis/a09_systematics.json")
    out = {"meta": C.provenance(__file__, {
        "finding": "Y6-03", "draws": "120 per direction, 6 directions (as the thesis rows)",
        "seeds": "study:y6-03:<key>:b<bmax>:<dir> (and :geom for the pose stream)",
        "known_burial_model": "true body = exposed prism + b m below the surface, b ~ U(0, bmax); inversion operator = the "
                              "same buried prism at the imaged pose"}), "rows": {}}
    for row, key, extra in ROWS:
        out["rows"][row] = {"released_key": key, "old_b0_stored": stored_table(st[key])}
        for bmax in (1.0, 0.5, 0.0):
            V = B.budget_vectors(f"study:y6-03:{key}:b{bmax:g}", bmax=bmax, known_burial=True, **BASE, **extra)
            out["rows"][row][f"new_known_b_U0_{bmax:g}"] = summarize(V)
            t = out["rows"][row][f"new_known_b_U0_{bmax:g}"]["table"]
            o = out["rows"][row]["old_b0_stored"]
            print(f"{row:42s} b<={bmax}: theta {o['theta_mean_range'][0]:.2f}-{o['theta_mean_range'][1]:.2f} -> "
                  f"{t['theta_mean_range'][0]:.2f}-{t['theta_mean_range'][1]:.2f} (SE {t['theta_mean_se_range'][0]:.2f}-{t['theta_mean_se_range'][1]:.2f}); "
                  f"p95 {t['theta_p95_range'][0]:.1f}-{t['theta_p95_range'][1]:.1f}; bias {o['bias_range'][0]:.2f}-{o['bias_range'][1]:.2f} -> "
                  f"{t['bias_range'][0]:.2f}-{t['bias_range'][1]:.2f}; dMs {t['dM_signed_range'][0]:+.1f} to {t['dM_signed_range'][1]:+.1f}; "
                  f">5deg dirs {t['n_dirs_mean_above_5deg']}  ({time.time()-t0:.0f}s)", flush=True)
            C.save("y6_03_known_burial.json", out)
    out["meta"]["wall_s"] = time.time() - t0
    print(C.save("y6_03_known_burial.json", out))


if __name__ == "__main__":
    main()

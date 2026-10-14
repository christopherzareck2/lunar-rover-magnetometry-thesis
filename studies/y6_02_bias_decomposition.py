"""Y6-02: what the budget table's "bias" column measures, row by row.

The released a09 ``budget`` reports "noise_free_bias" = mean over draws of the angle between the noise-free recovered
vector and the truth, with burial, pose, site dipole, tilt axis and susceptibility error redrawn every draw. That is a
mean angular error (a per-outcrop systematic loss), not the angle of the mean vector, which is what survives averaging
over outcrops whose errors are independent. This script reruns every row of Table 5.2 (thesis Table "Combined error
budget") with the released seeds, checks that the stored means come back, and adds for each row and direction:

  * mean noise-free angular error (= the table's "Bias"), with its standard error;
  * angle of the mean noise-free vector (the part that a mean over independent outcrops keeps), with a bootstrap SE;
  * the same two quantities for the full recovered vectors (white noise and drift on);
  * the mean signed intensity error of the noise-free vectors.

Modified copy: ``_budget.budget_vectors`` (a09 ``budget`` with the vectors kept; see its header). Released seeds are
reused on purpose (labels "a09:<key>"), so these are the same draws as the stored table.

Run:  python studies/y6_02_bias_decomposition.py      (about 3 min)
Out:  studies/results/y6_02_bias_decomposition.json
"""
from __future__ import annotations

import time

import numpy as np

import _common as C
import _budget as B

CHI = 2e-3
L2, L14, L12 = np.log(2.0), np.log(1.4), np.log(1.2)
D6 = C.DIRS6
SINGLE = dict(dx=0.5, R_site=1e6, pose=(0.0, 0.0), bmax=0.0, drift_rms=0.0, tilt_nT=0.0, chi=0.0, chi_err=0.0, n=80)

# (table row, a09 key, budget arguments) -- argument lists copied from analysis/a09_systematics.py parts Z, U, V, G, Y, X, q, f-j
ROWS = [
    ("Noise only", "Z_single_none", dict(SINGLE)),
    ("Site field alone", "Z_single_site", dict(SINGLE, R_site=530.0)),
    ("Pose alone", "Z_single_pose", dict(SINGLE, pose=(3.0, 0.15))),
    ("Burial alone (b<=1 m)", "Z_single_burial", dict(SINGLE, bmax=1.0)),
    ("Drift alone (0.03 nT)", "Z_single_drift", dict(SINGLE, drift_rms=0.03)),
    ("Tilt alone (0.05 nT)", "Z_single_tilt", dict(SINGLE, tilt_nT=0.05)),
    ("Induced magnetization alone", "Z_single_induced", dict(SINGLE, chi=CHI, chi_err=L2)),
    ("All, full tolerances", "U_budget_full_offcentre", dict(dx=0.5, R_site=530.0, pose=(3.0, 0.15), bmax=1.0, drift_rms=0.03, tilt_nT=0.05, chi=CHI, chi_err=L2)),
    ("All, half tolerances, off center", "V_budget_half_offcentre", dict(dx=0.5, R_site=750.0, pose=(1.5, 0.075), bmax=0.5, drift_rms=0.015, tilt_nT=0.025, chi=CHI, chi_err=L14)),
    ("All, half tolerances, root fitted", "F2_half_offcentre_root", dict(dx=0.5, R_site=750.0, pose=(1.5, 0.075), bmax=0.5, drift_rms=0.015, tilt_nT=0.025, chi=CHI, chi_err=L14, root=1.0)),
    ("All, quarter tolerances", "Y_budget_quarter_offcentre", dict(dx=0.5, R_site=1060.0, pose=(0.75, 0.0375), bmax=0.25, drift_rms=0.0075, tilt_nT=0.0125, chi=CHI, chi_err=L12)),
    ("All, half tolerances, centered", "X_budget_half_centred", dict(dx=0.0, R_site=750.0, pose=(1.5, 0.075), bmax=0.5, drift_rms=0.015, tilt_nT=0.025, chi=CHI, chi_err=L14)),
    ("Requirement set, b<=0.5 m", "REQ_a_dx0.1_b0.5", dict(dx=0.1, R_site=530.0, pose=(1.5, 0.075), bmax=0.5, drift_rms=0.01, tilt_nT=0.05, chi=CHI, chi_err=L14, n=120, dirs=D6)),
    ("Requirement set, b<=1 m", "REQ_b_dx0.1_b1", dict(dx=0.1, R_site=530.0, pose=(1.5, 0.075), bmax=1.0, drift_rms=0.01, tilt_nT=0.05, chi=CHI, chi_err=L14, n=120, dirs=D6)),
    ("Requirement set, 'burial known' (b=0)", "REQ_c_dx0.5_b0", dict(dx=0.5, R_site=530.0, pose=(1.5, 0.075), bmax=0.0, drift_rms=0.01, tilt_nT=0.05, chi=CHI, chi_err=L14, n=120, dirs=D6)),
    ("Mitigated set, b<=0.5 m", "REQF_a_dx0.1_b0.5", dict(dx=0.1, R_site=530.0, pose=(1.5, 0.075), bmax=0.5, drift_rms=0.01, tilt_nT=0.05, chi=CHI, chi_err=L14, n=120, dirs=D6, fit_tilt=True, reverse=True)),
    ("Mitigated set, b<=1 m", "REQF_b_dx0.1_b1", dict(dx=0.1, R_site=530.0, pose=(1.5, 0.075), bmax=1.0, drift_rms=0.01, tilt_nT=0.05, chi=CHI, chi_err=L14, n=120, dirs=D6, fit_tilt=True, reverse=True)),
    ("Mitigated set, 'burial known' (b=0)", "REQF_c_dx0.5_b0", dict(dx=0.5, R_site=530.0, pose=(1.5, 0.075), bmax=0.0, drift_rms=0.01, tilt_nT=0.05, chi=CHI, chi_err=L14, n=120, dirs=D6, fit_tilt=True, reverse=True)),
    ("Mitigated set, chi=5e-3, b<=0.5 m", "REQF_chi5_a_dx0.1_b0.5", dict(dx=0.1, R_site=530.0, pose=(1.5, 0.075), bmax=0.5, drift_rms=0.01, tilt_nT=0.05, chi=5e-3, chi_err=L14, n=120, dirs=D6, fit_tilt=True, reverse=True)),
    ("Mitigated set, chi=5e-3, 'burial known' (b=0)", "REQF_chi5_c_dx0.5_b0", dict(dx=0.5, R_site=530.0, pose=(1.5, 0.075), bmax=0.0, drift_rms=0.01, tilt_nT=0.05, chi=5e-3, chi_err=L14, n=120, dirs=D6, fit_tilt=True, reverse=True)),
]


def angles(V, mt):
    return np.array([C.dir_err_deg(v, mt) for v in V])


def main():
    t0 = time.time()
    st = C.stored("results/analysis/a09_systematics.json")
    out = {"meta": C.provenance(__file__, {
        "finding": "Y6-02", "seeds": "released a09 labels a09:<key>:<dir> (same draws as the stored Table 5.2)",
        "definitions": {
            "mean_nf_angle": "mean over draws of angle(M_nf_i, M_true) = the table's 'Bias' (white noise and drift off)",
            "angle_of_mean_nf": "angle(mean_i M_nf_i, M_true): the systematic part common to all draws",
            "mean_total_angle": "mean over draws of angle(M_rec_i, M_true) = the table's 'theta_err mean'",
            "angle_of_mean_total": "angle(mean_i M_rec_i, M_true)"}}), "rows": {}}
    worst = 0.0
    for row, key, kw in ROWS:
        V = B.budget_vectors(f"a09:{key}", **kw)
        r = {"a09_key": key, "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items() if k != "dirs"},
             "directions": {}}
        for dn, v in V.items():
            mt = v["mt"]
            a_nf, a_tot = angles(v["Mnf"], mt), angles(v["Mrec"], mt)
            dm_nf = 100 * (np.linalg.norm(v["Mnf"], axis=1) - np.linalg.norm(mt)) / np.linalg.norm(mt)
            s = st[key][dn]
            dev = max(abs(a_nf.mean() - s["noise_free_bias"]["dir_err_deg"]["mean"]),
                      abs(a_tot.mean() - s["all_sources"]["dir_err_deg"]["mean"]))
            worst = max(worst, dev)
            r["directions"][dn] = {
                "mean_nf_angle": C.mean_se(a_nf),
                "angle_of_mean_nf": C.angle_of_mean(v["Mnf"], mt, seed_label=f"study:y6-02:bs:nf:{key}:{dn}"),
                "mean_total_angle": C.mean_se(a_tot),
                "angle_of_mean_total": C.angle_of_mean(v["Mrec"], mt, seed_label=f"study:y6-02:bs:tot:{key}:{dn}"),
                "dM_signed_nf_mean_pct": C.mean_se(dm_nf),
                "stored_bias_mean": s["noise_free_bias"]["dir_err_deg"]["mean"],
                "stored_total_mean": s["all_sources"]["dir_err_deg"]["mean"]}
        out["rows"][row] = r
        mn = [d["mean_nf_angle"]["mean"] for d in r["directions"].values()]
        am = [d["angle_of_mean_nf"]["value"] for d in r["directions"].values()]
        print(f"{row:48s} mean nf angle {min(mn):5.2f}-{max(mn):5.2f} | angle of mean nf {min(am):5.2f}-{max(am):5.2f}  ({time.time()-t0:.0f}s)", flush=True)
        out["meta"]["max_abs_diff_vs_stored_deg"] = worst
        C.save("y6_02_bias_decomposition.json", out)
    out["meta"]["wall_s"] = time.time() - t0
    print("max deviation from stored means:", worst)
    print(C.save("y6_02_bias_decomposition.json", out))


if __name__ == "__main__":
    main()

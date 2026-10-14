"""Flow-mean simulation: ten outcrops of one lava flow, independent per-outcrop errors plus errors shared by the flow.

Question (follows from finding Y6-02): the budget's per-outcrop "bias" is a mean angular loss. How large is the error of
a flow-mean direction built from ten outcrops, and does the Fisher alpha95 of that mean cover the true direction?

Model, per flow (500 flows per configuration):
  * the ten outcrops share the magnetization direction (each of the six test directions of the requirement-set rows,
    evaluated on the same draws) and the survey frame;
  * per outcrop, independently: buried extent b ~ U(0, bmax), imaging pose error (pointing az/el ~ U(+-p),
    translation ~ U(+-t) per axis), site field (100 nT at the outcrop from a dipole at R_site, random azimuth and
    moment), tilt axis ~ U(0, 2 pi), susceptibility error ln(chi_est/chi) ~ N(0, s^2), an OU drift realization and
    white noise (0.5 nT), exactly as one draw of the a09 combined budget;
  * loop offset: dx from the outcrop centre, either along +x for every outcrop ("shared offset", the budget's own
    convention, a worst case) or in an independent random azimuth per outcrop ("random offset");
  * shared by the flow: a survey-frame rotation, applied to every outcrop's recovered vector: none, heading only
    (rotation about vertical, U(+-1 deg)), or full frame (three small rotations, each U(+-1 deg)).
Sets (thesis Table 5.1): half tolerance (loops 0.5 m off centre, a09 row V) and the requirement set (loops 0.1 m off
centre, b <= 0.5 m, a09 row REQ_a); the mitigated requirement set (each loop driven both ways, tilt heading terms
fitted, a09 row REQF_a) is added.

Per flow: Fisher mean of the ten unit vectors, its angle to the truth, R, k = (N-1)/(N-R) and
alpha95 = arccos(1 - (N-R)/R (20^(1/(N-1)) - 1)); "covered" if the truth lies within alpha95.
Fisher-independence prediction: from the per-outcrop errors of the same simulation, kappa = 2 / E[theta^2]
(small-angle Fisher), so the mean of N independent outcrops has E[theta_F] = sqrt(pi / (2 N kappa)) and
95% cone sqrt(-2 ln 0.05 / (N kappa)); nominal alpha95 coverage 95%.

Seeds: every outcrop draws its inputs from seed_for("study:flow:<flow>:<outcrop>") as uniforms/normals that are scaled to
each set's tolerances, so the sets and the offset cases are paired (common random numbers); the shared frame rotation
uses seed_for("study:flow:<flow>:frame").

Run:  python studies/flow_mean.py [--flows 500]      (about 15-25 min)
Out:  studies/results/flow_mean.json
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import _common as C
import _budget as B
import _fastprism as FP

N_OUT = 10
SETS = {
    "half_tolerance": dict(dx=0.5, R_site=750.0, pose=(1.5, 0.075), bmax=0.5, drift_rms=0.015, tilt_nT=0.025,
                           chi=2e-3, chi_err=np.log(1.4), reverse=False, fit_tilt=False, a09_row="V_budget_half_offcentre"),
    "requirement_set": dict(dx=0.1, R_site=530.0, pose=(1.5, 0.075), bmax=0.5, drift_rms=0.01, tilt_nT=0.05,
                            chi=2e-3, chi_err=np.log(1.4), reverse=False, fit_tilt=False, a09_row="REQ_a_dx0.1_b0.5"),
    "mitigated_set": dict(dx=0.1, R_site=530.0, pose=(1.5, 0.075), bmax=0.5, drift_rms=0.01, tilt_nT=0.05,
                          chi=2e-3, chi_err=np.log(1.4), reverse=True, fit_tilt=True, a09_row="REQF_a_dx0.1_b0.5"),
}
SHARED = ("none", "heading_1deg", "frame_1deg")


def rot(axis, ang):
    c, s = np.cos(ang), np.sin(ang)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def outcrop_inputs(flow, k):
    """Unit-scale random inputs of one outcrop (shared by all sets and offset cases)."""
    r = C.rng_for(f"study:flow:{flow}:{k}")
    return {"ub": r.uniform(), "upose": r.uniform(-1, 1, 5), "site_az": r.uniform(0, 2 * np.pi), "site_m": r.normal(size=3),
            "tilt_axis": r.uniform(0, 2 * np.pi), "eta": r.normal(), "drift_seed": int(r.integers(2 ** 62)),
            "noise_seed": int(r.integers(2 ** 62)), "offset_az": r.uniform(0, 2 * np.pi)}


def frame_rotation(flow, kind):
    r = C.rng_for(f"study:flow:{flow}:frame")
    a = np.radians(r.uniform(-1.0, 1.0, 3))
    if kind == "none":
        return np.eye(3)
    if kind == "heading_1deg":
        return rot("z", a[0])
    return rot("z", a[0]) @ rot("y", a[1]) @ rot("x", a[2])


def fisher_stats(U):
    """U: (N, 3) unit vectors. Returns mean unit vector, R, k, alpha95 (deg)."""
    S = U.sum(0); R = np.linalg.norm(S); N = len(U)
    a = 1 - (N - R) / R * (20.0 ** (1 / (N - 1)) - 1)
    return S / R, R, (N - 1) / max(N - R, 1e-300), float(np.degrees(np.arccos(np.clip(a, -1, 1))))


def unit(V):
    return V / np.linalg.norm(V, axis=-1, keepdims=True)


def run_set(name, cfg, n_flows, t0):
    dirs = C.DIRS6; names = list(dirs); MT = np.array([C.M_NRM_MIN * dirs[d] for d in names])
    model = B.OutcropModel(cfg["dx"], cfg["reverse"], cfg["fit_tilt"], cfg["R_site"], 100.0, cfg["tilt_nT"],
                           cfg["chi"], cfg["chi_err"], cfg["drift_rms"], dirs)
    p, t = cfg["pose"]
    rec = {oc: np.zeros((n_flows, N_OUT, len(names), 3)) for oc in ("shared_offset", "random_offset")}
    nf = {oc: np.zeros((n_flows, N_OUT, len(names), 3)) for oc in ("shared_offset", "random_offset")}
    for f in range(n_flows):
        for k in range(N_OUT):
            x = outcrop_inputs(f, k)
            args = dict(b=cfg["bmax"] * x["ub"], az_deg=p * x["upose"][0], el_deg=p * x["upose"][1],
                        trans=t * x["upose"][2:], site_az=x["site_az"], site_m=x["site_m"], tilt_axis=x["tilt_axis"],
                        eta=x["eta"], drift_seed=x["drift_seed"], noise_seed=x["noise_seed"])
            for oc, az in (("shared_offset", 0.0), ("random_offset", x["offset_az"])):
                o = model.evaluate(offset_az=az, **args)
                rec[oc][f, k] = o["Mrec"]; nf[oc][f, k] = o["Mnf"]
        if (f + 1) % 50 == 0:
            print(f"  [{name}] {f+1}/{n_flows} flows ({time.time()-t0:.0f}s)", flush=True)

    res = {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}, "cases": {}}
    for oc in rec:
        for sh in SHARED:
            Rf = np.array([frame_rotation(f, sh) for f in range(n_flows)])            # (F, 3, 3)
            Vr = np.einsum("fij,fkdj->fkdi", Rf, rec[oc])
            case = {}
            for di, dn in enumerate(names):
                u = MT[di] / np.linalg.norm(MT[di])
                Uo = unit(Vr[:, :, di, :])                                             # (F, N, 3)
                th_out = np.degrees(np.arccos(np.clip(Uo @ u, -1, 1))).ravel()
                kappa = 2.0 / np.mean(np.radians(th_out) ** 2)
                thF, a95, cov, kk = [], [], [], []
                for f in range(n_flows):
                    m, R, kf, a = fisher_stats(Uo[f])
                    e = float(np.degrees(np.arccos(np.clip(m @ u, -1, 1))))
                    thF.append(e); a95.append(a); cov.append(e <= a); kk.append(kf)
                thF, a95, cov = np.array(thF), np.array(a95), np.array(cov, float)
                grand = unit(unit(Vr[:, :, di, :]).reshape(-1, 3).sum(0))
                case[dn] = {
                    "outcrop_dir_err": C.mean_se(th_out),
                    "flow_mean_dir_err": C.mean_se(thF),
                    "alpha95": C.mean_se(a95),
                    "coverage": {"value": float(cov.mean()), "se": float(np.sqrt(cov.mean() * (1 - cov.mean()) / n_flows))},
                    "systematic_angle_of_grand_mean_deg": float(np.degrees(np.arccos(np.clip(grand @ u, -1, 1)))),
                    "fisher_independence_prediction": {
                        "kappa_from_outcrops": float(kappa),
                        "flow_mean_dir_err_mean": float(np.degrees(np.sqrt(np.pi / (2 * N_OUT * kappa)))),
                        "flow_mean_dir_err_p95": float(np.degrees(np.sqrt(-2 * np.log(0.05) / (N_OUT * kappa)))),
                        "coverage": 0.95},
                    "ratio_observed_to_predicted_mean": float(thF.mean() / np.degrees(np.sqrt(np.pi / (2 * N_OUT * kappa))))}
            # noise-free per-outcrop systematic part (no shared rotation): what averaging cannot remove
            if sh == "none":
                for di, dn in enumerate(names):
                    u = MT[di] / np.linalg.norm(MT[di])
                    Un = unit(nf[oc][:, :, di, :])
                    gm = unit(Un.reshape(-1, 3).sum(0))
                    case[dn]["noise_free_outcrop_err"] = C.mean_se(np.degrees(np.arccos(np.clip(Un @ u, -1, 1))).ravel())
                    case[dn]["noise_free_angle_of_grand_mean_deg"] = float(np.degrees(np.arccos(np.clip(gm @ u, -1, 1))))
                    fm = np.array([np.degrees(np.arccos(np.clip(fisher_stats(Un[f])[0] @ u, -1, 1))) for f in range(n_flows)])
                    case[dn]["noise_free_flow_mean_dir_err"] = C.mean_se(fm)
            res["cases"][f"{oc}|{sh}"] = case
            rng_ = lambda key: [min(case[d][key]["mean"] for d in names), max(case[d][key]["mean"] for d in names)]
            covs = [case[d]["coverage"]["value"] for d in names]
            pr = [case[d]["fisher_independence_prediction"]["flow_mean_dir_err_mean"] for d in names]
            print(f"[{name} {oc:13s} {sh:12s}] outcrop {rng_('outcrop_dir_err')[0]:.2f}-{rng_('outcrop_dir_err')[1]:.2f} | "
                  f"flow mean {rng_('flow_mean_dir_err')[0]:.2f}-{rng_('flow_mean_dir_err')[1]:.2f} "
                  f"(Fisher pred {min(pr):.2f}-{max(pr):.2f}) | a95 {rng_('alpha95')[0]:.2f}-{rng_('alpha95')[1]:.2f} | "
                  f"coverage {min(covs):.3f}-{max(covs):.3f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--flows", type=int, default=500)
    ap.add_argument("--sets", default=",".join(SETS)); a = ap.parse_args()
    t0 = time.time()
    sx = B.survey_points(0.5, False)
    out = {"meta": C.provenance(__file__, {
        "study": "flow mean (outline-v2 item 3)", "n_flows": a.flows, "n_outcrops_per_flow": N_OUT,
        "directions": list(C.DIRS6), "fast_operator_check": FP.check(sx, [B.body_bounds(0.0), B.body_bounds(0.4)]),
        "seeds": "study:flow:<flow>:<outcrop> (inputs), study:flow:<flow>:frame (shared rotation)"}), "sets": {}}
    for name in a.sets.split(","):
        out["sets"][name] = run_set(name, SETS[name], a.flows, t0)
        out["meta"]["wall_s"] = time.time() - t0
        C.save("flow_mean.json", out)
    print(C.save("flow_mean.json", out))


if __name__ == "__main__":
    main()

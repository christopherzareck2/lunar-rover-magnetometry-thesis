"""Which error source leaves a systematic (flow-shared) direction bias once the loop offset is random per outcrop?

Companion to flow_mean.py. With the loop offset in an independent random azimuth for every outcrop, the flow-mean
simulation still shows a systematic error of ~1.5-2.3 deg for inclined magnetizations (angle of the grand-mean
noise-free vector). This script switches the requirement-set sources on one at a time (all others at zero) and reports,
per magnetization direction, the angle of the grand-mean noise-free vector over 2000 outcrops (bootstrap SE), and the
inclination of that mean minus the true inclination (positive = recovered direction steeper).

Same model and inputs as flow_mean.py (``_budget.OutcropModel``; seed labels study:flowbias:<source>:<k>).

Run:  python studies/flow_mean_bias_source.py      (about 3 min)
Out:  studies/results/flow_mean_bias_source.json
"""
from __future__ import annotations

import time

import numpy as np

import _common as C
import _budget as B

N = 2000
REQ = dict(dx=0.1, R_site=530.0, pose=(1.5, 0.075), bmax=0.5, drift_rms=0.01, tilt_nT=0.05, chi=2e-3, chi_err=np.log(1.4))
SOURCES = {"all": {}, "burial_only": dict(only="burial"), "pose_only": dict(only="pose"), "site_only": dict(only="site"),
           "tilt_only": dict(only="tilt"), "induced_only": dict(only="induced"), "drift_only": dict(only="drift"),
           "none_centering_only": dict(only="none")}


def inc_deg(v):
    return float(np.degrees(np.arctan2(-v[2], np.hypot(v[0], v[1]))))


def main():
    t0 = time.time()
    out = {"meta": C.provenance(__file__, {"N_outcrops": N, "set": "requirement set, loops 0.1 m off centre in a random "
                                           "azimuth per outcrop", "seeds": "study:flowbias:<source>:<k>"}), "sources": {}}
    for name, spec in SOURCES.items():
        only = spec.get("only")
        on = lambda s: only is None or only == s
        m = B.OutcropModel(REQ["dx"], False, False, REQ["R_site"] if on("site") else 1e6, 100.0,
                           REQ["tilt_nT"] if on("tilt") else 0.0, REQ["chi"] if on("induced") else 0.0,
                           REQ["chi_err"], REQ["drift_rms"] if on("drift") else 0.0, C.DIRS6)
        p, t = REQ["pose"] if on("pose") else (0.0, 0.0)
        V = np.zeros((N, len(C.DIRS6), 3))
        for k in range(N):
            r = C.rng_for(f"study:flowbias:{name}:{k}")
            up = r.uniform(-1, 1, 5)
            o = m.evaluate(b=REQ["bmax"] * r.uniform() if on("burial") else 0.0, az_deg=p * up[0], el_deg=p * up[1],
                           trans=t * up[2:], site_az=r.uniform(0, 2 * np.pi), site_m=r.normal(size=3),
                           tilt_axis=r.uniform(0, 2 * np.pi), eta=r.normal(), drift_seed=int(r.integers(2 ** 62)),
                           offset_az=r.uniform(0, 2 * np.pi))
            V[k] = o["Mdrift"] if "Mdrift" in o else o["Mnf"]
        res = {}
        for di, dn in enumerate(C.DIRS6):
            mt = m.MT[di]
            U = V[:, di] / np.linalg.norm(V[:, di], axis=1, keepdims=True)
            aom = C.angle_of_mean(U, mt, n_boot=500, seed_label=f"study:flowbias:bs:{name}:{dn}")
            g = U.mean(0)
            res[dn] = {"angle_of_grand_mean_deg": aom, "inclination_bias_deg": inc_deg(g) - inc_deg(mt),
                       "mean_outcrop_err_deg": C.mean_se([C.dir_err_deg(v, mt) for v in V[:, di]])}
        out["sources"][name] = res
        print(f"[{name:20s}] angle of grand mean: " + " ".join(f"{dn} {res[dn]['angle_of_grand_mean_deg']['value']:.2f}" for dn in res)
              + " | dI: " + " ".join(f"{res[dn]['inclination_bias_deg']:+.2f}" for dn in res) + f"  ({time.time()-t0:.0f}s)", flush=True)
        C.save("flow_mean_bias_source.json", out)
    out["meta"]["wall_s"] = time.time() - t0
    print(C.save("flow_mean_bias_source.json", out))


if __name__ == "__main__":
    main()

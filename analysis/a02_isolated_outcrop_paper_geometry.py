"""a02 - Isolated outcrop (paper Section 4.2, Figures 4 and 5).

Geometry (paper Sections 3.3.3 and 3.5.1, Figure 1):
  4 x 4 x 1 m prism standing on the ground (z = 0..1 m), sensor at z = 0.5 m (mid-height of the
  outcrop), 1 m standoff from the faces, 1 cm along-track sampling.
  Drive-around: closed rectangular loop at 1 m standoff (2399 stations).
  Drive-by: straight pass at 1 m standoff from x = -8 to +8 m (1601 stations).

Implementation: the prism is centred on the origin (z = -0.5..0.5) and the sensor is at z = 0.0, which is
the same relative geometry as prism 0..1 / sensor 0.5 (the ground is non-magnetic, so only relative position
matters). With the body centred on the origin, the imaging mis-pose rotation in
engine/isolated_outcrop_1cm.warped_recovery is about the body centre, as the paper states.

Parts (all K = 1 closed-form weighted least squares, sigma_B = 0.50 nT per axis, i.i.d. Gaussian):
  1. Baseline, N = 1000 per survey.
  2. Detection sweep, B_NRM/B_ref in {0.1 ... 100}, N = 200 per point (Figure 4 uses a02b, N = 1000).
  3. Imaging mis-pose grid, pointing 0..6 deg x range 0..0.30 m, N = 100 typical + 32 worst per cell
     (Figure 5).

Intensity-error convention: the paper's intensity error is |(|M_rec| - |M_true|)| / |M_true|, i.e. an ABSOLUTE
error per recovery. Both the signed and absolute statistics are stored; the figures and text use the
absolute median, which is the typical single-survey error. The signed mean is the bias.

Run:  python analysis/a02_isolated_outcrop_paper_geometry.py
Out:  results/analysis/a02_isolated_outcrop_paper_geometry.json
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "engine"))
os.chdir(os.path.join(ROOT, "engine"))

import isolated_outcrop_1cm as ISO
from inversion_helpers import (prism_cell_operator, linear_mc, dir_err_deg, seed_for,
                               M_NRM_MIN, SIGMA_B, STANDOFF, PRISM_DIMS)
from core import generate_traverse

SPACING = 0.01
SENSOR_Z_BODY_FRAME = 0.0      # = 0.5 m above the ground for a prism standing on z = 0
SURVEYS = {"drive_around": ("full_orbit", 5.0), "drive_by": ("drive_by", 6.0)}
RATIOS = [0.1, 0.5, 1, 2, 3, 4, 5, 10, 25, 50, 100]
PGRID = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
RGRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
N_BASE, N_SWEEP, N_TYP, N_WORST = 1000, 200, 100, 32
M_TRUE = M_NRM_MIN * np.array([0.0, 0.0, -1.0])
BND = ISO.bnd                                          # prism centred on the origin


def survey(name):
    kind, approach = SURVEYS[name]
    sp = {"cx": 0.0, "cy": 0.0, "sx": PRISM_DIMS[0], "sy": PRISM_DIMS[1], "sz": PRISM_DIMS[2]}
    return np.asarray(generate_traverse(traverse_type=kind, shape_params=sp,
                                        sensor_height=SENSOR_Z_BODY_FRAME, spacing=SPACING,
                                        orbit_offset=STANDOFF, approach_distance=approach,
                                        shape="rect"), float)


def stats(a):
    a = np.asarray(a, float)
    return {k: float(v) for k, v in dict(
        median=np.median(a), mean=np.mean(a), std=np.std(a), p5=np.percentile(a, 5),
        p25=np.percentile(a, 25), p75=np.percentile(a, 75), p95=np.percentile(a, 95),
        max=np.max(a)).items()} | {"N": int(a.size)}


def recover(G, mt, n, seed):
    M_rec, rms = linear_mc(G, mt, G, SIGMA_B, n, seed)
    th = np.array([dir_err_deg(m, mt) for m in M_rec])
    dm = 100 * (np.linalg.norm(M_rec, axis=1) - np.linalg.norm(mt)) / np.linalg.norm(mt)
    return {"dir_err_deg": stats(th), "dM_signed_pct": stats(dm), "dM_abs_pct": stats(np.abs(dm)),
            "resid_rms": stats(rms)}


def main():
    t0 = time.time()
    out = {"meta": dict(script="analysis/a02_isolated_outcrop_paper_geometry.py",
                        geometry="prism 4x4x1 m on ground (z=0..1), sensor z=0.5 m, standoff 1 m",
                        spacing_m=SPACING, sigma_B_nT=SIGMA_B, M_true_Am=M_TRUE.tolist(),
                        drive_by_extent_m=[-8, 8], ratios=RATIOS, pointing_deg=PGRID, range_m=RGRID,
                        N_baseline=N_BASE, N_sweep=N_SWEEP, N_lidar_typical=N_TYP, N_lidar_worst=N_WORST,
                        seeds="seed_for('a02:<part>:<survey>:<point>')"),
           "baseline": {}, "detection_sweep": {}, "lidar_grid": {}}
    for name in SURVEYS:
        sxyz = survey(name)
        G = prism_cell_operator(BND, sxyz)
        d = (G @ M_TRUE).reshape(-1, 3)
        b = recover(G, M_TRUE, N_BASE, seed_for(f"a02:base:{name}"))
        b.update(n_stations=int(len(sxyz)), x_extent_m=[float(sxyz[:, 0].min()), float(sxyz[:, 0].max())],
                 peak_field_nT=float(np.max(np.linalg.norm(d, axis=1))),
                 signal_rms_per_axis_nT=float(np.sqrt(np.mean(d ** 2))))
        out["baseline"][name] = b
        print(f"[baseline {name}] n={b['n_stations']} peak={b['peak_field_nT']:.3f} nT "
              f"theta med {b['dir_err_deg']['median']:.2f} mean {b['dir_err_deg']['mean']:.2f} | "
              f"|dM| med {b['dM_abs_pct']['median']:.2f}% | signed mean {b['dM_signed_pct']['mean']:+.2f}% "
              f"std {b['dM_signed_pct']['std']:.2f}%", flush=True)
        out["detection_sweep"][name] = []
        for r in RATIOS:
            s = recover(G, M_TRUE * r, N_SWEEP, seed_for(f"a02:sweep:{name}:{r}"))
            s["ratio"] = r
            out["detection_sweep"][name].append(s)
            print(f"   ratio {r:6.1f}: theta med {s['dir_err_deg']['median']:6.2f}  |dM| med {s['dM_abs_pct']['median']:8.3f}%", flush=True)
    for name in SURVEYS:
        sxyz = survey(name)
        out["lidar_grid"][name] = []
        for pd in PGRID:
            for rm in RGRID:
                de, me = ISO.warped_recovery(BND, sxyz, M_TRUE, SIGMA_B, N_TYP,
                                             seed_for(f"a02:lidar:{name}:p{pd}:r{rm}:typ"), pd, rm, "random")
                dw, mw = ISO.warped_recovery(BND, sxyz, M_TRUE, SIGMA_B, N_WORST,
                                             seed_for(f"a02:lidar:{name}:p{pd}:r{rm}:worst"), pd, rm, "worst")
                rec = {"pointing_deg": pd, "range_m": rm,
                       "typical": {"dir_err_deg": stats(de), "dM_abs_pct": stats(np.abs(me)), "dM_signed_pct": stats(me)},
                       "worst": {"dir_err_deg": stats(dw), "dM_abs_pct": stats(np.abs(mw))}}
                out["lidar_grid"][name].append(rec)
                print(f"   lidar {name:12s} p={pd:.0f} r={rm:.2f}: theta med {rec['typical']['dir_err_deg']['median']:5.2f} "
                      f"|dM| med {rec['typical']['dM_abs_pct']['median']:5.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    out["meta"]["wall_clock_s"] = time.time() - t0
    os.makedirs(os.path.join(ROOT, "results", "analysis"), exist_ok=True)
    fn = "a02_isolated_outcrop_paper_geometry.json"
    with open(os.path.join(ROOT, "results", "analysis", fn), "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote results/analysis/{fn} ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()

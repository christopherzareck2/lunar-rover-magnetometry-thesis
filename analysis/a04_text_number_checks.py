"""a04 - Numbers quoted in Sections 4.2 and 4.4 that a02 and a03 do not print.

  1. Isolated outcrop, paper geometry: per-component std of the recovered M (drive-by / drive-around),
     and 95th percentile of the direction error (drive-around).
  2. Imaging mis-pose grid: does pointing or range error dominate direction and intensity errors?
  3. Three-layer: nearest-face distance of each flow from the ground traverse (y = -1, z = 0.5).

Run:  python analysis/a04_text_number_checks.py
Out:  results/analysis/a04_text_number_checks.json
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "engine")); sys.path.insert(0, HERE); os.chdir(os.path.join(ROOT, "engine"))
import a02_isolated_outcrop_paper_geometry as A2
from inversion_helpers import prism_cell_operator, linear_mc, seed_for, SIGMA_B

out = {}
for name in A2.SURVEYS:
    sxyz = A2.survey(name)
    G = prism_cell_operator(A2.BND, sxyz)
    M, _ = linear_mc(G, A2.M_TRUE, G, SIGMA_B, 1000, seed_for(f"a02:base:{name}"))  # same seed as a02 baseline
    out[f"{name}_component_std_Am"] = dict(zip("xyz", M.std(axis=0).tolist()))
a = out["drive_by_component_std_Am"]; b = out["drive_around_component_std_Am"]
out["drive_by_over_drive_around_std"] = {k: a[k] / b[k] for k in "xyz"}

res = json.load(open(os.path.join(ROOT, "results", "analysis", "a02_isolated_outcrop_paper_geometry.json")))
out["drive_around_theta_p95"] = res["baseline"]["drive_around"]["dir_err_deg"]["p95"]
out["drive_by_theta_p95"] = res["baseline"]["drive_by"]["dir_err_deg"]["p95"]
for name, g in res["lidar_grid"].items():
    cell = {(r["pointing_deg"], r["range_m"]): r["typical"] for r in g}
    out[f"lidar_{name}"] = {
        "corner_0_0": [cell[(0.0, 0.0)]["dir_err_deg"]["median"], cell[(0.0, 0.0)]["dM_abs_pct"]["median"]],
        "pointing_6_range_0": [cell[(6.0, 0.0)]["dir_err_deg"]["median"], cell[(6.0, 0.0)]["dM_abs_pct"]["median"]],
        "pointing_0_range_0.30": [cell[(0.0, 0.30)]["dir_err_deg"]["median"], cell[(0.0, 0.30)]["dM_abs_pct"]["median"]],
        "pointing_6_range_0.30": [cell[(6.0, 0.30)]["dir_err_deg"]["median"], cell[(6.0, 0.30)]["dM_abs_pct"]["median"]]}

ys, zs = -1.0, 0.5
out["three_layer_nearest_face_m"] = [float(np.hypot(0 - ys, max(zb - zs, 0) + max(zs - zt, 0))) for zb, zt in [(0, 5), (5, 10), (10, 15)]]
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(ROOT, "results", "analysis", "a04_text_number_checks.json"), "w"), indent=1)

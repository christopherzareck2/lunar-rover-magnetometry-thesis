"""a05 - Background-magnetization Monte Carlo and correlated-noise sweep for the isolated outcrop
(paper Sections 3.3.1, 5.1 and 5.5.2).

Runs engine/scripts/background_magnetization_mc.py and engine/scripts/correlated_noise_requirement.py in the
paper geometry: a 4 x 4 x 1 m prism standing on the ground, sensor at mid-height (0.5 m above the ground),
drive-by from x = -8 to +8 m, as in analysis/a02. The survey helper in both scripts is replaced by one that
builds this geometry (the prism is centred on the origin, so the sensor is at z = 0 and the background slab
is attached to the prism base at z = -0.5 m).

Run:  python analysis/a05_background_and_correlated_noise.py [background|noise|both]
Out:  results/analysis/a05_background_magnetization_mc_paper_geometry.json
      results/analysis/a05_correlated_noise_requirement_paper_geometry.json
"""
import os, sys, json, importlib

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ENGINE = os.path.join(ROOT, "engine")
sys.path.insert(0, ENGINE); sys.path.insert(0, os.path.join(ENGINE, "scripts")); os.chdir(ENGINE)

import inversion_helpers as H
from core import generate_traverse

NOTE = ("paper geometry: sensor at z = 0 in the body frame of the prism centred on the origin, i.e. mid-height "
        "of a 4x4x1 m prism standing on the ground (sensor 0.5 m above ground); drive-by x = -8..+8 m. "
        "'sensor_height_m' in meta is the body-frame value. Run by analysis/a05_background_and_correlated_noise.py")


def paper_survey(traverse_type, standoff, dims=H.PRISM_DIMS, spacing=0.5, center=(0., 0., 0.)):
    sp = {"cx": center[0], "cy": center[1], "sx": dims[0], "sy": dims[1], "sz": dims[2]}
    approach = 6.0 if traverse_type == "drive_by" else 5.0
    return generate_traverse(traverse_type=traverse_type, shape_params=sp, sensor_height=0.0,
                             spacing=spacing, orbit_offset=standoff, approach_distance=approach, shape="rect")


def run(modname, outname):
    H.outcrop_survey = paper_survey
    H.SENSOR_H = 0.0
    if "correlated_noise_sensitivity" in sys.modules:
        sys.modules["correlated_noise_sensitivity"].outcrop_survey = paper_survey
    mod = importlib.import_module(modname)
    mod.outcrop_survey = paper_survey
    mod.SENSOR_H = 0.0
    out = os.path.join(ROOT, "results", "analysis", outname)
    mod.RESULTS_PATH = out
    mod.main()
    d = json.load(open(out))
    d.setdefault("meta", {})["geometry_note"] = NOTE
    json.dump(d, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    os.makedirs(os.path.join(ROOT, "results", "analysis"), exist_ok=True)
    if which in ("background", "both"):
        run("background_magnetization_mc", "a05_background_magnetization_mc_paper_geometry.json")
    if which in ("noise", "both"):
        run("correlated_noise_requirement", "a05_correlated_noise_requirement_paper_geometry.json")

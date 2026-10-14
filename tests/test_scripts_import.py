"""Every frozen simulation script still imports through the engine aliases (without running its main()).

Each script is imported in a fresh interpreter started in the repository root with ONLY its own folder handling
sys.path (as when it is run as `python <script>`), so this checks the compatibility aliases in engine/ exactly as
the scripts use them. analysis/a04 has no main guard (it computes and writes at import), so it is only compiled.
"""
import os
import py_compile
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUARDED = ["analysis/a02_isolated_outcrop_paper_geometry.py", "analysis/a02b_detection_sweep_N1000.py",
           "analysis/a03_three_layer_direction_vs_distance.py", "analysis/a05_background_and_correlated_noise.py",
           "analysis/a06_isolated_robustness.py", "analysis/a07_three_layer_wall_extent.py",
           "analysis/a08_skatepark_neighbor_directions.py", "analysis/a09_systematics.py",
           "engine/inversion_approach_comparison.py", "engine/skatepark_1cm_dr_sweep.py", "engine/three_layer_wall.py",
           "engine/scripts/background_magnetization_mc.py", "engine/scripts/correlated_noise_requirement.py",
           "engine/scripts/elevated_traverse_height_sweep.py"]

RUNNER = ("import runpy, sys, os; p = sys.argv[1]; sys.path.insert(0, os.path.dirname(os.path.abspath(p))); "
          "runpy.run_path(p, run_name='imported'); print('IMPORT-OK')")


@pytest.mark.parametrize("script", GUARDED)
def test_script_imports_through_aliases(script):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}     # no help from PYTHONPATH=src
    env["MPLBACKEND"] = "Agg"
    r = subprocess.run([sys.executable, "-c", RUNNER, os.path.join(ROOT, script)], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "IMPORT-OK" in r.stdout, r.stderr[-1500:]


def test_a04_compiles():
    py_compile.compile(os.path.join(ROOT, "analysis", "a04_text_number_checks.py"), doraise=True)

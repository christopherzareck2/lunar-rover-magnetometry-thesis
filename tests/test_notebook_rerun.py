"""The notebook's RERUN flag must do what it says.

  RERUN = False  Part 1 draws no noise and runs no inversion (any attempt raises), reads the stored results, and
                 Part 2 plus the summary still reproduce all 374 numbers.
  RERUN = True   Part 1 draws the noise, inverts it and asserts agreement with the stored single draw.

The notebook's code cells are run in order in one fresh Python process (no Jupyter kernel needed). The final
"rerun the simulations" cell is skipped for RERUN = True because it would take hours.
Run from the repository root:  python -m unittest discover -s tests -v
"""
import os, sys, json, subprocess, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "notebook", "reproduce_paper.ipynb")

RUNNER = r'''
import sys, json, os
with open(sys.argv[1]) as f:
    nb = json.load(f)
rerun = sys.argv[2] == "1"
import numpy as np
def forbid():
    def _boom(*a, **k):
        raise AssertionError("random numbers or an inversion were requested while RERUN = False")
    np.random.default_rng = _boom
    np.linalg.solve = _boom
code = [''.join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
if rerun:
    code = code[:-1]                      # do not launch the multi-hour reruns
g = {"__name__": "__main__"}
for i, src in enumerate(code):
    exec(compile(src, f"<code cell {i}>", "exec"), g)
    if i == 0 and not rerun:
        forbid()                          # after the imports; from here on nothing may draw or invert
print("NOTEBOOK-OK", g["RERUN"], g["count"]())
'''


def run_notebook(rerun):
    env = dict(os.environ, MPLBACKEND="Agg", LUNAR_RERUN="1" if rerun else "0")
    return subprocess.run([sys.executable, "-c", RUNNER, NB, "1" if rerun else "0"], cwd=os.path.join(ROOT, "notebook"),
                          capture_output=True, text=True, env=env, timeout=1200)


class NotebookRerunFlag(unittest.TestCase):
    def test_rerun_false_draws_nothing_and_loads_stored_results(self):
        r = run_notebook(False)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("NOTEBOOK-OK False (374, 374)", r.stdout)
        self.assertIn("RERUN = False: no noise drawn", r.stdout)
        self.assertIn("stored single-draw result", r.stdout)
        self.assertIn("none redrawn", r.stdout)

    def test_rerun_true_redraws_and_matches_stored(self):
        r = run_notebook(True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("NOTEBOOK-OK True (374, 374)", r.stdout)
        self.assertNotIn("none redrawn", r.stdout)
        self.assertIn("median direction error 3.0 deg (stored 3.0)", r.stdout)

    def test_stored_single_draw_matches_recomputation(self):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "notebook", "part1_single_draw.py"), "--check"],
                           cwd=ROOT, capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr[-2000:])

    def test_rerun_time_statements_agree(self):
        """README and notebook must give the same rerun time (3.5 to 4 hours); the old '2.5' must not return."""
        for path in (os.path.join(ROOT, "README.md"), NB):
            with open(path) as f:
                text = f.read()
            self.assertIn("3.5 to 4 hours", text, path)
            self.assertNotIn("2.5 hours", text, path)
            self.assertNotIn("2.5 h ", text, path)


if __name__ == "__main__":
    unittest.main()

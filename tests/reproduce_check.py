"""Quick reproduction check (about a minute): rerun two cheap parts of the analysis from scratch and compare
them with the stored results in results/analysis/.

  a06 part E  sampling interval vs noise ASD (Table 3, Section 5.1)
  a09 part M  heterogeneous ground vs depth to its top (Section 5.5)

Every run is seeded (seed_for in engine/inversion_helpers.py), so the numbers must match to rounding.
Run from the repository root:  python tests/reproduce_check.py
"""
import os, sys, json, io, contextlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import a06_isolated_robustness as A6        # changes directory to engine/
import a09_systematics as A9


def leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{p}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{p}[{i}]")
    elif isinstance(o, (int, float)) and o is not None:
        yield p, float(o)


def compare(name, new, stored, rtol=1e-6):
    a, b = dict(leaves(new)), dict(leaves(stored))
    common = sorted(set(a) & set(b))
    bad = [k for k in common if not np.isclose(a[k], b[k], rtol=rtol, atol=1e-9)]
    print(f"{name}: {len(common)} numbers compared, {len(bad)} differ")
    for k in bad[:10]:
        print(f"   {k}: rerun {a[k]:.6g} vs stored {b[k]:.6g}")
    return not bad and len(common) > 0


ok = True
stored6 = json.load(open(os.path.join(ROOT, "results", "analysis", "a06_isolated_robustness.json")))
out = {}
with contextlib.redirect_stdout(io.StringIO()):
    A6.part_E(out)
ok &= compare("a06 part E", out["E_sampling_vs_ASD"], stored6["E_sampling_vs_ASD"])
stored9 = json.load(open(os.path.join(ROOT, "results", "analysis", "a09_systematics.json")))
out = {}
with contextlib.redirect_stdout(io.StringIO()):
    A9.part_M(out)
ok &= compare("a09 part M", out["M_background_depth"], stored9["M_background_depth"])
print("REPRODUCED" if ok else "MISMATCH")
sys.exit(0 if ok else 1)

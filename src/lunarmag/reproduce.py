"""Fast reproduction from stored results (``make reproduce``, a few minutes) and comparison of a full rerun.

``fast()`` reruns every seeded analysis part that takes seconds (a06 parts A-I, a09 parts M and N, the notebook's
single draw), requires the stored numbers bit for bit, recounts the 374 quoted numbers, and redraws every thesis
figure to a scratch folder to check that it matches the manifest. ``compare(new_results, stored_results)``
compares a full rerun (``make rerun``) with the stored results, file by file.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

from .paths import ANALYSIS, ROOT

A6_PARTS = "ABCDEFGHI"
A9_PARTS = "MN"


def _leaves(o: Any, p: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(o, dict):
        yield f"{p}<dict>", len(o)
        for k, v in o.items():
            yield from _leaves(v, f"{p}/{k}")
    elif isinstance(o, list):
        yield f"{p}<list>", len(o)
        for i, v in enumerate(o):
            yield from _leaves(v, f"{p}[{i}]")
    else:
        yield p, o


def diff(new: Any, ref: Any, rtol: float = 0.0, ignore: tuple[str, ...] = ("wall_clock_s", "runtime")) -> list[str]:
    """Paths whose values differ (exactly, or beyond ``rtol`` for floats); keys containing ``ignore`` are skipped."""
    a, b = dict(_leaves(new)), dict(_leaves(ref))
    out = [f"{k}: only in one" for k in sorted(set(a) ^ set(b)) if not any(i in k for i in ignore)]
    for k in sorted(set(a) & set(b)):
        if any(i in k for i in ignore):
            continue
        x, y = a[k], b[k]
        if isinstance(x, float) and isinstance(y, float) and not (math.isfinite(x) and math.isfinite(y)):
            if not (x == y or (math.isnan(x) and math.isnan(y))):
                out.append(f"{k}: {x!r} vs {y!r}")
        elif isinstance(x, float) and isinstance(y, (int, float)) and rtol > 0:
            if abs(x - y) > rtol * max(abs(x), abs(y)):
                out.append(f"{k}: {x!r} vs {y!r}")
        elif x != y:
            out.append(f"{k}: {x!r} vs {y!r}")
    return out


def _import_analysis(name: str):
    cwd = os.getcwd()
    if str(ANALYSIS) not in sys.path:
        sys.path.insert(0, str(ANALYSIS))
    try:
        return __import__(name)
    finally:
        os.chdir(cwd)


def fast(figures: bool = True) -> bool:
    ok = True
    t0 = time.time()
    A6 = _import_analysis("a06_isolated_robustness")
    A9 = _import_analysis("a09_systematics")
    s6 = json.load(open(ROOT / "results" / "analysis" / "a06_isolated_robustness.json"))
    s9 = json.load(open(ROOT / "results" / "analysis" / "a09_systematics.json"))
    for mod, parts, stored, tag in ((A6, A6_PARTS, s6, "a06"), (A9, A9_PARTS, s9, "a09")):
        for p in parts:
            out: dict = {}
            with contextlib.redirect_stdout(io.StringIO()):
                getattr(mod, "part_" + p)(out)
            for key, val in out.items():
                d = diff(json.loads(json.dumps(val)), stored[key], rtol=float(os.environ.get("LUNARMAG_RTOL", "0")))
                n = sum(1 for _ in _leaves(val))
                ok &= not d
                print(f"  {tag} part {p} {key}: {n} numbers, {'identical' if not d else f'{len(d)} DIFFER, e.g. ' + d[0]}")
    sys.path.insert(0, str(ROOT / "notebook"))
    import part1_single_draw as P1
    import paper_numbers as PN
    same = P1.check(P1.compute(), json.load(open(ROOT / "results" / "notebook" / "part1_single_draw.json")))
    ok &= same
    print(f"  notebook single draw: {'matches stored' if same else 'DIFFERS'}")
    m, n = PN.count()
    ok &= m == n
    print(f"  quoted numbers (notebook/paper_numbers.py): {m} of {n} reproduce")
    if figures:
        from . import figures as F
        man_path = ROOT / "figures" / "manifest.json"
        man = {e.get("figure_id", e["figure"]): e for e in json.load(open(man_path))["figures"]} if man_path.exists() else {}
        with tempfile.TemporaryDirectory() as tmp:
            for spec in F.FIGURES:
                rec = F.build(spec, out_dir=tmp, png_dir=None, values_dir=tmp)
                ref = man.get(spec.number)
                same_vals = ref is None or rec["plotted_values_sha256"] == ref["plotted_values_sha256"]
                same_pdf = ref is None or rec["pdf_sha256"] == ref["pdf_sha256"]
                ok &= same_vals
                print(f"  Fig. {spec.number}: plotted values {'match' if same_vals else 'DIFFER from'} the manifest; "
                      f"PDF {'identical' if same_pdf else 'differs (fonts or library version)'}")
    print(f"{'REPRODUCED' if ok else 'MISMATCH'} ({time.time() - t0:.0f} s)")
    return ok


def compare(new_dir: Path | str, ref_dir: Path | str = ROOT / "results", rtol: float = 0.0) -> bool:
    """Compare every JSON under ``new_dir`` with the file of the same relative path under ``ref_dir``."""
    new_dir, ref_dir = Path(new_dir), Path(ref_dir)
    ok = True
    for f in sorted(new_dir.rglob("*.json")):
        ref = ref_dir / f.relative_to(new_dir)
        if not ref.exists():
            print(f"  {f.relative_to(new_dir)}: no stored counterpart"); continue
        d = diff(json.load(open(f)), json.load(open(ref)), rtol=rtol)
        ok &= not d
        print(f"  {f.relative_to(new_dir)}: {'identical' if not d else f'{len(d)} differences, e.g. {d[0]}'}")
    return ok

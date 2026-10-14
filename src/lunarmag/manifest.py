"""The figure manifest: thesis figure -> script -> stored result file -> sha256.

``write(records)`` writes ``figures/manifest.json`` (machine-readable, checked by ``tests/test_figures.py``) and
``figures/MANIFEST.md`` (the same table for people), with the package and library versions that drew the figures.
(The git commit is left out on purpose: the manifest is committed with the figures it describes.)
"""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from . import __version__
from .paths import ROOT

MANIFEST_JSON = ROOT / "figures" / "manifest.json"
MANIFEST_MD = ROOT / "figures" / "MANIFEST.md"
GEOMETRY_SOURCES = {"3.1": "geometry constants (as `analysis/a02`, `a09`)",
                    "3.2": "scene definition in `engine/skatepark_1cm_dr_sweep.py`",
                    "3.3": "`inversion_helpers.three_layer_mesh` and the traverse constants"}


def environment(git: bool = True) -> dict[str, str]:
    import matplotlib
    import numpy
    out = {"lunarmag": __version__, "python": platform.python_version(), "numpy": numpy.__version__,
           "matplotlib": matplotlib.__version__}
    if not git:
        return out
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True)
        if rev.returncode == 0:
            out["git_commit"] = rev.stdout.strip() + ("+uncommitted" if dirty.stdout.strip() else "")
    except OSError:
        pass
    return out


def _num_key(n: str) -> list:
    out = []
    for x in n.split("."):
        out.append((0, int(x)) if x.isdigit() else (1, x))
    return out


def write(records: list[dict[str, Any]], external: list[dict[str, Any]] | None = None) -> tuple[Path, Path]:
    from .figures.external import figure_numbers
    nums = figure_numbers()
    for r in records:                     # the thesis number from the last build, if any (chapters move)
        r["figure_id"] = r.get("figure_id", r["figure"])
        r["figure"] = nums.get(r["label"], r["figure_id"])
    external = external or []
    allrec = sorted(records + external, key=lambda r: _num_key(r["figure"]) if r["figure"][:1].isalnum() and r["figure"][:1] != "n" else [(2, "")])
    data = {"description": "Thesis figures: file, the code that draws it, the stored inputs it reads (sha256), and the "
                           "sha256 of the figure PDF (and of its plotted values for the lunarmag figures). "
                           "Regenerate with `make figures`. Figure numbers are those of the thesis (kept from the previous manifest).",
            "environment": environment(git=False), "figures": records, "external_figures": external}
    MANIFEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_JSON.write_text(json.dumps(data, indent=1) + "\n")
    lines = ["# Figure manifest", "",
             "Written by `make figures` (`lunarmag figures`). Every PDF in `thesis/MIT-thesis-template/figures/` is "
             "listed here, drawn in the thesis house style (`lunarmag.plotting`) and made PDF/A-1b ready (complete "
             "CIDSets, no transparency). `tests/test_figures.py` checks that the lunarmag figures plot exactly the "
             "values of the companion-paper scripts, that every figure PDF is PDF/A-1b clean, and that these hashes "
             "match the files in the thesis. Figure numbers are those of the thesis (kept from the previous manifest).", "",
             f"Environment: {', '.join(f'{k} {v}' for k, v in data['environment'].items())}.", "",
             "| Fig. | File (thesis/MIT-thesis-template/figures/) | Drawn by | Stored inputs read (sha256) | PDF sha256 |",
             "|---|---|---|---|---|"]
    for r in allrec:
        res = "<br>".join(f"`{k}` ({v[:12]})" for k, v in r["results"].items()) or \
            GEOMETRY_SOURCES.get(r.get("figure_id", ""), "")
        lines.append(f"| {r['figure']} | `{Path(r['file']).name}` | `{r['script']}` | {res} | `{r['pdf_sha256'][:16]}` |")
    lines += ["", "Plotted values of the lunarmag figures: `figures/values/<file>.values.json`. Previews (PNG, 300 dpi): "
              "`figures/preview/` (Chapters 3-5), `studies/figures/`, `figures/ch6/png/`, `ml/figures/`.", ""]
    MANIFEST_MD.write_text("\n".join(lines))
    return MANIFEST_JSON, MANIFEST_MD

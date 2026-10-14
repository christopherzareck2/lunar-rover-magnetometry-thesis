"""Repository paths used by the figure, manifest and check code.

The package lives in ``<repo>/src/lunarmag``; stored results are in ``<repo>/results`` and the thesis LaTeX in
``<repo>/thesis/MIT-thesis-template``. Set ``LUNARMAG_ROOT`` to point at a different checkout.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("LUNARMAG_ROOT", Path(__file__).resolve().parents[2]))
RESULTS = ROOT / "results"
RES_ANALYSIS = RESULTS / "analysis"
RES_ENGINE = RESULTS / "engine"
RES_NOTEBOOK = RESULTS / "notebook"
THESIS_DIR = ROOT / "thesis" / "MIT-thesis-template"
THESIS_FIGURES = THESIS_DIR / "figures"
ENGINE = ROOT / "engine"
ANALYSIS = ROOT / "analysis"


def result(name: str) -> Path:
    """Path of a stored result file by name, searching results/analysis then results/engine."""
    for sub in (RES_ANALYSIS, RES_ENGINE, RES_NOTEBOOK):
        p = sub / name
        if p.exists():
            return p
    raise FileNotFoundError(name)


def rel(p: Path | str) -> str:
    """Path relative to the repository root, with forward slashes (absolute if outside the repository)."""
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()

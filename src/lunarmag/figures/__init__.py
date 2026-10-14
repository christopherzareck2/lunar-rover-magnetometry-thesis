"""The thesis figures, one entry per figure, and the code that builds them.

``build(number)`` draws a figure from the stored results in the thesis style, writes ``<stem>.pdf`` (PDF/A-1b
ready: no transparency, CIDSet added) and a ``.png`` preview to ``thesis/MIT-thesis-template/figures/``, a 300 dpi PNG
preview to ``figures/preview/`` and the plotted numbers to ``figures/values/<stem>.values.json``, and returns a manifest record (figure, script, result
files, output sha256). ``lunarmag figures`` and ``make figures`` call it for every figure.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .. import plotting as st
from ..paths import ROOT, THESIS_FIGURES, rel, result


@dataclass(frozen=True)
class FigureSpec:
    number: str                    #: thesis figure number in the 2026-10-13 build, e.g. "4.1"
    stem: str                      #: file name used by \includegraphics (unchanged from the thesis source)
    label: str                     #: LaTeX label of the figure
    width: str                     #: width key of plotting.WIDTH = the \includegraphics width in the thesis
    module: str                    #: module that holds data_/draw_ functions
    func: str                      #: suffix of data_<func> / draw_<func>
    results: tuple[str, ...]       #: stored result files read
    paper_script: Optional[str]    #: companion-paper script that drew the earlier version (for value checks)
    title: str = ""
    kwargs: dict = field(default_factory=dict)


FIGURES: list[FigureSpec] = [
    FigureSpec("3.1", "fig01_isolated_outcrop_geometry", "fig:outcrop_geometry", "full", "geometry_figs",
               "outcrop_geometry", (), "figures/scripts/fig01_isolated_outcrop_geometry.py",
               "Isolated outcrop model and survey paths"),
    FigureSpec("3.2", "fig02_skatepark_geometry", "fig:skatepark_geometry", "full", "geometry_figs",
               "skatepark_geometry", (), "figures/scripts/fig02_skatepark_geometry.py",
               "Skate park scene"),
    FigureSpec("3.3", "fig03_three_layer_geometry", "fig:threelayer_geometry", "full", "geometry_figs",
               "threelayer_geometry", (), "figures/scripts/fig03_three_layer_geometry.py",
               "Three-layer wall model and traverses"),
    FigureSpec("4.1", "fig04_detection_envelope", "fig:detection_envelope", "0.7", "results_figs",
               "detection_envelope", ("a02b_detection_sweep_N1000.json",),
               "figures/scripts/fig04_detection_envelope.py", "Recovery vs NRM field strength"),
    FigureSpec("4.2", "fig05_lidar_sensitivity", "fig:lidar_sensitivity", "full", "results_figs",
               "lidar_sensitivity", ("a02_isolated_outcrop_paper_geometry.json",),
               "figures/scripts/fig05_lidar_sensitivity.py", "Sensitivity to imaging mis-pose"),
    FigureSpec("4.3", "fig06_skatepark_recovery", "fig:skatepark_recovery", "0.7", "results_figs",
               "skatepark_recovery", ("skatepark_1cm_dr_sweep_results.json",),
               "figures/scripts/fig06_skatepark_recovery.py", "Central-outcrop recovery vs neighbor gap"),
    FigureSpec("4.4", "fig07_second_traverse_height", "fig:second_traverse", "0.7", "results_figs",
               "second_traverse_height", ("elevated_traverse_height_sweep_results.json",),
               "figures/scripts/fig07_second_traverse_height.py", "Three-flow recovery vs second-traverse height"),
    FigureSpec("5.1", "fig08_noise_requirement", "fig:noise_requirement", "full", "results_figs",
               "noise_requirement", ("a06_isolated_robustness.json",),
               "figures/scripts/fig08_noise_requirement.py", "Correlated noise and slow drift"),
    FigureSpec("5.2", "thesis_fig_budget", "fig:budget", "0.9", "results_figs", "budget",
               ("a09_systematics.json",), "thesis/MIT-thesis-template/figures/src/thesis_fig_budget.py",
               "Combined error budget"),
]

BY_NUMBER = {f.number: f for f in FIGURES}
VALUES_DIR = ROOT / "figures" / "values"
PREVIEW_DIR = ROOT / "figures" / "preview"


def _module(spec: FigureSpec):
    import importlib
    return importlib.import_module(f"{__name__}.{spec.module}")


def figure_data(spec: FigureSpec) -> dict[str, Any]:
    """The numbers the figure plots (read from the stored results; no simulation)."""
    return getattr(_module(spec), f"data_{spec.func}")()


def draw(spec: FigureSpec, data: Optional[dict] = None, **kw):
    """Draw a figure (inside the thesis style) and return the matplotlib Figure."""
    import logging
    import warnings

    class _Catch(logging.Handler):
        def __init__(self):
            super().__init__(logging.WARNING); self.msgs = []

        def emit(self, record):
            self.msgs.append(record.getMessage())

    mod = _module(spec)
    data = figure_data(spec) if data is None else data
    h = _Catch()
    log = logging.getLogger("matplotlib")
    log.addHandler(h)
    try:
        with st.style(), warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*[Gg]lyph.*missing.*")
            fig = getattr(mod, f"draw_{spec.func}")(data, **{**spec.kwargs, **kw})
            fig.canvas.draw()
            st.lint_text(fig)
    finally:
        log.removeHandler(h)
    glyph = [m for m in h.msgs if "glyph" in m.lower()]
    if glyph:
        raise ValueError(f"Figure {spec.number}: missing glyphs: {glyph[:3]}")
    return fig


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(spec: FigureSpec, out_dir: Path | str = THESIS_FIGURES, pdfa: bool = True,
          png_dir: Path | str | None = PREVIEW_DIR, values_dir: Path | str = VALUES_DIR, **kw) -> dict[str, Any]:
    """Draw and save one figure; returns its manifest record."""
    from .. import pdfa as pdfa_mod
    data = figure_data(spec)
    with st.style():
        fig = draw(spec, data, **kw)
        paths = st.save(fig, out_dir, spec.stem, png_dir=png_dir)
    pdf = Path(paths[0])
    report = pdfa_mod.make_pdfa1b_ready(pdf) if pdfa else {}
    values_dir = Path(values_dir)
    values_dir.mkdir(parents=True, exist_ok=True)
    vpath = values_dir / f"{spec.stem}.values.json"
    vpath.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    script = f"src/lunarmag/figures/{spec.module}.py"
    return {
        "figure": spec.number, "label": spec.label, "title": spec.title, "file": rel(pdf),
        "script": f"{script} (draw_{spec.func})", "command": f"lunarmag figures --only {spec.number}",
        "results": {rel(result(r)): sha256(result(r)) for r in spec.results},
        "plotted_values": rel(vpath), "plotted_values_sha256": sha256(vpath),
        "pdf_sha256": sha256(pdf), "png": rel(paths[1]) if len(paths) > 1 else None,
        "width": spec.width, "pdfa1b_ready": bool(pdfa), **({"pdfa": report} if pdfa else {}),
    }

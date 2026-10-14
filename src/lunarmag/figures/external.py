"""Thesis figures drawn by scripts outside ``lunarmag.figures`` (Chapters 5-7), and the PDF/A pass over every figure.

The figures of Chapters 3-5 that come from the simulation results are drawn by ``lunarmag.figures``. The newer
chapters have their own drivers, which keep their data code next to the study that produced it:

* Chapter 5 studies (``studies/``): Sobol indices and the flow-mean simulation.
* Chapter 6 mission figures (``figures/ch6/``): architecture, ConOps, surface time, Pareto front, boom.
* Image figures (``figures/scripts/fig_target_imagery.py``, ``fig_site_map.py``): real imagery (Chapter 1) and the site map
  (Chapter 6). They read reduced inputs in ``figures/inputs/``, made from public NASA/USGS/PDS downloads by
  ``figures/scripts/prepare_image_inputs.py`` (raw files live in ``data/external/``, not in git).
* Chapter 7 learning study (``ml/``): burial sweep and calibration.

All of them draw with ``lunarmag.plotting`` (the thesis house style). ``run_external`` runs each driver and copies its
PDF into ``thesis/MIT-thesis-template/figures/`` under the name the chapter includes. ``postprocess_dir`` then
makes every PDF in that folder PDF/A-1b ready: complete CIDSets (``pdfa.add_cidset``; an incomplete set is
rebuilt) and a check that no transparency is left. ``make figures`` does both.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..paths import ROOT, THESIS_FIGURES, rel


@dataclass(frozen=True)
class ExternalFigure:
    stem: str                      #: file name in thesis/MIT-thesis-template/figures (as \includegraphics)
    label: str                     #: LaTeX label of the figure (number resolved from the build's .aux)
    script: str                    #: driver, relative to the repository root
    source: str | None             #: PDF the driver writes, if not written straight into the thesis folder
    results: tuple[str, ...]       #: stored inputs it reads (hashed in the manifest)
    title: str


EXTERNAL: list[ExternalFigure] = [
    ExternalFigure("ch1_target_imagery", "fig:intro_target", "figures/scripts/fig_target_imagery.py", None,
                   ("figures/inputs/nac_M113941548L.png", "figures/inputs/nac_M113941548L.json",
                    "figures/inputs/nac_M1345757107L.png", "figures/inputs/nac_M1345757107L.json",
                    "figures/inputs/apollo15_AS15-85-11451.png"), "The target: LROC NAC and Apollo 15 imagery"),
    ExternalFigure("ch5_sobol_indices", "fig:sobol", "studies/fig_sobol.py", "studies/figures/study_sobol_indices.pdf",
                   ("studies/results/sobol.json",), "Sobol indices of the noise-free direction error"),
    ExternalFigure("ch5_flow_mean", "fig:flow_mean", "studies/fig_flow_mean.py", "studies/figures/study_flow_mean.pdf",
                   ("studies/results/flow_mean.json",), "Flow-mean direction error and alpha95 coverage"),
    ExternalFigure("ch6_architecture", "fig:mis_architecture", "figures/ch6/make_ch6.py", None,
                   ("figures/ch6/ch6_architecture.py",), "Mission architecture (diagram)"),
    ExternalFigure("ch6_conops", "fig:mis_conops", "figures/ch6/make_ch6.py", None,
                   ("figures/ch6/ch6_data.py", "results/analysis/a06_isolated_robustness.json", "results/analysis/a09_systematics.json"), "Concept of operations"),
    ExternalFigure("ch6_surface_time", "fig:mis_surface_time", "figures/ch6/make_ch6.py", None,
                   ("figures/ch6/ch6_data.py", "results/analysis/a06_isolated_robustness.json", "results/analysis/a09_systematics.json"), "Surface time per outcrop"),
    ExternalFigure("ch6_pareto", "fig:mis_pareto", "figures/ch6/make_ch6.py", None,
                   ("figures/ch6/ch6_data.py", "results/analysis/a06_isolated_robustness.json", "results/analysis/a09_systematics.json"), "Trade-study Pareto front"),
    ExternalFigure("ch6_boom", "fig:mis_boom", "figures/ch6/make_ch6.py", None,
                   ("figures/ch6/ch6_data.py", "results/analysis/a06_isolated_robustness.json", "results/analysis/a09_systematics.json"), "Boom length trade"),
    ExternalFigure("ch6_site_map", "fig:mis_sites", "figures/scripts/fig_site_map.py", None,
                   ("figures/inputs/lola_hillshade_nearside.png", "figures/inputs/site_coordinates.csv"),
                   "Candidate sites on the nearside (LOLA hillshade)"),
    ExternalFigure("ch7_burial", "fig:ai_burial", "ml/fig_ml.py", "ml/figures/ml_fig1_burial.pdf",
                   ("ml/results/sbi_results.json",), "Learned vs least-squares error against buried extent"),
    ExternalFigure("ch7_calib", "fig:ai_calib", "ml/fig_ml.py", "ml/figures/ml_fig2_calib.pdf",
                   ("ml/results/sbi_results.json", "ml/data/test_prior.npz", "ml/data/test_ood_pose.npz",
                    "ml/data/test_none.npz"), "Network calibration and the Cramer-Rao check"),
]


def run_external(python: str = sys.executable) -> list[ExternalFigure]:
    """Run every driver once (each script only once even if it draws several figures) and copy outputs in place."""
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]))
    for script in dict.fromkeys(e.script for e in EXTERNAL):
        r = subprocess.run([python, str(ROOT / script)], cwd=ROOT, env=env, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"{script} failed:\n{r.stderr[-2000:]}")
        print(f"ran {script}")
    for e in EXTERNAL:
        if e.source:
            shutil.copyfile(ROOT / e.source, THESIS_FIGURES / f"{e.stem}.pdf")
    return EXTERNAL


def postprocess_dir(folder: Path | str = THESIS_FIGURES) -> list[dict[str, Any]]:
    """Make every PDF in ``folder`` PDF/A-1b ready (complete CIDSets, no transparency). Raises on transparency."""
    from .. import pdfa
    out, bad = [], []
    for pdf in sorted(Path(folder).glob("*.pdf")):
        n = pdfa.add_cidset(pdf, replace_incomplete=True)
        tr = pdfa.scan_transparency(pdf)
        if tr:
            bad.append(f"{pdf.name}: {tr}")
        out.append({"file": pdf.name, "cidsets_written": n, "transparency": tr or None})
    if bad:
        raise RuntimeError("transparency left in figure PDFs (PDF/A-1 forbids it): " + "; ".join(bad))
    return out


def figure_numbers(aux_dir: Path | str = ROOT / "thesis" / "build" / "src") -> dict[str, str]:
    """{label: figure number} from the last thesis build's .aux files (empty if the thesis was not built)."""
    import re
    nums: dict[str, str] = {}
    main = Path(aux_dir) / "MIT-Thesis.aux"
    if not main.exists():
        return nums
    text = main.read_text(errors="ignore")
    files = [main] + [Path(aux_dir) / f for f in re.findall(r"\\@input\{([^}]*)\}", text)]   # only what the build used
    for aux in files:
        if aux.exists():
            for m in re.finditer(r"\\newlabel\{(fig:[^}]*)\}\{\{([^}]*)\}", aux.read_text(errors="ignore")):
                nums[m.group(1)] = m.group(2)
    return nums


def _numbers_from_manifest(path: Path | str = ROOT / "figures" / "manifest.json") -> dict[str, str]:
    """{label: figure number} as last written to the manifest (used when the thesis has not been built here)."""
    import json
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return {r["label"]: r["figure"] for r in data.get("external_figures", []) if r.get("figure") != "not included"}


def records() -> list[dict[str, Any]]:
    """Manifest records for the external figures (after ``run_external`` and ``postprocess_dir``).

    Figure numbers come from the last thesis build. In a fresh checkout that has not built the thesis yet, the
    numbers already in ``figures/manifest.json`` are kept.
    """
    from . import sha256
    nums = figure_numbers() or _numbers_from_manifest()
    recs = []
    for e in EXTERNAL:
        pdf = THESIS_FIGURES / f"{e.stem}.pdf"
        recs.append({"figure": nums.get(e.label, "not included"), "label": e.label, "title": e.title, "file": rel(pdf),
                     "script": e.script, "command": "make figures", "source_pdf": e.source,
                     "results": {r: sha256(ROOT / r) for r in e.results if (ROOT / r).exists()},
                     "pdf_sha256": sha256(pdf), "pdfa1b_ready": True})
    return recs

"""Figure: Sobol indices of the noise-free direction error per error source (studies/results/sobol.json).

Requirement set (``--set mitigated_set`` for the other). One row per error source, sorted by the total index of the
vector output. Markers: total index S_T (blue squares) and first-order index S_1 (vermillion circles), with 95%
bootstrap intervals. (A) Output = direction error theta averaged over the six test directions. (B) Output = the
perpendicular direction-error vector of all six directions (generalized indices; additive, so S_1 sums to ~0.9). Thesis style module (lunarmag.plotting) if
present, else plain matplotlib; no transparency (PDF/A-1).

Run:  python studies/fig_sobol.py
Out:  studies/figures/study_sobol_indices[_mitigated_set].pdf (+ .png preview)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
try:
    from lunarmag import plotting as P
except Exception:                      # plain matplotlib fallback
    P = None
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {"burial": "Buried extent", "pose": "Imaging pose", "site": "Site-field curvature", "tilt": "Sensor tilt",
          "induced": "Induced magnetization", "drift": "Slow drift", "centering": "Loop centering"}
TITLES = {"requirement_set": "Requirement set", "mitigated_set": "Mitigated set"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="requirement_set"); a = ap.parse_args()
    d = json.load(open(os.path.join(HERE, "results", "sobol.json")))
    S = d["sets"][a.set]
    panels = [(S["outputs"]["mean_over_directions"], "Direction error $\\theta$"),
              (S["vector_outputs"]["all_directions"], "Direction-error vector")]
    ref = panels[1][0]["groups"]
    order = sorted(ref, key=lambda g: ref[g]["ST"])                       # largest at the top
    blue = P.INTENSITY if P else "#0072B2"; verm = P.DIRECTION if P else "#D55E00"
    ink = P.INK if P else "black"; grid = P.GRID if P else "#E3E3E3"
    ctx = P.style() if P else plt.rc_context({})
    with ctx:
        fig = P.figure("full", 2.7) if P else plt.figure(figsize=(6.5, 2.7))
        axes = fig.subplots(1, 2, sharey=True, gridspec_kw=dict(wspace=0.08, left=0.2, right=0.98, bottom=0.17, top=0.88))
        y = np.arange(len(order))
        for ax, (o, title) in zip(axes, panels):
            G = o["groups"]
            for g_i, g in enumerate(order):
                st, s1 = G[g]["ST"], G[g]["S1"]
                ax.errorbar(st, g_i + 0.15, xerr=[[st - G[g]["ST_ci95"][0]], [G[g]["ST_ci95"][1] - st]], fmt="s",
                            color=blue, ms=4.5, lw=0.9, capsize=2, label=r"Total, $S_T$" if g_i == 0 else None, zorder=3)
                ax.errorbar(s1, g_i - 0.15, xerr=[[s1 - G[g]["S1_ci95"][0]], [G[g]["S1_ci95"][1] - s1]], fmt="o",
                            color=verm, ms=4.5, lw=0.9, capsize=2, label=r"First order, $S_1$" if g_i == 0 else None, zorder=3)
            ax.axvline(0, color=ink, lw=0.6)
            ax.set_xlim(-0.12, 1.0)
            ax.grid(axis="x", color=grid, lw=0.5, zorder=0)
            ax.set_xlabel("Sobol index (share of variance)")
            ax.set_title(f"{title}; sum of $S_1$ = {o['sum_S1']:.2f}", fontsize=8.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(axis="y", length=0)
        axes[0].set_yticks(y); axes[0].set_yticklabels([LABELS[g] for g in order])
        axes[1].legend(loc="lower right", frameon=True, fontsize=7.5)
        name = "study_sobol_indices" + ("" if a.set == "requirement_set" else f"_{a.set}")
        if P:
            P.panel_label(axes[0], "A", x=-0.01, y=1.04); P.panel_label(axes[1], "B", x=-0.01, y=1.04)
            P.lint_text(fig)
            paths = P.save(fig, os.path.join(HERE, "figures"), name, png_dir=os.path.join(HERE, "figures"))
        else:
            os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
            paths = [os.path.join(HERE, "figures", f"{name}.pdf")]
            fig.savefig(paths[0]); fig.savefig(paths[0].replace(".pdf", ".png"), dpi=300)
    print(*paths, sep="\n")


if __name__ == "__main__":
    main()

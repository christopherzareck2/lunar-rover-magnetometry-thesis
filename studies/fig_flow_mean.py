"""Figure: flow-mean direction error and alpha95 coverage for ten outcrops per flow (studies/results/flow_mean.json).

Rows: the three tolerance sets, each with the loop offset shared by all outcrops (+x, the budget's convention) or in an
independent random azimuth per outcrop. All rows include the shared heading error (+-1 deg, common to the flow).
(A) Mean error of the flow-mean direction over 500 flows: one small grey dot per magnetization direction, the vermillion
circle is the mean over the six directions, and the blue square is the Fisher-independence prediction from the
per-outcrop errors of the same runs. (B) Fraction of flows whose Fisher alpha95 contains the true direction; the dashed
line is the nominal 95%.
Thesis style module (lunarmag.plotting) if present, else plain matplotlib; no transparency (PDF/A-1).

Run:  python studies/fig_flow_mean.py [--shared heading_1deg]
Out:  studies/figures/study_flow_mean.pdf (+ .png preview)
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
except Exception:
    P = None
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SETS = [("half_tolerance", "Half tolerance"), ("requirement_set", "Requirement set"), ("mitigated_set", "Mitigated set")]
OFFS = [("shared_offset", "offset shared"), ("random_offset", "offset random")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--shared", default="heading_1deg"); a = ap.parse_args()
    d = json.load(open(os.path.join(HERE, "results", "flow_mean.json")))
    rows = [(s, o, f"{sl}, {ol}") for s, sl in SETS if s in d["sets"] for o, ol in OFFS]
    verm = P.DIRECTION if P else "#D55E00"; blue = P.INTENSITY if P else "#0072B2"
    grey = P.MUTED if P else "#6E6E6E"; guide = P.GUIDE if P else "#9A9A9A"; grid = P.GRID if P else "#E3E3E3"
    ctx = P.style() if P else plt.rc_context({})
    with ctx:
        fig = P.figure("full", 2.8) if P else plt.figure(figsize=(6.5, 2.8))
        axA, axB = fig.subplots(1, 2, sharey=True, gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.08, left=0.255,
                                                                       right=0.98, bottom=0.17, top=0.9))
        y = np.arange(len(rows))[::-1]
        for yi, (s, o, lab) in zip(y, rows):
            case = d["sets"][s]["cases"][f"{o}|{a.shared}"]
            fm = np.array([case[k]["flow_mean_dir_err"]["mean"] for k in case])
            pr = np.array([case[k]["fisher_independence_prediction"]["flow_mean_dir_err_mean"] for k in case])
            cv = np.array([case[k]["coverage"]["value"] for k in case]) * 100
            axA.plot(fm, np.full(fm.size, yi), "o", ms=2.6, color=grey, zorder=2, label="Each direction" if yi == y[0] else None)
            axA.plot(fm.mean(), yi + 0.22, "o", ms=5, color=verm, zorder=3, label="Simulated, mean of directions" if yi == y[0] else None)
            axA.plot(pr.mean(), yi - 0.22, "s", ms=4.5, color=blue, zorder=3,
                     label="Fisher, independent outcrops" if yi == y[0] else None)
            axB.plot(cv, np.full(cv.size, yi), "o", ms=2.6, color=grey, zorder=2)
            axB.plot(cv.mean(), yi + 0.22, "o", ms=5, color=verm, zorder=3)
        axA.set_yticks(y); axA.set_yticklabels([r[2] for r in rows])
        axA.set_xlim(0, None); axA.set_xlabel(r"Mean error of the flow-mean direction ($^{\circ}$)")
        axB.axvline(95, color=guide, ls=(0, (4, 2.5)), lw=0.9, zorder=1)
        axB.text(93.5, y[0] + 0.5, "nominal 95%", ha="right", va="center", fontsize=7.5, color=grey)
        axB.set_xlim(0, 100); axB.set_xlabel(r"Flows with the truth inside $\alpha_{95}$ (%)")
        for ax in (axA, axB):
            ax.grid(axis="x", color=grid, lw=0.5, zorder=0)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(axis="y", length=0)
        axA.legend(loc="center right", frameon=True, handletextpad=0.3, fontsize=7.5)
        if P:
            P.panel_label(axA, "A"); P.panel_label(axB, "B")
            P.lint_text(fig)
            paths = P.save(fig, os.path.join(HERE, "figures"), "study_flow_mean", png_dir=os.path.join(HERE, "figures"))
        else:
            os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
            paths = [os.path.join(HERE, "figures", "study_flow_mean.pdf")]
            fig.savefig(paths[0]); fig.savefig(paths[0].replace(".pdf", ".png"), dpi=300)
    print(*paths, sep="\n")


if __name__ == "__main__":
    main()

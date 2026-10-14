"""Figure 6.1: mission architecture (schematic, not to scale except panel B).

(A) Side view: a CLPS-class lander with a reference fluxgate on a mast, a small rover with a boom-mounted
three-axis fluxgate, mast cameras or LiDAR, an arm-mounted susceptibility probe and optional ground-penetrating
radar, at an outcrop that protrudes through the regolith with an unknown buried extent b.
(B) Plan view, to scale: the 4 x 4 m outcrop of Sec. 3.3.3, the 24 m loop at 1 m standoff and the 40 m loop at
3 m standoff (each driven in both directions), the 10 m baseline traverse, and a calibration-turn site.
No data are plotted; geometry follows Ch. 3.
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

import ch6_data as D
from ch6_common import finish, st

INK, MUTED, SHADE = st.INK, st.MUTED, st.SHADE
REG = "#D9CBB0"          # regolith (opaque)
ROCK = st.SRC            # magnetized outcrop (viridis top, as in Ch. 3 figures)
ROOT = "#9FB8AE"         # buried extent, drawn lighter (pre-blended, no alpha)
PATH = st.TRAVERSE
FS = st.BASE_PT - 1


def arrow(ax, p0, p1, color=INK, lw=0.8, style="-|>", ms=7, ls="-", **kw):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0, **kw))


def side_view(ax):
    ax.set_xlim(0, 10.7); ax.set_ylim(-1.5, 3.35); ax.set_aspect("equal"); ax.axis("off")
    # ground and regolith
    ax.add_patch(Rectangle((0, -1.5), 10.6, 1.5, fc=REG, ec="none", zorder=0))
    ax.plot([0, 10.6], [0, 0], color=INK, lw=0.8, zorder=2)
    ax.text(0.15, -1.32, "regolith (thickness $h$)", fontsize=FS, color=INK, va="bottom")
    # break marks for the ~1 km gap
    for x in (3.35, 3.55):
        ax.plot([x - 0.1, x + 0.1], [-0.25, 0.25], color=INK, lw=0.8, zorder=3)
    ax.annotate("", xy=(2.3, 2.95), xytext=(7.4, 2.95),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.7, shrinkA=0, shrinkB=0))
    ax.text(4.85, 3.02, r"$\lesssim 1$ km (lander to outcrop)", fontsize=FS, color=MUTED, ha="center", va="bottom")

    # lander
    ax.add_patch(Polygon([[0.55, 0.0], [0.8, 0.55], [1.9, 0.55], [2.15, 0.0]], closed=False, fill=False,
                         ec=INK, lw=0.9, zorder=3))
    ax.add_patch(Rectangle((0.75, 0.55), 1.2, 0.75, fc="#E3E3E3", ec=INK, lw=0.9, zorder=3))
    ax.plot([1.85, 1.85], [1.3, 2.2], color=INK, lw=1.0, zorder=3)
    ax.add_patch(Rectangle((1.77, 2.2), 0.16, 0.16, fc=st.INTENSITY, ec=INK, lw=0.6, zorder=4))
    ax.text(2.02, 2.28, "reference\nfluxgate", fontsize=FS, va="center", ha="left")
    ax.plot([1.0, 1.25], [1.3, 1.65], color=INK, lw=0.8)
    ax.add_patch(Polygon([[0.95, 1.6], [1.4, 1.78], [1.32, 1.9]], fc="white", ec=INK, lw=0.6, zorder=4))
    arrow(ax, (1.15, 1.85), (0.75, 2.45), color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.text(0.75, 2.5, "to Earth", fontsize=FS, color=MUTED, va="bottom", ha="center")
    ax.text(1.35, -0.12, "lander", fontsize=FS, ha="center", va="top")

    # outcrop with buried extent
    ox0, ox1 = 8.0, 10.2
    ax.add_patch(Rectangle((ox0, 0.0), ox1 - ox0, 1.0, fc=ROCK, ec=INK, lw=0.8, zorder=3))
    ax.add_patch(Rectangle((ox0, -0.7), ox1 - ox0, 0.7, fc=ROOT, ec=INK, lw=0.7, ls=(0, (3, 2)), zorder=2))
    ax.annotate("", xy=(10.38, 0.0), xytext=(10.38, -0.7),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.6, shrinkA=0, shrinkB=0))
    ax.text(10.44, -0.35, "$b$", fontsize=FS, va="center")
    ax.text((ox0 + ox1) / 2, 1.08, "outcrop", fontsize=FS, color=INK, ha="center", va="bottom", zorder=5)
    for xa in (8.5, 9.1, 9.7):
        arrow(ax, (xa, 0.85), (xa, 0.18), color=INK, lw=0.8, ms=6, zorder=5)
    ax.text((ox0 + ox1) / 2, -0.38, "buried\nextent", fontsize=FS, color=INK, ha="center", va="center", zorder=5)

    # rover
    rx = 5.6
    ax.add_patch(Rectangle((rx, 0.22), 1.25, 0.42, fc="#E3E3E3", ec=INK, lw=0.9, zorder=3))
    for wx in (rx + 0.2, rx + 0.62, rx + 1.05):
        ax.add_patch(Circle((wx, 0.15), 0.15, fc="white", ec=INK, lw=0.8, zorder=4))
    ax.plot([rx + 0.3, rx + 0.3], [0.64, 1.25], color=INK, lw=0.9)          # camera mast
    ax.add_patch(Rectangle((rx + 0.16, 1.25), 0.3, 0.16, fc="white", ec=INK, lw=0.7, zorder=4))
    ax.text(rx - 0.05, 1.5, "stereo cameras\nand/or LiDAR", fontsize=FS, ha="right", va="bottom")
    ax.plot([rx + 1.25, rx + 2.0], [0.55, 0.5], color=INK, lw=1.0)          # boom toward outcrop
    ax.add_patch(Rectangle((rx + 1.95, 0.42), 0.16, 0.16, fc=st.DIRECTION, ec=INK, lw=0.6, zorder=4))
    ax.text(rx + 2.05, 1.65, "boom-mounted\nthree-axis fluxgate\n(sensor 0.5 m up)", fontsize=FS, ha="center", va="bottom")
    arrow(ax, (rx + 2.05, 1.62), (rx + 2.03, 0.62), color=MUTED, lw=0.6, ms=5)
    ax.plot([rx + 0.9, rx + 1.3, rx + 1.55], [0.64, 0.95, 0.05], color=INK, lw=0.8)  # arm
    ax.add_patch(Rectangle((rx + 1.5, 0.0), 0.1, 0.07, fc=INK, ec=INK, lw=0.5, zorder=4))
    ax.text(rx + 0.6, -0.12, "rover", fontsize=FS, ha="center", va="top")
    ax.text(rx + 1.2, -0.62, "susceptibility probe,\noptional GPR", fontsize=FS, ha="center", va="top")
    ax.annotate("", xy=(rx + 1.55, -0.02), xytext=(rx + 1.4, -0.58),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.6, mutation_scale=5, shrinkA=0, shrinkB=0))


def plan_view(ax):
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((-2, -2), 4, 4, fc=ROCK, ec=INK, lw=0.8, zorder=2))
    for s, lab, ls in ((3.0, "loop 1: 1 m standoff, 24 m", "-"), (5.0, "loop 2: 3 m standoff, 40 m", (0, (5, 2.5)))):
        xs = [-s, s, s, -s, -s]; ys = [-s, -s, s, s, -s]
        ax.plot(xs, ys, color=PATH, lw=1.2, ls=ls, zorder=3)
        # paired arrows: both directions
        arrow(ax, (-0.6, -s), (0.6, -s), color=PATH, lw=1.0, ms=7, zorder=4)
        arrow(ax, (0.6, s), (-0.6, s), color=PATH, lw=1.0, ms=7, zorder=4)
    ax.text(0, -3.3, "loop 1 (1 m)", fontsize=FS, ha="center", va="top", color=INK)
    ax.text(0, -5.3, "loop 2 (3 m)", fontsize=FS, ha="center", va="top", color=INK)
    ax.text(0, 0, "outcrop", fontsize=FS - 1, ha="center", va="center", color=INK, zorder=5)
    # baseline traverse leaving the site, and a calibration-turn site
    ax.plot([5.0, 15.0], [-5.0, -5.0], color=st.INTENSITY, lw=1.2, ls=(0, (1, 1.5)), zorder=3)
    ax.text(10.0, -5.35, "10 m baseline traverse", fontsize=FS, ha="center", va="top", color=INK)
    ax.add_patch(Circle((12.0, 2.0), 1.0, fill=False, ec=MUTED, lw=0.8, ls=(0, (2, 1.5))))
    arrow(ax, (12.0 + np.cos(0.3), 2.0 + np.sin(0.3)), (12.0 + np.cos(0.6), 2.0 + np.sin(0.6)), color=MUTED, ms=7)
    ax.text(12.0, 3.25, "calibration\nturns", fontsize=FS, ha="center", va="bottom", color=INK)
    ax.text(-5.0, 5.4, "each loop driven both ways", fontsize=FS, ha="left", va="bottom", color=INK)
    ax.set_xlim(-6.2, 15.5); ax.set_ylim(-7.0, 7.3)
    ax.set_xticks([-5, 0, 5, 10, 15]); ax.set_yticks([-5, 0, 5])
    ax.set_xlabel("$x$ (m)"); ax.set_ylabel("$y$ (m)")


def main():
    with st.style():
        fig = st.figure("full", 2.15)
        ax = fig.add_axes([0.0, 0.0, 0.6, 1.0])
        bx = fig.add_axes([0.675, 0.19, 0.315, 0.76])
        side_view(ax); plan_view(bx)
        st.panel_label(ax, "A", x=0.005, y=0.995); st.panel_label(bx, "B", x=-0.12, y=1.02)
        finish(fig, "ch6_architecture")


if __name__ == "__main__":
    main()

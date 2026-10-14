"""Scenario-geometry figures of thesis Chapter 3 (Figs. 3.1-3.3), drawn from the scene definitions in the code.

Same visual language as the companion paper's Figures 1-3, redrawn at the thesis text width: plan views on top,
true-scale sections sharing their horizontal axis, a viridis magnetization scale (zero = ground, top = M_ref),
x / o / arrows for magnetization into / out of / in the page, rover paths in orange, start and end as a black dot.
All three share one canvas width, panel width and legend band, so they print at a consistent size.

``data_*`` returns the geometry that is drawn (bodies, paths, magnetization signs), taken from the same constants
as the simulations (``engine/skatepark_1cm_dr_sweep.py`` for the skate park, ``inversion_helpers`` for the wall).
"""
from __future__ import annotations

import os
import sys
from typing import Any

import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.transforms import Bbox

from .. import plotting as st
from ..paths import ENGINE

FigData = dict[str, Any]

# ---------------------------------------------------------------------------------------------- shared layout
W = st.WIDTH["full"]
COL_W, COL_GAP, SHARE_GAP = 2.7, 0.5, 0.09         # panel width, gap between columns, gap plan -> section (in)
COL_X = [0.52, 0.52 + COL_W + COL_GAP]             # left edges of the two columns (in)
COL_W_SINGLE = 3.4                                 # Fig. 3.1 has one column, drawn a little larger
GRID_TOP = 0.62                                    # band for the colorbar above the panels (in)
LEG_DROP, LEG_H = 0.42, 0.30                       # legend band under the panels (in)
TRAV_LW = 1.6
LS_LOOP, LS_BY, LS_ELEV = "-", (0, (6, 2, 1.5, 2)), (0, (4, 2.2))
MARK_LW = 1.0


class GeoGrid:
    """A figure on which axes are placed in inches measured from the top-left corner."""

    def __init__(self, height_in: float):
        self.H = height_in
        self.fig = st.figure("full", height_in)

    def axes(self, x, top, w, h, **kw):
        return self.fig.add_axes([x / W, 1 - (top + h) / self.H, w / W, h / self.H], **kw)


def h_true(xlim, ylim, ve=1.0, width=COL_W):
    """Panel height (in) that draws a ``width``-wide panel at vertical exaggeration ``ve``."""
    return width * (ylim[1] - ylim[0]) * ve / (xlim[1] - xlim[0])


def fig_height(block_bottom):
    return block_bottom + LEG_DROP + LEG_H


def colorbar(g: GeoGrid):
    cw, ch = 0.36 * W, 0.11
    cax = g.fig.add_axes([(W - cw) / 2 / W - 0.06, 1 - (0.20 + ch) / g.H, cw / W, ch / g.H])
    cb = g.fig.colorbar(cm.ScalarMappable(norm=st.NORM_M, cmap=st.VIRIDIS), cax=cax, orientation="horizontal",
                        ticks=[0, 0.6, 1.2, 1.8, 2.4])
    cb.ax.set_xticklabels(["0", "0.6", "1.2", "1.8", "2.4"])
    cb.ax.tick_params(length=2, pad=1.5, labelsize=st.BASE_PT - 1)
    cb.outline.set_linewidth(0.6)
    cax.text(1.03, 0.5, r"$|\mathbf{M}_{\mathrm{NRM}}|$ ($\times10^{-3}$ A m$^{-1}$)", transform=cax.transAxes,
             ha="left", va="center")
    return cb


def bottom_legend(g: GeoGrid, handles, labels, block_bottom):
    return g.fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1 - (block_bottom + LEG_DROP) / g.H),
                        ncol=len(handles), frameon=False, handlelength=2.2, columnspacing=1.2, handletextpad=0.5,
                        fontsize=st.BASE_PT - 1)


def mark_into(ax, x, y, s=26, lw=MARK_LW, color="black"):
    ax.scatter(x, y, marker="x", s=s, c=color, linewidths=lw, zorder=6)


def mark_out(ax, x, y, s=26, lw=MARK_LW, color="black"):
    ax.scatter(x, y, marker="o", s=s, facecolors="none", edgecolors=color, linewidths=lw, zorder=6)


def arrow(ax, x, y, dx, dy, lw=1.0, ms=7):
    ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y), zorder=7,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=lw, mutation_scale=ms, shrinkA=0, shrinkB=0))


def start_dot(ax, x, y):
    ax.scatter([x], [y], s=22, c="black", edgecolors="white", linewidths=0.9, zorder=9)


def start_handle(ax):
    return ax.scatter([], [], s=22, c="black", edgecolors="white", linewidths=0.9, label="Start/end")


def trav(ax, xs, ys, ls="-", label=None, zorder=6):
    ax.plot(xs, ys, color=st.TRAVERSE, lw=TRAV_LW, ls=ls, zorder=zorder, label=label, solid_capstyle="butt",
            dash_capstyle="butt")


def ground(ax, x0, x1, depth):
    ax.set_facecolor("white")
    ax.add_patch(Rectangle((x0, -depth), x1 - x0, depth, fc=st.BG, ec="none", zorder=1))


def guide(ax, xs, z):
    """Thin grey dotted line at the sensor height in a vertical section (not a survey path)."""
    ax.plot(xs, [z, z], ls=(0, (1, 1.6)), color="#8C8C8C", lw=0.8, zorder=4)


def triad(ax, x0, y0, L, labels=("x", "y"), out_of_page=None, Lz=None, color="white"):
    Lz = L if Lz is None else Lz
    for (dx, dy), lab in (((L, 0), labels[0]), ((0, Lz), labels[1])):
        ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0), zorder=12,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, mutation_scale=7, shrinkA=0, shrinkB=0))
        ax.text(x0 + dx * 1.3, y0 + dy * 1.3, f"${lab}$", color=color, ha="center", va="center", zorder=12,
                fontsize=st.BASE_PT - 1)
    if out_of_page:
        ax.scatter([x0], [y0], s=26, facecolors="none", edgecolors=color, lw=0.8, zorder=12)
        ax.scatter([x0], [y0], s=2.5, c=color, zorder=12)
        ax.text(x0 - 0.22 * L, y0 - 0.25 * L, f"${out_of_page}$", color=color, ha="right", va="top", zorder=12,
                fontsize=st.BASE_PT - 1)


def section_axes_style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)


def plan_axes_style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)


def save_kwargs(fig):
    w, h = fig.get_size_inches()
    return dict(bbox_inches=Bbox.from_bounds(0, 0, w, h))


# ============================================================================================== Fig. 3.1
HALF, HEIGHT, Z_S, XBY = 2.0, 1.0, 0.5, 8.0
L1 = HALF + 1.0


def data_outcrop_geometry() -> FigData:
    """Prism 4 x 4 x 1 m on the ground, M = 2.4e-3 A/m along -z, sensor 0.5 m up; drive-around loop 1 m outside the
    faces from (-3, -3); drive-by along y = -3 m from x = -8 to 8 m (as analysis/a02 and a09)."""
    return {"prism_xyz_m": [[-HALF, HALF], [-HALF, HALF], [0.0, HEIGHT]], "M_Am": [0.0, 0.0, -2.4e-3],
            "sensor_z_m": Z_S,
            "drive_around_xy_m": [[-L1, -L1], [L1, -L1], [L1, L1], [-L1, L1], [-L1, -L1]],
            "drive_by_xy_m": [[-XBY, -L1], [XBY, -L1]]}


def draw_outcrop_geometry(d: FigData) -> plt.Figure:
    XL, YA, ZB = (-8.8, 8.8), (-7.0, 7.0), (-1.1, 2.1)
    cw = COL_W_SINGLE
    hA, hB = h_true(XL, YA, width=cw), h_true(XL, ZB, width=cw)
    bottom = GRID_TOP + hA + SHARE_GAP + hB
    g = GeoGrid(fig_height(bottom)); fig = g.fig
    x0 = (W - cw) / 2
    axA = g.axes(x0, GRID_TOP, cw, hA)
    axB = g.axes(x0, GRID_TOP + hA + SHARE_GAP, cw, hB, sharex=axA)
    (px0, px1), (py0, py1), (pz0, pz1) = d["prism_xyz_m"]
    axA.set_facecolor(st.BG)
    axA.add_patch(Rectangle((px0, py0), px1 - px0, py1 - py0, fc=st.SRC, ec="black", lw=1.0, zorder=3))
    for xm in (-1, 1):
        for ym in (-1, 1):
            mark_into(axA, xm, ym)
    s1 = np.array(d["drive_around_xy_m"])
    trav(axA, s1[:, 0], s1[:, 1], LS_LOOP, "Drive-around")
    by = np.array(d["drive_by_xy_m"])
    trav(axA, by[:, 0], by[:, 1], LS_BY, "Drive-by", zorder=5)
    start_dot(axA, s1[0, 0], s1[0, 1])
    sh = start_handle(axA)
    triad(axA, -7.4, -5.5, 1.5, ("x", "y"), out_of_page="z")
    axA.set_xlim(*XL); axA.set_ylim(*YA); axA.set_aspect("equal")
    axA.set_yticks([-5, 0, 5]); axA.set_ylabel(r"$y$ (m)")
    plt.setp(axA.get_xticklabels(), visible=False)
    st.panel_label(axA, "A", x=0.03, y=0.96, dark=True)
    plan_axes_style(axA)

    ground(axB, -12, 12, 5)
    axB.add_patch(Rectangle((px0, pz0), px1 - px0, pz1 - pz0, fc=st.SRC, ec="black", lw=1.0, zorder=3))
    for xa in (-1.0, 1.0):
        arrow(axB, xa, 0.88, 0, -0.76)
    guide(axB, [-12, 12], d["sensor_z_m"])
    axB.set_ylim(*ZB); axB.set_aspect("equal")
    axB.set_xticks([-5, 0, 5]); axB.set_yticks([0, 1, 2])
    axB.set_xlabel(r"$x$ (m)"); axB.set_ylabel(r"$z$ (m)")
    st.panel_label(axB, "B", x=0.03, y=0.94)
    section_axes_style(axB)

    h, l = axA.get_legend_handles_labels()
    mx = axB.scatter([], [], marker="x", s=26, c="black", linewidths=MARK_LW)
    bottom_legend(g, h + [mx], l + [r"$\mathbf{M}$ into page"], bottom)
    colorbar(g)
    return fig


# ============================================================================================== Fig. 3.2
def _skatepark():
    if str(ENGINE) not in sys.path:
        sys.path.insert(0, str(ENGINE))
    import skatepark_1cm_dr_sweep as SP          # scene definition only; nothing is run
    return SP


SK_DR, SK_LOOP = 1.0, 3.0


def data_skatepark_geometry() -> FigData:
    """The five bodies of the skate park at Delta_r = 1 m for cases A and B, from engine/skatepark_1cm_dr_sweep.py."""
    SP = _skatepark()
    out: FigData = {"delta_r_m": SK_DR, "loop_half_m": SK_LOOP}
    for case in ("A", "B"):
        cfgs, _, _ = SP.outcrop_configs_for(SK_DR, case)
        out[case] = [{"shape": c["shape"], "params": {k: float(v) for k, v in c["params"].items()},
                      "center": [float(v) for v in c["center"]],
                      "magnetization": [float(v) for v in c["magnetization"]]} for c in cfgs]
    return out


def _sk_body_plan(ax, cfg, sym):
    stp, p, (cx, cy, cz) = cfg["shape"], cfg["params"], cfg["center"]
    kw = dict(facecolor=st.SRC, edgecolor="black", lw=1.0, zorder=3)
    if stp == "rect":
        ax.add_patch(Rectangle((cx - p["sx"] / 2, cy - p["sy"] / 2), p["sx"], p["sy"], **kw)); dd = 1.0
    elif stp == "pyramid_z+":
        a = p["a"]; ax.add_patch(Rectangle((cx - a / 2, cy - a / 2), a, a, **kw))
        for sx_, sy_ in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            ax.plot([cx + sx_ * a / 2, cx], [cy + sy_ * a / 2, cy], color="black", lw=0.6, zorder=4)
        dd = a / 4
    else:
        r = p["r"]; ax.add_patch(Circle((cx, cy), r, **kw)); dd = 0.42 * r
    for sx in (-dd, dd):
        for sy in (-dd, dd):
            s = 18 if (stp in ("rect", "pyramid_z+") or p.get("r", 9) > 1.3) else 10
            (mark_into if sym == "x" else mark_out)(ax, cx + sx, cy + sy, s=s, lw=0.8)


def _sk_nearest(cfg):
    return cfg["params"]["a"] / 2 if cfg["shape"] == "pyramid_z+" else cfg["params"]["r"]


def _sk_plan(ax, cfgs, letter, triad_on):
    L = SK_LOOP
    ax.set_facecolor(st.BG)
    for c in cfgs:
        _sk_body_plan(ax, c, "x" if c["magnetization"][2] < 0 else "o")
    sq = np.array([[-L, -L], [L, -L], [L, L], [-L, L], [-L, -L]])
    trav(ax, sq[:, 0], sq[:, 1], LS_LOOP, "Drive-around", zorder=5)
    start_dot(ax, -L, -L)
    if triad_on:
        triad(ax, -8.7, -7.2, 1.4, ("x", "y"), out_of_page="z")
    for c in cfgs[1:]:
        cx, cy, _ = c["center"]; e = _sk_nearest(c)
        if abs(cx) > abs(cy):
            s = np.sign(cx); ax.plot([s * L, cx - s * e], [0, 0], color="white", lw=1.6, zorder=2)
        else:
            s = np.sign(cy); ax.plot([0, 0], [s * L, cy - s * e], color="white", lw=1.6, zorder=2)
    ax.annotate(r"$\Delta r$", xy=(L + 0.5 * SK_DR, 0.15), xytext=(L + 1.3, 2.3), color="white", ha="center",
                zorder=12, arrowprops=dict(arrowstyle="-|>", color="white", lw=0.9, mutation_scale=7,
                                           connectionstyle="arc3,rad=0.3"))
    ax.set_xlim(-10.4, 10.4); ax.set_ylim(-9.1, 7.6); ax.set_aspect("equal")
    ax.set_xticks([-5, 0, 5]); ax.set_yticks([-5, 0, 5])
    ax.set_xlabel(r"$x$ (m)")
    st.panel_label(ax, letter, x=0.03, y=0.96, dark=True)
    plan_axes_style(ax)


def _sk_section(ax, cfgs):
    ax.set_facecolor("white")
    ax.add_patch(Rectangle((-13, -3), 26, 3, fc=st.BG, ec="none", zorder=1))
    for c in cfgs:
        stp, p, (cx, cy, cz) = c["shape"], c["params"], c["center"]
        if abs(cy) > 1e-9:
            continue
        up = c["magnetization"][2] > 0
        kw = dict(facecolor=st.SRC, edgecolor="black", lw=1.0, zorder=3)
        if stp == "rect":
            ax.add_patch(Rectangle((cx - p["sx"] / 2, 0), p["sx"], p["sz"], **kw)); xs, zc, L = (cx - 1, cx + 1), 0.5, 0.35
        elif stp == "sphere":
            ax.add_patch(Circle((cx, cz), p["r"], **kw)); xs, zc, L = (cx,), cz, 0.45
        elif stp == "hemisphere_z+":
            t = np.linspace(0, np.pi, 100)
            ax.add_patch(Polygon(np.c_[cx + p["r"] * np.cos(t), p["r"] * np.sin(t)], closed=True, **kw))
            xs, zc, L = (cx,), 0.45 * p["r"], 0.45
        else:
            continue
        for x in xs:
            arrow(ax, x, zc - (L if up else -L), 0, (2 * L if up else -2 * L))
    guide(ax, [-13, 13], 0.5)
    ax.set_ylim(-1.2, 2.6); ax.set_aspect("equal")
    ax.set_xticks([-5, 0, 5]); ax.set_yticks([0, 2])
    ax.set_xlabel(r"$x$ (m)"); ax.set_ylabel(r"$z$ (m)")
    st.panel_label(ax, "C", x=0.03, y=0.94)
    section_axes_style(ax)


def draw_skatepark_geometry(d: FigData) -> plt.Figure:
    XL, YP, ZC = (-10.4, 10.4), (-9.1, 7.6), (-1.2, 2.6)
    hP, hC = h_true(XL, YP), h_true(XL, ZC)
    bottom = GRID_TOP + hP + SHARE_GAP + hC
    g = GeoGrid(fig_height(bottom)); fig = g.fig
    x0, x1 = COL_X
    axA = g.axes(x0, GRID_TOP, COL_W, hP)
    axB = g.axes(x1, GRID_TOP, COL_W, hP, sharey=axA)
    axC = g.axes(x1, GRID_TOP + hP + SHARE_GAP, COL_W, hC, sharex=axB)
    _sk_plan(axA, d["A"], "A", triad_on=True)
    _sk_plan(axB, d["B"], "B", triad_on=False)
    axA.set_ylabel(r"$y$ (m)")
    plt.setp(axB.get_yticklabels(), visible=False)
    plt.setp(axB.get_xticklabels(), visible=False); axB.set_xlabel("")
    _sk_section(axC, d["B"])
    loop = mlines.Line2D([], [], color=st.TRAVERSE, lw=TRAV_LW)
    sh = mlines.Line2D([], [], marker="o", ls="none", ms=4.5, mfc="black", mec="white", mew=0.9)
    gap = mlines.Line2D([], [], color="white", lw=1.8, path_effects=[pe.Stroke(linewidth=3.6, foreground=st.BG), pe.Normal()])
    mx = mlines.Line2D([], [], marker="x", ls="none", ms=5, mec="black", mew=MARK_LW)
    mo = mlines.Line2D([], [], marker="o", ls="none", ms=5, mfc="none", mec="black", mew=MARK_LW)
    bottom_legend(g, [loop, sh, gap, mx, mo], ["Drive-around", "Start/end", r"Gap $\Delta r$", r"$\mathbf{M}$ into page",
                                               r"$\mathbf{M}$ out of page"], bottom)
    colorbar(g)
    return fig


# ============================================================================================== Fig. 3.3
TL_LAYERS = [(0, 5, "Flow 1", (0, 0, -1)), (5, 10, "Flow 2", (0, 1, 0)), (10, 15, "Flow 3", (0, 0, 1))]
TL_XW, TL_YW, TL_Y_ROV, TL_Z_LO, TL_Z_HI, TL_X_TR = 50.0, 50.0, -1.0, 0.5, 12.5, 120.0


def data_threelayer_geometry() -> FigData:
    """Three 5 m flows, x in [-50, 50] m, y in [0, 50] m, z in [0, 15] m (inversion_helpers.three_layer_mesh);
    M per flow -z, +y, +z; ground traverse y = -1 m, z = 0.5 m, elevated traverse z = 12.5 m, x = -120..120 m."""
    from ..inversion_helpers import three_layer_mesh
    mesh = three_layer_mesh()
    return {"mesh_origin_m": [float(v) for v in mesh.origin], "mesh_h_m": [[float(v) for v in h] for h in mesh.h],
            "flows": [{"z_m": [z0, z1], "name": n, "M_dir": list(m)} for z0, z1, n, m in TL_LAYERS],
            "traverse_y_m": TL_Y_ROV, "traverse_z_m": [TL_Z_LO, TL_Z_HI], "traverse_x_m": [-TL_X_TR, TL_X_TR]}


def draw_threelayer_geometry(d: FigData) -> plt.Figure:
    VE = 3.0
    XA = 75.0
    YA, ZL, YC = (-10.0, 60.0), (-2.5, 17.5), (-8.0, 62.0)
    XLIM = (-XA, XA)
    hA, hB, hC = h_true(XLIM, YA), h_true(XLIM, ZL, VE), h_true(YC, ZL)
    bottom = GRID_TOP + hA + SHARE_GAP + hB
    g = GeoGrid(fig_height(bottom)); fig = g.fig
    x0, x1 = COL_X
    axA = g.axes(x0, GRID_TOP, COL_W, hA)
    axB = g.axes(x0, GRID_TOP + hA + SHARE_GAP, COL_W, hB, sharex=axA)
    axC = g.axes(x1, GRID_TOP + (bottom - GRID_TOP - hC) / 2 - 0.1, COL_W, hC)
    yr, (zlo, zhi), (xt0, xt1) = d["traverse_y_m"], d["traverse_z_m"], d["traverse_x_m"]

    axA.set_facecolor(st.BG)
    axA.add_patch(Rectangle((-TL_XW, 0), 2 * TL_XW, TL_YW, fc=st.SRC, ec="black", lw=1.0, zorder=3))
    for xa in (-35, -12, 12, 35):
        for ya in (15, 35):
            mark_out(axA, xa, ya, s=18, lw=0.9)
    trav(axA, [xt0, xt1], [yr, yr], LS_LOOP, zorder=6)
    triad(axA, -68, 16, 11, ("x", "y"), out_of_page="z")
    axA.set_xlim(*XLIM); axA.set_ylim(*YA); axA.set_aspect("equal")
    axA.set_xticks([-50, 0, 50]); axA.set_yticks([0, 25, 50])
    axA.set_ylabel(r"$y$ (m)"); plt.setp(axA.get_xticklabels(), visible=False)
    st.panel_label(axA, "A", x=0.03, y=0.95, dark=True)
    plan_axes_style(axA)

    ground(axB, -140, 140, 12)
    for fl in d["flows"]:
        z0, z1 = fl["z_m"]; m = fl["M_dir"]
        axB.add_patch(Rectangle((-TL_XW, z0), 2 * TL_XW, z1 - z0, fc=st.SRC, ec="black", lw=1.0, zorder=3))
        zc = 0.5 * (z0 + z1)
        for xa in (-35, -12, 12, 35):
            if m[1] == 0:
                arrow(axB, xa, zc - m[2] * 1.6, 0, m[2] * 3.2)
            else:
                mark_into(axB, xa, zc, s=14, lw=0.9)
    trav(axB, [xt0, xt1], [zlo, zlo], LS_LOOP, "Ground traverse", zorder=6)
    trav(axB, [xt0, xt1], [zhi, zhi], LS_ELEV, "Elevated traverse", zorder=6)
    axB.set_ylim(*ZL); axB.set_aspect(VE)
    axB.set_xticks([-50, 0, 50]); axB.set_yticks([0, 5, 10, 15])
    axB.set_xlabel(r"$x$ (m, along strike)"); axB.set_ylabel(r"$z$ (m)")
    st.panel_label(axB, "B", x=0.03, y=0.94)
    section_axes_style(axB)

    ground(axC, -14, 92, 18)
    for fl in d["flows"]:
        z0, z1 = fl["z_m"]; m = fl["M_dir"]
        axC.add_patch(Rectangle((0, z0), TL_YW, z1 - z0, fc=st.SRC, ec="black", lw=1.0, zorder=3))
        zc = 0.5 * (z0 + z1)
        for ya in (12, 25, 38):
            if m[1] == 0:
                arrow(axC, ya, zc - m[2] * 1.7, 0, m[2] * 3.4)
            else:
                arrow(axC, ya - 4, zc, 8, 0)
        axC.text(TL_YW + 1.5, zc, fl["name"], ha="left", va="center", fontsize=st.BASE_PT - 1, zorder=8)
    axC.set_xlim(*YC); axC.set_ylim(*ZL); axC.set_aspect("equal")
    axC.set_xticks([0, 25, 50]); axC.set_yticks([0, 5, 10, 15])
    axC.set_xlabel(r"$y$ (m, into the wall)"); axC.set_ylabel(r"$z$ (m)")
    st.panel_label(axC, "C", x=0.03, y=0.95)
    section_axes_style(axC)

    hL, lL = axB.get_legend_handles_labels()
    mx = mlines.Line2D([], [], marker="x", ls="none", ms=5, mec="black", mew=MARK_LW)
    mo = mlines.Line2D([], [], marker="o", ls="none", ms=5, mfc="none", mec="black", mew=MARK_LW)
    bottom_legend(g, hL + [mx, mo], lL + [r"$\mathbf{M}$ into page", r"$\mathbf{M}$ out of page"], bottom)
    colorbar(g)
    return fig

"""Figure 2 - Multi-body ("skate park") scene, drawn from the scene definition used for every
Section 4.3 number (engine/skatepark_1cm_dr_sweep.py: secondary_objects_for, outcrop_configs_for).

(A) Case A, plan view: all five bodies magnetized along -z (x = into the page).
(B) Case B, plan view: east (hemisphere) and west (sphere) neighbours reversed to +z (o = out of page).
(C) Case B, xz section at y = 0 through the west sphere, central prism and east hemisphere, showing the
    sensor height (z = 0.5 m) where the traverse crosses the section.
Scene shown at Delta_r = 1 m (the closest spacing in the sweep); Delta_r is the gap between the
traverse and the nearest point of each neighbour, swept 1-10 m in Figure 6.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
import paperstyle as ps

sys.path.insert(0, os.path.join(ps.ROOT, "engine"))
import skatepark_1cm_dr_sweep as SP  # scene definition only; nothing is run

DR = 1.0
XL, YP, ZC = (-10.4, 10.4), (-9.1, 7.6), (-1.2, 2.6)   # plan x/y and section z limits (m), all to scale
LOOP = 3.0


def body_plan(ax, cfg, sym):
    st, p, (cx, cy, cz) = cfg["shape"], cfg["params"], cfg["center"]
    kw = dict(facecolor=ps.SRC, edgecolor="black", lw=2, zorder=3)
    if st == "rect":
        ax.add_patch(Rectangle((cx - p["sx"] / 2, cy - p["sy"] / 2), p["sx"], p["sy"], **kw)); d = 1.0
    elif st == "pyramid_z+":
        a = p["a"]; ax.add_patch(Rectangle((cx - a / 2, cy - a / 2), a, a, **kw))
        for sx_, sy_ in ((-1, -1), (1, -1), (1, 1), (-1, 1)):   # edges from the base corners to the apex
            ax.plot([cx + sx_ * a / 2, cx], [cy + sy_ * a / 2, cy], color="black", lw=1.0, zorder=4)
        d = a / 4
    else:
        r = p["r"]; ax.add_patch(Circle((cx, cy), r, **kw))
        d = 0.42 * r
    for sx in (-d, d):
        for sy in (-d, d):
            sz = 85 if (st in ("rect", "pyramid_z+") or p.get("r", 9) > 1.3) else 45
            (ps.mark_into if sym == "x" else ps.mark_out)(ax, cx + sx, cy + sy, s=sz, lw=1.6)


def nearest_edge(cfg):
    st, p, (cx, cy, _) = cfg["shape"], cfg["params"], cfg["center"]
    if st == "pyramid_z+":
        return p["a"] / 2
    return p["r"]


def plan_panel(ax, case, letter, legend):
    cfgs, _, _ = SP.outcrop_configs_for(DR, case)
    ax.set_facecolor(ps.BG)
    for c in cfgs:
        body_plan(ax, c, "x" if c["magnetization"][2] < 0 else "o")
    sq = np.array([[-LOOP, -LOOP], [LOOP, -LOOP], [LOOP, LOOP], [-LOOP, LOOP], [-LOOP, -LOOP]])
    ps.trav(ax, sq[:, 0], sq[:, 1], ps.LS_LOOP, "Drive-around", zorder=5)
    ps.start_dot(ax, -LOOP, -LOOP)
    ps.start_legend_handle(ax)
    if legend:
        ps.triad(ax, -8.9, -7.3, 1.4, ("x", "y"), color="white", out_of_page="z")
    ax.plot([], [], color="white", lw=3, label=r"Gap $\Delta r$ to each neighbor")
    # Delta_r gaps (white) from the traverse to each neighbour's nearest point
    for c in cfgs[1:]:
        cx, cy, _ = c["center"]; e = nearest_edge(c)
        if abs(cx) > abs(cy):
            s = np.sign(cx); ax.plot([s * LOOP, cx - s * e], [0, 0], color="white", lw=3, zorder=2)
        else:
            s = np.sign(cy); ax.plot([0, 0], [s * LOOP, cy - s * e], color="white", lw=3, zorder=2)
    ax.annotate(r"$\Delta r$", xy=(LOOP + 0.5 * DR, 0.15), xytext=(LOOP + 1.2, 2.2), color="white",
                fontsize=16, ha="center", zorder=12,
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.8, connectionstyle="arc3,rad=0.3"))
    ax.set_xlim(*XL); ax.set_ylim(*YP); ax.set_aspect("equal")
    ax.set_xticks([-5, 0, 5]); ax.set_yticks([-5, 0, 5])
    ax.set_xlabel(r"$x$ (m)")
    ps.panel_letter(ax, letter, dark=True, size=24)


def section_panel(ax):
    cfgs, _, _ = SP.outcrop_configs_for(DR, "B")
    ax.set_facecolor("white")
    ax.add_patch(Rectangle((-13, -3), 26, 3, color=ps.BG, zorder=1))
    for c in cfgs:
        st, p, (cx, cy, cz) = c["shape"], c["params"], c["center"]
        if abs(cy) > 1e-9:
            continue
        up = c["magnetization"][2] > 0
        kw = dict(facecolor=ps.SRC, edgecolor="black", lw=2, zorder=3)
        if st == "rect":
            ax.add_patch(Rectangle((cx - p["sx"] / 2, 0), p["sx"], p["sz"], **kw)); xs, zc, L = (cx - 1, cx + 1), 0.5, 0.35
        elif st == "sphere":
            ax.add_patch(Circle((cx, cz), p["r"], **kw)); xs, zc, L = (cx,), cz, 0.45
        elif st == "hemisphere_z+":
            t = np.linspace(0, np.pi, 100)
            ax.add_patch(Polygon(np.c_[cx + p["r"] * np.cos(t), p["r"] * np.sin(t)], closed=True, **kw)); xs, zc, L = (cx,), 0.45 * p["r"], 0.45
        for x in xs:
            ps.arrow(ax, x, zc - (L if up else -L), 0, (2 * L if up else -2 * L), lw=1.5, ms=10)
    ps.guide(ax, [-13, 13], 0.5)
    ax.set_ylim(*ZC); ax.set_aspect("equal")
    ax.set_xticks([-5, 0, 5]); ax.set_yticks([0, 2])
    ax.set_xlabel(r"$x$ (m)"); ax.set_ylabel(r"$z$ (m)")
    ps.panel_letter(ax, "C", dark=False, size=24, y=0.97)


def main():
    T = ps.GRID_TOP; hP, hC = ps.h_true(XL, YP), ps.h_true(XL, ZC)
    bottom = T + hP + ps.SHARE_GAP + hC
    g = ps.GeoGrid(ps.fig_height(bottom)); fig = g.fig
    x0, x1 = ps.COL_X
    axA = g.axes(x0, T, ps.COL_W, hP)
    axB = g.axes(x1, T, ps.COL_W, hP, sharey=axA)
    axC = g.axes(x1, T + hP + ps.SHARE_GAP, ps.COL_W, hC, sharex=axB)
    plan_panel(axA, "A", "A", legend=True)
    plan_panel(axB, "B", "B", legend=False)
    axA.set_ylabel(r"$y$ (m)")
    plt.setp(axB.get_yticklabels(), visible=False)
    plt.setp(axB.get_xticklabels(), visible=False); axB.set_xlabel("")
    section_panel(axC)
    hA, lA = axA.get_legend_handles_labels(); hC, lC = axC.get_legend_handles_labels()
    mx = axC.scatter([], [], marker="x", s=120, c="black", linewidths=2.0, label=r"$\mathbf{M}$ into page ($-z$)")
    mo = axC.scatter([], [], marker="o", s=120, facecolors="none", edgecolors="black", linewidths=2.0, label=r"$\mathbf{M}$ out of page ($+z$)")
    import matplotlib.lines as mlines
    import matplotlib.patheffects as pe
    gap = mlines.Line2D([], [], color="white", lw=4, path_effects=[pe.Stroke(linewidth=6.5, foreground=ps.BG), pe.Normal()])
    hA = [h for h, l in zip(hA, lA) if "Gap" not in l]; lA = [l for l in lA if "Gap" not in l]
    ps.bottom_legend(g, hA + [gap] + hC + [mx, mo], lA + [r"Gap $\Delta r$"] + lC + [r"$\mathbf{M}$ into page", r"$\mathbf{M}$ out of page"],
                     bottom)
    ps.colorbar_mid(g)
    ps.save_fixed(fig, "fig02_skatepark_geometry")


if __name__ == "__main__":
    main()

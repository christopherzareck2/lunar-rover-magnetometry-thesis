"""Figure 3 - Three-layer wall geometry (same layout language as Figures 1 and 2), drawn from the
constants of engine/three_layer_wall.py and inversion_helpers.three_layer_mesh:

  flows: x in [-50, 50] m, y in [0, 50] m (into the wall), z in [0,5], [5,10], [10,15] m
  M: flow 1 (bottom) -z, flow 2 +y (into the wall), flow 3 (top) +z, |M| = 2.4e-3 A/m
  ground traverse: y = -1 m (1 m standoff from the wall face), z = 0.5 m, x = -120..+120 m
  elevated traverse (Table 4): same line at z = 12.5 m; the height sweep is Figure 7

(A) plan view (xy), to scale: top of the wall (flow 3) and the traverse line in front of the face
(B) the wall face seen from the rover side (xz), vertical exaggeration 3
(C) vertical section through x = 0 (yz), to scale
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import paperstyle as ps

LAYERS = [(0, 5, "Flow 1", (0, 0, -1)), (5, 10, "Flow 2", (0, 1, 0)), (10, 15, "Flow 3", (0, 0, 1))]
XW, YW = 50.0, 50.0
Y_ROV, Z_LO, Z_HI, X_TR = -1.0, 0.5, 12.5, 120.0
VE = 3.0      # wall face (B); plan (A) and section (C) are to scale
YA, ZL, YC = (-10.0, 60.0), (-2.5, 17.5), (-8.0, 62.0)
XA = 75.0     # half-width of the plan view (A); the traverse continues to +-120 m


def panel_plan(ax):
    ax.set_facecolor(ps.BG)
    ax.add_patch(Rectangle((-XW, 0), 2 * XW, YW, fc=ps.SRC, ec="black", lw=2, zorder=3))
    for xa in (-35, -12, 12, 35):
        for ya in (15, 35):
            ps.mark_out(ax, xa, ya, s=110, lw=1.8)
    ps.trav(ax, [-X_TR, X_TR], [Y_ROV, Y_ROV], ps.LS_LOOP, zorder=6)
    ps.triad(ax, -68, 16, 11, ("x", "y"), color="white", out_of_page="z")
    ax.set_xlim(-XA, XA); ax.set_ylim(*YA); ax.set_aspect("equal")
    ax.set_xticks([-50, 0, 50]); ax.set_yticks([0, 25, 50])
    ax.set_ylabel(r"$y$ (m)"); plt.setp(ax.get_xticklabels(), visible=False)
    ps.panel_letter(ax, "A", dark=True, size=24)


def panel_face(ax):
    ps.ground(ax, -140, 140, 12)
    for z0, z1, name, m in LAYERS:
        ax.add_patch(Rectangle((-XW, z0), 2 * XW, z1 - z0, fc=ps.SRC, ec="black", lw=2, zorder=3))
        zc = 0.5 * (z0 + z1)
        for xa in (-35, -12, 12, 35):
            if m[1] == 0:
                ps.arrow(ax, xa, zc - m[2] * 1.6, 0, m[2] * 3.2)
            else:
                ps.mark_into(ax, xa, zc, s=80, lw=1.8)
    ps.trav(ax, [-X_TR, X_TR], [Z_LO, Z_LO], ps.LS_LOOP, "Ground traverse", zorder=6)
    ps.trav(ax, [-X_TR, X_TR], [Z_HI, Z_HI], ps.LS_ELEV, "Elevated traverse", zorder=6)
    ax.set_ylim(*ZL); ax.set_aspect(VE)
    ax.set_xticks([-50, 0, 50]); ax.set_yticks([0, 5, 10, 15])
    ax.set_xlabel(r"$x$ (m, along strike)"); ax.set_ylabel(r"$z$ (m)")
    ps.panel_letter(ax, "B", dark=False, size=24, y=0.97)


def panel_section(ax):
    ps.ground(ax, -14, 92, 18)
    for z0, z1, name, m in LAYERS:
        ax.add_patch(Rectangle((0, z0), YW, z1 - z0, fc=ps.SRC, ec="black", lw=2, zorder=3))
        zc = 0.5 * (z0 + z1)
        for ya in (12, 25, 38):
            if m[1] == 0:
                ps.arrow(ax, ya, zc - m[2] * 1.7, 0, m[2] * 3.4, lw=1.5, ms=10)
            else:
                ps.arrow(ax, ya - 4, zc, 8, 0, lw=1.5, ms=10)
        ax.text(YW + 1.2, zc, name, ha="left", va="center", fontsize=11.5, zorder=8)
    ax.set_xlim(*YC); ax.set_ylim(*ZL); ax.set_aspect("equal")
    ax.set_xticks([0, 25, 50]); ax.set_yticks([0, 5, 10, 15])
    ax.set_xlabel(r"$y$ (m, into the wall)"); ax.set_ylabel(r"$z$ (m)")
    ps.panel_letter(ax, "C", dark=False, size=24, y=0.97)


def main():
    T = ps.GRID_TOP; XLIM = (-XA, XA)
    hA, hB, hC = ps.h_true(XLIM, YA), ps.h_true(XLIM, ZL, VE), ps.h_true(YC, ZL)
    bottom = T + hA + ps.SHARE_GAP + hB
    g = ps.GeoGrid(ps.fig_height(bottom)); fig = g.fig
    x0, x1 = ps.COL_X
    axA = g.axes(x0, T, ps.COL_W, hA)
    axB = g.axes(x0, T + hA + ps.SHARE_GAP, ps.COL_W, hB, sharex=axA)
    axC = g.axes(x1, T + (bottom - T - hC) / 2 - 0.2, ps.COL_W, hC)   # centred beside A and B (label space below)
    panel_plan(axA); panel_face(axB); panel_section(axC)
    hL, lL = axB.get_legend_handles_labels()
    mx = axC.scatter([], [], marker="x", s=120, c="black", linewidths=2.0)
    mo = axC.scatter([], [], marker="o", s=120, facecolors="none", edgecolors="black", linewidths=2.0)
    ps.bottom_legend(g, hL + [mx, mo], lL + [r"$\mathbf{M}$ into page", r"$\mathbf{M}$ out of page"], bottom)
    ps.colorbar_mid(g)
    ps.save_fixed(fig, "fig03_three_layer_geometry")


if __name__ == "__main__":
    main()

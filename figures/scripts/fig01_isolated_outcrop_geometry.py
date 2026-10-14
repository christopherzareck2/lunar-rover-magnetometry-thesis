"""Figure 1 - Isolated outcrop geometry and survey paths (same layout as Figures 2 and 3).

Geometry (identical to analysis/a02_isolated_outcrop_paper_geometry.py and a09):
  prism 4 x 4 x 1 m on the ground (x, y in [-2, 2], z in [0, 1]), M = 2.4e-3 A/m along -z;
  sensor at z = 0.5 m; drive-around = closed loop 1 m outside the faces (x, y = +-3 m), starting and
  ending at (-3, -3); drive-by = straight pass along y = -3 m from x = -8 to +8 m.
(A) plan view (xy); (B) vertical section through y = 0 (xz), to scale, directly under (A) on the same x axis.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import paperstyle as ps

HALF, H, Z_S, XBY = 2.0, 1.0, 0.5, 8.0
L1 = HALF + 1.0
XL, YA, ZB = (-8.8, 8.8), (-7.0, 7.0), (-1.1, 2.1)


def square(L):
    return np.array([[-L, -L], [L, -L], [L, L], [-L, L], [-L, -L]])


def main():
    hA, hB = ps.h_true(XL, YA), ps.h_true(XL, ZB)
    T = ps.GRID_TOP
    bottom = T + hA + ps.SHARE_GAP + hB
    g = ps.GeoGrid(ps.fig_height(bottom)); fig = g.fig
    x0 = (ps.GEO_W - ps.COL_W) / 2          # single column, centred under the colorbar
    axA = g.axes(x0, T, ps.COL_W, hA)
    axB = g.axes(x0, T + hA + ps.SHARE_GAP, ps.COL_W, hB, sharex=axA)

    # (A) plan view
    axA.set_facecolor(ps.BG)
    axA.add_patch(Rectangle((-HALF, -HALF), 2 * HALF, 2 * HALF, fc=ps.SRC, ec="black", lw=2, zorder=3))
    for xm in (-1, 1):
        for ym in (-1, 1):
            ps.mark_into(axA, xm, ym, s=190)
    s1 = square(L1)
    ps.trav(axA, s1[:, 0], s1[:, 1], ps.LS_LOOP, "Drive-around")
    ps.trav(axA, [-XBY, XBY], [-L1, -L1], ps.LS_BY, "Drive-by", zorder=5)
    ps.start_dot(axA, -L1, -L1)
    ps.start_legend_handle(axA)
    ps.triad(axA, -7.6, -5.6, 1.5, ("x", "y"), color="white", out_of_page="z")
    axA.set_xlim(*XL); axA.set_ylim(*YA); axA.set_aspect("equal")
    axA.set_yticks([-5, 0, 5]); axA.set_ylabel(r"$y$ (m)")
    plt.setp(axA.get_xticklabels(), visible=False)
    ps.panel_letter(axA, "A", dark=True, size=24)

    # (B) section through y = 0, to scale
    ps.ground(axB, -12, 12, 5)
    axB.add_patch(Rectangle((-HALF, 0), 2 * HALF, H, fc=ps.SRC, ec="black", lw=2, zorder=3))
    for xa in (-1.0, 1.0):
        ps.arrow(axB, xa, 0.88, 0, -0.76, lw=1.5, ms=10)
    ps.guide(axB, [-12, 12], Z_S)
    axB.set_ylim(*ZB); axB.set_aspect("equal")
    axB.set_xticks([-5, 0, 5]); axB.set_yticks([0, 1, 2])
    axB.set_xlabel(r"$x$ (m)"); axB.set_ylabel(r"$z$ (m)")
    ps.panel_letter(axB, "B", dark=False, size=24, y=0.97)

    hA_, lA_ = axA.get_legend_handles_labels()
    mx = axB.scatter([], [], marker="x", s=120, c="black", linewidths=2.0)
    ps.bottom_legend(g, hA_ + [mx], lA_ + [r"$\mathbf{M}$ into page"], bottom)
    ps.colorbar_mid(g)
    ps.save_fixed(fig, "fig01_isolated_outcrop_geometry")


if __name__ == "__main__":
    main()

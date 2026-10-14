"""Figure 5 - Sensitivity of isolated-outcrop recovery to imaging mis-pose (pointing and range errors), (A) drive-around, (B) drive-by.

Data: results/analysis/a02_isolated_outcrop_paper_geometry.json, "lidar_grid". Grid: pointing error bound
0-6 deg x range error bound 0-0.30 m (7 x 7). Each cell: N = 100 inversions with the assumed outcrop
rigidly rotated and translated by a random draw within the bounds; data from the true geometry.
Lower-left triangle (orange): median direction error. Upper-right triangle (blue): median |dM|.
"""
import os, json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import cm, colors as mcolors
import paperstyle as ps


def grid(recs, key):
    P = sorted({r["pointing_deg"] for r in recs}); R = sorted({r["range_m"] for r in recs})
    Z = np.zeros((len(P), len(R)))
    for r in recs:
        Z[P.index(r["pointing_deg"]), R.index(r["range_m"])] = r["typical"][key]["median"]
    return np.array(P), np.array(R), Z


def panel(fig, ax, recs, letter, cax_top, cax_bot):
    P, R, D = grid(recs, "dir_err_deg")
    _, _, M = grid(recs, "dM_abs_pct")
    nd = mcolors.Normalize(np.floor(D.min() * 2) / 2, np.ceil(D.max() * 2) / 2)
    nm = mcolors.Normalize(0, np.ceil(M.max() / 2) * 2)
    co, cb = matplotlib.colormaps["Oranges"], matplotlib.colormaps["Blues"]
    for i in range(len(P)):
        for j in range(len(R)):
            x0, y0 = j - 0.5, i - 0.5
            ax.add_patch(Polygon([(x0, y0), (x0 + 1, y0), (x0, y0 + 1)], closed=True,
                                 fc=co(0.08 + 0.85 * nd(D[i, j])), ec="white", lw=0.8))
            ax.add_patch(Polygon([(x0 + 1, y0), (x0 + 1, y0 + 1), (x0, y0 + 1)], closed=True,
                                 fc=cb(0.08 + 0.85 * nm(M[i, j])), ec="white", lw=0.8))
    ax.set_xlim(-0.5, len(R) - 0.5); ax.set_ylim(-0.5, len(P) - 0.5)
    ax.set_xticks(range(len(R))[::2]); ax.set_xticklabels([f"{r:.2f}" for r in R[::2]])
    ax.set_yticks(range(len(P))[::2]); ax.set_yticklabels([f"{p:.0f}" for p in P[::2]])
    ax.set_xlabel("Range Error (m)")
    ax.set_aspect("equal")
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=30, fontweight="bold", ha="right", va="bottom")
    for cax, cmap, norm, lab, col in ((cax_top, co, nd, r"$\blacktriangleleft$ $\vec{M}_{\rm NRM}$ Direction Error ($\degree$)", ps.ORANGE),
                                     (cax_bot, cb, nm, r"$|M_{\rm NRM}|$ Error (%)", ps.BLUE)):
        sm = cm.ScalarMappable(norm=norm, cmap=mcolors.LinearSegmentedColormap.from_list(
            "t", [cmap(0.08), cmap(0.93)]))
        c = fig.colorbar(sm, cax=cax, orientation="horizontal")
        c.outline.set_linewidth(1.4)
        c.ax.tick_params(labelsize=13)
    cax_top.set_title(r"$\blacktriangleleft$ Direction error, median ($\degree$)".replace(r"$\blacktriangleleft$", "◣"),
                      color=ps.ORANGE, fontsize=15, pad=6, fontfamily="Arial Unicode MS")
    cax_bot.set_xlabel("◥ |M$_{\\rm NRM}$| error, median (%)", color=ps.BLUE, fontsize=15, labelpad=6, fontfamily="Arial Unicode MS")
    return D, M


def main():
    d = json.load(open(os.path.join(ps.RES_ANALYSIS, "a02_isolated_outcrop_paper_geometry.json")))["lidar_grid"]
    fig = plt.figure(figsize=(13, 7.6))
    axA = fig.add_axes([0.07, 0.20, 0.40, 0.60]); axB = fig.add_axes([0.56, 0.20, 0.40, 0.60])
    DA, MA = panel(fig, axA, d["drive_around"], "A", fig.add_axes([0.12, 0.88, 0.30, 0.025]), fig.add_axes([0.12, 0.07, 0.30, 0.025]))
    DB, MB = panel(fig, axB, d["drive_by"], "B", fig.add_axes([0.61, 0.88, 0.30, 0.025]), fig.add_axes([0.61, 0.07, 0.30, 0.025]))
    axA.set_ylabel(r"Pointing Error ($\degree$)")
    ps.save(fig, "fig05_lidar_sensitivity")
    for n, D, M in (("drive_around", DA, MA), ("drive_by", DB, MB)):
        print(f"{n}: direction median range {D.min():.2f}-{D.max():.2f} deg; |dM| median range {M.min():.2f}-{M.max():.2f} %")
        sub = (slice(0, 4), slice(0, 4))
        print(f"   0-3 deg / 0-0.15 m sub-grid: direction max {D[sub].max():.2f}, |dM| max {M[sub].max():.2f}")


if __name__ == "__main__":
    main()

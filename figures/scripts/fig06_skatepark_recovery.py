"""Figure 6 - Central-outcrop recovery versus neighbour separation Delta_r (skate park).

Data: results/engine/skatepark_1cm_dr_sweep_results.json (engine/skatepark_1cm_dr_sweep.py,
1 cm sampling, N = 250 inversions per point, sigma_B = 0.50 nT, 0.25 m mesh, K = 1 inversion of the
central prism with all five bodies in the data). Median direction error (orange, left) and median
absolute intensity error (blue, right). Solid = Case A (all neighbours parallel); dashed = Case B
(east and west neighbours antiparallel).
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paperstyle as ps


def series(sw, case, key):
    rows = sorted(sw[case].values(), key=lambda r: r["delta_r_m"])
    return np.array([r["delta_r_m"] for r in rows]), np.array([r[key]["median"] for r in rows])


def main():
    d = json.load(open(os.path.join(ps.RES_ENGINE, "skatepark_1cm_dr_sweep_results.json")))["sweep"]
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    ax2 = ax.twinx()
    for case, ls in (("A", "-"), ("B", (0, (5, 3)))):
        x, th = series(d, case, "dir_err_deg")
        _, dm = series(d, case, "mag_err_pct_abs")
        ax.plot(x, th, ls=ls, marker="o", color=ps.ORANGE, lw=2.4, ms=8, zorder=4)
        ax2.plot(x, dm, ls=ls, marker="s", color=ps.BLUE, lw=2.4, ms=8, zorder=4)
    ax.set_xlim(0.5, 10.5); ax.set_xticks([2, 4, 6, 8, 10])
    ax.set_ylim(0, 6); ax2.set_ylim(0, 50)
    ax.set_xlabel(r"$\Delta r$ (m)")
    ax.set_ylabel(r"$\vec{M}_{\rm NRM}$ Direction Error ($\degree$)", color=ps.ORANGE)
    ax2.set_ylabel(r"$|M_{\rm NRM}|$ Error (%)", color=ps.BLUE)
    ax.tick_params(axis="y", colors=ps.ORANGE); ax2.tick_params(axis="y", colors=ps.BLUE)
    ax.grid(True, color="#E6E6E6", lw=0.8, zorder=0)
    h = [Line2D([], [], color=ps.ORANGE, marker="o", lw=2.4, ms=8),
         Line2D([], [], color=ps.BLUE, marker="s", lw=2.4, ms=8),
         Line2D([], [], color="black", lw=2.2),
         Line2D([], [], color="black", lw=2.2, ls=(0, (5, 3)))]
    ax.legend(h, [r"$\vec{M}_{\rm NRM}$ Direction Error ($\degree$)", r"$|M_{\rm NRM}|$ Error (%)",
                  "All neighbors parallel (A)", "E and W neighbors reversed (B)"],
              loc="upper right", fontsize=12.5, framealpha=0.95)
    ps.save(fig, "fig06_skatepark_recovery")


if __name__ == "__main__":
    main()

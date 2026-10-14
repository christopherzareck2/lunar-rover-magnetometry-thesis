"""Figure 4 - Isolated-outcrop recovery versus normalized field strength (drive-around).

Data: results/analysis/a02b_detection_sweep_N1000.json, "detection_sweep" -> drive_around.
Each point is the median of 1000 inversions; the 1x point is the Section 4.2 baseline (K = 1, 1 cm sampling, sigma_B = 0.50 nT per axis).
Orange: median direction error (left axis). Blue: median absolute intensity error |dM| (right, log).
Shaded: B_NRM >= B_ref.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
import paperstyle as ps


def load():
    return json.load(open(os.path.join(ps.RES_ANALYSIS, "a02b_detection_sweep_N1000.json")))


def main():
    d = load()["detection_sweep"]["drive_around"]
    r = np.array([p["ratio"] for p in d])
    th = np.array([p["dir_err_deg"]["median"] for p in d])
    dm = np.array([p["dM_abs_pct"]["median"] for p in d])

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    ax.axvspan(1.0, 100.0, color=ps.GREY, alpha=0.9, zorder=0, lw=0)
    ax.axvline(1.0, color="#8C8C8C", ls="--", lw=2, zorder=1)
    l1, = ax.plot(r, th, "-o", color=ps.ORANGE, lw=2.4, ms=9, zorder=4)
    ax.set_xscale("log"); ax.set_xlim(0.1, 100)
    ax.set_ylim(-1.5, 32)
    ax.set_xlabel(r"$B_{\rm NRM}/B_{\rm ref}$")
    ax.set_ylabel(r"$\vec{M}_{\rm NRM}$ Direction Error ($\degree$)", color=ps.ORANGE)
    ax.tick_params(axis="y", colors=ps.ORANGE)
    ax.grid(True, which="major", color="white", alpha=0.35, lw=0.8)

    ax2 = ax.twinx()
    l2, = ax2.plot(r, dm, "-s", color=ps.BLUE, lw=2.4, ms=9, zorder=4)
    ax2.set_yscale("log"); ax2.set_ylim(0.02, 60)
    ax2.set_ylabel(r"$|M_{\rm NRM}|$ Error (%)", color=ps.BLUE)
    ax2.tick_params(axis="y", colors=ps.BLUE)
    for sp in ("left",):
        ax.spines[sp].set_color("black")
    l3, = ax.plot([], [], ls="--", color="#8C8C8C", lw=2)
    ax.legend([l1, l2, l3], [r"$\vec{M}_{\rm NRM}$ Direction Error ($\degree$)", r"$|M_{\rm NRM}|$ Error (%)",
                             r"$M_{\rm ref} = 2.4\times10^{-3}$ A m$^{-1}$"],
              loc="upper right", fontsize=13, framealpha=0.95)
    ps.save(fig, "fig04_detection_envelope")


if __name__ == "__main__":
    main()

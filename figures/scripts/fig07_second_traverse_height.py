"""Figure 7 - Three-layer recovery with a second, elevated traverse as a function of its height z2.

Data: results/engine/elevated_traverse_height_sweep_results.json
(engine/scripts/elevated_traverse_height_sweep.py): ground traverse fixed at z = 0.5 m,
second traverse at z2, both at y = -1 m, x = -120..+120 m, 1 cm sampling, joint K = 3 inversion
(undamped least squares), N = 500 inversions per height, sigma_B = 0.50 nT.
Mean direction error per flow (flow 1 at the bottom is the oldest, flow 3 at the top the youngest). The condition number of the joint operator
(minimum 19.6 at z2 = 10 m) is in the JSON and quoted in the text, not plotted. z2 = 0.5 m is the same line driven twice.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
import paperstyle as ps

# three clearly different colors (none is the blue used for intensity error elsewhere); line style and marker
# also separate the flows, as line style separates the two cases in Figure 6
FLOWS = [("layer_1", "Flow 1 (bottom)", "#8C2D04", "o", "-"),
         ("layer_2", "Flow 2 (middle)", ps.ORANGE, "s", (0, (6, 2.5))),
         ("layer_3", "Flow 3 (top)", "#6A3D9A", "^", (0, (1.2, 1.4)))]

SOLVER = "closed_form_ls"          # the undamped least-squares solution, as in Table 4


def main():
    d = json.load(open(os.path.join(ps.RES_ENGINE, "elevated_traverse_height_sweep_results.json")))
    H = sorted(d["heights"].values(), key=lambda h: h["z2_m"])
    z = np.array([h["z2_m"] for h in H])
    fig, ax = plt.subplots(figsize=(9, 6))
    thr = d["headline"]["thresholds"][SOLVER]["all_three_flows_within_5deg"]["interpolated_min_z2_m"]["x"]
    worst = np.max([[h["per_layer_" + SOLVER][k]["dir_err_deg"]["mean"] for k, *_ in FLOWS] for h in H], axis=1)
    # upper crossing: last height where the worst flow is below 5 deg, interpolated to the next sample
    ok = np.where(worst < 5)[0]
    i = ok.max()
    thr_hi = z[i] + (5 - worst[i]) * (z[i + 1] - z[i]) / (worst[i + 1] - worst[i]) if i + 1 < len(z) else z[i]

    ax.grid(True, color="#E6E6E6", lw=0.8, zorder=0)
    ax.axvspan(thr, thr_hi, color=ps.GREY, alpha=0.45, lw=0, zorder=0.5)
    zc = (thr + thr_hi) / 2
    ax.text(zc, 12.5, "all flows $<5\\degree$", ha="center", va="bottom", fontsize=13)
    ax.text(zc, 12.0, f"$z_2$ = {thr:.1f}\u2013{thr_hi:.1f} m", ha="center", va="top", fontsize=13)
    ax.axhline(5, color="#8C8C8C", ls="--", lw=1.6, zorder=1)
    ax.text(0.25, 5.5, "5\u00b0 goal", ha="left", va="bottom", fontsize=13, color="#6E6E6E")
    for key, lab, col, mk, ls in FLOWS:
        m = np.array([h["per_layer_" + SOLVER][key]["dir_err_deg"]["mean"] for h in H])
        ax.plot(z, m, marker=mk, color=col, ls=ls, lw=2.6, ms=9, mec="black", mew=0.8, label=lab, zorder=4)
    ax.set_xlim(0, 15.5); ax.set_ylim(0, 30)
    ax.set_xlabel(r"Second-traverse sensor height $z_2$ (m)")
    ax.set_ylabel(r"$\vec{M}_{\rm NRM}$ Direction Error ($\degree$)")
    ax.legend(loc="upper right", fontsize=13, handlelength=3.2, framealpha=0.95)

    ps.save(fig, "fig07_second_traverse_height")


if __name__ == "__main__":
    main()

"""Figure 8 - How correlated noise and slow drift degrade the isolated-outcrop recovery (drive-around).

Data: results/analysis/a06_isolated_robustness.json
  (A) part A3: mean direction error vs OU correlation length L (total 0.50 nT per axis), inverted without
      nuisance terms and with the five rover- and local-frame offsets fitted. L = inf is a constant offset.
      Dotted: prediction for short L, white-noise error x sqrt(2L/dx) (dx = 1 cm).
  (B) part G2: mean direction error vs the rms variation of a slow drift (L = 100 m) about its survey mean,
      added to 0.50 nT white noise, offsets fitted.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
import paperstyle as ps


def main():
    a6 = json.load(open(os.path.join(ps.RES_ANALYSIS, "a06_isolated_robustness.json")))
    A3 = a6["A_dc_offsets"]["drive_around"]["A3_correlated"]
    finite = [r for r in A3 if r["L_m"] != "inf"]
    L = np.array([r["L_m"] for r in finite]) * 100.0          # cm
    none = np.array([r["none"]["dir_err_deg"]["mean"] for r in finite])
    fit = np.array([r["rover_world_dc"]["dir_err_deg"]["mean"] for r in finite])
    inf = [r for r in A3 if r["L_m"] == "inf"][0]
    white = a6["A_dc_offsets"]["drive_around"]["A1_iid"]["none"]["dir_err_deg"]["mean"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.45, 1]})
    ax.axhline(5, color="#8C8C8C", ls="--", lw=1.8, zorder=1)
    ax.axhline(white, color="black", ls=":", lw=1.4, zorder=1, label="White noise only, 0.5 nT")
    ax.plot(L, none, "-o", color=ps.ORANGE, lw=2.4, ms=8, label="Offsets not fitted", zorder=4)
    ax.plot(L, fit, "-s", color=ps.BLUE, lw=2.4, ms=8, label="Rover and local offsets fitted", zorder=4)
    Ls = np.logspace(np.log10(0.5), np.log10(8), 50)
    ax.plot(Ls, white * np.sqrt(2 * Ls / 1.0), ls="-.", color="#6A3D9A", lw=2.6,
            label=r"$\theta_{\rm white}\sqrt{2L/\Delta x}$", zorder=6)
    xinf = L.max() * 3.2
    ax.plot([xinf], [inf["none"]["dir_err_deg"]["mean"]], "o", color=ps.ORANGE, ms=9, zorder=5)
    ax.plot([xinf], [inf["rover_world_dc"]["dir_err_deg"]["mean"]], "s", color=ps.BLUE, ms=9, zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(0.35, xinf * 1.8)
    ticks = [1, 10, 100, 1000, 10000]
    ax.set_xticks(ticks + [xinf])
    ax.set_xticklabels(["1 cm", "10 cm", "1 m", "10 m", "100 m", r"$\infty$"])
    ax.set_ylim(0, 92)
    ax.set_xlabel(r"Noise correlation length $L$ (1 cm = 1 s at 1 cm s$^{-1}$)")
    ax.set_ylabel(r"$\vec{M}_{\rm NRM}$ Direction Error ($\degree$)")
    ax.text(120, 6.0, r"$5\degree$ goal", fontsize=13, color="#6E6E6E")
    ax.legend(loc="upper center", bbox_to_anchor=(0.56, 0.995), fontsize=13, ncol=2)
    ps.panel_letter(ax, "A", dark=False)

    G2 = a6["G_gradient_and_drift"]["drive_around"]["drift_L100m"]
    x = np.array([0.0] + [r["rms_about_mean_nT"] for r in G2[1:]])
    y = np.array([r["offsets_fitted"]["dir_err_deg"]["mean"] for r in G2])
    bx.axhline(5, color="#8C8C8C", ls="--", lw=1.8, zorder=1)
    bx.plot(x, y, "-s", color=ps.BLUE, lw=2.4, ms=8, zorder=4)
    bx.set_xlabel("Slow drift about survey mean (nT rms per axis)")
    bx.set_ylabel(r"$\vec{M}_{\rm NRM}$ Direction Error ($\degree$)")
    bx.set_xlim(-0.005, 0.155); bx.set_ylim(0, 16)
    bx.text(0.100, 5.3, r"$5\degree$ goal", fontsize=13, color="#6E6E6E")
    ps.panel_letter(bx, "B", dark=False)
    fig.tight_layout(w_pad=3)
    ps.save(fig, "fig08_noise_requirement")


if __name__ == "__main__":
    main()

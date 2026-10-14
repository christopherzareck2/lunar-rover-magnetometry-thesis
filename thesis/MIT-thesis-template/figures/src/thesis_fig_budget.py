"""Thesis Figure (combined error budget of Table 5.1 as bars). Plotting only.

Reads the STORED results file results/analysis/a09_systematics.json of the code release and draws
bars. For each case, the plotted values are the min and max over magnetization directions of
all_sources/dir_err_deg/mean (bar) and noise_free_bias/dir_err_deg/mean (dark segment).

Usage: python thesis_fig_budget.py /path/to/a09_systematics.json /output/dir
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": ["Arial", "Liberation Sans"], "mathtext.fontset": "custom", "mathtext.rm": "sans",
                     "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold", "pdf.fonttype": 42,
                     "axes.linewidth": 1.6})
a9 = json.load(open(sys.argv[1]))
OUT = sys.argv[2]

GROUPS = [
    ("One source at a time (loops 0.5 m off center)", [
        ("Noise only", "Z_single_none"), ("Site field", "Z_single_site"), ("Pose", "Z_single_pose"),
        ("Burial (b ≤ 1 m)", "Z_single_burial"), ("Drift (0.03 nT)", "Z_single_drift"),
        ("Tilt (0.05 nT)", "Z_single_tilt"), ("Induced magnetization", "Z_single_induced")]),
    ("All sources together", [
        ("Full tolerances, off center", "U_budget_full_offcentre"), ("Half, off center", "V_budget_half_offcentre"),
        ("Half, off center, root fitted", "F2_half_offcentre_root"), ("Quarter, off center", "Y_budget_quarter_offcentre"),
        ("Half, centered", "X_budget_half_centred")]),
    ("Requirement set", [
        ("b ≤ 0.5 m, loops 0.1 m off", "REQ_a_dx0.1_b0.5"), ("b ≤ 1 m, loops 0.1 m off", "REQ_b_dx0.1_b1"),
        ("Burial known, 0.5 m off", "REQ_c_dx0.5_b0")]),
    ("Requirement set, both mitigations", [
        ("b ≤ 0.5 m, loops 0.1 m off", "REQF_a_dx0.1_b0.5"), ("b ≤ 1 m, loops 0.1 m off", "REQF_b_dx0.1_b1"),
        ("Burial known, 0.5 m off", "REQF_c_dx0.5_b0")]),
]


def span(key, k):
    v = [r[k]["dir_err_deg"]["mean"] for d, r in a9[key].items() if d != "config"]
    return min(v), max(v)


def main():
    n = sum(len(items) for _, items in GROUPS) + len(GROUPS)
    fig, ax = plt.subplots(figsize=(10.5, 0.52 * n + 1.4))
    y, yt, yl, table = 0, [], [], []
    ax.axvline(5, ls="--", color="k", lw=1.4, zorder=1)
    ax.axvline(16.2, color="0.7", lw=0.8, zorder=1)
    for g, items in GROUPS:
        ax.text(0.2, y, g, fontsize=15, fontweight="bold", va="center", zorder=6,
                bbox=dict(fc="white", ec="none", pad=2.5))
        y += 1
        for name, key in items:
            lo, hi = span(key, "all_sources")
            blo, bhi = span(key, "noise_free_bias")
            ax.plot([lo, hi], [y, y], lw=11, color="#F2C29B", solid_capstyle="butt", zorder=3)
            if bhi - blo > 1e-6:
                ax.plot([blo, bhi], [y, y], lw=5, color="#5B2A86", solid_capstyle="butt", zorder=4)
            else:  # a single value to within 1e-6 deg (zero bias): a segment of zero length would be invisible, so mark the value itself
                ax.plot([blo], [y], marker="D", ms=8, color="#5B2A86", ls="none", zorder=12, clip_on=False)
            ax.text(16.4, y, f"{lo:.1f}–{hi:.1f}°", va="center", ha="left", fontsize=14)
            table.append((key, round(lo, 4), round(hi, 4), round(blo, 4), round(bhi, 4)))
            yt.append(y); yl.append(name)
            y += 1
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=14)
    ax.set_ylim(y - 0.4, -0.6)
    ax.set_xlim(0, 19.6)
    ax.set_xticks(range(0, 17, 2))
    ax.tick_params(axis="x", labelsize=14)
    ax.set_xlabel("Direction error (deg)", fontsize=15)
    ax.plot([], [], lw=11, color="#F2C29B", label="Mean error, range over directions")
    ax.plot([], [], lw=5, color="#5B2A86", label="Noise-free bias, range")
    ax.plot([], [], marker="D", ms=7, color="#5B2A86", ls="none", label="Noise-free bias below $10^{-6}$° (point)")
    ax.plot([], [], ls="--", color="k", lw=1.4, label="5° goal")
    ax.legend(loc="upper right", bbox_to_anchor=(16.0, 0.6), bbox_transform=ax.transData, fontsize=13, frameon=True,
              framealpha=1.0)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "thesis_fig_budget.pdf"), metadata={"CreationDate": None})
    fig.savefig(os.path.join(OUT, "thesis_fig_budget.png"), dpi=150)
    json.dump(table, open(os.path.join(OUT, "thesis_fig_budget_plotted_values.json"), "w"), indent=1)
    print("wrote", OUT, "font:", plt.rcParams["font.family"])


if __name__ == "__main__":
    main()

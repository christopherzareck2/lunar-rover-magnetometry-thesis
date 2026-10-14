"""Result figures of thesis Chapters 4 and 5 (Figs. 4.1-4.4, 5.1, 5.2), drawn from the stored results only.

Each figure has a ``data_<name>()`` function that reads the stored JSON and returns exactly the plotted numbers
(written next to the figure as ``<name>.values.json`` and compared with the companion-paper scripts by
``tests/test_figures.py``), and a ``draw_<name>(d)`` function that draws them in the thesis style.
"""
from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.transforms import blended_transform_factory

from .. import plotting as st
from ..paths import result

FigData = dict[str, Any]


def _load(name: str) -> Any:
    with open(result(name)) as f:
        return json.load(f)


def _arr(x) -> list[float]:
    return [float(v) for v in x]


# =============================================================================================== Fig. 4.1
def data_detection_envelope() -> FigData:
    """Median direction and |intensity| errors vs B_NRM/B_ref, drive-around (a02b, 1000 inversions per point)."""
    d = _load("a02b_detection_sweep_N1000.json")["detection_sweep"]["drive_around"]
    return {"ratio": _arr(p["ratio"] for p in d),
            "dir_err_median_deg": _arr(p["dir_err_deg"]["median"] for p in d),
            "dM_abs_median_pct": _arr(p["dM_abs_pct"]["median"] for p in d)}


def draw_detection_envelope(d: FigData) -> plt.Figure:
    fig = st.figure("0.7", 3.7, layout="constrained")
    ax, bx = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    for a in (ax, bx):
        a.axvspan(1.0, 100.0, color=st.SHADE, lw=0, zorder=0)
        a.axvline(1.0, color=st.GUIDE, ls=(0, (4, 2.5)), lw=0.9, zorder=1)
        a.grid(True, which="major", axis="y", zorder=0.5)
        a.set_axisbelow(True)
    r = np.array(d["ratio"])
    ax.plot(r, d["dir_err_median_deg"], "-o", color=st.DIRECTION, mec="white", mew=0.5, zorder=4)
    bx.plot(r, d["dM_abs_median_pct"], "-s", color=st.INTENSITY, mec="white", mew=0.5, zorder=4)
    ax.set_xscale("log"); ax.set_xlim(0.1, 100)
    ax.set_ylim(0, 32); ax.set_yticks([0, 10, 20, 30])
    bx.set_yscale("log"); bx.set_ylim(0.02, 60)
    ax.set_ylabel(r"Direction error, median ($^{\circ}$)")
    bx.set_ylabel(r"$|$Intensity error$|$, median (%)")
    bx.set_xlabel(r"$B_{\mathrm{NRM}}/B_{\mathrm{ref}}$")
    ax.text(1.12, 29, r"$M_{\mathrm{ref}} = 2.4\times10^{-3}$ A m$^{-1}$", fontsize=st.BASE_PT - 1, color=st.MUTED,
            va="top")
    st.panel_label(ax, "A"); st.panel_label(bx, "B")
    return fig


# =============================================================================================== Fig. 4.2
def data_lidar_sensitivity() -> FigData:
    """Median direction and |intensity| errors over the pointing x range grid of imaging mis-pose (a02 lidar_grid)."""
    g = _load("a02_isolated_outcrop_paper_geometry.json")["lidar_grid"]
    out: FigData = {}
    for name, recs in g.items():
        P = sorted({r["pointing_deg"] for r in recs}); R = sorted({r["range_m"] for r in recs})
        D = np.zeros((len(P), len(R))); M = np.zeros_like(D)
        for r in recs:
            i, j = P.index(r["pointing_deg"]), R.index(r["range_m"])
            D[i, j] = r["typical"]["dir_err_deg"]["median"]; M[i, j] = r["typical"]["dM_abs_pct"]["median"]
        out[name] = {"pointing_deg": _arr(P), "range_m": _arr(R), "dir_err_median_deg": D.tolist(),
                     "dM_abs_median_pct": M.tolist()}
    return out


def _tri_panel(fig, ax, rec):
    P, R = rec["pointing_deg"], rec["range_m"]
    D, M = np.array(rec["dir_err_median_deg"]), np.array(rec["dM_abs_median_pct"])
    nd = mcolors.Normalize(np.floor(D.min() * 2) / 2, np.ceil(D.max() * 2) / 2)
    nm = mcolors.Normalize(0, np.ceil(M.max() / 2) * 2)
    co, cb = plt.get_cmap("Oranges"), plt.get_cmap("Blues")
    for i in range(len(P)):
        for j in range(len(R)):
            x0, y0 = j - 0.5, i - 0.5
            ax.add_patch(Polygon([(x0, y0), (x0 + 1, y0), (x0, y0 + 1)], closed=True,
                                 fc=co(0.08 + 0.85 * nd(D[i, j])), ec="white", lw=0.5))
            ax.add_patch(Polygon([(x0 + 1, y0), (x0 + 1, y0 + 1), (x0, y0 + 1)], closed=True,
                                 fc=cb(0.08 + 0.85 * nm(M[i, j])), ec="white", lw=0.5))
    ax.set_xlim(-0.5, len(R) - 0.5); ax.set_ylim(-0.5, len(P) - 0.5); ax.set_aspect("equal")
    ax.set_xticks(range(len(R))[::2]); ax.set_xticklabels([f"{r:.2f}" for r in R[::2]])
    ax.set_yticks(range(len(P))[::2]); ax.set_yticklabels([f"{p:.0f}" for p in P[::2]])
    ax.set_xlabel("Range error bound (m)")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    return (co, nd), (cb, nm)


def draw_lidar_sensitivity(d: FigData) -> plt.Figure:
    W, H = st.WIDTH["full"], 3.95
    fig = st.figure("full", H)
    size, bottom = 2.3, 0.48

    def box(x, y, w, h):                          # inches from the bottom-left corner -> figure fraction
        return [x / W, y / H, w / W, h / H]

    for x0, name, letter, title in ((0.62, "drive_around", "A", "Drive-around"), (3.78, "drive_by", "B", "Drive-by")):
        ax = fig.add_axes(box(x0, bottom, size, size))
        (co, nd), (cb, nm) = _tri_panel(fig, ax, d[name])
        if letter == "A":
            ax.set_ylabel(r"Pointing error bound ($^{\circ}$)")
        ax.text(0.0, 1.035, title, transform=ax.transAxes, ha="left", va="bottom")
        st.panel_label(ax, letter, x=-0.06, y=1.035)
        for row, (cmap, norm, lab, lower) in enumerate(((co, nd, r"Direction error ($^{\circ}$)", True),
                                                       (cb, nm, r"$|$Intensity error$|$ (%)", False))):
            yb = bottom + size + 0.80 - row * 0.40
            cax = fig.add_axes(box(x0 + 0.26, yb, 1.05, 0.085))
            sm = cm.ScalarMappable(norm=norm, cmap=mcolors.LinearSegmentedColormap.from_list(
                "t", [cmap(0.08), cmap(0.93)]))
            c = fig.colorbar(sm, cax=cax, orientation="horizontal")
            c.outline.set_linewidth(0.5)
            c.ax.tick_params(labelsize=st.BASE_PT - 1.5, length=2, pad=1.5)
            c.ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            cax.text(1.06, 0.5, lab, transform=cax.transAxes, ha="left", va="center", fontsize=st.BASE_PT - 1)
            icon = fig.add_axes(box(x0 + 0.05, yb - 0.02, 0.13, 0.13))
            tri = [(0, 0), (1, 0), (0, 1)] if lower else [(1, 0), (1, 1), (0, 1)]
            icon.add_patch(Polygon(tri, closed=True, fc=cmap(0.7), ec="none"))
            icon.set_xlim(0, 1); icon.set_ylim(0, 1); icon.axis("off")
    return fig


# =============================================================================================== Fig. 4.3
def data_skatepark_recovery() -> FigData:
    """Median direction and |intensity| errors of the central outcrop vs neighbor gap, cases A and B."""
    sw = _load("skatepark_1cm_dr_sweep_results.json")["sweep"]
    out: FigData = {}
    for case in ("A", "B"):
        rows = sorted(sw[case].values(), key=lambda r: r["delta_r_m"])
        out[case] = {"delta_r_m": _arr(r["delta_r_m"] for r in rows),
                     "dir_err_median_deg": _arr(r["dir_err_deg"]["median"] for r in rows),
                     "dM_abs_median_pct": _arr(r["mag_err_pct_abs"]["median"] for r in rows)}
    return out


def draw_skatepark_recovery(d: FigData) -> plt.Figure:
    fig = st.figure("0.7", 3.7, layout="constrained")
    ax, bx = fig.subplots(2, 1, sharex=True)
    styles = {"A": dict(ls="-", mfc=None), "B": dict(ls=(0, (4, 2)), mfc="white")}
    for case, s in styles.items():
        r = d[case]
        ax.plot(r["delta_r_m"], r["dir_err_median_deg"], marker="o", color=st.DIRECTION, ls=s["ls"],
                mfc=s["mfc"] or st.DIRECTION, mew=0.9, zorder=4)
        bx.plot(r["delta_r_m"], r["dM_abs_median_pct"], marker="s", color=st.INTENSITY, ls=s["ls"],
                mfc=s["mfc"] or st.INTENSITY, mew=0.9, zorder=4)
    for a in (ax, bx):
        a.grid(True, axis="y"); a.set_axisbelow(True)
    ax.set_xlim(0.5, 10.5); bx.set_xticks([1, 2, 4, 6, 8, 10])
    ax.set_ylim(0, 6); bx.set_ylim(0, 50)
    ax.set_ylabel(r"Direction error, median ($^{\circ}$)")
    bx.set_ylabel(r"$|$Intensity error$|$, median (%)")
    bx.set_xlabel(r"Neighbor gap $\Delta r$ (m)")
    h = [Line2D([], [], color=st.INK, ls="-", marker="o", mfc=st.INK, ms=3.5),
         Line2D([], [], color=st.INK, ls=(0, (4, 2)), marker="o", mfc="white", ms=3.5)]
    ax.legend(h, ["All neighbors parallel", "E and W neighbors reversed"], loc="upper right")
    st.panel_label(ax, "A"); st.panel_label(bx, "B")
    return fig


# =============================================================================================== Fig. 4.4
FLOW_KEYS = [("layer_1", "Flow 1 (bottom, oldest)", "o", "-"),
             ("layer_2", "Flow 2 (middle)", "s", (0, (5, 2))),
             ("layer_3", "Flow 3 (top, youngest)", "^", (0, (1, 1.2)))]


def data_second_traverse_height() -> FigData:
    """Mean direction error of each flow vs second-traverse height z2 (undamped LS, 500 inversions per height),
    and the interpolated heights between which all three flows are below 5 deg."""
    dd = _load("elevated_traverse_height_sweep_results.json")
    H = sorted(dd["heights"].values(), key=lambda h: h["z2_m"])
    z = np.array([h["z2_m"] for h in H])
    solver = "closed_form_ls"
    flows = {k: _arr(h["per_layer_" + solver][k]["dir_err_deg"]["mean"] for h in H) for k, *_ in FLOW_KEYS}
    thr = dd["headline"]["thresholds"][solver]["all_three_flows_within_5deg"]["interpolated_min_z2_m"]["x"]
    worst = np.max([[flows[k][i] for k, *_ in FLOW_KEYS] for i in range(len(z))], axis=1)
    i = np.where(worst < 5)[0].max()
    thr_hi = z[i] + (5 - worst[i]) * (z[i + 1] - z[i]) / (worst[i + 1] - worst[i]) if i + 1 < len(z) else z[i]
    return {"z2_m": _arr(z), "flows": flows, "band_m": [float(thr), float(thr_hi)]}


def draw_second_traverse_height(d: FigData) -> plt.Figure:
    fig = st.figure("0.7", 2.9, layout="constrained")
    ax = fig.subplots()
    lo, hi = d["band_m"]
    ax.axvspan(lo, hi, color=st.SHADE, lw=0, zorder=0)
    ax.grid(True, axis="y"); ax.set_axisbelow(True)
    st.goal_line(ax, 5.0, r"$5^{\circ}$ goal", x_text=0.3)
    ax.text((lo + hi) / 2, 29.3, f"all flows $<5^{{\\circ}}$\n$z_2$ = {lo:.1f}" + r"$\endash$" + f"{hi:.1f} m",
            ha="center", va="top", fontsize=st.BASE_PT - 1, linespacing=1.3)
    for (key, lab, mk, ls), col in zip(FLOW_KEYS, st.FLOWS):
        ax.plot(d["z2_m"], d["flows"][key], marker=mk, color=col, ls=ls, mec="white", mew=0.5, label=lab, zorder=4)
    ax.set_xlim(0, 15.5); ax.set_ylim(0, 30)
    ax.set_xlabel(r"Second-traverse sensor height $z_2$ (m)")
    ax.set_ylabel(r"Direction error, mean ($^{\circ}$)")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.52), handlelength=2.8)
    return fig


# =============================================================================================== Fig. 5.1
def data_noise_requirement() -> FigData:
    """(A) mean direction error vs OU correlation length (a06 A3) with and without offsets fitted, the white-noise
    error and its sqrt(2L/dx) scaling; (B) mean direction error vs slow-drift rms (a06 G2), offsets fitted."""
    a6 = _load("a06_isolated_robustness.json")
    A3 = a6["A_dc_offsets"]["drive_around"]["A3_correlated"]
    finite = [r for r in A3 if r["L_m"] != "inf"]
    inf = [r for r in A3 if r["L_m"] == "inf"][0]
    G2 = a6["G_gradient_and_drift"]["drive_around"]["drift_L100m"]
    return {"L_cm": _arr(np.array([r["L_m"] for r in finite]) * 100.0),
            "none_deg": _arr(r["none"]["dir_err_deg"]["mean"] for r in finite),
            "offsets_deg": _arr(r["rover_world_dc"]["dir_err_deg"]["mean"] for r in finite),
            "inf_none_deg": float(inf["none"]["dir_err_deg"]["mean"]),
            "inf_offsets_deg": float(inf["rover_world_dc"]["dir_err_deg"]["mean"]),
            "white_deg": float(a6["A_dc_offsets"]["drive_around"]["A1_iid"]["none"]["dir_err_deg"]["mean"]),
            "drift_rms_nT": _arr([0.0] + [r["rms_about_mean_nT"] for r in G2[1:]]),
            "drift_deg": _arr(r["offsets_fitted"]["dir_err_deg"]["mean"] for r in G2)}


def draw_noise_requirement(d: FigData) -> plt.Figure:
    fig = st.figure("full", 2.75, layout="constrained")
    ax, bx = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.45, 1]})
    L = np.array(d["L_cm"])
    xinf = L.max() * 3.2
    for a in (ax, bx):
        a.grid(True, axis="y"); a.set_axisbelow(True)
    st.goal_line(ax, 5.0, r"$5^{\circ}$ goal", x_text=150)
    ax.axhline(d["white_deg"], color=st.INK, ls=(0, (1, 1.5)), lw=0.9, zorder=1, label="White noise only, 0.5 nT")
    ax.plot(L, d["none_deg"], "-o", color=st.DIRECTION, mec="white", mew=0.5, label="Offsets not fitted", zorder=4)
    ax.plot(L, d["offsets_deg"], "-s", color=st.INTENSITY, mec="white", mew=0.5,
            label="Rover and local offsets fitted", zorder=4)
    Ls = np.logspace(np.log10(0.5), np.log10(8), 50)
    ax.plot(Ls, d["white_deg"] * np.sqrt(2 * Ls / 1.0), ls=(0, (5, 1.5, 1, 1.5)), color=st.OKABE_ITO["purple"], lw=1.3,
            label=r"$\theta_{\mathrm{white}}\sqrt{2L/\Delta x}$", zorder=6)
    ax.plot([xinf], [d["inf_none_deg"]], "o", color=st.DIRECTION, mec="white", mew=0.5, zorder=5)
    ax.plot([xinf], [d["inf_offsets_deg"]], "s", color=st.INTENSITY, mec="white", mew=0.5, zorder=5)
    ax.set_xscale("log"); ax.set_xlim(0.35, xinf * 1.8)
    ax.set_xticks([1, 10, 100, 1000, 10000, xinf])
    ax.set_xticklabels(["1 cm", "10 cm", "1 m", "10 m", "100 m", r"$\infty$"])
    ax.minorticks_off()
    ax.set_ylim(0, 92)
    ax.set_xlabel(r"Noise correlation length $L$ (1 cm = 1 s at 1 cm s$^{-1}$)")
    ax.set_ylabel(r"Direction error, mean ($^{\circ}$)")
    ax.legend(loc="upper left", ncol=1, fontsize=st.BASE_PT - 1.5)
    bx.plot(d["drift_rms_nT"], d["drift_deg"], "-s", color=st.INTENSITY, mec="white", mew=0.5, zorder=4)
    st.goal_line(bx, 5.0, r"$5^{\circ}$ goal", x_text=0.10)
    bx.set_xlabel("Slow-drift rms about survey mean (nT)")
    bx.set_ylabel(r"Direction error, mean ($^{\circ}$)")
    bx.set_xlim(-0.005, 0.155); bx.set_ylim(0, 16)
    st.panel_label(ax, "A"); st.panel_label(bx, "B")
    return fig


# =============================================================================================== Fig. 5.2
BUDGET_GROUPS = [
    ("One source at a time (loops 0.5 m off center)", [
        ("Noise only", "Z_single_none"), ("Site field", "Z_single_site"), ("Pose", "Z_single_pose"),
        (r"Burial ($b \leq 1$ m)", "Z_single_burial"), ("Drift (0.03 nT)", "Z_single_drift"),
        ("Tilt (0.05 nT)", "Z_single_tilt"), ("Induced magnetization", "Z_single_induced")]),
    ("All sources together", [
        ("Full tolerances, off center", "U_budget_full_offcentre"), ("Half, off center", "V_budget_half_offcentre"),
        ("Half, off center, root fitted", "F2_half_offcentre_root"), ("Quarter, off center", "Y_budget_quarter_offcentre"),
        ("Half, centered", "X_budget_half_centred")]),
    ("Requirement set", [
        (r"$b \leq 0.5$ m, loops 0.1 m off", "REQ_a_dx0.1_b0.5"), (r"$b \leq 1$ m, loops 0.1 m off", "REQ_b_dx0.1_b1"),
        (r"No buried part ($b = 0$), 0.5 m off", "REQ_c_dx0.5_b0"),
        (r"Known depth $b \leq 1$ m, 0.5 m off", "y6_03:Requirement set, burial known")]),
    ("Requirement set, both mitigations", [
        (r"$b \leq 0.5$ m, loops 0.1 m off", "REQF_a_dx0.1_b0.5"), (r"$b \leq 1$ m, loops 0.1 m off", "REQF_b_dx0.1_b1"),
        (r"No buried part ($b = 0$), 0.5 m off", "REQF_c_dx0.5_b0"),
        (r"Known depth $b \leq 1$ m, 0.5 m off", "y6_03:Mitigated set, burial known")]),
]
#: The rows keyed REQ_c / REQF_c are the b = 0 runs (review finding Y6-03), so they are labeled "No buried part"
#: by default (they were "Burial known" before 2026-10-13). The known-depth rows come from the later rerun
#: studies/results/y6_03_known_burial.json (true b drawn in 0..1 m and put into the model; Table 5.1, last two rows).
#: REVIEW_LABELS is kept so --review-labels still runs; it now changes only the legend wording.
REVIEW_LABELS = {"Burial known, 0.5 m off": r"No buried part ($b = 0$), 0.5 m off"}
Y6_03 = "studies/results/y6_03_known_burial.json"


def data_budget() -> FigData:
    """Per budget case: min and max over magnetization directions of the mean direction error with all sources
    (bar) and of the noise-free mean direction error (dark bar), from a09_systematics.json (Table 5.1)."""
    a9 = _load("a09_systematics.json")
    from ..paths import ROOT
    with open(ROOT / Y6_03) as f:
        y6 = json.load(f)["rows"]

    def span(key, k):
        if key.startswith("y6_03:"):                    # known-depth rerun: true b (0..1 m) in the model
            dirs = y6[key.split(":", 1)[1]]["new_known_b_U0_1"]["directions"]
            kk = "dir_err" if k == "all_sources" else "bias_mean_nf_angle"
            v = [r[kk]["mean"] for r in dirs.values()]
        else:
            v = [r[k]["dir_err_deg"]["mean"] for dn, r in a9[key].items() if dn != "config"]
        return float(min(v)), float(max(v))

    rows = []
    for g, items in BUDGET_GROUPS:
        for label, key in items:
            lo, hi = span(key, "all_sources"); blo, bhi = span(key, "noise_free_bias")
            rows.append({"group": g, "label": label, "key": key, "all_sources_deg": [lo, hi],
                         "noise_free_deg": [blo, bhi]})
    return {"rows": rows}


def draw_budget(d: FigData, review_labels: bool = False) -> plt.Figure:
    rows = d["rows"]
    groups = list(dict.fromkeys(r["group"] for r in rows))
    n = len(rows) + len(groups)
    H = 0.205 * n + 1.15
    fig = st.figure("0.9", H)
    ax = fig.add_axes([0.355, 0.85 / H, 0.53, 1 - 0.97 / H])
    light, dark = "#F2B07E", "#5B2A86"           # pre-blended light vermillion; dark purple (both opaque)
    y, yt, yl = 0, [], []
    st.goal_line(ax, 5.0, None, orientation="v")
    for g in groups:
        ax.text(0.012, y, g, transform=blended_transform_factory(fig.transFigure, ax.transData), ha="left",
                va="center", family=st.BOLD_FAMILY, fontsize=st.BASE_PT - 0.5)
        y += 1
        for r in (r for r in rows if r["group"] == g):
            lo, hi = r["all_sources_deg"]; blo, bhi = r["noise_free_deg"]
            ax.plot([lo, hi], [y, y], lw=6.5, color=light, solid_capstyle="butt", zorder=3)
            if bhi - blo > 1e-6:
                ax.plot([blo, bhi], [y, y], lw=3.0, color=dark, solid_capstyle="butt", zorder=4)
            else:
                ax.plot([blo], [y], marker="D", ms=3.8, color=dark, ls="none", zorder=12, clip_on=False)
            ax.text(1.015, y, f"{lo:.1f}" + r"$\endash$" + f"{hi:.1f}" + r"$^{\circ}$", transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=st.BASE_PT - 1)
            lab = REVIEW_LABELS.get(r["label"], r["label"]) if review_labels else r["label"]
            yt.append(y); yl.append(lab)
            y += 1
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=st.BASE_PT - 1)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(y - 0.4, -0.6)
    ax.set_xlim(0, 16.2); ax.set_xticks(range(0, 17, 2))
    ax.grid(True, axis="x"); ax.set_axisbelow(True)
    ax.set_xlabel(r"Direction error ($^{\circ}$)")
    h = [Line2D([], [], lw=6.5, color=light), Line2D([], [], lw=3.0, color=dark),
         Line2D([], [], marker="D", ms=3.8, color=dark, ls="none"),
         Line2D([], [], color=st.GUIDE, ls=(0, (4, 2.5)), lw=0.9)]
    lab = ["Mean error, range over directions",
           "Noise-free mean error, range",
           r"Noise-free value below $10^{-6}$ deg", r"$5^{\circ}$ goal"]
    fig.legend(h, lab, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
               fontsize=st.BASE_PT - 1, columnspacing=1.5)
    return fig

"""Figures for the amortized-inference study (thesis Figs. 7.x, files ch7_burial.pdf and ch7_calib.pdf), drawn in the
thesis house style (lunarmag.plotting: Computer Modern, Okabe-Ito colours, exact thesis width, no transparency).

  ml_fig1_burial   (A) mean direction error and (B) mean signed intensity error against the true buried extent b,
                   every other nuisance drawn from the training prior, |M| = M_ref, random directions.
  ml_fig2_calib    (A) coverage of the predicted 3-D credible ellipsoids (prior test set, and pose errors twice the
                   training range), against the least-squares formal covariance;
                   (B) mean-square error in the linear-Gaussian (noise-only) case relative to the Cramer-Rao bound,
                   with the van Trees bound for the same prior.

The numbers are computed exactly as in the first version of this script (2026-10-13 ML study); only
the drawing changed (restyled in the thesis house style; tests/test_figures.py checks the plotted values).
Colour roles follow the captions: least squares blue, root-body fit grey, profile fit black (dashed), network orange.

Run:  python ml/fig_ml.py
Out:  ml/figures/ml_fig1_burial.{pdf,png}, ml/figures/ml_fig2_calib.{pdf,png} (PDF/A-1b ready). `make figures` copies
      them into the thesis as figures/ch7_burial.pdf and figures/ch7_calib.pdf.
"""
import os, sys, json, glob
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
sys.path.insert(0, os.path.join(C.ROOT, "src"))
from lunarmag import plotting as st                 # noqa: E402  (thesis house style)
from lunarmag.pdfa import make_pdfa1b_ready         # noqa: E402
import matplotlib.pyplot as plt                     # noqa: E402
from train import features, MODELS                  # noqa: E402
from evaluate import ensemble_predict               # noqa: E402

GREY_D = "#7F7F7F"
STY = {  # estimator: (label, colour, marker, linestyle)
    "LS11": ("LS, 11 columns (thesis)", st.INTENSITY, "s", "-"),
    "LS17": ("LS, 17 columns", st.INTENSITY, "D", (0, (5, 3))),
    "LS11_root1m": ("LS + 1 m root body", GREY_D, "^", "-"),
    "profile": ("Profile LS over $b$", st.INK, "v", (0, (5, 3))),
    "NN": ("Neural estimator (ensemble)", st.DIRECTION, "o", "-"),
}


def save(fig, name):
    os.makedirs(C.FIGS, exist_ok=True)
    st.lint_text(fig)
    pdf, png = st.save(fig, C.FIGS, name, png_dir=C.FIGS)
    make_pdfa1b_ready(pdf)
    print(f"wrote ml/figures/{name}.pdf/.png")


def panel(ax, letter):
    """Bold uppercase panel letter (A, B), the house style of every other thesis figure."""
    st.panel_label(ax, letter.upper())


def data_burial(R):
    bs = np.round(np.arange(0, 1.01, 0.1), 1)
    rows = {}
    for k in STY:
        g = [R["sets"][f"test_b{b:.1f}"][k] for b in bs]
        rows[k] = {"dir_mean": [x["dir_err_deg"]["mean"] for x in g], "dir_sem": [x["dir_err_deg"]["sem"] for x in g],
                   "dM_mean": [x["int_err_signed_pct"]["mean"] for x in g],
                   "dM_sem": [x["int_err_signed_pct"]["sem"] for x in g]}
    return bs, rows


def fig1(R):
    bs, rows = data_burial(R)
    with st.style():
        fig = st.figure("full", 2.9)
        axs = fig.subplots(1, 2, gridspec_kw=dict(left=0.08, right=0.985, bottom=0.15, top=0.78, wspace=0.28))
        for k, (lab, col, mk, ls) in STY.items():
            r = rows[k]
            e, se = np.array(r["dir_mean"]), np.array(r["dir_sem"])
            dm, dse = np.array(r["dM_mean"]), np.array(r["dM_sem"])
            kw = dict(color=col, marker=mk, ls=ls, lw=1.1, ms=3.8, mfc="white" if k in ("LS17", "profile") else col,
                      mew=0.9, elinewidth=0.8, label=lab)
            axs[0].errorbar(bs, e, yerr=2 * se, capsize=0, **kw)
            axs[1].errorbar(bs, dm, yerr=2 * dse, capsize=0, **kw)
        axs[0].axhline(5.0, color=GREY_D, lw=0.9, ls=(0, (1, 1.6)))
        axs[0].text(1.0, 5.2, r"$5^{\circ}$ goal", color=st.MUTED, fontsize=st.BASE_PT - 1, ha="right", va="bottom")
        axs[1].axhline(0.0, color=GREY_D, lw=0.9, ls=(0, (1, 1.6)))
        axs[0].set_ylabel(r"Mean direction error ($^{\circ}$)"); axs[1].set_ylabel("Mean signed intensity error (%)")
        for ax, L in zip(axs, "ab"):
            ax.set_xlabel("True buried extent $b$ (m)"); ax.set_xlim(-0.03, 1.03)
            ax.grid(True, axis="y"); ax.set_axisbelow(True)
            panel(ax, L)
        axs[0].set_ylim(0, None)
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=st.BASE_PT - 1,
                   columnspacing=1.2, handlelength=2.6)
        save(fig, "ml_fig1_burial")


def data_calib(R):
    """Coverage curves and MSE/CRLB values, computed exactly as in the first version of this script."""
    sv = C.Survey(0.5); dof = sv.nd - sv.k
    full = sorted(glob.glob(os.path.join(MODELS, "full_s*.npz")))
    none = sorted(glob.glob(os.path.join(MODELS, "none_s*.npz")))
    lv = np.linspace(0.02, 0.98, 49)
    q = stats.chi2.ppf(lv, 3)

    def cov_curve(Mhat, P, M):
        r = M - Mhat
        d2 = np.einsum("ni,nij,nj->n", r, P, r)
        return np.array([np.mean(d2 <= qq) for qq in q])
    curves = []
    for setname, ls, lab in (("test_prior", "-", "Network, prior test draws"),
                             ("test_ood_pose", (0, (5, 3)), "Network, pose errors 2x prior")):
        o = dict(np.load(os.path.join(C.DATA, f"{setname}.npz")))
        mu, cov, _, _ = ensemble_predict(full, features(o, dof))
        curves.append((cov_curve(mu, np.linalg.inv(cov), o["M"]), st.DIRECTION, ls, lab))
    o = dict(np.load(os.path.join(C.DATA, "test_prior.npz")))
    curves.append((cov_curve(o["est_LS11"], np.broadcast_to(np.linalg.inv(sv.cov["LS11"]), (len(o["M"]), 3, 3)), o["M"]),
                   st.INTENSITY, "-", "LS formal covariance, prior"))
    o = dict(np.load(os.path.join(C.DATA, "test_none.npz"))); M = o["M"]; X = features(o, dof)
    B = R["bounds"]; crlb = B["crlb_trace"]
    rows = [("LS,\n11 columns", o["est_LS11"], st.INTENSITY, "s")]
    for i, pth in enumerate(none):
        rows.append((f"Network,\nseed {i}", ensemble_predict([pth], X)[0], st.DIRECTION, "o"))
    rows.append(("Network,\nensemble", ensemble_predict(none, X)[0], st.DIRECTION, "o"))
    eff = []
    for lab, Mh, col, mk in rows:
        e2 = np.sum((Mh - M) ** 2, 1)
        eff.append((lab, e2.mean() / crlb, 2 * e2.std() / np.sqrt(len(e2)) / crlb, col, mk))
    return lv, curves, eff, B["bcrb_over_crlb"]


def fig2(R):
    lv, curves, eff, vt = data_calib(R)
    with st.style():
        fig = st.figure("full", 3.0)
        axs = fig.subplots(1, 2, gridspec_kw=dict(left=0.085, right=0.985, bottom=0.21, top=0.92, wspace=0.27))
        ax = axs[0]
        ax.plot([0, 1], [0, 1], color=GREY_D, lw=0.9, ls=(0, (1, 1.6)))
        for y, col, ls, lab in curves:
            ax.plot(lv, y, color=col, lw=1.4, ls=ls, label=lab)
        ax.set_xlabel("Nominal credible level"); ax.set_ylabel("Empirical coverage")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(True); ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=st.BASE_PT - 1.5, handlelength=2.4, frameon=False)
        panel(ax, "a")
        ax = axs[1]
        for i, (lab, v, se, col, mk) in enumerate(eff):
            ax.errorbar(i, v, yerr=se, color=col, marker=mk, ms=4.5, lw=1.1, capsize=3, mew=1.0)
        ax.axhline(1.0, color=st.INK, lw=0.9, label="Cramér–Rao bound (unbiased estimators)")
        ax.axhline(vt, color=GREY_D, lw=0.9, ls=(0, (4, 2)), label="van Trees bound (any estimator, this prior)")
        ax.legend(loc="lower right", fontsize=st.BASE_PT - 1.5)
        ax.set_xticks(range(len(eff))); ax.set_xticklabels([r[0] for r in eff], fontsize=st.BASE_PT - 1.5)
        ax.set_xlim(-0.5, len(eff) - 0.5); ax.set_ylim(0.95, 1.06)
        ax.set_ylabel("MSE / Cramér–Rao bound")
        ax.grid(True, axis="y"); ax.set_axisbelow(True)
        panel(ax, "b")
        save(fig, "ml_fig2_calib")


if __name__ == "__main__":
    R = json.load(open(os.path.join(C.RESULTS, "sbi_results.json")))
    fig1(R)
    needed = [os.path.join(C.DATA, f"{n}.npz") for n in ("test_prior", "test_ood_pose", "test_none")]
    if all(os.path.exists(f) for f in needed):
        fig2(R)
    else:
        # ml/data is large and not tracked by git; the stored calibration figure is left as it is.
        print("ml/data not found: ml_fig2_calib not redrawn. `bash ml/run_all.sh` regenerates the data (about 30 min).")

"""Evaluate the trained networks and the least-squares estimators on every held-out test set.

Estimators
  LS11          thesis estimator: magnetization + 3 offsets + 5 gradient terms (Eq. ls with the augmented matrix)
  LS17          + 6 heading terms (constant tilt)
  LS11_root1m   LS11 + a separately magnetized 1 m root body (thesis 'root fit')
  LS17_root1m   LS17 + the same root body
  profile       profile least squares over the buried extent: G(b) on a 0.05 m grid, 17 nuisance columns,
                keep the b with the smallest residual (a classical nonlinear estimator; pose not modelled)
  NN            deep ensemble of Gaussian-output networks trained on the full nuisance prior (moment-matched)
  NN_s0         the first ensemble member alone
  NN_none       networks trained on the linear-Gaussian (noise-only) simulator, for the efficiency check

Run:  python ml/evaluate.py
Out:  ml/results/sbi_results.json
"""
import os, sys, json, glob, time
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from train import features, load_net, MODELS
from generate import SETS

LEVELS = (0.5, 0.68, 0.9, 0.95)
LS_KEYS = ["LS11", "LS17", "LS11_root1m", "LS17_root1m", "profile"]


def ensemble_predict(paths, X):
    mus, covs = [], []
    for p in paths:
        net, mu_x, sd_x = load_net(p)
        m, c = net.predict(((X - mu_x) / sd_x).astype(np.float32))
        mus.append(m); covs.append(c)
    mus = np.stack(mus); covs = np.stack(covs)
    mu = mus.mean(0)
    cov = (covs + np.einsum("eni,enj->enij", mus, mus)).mean(0) - np.einsum("ni,nj->nij", mu, mu)
    return mu, cov, mus, covs


def coverage(Mhat, cov, M):
    r = M - Mhat
    d2 = np.einsum("ni,nij,nj->n", r, np.linalg.inv(cov), r)
    return {f"{lv:.2f}": float(np.mean(d2 <= stats.chi2.ppf(lv, 3))) for lv in LEVELS}


def cone_coverage(Mhat, cov, M, rng, ns=1024, level=0.95):
    """Predicted direction cone: the `level` quantile of the angle between draws from N(Mhat, cov) and Mhat.
    Returns the fraction of cases whose true direction lies inside, and the median cone half-angle."""
    Lc = np.linalg.cholesky(cov)
    th = np.zeros(len(M))
    for i in range(0, len(M), 500):
        z = rng.normal(size=(ns, 3))                     # fresh draws per block
        s = Mhat[i:i + 500, None, :] + np.einsum("nij,sj->nsi", Lc[i:i + 500], z)
        u = Mhat[i:i + 500] / np.linalg.norm(Mhat[i:i + 500], axis=1, keepdims=True)
        c = np.sum(s * u[:, None, :], 2) / np.linalg.norm(s, axis=2)
        th[i:i + 500] = np.quantile(np.degrees(np.arccos(np.clip(c, -1, 1))), level, axis=1)
    e = C.dir_err(Mhat, M)
    return {"covered": float(np.mean(e <= th)), "median_halfangle_deg": float(np.median(th))}


def metrics(Mhat, M):
    e = C.dir_err(Mhat, M); dm = C.int_err_signed(Mhat, M)
    return {"dir_err_deg": C.summary(e), "int_err_signed_pct": C.summary(dm),
            "int_err_abs_pct_median": float(np.median(np.abs(dm))),
            "mse_Mref2": float(np.mean(np.sum((Mhat - M) ** 2, 1))),
            "mse_sem": float(np.std(np.sum((Mhat - M) ** 2, 1)) / np.sqrt(len(M)))}


def van_trees(sv):
    """Bayesian (van Trees) bound on the M block for the noise-only simulator: data Fisher information of
    [G E Ng] plus the prior information of M (direction uniform, ln|M| ~ N(ln 2, 0.35^2)), of the offsets
    (N(0, 50^2) nT) and of the gradient coefficients (N(0, 1))."""
    s = 0.35; mu0 = np.log(2.0)
    w, wt = np.polynomial.hermite_e.hermegauss(80); wt = wt / wt.sum()
    r = np.exp(mu0 + s * w)
    Jp = float(np.sum(wt * (3.0 + w / s) ** 2 / r ** 2) / 3.0)   # per component, isotropic
    A = np.column_stack([sv.G, sv.E, sv.Ng])
    J = A.T @ A / C.SIGMA_B ** 2 + np.diag([Jp] * 3 + [1 / 50.0 ** 2] * 3 + [1.0] * 5)
    B = np.linalg.inv(J)[:3, :3]
    return {"J_prior_M_per_component": Jp, "bcrb_trace": float(np.trace(B)),
            "crlb_trace": float(np.trace(sv.cov["LS11"])), "bcrb_over_crlb": float(np.trace(B) / np.trace(sv.cov["LS11"]))}


def main():
    sv = C.Survey(0.5); dof = sv.nd - sv.k
    full = sorted(glob.glob(os.path.join(MODELS, "full_s*.npz")))
    none = sorted(glob.glob(os.path.join(MODELS, "none_s*.npz")))
    rng = np.random.default_rng(C.seed_for("ml:evaluate"))
    out = {"meta": {"dx_m": 0.5, "n_stations": sv.n, "n_features": sv.k - sv.n_inv + 1, "models_full": [os.path.basename(p) for p in full],
                    "models_none": [os.path.basename(p) for p in none], "sigma_B_nT": C.SIGMA_B, "M_ref_Am": C.MREF,
                    "units": "magnetization in units of M_ref; mse in M_ref^2"},
           "bounds": van_trees(sv), "sets": {}}
    tests = [s for s in SETS if s.startswith("test_")]
    for name in tests:
        f = os.path.join(C.DATA, f"{name}.npz")
        if not os.path.exists(f):
            print(f"missing {name}"); continue
        o = dict(np.load(f)); M = o["M"]; X = features(o, dof)
        res = {"n": int(len(M))}
        for k in LS_KEYS:
            res[k] = metrics(o[f"est_{k}"], M)
        for k in ("LS11", "LS17"):
            covk = np.broadcast_to(sv.cov[k], (len(M), 3, 3))
            res[k]["coverage"] = coverage(o[f"est_{k}"], covk, M)
            res[k]["cone95"] = cone_coverage(o[f"est_{k}"], np.ascontiguousarray(covk), M, rng)
        mu, cov, mus, covs = ensemble_predict(full, X)
        res["NN"] = metrics(mu, M); res["NN"]["coverage"] = coverage(mu, cov, M); res["NN"]["cone95"] = cone_coverage(mu, cov, M, rng)
        res["NN_members"] = [{"dir_err_mean": float(C.dir_err(m, M).mean()), "dir_err_p95": float(np.percentile(C.dir_err(m, M), 95)),
                              "mse_Mref2": float(np.mean(np.sum((m - M) ** 2, 1))), "coverage": coverage(m, c, M)}
                             for m, c in zip(mus, covs)]
        if name == "test_prior":       # coverage conditional on burial depth, intensity and direction
            r = M - mu; d2 = np.einsum("ni,nij,nj->n", r, np.linalg.inv(cov), r); q95 = stats.chi2.ppf(0.95, 3)
            inc = np.degrees(np.arcsin(np.clip(-M[:, 2] / np.linalg.norm(M, axis=1), -1, 1)))
            groups = {"b_m": (o["b"], [0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                      "intensity_Mref": (np.linalg.norm(M, axis=1), [0, 1.5, 2.0, 2.5, 10]),
                      "abs_inclination_deg": (np.abs(inc), [0, 30, 60, 90.01]),
                      "pose_max_deg": (np.maximum(np.abs(o["az"]), np.abs(o["el"])), [0, 1, 2, 3.01]),
                      "drift_rms_nT": (o["drift_rms"], [0, 0.01, 0.02, 0.0301])}
            cc = {}
            for gname, (v, edges) in groups.items():
                cc[gname] = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    sel = (v >= lo) & (v < hi); k = int(sel.sum()); p = float(np.mean(d2[sel] <= q95))
                    zw = 1.96; den = 1 + zw ** 2 / k; cen = (p + zw ** 2 / (2 * k)) / den
                    hw = zw * np.sqrt(p * (1 - p) / k + zw ** 2 / (4 * k ** 2)) / den
                    cc[gname].append({"lo": lo, "hi": hi, "n": k, "cov95": p, "wilson95": [cen - hw, cen + hw],
                                      "dir_err_mean_NN": float(C.dir_err(mu[sel], M[sel]).mean()),
                                      "dir_err_mean_profile": float(C.dir_err(o["est_profile"][sel], M[sel]).mean())})
            res["NN"]["conditional_coverage95"] = cc
        if none:
            mu0, cov0, _, _ = ensemble_predict(none, X)
            res["NN_none"] = metrics(mu0, M); res["NN_none"]["coverage"] = coverage(mu0, cov0, M)
        if "b" in o:
            bb = o["b"]; res["profile_bhat"] = {"rmse_m": float(np.sqrt(np.mean((o["bhat_profile"] - bb) ** 2))),
                                               "corr": float(np.corrcoef(o["bhat_profile"], bb)[0, 1]) if bb.std() > 0 else None}
        out["sets"][name] = res
        print(f"{name:20s} " + " ".join(f"{k} {res[k]['dir_err_deg']['mean']:.2f}/{res[k]['dir_err_deg']['p95']:.1f}"
                                       for k in ["LS11", "LS17", "LS11_root1m", "profile", "NN"] + (["NN_none"] if none else [])), flush=True)
    # inference cost per survey (features already compressed; the compression itself is one 3n x k product)
    o = dict(np.load(os.path.join(C.DATA, "test_none.npz"))); X = features(o, dof)[:2000]
    net, mu_x, sd_x = load_net(full[0]); t0 = time.time(); net.predict(((X - mu_x) / sd_x).astype(np.float32)); tn = (time.time() - t0) / len(X)
    d = rng.normal(0, 1, sv.nd); t0 = time.time()
    for _ in range(50):
        sv.profile_ls(d)
    tp = (time.time() - t0) / 50
    t0 = time.time()
    for _ in range(50):
        sv.Q.T @ d
    tq = (time.time() - t0) / 50
    nets = [load_net(p) for p in full]
    t0 = time.time()
    for _ in range(50):
        z = sv.Q.T @ d; rr = (d @ d - z @ z) / C.SIGMA_B ** 2
        x = np.concatenate([z[sv.n_inv:], [np.log(rr / dof)]])[None]
        for net_, mx, sx_ in nets:
            net_.predict(((x - mx) / sx_).astype(np.float32))
    te = (time.time() - t0) / 50
    out["timing_s_per_survey"] = {"nn_forward_batched": tn, "compression_QTd": tq, "profile_ls_21_grid": tp,
                                  "ensemble_single_survey_end_to_end": te}
    json.dump(out, open(os.path.join(C.RESULTS, "sbi_results.json"), "w"), indent=1)
    print("bounds", out["bounds"]); print("timing", out["timing_s_per_survey"])


if __name__ == "__main__":
    main()

"""Along-track noise correlation (paper Section 5.1); run in the paper geometry by analysis/a05_background_and_correlated_noise.py.

This run sweeps the correlation length L on a fine grid through
the short-correlation regime and states the maximum tolerable L in cm and, at 1 cm/s, in seconds.

Part 1: the inversion assumes i.i.d. noise (cold-start weighted least squares) while the data are
correlated; isolated drive-around and drive-by (K = 1) and the three-layer wall (K = 3).
Part 2: noise-aware inversion by generalized least squares, whitening data and operator with the
true station covariance C = sigma_w^2 I + sigma_c^2 exp(-|s_i - s_j| / L), on matched noise draws.

The noise generator is engine/correlated_noise_sensitivity.py.

Run:  python analysis/a05_background_and_correlated_noise.py noise
"""
import json
import os
import sys
import time

import numpy as np
import scipy
import scipy.linalg as sla
import discretize
import simpeg

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import inversion_helpers as H
from inversion_helpers import (prism_cell_operator, prism_bounds, ls_operator,
                               recovery_metrics, seed_for, outcrop_survey,
                               survey_geom_dict, layer_slices, three_layer_mesh,
                               M_NRM_MIN, STANDOFF, SENSOR_H, PRISM_DIMS,
                               MASTER_SEED, ACTIVE_SEED)
# noise model from correlated_noise_sensitivity.py
from correlated_noise_sensitivity import (make_noise, effective_sample_count,
                                          along_track_ds, SIGMA_TOT, SIGMA_W, SIGMA_C)
from core import generate_rille_traverse

RESULTS_PATH = os.path.join(os.path.dirname(_REPO), "results", "engine", "correlated_noise_requirement_results.json")  # a05 writes to results/analysis instead

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
SPACING = 0.01
N_MC = 500                 # naive sweep
N_GLS = 200                # noise-aware arm (matched draws vs naive)
TRAVERSE_SPEED_MPS = 0.01  # 1 cm/s, for the cm -> seconds conversion

# Fine grid through the short-correlation regime, plus coarse anchors for continuity
# with correlated_noise_sensitivity.py.
L_FINE_CM = [0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0]
L_COARSE_M = [1.0, 2.0]
L_SWEEP = ([("iid_ref", None)]
           + [(f"{c:g}cm", c / 100.0) for c in L_FINE_CM]
           + [(f"{m:g}m", m) for m in L_COARSE_M]
           + [("whole", np.inf)])

L_GLS_ISO_CM = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
L_GLS_WALL_CM = [10.0, 50.0]

DRIVE = {"drive_around": "full_orbit", "drive_by": "drive_by"}
HEADLINE_SURVEY = "drive_around"
THRESHOLDS = [5.0, 10.0]

# Three-layer scenario (ground traverse of engine/three_layer_wall.py)
LAYER_NAMES = ["bottom(nearest)", "middle", "top"]
M_LAYERS = M_NRM_MIN * np.array([[0., 0., -1.], [0., 1., 0.], [0., 0., 1.]])
M_TRUE_9 = np.concatenate([M_LAYERS[:, 0], M_LAYERS[:, 1], M_LAYERS[:, 2]])
WALL_Y, WALL_Z = -1.0, 0.5
WALL_X0, WALL_X1 = -120.0, 120.0

m_true_iso = M_NRM_MIN * np.array([0., 0., -1.])
bnd = prism_bounds((0., 0., 0.), PRISM_DIMS)

SEEDS_USED = {}


def record_seed(label):
    s = int(seed_for(label))
    SEEDS_USED[label] = s
    return s


# --------------------------------------------------------------------------- #
# Naive (i.i.d.-assuming) inversion, identical to correlated_noise_sensitivity.recover
# --------------------------------------------------------------------------- #
def recover_naive(G, m_true_full, noise):
    K = ls_operator(G)
    d_clean = G @ np.asarray(m_true_full, float)
    D = d_clean[None, :] + noise
    M_rec = D @ K.T
    resid_rms = np.sqrt(np.mean(((M_rec @ G.T - D) / SIGMA_TOT) ** 2, axis=1))
    return M_rec, resid_rms


# --------------------------------------------------------------------------- #
# Noise-aware (GLS) inversion via exact whitening
# --------------------------------------------------------------------------- #
def station_arclength(sxyz):
    """Cumulative along-track distance used by the imported OU generator."""
    return np.r_[0.0, np.cumsum(along_track_ds(sxyz))]


def ou_covariance(sxyz, L):
    """Per-axis station covariance of the imported noise model: sigma_w^2 I + sigma_c^2 R."""
    s = station_arclength(sxyz)
    nR = len(s)
    if np.isinf(L):
        R = np.ones((nR, nR))
    else:
        R = np.exp(-np.abs(s[:, None] - s[None, :]) / L)
    C = SIGMA_C ** 2 * R
    C[np.diag_indices_from(C)] += SIGMA_W ** 2
    return C


def whitener(sxyz, L):
    """Lower Cholesky factor Lc of the per-axis covariance (in place, memory-conscious)."""
    C = ou_covariance(sxyz, L)
    Lc = sla.cholesky(C, lower=True, overwrite_a=True, check_finite=False)
    del C
    return Lc


def whiten_stationmajor(Lc, A, nR, n_axes=3):
    """Apply Lc^-1 per axis to a station-major array A of shape (nR*n_axes, k)."""
    A3 = np.asarray(A, float).reshape(nR, n_axes, -1)
    out = np.empty_like(A3)
    for a in range(n_axes):
        out[:, a, :] = sla.solve_triangular(Lc, A3[:, a, :], lower=True,
                                            check_finite=False)
    return out.reshape(nR * n_axes, -1)


def recover_gls(G, m_true_full, noise, sxyz, L):
    """Exact GLS recovery using the true non-diagonal covariance (whitening transform).

    Returns (M_rec, resid_rms_whitened, resid_rms_iid_weighted).
    """
    nR = len(sxyz)
    Lc = whitener(sxyz, L)
    G_w = whiten_stationmajor(Lc, G, nR)                       # (nD, p)
    d_clean = G @ np.asarray(m_true_full, float)
    D = d_clean[None, :] + noise                               # (n, nD)
    D_w = whiten_stationmajor(Lc, D.T, nR).T                   # (n, nD)
    del Lc
    K_w = ls_operator(G_w)
    M_rec = D_w @ K_w.T
    resid_w = M_rec @ G_w.T - D_w
    rms_w = np.sqrt(np.mean(resid_w ** 2, axis=1))
    resid_i = M_rec @ G.T - D
    rms_i = np.sqrt(np.mean((resid_i / SIGMA_TOT) ** 2, axis=1))
    return M_rec, rms_w, rms_i


def estimator_covariance_comparison(G, sxyz, L, m_true_full):
    """EXACT (Monte-Carlo-free) covariances of the naive OLS and noise-aware GLS estimators.

    Both estimators are unbiased, so the entire difference in recovery accuracy is the
    difference between
        Cov_OLS = (G'G)^-1 G' C G (G'G)^-1        (i.i.d.-assuming weights, true C)
        Cov_GLS = (G' C^-1 G)^-1                  (Gauss-Markov optimum)
    Gauss-Markov guarantees Cov_OLS >= Cov_GLS; this quantifies BY HOW MUCH, independently
    of the Monte Carlo, and also reports how different the two estimator operators are, so
    "GLS does not help" cannot be confused with "GLS was not actually applied".
    """
    nR = len(sxyz)
    p = G.shape[1]
    C = ou_covariance(sxyz, L)
    Lc = sla.cholesky(C, lower=True, check_finite=False)
    G3 = G.reshape(nR, 3, p)
    Gw = np.empty_like(G3)
    CG = np.empty_like(G3)
    for a in range(3):
        Gw[:, a, :] = sla.solve_triangular(Lc, G3[:, a, :], lower=True, check_finite=False)
        CG[:, a, :] = C @ G3[:, a, :]
    Gw = Gw.reshape(nR * 3, p)
    CG = CG.reshape(nR * 3, p)
    GtG_inv = np.linalg.inv(G.T @ G)
    cov_ols = GtG_inv @ (G.T @ CG) @ GtG_inv
    cov_gls = np.linalg.inv(Gw.T @ Gw)
    # effective operator of the GLS estimator acting on RAW data, for an honest
    # "the estimators really are different" check
    K_ols = ls_operator(G)
    K_g = np.linalg.solve(Gw.T @ Gw, Gw.T).reshape(p, nR, 3)
    Kg_raw = np.empty((p, nR, 3))
    for a in range(3):
        Kg_raw[:, :, a] = sla.solve_triangular(Lc.T, K_g[:, :, a].T, lower=False,
                                               check_finite=False).T
    Kg_raw = Kg_raw.reshape(p, nR * 3)
    del C, Lc
    tmag = np.linalg.norm(np.asarray(m_true_full, float))
    return {
        "correlation_length_m": float(L),
        "trace_cov_ols": float(np.trace(cov_ols)),
        "trace_cov_gls": float(np.trace(cov_gls)),
        "trace_ratio_ols_over_gls": float(np.trace(cov_ols) / np.trace(cov_gls)),
        "eigenvalues_cov_ols": [float(v) for v in np.linalg.eigvalsh(cov_ols)],
        "eigenvalues_cov_gls": [float(v) for v in np.linalg.eigvalsh(cov_gls)],
        "per_component_std_over_Mtrue_ols":
            [float(v) for v in np.sqrt(np.diag(cov_ols)) / tmag],
        "per_component_std_over_Mtrue_gls":
            [float(v) for v in np.sqrt(np.diag(cov_gls)) / tmag],
        "operator_relative_difference_frobenius":
            float(np.linalg.norm(Kg_raw - K_ols) / np.linalg.norm(K_ols)),
        "max_possible_variance_reduction_pct":
            float(100.0 * (1.0 - np.trace(cov_gls) / np.trace(cov_ols))),
    }


def verify_covariance_model(n_draws=20000, nR=300):
    """Empirical covariance of the imported generator vs the analytic model used for GLS."""
    sxyz = generate_rille_traverse(0.0, (nR - 1) * SPACING, -1.0, 0.5, SPACING)
    out = {}
    for L in (0.05, 0.20):
        rng = np.random.default_rng(record_seed(f"runC_covcheck_L{L}"))
        n = make_noise(sxyz, L, n_draws, rng).reshape(n_draws, len(sxyz), 3)
        emp = np.cov(n[:, :, 0], rowvar=False)
        ana = ou_covariance(sxyz, L)
        rel = float(np.max(np.abs(emp - ana)) / ana[0, 0])
        out[f"L_{L}m"] = {
            "n_draws": n_draws, "n_stations": int(len(sxyz)),
            "analytic_var": float(ana[0, 0]), "empirical_var": float(emp[0, 0]),
            "max_abs_deviation_over_var": rel,
            "analytic_lag1_corr": float(ana[0, 1] / ana[0, 0]),
            "empirical_lag1_corr": float(emp[0, 1] / emp[0, 0]),
            "agrees_within_5pct": bool(rel < 0.05)}
        print(f"    L={L} m: var analytic {ana[0,0]:.6f} vs empirical {emp[0,0]:.6f} | "
              f"lag-1 corr {ana[0,1]/ana[0,0]:.5f} vs {emp[0,1]/emp[0,0]:.5f} | "
              f"max|dev|/var={rel*100:.2f}% -> "
              f"{'OK' if out[f'L_{L}m']['agrees_within_5pct'] else 'MISMATCH'}")
    return out


# --------------------------------------------------------------------------- #
# Threshold helpers
# --------------------------------------------------------------------------- #
def crossing(xs, ys, level):
    """Largest x at which piecewise-linear y(x) is still <= level (ys INCREASING in x)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    if ys[0] > level:
        return {"x": None, "status": "exceeded_at_smallest_L_sampled"}
    for i in range(1, len(xs)):
        if ys[i] > level:
            f = (level - ys[i - 1]) / (ys[i] - ys[i - 1])
            return {"x": float(xs[i - 1] + f * (xs[i] - xs[i - 1])),
                    "status": "interpolated"}
    return {"x": float(xs[-1]), "status": "not_exceeded_over_sampled_range"}


def requirement_block(L_m, err, level):
    c = crossing(L_m, err, level)
    if c["x"] is None:
        return {"L_max_m": None, "L_max_cm": None, "tau_max_s": None,
                "status": c["status"]}
    return {"L_max_m": c["x"], "L_max_cm": c["x"] * 100.0,
            "tau_max_s": c["x"] / TRAVERSE_SPEED_MPS,
            "status": c["status"]}


def fmt_req(r):
    if r["L_max_cm"] is None:
        return f"none ({r['status']})"
    return f"L <= {r['L_max_cm']:.2f} cm (tau <= {r['tau_max_s']:.2f} s) [{r['status']}]"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    t_start = time.time()
    print(f"simpeg {simpeg.__version__} | discretize {discretize.__version__} | "
          f"numpy {np.__version__}")
    print(f"MASTER_SEED={MASTER_SEED} ACTIVE_SEED={ACTIVE_SEED} | N_MC={N_MC} "
          f"N_GLS={N_GLS} | dx={SPACING} m")
    print(f"noise model (imported from correlated_noise_sensitivity.py): "
          f"sigma_total={SIGMA_TOT} nT, sigma_white={SIGMA_W} nT, "
          f"sigma_OU={SIGMA_C:.4f} nT")

    print("\n=== VALIDATION: analytic GLS covariance vs imported generator ===")
    covchk = verify_covariance_model()

    results = {"isolated_outcrop": {}, "three_layer": {}, "effective_samples": {},
               "noise_aware_gls": {}, "headline": {}}

    # ---- PART 1a: isolated outcrop, naive inversion ----
    print("\n=== PART 1a: isolated outcrop (K=1), naive i.i.d.-assuming inversion ===")
    iso_surveys = {}
    for name, ttype in DRIVE.items():
        sxyz = outcrop_survey(ttype, STANDOFF, spacing=SPACING)
        iso_surveys[name] = sxyz
        G = prism_cell_operator(bnd, sxyz)
        sig_rms = float(np.sqrt(np.mean((G @ m_true_iso) ** 2)))
        results["isolated_outcrop"][name] = {}
        results["effective_samples"].setdefault(name, {})
        print(f"-- {name} ({len(sxyz)} stations, nD={3*len(sxyz)}, "
              f"signal rms={sig_rms:.4f} nT) --")
        for lname, L in L_SWEEP:
            rng = np.random.default_rng(record_seed(f"runC_iso_{name}_{lname}"))
            noise = make_noise(sxyz, L, N_MC, rng)
            M_rec, rms = recover_naive(G, m_true_iso, noise)
            blk = recovery_metrics(M_rec, m_true_iso, rms, SIGMA_TOT,
                                   survey_geom_dict(sxyz, {"type": ttype}))
            neff = effective_sample_count(sxyz, L)
            blk["correlation_length_label"] = lname
            blk["correlation_length_m"] = None if L is None else (
                None if np.isinf(L) else float(L))
            blk["correlation_length_is_inf"] = bool(L is not None and np.isinf(L))
            blk["tau_s_at_1cm_per_s"] = (None if L is None or np.isinf(L)
                                         else float(L / TRAVERSE_SPEED_MPS))
            blk["N_eff_per_axis"] = neff
            blk["signal_rms_nT"] = sig_rms
            results["isolated_outcrop"][name][lname] = blk
            results["effective_samples"][name][lname] = neff
            de, me = blk["dir_err_deg"], blk["mag_err_pct"]
            print(f"   L={lname:8s} rho1={neff['rho_lag1']:.4f} N_eff={neff['N_eff']:8.1f} | "
                  f"theta mean={de['mean']:6.2f} med={de['median']:6.2f} deg | "
                  f"dM mean={me['mean']:+8.2f}% | rms={rms.mean():.4f}", flush=True)
            del noise

    # ---- PART 1b: three-layer wall, naive inversion ----
    print("\n=== PART 1b: three-layer wall (K=3), baseline traverse, naive inversion ===")
    mesh = three_layer_mesh()
    sxyz3 = generate_rille_traverse(WALL_X0, WALL_X1, WALL_Y, WALL_Z, SPACING)
    sim3 = H._sim_for(mesh, sxyz3, mesh.n_cells)
    G3 = sim3.G.copy()
    sig3 = [float(np.sqrt(np.mean((G3 @ np.where(
        np.isin(np.arange(9), [i, 3 + i, 6 + i]), M_TRUE_9, 0.0)) ** 2))) for i in range(3)]
    print(f"-- {len(sxyz3)} stations, nD={3*len(sxyz3)}, per-flow signal rms="
          f"{[round(v,4) for v in sig3]} nT --")
    results["effective_samples"]["three_layer"] = {}
    for lname, L in L_SWEEP:
        rng = np.random.default_rng(record_seed(f"runC_wall_{lname}"))
        noise = make_noise(sxyz3, L, N_MC, rng)
        M_rec, rms = recover_naive(G3, M_TRUE_9, noise)
        neff = effective_sample_count(sxyz3, L)
        per_layer = {}
        for i, Mi in enumerate(layer_slices(M_rec)):
            per_layer[f"layer_{i+1}_{LAYER_NAMES[i]}"] = recovery_metrics(
                Mi, M_LAYERS[i], rms, SIGMA_TOT, {"type": "rille_traverse"})
            per_layer[f"layer_{i+1}_{LAYER_NAMES[i]}"]["signal_rms_nT"] = sig3[i]
        results["three_layer"][lname] = {
            "correlation_length_label": lname,
            "correlation_length_m": None if L is None else (
                None if np.isinf(L) else float(L)),
            "correlation_length_is_inf": bool(L is not None and np.isinf(L)),
            "tau_s_at_1cm_per_s": (None if L is None or np.isinf(L)
                                   else float(L / TRAVERSE_SPEED_MPS)),
            "N_eff_per_axis": neff, "per_layer": per_layer}
        results["effective_samples"]["three_layer"][lname] = neff
        b = per_layer[f"layer_1_{LAYER_NAMES[0]}"]
        de, me = b["dir_err_deg"], b["mag_err_pct"]
        print(f"   L={lname:8s} rho1={neff['rho_lag1']:.4f} N_eff={neff['N_eff']:8.1f} | "
              f"nearest flow theta mean={de['mean']:6.2f} med={de['median']:6.2f} deg | "
              f"dM mean={me['mean']:+8.2f}% | rms={rms.mean():.4f}", flush=True)
        del noise

    # ---- headline requirement ----
    print("\n=== HEADLINE REQUIREMENT ===")
    head = {}
    for name in DRIVE:
        blocks = results["isolated_outcrop"][name]
        finite = [(blocks[l]["correlation_length_m"], blocks[l]["dir_err_deg"]["mean"],
                   blocks[l]["dir_err_deg"]["median"])
                  for l, _ in L_SWEEP if blocks[l]["correlation_length_m"] is not None]
        Ls = [f[0] for f in finite]
        head[name] = {}
        for lvl in THRESHOLDS:
            head[name][f"dir_err_mean_under_{lvl:g}deg"] = requirement_block(
                Ls, [f[1] for f in finite], lvl)
            head[name][f"dir_err_median_under_{lvl:g}deg"] = requirement_block(
                Ls, [f[2] for f in finite], lvl)
        if name == HEADLINE_SURVEY:
            for lvl in THRESHOLDS:
                r = head[name][f"dir_err_mean_under_{lvl:g}deg"]
                print(f"  isolated {name}: direction error (ensemble MEAN) stays under "
                      f"{lvl:g} deg for {fmt_req(r)}")
                rm = head[name][f"dir_err_median_under_{lvl:g}deg"]
                print(f"  isolated {name}: direction error (MEDIAN)        stays under "
                      f"{lvl:g} deg for {fmt_req(rm)}")
    wall_finite = [(results["three_layer"][l]["correlation_length_m"],
                    results["three_layer"][l]["per_layer"][f"layer_1_{LAYER_NAMES[0]}"]
                    ["dir_err_deg"]["mean"])
                   for l, _ in L_SWEEP
                   if results["three_layer"][l]["correlation_length_m"] is not None]
    head["three_layer_nearest_flow"] = {
        f"dir_err_mean_under_{lvl:g}deg": requirement_block(
            [w[0] for w in wall_finite], [w[1] for w in wall_finite], lvl)
        for lvl in THRESHOLDS}
    for lvl in THRESHOLDS:
        r = head["three_layer_nearest_flow"][f"dir_err_mean_under_{lvl:g}deg"]
        print(f"  three-layer nearest flow: direction error stays under {lvl:g} deg for "
              f"{fmt_req(r)}")
    results["headline"] = head

    # ---- PART 2: noise-aware (GLS) inversion, matched draws ----
    print(f"\n=== PART 2: noise-aware GLS inversion (exact whitening), N={N_GLS} ===")
    gls = {"isolated_drive_around": {}, "three_layer": {}}
    sxyz_da = iso_surveys[HEADLINE_SURVEY]
    G_da = prism_cell_operator(bnd, sxyz_da)
    print(f"-- isolated {HEADLINE_SURVEY} ({len(sxyz_da)} stations) --")
    for c in L_GLS_ISO_CM:
        L = c / 100.0
        lname = f"{c:g}cm"
        t0 = time.time()
        rng = np.random.default_rng(record_seed(f"runC_gls_iso_{lname}"))
        noise = make_noise(sxyz_da, L, N_GLS, rng)
        M_n, rms_n = recover_naive(G_da, m_true_iso, noise)
        M_g, rms_gw, rms_gi = recover_gls(G_da, m_true_iso, noise, sxyz_da, L)
        bn = recovery_metrics(M_n, m_true_iso, rms_n, SIGMA_TOT, {"type": "full_orbit"})
        bg = recovery_metrics(M_g, m_true_iso, rms_gw, SIGMA_TOT, {"type": "full_orbit"})
        bg["residual_rms_iid_weighted"] = {"mean": float(rms_gi.mean()),
                                           "std": float(rms_gi.std())}
        recov = (100.0 * (bn["dir_err_deg"]["mean"] - bg["dir_err_deg"]["mean"])
                 / max(bn["dir_err_deg"]["mean"] - 0.0, 1e-12))
        gls["isolated_drive_around"][lname] = {
            "correlation_length_m": L, "tau_s_at_1cm_per_s": L / TRAVERSE_SPEED_MPS,
            "N_mc": N_GLS, "matched_noise_draws": True,
            "naive_iid_assuming": bn, "noise_aware_gls": bg,
            "dir_err_reduction_pct_of_naive": float(recov)}
        print(f"   L={lname:6s} naive theta={bn['dir_err_deg']['mean']:6.2f} deg "
              f"(rms {rms_n.mean():.4f}) -> GLS theta={bg['dir_err_deg']['mean']:6.2f} deg "
              f"(whitened rms {rms_gw.mean():.4f}, iid-weighted rms {rms_gi.mean():.4f}) | "
              f"recovered {recov:5.1f}% of the naive error | dM "
              f"{bn['mag_err_pct']['mean']:+7.2f}% -> {bg['mag_err_pct']['mean']:+7.2f}% "
              f"| {time.time()-t0:.1f}s", flush=True)
        del noise

    print(f"\n-- EXACT estimator-covariance comparison, isolated {HEADLINE_SURVEY} "
          f"(no Monte Carlo) --")
    covcmp = {}
    for c in L_GLS_ISO_CM:
        e = estimator_covariance_comparison(G_da, sxyz_da, c / 100.0, m_true_iso)
        covcmp[f"{c:g}cm"] = e
        print(f"   L={c:g}cm: trace(cov) OLS {e['trace_cov_ols']:.6e} vs GLS "
              f"{e['trace_cov_gls']:.6e} | best possible variance reduction "
              f"{e['max_possible_variance_reduction_pct']:6.3f}% | estimator operators "
              f"differ by {e['operator_relative_difference_frobenius']*100:5.1f}% (Frobenius)")
    gls["estimator_covariance_comparison_isolated_drive_around"] = covcmp

    print(f"\n-- three-layer wall ({len(sxyz3)} stations; dense "
          f"{len(sxyz3)}x{len(sxyz3)} covariance = "
          f"{len(sxyz3)**2*8/1e9:.2f} GB per Cholesky) --")
    for c in L_GLS_WALL_CM:
        L = c / 100.0
        lname = f"{c:g}cm"
        t0 = time.time()
        rng = np.random.default_rng(record_seed(f"runC_gls_wall_{lname}"))
        noise = make_noise(sxyz3, L, N_GLS, rng)
        M_n, rms_n = recover_naive(G3, M_TRUE_9, noise)
        M_g, rms_gw, rms_gi = recover_gls(G3, M_TRUE_9, noise, sxyz3, L)
        entry = {"correlation_length_m": L, "tau_s_at_1cm_per_s": L / TRAVERSE_SPEED_MPS,
                 "N_mc": N_GLS, "matched_noise_draws": True,
                 "naive_iid_assuming": {}, "noise_aware_gls": {}}
        for i, (Mi_n, Mi_g) in enumerate(zip(layer_slices(M_n), layer_slices(M_g))):
            kn = f"layer_{i+1}_{LAYER_NAMES[i]}"
            entry["naive_iid_assuming"][kn] = recovery_metrics(
                Mi_n, M_LAYERS[i], rms_n, SIGMA_TOT, {"type": "rille_traverse"})
            entry["noise_aware_gls"][kn] = recovery_metrics(
                Mi_g, M_LAYERS[i], rms_gw, SIGMA_TOT, {"type": "rille_traverse"})
        entry["noise_aware_gls"]["residual_rms_iid_weighted"] = {
            "mean": float(rms_gi.mean()), "std": float(rms_gi.std())}
        k1 = f"layer_1_{LAYER_NAMES[0]}"
        en = entry["naive_iid_assuming"][k1]["dir_err_deg"]["mean"]
        eg = entry["noise_aware_gls"][k1]["dir_err_deg"]["mean"]
        entry["nearest_flow_dir_err_reduction_pct_of_naive"] = float(
            100.0 * (en - eg) / max(en, 1e-12))
        gls["three_layer"][lname] = entry
        print(f"   L={lname:6s} nearest flow naive theta={en:6.2f} deg "
              f"(rms {rms_n.mean():.4f}) -> GLS theta={eg:6.2f} deg "
              f"(whitened rms {rms_gw.mean():.4f}) | recovered "
              f"{entry['nearest_flow_dir_err_reduction_pct_of_naive']:5.1f}% "
              f"| {time.time()-t0:.1f}s", flush=True)
        del noise
    results["noise_aware_gls"] = gls

    # requirement with the noise-aware (GLS) inversion, on the GLS grid
    gl = gls["isolated_drive_around"]
    Lg = [gl[f"{c:g}cm"]["correlation_length_m"] for c in L_GLS_ISO_CM]
    Eg = [gl[f"{c:g}cm"]["noise_aware_gls"]["dir_err_deg"]["mean"] for c in L_GLS_ISO_CM]
    results["headline"]["isolated_drive_around_with_noise_aware_gls"] = {
        f"dir_err_mean_under_{lvl:g}deg": requirement_block(Lg, Eg, lvl)
        for lvl in THRESHOLDS}
    print("\n  requirement with the noise-aware (GLS) inversion, isolated drive-around:")
    for lvl in THRESHOLDS:
        r = results["headline"]["isolated_drive_around_with_noise_aware_gls"][
            f"dir_err_mean_under_{lvl:g}deg"]
        print(f"      direction error stays under {lvl:g} deg for {fmt_req(r)}")

    payload = {
        "meta": {
            "script": "scripts/correlated_noise_requirement.py",
            "run": "correlated noise as a requirement",
            "question": ("maximum along-track noise correlation length at which isolated "
                         "drive-around direction error stays under 5 deg and under 10 deg, "
                         "expressed in cm and in seconds at 1 cm/s"),
            "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
            "library_versions": {"simpeg": simpeg.__version__,
                                 "discretize": discretize.__version__,
                                 "numpy": np.__version__, "scipy": scipy.__version__},
            "master_seed": MASTER_SEED, "active_seed": ACTIVE_SEED,
            "seed_scheme": "seed_for(name) = sha256(f'{ACTIVE_SEED}:{name}')[:8] (hex->int)",
            "seeds_resolved": SEEDS_USED,
            "N_mc_naive": N_MC, "N_mc_gls": N_GLS,
            "sampling_m": SPACING,
            "traverse_speed_m_per_s": TRAVERSE_SPEED_MPS,
            "noise_model": {
                "source": ("imported from correlated_noise_sensitivity.py (make_noise, "
                           "effective_sample_count, along_track_ds, SIGMA_TOT/W/C); that "
                           "file was read and imported, never modified"),
                "description": ("n = white(sigma_w) + stationary OU/AR(1) along-track drift "
                                "(sigma_c, rho(ds) = exp(-ds/L)); total per-axis variance "
                                "fixed at (0.50 nT)^2 for every L"),
                "sigma_total_nT": SIGMA_TOT, "sigma_white_nT": SIGMA_W,
                "sigma_OU_nT": SIGMA_C,
                "drift_axes": "per-axis OU (each field component drifts coherently)",
                "iid_ref": "pure white 0.50 nT, no correlated component (anchor)",
                "whole": "L -> inf: per-survey per-axis static offset",
            },
            "L_sweep_labels": [x[0] for x in L_SWEEP],
            "L_fine_grid_cm": L_FINE_CM,
            "inversion": {
                "part_1_naive": ("cold-start zero-reference weighted least squares, "
                                 "K = ls_operator(G); assumes i.i.d. noise internally even "
                                 "though the data are correlated. Magnetization components only, geometry fixed "
                                 "and exactly known. K=1 (3 free params) for the isolated "
                                 "outcrop, K=3 (9 free params) for the wall."),
                "part_2_noise_aware": ("exact generalized least squares via a whitening "
                                       "transform applied to BOTH data and forward operator: "
                                       "C = sigma_w^2 I + sigma_c^2 exp(-|s_i-s_j|/L) per "
                                       "axis, C = Lc Lc^T, solve the Lc^-1-whitened system, "
                                       "giving (G^T C^-1 G)^-1 G^T C^-1 d. Matched noise "
                                       "draws against the naive arm."),
                "beta_selection": "none (closed-form least squares; no regularization, no beta)",
                "stopping_criterion": ("none (direct linear solve); the normalized residual "
                                       "rms is reported instead"),
                "residual_rms_conventions": {
                    "naive": "sqrt(mean((r/0.50 nT)^2)) - i.i.d.-assumed weighting",
                    "gls_whitened": ("sqrt(mean(r_white^2)) = sqrt(chi^2/nD) under the TRUE "
                                     "covariance"),
                    "gls_iid_weighted": ("the GLS solution's residual re-scored with the "
                                         "i.i.d. weighting, for comparability with the "
                                         "naive column"),
                },
                "part_2_exact_check": ("both estimators are unbiased, so the accuracy gap is "
                                       "entirely Cov_OLS = (G'G)^-1 G' C G (G'G)^-1 versus "
                                       "Cov_GLS = (G' C^-1 G)^-1. These are computed exactly "
                                       "(no Monte Carlo) alongside the relative Frobenius "
                                       "difference of the two estimator operators, so a null "
                                       "result cannot be confused with the whitening not "
                                       "having been applied"),
            },
            "isolated_scenario": {
                "dims_m": list(PRISM_DIMS), "M_true_Apm": m_true_iso.tolist(),
                "M_NRM_min_Apm": M_NRM_MIN, "standoff_m": STANDOFF,
                "sensor_height_m": SENSOR_H,
                "surveys": {k: {"type": v, "n_stations": int(len(iso_surveys[k])),
                                "n_data": int(3 * len(iso_surveys[k])),
                                "path_length_m": float(along_track_ds(iso_surveys[k]).sum()),
                                "x_range_m": [float(iso_surveys[k][:, 0].min()),
                                              float(iso_surveys[k][:, 0].max())]}
                            for k, v in DRIVE.items()},
            },
            "three_layer_scenario": {
                "layer_names": LAYER_NAMES, "M_layers_Apm": M_LAYERS.tolist(),
                "traverse_y_m": WALL_Y, "traverse_z_m": WALL_Z,
                "traverse_x_range_m": [WALL_X0, WALL_X1], "spacing_m": SPACING,
                "n_stations": int(len(sxyz3)), "n_data": int(3 * len(sxyz3)),
                "per_flow_signal_rms_nT": sig3,
            },
            "dM_convention": "mag_err_pct = 100*(|M_rec|-|M_true|)/|M_true| (SIGNED)",
            "threshold_definition": ("requirement is the largest L at which the ENSEMBLE-MEAN "
                                     "direction error is still at or below the threshold "
                                     "(median also reported for the isolated surveys)"),
        },
        "covariance_model_validation": covchk,
        "isolated_outcrop": results["isolated_outcrop"],
        "three_layer": results["three_layer"],
        "effective_samples": results["effective_samples"],
        "noise_aware_gls": results["noise_aware_gls"],
        "headline": results["headline"],
        "wall_clock_total_s": time.time() - t_start,
    }
    os.makedirs(os.path.join(_REPO, "results"), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {RESULTS_PATH}")
    print(f"total runtime {time.time()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

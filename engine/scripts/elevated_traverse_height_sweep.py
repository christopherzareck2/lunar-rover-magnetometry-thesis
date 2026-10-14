"""Second-traverse height sweep for the three-layer wall (paper Section 4.4, Figure 7).

Sweeps the height z2 of the second traverse (the first stays at z = 0.5 m; same y = -1 m standoff,
x = -120..120 m, 1 cm sampling) and reports the smallest height at which all three flows recover
to within 5 deg, and separately 10 deg, together with the joint operator's condition number.

Geometry and solvers are those of engine/three_layer_wall.py (iterative ProjectedGNCG with
zero-reference damping, and the undamped closed-form least squares). One matched noise matrix is
reused for the fixed low traverse at every height so that heights differ only in geometry.

Run:  python engine/scripts/elevated_traverse_height_sweep.py
Out:  results/engine/elevated_traverse_height_sweep_results.json
"""
import contextlib as _ctx
import io as _io
import json
import os
import sys
import time

import numpy as np
import discretize
import simpeg
from simpeg import (data, data_misfit, directives, inverse_problem, inversion,
                    maps, optimization, regularization)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import inversion_helpers as H
from inversion_helpers import (seed_for, layer_slices, recovery_metrics, survey_geom_dict,
                               ls_operator, three_layer_mesh, operator_diagnostics,
                               M_NRM_MIN, SIGMA_B, MASTER_SEED, ACTIVE_SEED,
                               DEFAULT_GNCG_OPT, DEFAULT_BETA0_RATIO)
from core import generate_rille_traverse

RESULTS_PATH = os.path.join(os.path.dirname(_REPO), "results", "engine",
                            "elevated_traverse_height_sweep_results.json")

# --------------------------------------------------------------------------- #
# Scenario (as engine/three_layer_wall.py)
# --------------------------------------------------------------------------- #
LAYER_Z = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]
LAYER_NAMES = ["bottom/oldest", "middle", "top/youngest"]
M_LAYERS = M_NRM_MIN * np.array([[0., 0., -1.], [0., 1., 0.], [0., 0., 1.]])
M_TRUE_9 = np.concatenate([M_LAYERS[:, 0], M_LAYERS[:, 1], M_LAYERS[:, 2]])

BASE_Y, BASE_Z = -1.0, 0.5
DX = 0.01
X0, X1 = -120.0, 120.0

# Second-traverse heights (m). 0.5 = no vertical separation (degenerate control).
Z2_SWEEP = [0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 12.5, 15.0]
MAST_LIMIT_M = 3.0        # above this a second traverse means driving up the slope

N_MC = 500
RMS_TOL = 0.05
THRESHOLDS = [5.0, 10.0]

# Reference values from engine/three_layer_wall.py (iterative solver, N = 1000), for comparison.
ANCHORS = {
    "single_base_traverse": {"cond": 238.6, "max_corr": 0.997,
                             "dir_deg": [2.59, 24.19, 33.99]},
    "pair_z12.5": {"cond": 20.98, "max_corr": 0.8696, "dir_deg": [2.01, 4.43, 2.02]},
}

SEEDS_USED = {}


def record_seed(label):
    s = int(seed_for(label))
    SEEDS_USED[label] = s
    return s


# --------------------------------------------------------------------------- #
# Forward
# --------------------------------------------------------------------------- #
def wall_sim(rx_locs):
    mesh = three_layer_mesh()
    return H._sim_for(mesh, rx_locs, mesh.n_cells)


def traverse(z, y=BASE_Y):
    return generate_rille_traverse(X0, X1, y, z, DX)


def per_layer_signal(G):
    out = []
    for i in range(3):
        m9 = np.zeros(9)
        m9[[i, 3 + i, 6 + i]] = M_LAYERS[i]
        d = G @ m9
        rms = float(np.sqrt(np.mean(d ** 2)))
        out.append({"layer": LAYER_NAMES[i], "signal_rms_nT": rms,
                    "snr": rms / float(SIGMA_B),
                    "peak_abs_component_nT": float(np.max(np.abs(d)))})
    return out


def nearest_face_distance(y_sensor, z_sensor):
    return [float(np.hypot(max(0.0 - y_sensor, 0.0),
                           max(zb - z_sensor, 0.0) + max(z_sensor - zt, 0.0)))
            for (zb, zt) in LAYER_Z]


# --------------------------------------------------------------------------- #
# 9-parameter damped solver (configuration identical to
# inversion_helpers.iterative_recover_M_ref and engine/three_layer_wall.gncg_solve_9;
# written out so the run can also return beta and the target)
# --------------------------------------------------------------------------- #
def gncg_solve_9(sim, dobs, sigma, m0, beta_seed, reference_model,
                 opt_kwargs=None, beta0_ratio=DEFAULT_BETA0_RATIO):
    opt_kwargs = DEFAULT_GNCG_OPT if opt_kwargs is None else opt_kwargs
    m0 = np.asarray(m0, float)
    nP = int(m0.size)
    nD = int(sim.survey.nD)
    data_obj = data.Data(survey=sim.survey, dobs=np.asarray(dobs, float))
    data_obj.standard_deviation = np.full(nD, float(sigma))
    dmis = data_misfit.L2DataMisfit(data=data_obj, simulation=sim)
    reg = regularization.Smallness(mesh=discretize.TensorMesh([np.ones(nP)]),
                                   mapping=maps.IdentityMap(nP=nP))
    reg.reference_model = np.asarray(reference_model, float)
    opt = optimization.ProjectedGNCG(lower=-np.inf, upper=np.inf, **opt_kwargs)
    invprob = inverse_problem.BaseInvProblem(dmis, reg, opt)
    target = directives.TargetMisfit(chifact=1.0)
    dlist = [directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio, random_seed=int(beta_seed)),
             directives.UpdatePreconditioner(), target]
    inv = inversion.BaseInversion(invprob, directiveList=dlist)
    with _ctx.redirect_stdout(_io.StringIO()):
        m_rec = np.asarray(inv.run(m0), float)
    phi_d = float(dmis(m_rec))
    return (m_rec, int(getattr(opt, "iter", -1)), phi_d, float(invprob.beta),
            float(target.target))


def verify_solver_fidelity(sim, dobs):
    """Assert the local solver gives the same result as inversion_helpers.iterative_recover_M_ref."""
    m_a, it_a, phi_a, _, _ = gncg_solve_9(sim, dobs, SIGMA_B, np.zeros(9), 1234, np.zeros(9))
    m_b, it_b, phi_b, _ = H.iterative_recover_M_ref(sim, dobs, SIGMA_B, np.zeros(9), 1234,
                                                    np.zeros(9),
                                                    opt_kwargs=DEFAULT_GNCG_OPT,
                                                    beta0_ratio=DEFAULT_BETA0_RATIO)
    assert np.allclose(m_a, m_b) and it_a == it_b and np.isclose(phi_a, phi_b), \
        "local gncg_solve_9 diverges from inversion_helpers.iterative_recover_M_ref"
    print("  solver fidelity vs iterative_recover_M_ref: identical")


# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #
def converged_mc(sim, noise, label):
    nD = int(sim.survey.nD)
    d_clean = np.asarray(sim.dpred(M_TRUE_9), float)
    assert noise.shape[1] == nD, "noise matrix does not match data size"
    beta_seed = record_seed(f"runB_beta_{label}")
    M_rec = np.empty((len(noise), 9))
    rms = np.empty(len(noise))
    iters = np.empty(len(noise), int)
    betas = np.empty(len(noise))
    target = None
    t0 = time.time()
    for k in range(len(noise)):
        m_rec, n_it, phi_d, beta, target = gncg_solve_9(
            sim, d_clean + noise[k], SIGMA_B, np.zeros(9), beta_seed + k, np.zeros(9))
        M_rec[k] = m_rec
        rms[k] = np.sqrt(phi_d / nD)
        iters[k] = n_it
        betas[k] = beta
        if k == 0 or (k + 1) % 250 == 0:
            print(f"      [{label}] {k+1}/{len(noise)}: iters={n_it} "
                  f"chi2={phi_d:.1f} (target {target:.1f}) rms={rms[k]:.4f} "
                  f"beta={beta:.3e} | {time.time()-t0:.1f}s", flush=True)
    conv = {"n_trials": int(len(noise)),
            "criterion": f"|normalized residual rms - 1| <= {RMS_TOL}",
            "n_converged": int(np.sum(np.abs(rms - 1.0) <= RMS_TOL)),
            "iters": {"min": int(iters.min()), "max": int(iters.max())},
            "target_phi_d": float(target), "nD": nD,
            "beta_mean": float(betas.mean()),
            "normalized_resid_rms": {"mean": float(rms.mean()), "std": float(rms.std()),
                                     "min": float(rms.min()), "max": float(rms.max())},
            "wall_clock_s": time.time() - t0}
    return M_rec, rms, conv


def per_layer_table(M_rec, rms, geom, signal):
    out = {}
    for i, Mi in enumerate(layer_slices(M_rec)):
        blk = recovery_metrics(Mi, M_LAYERS[i], rms, SIGMA_B, geom, per_component=True)
        tmag = np.linalg.norm(M_LAYERS[i])
        abs_err = np.abs(100.0 * (np.linalg.norm(Mi, axis=1) - tmag) / tmag)
        blk["mag_err_abs_pct"] = {"mean": float(abs_err.mean()), "std": float(abs_err.std())}
        blk["layer_name"] = LAYER_NAMES[i]
        blk["layer_z_m"] = list(LAYER_Z[i])
        blk["signal"] = signal[i]
        out[f"layer_{i+1}"] = blk
    return out


def print_table(title, table, dists):
    print(f"    {title}")
    print(f"      {'flow':15s} {'z (m)':>10s} {'dist':>7s} {'sig rms':>8s} "
          f"{'theta_err (deg)':>19s} {'dM signed (%)':>19s}")
    for i in range(3):
        b = table[f"layer_{i+1}"]
        de, me = b["dir_err_deg"], b["mag_err_pct"]
        print(f"      {b['layer_name']:15s} {str([int(v) for v in b['layer_z_m']]):>10s} "
              f"{dists[i]:7.3f} {b['signal']['signal_rms_nT']:8.4f} "
              f"{de['mean']:8.2f} +/- {de['std']:6.2f} "
              f"{me['mean']:8.2f} +/- {me['std']:6.2f}")


def max_dir_err(table):
    return float(max(table[f"layer_{i+1}"]["dir_err_deg"]["mean"] for i in range(3)))


def crossing(xs, ys, level):
    """Smallest x where piecewise-linear y(x) drops to <= level (ys DECREASING in x)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    if ys[0] <= level:
        return {"x": float(xs[0]), "status": "below_lowest_sampled"}
    for i in range(1, len(xs)):
        if ys[i] <= level:
            f = (ys[i - 1] - level) / (ys[i - 1] - ys[i])
            return {"x": float(xs[i - 1] + f * (xs[i] - xs[i - 1])),
                    "status": "interpolated"}
    return {"x": None, "status": "not_reached"}


def fmt_cross(c):
    if c["x"] is None:
        return "never (not reached)"
    tag = {"interpolated": "", "below_lowest_sampled": " (already met at lowest z sampled)"}
    return f"{c['x']:.3g} m{tag[c['status']]}"


def smallest_grid_z(zs, maxerr, level):
    """Smallest SAMPLED z (excluding the degenerate z2 == BASE_Z control) meeting `level`."""
    ok = [z for z, e in zip(zs, maxerr) if z > BASE_Z and e <= level]
    return float(min(ok)) if ok else None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    t_start = time.time()
    print(f"simpeg {simpeg.__version__} | discretize {discretize.__version__} | "
          f"numpy {np.__version__}")
    print(f"MASTER_SEED={MASTER_SEED} ACTIVE_SEED={ACTIVE_SEED} | N_MC={N_MC} | "
          f"sigma_B={SIGMA_B} nT/axis | dx={DX} m | x=[{X0},{X1}] y={BASE_Y}")

    sxyz_low = traverse(BASE_Z)
    sim_low = wall_sim(sxyz_low)
    nD_low = int(sim_low.survey.nD)
    dists_low = nearest_face_distance(BASE_Y, BASE_Z)
    print(f"\nlow traverse: z={BASE_Z} m, {len(sxyz_low)} stations, nD={nD_low}")

    print("\nsolver fidelity:")
    rng_probe = np.random.default_rng(record_seed("runB_fidelity_probe"))
    verify_solver_fidelity(sim_low,
                           sim_low.dpred(M_TRUE_9) + rng_probe.normal(0, SIGMA_B, nD_low))

    # matched noise: ONE low-traverse matrix reused at every height
    noise_low = np.random.default_rng(
        record_seed(f"runB_noise_low_y{BASE_Y}_z{BASE_Z}_dx{DX}")).normal(
        0., SIGMA_B, (N_MC, nD_low))

    # ---- baseline: single traverse (anchor cond ~ 238, max corr ~ 0.997) ----
    print(f"\n=== BASELINE: single traverse at z={BASE_Z} m (no second traverse) ===")
    diag_low = operator_diagnostics(sim_low.G)
    sig_low = per_layer_signal(sim_low.G)
    geom_low = survey_geom_dict(sxyz_low, {"spacing_m": DX, "n_data": nD_low})
    print(f"  cond={diag_low['condition_number']:.2f} (anchor "
          f"{ANCHORS['single_base_traverse']['cond']}) | max inter-layer corr="
          f"{diag_low['interlayer_correlation_max']:.5f} (anchor "
          f"{ANCHORS['single_base_traverse']['max_corr']})")
    M_b, rms_b, conv_b = converged_mc(sim_low, noise_low, "baseline_single")
    tab_b_it = per_layer_table(M_b, rms_b, geom_low, sig_low)
    print_table(f"BASELINE - ITERATIVE, N={N_MC}", tab_b_it, dists_low)
    K_b = ls_operator(sim_low.G)
    D_b = (sim_low.G @ M_TRUE_9)[None, :] + noise_low
    M_bls = D_b @ K_b.T
    rms_bls = np.sqrt(np.mean(((M_bls @ sim_low.G.T - D_b) / SIGMA_B) ** 2, axis=1))
    tab_b_ls = per_layer_table(M_bls, rms_bls, geom_low, sig_low)
    print_table(f"BASELINE - CLOSED-FORM LS, N={N_MC}", tab_b_ls, dists_low)
    del D_b
    baseline = {"z2_m": None, "n_traverses": 1, "n_data": nD_low,
                "n_stations": int(len(sxyz_low)),
                "operator_diagnostics": diag_low, "per_layer_signal": sig_low,
                "nearest_face_distance_m": dict(zip(LAYER_NAMES, dists_low)),
                "per_layer_iterative": tab_b_it, "per_layer_closed_form_ls": tab_b_ls,
                "convergence": conv_b,
                "max_dir_err_deg_iterative": max_dir_err(tab_b_it),
                "max_dir_err_deg_closed_form_ls": max_dir_err(tab_b_ls),
                "anchor": ANCHORS["single_base_traverse"]}

    # ---- height sweep ----
    print(f"\n=== SWEEP: second-traverse height z2 in {Z2_SWEEP} m, N={N_MC} ===")
    heights = {}
    for z2 in Z2_SWEEP:
        t0 = time.time()
        key = f"z2_{z2:g}m"
        sxyz_hi = traverse(z2)
        sxyz_pair = np.vstack([sxyz_low, sxyz_hi])
        sim_pair = wall_sim(sxyz_pair)
        nD_pair = int(sim_pair.survey.nD)
        # data ordering must be the concatenation of the per-traverse blocks for the
        # matched hstacked noise to line up
        assert np.allclose(sim_pair.dpred(M_TRUE_9),
                           np.concatenate([sim_low.dpred(M_TRUE_9),
                                           wall_sim(sxyz_hi).dpred(M_TRUE_9)])), \
            "pair data ordering is not the concatenation of the per-traverse blocks"
        diag = operator_diagnostics(sim_pair.G)
        sig = per_layer_signal(sim_pair.G)
        sig_hi = per_layer_signal(wall_sim(sxyz_hi).G)
        dists_hi = nearest_face_distance(BASE_Y, z2)
        geom = survey_geom_dict(sxyz_pair, {"spacing_m": DX, "n_data": nD_pair,
                                            "z_low_m": BASE_Z, "z_high_m": z2})
        print(f"\n-- z2 = {z2} m | nD={nD_pair} | cond={diag['condition_number']:.2f} "
              f"| max inter-layer corr={diag['interlayer_correlation_max']:.5f} "
              f"| upper-traverse nearest-face dist="
              f"{[round(v,2) for v in dists_hi]} m --")
        noise_hi = np.random.default_rng(
            record_seed(f"runB_noise_high_y{BASE_Y}_z{z2}_dx{DX}")).normal(
            0., SIGMA_B, (N_MC, nD_low))
        noise_pair = np.hstack([noise_low, noise_hi])
        M_it, rms_it, conv = converged_mc(sim_pair, noise_pair, key)
        tab_it = per_layer_table(M_it, rms_it, geom, sig)
        print_table(f"z2={z2} m - ITERATIVE, N={N_MC}", tab_it, dists_low)
        K = ls_operator(sim_pair.G)
        D = (sim_pair.G @ M_TRUE_9)[None, :] + noise_pair
        M_ls = D @ K.T
        rms_ls = np.sqrt(np.mean(((M_ls @ sim_pair.G.T - D) / SIGMA_B) ** 2, axis=1))
        tab_ls = per_layer_table(M_ls, rms_ls, geom, sig)
        print_table(f"z2={z2} m - CLOSED-FORM LS, N={N_MC}", tab_ls, dists_low)
        del D, noise_pair, noise_hi
        heights[key] = {
            "z2_m": z2, "n_traverses": 2, "n_data": nD_pair,
            "n_stations": int(len(sxyz_pair)),
            "vertical_separation_m": z2 - BASE_Z,
            "operator_diagnostics": diag,
            "per_layer_signal_pair": sig,
            "per_layer_signal_upper_traverse_only": sig_hi,
            "nearest_face_distance_upper_m": dict(zip(LAYER_NAMES, dists_hi)),
            "per_layer_iterative": tab_it, "per_layer_closed_form_ls": tab_ls,
            "convergence": conv,
            "max_dir_err_deg_iterative": max_dir_err(tab_it),
            "max_dir_err_deg_closed_form_ls": max_dir_err(tab_ls),
            "wall_clock_s": time.time() - t0,
        }
        print(f"    max per-flow direction error: iterative "
              f"{max_dir_err(tab_it):.2f} deg | LS {max_dir_err(tab_ls):.2f} deg | "
              f"rms {conv['normalized_resid_rms']['mean']:.4f} | "
              f"{time.time()-t0:.1f}s", flush=True)

    # ---- headline ----
    print("\n=== HEADLINE ===")
    zs = [heights[f"z2_{z:g}m"]["z2_m"] for z in Z2_SWEEP]
    conds = [heights[f"z2_{z:g}m"]["operator_diagnostics"]["condition_number"]
             for z in Z2_SWEEP]
    corrs = [heights[f"z2_{z:g}m"]["operator_diagnostics"]["interlayer_correlation_max"]
             for z in Z2_SWEEP]
    head = {"thresholds": {}, "condition_number_vs_z2": dict(zip(
        [f"{z:g}" for z in zs], conds))}
    for sol in ("iterative", "closed_form_ls"):
        errs = [heights[f"z2_{z:g}m"][f"max_dir_err_deg_{sol}"] for z in Z2_SWEEP]
        blk = {}
        for lvl in THRESHOLDS:
            # exclude the degenerate z2 == BASE_Z control from the interpolation
            zz = [z for z in zs if z > BASE_Z]
            ee = [e for z, e in zip(zs, errs) if z > BASE_Z]
            blk[f"all_three_flows_within_{lvl:g}deg"] = {
                "interpolated_min_z2_m": crossing(zz, ee, lvl),
                "smallest_sampled_z2_m": smallest_grid_z(zs, errs, lvl),
            }
        blk["max_dir_err_deg_vs_z2"] = dict(zip([f"{z:g}" for z in zs], errs))
        head["thresholds"][sol] = blk
        print(f"  [{sol}] max per-flow direction error vs z2 (m):")
        for z, e, c, r in zip(zs, errs, conds, corrs):
            flag = " (degenerate: no vertical separation)" if z == BASE_Z else ""
            print(f"      z2={z:5g} m  max theta={e:6.2f} deg  cond={c:8.2f}  "
                  f"max corr={r:.5f}{flag}")
        for lvl in THRESHOLDS:
            b = blk[f"all_three_flows_within_{lvl:g}deg"]
            print(f"      all three flows within {lvl:g} deg: interpolated z2 >= "
                  f"{fmt_cross(b['interpolated_min_z2_m'])}; smallest sampled z2 = "
                  f"{b['smallest_sampled_z2_m']} m")

    # diminishing returns on the condition number
    zz = [z for z in zs if z > BASE_Z]
    cc = [c for z, c in zip(zs, conds) if z > BASE_Z]
    dcond = [(zz[i], cc[i], cc[i] - cc[i - 1]) for i in range(1, len(zz))]
    best = min(cc)
    within10 = [z for z, c in zip(zz, cc) if c <= 1.10 * best]
    head["diminishing_returns"] = {
        "min_condition_number": float(best),
        "z2_at_min_condition_number_m": float(zz[int(np.argmin(cc))]),
        "smallest_z2_within_10pct_of_min_cond_m": float(min(within10)),
        "condition_number_step_changes": [
            {"z2_m": z, "cond": float(c), "delta_vs_previous_z2": float(d)}
            for z, c, d in dcond],
    }
    print(f"\n  condition number: min={best:.2f} at z2="
          f"{zz[int(np.argmin(cc))]:g} m; first within 10% of that minimum at z2="
          f"{min(within10):g} m")
    print("  per-step change in cond:")
    for z, c, d in dcond:
        print(f"      z2={z:5g} m cond={c:8.2f} (delta {d:+8.2f})")

    # operational side of the mast/slope line
    op = {}
    for sol in ("iterative", "closed_form_ls"):
        for lvl in THRESHOLDS:
            b = head["thresholds"][sol][f"all_three_flows_within_{lvl:g}deg"]
            z_req = b["smallest_sampled_z2_m"]
            op[f"{sol}_{lvl:g}deg"] = {
                "smallest_sampled_z2_m": z_req,
                "mast_limit_m": MAST_LIMIT_M,
                "side": (None if z_req is None else
                         ("boom_or_mast" if z_req <= MAST_LIMIT_M else "drive_up_the_slope")),
            }
    head["operational_classification"] = op
    print(f"\n  operational classification (mast/boom plausible at <= {MAST_LIMIT_M} m):")
    for k, v in op.items():
        print(f"      {k:28s} z2={v['smallest_sampled_z2_m']} m -> {v['side']}")

    # z = 12.5 m anchor comparison
    a = heights["z2_12.5m"]
    anchor_cmp = {
        "anchor": ANCHORS["pair_z12.5"],
        "reproduced": {
            "cond": a["operator_diagnostics"]["condition_number"],
            "max_corr": a["operator_diagnostics"]["interlayer_correlation_max"],
            "dir_deg_iterative": [a["per_layer_iterative"][f"layer_{i+1}"]["dir_err_deg"]["mean"]
                                  for i in range(3)]},
        "note": ("reference is the three_layer_wall.py pair (N = 1000, its own seeds); this run "
                 "uses N = 500 and different seeds, so agreement is to Monte-Carlo error"),
    }
    head["z12.5_anchor_comparison"] = anchor_cmp
    print(f"\n  z2=12.5 m vs three_layer_wall.py pair: cond {anchor_cmp['reproduced']['cond']:.2f} "
          f"(anchor {ANCHORS['pair_z12.5']['cond']}) | max corr "
          f"{anchor_cmp['reproduced']['max_corr']:.4f} (anchor "
          f"{ANCHORS['pair_z12.5']['max_corr']}) | dir "
          f"{[round(v,2) for v in anchor_cmp['reproduced']['dir_deg_iterative']]} deg "
          f"(anchor {ANCHORS['pair_z12.5']['dir_deg']})")

    payload = {
        "meta": {
            "script": "scripts/elevated_traverse_height_sweep.py",
            "run": "minimum useful second-traverse height",
            "question": ("smallest second-traverse sensor height at which all three flows "
                         "recover to within 5 deg, and separately 10 deg, with everything "
                         "but z held fixed"),
            "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
            "library_versions": {"simpeg": simpeg.__version__,
                                 "discretize": discretize.__version__,
                                 "numpy": np.__version__},
            "master_seed": MASTER_SEED, "active_seed": ACTIVE_SEED,
            "seed_scheme": "seed_for(name) = sha256(f'{ACTIVE_SEED}:{name}')[:8] (hex->int)",
            "seeds_resolved": SEEDS_USED,
            "N_mc": N_MC, "sigma_B_nT": SIGMA_B,
            "noise_model": ("i.i.d. zero-mean Gaussian sigma_B = 0.50 nT per axis per "
                            "station; ONE matched low-traverse noise matrix reused at every "
                            "height, per-height matrix for the upper traverse"),
            "sampling_m": DX,
            "scenario": {
                "layers_z_m": [list(z) for z in LAYER_Z], "layer_names": LAYER_NAMES,
                "M_layers_Apm": M_LAYERS.tolist(), "M0_Apm": M_NRM_MIN,
                "polarity": "bottom -z_hat / middle +y_hat (into wall) / top +z_hat",
                "each_layer_is_one_exact_prism": True,
                "wall_face_y_m": 0.0, "x_along_strike_m": [-50.0, 50.0],
                "y_into_wall_m": [0.0, 50.0],
                "traverse_x_range_m": [X0, X1], "traverse_y_m": BASE_Y,
                "traverse_spacing_m": DX, "standoff_m": abs(BASE_Y),
                "traverse_1_z_m": BASE_Z, "traverse_2_z_sweep_m": Z2_SWEEP,
                "held_fixed": ("standoff y = -1 m, x extent, 1 cm spacing and path length are "
                               "identical for both traverses at every height; only z changes"),
            },
            "inversion": {
                "free_parameters": 9, "K": 3,
                "geometry": "fixed and exactly known",
                "iterative": (f"ProjectedGNCG({DEFAULT_GNCG_OPT}) + Smallness(zero reference) "
                              f"+ BetaEstimate_ByEig(beta0_ratio={DEFAULT_BETA0_RATIO}) + "
                              "UpdatePreconditioner + TargetMisfit(chifact=1.0); verified "
                              "identical to inversion_helpers.iterative_recover_M_ref"),
                "closed_form_ls": ("undamped zero-referenced weighted least squares "
                                   "(inversion_helpers.ls_operator)"),
                "beta_selection": f"BetaEstimate_ByEig(beta0_ratio={DEFAULT_BETA0_RATIO})",
                "stopping_criterion": "TargetMisfit(chifact=1.0), target phi_d = nD",
                "phi_d_convention": ("simpeg L2DataMisfit returns sum((r/sigma)^2) = chi^2 "
                                     "(no 0.5 prefactor); target = nD, NOT 0.5*nD; "
                                     "normalized residual rms = sqrt(phi_d/nD)"),
                "convergence_criterion": f"|normalized residual rms - 1| <= {RMS_TOL}",
            },
            "dM_convention": "mag_err_pct = 100*(|M_rec|-|M_true|)/|M_true| (SIGNED)",
            "threshold_definition": ("a height 'recovers all three flows' when the MAXIMUM "
                                     "over the three flows of the ensemble-MEAN direction "
                                     "error is at or below the threshold"),
            "mast_limit_m": MAST_LIMIT_M,
            "anchors": ANCHORS,
        },
        "baseline_single_traverse": baseline,
        "heights": heights,
        "headline": head,
        "wall_clock_total_s": time.time() - t_start,
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {RESULTS_PATH}")
    print(f"total runtime {time.time()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

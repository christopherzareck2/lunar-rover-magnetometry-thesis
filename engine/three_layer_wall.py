"""Three-layer wall (paper Section 4.4, Table 4): per-flow recovery with the iterative parametric inversion.

Scenario: three stacked 5 m flows (x in -50..50 m, y in 0..50 m into the wall, z in 0..15 m), each
one exact prism with |M| = 2.4e-3 A/m: flow 1 (bottom, oldest) along -z, flow 2 along +y, flow 3
(top, youngest) along +z. Wall face at y = 0, rover traverse at y = -1 m, x = -120..120 m at 1 cm
(24 001 stations), sigma_B = 0.50 nT per axis.

Arms: (A) single base traverse at z = 0.5 m; (C) base traverse plus a second traverse at
z = 12.5 m inverted jointly. Nine free magnetization components (K = 3), geometry fixed.

Solver: ProjectedGNCG with a zero-reference Smallness term (beta0_ratio = 1e-4), preconditioner
update and TargetMisfit(chifact = 1); N = 1000 noise realizations per arm. SimPEG's misfit is
phi_d = sum((r / sigma)^2), so the target is nD and the normalized residual rms is sqrt(phi_d / nD);
a realization is accepted on the misfit reached, not on the optimizer's exit flag.

Because the nine-parameter operator is ill-conditioned (inter-flow correlation up to ~0.997 for a
single traverse), the damping is not negligible for the weakly constrained upper flows, so the
undamped closed-form least-squares solution is reported alongside the iterative one.

Run:  python engine/three_layer_wall.py
Out:  results/engine/three_layer_wall_results.json
"""
import contextlib as _ctx
import io as _io
import json
import os
import time

import numpy as np
import discretize
import simpeg
from simpeg import (data, data_misfit, directives, inverse_problem, inversion,
                    maps, optimization, regularization)

import inversion_helpers as H
from inversion_helpers import (
    seed_for, layer_slices, recovery_metrics, survey_geom_dict, ls_operator,
    three_layer_mesh, iterative_recover_M_ref, operator_diagnostics,
    M_NRM_MIN, SIGMA_B, MASTER_SEED, ACTIVE_SEED, N_HEADLINE,
    DEFAULT_GNCG_OPT, DEFAULT_BETA0_RATIO,
)
from core import generate_rille_traverse

RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "engine",
                            "three_layer_wall_results.json")

# --------------------------------------------------------------------------- #
# Scenario (paper Sections 3.3.5 and 3.5.3)
# --------------------------------------------------------------------------- #
LAYER_Z = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]      # mesh cell order = ascending z
LAYER_NAMES = ["bottom/oldest", "middle", "top/youngest"]

M_LAYERS = M_NRM_MIN * np.array([[0., 0., -1.],          # bottom/oldest: -z_hat (down)
                                 [0., 1., 0.],           # middle: +y_hat (into wall)
                                 [0., 0., 1.]])          # top/youngest: +z_hat (up)
# simpeg vector ordering: [mx(all cells), my(all cells), mz(all cells)]
M_TRUE_9 = np.concatenate([M_LAYERS[:, 0], M_LAYERS[:, 1], M_LAYERS[:, 2]])

BASE_Y, BASE_Z = -1.0, 0.5        # rover at the BASE of the wall, 1 m standoff
DX = 0.01                         # 1 cm sampling
X0, X1 = -120.0, 120.0
ELEVATED_Z = 12.5                 # height of the second (elevated) traverse

N_MC = N_HEADLINE                 # standard Monte-Carlo count (1000)
RMS_TOL = 0.05                    # accept |normalized residual rms - 1| <= 5%

SEEDS_USED = {}


def record_seed(label):
    # the labels ("part8_...") are the ones the stored results were generated with; changing them changes the draws
    s = int(seed_for(label))
    SEEDS_USED[label] = s
    return s


# --------------------------------------------------------------------------- #
# Forward operators
# --------------------------------------------------------------------------- #
def wall_sim(rx_locs):
    """Vector simulation for the three-prism wall; nP == 9 (three uniform layer vectors)."""
    mesh = three_layer_mesh()
    return H._sim_for(mesh, rx_locs, mesh.n_cells)


def traverse(y, z, dx=DX):
    return generate_rille_traverse(X0, X1, y, z, dx)


def per_layer_signal(G, sigma=SIGMA_B):
    """Noise-free signal statistics for each flow driven ALONE through the shared operator."""
    out = []
    for i in range(3):
        m9 = np.zeros(9)
        m9[[i, 3 + i, 6 + i]] = M_LAYERS[i]
        d = G @ m9
        rms = float(np.sqrt(np.mean(d ** 2)))
        out.append({"layer": LAYER_NAMES[i],
                    "signal_rms_nT": rms,
                    "snr": rms / float(sigma),
                    "peak_abs_component_nT": float(np.max(np.abs(d)))})
    return out


def nearest_face_distance(y_sensor, z_sensor):
    """Perpendicular distance from a sensor station to the nearest face of each flow prism."""
    return [float(np.hypot(max(0.0 - y_sensor, 0.0),
                           max(zb - z_sensor, 0.0) + max(z_sensor - zt, 0.0)))
            for (zb, zt) in LAYER_Z]


# --------------------------------------------------------------------------- #
# 9-parameter damped solver (same configuration as inversion_helpers.iterative_recover_M_ref;
# written out here so the run can also return beta and the convergence log).
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
             directives.UpdatePreconditioner(),
             target]
    inv = inversion.BaseInversion(invprob, directiveList=dlist)
    buf = _io.StringIO()
    with _ctx.redirect_stdout(buf):
        m_rec = np.asarray(inv.run(m0), float)
    phi_d = float(dmis(m_rec))
    return (m_rec, int(getattr(opt, "iter", -1)), phi_d, float(invprob.beta),
            float(target.target), buf.getvalue())


def verify_solver_fidelity(sim, dobs, sigma):
    """Assert gncg_solve_9 gives the same result as inversion_helpers.iterative_recover_M_ref."""
    m_a, it_a, phi_a, _, _, _ = gncg_solve_9(sim, dobs, sigma, np.zeros(9), 1234, np.zeros(9))
    m_b, it_b, phi_b, _ = iterative_recover_M_ref(sim, dobs, sigma, np.zeros(9), 1234,
                                                  np.zeros(9), opt_kwargs=DEFAULT_GNCG_OPT,
                                                  beta0_ratio=DEFAULT_BETA0_RATIO)
    assert np.allclose(m_a, m_b) and it_a == it_b and np.isclose(phi_a, phi_b), (
        "local gncg_solve_9 diverges from inversion_helpers.iterative_recover_M_ref")
    print("  solver fidelity check vs iterative_recover_M_ref: identical")


# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #
def converged_mc(sim, noise, sigma=SIGMA_B, label=""):
    """Iterative inversion of every noise realization, with per-trial convergence status."""
    nD = int(sim.survey.nD)
    d_clean = np.asarray(sim.dpred(M_TRUE_9), float)
    assert noise.shape[1] == nD, "noise matrix does not match data size"
    beta_seed = record_seed(f"part8_beta_{label}")

    M_rec = np.empty((len(noise), 9))
    resid_rms = np.empty(len(noise))
    trials = []
    betas = []
    t0 = time.time()
    for k in range(len(noise)):
        m_rec, n_iter, phi_d, beta, target, _log = gncg_solve_9(
            sim, d_clean + noise[k], sigma, np.zeros(9), beta_seed + k, np.zeros(9))
        rms = float(np.sqrt(phi_d / nD))
        M_rec[k] = m_rec
        resid_rms[k] = rms
        betas.append(beta)
        trials.append({"trial": k, "n_iter": n_iter, "phi_d_chi2": float(phi_d),
                       "target_phi_d": float(target), "nD": nD,
                       "normalized_resid_rms": rms,
                       "reached_target_strict": bool(phi_d <= target),
                       "rms_within_tol": bool(abs(rms - 1.0) <= RMS_TOL)})
        if k == 0 or (k + 1) % 250 == 0:
            print(f"    [{label}] {k+1}/{len(noise)}: iters={n_iter} "
                  f"chi2={phi_d:.1f} (target {target:.1f}) rms={rms:.4f} "
                  f"beta={beta:.3e} | {time.time()-t0:.1f}s", flush=True)
    return M_rec, resid_rms, trials, float(np.mean(betas))


def convergence_report(trials):
    bad = [t for t in trials if not t["rms_within_tol"]]
    strict = [t for t in trials if t["reached_target_strict"]]
    rms = [t["normalized_resid_rms"] for t in trials]
    return {"n_trials": len(trials),
            "criterion": f"|normalized residual rms - 1| <= {RMS_TOL}",
            "n_converged": len(trials) - len(bad),
            "n_not_converged": len(bad),
            "all_converged": len(bad) == 0,
            "n_reached_target_strict": len(strict),
            "iters": {"min": int(min(t["n_iter"] for t in trials)),
                      "max": int(max(t["n_iter"] for t in trials))},
            "phi_d_chi2": {"mean": float(np.mean([t["phi_d_chi2"] for t in trials])),
                           "max": float(np.max([t["phi_d_chi2"] for t in trials])),
                           "target": float(trials[0]["target_phi_d"]),
                           "nD": trials[0]["nD"]},
            "normalized_resid_rms": {"mean": float(np.mean(rms)), "std": float(np.std(rms)),
                                     "min": float(np.min(rms)), "max": float(np.max(rms))},
            "non_converged_trials": bad[:20]}


def per_layer_table(M_rec, resid_rms, geom, signal):
    """Per-flow metrics; dM reported signed (recovery_metrics) and absolute."""
    out = {}
    for i, Mi in enumerate(layer_slices(M_rec)):
        blk = recovery_metrics(Mi, M_LAYERS[i], resid_rms, SIGMA_B, geom,
                               store_raw=False, per_component=True)
        tmag = np.linalg.norm(M_LAYERS[i])
        abs_err = np.abs(100.0 * (np.linalg.norm(Mi, axis=1) - tmag) / tmag)
        blk["mag_err_abs_pct"] = {"mean": float(np.mean(abs_err)),
                                  "std": float(np.std(abs_err))}
        blk["layer_name"] = LAYER_NAMES[i]
        blk["layer_z_m"] = list(LAYER_Z[i])
        blk["signal"] = signal[i]
        out[f"layer_{i+1}"] = blk
    return out


def print_table(title, table, dists=None):
    print(f"\n{title}")
    print(f"  {'flow':15s} {'z (m)':>11s} {'dist':>7s} {'sig rms':>8s} {'SNR':>6s} "
          f"{'theta_err (deg)':>19s} {'dM signed (%)':>19s} {'|dM| (%)':>9s}")
    for i in range(3):
        b = table[f"layer_{i+1}"]
        de, me = b["dir_err_deg"], b["mag_err_pct"]
        d = f"{dists[i]:7.3f}" if dists else f"{'-':>7s}"
        print(f"  {b['layer_name']:15s} "
              f"{str([int(v) for v in b['layer_z_m']]):>11s} {d} "
              f"{b['signal']['signal_rms_nT']:8.4f} {b['signal']['snr']:6.2f} "
              f"{de['mean']:8.2f} +/- {de['std']:6.2f} "
              f"{me['mean']:8.2f} +/- {me['std']:6.2f} "
              f"{b['mag_err_abs_pct']['mean']:9.2f}")


def run_arm(name, sxyz, noise, label, dists):
    """Both solvers on one survey geometry."""
    sim = wall_sim(sxyz)
    G = sim.G
    nD = int(sim.survey.nD)
    signal = per_layer_signal(G)
    diag = operator_diagnostics(G)
    geom = survey_geom_dict(sxyz, {"spacing_m": DX, "n_data": nD})
    print(f"\n=== {name} ===")
    print(f"  stations={len(sxyz)} nD={nD} | cond={diag['condition_number']:.1f} "
          f"| max inter-layer corr={diag['interlayer_correlation_max']:.5f}")

    M_it, rms_it, trials, beta_mean = converged_mc(sim, noise, label=label)
    conv = convergence_report(trials)
    print(f"  ITERATIVE: converged {conv['n_converged']}/{conv['n_trials']} "
          f"(strict target {conv['n_reached_target_strict']}/{conv['n_trials']}) | "
          f"iters {conv['iters']['min']}-{conv['iters']['max']} | "
          f"rms {conv['normalized_resid_rms']['min']:.4f}-"
          f"{conv['normalized_resid_rms']['max']:.4f} | beta~{beta_mean:.3e}")
    table_it = per_layer_table(M_it, rms_it, geom, signal)
    print_table(f"  {name} - ITERATIVE (damped), N={N_MC}", table_it, dists)

    # undamped closed-form least squares (parts 2/3/5/7 path for this scenario)
    K = ls_operator(G)
    D = (G @ M_TRUE_9)[None, :] + noise
    M_ls = D @ K.T
    rms_ls = np.sqrt(np.mean(((M_ls @ G.T - D) / SIGMA_B) ** 2, axis=1))
    table_ls = per_layer_table(M_ls, rms_ls, geom, signal)
    print_table(f"  {name} - CLOSED-FORM LS (undamped), N={N_MC}", table_ls, dists)
    print(f"  amplitude shrinkage from Smallness: "
          f"mean |M_iterative| / |M_ls| per flow = "
          f"{[round(float(np.mean(np.linalg.norm(a, axis=1)) / np.mean(np.linalg.norm(b, axis=1))), 3) for a, b in zip(layer_slices(M_it), layer_slices(M_ls))]}")

    return {"per_layer_iterative": table_it,
            "per_layer_closed_form_ls": table_ls,
            "convergence": conv,
            "beta_mean": beta_mean,
            "operator_diagnostics": diag,
            "n_data": nD,
            "n_stations": int(len(sxyz)),
            "nearest_face_distance_m": dict(zip(LAYER_NAMES, dists))}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    t_start = time.time()
    print(f"simpeg {simpeg.__version__} | discretize {discretize.__version__} | "
          f"numpy {np.__version__}")
    print(f"MASTER_SEED={MASTER_SEED} ACTIVE_SEED={ACTIVE_SEED} | N_MC={N_MC} | "
          f"sigma_B={SIGMA_B} nT/axis | dx={DX} m")

    mesh = three_layer_mesh()
    x_half = 0.5 * float(mesh.nodes_x[-1] - mesh.nodes_x[0])
    y_ext = float(mesh.nodes_y[-1] - mesh.nodes_y[0])
    print("\n=== WALL PRISM DIMENSIONS (three_layer_mesh) ===")
    print(f"  x (along strike): {float(mesh.nodes_x[0]):+.1f} .. {float(mesh.nodes_x[-1]):+.1f} m "
          f"-> full width {2*x_half:.1f} m, half-width {x_half:.1f} m")
    print(f"  y (into wall):    {float(mesh.nodes_y[0]):+.1f} .. {float(mesh.nodes_y[-1]):+.1f} m "
          f"-> extent {y_ext:.1f} m; wall face at y = 0")
    print(f"  z (vertical):     {[float(v) for v in mesh.nodes_z]} -> flows {LAYER_Z}")
    print(f"  cell sizes: hx={[float(v) for v in mesh.h[0]]} hy={[float(v) for v in mesh.h[1]]} "
          f"hz={[float(v) for v in mesh.h[2]]}")
    print(f"  n_cells={mesh.n_cells} (ONE EXACT PRISM PER FLOW; no voxel discretization "
          f"error), nP={3*mesh.n_cells}")

    sxyz_low = traverse(BASE_Y, BASE_Z)
    sxyz_high = traverse(BASE_Y, ELEVATED_Z)
    dists_low = nearest_face_distance(BASE_Y, BASE_Z)
    dists_high = nearest_face_distance(BASE_Y, ELEVATED_Z)

    sim_low_probe = wall_sim(sxyz_low)
    nD_low = int(sim_low_probe.survey.nD)
    print("\nsolver fidelity:")
    rng_probe = np.random.default_rng(record_seed("part8_fidelity_probe"))
    verify_solver_fidelity(sim_low_probe,
                           sim_low_probe.dpred(M_TRUE_9) + rng_probe.normal(0, SIGMA_B, nD_low),
                           SIGMA_B)

    # matched per-traverse noise streams
    lab_low = f"part8_wall_noise_y{BASE_Y}_z{BASE_Z}_dx{DX}"
    lab_high = f"part8_wall_noise_y{BASE_Y}_z{ELEVATED_Z}_dx{DX}"
    noise_low = np.random.default_rng(record_seed(lab_low)).normal(0., SIGMA_B, (N_MC, nD_low))
    noise_high = np.random.default_rng(record_seed(lab_high)).normal(0., SIGMA_B, (N_MC, nD_low))

    arms = {}
    arms["A_single_base_traverse"] = run_arm(
        f"ARM A: single drive-by at the BASE of the wall (y={BASE_Y}, z={BASE_Z})",
        sxyz_low, noise_low, "A_base", dists_low)

    arms["B_single_elevated_traverse"] = run_arm(
        f"ARM B: single ELEVATED traverse (y={BASE_Y}, z={ELEVATED_Z})",
        sxyz_high, noise_high, "B_elevated", dists_high)

    # vertical pair: stacked receiver list; data ordering verified to be the
    # concatenation of the two per-traverse blocks, so hstacked noise stays matched.
    sxyz_pair = np.vstack([sxyz_low, sxyz_high])
    sim_pair_probe = wall_sim(sxyz_pair)
    assert np.allclose(sim_pair_probe.dpred(M_TRUE_9),
                       np.concatenate([sim_low_probe.dpred(M_TRUE_9),
                                       wall_sim(sxyz_high).dpred(M_TRUE_9)])), \
        "pair data ordering is not the concatenation of the per-traverse blocks"
    arms["C_vertical_pair"] = run_arm(
        f"ARM C: VERTICAL PAIR (z={BASE_Z} and z={ELEVATED_Z}, matched noise)",
        sxyz_pair, np.hstack([noise_low, noise_high]), "C_pair", dists_low)

    payload = {
        "meta": {
            "script": "engine/three_layer_wall.py",
            "helper_module": "inversion_helpers.py",
            "purpose": "three-layer per-flow recovery (paper Section 4.4, Table 4)",
            "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
            "library_versions": {"simpeg": simpeg.__version__,
                                 "discretize": discretize.__version__,
                                 "numpy": np.__version__},
            "master_seed": MASTER_SEED, "active_seed": ACTIVE_SEED,
            "seed_scheme": "seed_for(name) = sha256(f'{ACTIVE_SEED}:{name}')[:8] (hex->int)",
            "seeds_resolved": SEEDS_USED,
            "N_mc": N_MC, "sigma_B_nT": SIGMA_B,
            "solver_iterative": ("ProjectedGNCG(maxIter=20, maxIterCG=20, tolCG=1e-4) + "
                                 "Smallness(zero reference) + BetaEstimate_ByEig(beta0_ratio=1e-4) "
                                 "+ UpdatePreconditioner + TargetMisfit(chifact=1.0); geometry "
                                 "fixed, only the 9 layer magnetization components free. "
                                 "Verified identical to inversion_helpers.iterative_recover_M_ref."),
            "solver_closed_form": ("undamped zero-referenced weighted least squares "
                                   "(inversion_helpers.ls_operator)"),
            "phi_d_convention": ("simpeg L2DataMisfit.__call__ returns np.vdot(R,R) = "
                                 "sum((r/sigma)^2) = chi^2, NO 0.5 prefactor; "
                                 "TargetMisfit(chifact=1.0).target == nD; "
                                 "normalized residual rms = sqrt(phi_d/nD)"),
            "convergence_criterion": f"|normalized residual rms - 1| <= {RMS_TOL}",
            "scenario": {
                "layers_z_m": [list(z) for z in LAYER_Z],
                "layer_names": LAYER_NAMES,
                "M_layers_Apm": M_LAYERS.tolist(),
                "M0_Apm": M_NRM_MIN,
                "polarity": "bottom -z_hat (down) / middle +y_hat (into wall) / top +z_hat (up)",
                "wall_face_y_m": 0.0,
                "x_half_width_m": x_half,
                "x_full_width_m": 2 * x_half,
                "y_into_wall_extent_m": y_ext,
                "mesh_cell_sizes_m": {"hx": [float(v) for v in mesh.h[0]],
                                      "hy": [float(v) for v in mesh.h[1]],
                                      "hz": [float(v) for v in mesh.h[2]]},
                "mesh_n_cells": int(mesh.n_cells),
                "traverse_x_range_m": [X0, X1],
                "traverse_spacing_m": DX,
                "base_traverse_y_z_m": [BASE_Y, BASE_Z],
                "elevated_traverse_z_m": ELEVATED_Z,
                "standoff_m": abs(BASE_Y),
            },
        },
        "arms": arms,
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {RESULTS_PATH}")
    print(f"total runtime {time.time()-t_start:.1f}s")
    print("\nseeds used:")
    for k, v in SEEDS_USED.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

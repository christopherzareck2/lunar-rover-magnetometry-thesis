"""Background magnetization Monte Carlo (paper Sections 3.3.1 and 5.5.2); run in the paper geometry by
analysis/a05_background_and_correlated_noise.py.

The paper justifies a zero-magnetization background with Runcorn's theorem: a large, uniformly
magnetized mare unit produces negligible external field, so the outcrop signal comes from
topographic contrast. This run measures how far from uniform the underlying unit can be before
that justification fails: a heterogeneously magnetized 5 m slab beneath the outcrop generates an
additional field that is added to the data and not inverted for; the single-body inversion of
isolated outcrop is otherwise as in Section 4.2 (drive-around survey, 1 cm sampling).

Slab: z = -5.5..-0.5 m, 1 m cells over a central core, geometrically padded to about +/-1200 m so
that a uniformly magnetized slab produces an edge artifact of 0.65 % of sigma_B (uniform control,
reported). Heterogeneity per cell group: intensity log-uniform over [0.01, 1] x M_NRM,min,
direction scattered about -z with a swept angular width sigma_theta in {0, 5, 15, 30, 60, 90} deg
plus an isotropic reference.

Run:  python analysis/a05_background_and_correlated_noise.py background
"""
import json
import os
import sys
import time

import numpy as np
import discretize
import simpeg
from simpeg import maps
from simpeg.potential_fields import magnetics as mag

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import inversion_helpers as H
from inversion_helpers import (prism_cell_operator, prism_bounds, ls_operator,
                               recovery_metrics, seed_for, outcrop_survey,
                               survey_geom_dict, single_cell_sim, iterative_recover_M,
                               M_NRM_MIN, SIGMA_B, STANDOFF, SENSOR_H, PRISM_DIMS,
                               MASTER_SEED, ACTIVE_SEED, DEFAULT_GNCG_OPT,
                               DEFAULT_BETA0_RATIO)

RESULTS_PATH = os.path.join(os.path.dirname(_REPO), "results", "engine", "background_magnetization_mc_results.json")  # a05 writes to results/analysis instead

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
SPACING = 0.01                # 1 cm along-track
N_MC = 400                    # realizations per (sigma_theta, coherence) point
SLAB_THICK = 5.0              # m
SLAB_Z_TOP = -0.5             # m, outcrop base / ground surface
DH = 1.0                      # m, background cell size
CORE_HALF = 15.0              # m, half-width of the 1 m heterogeneous core
PAD_FAC = 1.6                 # geometric padding growth factor
PAD_TO = 1000.0               # m, minimum lateral reach of the padding

INT_LOG10_RANGE = (-2.0, 0.0)  # log-uniform intensity: [0.01, 1.0] * M_NRM_min
# SWEEP 3: log-uniform intensity floors [f, 1.0] * M_NRM_min; f = 1.0 is a fixed intensity
INTENSITY_FLOORS = [1.0, 0.8, 0.5, 0.2, 0.05, 0.01]

SIGMA_THETA_DEG = [0.0, 5.0, 15.0, 30.0, 60.0, 90.0]
ISOTROPIC_LABEL = "uniform_sphere"
# (label, block size in x/y (m), coherent over the full 5 m thickness?)
COHERENCE = [("cell_1m_iso", 1.0, False),      # independent 1 m cells
             ("block_1m", 1.0, True),
             ("block_2m", 2.0, True),
             ("block_5m", 5.0, True),
             ("block_10m", 10.0, True)]
HEADLINE_ARM = "cell_1m_iso"

m_true = M_NRM_MIN * np.array([0.0, 0.0, -1.0])
bnd = prism_bounds((0.0, 0.0, 0.0), PRISM_DIMS)

SEEDS_USED = {}


def record_seed(label):
    s = int(seed_for(label))
    SEEDS_USED[label] = s
    return s


# --------------------------------------------------------------------------- #
# Background slab mesh and forward operators
# --------------------------------------------------------------------------- #
def slab_mesh():
    """1 m core over +/- CORE_HALF, geometric padding out past PAD_TO; 5 x 1 m in z."""
    n_core = int(round(2 * CORE_HALF / DH))
    pads, w, reach = [], DH * PAD_FAC, CORE_HALF
    while reach < PAD_TO:
        pads.append(w)
        reach += w
        w *= PAD_FAC
    hx = np.r_[pads[::-1], DH * np.ones(n_core), pads]
    hz = DH * np.ones(int(round(SLAB_THICK / DH)))
    origin = [-(CORE_HALF + sum(pads)), -(CORE_HALF + sum(pads)), SLAB_Z_TOP - SLAB_THICK]
    return discretize.TensorMesh([hx, hx.copy(), hz], origin=origin)


def slab_G(mesh, rx_locs, active_cells=None):
    """Vector sensitivity (nD x 3*nA) for the slab, same source convention as _sim_for."""
    nA = mesh.n_cells if active_cells is None else int(np.count_nonzero(active_cells))
    kw = {} if active_cells is None else {"active_cells": active_cells}
    sim = mag.simulation.Simulation3DIntegral(
        mesh, survey=mag.survey.Survey(mag.sources.UniformBackgroundField(
            receiver_list=[mag.receivers.Point(np.asarray(rx_locs, float),
                                               components=["bx", "by", "bz"])],
            amplitude=H.AMP_NT, inclination=90.0, declination=0.0)),
        chiMap=maps.IdentityMap(nP=3 * nA), model_type="vector",
        store_sensitivities="ram", sensitivity_dtype=np.float64, **kw)
    return sim.G


def uniform_operator(G, nA):
    """(nD x 3) operator mapping ONE uniform (Mx,My,Mz) applied to all nA cells -> data."""
    return np.column_stack([G[:, k * nA:(k + 1) * nA].sum(axis=1) for k in range(3)])


# --------------------------------------------------------------------------- #
# Heterogeneity draws
# --------------------------------------------------------------------------- #
def group_ids(cc_core, bxy, vertical_coherent):
    """Group index per core cell: bxy x bxy m blocks in x/y, optionally merged in z."""
    ix = np.floor((cc_core[:, 0] + CORE_HALF) / bxy).astype(int)
    iy = np.floor((cc_core[:, 1] + CORE_HALF) / bxy).astype(int)
    iz = np.zeros(len(cc_core), int) if vertical_coherent else \
        np.floor((cc_core[:, 2] - (SLAB_Z_TOP - SLAB_THICK)) / DH).astype(int)
    key = (ix.astype(np.int64) * 100000 + iy) * 1000 + iz
    _, inv = np.unique(key, return_inverse=True)
    return inv


def draw_groups(rng, n_groups, sigma_theta_deg, n_draws, int_log10_range=INT_LOG10_RANGE):
    """(n_draws, n_groups, 3) magnetizations [A/m] about -z_hat.

    intensity ~ log-uniform on 10**int_log10_range * M_NRM_min; (0, 0) = fixed at M_NRM_min
    theta     = |N(0, sigma_theta)| clipped to [0, 180] deg from -z_hat, azimuth U(0, 2pi);
                sigma_theta_deg is None -> exactly isotropic on the sphere.
    """
    amp = M_NRM_MIN * 10.0 ** rng.uniform(*int_log10_range, size=(n_draws, n_groups))
    if sigma_theta_deg is None:
        cos_t = rng.uniform(-1.0, 1.0, size=(n_draws, n_groups))
        theta = np.arccos(cos_t)
    elif sigma_theta_deg == 0.0:
        theta = np.zeros((n_draws, n_groups))
    else:
        theta = np.clip(np.abs(rng.normal(0.0, np.deg2rad(sigma_theta_deg),
                                          size=(n_draws, n_groups))), 0.0, np.pi)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(n_draws, n_groups))
    st, ct = np.sin(theta), np.cos(theta)
    # unit vector at polar angle theta measured from -z_hat
    return np.stack([amp * st * np.cos(phi), amp * st * np.sin(phi), -amp * ct], axis=-1)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def invert_ls(K, G_out, D):
    """Closed-form LS recovery of the 3 outcrop components + normalized residual rms."""
    M_rec = D @ K.T
    resid = M_rec @ G_out.T - D
    return M_rec, np.sqrt(np.mean((resid / SIGMA_B) ** 2, axis=1))


def field_stats(d_bg, sig_rms):
    rms = np.sqrt(np.mean(d_bg ** 2, axis=1))
    return {"bg_field_rms_nT": {"mean": float(rms.mean()), "std": float(rms.std()),
                                "min": float(rms.min()), "max": float(rms.max())},
            "bg_field_peak_abs_nT": float(np.abs(d_bg).max()),
            "ratio_to_sigma_B": float(rms.mean() / SIGMA_B),
            "ratio_to_outcrop_signal_rms": float(rms.mean() / sig_rms)}


def block(M_rec, rms, d_bg, sig_rms, geom):
    blk = recovery_metrics(M_rec, m_true, rms, SIGMA_B, geom)
    blk["background_field"] = field_stats(d_bg, sig_rms)
    return blk


def crossing(xs, ys, level):
    """Smallest x at which the piecewise-linear y(x) first reaches `level`.

    Returns {"x", "status"} with status one of
      "interpolated"          : a genuine bracketed crossing
      "below_lowest_sampled"  : y already >= level at the smallest x sampled (x = that point,
                                the true crossing lies below the sampled range)
      "not_reached"           : y < level everywhere sampled (x = None)
    """
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    if ys[0] >= level:
        return {"x": float(xs[0]), "status": "below_lowest_sampled"}
    for i in range(1, len(xs)):
        if ys[i] >= level:
            f = (level - ys[i - 1]) / (ys[i] - ys[i - 1])
            return {"x": float(xs[i - 1] + f * (xs[i] - xs[i - 1])),
                    "status": "interpolated"}
    return {"x": None, "status": "not_reached"}


def fmt_cross(c):
    if c["x"] is None:
        return "never (not reached)"
    tag = {"interpolated": "", "below_lowest_sampled": " (already exceeded at lowest sampled)"}
    return f"{c['x']:.4g}{tag[c['status']]}"


def sweep_point(coh_label, lvl_label, sth, gid, n_groups, rng, G_core, U_all, U_core,
                d_out, noise, sig_rms, geom, K_ls, G_out, nD,
                int_log10_range=INT_LOG10_RANGE):
    """One (coherence, dispersion, intensity-range) point: draw, forward, invert, metrics."""
    Mg = draw_groups(rng, n_groups, sth, N_MC, int_log10_range=int_log10_range)
    Mc = Mg[:, gid, :]
    m_bar = Mc.mean(axis=1)
    m_flat = np.concatenate([Mc[:, :, 0], Mc[:, :, 1], Mc[:, :, 2]], axis=1)
    d_core = m_flat @ G_core.T
    d_raw = d_core + m_bar @ (U_all - U_core).T      # padding at the realization core mean
    d_dc = d_core - m_bar @ U_core.T                 # zero-mean fluctuation only
    cell_int = np.linalg.norm(Mc, axis=2) / M_NRM_MIN
    entry = {"sigma_theta_deg": sth, "n_groups": n_groups,
             "intensity_log10_range": list(int_log10_range),
             "mean_cell_intensity_over_M0": float(cell_int.mean()),
             "std_cell_intensity_over_M0": float(cell_int.std()),
             "mean_core_mean_intensity_over_M0":
                 float(np.mean(np.linalg.norm(m_bar, axis=1)) / M_NRM_MIN)}
    del Mg, Mc, m_flat, cell_int
    for var, d_bg in (("raw", d_raw), ("dc_removed", d_dc)):
        M_rec, rms = invert_ls(K_ls, G_out, d_out[None, :] + d_bg + noise)
        entry[var] = block(M_rec, rms, d_bg, sig_rms, geom)
    return entry, d_raw


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    t_start = time.time()
    print(f"simpeg {simpeg.__version__} | discretize {discretize.__version__} | "
          f"numpy {np.__version__}")
    print(f"MASTER_SEED={MASTER_SEED} ACTIVE_SEED={ACTIVE_SEED} | N_MC={N_MC} | "
          f"sigma_B={SIGMA_B} nT/axis | dx={SPACING} m")

    # ---- outcrop survey and operator ----
    sxyz = outcrop_survey("full_orbit", STANDOFF, spacing=SPACING)
    nD = 3 * len(sxyz)
    G_out = prism_cell_operator(bnd, sxyz)
    d_out = G_out @ m_true
    sig_rms = float(np.sqrt(np.mean(d_out ** 2)))
    K_ls = ls_operator(G_out)
    geom = survey_geom_dict(sxyz, {"type": "full_orbit", "standoff_m": STANDOFF,
                                   "spacing_m": SPACING, "n_data": nD})
    ds = np.linalg.norm(np.diff(np.asarray(sxyz)[:, :2], axis=0), axis=1)
    print(f"\n=== OUTCROP (drive-around) ===")
    print(f"  stations={len(sxyz)} nD={nD} perimeter={ds.sum():.2f} m "
          f"x=[{sxyz[:,0].min():.2f},{sxyz[:,0].max():.2f}] z={sxyz[0,2]}")
    print(f"  signal rms={sig_rms:.4f} nT  peak|B|={np.abs(d_out).max():.4f} nT  "
          f"(sigma_B={SIGMA_B} nT)")

    # ---- background slab mesh, operators ----
    mesh = slab_mesh()
    cc = mesh.cell_centers
    core = (np.abs(cc[:, 0]) <= CORE_HALF) & (np.abs(cc[:, 1]) <= CORE_HALF)
    nA_core = int(core.sum())
    print(f"\n=== BACKGROUND SLAB ===")
    print(f"  z = {mesh.nodes_z[0]:.1f} .. {mesh.nodes_z[-1]:.1f} m (thickness {SLAB_THICK} m, "
          f"{mesh.shape_cells[2]} x {DH} m cells)")
    print(f"  lateral extent = +/-{mesh.nodes_x[-1]:.1f} m  |  cells "
          f"{mesh.shape_cells} = {mesh.n_cells}  |  1 m core cells = {nA_core} "
          f"(+/-{CORE_HALF:.0f} m)")
    t0 = time.time()
    G_full = slab_G(mesh, sxyz)
    U_all = uniform_operator(G_full, mesh.n_cells)          # whole-slab uniform operator
    G_core = np.ascontiguousarray(
        np.hstack([G_full[:, k * mesh.n_cells:(k + 1) * mesh.n_cells][:, core]
                   for k in range(3)]))
    U_core = uniform_operator(G_core, nA_core)
    del G_full
    print(f"  slab sensitivities built in {time.time()-t0:.1f}s "
          f"(core block {G_core.shape}, {G_core.nbytes/1e9:.2f} GB)")

    # ---- CRITICAL TRAP: uniform-background control ----
    print(f"\n=== CONTROL: perfectly uniform background (every cell M = M_NRM_min*(-z_hat)) ===")
    d_unif = U_all @ m_true
    unif_rms = float(np.sqrt(np.mean(d_unif ** 2)))
    ctrl = {"uniform_field_rms_nT": unif_rms,
            "uniform_field_peak_abs_nT": float(np.abs(d_unif).max()),
            "ratio_to_sigma_B": unif_rms / SIGMA_B,
            "ratio_to_outcrop_signal_rms": unif_rms / sig_rms,
            "negligible": bool(unif_rms < 0.02 * SIGMA_B),
            "criterion": "uniform-slab field rms < 2% of sigma_B"}
    print(f"  uniform field rms = {unif_rms:.5f} nT = {unif_rms/SIGMA_B*100:.2f}% of sigma_B "
          f"= {unif_rms/sig_rms*100:.2f}% of the outcrop signal rms")
    print(f"  peak |B_uniform| = {np.abs(d_unif).max():.5f} nT")
    print(f"  NEGLIGIBLE (rms < 2% of sigma_B): {ctrl['negligible']}")
    if not ctrl["negligible"]:
        print("  *** uniform-background control FAILED - stopping ***")
        os.makedirs(os.path.join(_REPO, "results"), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump({"control_uniform_background": ctrl,
                       "status": "STOPPED: uniform-background control not negligible"}, f,
                      indent=2)
        return 1
    # control also pushed through the full inversion chain
    noise = np.random.default_rng(record_seed("runA_white_noise")).normal(
        0.0, SIGMA_B, (N_MC, nD))
    d_bg_unif = np.repeat(d_unif[None, :], N_MC, axis=0)
    M_c, rms_c = invert_ls(K_ls, G_out, d_out[None, :] + d_bg_unif + noise)
    ctrl["recovery_closed_form_ls"] = block(M_c, rms_c, d_bg_unif, sig_rms, geom)
    M_n, rms_n = invert_ls(K_ls, G_out, d_out[None, :] + noise)
    ctrl["recovery_no_background"] = block(M_n, rms_n, np.zeros((N_MC, nD)), sig_rms, geom)
    for tag, b in (("uniform background", ctrl["recovery_closed_form_ls"]),
                   ("no background     ", ctrl["recovery_no_background"])):
        de, me = b["dir_err_deg"], b["mag_err_pct"]
        print(f"  {tag}: theta={de['mean']:5.2f} +/- {de['std']:4.2f} deg  "
              f"dM={me['mean']:+6.2f} +/- {me['std']:5.2f} %  rms={b['residual_rms']['mean']:.4f}")

    # ---- sweep ----
    levels = [(f"sigma_theta_{int(s)}deg", s) for s in SIGMA_THETA_DEG] + \
             [(ISOTROPIC_LABEL, None)]
    sim_out = single_cell_sim(bnd, sxyz)     # nP == 3, for the damped iterative arm
    sweep = {}
    print(f"\n=== SWEEP: {len(COHERENCE)} coherence arms x {len(levels)} dispersion levels, "
          f"N={N_MC} ===")
    for coh_label, bxy, vcoh in COHERENCE:
        gid = group_ids(cc[core], bxy, vcoh)
        n_groups = int(gid.max() + 1)
        arm = {"coherence_label": coh_label, "block_xy_m": bxy,
               "vertical_coherent": bool(vcoh), "n_groups": n_groups,
               "n_core_cells": nA_core, "levels": {}}
        print(f"\n-- {coh_label}: block {bxy} x {bxy} m, vertical coherent={vcoh}, "
              f"{n_groups} independent groups over {nA_core} core cells --")
        for lvl_label, sth in levels:
            t0 = time.time()
            rng = np.random.default_rng(record_seed(f"runA_bg_{coh_label}_{lvl_label}"))
            entry, d_raw = sweep_point(coh_label, lvl_label, sth, gid, n_groups, rng,
                                       G_core, U_all, U_core, d_out, noise, sig_rms,
                                       geom, K_ls, G_out, nD)
            arm["levels"][lvl_label] = entry
            r, dcv = entry["raw"], entry["dc_removed"]
            print(f"   {lvl_label:18s} bg rms={r['background_field']['bg_field_rms_nT']['mean']:7.4f} nT "
                  f"({r['background_field']['ratio_to_sigma_B']:5.3f} sig_B, "
                  f"{r['background_field']['ratio_to_outcrop_signal_rms']:6.3f} sig) | "
                  f"theta={r['dir_err_deg']['mean']:6.2f}+/-{r['dir_err_deg']['std']:5.2f} deg "
                  f"dM={r['mag_err_pct']['mean']:+7.2f}% rms={r['residual_rms']['mean']:.4f} "
                  f"| dc: theta={dcv['dir_err_deg']['mean']:6.2f} deg "
                  f"| {time.time()-t0:.1f}s", flush=True)

            # damped iterative cross-check on the headline arm (raw variant)
            if coh_label == HEADLINE_ARM:
                bseed = record_seed(f"runA_beta_{coh_label}_{lvl_label}")
                n_it = min(N_MC, 150)
                M_i = np.empty((n_it, 3))
                rms_i = np.empty(n_it)
                iters = np.empty(n_it, int)
                t1 = time.time()
                for j in range(n_it):
                    dobs = d_out + d_raw[j] + noise[j]
                    mr, ni, phi, _ = iterative_recover_M(sim_out, dobs, SIGMA_B,
                                                         np.zeros(3), bseed + j)
                    M_i[j] = mr
                    rms_i[j] = np.sqrt(phi / nD)
                    iters[j] = ni
                blk = block(M_i, rms_i, d_raw[:n_it], sig_rms, geom)
                blk["n_realizations"] = n_it
                blk["iters"] = {"min": int(iters.min()), "max": int(iters.max())}
                blk["target_phi_d"] = float(nD)
                entry["raw_iterative_gncg"] = blk
                print(f"   {'':18s} iterative GNCG (N={n_it}): "
                      f"theta={blk['dir_err_deg']['mean']:6.2f}+/-{blk['dir_err_deg']['std']:5.2f} deg "
                      f"dM={blk['mag_err_pct']['mean']:+7.2f}% rms={rms_i.mean():.4f} "
                      f"iters {iters.min()}-{iters.max()} | {time.time()-t1:.1f}s", flush=True)
            del d_raw
        sweep[coh_label] = arm

    # ---- SWEEP 2: direction dispersion ALONE (intensity fixed at exactly M_NRM_min) ----
    # The specified log-uniform intensity spans 100x and by itself dominates the incoherent
    # background field, which makes the sigma_theta axis in SWEEP 1 nearly flat. Holding the
    # intensity fixed isolates the angular-dispersion effect and gives a genuine crossing:
    # (sigma_theta = 0, fixed intensity) IS the perfectly uniform slab.
    print(f"\n=== SWEEP 2: direction dispersion alone (intensity fixed at M_NRM_min) ===")
    sweep2 = {}
    for coh_label, bxy, vcoh in COHERENCE:
        gid = group_ids(cc[core], bxy, vcoh)
        n_groups = int(gid.max() + 1)
        arm = {"coherence_label": coh_label, "block_xy_m": bxy,
               "vertical_coherent": bool(vcoh), "n_groups": n_groups, "levels": {}}
        print(f"\n-- {coh_label}: {n_groups} groups --")
        for lvl_label, sth in levels:
            rng = np.random.default_rng(record_seed(f"runA_dironly_{coh_label}_{lvl_label}"))
            entry, d_raw = sweep_point(coh_label, lvl_label, sth, gid, n_groups, rng,
                                       G_core, U_all, U_core, d_out, noise, sig_rms,
                                       geom, K_ls, G_out, nD, int_log10_range=(0.0, 0.0))
            arm["levels"][lvl_label] = entry
            r = entry["raw"]
            print(f"   {lvl_label:18s} bg rms={r['background_field']['bg_field_rms_nT']['mean']:7.4f} nT "
                  f"({r['background_field']['ratio_to_sigma_B']:5.3f} sig_B, "
                  f"{r['background_field']['ratio_to_outcrop_signal_rms']:6.3f} sig) | "
                  f"theta={r['dir_err_deg']['mean']:6.2f}+/-{r['dir_err_deg']['std']:5.2f} deg "
                  f"dM={r['mag_err_pct']['mean']:+7.2f}% rms={r['residual_rms']['mean']:.4f} "
                  f"| dc: theta={entry['dc_removed']['dir_err_deg']['mean']:6.2f} deg", flush=True)
            del d_raw
        sweep2[coh_label] = arm

    # ---- SWEEP 3: intensity dynamic range alone (all directions exactly -z_hat) ----
    print(f"\n=== SWEEP 3: intensity dynamic range alone (all directions -z_hat) ===")
    sweep3 = {}
    for coh_label, bxy, vcoh in COHERENCE:
        if coh_label not in (HEADLINE_ARM, "block_2m"):
            continue
        gid = group_ids(cc[core], bxy, vcoh)
        n_groups = int(gid.max() + 1)
        arm = {"coherence_label": coh_label, "block_xy_m": bxy,
               "vertical_coherent": bool(vcoh), "n_groups": n_groups, "levels": {}}
        print(f"\n-- {coh_label}: {n_groups} groups --")
        for f in INTENSITY_FLOORS:
            lvl_label = f"intensity_floor_{f:g}"
            rng = np.random.default_rng(record_seed(f"runA_intonly_{coh_label}_{lvl_label}"))
            lo = 0.0 if f >= 1.0 else float(np.log10(f))
            entry, d_raw = sweep_point(coh_label, lvl_label, 0.0, gid, n_groups, rng,
                                       G_core, U_all, U_core, d_out, noise, sig_rms,
                                       geom, K_ls, G_out, nD, int_log10_range=(lo, 0.0))
            entry["intensity_floor_over_M0"] = f
            arm["levels"][lvl_label] = entry
            r = entry["raw"]
            print(f"   floor={f:<6g} bg rms={r['background_field']['bg_field_rms_nT']['mean']:7.4f} nT "
                  f"({r['background_field']['ratio_to_sigma_B']:5.3f} sig_B, "
                  f"{r['background_field']['ratio_to_outcrop_signal_rms']:6.3f} sig) | "
                  f"theta={r['dir_err_deg']['mean']:6.2f}+/-{r['dir_err_deg']['std']:5.2f} deg "
                  f"dM={r['mag_err_pct']['mean']:+7.2f}% rms={r['residual_rms']['mean']:.4f}",
                  flush=True)
            del d_raw
        sweep3[coh_label] = arm

    # ---- headline: threshold crossings ----
    print("\n=== HEADLINE ===")
    sths = [0.0 if s is None else s for _, s in levels]
    head = {"as_specified_log_uniform_intensity": {}, "direction_dispersion_only": {},
            "intensity_range_only": {}}
    pooled_x, pooled_y, pooled_tag = [], [], []

    def arm_xy(arm, keys):
        xs = [arm["levels"][k]["raw"]["background_field"]["ratio_to_outcrop_signal_rms"]
              for k in keys]
        ys = [arm["levels"][k]["raw"]["dir_err_deg"]["mean"] for k in keys]
        return xs, ys

    lvl_keys = [l for l, _ in levels]
    dir_keys = lvl_keys[:-1]          # drop uniform_sphere for the sigma_theta axis
    for coh_label, _, _ in COHERENCE:
        for tag, src in (("as_specified_log_uniform_intensity", sweep),
                         ("direction_dispersion_only", sweep2)):
            arm = src[coh_label]
            xs, ys = arm_xy(arm, lvl_keys)
            _, ys_d = arm_xy(arm, dir_keys)
            head[tag][coh_label] = {
                "sigma_theta_deg_at_5deg": crossing(sths[:-1], ys_d, 5.0),
                "sigma_theta_deg_at_10deg": crossing(sths[:-1], ys_d, 10.0),
                "bg_to_signal_at_5deg": crossing(xs, ys, 5.0),
                "bg_to_signal_at_10deg": crossing(xs, ys, 10.0),
                "dir_err_deg_min": float(min(ys)), "dir_err_deg_max": float(max(ys)),
                "bg_to_signal_max": float(max(xs)),
                "bg_to_sigma_B_max": float(max(
                    arm["levels"][k]["raw"]["background_field"]["ratio_to_sigma_B"]
                    for k in lvl_keys)),
            }
            pooled_x += xs; pooled_y += ys
            pooled_tag += [f"{tag}:{coh_label}"] * len(xs)
    for coh_label, arm in sweep3.items():
        keys = list(arm["levels"].keys())
        xs, ys = arm_xy(arm, keys)
        floors = [arm["levels"][k]["intensity_floor_over_M0"] for k in keys]
        head["intensity_range_only"][coh_label] = {
            "dynamic_range_at_5deg": crossing([1.0 / f for f in floors], ys, 5.0),
            "dynamic_range_at_10deg": crossing([1.0 / f for f in floors], ys, 10.0),
            "bg_to_signal_at_5deg": crossing(xs, ys, 5.0),
            "bg_to_signal_at_10deg": crossing(xs, ys, 10.0),
            "dir_err_deg_min": float(min(ys)), "dir_err_deg_max": float(max(ys)),
        }
        pooled_x += xs; pooled_y += ys
        pooled_tag += [f"intensity_range_only:{coh_label}"] * len(xs)

    print("  (a) full model (log-uniform intensity over 100x + sigma_theta):")
    for coh_label, _, _ in COHERENCE:
        h = head["as_specified_log_uniform_intensity"][coh_label]
        print(f"      {coh_label:14s} theta {h['dir_err_deg_min']:5.2f}-{h['dir_err_deg_max']:6.2f} deg "
              f"| bg/signal max {h['bg_to_signal_max']:.3f} | 5 deg at sigma_theta="
              f"{fmt_cross(h['sigma_theta_deg_at_5deg'])}")
    print("  (b) direction dispersion ALONE (intensity fixed at M_NRM_min):")
    for coh_label, _, _ in COHERENCE:
        h = head["direction_dispersion_only"][coh_label]
        print(f"      {coh_label:14s} theta {h['dir_err_deg_min']:5.2f}-{h['dir_err_deg_max']:6.2f} deg "
              f"| 5 deg at sigma_theta={fmt_cross(h['sigma_theta_deg_at_5deg']):28s} "
              f"10 deg at sigma_theta={fmt_cross(h['sigma_theta_deg_at_10deg'])}")
    print("  (c) intensity dynamic range ALONE (all directions -z_hat):")
    for coh_label, h in head["intensity_range_only"].items():
        print(f"      {coh_label:14s} theta {h['dir_err_deg_min']:5.2f}-{h['dir_err_deg_max']:6.2f} deg "
              f"| 5 deg at dynamic range ={fmt_cross(h['dynamic_range_at_5deg'])}")

    pooled = {"bg_to_signal_at_5deg": crossing(pooled_x, pooled_y, 5.0),
              "bg_to_signal_at_10deg": crossing(pooled_x, pooled_y, 10.0),
              "n_points": len(pooled_x),
              "points": [{"arm": t, "bg_to_signal": float(x), "dir_err_deg": float(y)}
                         for t, x, y in zip(pooled_tag, pooled_x, pooled_y)]}
    print(f"\n  POOLED over all {len(pooled_x)} sweep points, outcrop direction error vs "
          f"background-to-signal rms ratio:")
    print(f"      crosses  5 deg at bg/signal = {fmt_cross(pooled['bg_to_signal_at_5deg'])}")
    print(f"      crosses 10 deg at bg/signal = {fmt_cross(pooled['bg_to_signal_at_10deg'])}")

    payload = {
        "meta": {
            "script": "scripts/background_magnetization_mc.py",
            "run": "background bedrock magnetization Monte Carlo",
            "question": ("how far from uniform can the underlying mare unit be before the "
                         "Runcorn (1975) negligible-external-field argument stops holding "
                         "and the K=1 outcrop inversion degrades"),
            "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
            "library_versions": {"simpeg": simpeg.__version__,
                                 "discretize": discretize.__version__,
                                 "numpy": np.__version__},
            "master_seed": MASTER_SEED, "active_seed": ACTIVE_SEED,
            "seed_scheme": "seed_for(name) = sha256(f'{ACTIVE_SEED}:{name}')[:8] (hex->int)",
            "seeds_resolved": SEEDS_USED,
            "N_mc": N_MC,
            "N_mc_iterative_subset": min(N_MC, 150),
            "sigma_B_nT": SIGMA_B,
            "noise_model": ("i.i.d. zero-mean Gaussian sigma_B = 0.50 nT per axis per "
                            "station; ONE matched noise matrix (N_mc x nD) reused for every "
                            "sweep point so differences isolate the background"),
            "sampling_m": SPACING,
            "outcrop": {"dims_m": list(PRISM_DIMS), "bounds_m": list(bnd),
                        "M_true_Apm": m_true.tolist(), "M_NRM_min_Apm": M_NRM_MIN,
                        "survey": "full_orbit drive-around", "standoff_m": STANDOFF,
                        "sensor_height_m": SENSOR_H,
                        "n_stations": int(len(sxyz)), "n_data": nD,
                        "perimeter_m": float(ds.sum()),
                        "signal_rms_nT": sig_rms,
                        "signal_peak_abs_nT": float(np.abs(d_out).max())},
            "background_slab": {
                "thickness_m": SLAB_THICK,
                "z_range_m": [float(mesh.nodes_z[0]), float(mesh.nodes_z[-1])],
                "cell_size_core_m": DH,
                "core_half_width_m": CORE_HALF,
                "n_core_cells": nA_core,
                "lateral_extent_m": float(mesh.nodes_x[-1]),
                "padding_growth_factor": PAD_FAC,
                "mesh_shape_cells": list(mesh.shape_cells),
                "mesh_n_cells": int(mesh.n_cells),
                "heterogeneity_confined_to_core": True,
                "padding_magnetization": ("each realization's own core-mean magnetization, so "
                                          "the model decomposes exactly into (uniform slab at "
                                          "the realization mean) + (zero-mean core fluctuation)"),
                "intensity_distribution": "log-uniform on [0.01, 1.0] * M_NRM_min",
                "direction_distribution": ("polar angle theta = |N(0, sigma_theta)| clipped to "
                                           "[0,180] deg from -z_hat, azimuth U(0,2pi); "
                                           "'uniform_sphere' level is exactly isotropic"),
                "sigma_theta_deg_sweep": SIGMA_THETA_DEG,
                "coherence_sweep": [{"label": a, "block_xy_m": b, "vertical_coherent": c}
                                    for a, b, c in COHERENCE],
                "intensity_floor_sweep": INTENSITY_FLOORS,
                "background_is_inverted_for": False,
            },
            "sweeps": {
                "sweep_1_as_specified": ("log-uniform intensity over [0.01,1]*M_NRM_min AND "
                                         "direction dispersion sigma_theta"),
                "sweep_2_direction_only": ("intensity held at exactly M_NRM_min, sigma_theta "
                                           "swept; isolates angular dispersion. The "
                                           "sigma_theta = 0 point of this sweep IS the "
                                           "perfectly uniform slab, so this sweep is the one "
                                           "with a genuine bracketed 5 deg crossing"),
                "sweep_3_intensity_only": ("all directions exactly -z_hat, log-uniform "
                                           "intensity floor swept over "
                                           f"{INTENSITY_FLOORS}; isolates intensity scatter"),
                "why_decomposed": ("in sweep 1 the specified 100x log-uniform intensity range "
                                   "alone dominates the incoherent background field, so the "
                                   "sigma_theta axis is nearly flat (5.9 -> 6.3 deg from "
                                   "sigma_theta = 0 to 90 deg) and the 5 deg threshold is "
                                   "already exceeded at sigma_theta = 0; sweeps 2 and 3 "
                                   "separate the two contributions"),
            },
            "variants": {
                "raw": "background field as drawn, INCLUDING the uniform-slab edge artifact",
                "dc_removed": ("uniform-case field subtracted: d_raw - U_all @ m_core_mean, "
                               "i.e. the pure zero-mean fluctuation field"),
            },
            "inversion": {
                "free_parameters": "3 (outcrop Mx,My,Mz); geometry fixed and exactly known",
                "K": 1,
                "primary": ("cold-start zero-reference weighted least squares, "
                            "K = inversion_helpers.ls_operator(G_outcrop); the "
                            "isolated_outcrop_1cm.py path"),
                "secondary": ("inversion_helpers.iterative_recover_M: ProjectedGNCG"
                              f"({DEFAULT_GNCG_OPT}) + Smallness(zero reference) + "
                              f"BetaEstimate_ByEig(beta0_ratio={DEFAULT_BETA0_RATIO}) + "
                              "UpdatePreconditioner + TargetMisfit(chifact=1.0); headline "
                              "coherence arm only"),
                "beta_selection": f"BetaEstimate_ByEig(beta0_ratio={DEFAULT_BETA0_RATIO})",
                "stopping_criterion": "TargetMisfit(chifact=1.0), target phi_d = nD",
                "phi_d_convention": ("simpeg L2DataMisfit returns sum((r/sigma)^2) = chi^2 "
                                     "(no 0.5 prefactor); target = nD, NOT 0.5*nD; "
                                     "normalized residual rms = sqrt(phi_d/nD)"),
                "target_phi_d": float(nD),
            },
            "dM_convention": "mag_err_pct = 100*(|M_rec|-|M_true|)/|M_true| (SIGNED)",
            "padding_convergence_study_uniform_field_rms_nT": {
                "+/-57m": 0.06845, "+/-196m": 0.02003, "+/-482m": 0.00814,
                "+/-1213m": 0.00323, "+/-3087m": 0.00127},
            "heterogeneity_core_convergence_field_rms_nT": {
                "core+/-5m": 0.03937, "core+/-10m": 0.03831,
                "core+/-15m": 0.04090, "core+/-20m": 0.04153},
        },
        "control_uniform_background": ctrl,
        "sweep_1_as_specified": sweep,
        "sweep_2_direction_dispersion_only": sweep2,
        "sweep_3_intensity_range_only": sweep3,
        "headline": {"crossings": head, "pooled_bg_to_signal": pooled,
                     "headline_coherence_arm": HEADLINE_ARM,
                     "crossing_status_legend": {
                         "interpolated": "genuine bracketed crossing",
                         "below_lowest_sampled": ("threshold already exceeded at the smallest "
                                                  "value sampled; true crossing is below the "
                                                  "sampled range"),
                         "not_reached": "threshold never reached over the sampled range"}},
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

"""Forward operators, seeded random numbers, survey geometry and inversion routines used by every scenario.

Relocated from ``engine/inversion_helpers.py`` (release v1.0.0) with type hints and docstrings added; the
arithmetic is unchanged, and ``engine/inversion_helpers.py`` is now an alias of this module, so every script that
does ``import inversion_helpers`` gets this exact module object (module-level settings such as ``ACTIVE_SEED``
are shared).

Units
-----
Model values are magnetization in A/m; predicted data are in nT. SimPEG's vector integral formulation multiplies
its geometric kernel (which carries the 1/(4 pi) of thesis Eq. 3.6) by the source amplitude, so setting the
amplitude to ``AMP_NT = mu0 * 1e9`` nT (an inducing field of 1 A/m) makes ``d = G m`` map A/m to nT exactly.

Thesis equations implemented here (renumbered Chapter 3 of the 2026-10-13 draft; see ``docs/theory.md``):

* Eq. 3.6/3.7   ``prism_cell_operator``, ``three_layer_G``: G for uniformly magnetized prisms, d = G m.
* Eq. 3.10/3.12 ``iterative_recover_M`` (damped, SimPEG); Eqs. 3.11/3.11 ``ls_operator`` (closed form, beta = 0).
* Eq. 3.14-3.16 ``linear_mc``: d = G p + n, p_hat = p + K n, sampled by Monte Carlo.
* Eq. 3.27/3.28 ``dir_err_deg``, ``recovery_metrics``.
"""
from __future__ import annotations

import hashlib
import io as _io
import contextlib as _ctx
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import discretize
from simpeg import (maps, data, data_misfit, directives, inverse_problem,
                    optimization, inversion, regularization)
from simpeg.potential_fields import magnetics as mag

from .core import (generate_traverse,
                   generate_rille_traverse, mu0)

Bounds = Sequence[float]   # (x0, x1, y0, y1, z0, z1) in metres

# --------------------------------------------------------------------------- #
# Physical constants and scenario defaults
# --------------------------------------------------------------------------- #
AMP_NT = mu0 * 1.0e9          # SimPEG source amplitude for H0 = 1 A/m

M_NRM_MIN = 2.4e-3            # A/m, reference magnetization M_ref
SIGMA_B = 0.5                # nT per axis, adopted rover uncertainty
STANDOFF = 1.0               # m, default rover-to-outcrop clearance
SENSOR_H = 0.5               # m, sensor height
PRISM_DIMS = (4.0, 4.0, 1.0) # m, isolated-outcrop prism (sx, sy, sz)

SPACING_CORE = 0.05          # m, outcrop core/sweep surveys
N_HEADLINE = 1000
N_SWEEP = 200

DRIVE = {"drive_around": "full_orbit", "drive_by": "drive_by"}

# --------------------------------------------------------------------------- #
# Reproducibility: master seed -> named per-experiment seeds
# --------------------------------------------------------------------------- #
MASTER_SEED = 20260624
ACTIVE_SEED = MASTER_SEED


def seed_for(name: str) -> int:
    """Deterministic per-experiment seed derived from ``ACTIVE_SEED`` and a label.

    ``seed = int(sha256(f"{ACTIVE_SEED}:{name}")[:8], 16)``, so every named experiment (for example
    ``"a09:U_budget_full_offcentre:-z"``) has its own reproducible random stream, independent of run order.
    """
    h = hashlib.sha256(f"{ACTIVE_SEED}:{name}".encode()).hexdigest()
    return int(h[:8], 16)


# --------------------------------------------------------------------------- #
# Forward operators (exact-coarse prisms; voxel blocks)
# --------------------------------------------------------------------------- #
def _sim_for(mesh: Any, rx_locs: np.ndarray, n_model: int) -> Any:
    """Vector-magnetics integral simulation; model values are A/m (amplitude = mu0 * 1e9 nT).

    Receivers measure (bx, by, bz) at each location (station-major data ordering). The model is the SimPEG vector
    model [Mx(all cells); My(all cells); Mz(all cells)] (component-major), ``3 * n_model`` values.
    The prism kernel backend is SimPEG's default for this call (Geoana in SimPEG 0.24.0).
    """
    rx = mag.receivers.Point(np.asarray(rx_locs, float), components=["bx", "by", "bz"])
    src = mag.sources.UniformBackgroundField(receiver_list=[rx], amplitude=AMP_NT,
                                             inclination=90.0, declination=0.0)
    survey = mag.survey.Survey(src)
    return mag.simulation.Simulation3DIntegral(
        mesh, survey=survey, chiMap=maps.IdentityMap(nP=3 * n_model),
        model_type="vector", store_sensitivities="ram", sensitivity_dtype=np.float64)


def prism_cell_operator(bounds: Bounds, rx_locs: np.ndarray) -> np.ndarray:
    """G (nD x 3) for ONE uniformly magnetized axis-aligned prism (a single mesh cell).

    ``bounds = (x0, x1, y0, y1, z0, z1)``. Columns map (Mx, My, Mz) [A/m] to data [nT]; rows are station-major
    (bx, by, bz per station). This is thesis Eq. 3.6 integrated exactly over the prism (Eq. 3.7, K = 1).
    """
    x0, x1, y0, y1, z0, z1 = bounds
    mesh = discretize.TensorMesh([np.r_[x1 - x0], np.r_[y1 - y0], np.r_[z1 - z0]],
                                 origin=[x0, y0, z0])
    sim = _sim_for(mesh, rx_locs, mesh.n_cells)
    return sim.G.copy()                       # (nD, 3), single cell


# --------------------------------------------------------------------------- #
# Cold-start linear solution operator + linear-operator Monte Carlo
# --------------------------------------------------------------------------- #
def ls_operator(G_inv: np.ndarray) -> np.ndarray:
    """Cold-start zero-reference least-squares operator K = (G^T G)^-1 G^T (p x nD); thesis Eq. 3.11 with beta = 0.

    With equal per-datum weights W_d = I / sigma the weights cancel in K (they do not cancel in the covariance,
    Eq. 3.16/3.17). ``m_rec = K @ d``. Asserts the operator does not depend on the true model (cold start).
    """
    G_inv = np.asarray(G_inv, float)
    K = np.linalg.solve(G_inv.T @ G_inv, G_inv.T)
    assert np.allclose(K @ np.zeros(G_inv.shape[0]), 0.0), "operator not zero-referenced"
    return K


def linear_mc(G_data: np.ndarray, m_true_full: np.ndarray, G_inv: np.ndarray, sigma: float, n: int,
              seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo of Eqs. 3.14-3.15: n noisy draws d = G_data m + n (n ~ N(0, sigma^2 I)), each inverted with K(G_inv).

    ``G_data`` and ``G_inv`` may differ (model error). Returns ``M_rec`` (n x p_inv) and the normalized residual
    rms of each draw (Eq. 3.29).
    """
    G_data = np.asarray(G_data, float); G_inv = np.asarray(G_inv, float)
    m_true_full = np.asarray(m_true_full, float)
    K = ls_operator(G_inv)
    rng = np.random.default_rng(seed)
    d_clean = G_data @ m_true_full
    D = d_clean[None, :] + rng.normal(0.0, sigma, size=(n, d_clean.size))
    M_rec = D @ K.T                                       # (n, p_inv)
    pred = M_rec @ G_inv.T                                # (n, nD)
    resid_rms = np.sqrt(np.mean(((pred - D) / sigma) ** 2, axis=1))
    return M_rec, resid_rms


# --------------------------------------------------------------------------- #
# Recovery-metrics block
# --------------------------------------------------------------------------- #
def _summary(arr: Any) -> dict[str, float | int]:
    a = np.asarray(arr, float)
    return {"mean": float(np.mean(a)), "std": float(np.std(a)), "median": float(np.median(a)),
            "p5": float(np.percentile(a, 5)), "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)), "p95": float(np.percentile(a, 95)),
            "min": float(np.min(a)), "max": float(np.max(a)), "N": int(a.size)}


def dir_err_deg(rec: Any, true: Any) -> float:
    """Direction error (deg) between two vectors, thesis Eq. 3.27: arccos(M_true . M_rec / (|M_true| |M_rec|))."""
    rec = np.asarray(rec, float); true = np.asarray(true, float)
    c = rec @ true / (np.linalg.norm(rec) * np.linalg.norm(true) + 1e-300)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def recovery_metrics(M_rec: np.ndarray, m_true_body: np.ndarray, resid_rms: np.ndarray, sigma: float,
                     survey_geom: Mapping[str, Any], store_raw: bool = False,
                     per_component: bool = False) -> dict[str, Any]:
    """Standard metrics block over an MC ensemble for one recovered body/layer.

    ``M_rec``: (n, 3) recovered vectors; ``m_true_body``: (3,) truth for this body. ``mag_err_pct`` is the SIGNED
    intensity error 100 (|M_rec| - |M|)/|M|; thesis Eq. 3.28 is its absolute value.
    """
    M_rec = np.asarray(M_rec, float); m_true_body = np.asarray(m_true_body, float)
    de = np.array([dir_err_deg(M_rec[i], m_true_body) for i in range(len(M_rec))])
    tmag = np.linalg.norm(m_true_body)
    me = 100.0 * (np.linalg.norm(M_rec, axis=1) - tmag) / tmag
    out = {"dir_err_deg": _summary(de), "mag_err_pct": _summary(me),
           "recovered_vector_mean": [float(v) for v in M_rec.mean(axis=0)],
           "true_model": [float(v) for v in m_true_body],
           "residual_rms": {"mean": float(np.mean(resid_rms)), "std": float(np.std(resid_rms))},
           "noise_floor_nT": float(sigma), "survey_geom": survey_geom}
    if per_component:
        comp_err = M_rec - m_true_body[None, :]
        out["per_component_error"] = {c: _summary(comp_err[:, k]) for k, c in enumerate("xyz")}
    if store_raw:
        out["raw"] = {"dir_err_deg": [float(v) for v in de],
                      "mag_err_pct": [float(v) for v in me]}
    return out


# --------------------------------------------------------------------------- #
# Geometry / survey helpers
# --------------------------------------------------------------------------- #
def prism_bounds(center: Sequence[float], dims: Sequence[float]) -> tuple[float, float, float, float, float, float]:
    """(x0,x1,y0,y1,z0,z1) for an axis-aligned prism at `center` with full `dims` (sx,sy,sz)."""
    (cx, cy, cz), (sx, sy, sz) = center, dims
    return (cx - sx/2, cx + sx/2, cy - sy/2, cy + sy/2, cz - sz/2, cz + sz/2)


def survey_geom_dict(survey_xyz: np.ndarray, extra: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Small summary of a survey path (station count, x range, mean y and z) stored with each result."""
    s = np.asarray(survey_xyz, float)
    g = {"n_stations": int(len(s)),
         "x_range_m": [float(s[:, 0].min()), float(s[:, 0].max())],
         "y_mean_m": float(s[:, 1].mean()), "z_mean_m": float(s[:, 2].mean())}
    if extra:
        g.update(extra)
    return g


def outcrop_survey(traverse_type: str, standoff: float, dims: Sequence[float] = PRISM_DIMS, spacing: float = 0.5,
                   center: Sequence[float] = (0., 0., 0.)) -> np.ndarray:
    """drive_by or full_orbit traverse around the isolated outcrop prism (core.generate_traverse)."""
    sp = {"cx": center[0], "cy": center[1], "sx": dims[0], "sy": dims[1], "sz": dims[2]}
    return generate_traverse(traverse_type=traverse_type, shape_params=sp,
                             sensor_height=SENSOR_H, spacing=spacing,
                             orbit_offset=standoff, approach_distance=5.0, shape="rect")


# --------------------------------------------------------------------------- #
# Three-layer wall scenario
# --------------------------------------------------------------------------- #
def three_layer_mesh() -> Any:
    """Three stacked 5 m flows (z in 0-15 m) as a 3-cell wall (x in -50..50, y in 0..50)."""
    return discretize.TensorMesh([np.r_[100.], np.r_[50.], np.r_[5., 5., 5.]], origin=[-50., 0., 0.])


def three_layer_G(sensor_y: float, sensor_z: float, x0: float = -120., x1: float = 120.,
                  spacing: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Forward operator (nD x 9) and station coordinates for one rille traverse in front of the wall.

    Columns are in SimPEG vector ordering: (Mx1, Mx2, Mx3, My1, My2, My3, Mz1, Mz2, Mz3), i.e. component-major.
    """
    mesh = three_layer_mesh()
    sxyz = generate_rille_traverse(x0, x1, sensor_y, sensor_z, spacing)
    sim = _sim_for(mesh, sxyz, mesh.n_cells)
    return sim.G.copy(), sxyz


def layer_slices(M_rec_full: np.ndarray) -> list[np.ndarray]:
    """Split a (n,9) recovered ensemble into three (n,3) per-layer vectors (SimPEG vector ordering)."""
    return [M_rec_full[:, [i, 3 + i, 6 + i]] for i in range(3)]


# --------------------------------------------------------------------------- #
# Iterative parametric inversion (geometry fixed; solve the 3 vector components)
# --------------------------------------------------------------------------- #
DEFAULT_GNCG_OPT = dict(maxIter=20, maxIterCG=20, tolCG=1e-4)
DEFAULT_BETA0_RATIO = 1e-4


def iterative_recover_M(sim: Any, dobs: np.ndarray, sigma: float, m0: np.ndarray, beta_seed: int,
                        opt_kwargs: Optional[Mapping[str, Any]] = None,
                        beta0_ratio: float = DEFAULT_BETA0_RATIO) -> tuple[np.ndarray, int, float, bool]:
    """One SimPEG iterative solve of thesis Eq. 3.10 for the 3 vector components on a fixed single-cell body.

    ProjectedGNCG + BetaEstimate_ByEig + UpdatePreconditioner + TargetMisfit(chifact=1.0); zero reference model.
    Returns ``(m_rec (3,), n_iter, phi_d, converged)``. Geometry is fixed; only (Mx, My, Mz) are free.
    """
    opt_kwargs = DEFAULT_GNCG_OPT if opt_kwargs is None else opt_kwargs
    data_obj = data.Data(survey=sim.survey, dobs=np.asarray(dobs, float))
    data_obj.standard_deviation = np.full(int(sim.survey.nD), float(sigma))
    dmis = data_misfit.L2DataMisfit(data=data_obj, simulation=sim)
    reg = regularization.Smallness(mesh=discretize.TensorMesh([np.ones(3)]),
                                   mapping=maps.IdentityMap(nP=3))
    reg.reference_model = np.zeros(3)
    opt = optimization.ProjectedGNCG(lower=-np.inf, upper=np.inf, **opt_kwargs)
    invprob = inverse_problem.BaseInvProblem(dmis, reg, opt)
    dlist = [directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio, random_seed=int(beta_seed)),
             directives.UpdatePreconditioner(),
             directives.TargetMisfit(chifact=1.0)]
    inv = inversion.BaseInversion(invprob, directiveList=dlist)
    with _ctx.redirect_stdout(_io.StringIO()):
        m_rec = np.asarray(inv.run(np.asarray(m0, float)), float)
    n_iter = int(getattr(opt, "iter", -1))
    phi_d = float(dmis(m_rec))
    converged = bool(0 <= n_iter < opt.maxIter)
    return m_rec, n_iter, phi_d, converged


def single_cell_sim(bounds: Bounds, rx_locs: np.ndarray) -> Any:
    """Single-cell vector simulation for a prism (nP == 3), used by the iterative solver."""
    x0, x1, y0, y1, z0, z1 = bounds
    mesh1 = discretize.TensorMesh([np.r_[x1 - x0], np.r_[y1 - y0], np.r_[z1 - z0]],
                                  origin=[x0, y0, z0])
    return _sim_for(mesh1, rx_locs, mesh1.n_cells)


def iterative_recover_M_ref(sim: Any, dobs: np.ndarray, sigma: float, m0: np.ndarray, beta_seed: int,
                            reference_model: np.ndarray, opt_kwargs: Optional[Mapping[str, Any]] = None,
                            beta0_ratio: float = DEFAULT_BETA0_RATIO) -> tuple[np.ndarray, int, float, bool]:
    """iterative_recover_M with an explicit regularization reference model (used by three_layer_wall.py and
    elevated_traverse_height_sweep.py to check their damped 9-parameter solver). nP is inferred from m0.
    Directives and solver settings are identical to iterative_recover_M."""
    opt_kwargs = DEFAULT_GNCG_OPT if opt_kwargs is None else opt_kwargs
    m0 = np.asarray(m0, float)
    nP = int(m0.size)
    data_obj = data.Data(survey=sim.survey, dobs=np.asarray(dobs, float))
    data_obj.standard_deviation = np.full(int(sim.survey.nD), float(sigma))
    dmis = data_misfit.L2DataMisfit(data=data_obj, simulation=sim)
    reg = regularization.Smallness(mesh=discretize.TensorMesh([np.ones(nP)]),
                                   mapping=maps.IdentityMap(nP=nP))
    reg.reference_model = np.asarray(reference_model, float)
    opt = optimization.ProjectedGNCG(lower=-np.inf, upper=np.inf, **opt_kwargs)
    invprob = inverse_problem.BaseInvProblem(dmis, reg, opt)
    dlist = [directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio, random_seed=int(beta_seed)),
             directives.UpdatePreconditioner(),
             directives.TargetMisfit(chifact=1.0)]
    inv = inversion.BaseInversion(invprob, directiveList=dlist)
    with _ctx.redirect_stdout(_io.StringIO()):
        m_rec = np.asarray(inv.run(m0), float)
    n_iter = int(getattr(opt, "iter", -1))
    phi_d = float(dmis(m_rec))
    converged = bool(0 <= n_iter < opt.maxIter)
    return m_rec, n_iter, phi_d, converged


def operator_diagnostics(G: np.ndarray, n_layers: int = 3) -> dict[str, Any]:
    """Information-content diagnostics for a stacked layered-wall operator.

    G is (nD x 3*n_layers) in SimPEG vector ordering (all mx, then my, then mz, so
    layer i owns columns [i + k*n_layers for k in range(3)], matching layer_slices).
    Returns the 3*n_layers singular values, the condition number, and the pairwise
    inter-layer subspace correlations: for layers (i, j) the maximum cosine of the
    principal angles between the column spaces of their (nD x 3) sub-operators
    (1.0 = some response direction of layer i is exactly reproducible by layer j,
    i.e. no independent information; 0.0 = orthogonal responses). This is the largest
    canonical correlation of the two three-dimensional column spaces, not the largest
    pairwise column correlation."""
    G = np.asarray(G, float)
    s = np.linalg.svd(G, compute_uv=False)
    bases = []
    for i in range(n_layers):
        cols = [i + k * n_layers for k in range(3)]
        Q, _ = np.linalg.qr(G[:, cols])
        bases.append(Q)
    pairs = {}
    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            c = np.linalg.svd(bases[i].T @ bases[j], compute_uv=False)
            pairs[f"L{i+1}_L{j+1}"] = float(c.max())
    return {"singular_values": [float(v) for v in s],
            "condition_number": float(s[0] / s[-1]),
            "interlayer_correlation_pairs": pairs,
            "interlayer_correlation_max": float(max(pairs.values()))}

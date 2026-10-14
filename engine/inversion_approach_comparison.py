"""Comparison of magnetization inversion approaches (paper Table 2).

Test case: 2 x 2 x 2 m cube, |M| = 0.08 A/m along -z, drive-around traverse at 1 m standoff
(sensor 0.5 m), 0.6 nT per-axis Gaussian noise, 0.25 m mesh cells, inversion domain equal to the
imaged body (512 active cells), N_MC = 20 noise realizations.

Approaches:
  (1) Cartesian voxel MVI: 3N parameters, linear, per-component least squares with sensitivity
      weighting and beta cooling to target misfit; cold start.
  (2) Spherical voxel MVI: (amplitude, theta, phi) per cell, nonlinear; initialized from the
      Cartesian solution (a zero cold start is degenerate because the angle Jacobian vanishes
      at |M| = 0).
  (3) Parametric (K = 1): uniform M over the known body, closed-form weighted least squares.

Metrics: direction error and recovered unit vector of the volume-averaged magnetization
(voxel cases) or of the recovered vector (parametric), mean and scatter over the realizations.

Run:  python engine/inversion_approach_comparison.py
Out:  results/engine/inversion_approach_comparison_results.json
"""
import os, io, json, time, contextlib
import numpy as np
import discretize
from simpeg import (maps, data, data_misfit, directives, inverse_problem, optimization,
                    inversion, regularization)
from simpeg.potential_fields import magnetics as mag
from simpeg.utils import mat_utils

from inversion_helpers import AMP_NT, SENSOR_H, dir_err_deg, ls_operator, seed_for
from core import generate_traverse

HERE = os.path.dirname(os.path.abspath(__file__))
SIDE, M_TRUE_AMP, SIGMA, DH, STANDOFF, SPACING = 2.0, 0.08, 0.6, 0.25, 1.0, 0.05
N_MC = 20
SEED_LABEL = "table2_cmp"


def build():
    n_core = int(6.0 / DH)                       # x,y in [-3,3]
    mesh = discretize.TensorMesh([DH * np.ones(n_core), DH * np.ones(n_core), DH * np.ones(int(4.0 / DH))],
                                 origin=[-3.0, -3.0, -1.0])       # z in [-1, 3]
    cc = mesh.cell_centers
    body = (np.abs(cc[:, 0]) <= SIDE / 2) & (np.abs(cc[:, 1]) <= SIDE / 2) & (cc[:, 2] >= 0.0) & (cc[:, 2] <= SIDE)
    nA = int(body.sum())
    sp = {"cx": 0.0, "cy": 0.0, "sx": SIDE, "sy": SIDE, "sz": SIDE}
    sxyz = generate_traverse("full_orbit", sp, sensor_height=SENSOR_H, spacing=SPACING,
                             orbit_offset=STANDOFF, approach_distance=5.0, shape="rect")
    rx = mag.receivers.Point(np.asarray(sxyz, float), components=["bx", "by", "bz"])
    src = mag.sources.UniformBackgroundField(receiver_list=[rx], amplitude=AMP_NT, inclination=90.0, declination=0.0)
    survey = mag.survey.Survey(src)
    sim = mag.simulation.Simulation3DIntegral(mesh, survey=survey, chiMap=maps.IdentityMap(nP=3 * nA),
                                              active_cells=body, model_type="vector",
                                              store_sensitivities="ram", sensitivity_dtype=np.float64)
    return mesh, body, nA, sxyz, survey, sim


def voxel_metrics(m_vec, nA, m_true_body):
    M = np.column_stack([m_vec[0:nA], m_vec[nA:2 * nA], m_vec[2 * nA:3 * nA]])
    avg = M.mean(axis=0)
    u = avg / (np.linalg.norm(avg) + 1e-300)
    return {"dir_err_deg": dir_err_deg(avg, m_true_body),
            "mag_err_pct": float(100 * (np.linalg.norm(avg) - np.linalg.norm(m_true_body)) / np.linalg.norm(m_true_body)),
            "unit_vector": [float(v) for v in u], "cell_dir_err_median_deg": float(np.median([dir_err_deg(r, m_true_body) for r in M]))}


def run_cartesian(sim, mesh, body, nA, dat, seed):
    wires = maps.Wires(("p0", nA), ("p1", nA), ("p2", nA))
    reg = (regularization.WeightedLeastSquares(mesh, active_cells=body, mapping=wires.p0)
           + regularization.WeightedLeastSquares(mesh, active_cells=body, mapping=wires.p1)
           + regularization.WeightedLeastSquares(mesh, active_cells=body, mapping=wires.p2))
    for r in reg.objfcts:
        r.reference_model = np.zeros(3 * nA)
    dmis = data_misfit.L2DataMisfit(data=dat, simulation=sim)
    opt = optimization.InexactGaussNewton(maxIter=25, maxIterCG=30, tolCG=1e-4)
    invprob = inverse_problem.BaseInvProblem(dmis, reg, opt)
    dlist = [directives.UpdateSensitivityWeights(every_iteration=False),
             directives.BetaEstimate_ByEig(beta0_ratio=1e1, random_seed=seed),
             directives.BetaSchedule(coolingFactor=3.0, coolingRate=1),
             directives.UpdatePreconditioner(), directives.TargetMisfit(chifact=1.0)]
    with contextlib.redirect_stdout(io.StringIO()):
        m = inversion.BaseInversion(invprob, directiveList=dlist).run(np.zeros(3 * nA))
    phi = float(dmis(m)) / dat.dobs.size
    return np.asarray(m, float), phi


def run_spherical(mesh, body, nA, survey, dat, m_cart, seed):
    sph = maps.SphericalSystem()
    sim_s = mag.simulation.Simulation3DIntegral(mesh, survey=survey, chiMap=sph, active_cells=body, model_type="vector",
                                                store_sensitivities="ram", sensitivity_dtype=np.float64)
    wires = maps.Wires(("amp", nA), ("theta", nA), ("phi", nA))
    reg = (regularization.WeightedLeastSquares(mesh, active_cells=body, mapping=wires.amp)
           + regularization.WeightedLeastSquares(mesh, active_cells=body, mapping=wires.theta)
           + regularization.WeightedLeastSquares(mesh, active_cells=body, mapping=wires.phi))
    for r in reg.objfcts:
        r.reference_model = np.zeros(3 * nA)
    # warm start from the Cartesian solution (Section 3.4.1)
    m0 = mat_utils.cartesian2spherical(m_cart.reshape((-1, 3), order="F")).reshape(-1, order="F")
    dmis = data_misfit.L2DataMisfit(data=dat, simulation=sim_s)
    opt = optimization.ProjectedGNCG(maxIter=25, lower=-np.inf, upper=np.inf, maxIterCG=30, tolCG=1e-4)
    invprob = inverse_problem.BaseInvProblem(dmis, reg, opt)
    dlist = [directives.UpdateSensitivityWeights(every_iteration=False),
             directives.BetaEstimate_ByEig(beta0_ratio=1e1, random_seed=seed),
             directives.ProjectSphericalBounds(), directives.BetaSchedule(coolingFactor=3.0, coolingRate=1),
             directives.UpdatePreconditioner(), directives.TargetMisfit(chifact=1.0)]
    with contextlib.redirect_stdout(io.StringIO()):
        ms = inversion.BaseInversion(invprob, directiveList=dlist).run(m0)
    m = np.asarray(sph * ms, float)
    phi = float(dmis(ms)) / dat.dobs.size
    return m, phi


def main():
    t0 = time.time()
    mesh, body, nA, sxyz, survey, sim = build()
    m_true_body = M_TRUE_AMP * np.array([0.0, 0.0, -1.0])
    m_full = np.concatenate([np.full(nA, m_true_body[0]), np.full(nA, m_true_body[1]), np.full(nA, m_true_body[2])])
    d_clean = np.asarray(sim.dpred(m_full), float)
    G_blk = np.column_stack([np.asarray(sim.dpred(np.concatenate([np.ones(nA) * (k == 0), np.ones(nA) * (k == 1), np.ones(nA) * (k == 2)])), float) for k in range(3)])
    K = ls_operator(G_blk)
    print(f"mesh {mesh.n_cells} cells, body {nA} active cells, {len(sxyz)} stations, nD {survey.nD}, peak |B| {np.abs(d_clean).max():.2f} nT, sigma {SIGMA} nT")

    out = {"meta": {"script": "inversion_approach_comparison.py", "test_case": "2x2x2 m cube, |M|=0.08 A/m (-z), drive-around 1 m standoff, "
                    f"sensor {SENSOR_H} m, {SIGMA} nT noise, {DH} m cells, inversion domain = body ({nA} cells)",
                    "n_stations": int(len(sxyz)), "nD": int(survey.nD), "spacing_m": SPACING, "N_MC": N_MC,
                    "peak_field_nT": float(np.abs(d_clean).max()), "true_unit_vector": [0.0, 0.0, -1.0]},
           "draws": [], "summary": {}}
    for i in range(N_MC):
        rng = np.random.default_rng(seed_for(f"{SEED_LABEL}_{i}"))
        dobs = d_clean + rng.normal(0.0, SIGMA, size=d_clean.size)
        dat = data.Data(survey, dobs=dobs, standard_deviation=SIGMA)
        m_par = K @ dobs
        rec = {"parametric": {"dir_err_deg": dir_err_deg(m_par, m_true_body),
                              "mag_err_pct": float(100 * (np.linalg.norm(m_par) - M_TRUE_AMP) / M_TRUE_AMP),
                              "unit_vector": [float(v) for v in m_par / np.linalg.norm(m_par)]}}
        try:
            m_c, phi_c = run_cartesian(sim, mesh, body, nA, dat, int(seed_for(f"{SEED_LABEL}_c_{i}")) % (2 ** 31))
            rec["cartesian_voxel"] = voxel_metrics(m_c, nA, m_true_body); rec["cartesian_voxel"]["phi_d_over_nD"] = phi_c
        except Exception as e:
            rec["cartesian_voxel"] = {"error": repr(e)}; m_c = None
        try:
            if m_c is None: raise RuntimeError("no Cartesian start")
            m_s, phi_s = run_spherical(mesh, body, nA, survey, dat, m_c, int(seed_for(f"{SEED_LABEL}_s_{i}")) % (2 ** 31))
            rec["spherical_voxel"] = voxel_metrics(m_s, nA, m_true_body); rec["spherical_voxel"]["phi_d_over_nD"] = phi_s
        except Exception as e:
            rec["spherical_voxel"] = {"error": repr(e)}
        out["draws"].append(rec)
        print(f"[draw {i:2d}] parametric {rec['parametric']['dir_err_deg']:5.2f} deg | cartesian {rec['cartesian_voxel'].get('dir_err_deg', float('nan')):6.2f} deg "
              f"(phi_d/nD {rec['cartesian_voxel'].get('phi_d_over_nD', float('nan')):.2f}) | spherical {rec['spherical_voxel'].get('dir_err_deg', float('nan')):6.2f} deg "
              f"(phi_d/nD {rec['spherical_voxel'].get('phi_d_over_nD', float('nan')):.2f})  ({time.time()-t0:.0f}s)", flush=True)

    for k in ("parametric", "cartesian_voxel", "spherical_voxel"):
        d = np.array([r[k]["dir_err_deg"] for r in out["draws"] if "dir_err_deg" in r[k]])
        u = np.array([r[k]["unit_vector"] for r in out["draws"] if "unit_vector" in r[k]])
        mg = np.array([r[k]["mag_err_pct"] for r in out["draws"] if "mag_err_pct" in r[k]])
        if d.size:
            um = u.mean(axis=0); um /= np.linalg.norm(um)
            out["summary"][k] = {"dir_err_deg_mean": float(d.mean()), "dir_err_deg_std": float(d.std()), "dir_err_deg_median": float(np.median(d)),
                                 "mag_err_pct_mean": float(mg.mean()), "unit_vector_mean": [float(v) for v in um], "n_ok": int(d.size)}
    out["meta"]["wall_clock_s"] = time.time() - t0
    out_dir = os.path.join(os.path.dirname(HERE), "results", "engine")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "inversion_approach_comparison_results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["summary"], indent=1)); print(f"total {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()

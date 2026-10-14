"""Sobol variance attribution of the combined-budget direction error.

Output: the direction error (deg) of the noise-free recovery of one outcrop (white noise off; slow drift on), for the
requirement-set tolerances of thesis Table 5.1, evaluated for the six test magnetization directions on the same input
draws; also the mean over the six directions as one scalar output.

Inputs (independent; grouped by error source, so the indices are per source):
  burial     b ~ U(0, 0.5 m)                                                         (1 variable)
  pose       pointing az, el ~ U(+-1.5 deg); translation x, y, z ~ U(+-0.075 m)      (5 variables)
  site       100 nT site field from a dipole at 530 m: source azimuth ~ U(0, 2 pi), moment direction uniform on
             the sphere (2 variables), i.e. a gradient-fit residual of ~0.02 nT     (3 variables)
  tilt       axis of a 0.05 nT constant tilt error ~ U(0, 2 pi)                      (1 variable)
  induced    ln(chi_est / chi) ~ N(0, (ln 1.4)^2), chi = 2e-3                         (1 variable)
  drift      one OU drift realization (L = 100 m, 0.01 nT rms about the survey mean), indexed by a uniform
             variable that seeds the generator: the whole path is ONE grouped input   (1 variable)
  centering  loop offset from the outcrop centre: radius ~ U(0, 0.1 m), azimuth ~ U(0, 2 pi)  (2 variables)
The released budget rows hold the offset fixed at 0.1 m along +x; here the 0.1 m centering requirement is treated as a
tolerance like the others. Surveys: the requirement set (loops at 1 and 3 m, offsets and gradient fitted) and the
mitigated set (each loop driven both ways, six tilt heading terms fitted).

Second output (added because the angle is a norm, so error sources that add as vectors interact strongly in it):
the two components of the recovered unit vector perpendicular to the truth, (e1, e2) in degrees. For this vector
output the generalized Sobol indices of Gamboa et al. (2014) are reported (component indices weighted by component
variance; for the six directions together, all 12 components). In the small-angle limit theta^2 = e1^2 + e2^2, so these
indices attribute the scatter part of the mean-square direction error, tr Cov(e).

Estimators (SALib is not installed in lunar-mag, so this is a numpy implementation of the standard ones):
  Saltelli et al. (2010) first order  S_i  = mean(f_B (f_ABi - f_A)) / V
  Jansen (1999) total order           ST_i = mean((f_A - f_ABi)^2) / (2 V)
with V the variance of [f_A, f_B], A and B the two halves of a scrambled Sobol' sequence in 2 x 14 dimensions and AB_i
equal to A with group i's columns taken from B (N (G + 2) model runs). 95% confidence intervals are percentile
bootstrap intervals over the N rows (1000 resamples). The estimates at N/2 are stored as a convergence check.

Seeds: scipy.stats.qmc.Sobol(d=28, scramble=True, seed=seed_for("study:sobol:<set>")).

Run:  python studies/sobol.py [--log2n 11] [--workers 3]      (about 15-30 min)
Out:  studies/results/sobol.json
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

import _common as C
import _budget as B

GROUPS = {"burial": 1, "pose": 5, "site": 3, "tilt": 1, "induced": 1, "drift": 1, "centering": 2}
D = sum(GROUPS.values())
SLICES, _i = {}, 0
for g, k in GROUPS.items():
    SLICES[g] = slice(_i, _i + k); _i += k
SETS = {"requirement_set": dict(reverse=False, fit_tilt=False), "mitigated_set": dict(reverse=True, fit_tilt=True)}
TOL = dict(bmax=0.5, pose=(1.5, 0.075), R_site=530.0, B0=100.0, tilt_nT=0.05, chi=2e-3, chi_err=np.log(1.4),
           drift_rms=0.01, dx_max=0.1)

_MODEL = None


def _init(set_name):
    global _MODEL
    s = SETS[set_name]
    _MODEL = B.OutcropModel(TOL["dx_max"], s["reverse"], s["fit_tilt"], TOL["R_site"], TOL["B0"], TOL["tilt_nT"],
                            TOL["chi"], TOL["chi_err"], TOL["drift_rms"], C.DIRS6)


def perp_basis(u):
    u = u / np.linalg.norm(u)
    a = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, u); e1 /= np.linalg.norm(e1)
    return e1, np.cross(u, e1)


def f(u):
    """For one row of unit-cube inputs u (D,): (6, 3) array, per direction [theta, e1, e2] in degrees, where theta is
    the direction error and (e1, e2) the components of the recovered unit vector perpendicular to the truth."""
    m = _MODEL
    b = TOL["bmax"] * u[SLICES["burial"]][0]
    up = u[SLICES["pose"]]; p, t = TOL["pose"]
    us = u[SLICES["site"]]
    z = 2 * us[1] - 1; ph = 2 * np.pi * us[2]
    mdir = np.array([np.sqrt(1 - z * z) * np.cos(ph), np.sqrt(1 - z * z) * np.sin(ph), z])
    eta = float(ndtri(np.clip(u[SLICES["induced"]][0], 1e-12, 1 - 1e-12)))
    seed = int(u[SLICES["drift"]][0] * 2 ** 53)
    uc = u[SLICES["centering"]]
    m.dx = TOL["dx_max"] * uc[0]
    o = m.evaluate(b=b, az_deg=p * (2 * up[0] - 1), el_deg=p * (2 * up[1] - 1), trans=t * (2 * up[2:5] - 1),
                   site_az=2 * np.pi * us[0], site_m=mdir, tilt_axis=2 * np.pi * u[SLICES["tilt"]][0], eta=eta,
                   drift_seed=seed, offset_az=2 * np.pi * uc[1])
    V = o["Mdrift"]
    out = np.zeros((len(V), 3))
    for k in range(len(V)):
        e1, e2 = perp_basis(m.MT[k]); w = V[k] / np.linalg.norm(V[k])
        out[k] = [C.dir_err_deg(V[k], m.MT[k]), np.degrees(w @ e1), np.degrees(w @ e2)]
    return out


def indices(fA, fB, fAB, idx=None):
    if idx is not None:
        fA, fB, fAB = fA[idx], fB[idx], fAB[:, idx]
    V = np.var(np.concatenate([fA, fB]), ddof=1)
    S = np.array([np.mean(fB * (fAB[i] - fA)) / V for i in range(len(fAB))])
    ST = np.array([0.5 * np.mean((fA - fAB[i]) ** 2) / V for i in range(len(fAB))])
    return S, ST, V


def gindices(fA, fB, fAB, idx=None):
    """Generalized Sobol indices of a vector output (Gamboa, Janon, Klein and Lagnoux 2014): per-component
    indices weighted by the component variances. fA, fB: (N, q); fAB: (G, N, q)."""
    if idx is not None:
        fA, fB, fAB = fA[idx], fB[idx], fAB[:, idx]
    V = np.var(np.concatenate([fA, fB]), axis=0, ddof=1)                    # (q,)
    S = np.array([np.sum(np.mean(fB * (fAB[i] - fA), axis=0)) / V.sum() for i in range(len(fAB))])
    ST = np.array([np.sum(0.5 * np.mean((fA - fAB[i]) ** 2, axis=0)) / V.sum() for i in range(len(fAB))])
    return S, ST, V.sum()


def analyse_vec(fA, fB, fAB, label):
    N = len(fA)
    S, ST, V = gindices(fA, fB, fAB)
    r = C.rng_for(f"study:sobol:bootstrap:vec:{label}")
    bs = [gindices(fA, fB, fAB, r.integers(0, N, N)) for _ in range(1000)]
    bS = np.array([x[0] for x in bs]); bST = np.array([x[1] for x in bs])
    Sh, STh, _ = gindices(fA[: N // 2], fB[: N // 2], fAB[:, : N // 2])
    allv = np.concatenate([fA, fB])
    out = {"total_variance_deg2": float(V), "mean_vector_deg": [float(v) for v in allv.mean(0)],
           "rms_direction_error_deg": float(np.sqrt(np.mean(np.sum(allv ** 2, axis=1) / (allv.shape[1] / 2)))),
           "N": int(N), "sum_S1": float(S.sum()), "groups": {}}
    for i, g in enumerate(GROUPS):
        out["groups"][g] = {
            "S1": float(S[i]), "S1_ci95": [float(np.percentile(bS[:, i], 2.5)), float(np.percentile(bS[:, i], 97.5))],
            "ST": float(ST[i]), "ST_ci95": [float(np.percentile(bST[:, i], 2.5)), float(np.percentile(bST[:, i], 97.5))],
            "S1_at_N_half": float(Sh[i]), "ST_at_N_half": float(STh[i])}
    return out


def analyse(fA, fB, fAB, label):
    N = len(fA)
    S, ST, V = indices(fA, fB, fAB)
    r = C.rng_for(f"study:sobol:bootstrap:{label}")
    bs = [indices(fA, fB, fAB, r.integers(0, N, N)) for _ in range(1000)]
    bS = np.array([x[0] for x in bs]); bST = np.array([x[1] for x in bs])
    Sh, STh, _ = indices(fA[: N // 2], fB[: N // 2], fAB[:, : N // 2])
    out = {"variance_deg2": float(V), "mean_deg": float(np.mean(np.concatenate([fA, fB]))),
           "sd_deg": float(np.sqrt(V)), "N": int(N), "sum_S1": float(S.sum()), "groups": {}}
    for i, g in enumerate(GROUPS):
        out["groups"][g] = {
            "S1": float(S[i]), "S1_ci95": [float(np.percentile(bS[:, i], 2.5)), float(np.percentile(bS[:, i], 97.5))],
            "S1_se_boot": float(bS[:, i].std(ddof=1)),
            "ST": float(ST[i]), "ST_ci95": [float(np.percentile(bST[:, i], 2.5)), float(np.percentile(bST[:, i], 97.5))],
            "ST_se_boot": float(bST[:, i].std(ddof=1)),
            "S1_at_N_half": float(Sh[i]), "ST_at_N_half": float(STh[i])}
    return out


def run_set(name, log2n, workers, t0):
    N = 2 ** log2n
    AB = qmc.Sobol(d=2 * D, scramble=True, seed=C.seed_for(f"study:sobol:{name}")).random_base2(log2n)
    A, Bm = AB[:, :D], AB[:, D:]
    rows = [A, Bm]
    for g in GROUPS:
        X = A.copy(); X[:, SLICES[g]] = Bm[:, SLICES[g]]; rows.append(X)
    X = np.vstack(rows)
    with Pool(workers, initializer=_init, initargs=(name,)) as pool:
        Y = np.array(pool.map(f, X, chunksize=64))
    print(f"  [{name}] {len(X)} model runs ({time.time()-t0:.0f}s)", flush=True)
    G = len(GROUPS)
    Ys = Y.reshape(G + 2, N, Y.shape[1], 3)
    T = Ys[..., 0]                                                           # direction errors (G+2, N, 6)
    E = Ys[..., 1:]                                                          # perpendicular components
    res = {"N": N, "model_runs": int(len(X)), "outputs": {}, "vector_outputs": {}}
    names = list(C.DIRS6)
    for k, dn in enumerate(names):
        res["outputs"][dn] = analyse(T[0, :, k], T[1, :, k], T[2:, :, k], f"{name}:{dn}")
        res["vector_outputs"][dn] = analyse_vec(E[0, :, k], E[1, :, k], E[2:, :, k], f"{name}:{dn}")
    res["outputs"]["mean_over_directions"] = analyse(T[0].mean(1), T[1].mean(1), T[2:].mean(2), f"{name}:mean")
    q = E.shape[2] * 2
    res["vector_outputs"]["all_directions"] = analyse_vec(E[0].reshape(N, q), E[1].reshape(N, q), E[2:].reshape(G, N, q), f"{name}:all")
    return res, Y


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--log2n", type=int, default=11)
    ap.add_argument("--workers", type=int, default=3); ap.add_argument("--sets", default=",".join(SETS))
    a = ap.parse_args()
    t0 = time.time()
    out = {"meta": C.provenance(__file__, {
        "study": "Sobol variance attribution (outline-v2 item 4)", "tolerances": {k: (list(v) if isinstance(v, tuple) else v) for k, v in TOL.items()},
        "groups": GROUPS, "estimators": "Saltelli 2010 (S1), Jansen 1999 (ST); bootstrap 1000",
        "output": "direction error (deg) of the noise-free recovery (white noise off, drift on), per direction and the "
                  "mean over the six directions; plus generalized indices of the perpendicular error vector (e1, e2)", "salib": "not installed in lunar-mag; numpy implementation"}), "sets": {}}
    for name in a.sets.split(","):
        res, Y = run_set(name, a.log2n, a.workers, t0)
        out["sets"][name] = res
        np.save(os.path.join(C.RESULTS, f"sobol_{name}_outputs.npy"), Y)
        mo = res["outputs"]["mean_over_directions"]
        print(f"[{name}] mean error {mo['mean_deg']:.2f} deg, sd {mo['sd_deg']:.2f}; " + "; ".join(
            f"{g} S1 {v['S1']:.3f} ST {v['ST']:.3f}" for g, v in mo["groups"].items()), flush=True)
        mv = res["vector_outputs"]["all_directions"]
        print(f"[{name}] vector: rms {mv['rms_direction_error_deg']:.2f} deg, sum S1 {mv['sum_S1']:.3f}; " + "; ".join(
            f"{g} S1 {v['S1']:.3f} ST {v['ST']:.3f}" for g, v in mv["groups"].items()), flush=True)
        out["meta"]["wall_s"] = time.time() - t0
        C.save("sobol.json", out)
    print(C.save("sobol.json", out))


if __name__ == "__main__":
    main()

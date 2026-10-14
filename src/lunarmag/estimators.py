"""The linear estimator, its covariance and the nuisance-parameter designs of thesis Chapter 3.

For a fixed source geometry the data are linear in the magnetization (thesis Eq. 3.14 [eq:th_data_model]),

    d = G_K p + n,            p_hat = K d,        K = (G_K^T W^2 G_K)^-1 G_K^T W^2      (Eqs. 3.11, 3.11)

and every quantity the thesis derives from it (covariance, small-error metrics, offset, gradient and tilt
columns) is a short NumPy expression, collected here with its equation number. Equation numbers follow the
renumbered Chapter 3 of the 2026-10-13 draft; the label in brackets is stable.

The functions that also exist in the frozen analysis scripts (``headings``, ``gradient_basis`` in
``analysis/a06_isolated_robustness.py``; ``tilt_error`` and the heading columns of ``budget`` in
``analysis/a09_systematics.py``) repeat their arithmetic exactly, and ``tests/test_estimators.py`` checks that
they return bit-identical arrays, so the stored results are explained by this documented code.

Ordering convention: data vectors are station-major, (bx, by, bz) of station 1, then station 2, ...
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------------------------- estimator
def least_squares_operator(G: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Thesis Eq. 3.11 [eq:ls] (the unweighted K = (G^T G)^-1 G^T for W = I): K = (G^T W^2 G)^-1 G^T W^2, W = diag(weights).

    Equal weights cancel in K, so ``weights=None`` gives the operator used by every scenario
    (``inversion_helpers.ls_operator``).
    """
    G = np.asarray(G, float)
    if weights is None:
        return np.linalg.solve(G.T @ G, G.T)
    w2 = np.asarray(weights, float) ** 2
    GtW2 = G.T * w2[None, :]
    return np.linalg.solve(GtW2 @ G, GtW2)


def estimator_covariance(G: np.ndarray, sigma: float | np.ndarray = 1.0,
                         noise_cov: Optional[np.ndarray] = None) -> np.ndarray:
    """Covariance of p_hat = K d.

    * i.i.d. noise of standard deviation ``sigma`` (scalar): thesis Eq. 3.16 [eq:th_covariance],
      C_p = sigma^2 (G^T G)^-1.
    * independent noise with per-datum ``sigma`` (array) and the weighted estimator W = 1/sigma:
      thesis Eq. 3.17 [eq:th_covariance_weighted], C_p = (G^T W^2 G)^-1.
    * any noise covariance ``noise_cov`` with the ordinary (unweighted) estimator: thesis Eq. 3.18
      [eq:th_covariance_correlated], C_p = K C_n K^T.
    """
    G = np.asarray(G, float)
    if noise_cov is not None:
        K = least_squares_operator(G)
        return K @ np.asarray(noise_cov, float) @ K.T
    s = np.asarray(sigma, float)
    if s.ndim == 0:
        return float(s) ** 2 * np.linalg.inv(G.T @ G)
    w2 = 1.0 / s ** 2
    return np.linalg.inv((G.T * w2[None, :]) @ G)


def small_error_moments(C: np.ndarray, M: Sequence[float]) -> tuple[float, float]:
    """Thesis Eq. 3.21 [eq:th_par_perp_variance] for the error eps = M_hat - M with covariance C (3 x 3).

    Returns (sigma_par^2, E|eps_perp|^2) = (u^T C u, tr C - u^T C u), u = M/|M|. By Eq. 3.20 the fractional
    intensity error is ~ eps_par/|M| and the direction error ~ |eps_perp|/|M| (radians) when both are small.
    E|eps_perp|^2 is a second moment (not the variance of theta).
    """
    u = np.asarray(M, float) / np.linalg.norm(M)
    par = float(u @ C @ u)
    return par, float(np.trace(C) - par)


def predicted_mean_direction_error_deg(C: np.ndarray, M: Sequence[float]) -> float:
    """Small-error prediction of the mean direction error from the covariance alone.

    If the perpendicular error is a circular 2-D Gaussian, |eps_perp| is Rayleigh distributed with
    E|eps_perp| = sqrt(pi E|eps_perp|^2 / 4), so E[theta] ~ sqrt(pi E|eps_perp|^2 / 4) / |M| (Eq. 3.20-3.21).
    The isolated-outcrop covariance is close to circular in the plane normal to -z, which the tests use.
    """
    _, perp2 = small_error_moments(C, M)
    return float(np.degrees(np.sqrt(np.pi * perp2 / 4.0) / np.linalg.norm(M)))


# --------------------------------------------------------------------------------------------- nuisance columns
def offset_columns(n_stations: int) -> np.ndarray:
    """Thesis Eq. 3.22 [eq:th_augmented_model]: E = [I3; I3; ...; I3] (3P x 3), a constant field in the local frame."""
    return np.tile(np.eye(3), (n_stations, 1))


def augmented_solution(G: np.ndarray, N: np.ndarray, d: np.ndarray) -> np.ndarray:
    """The solution of thesis Eq. 3.22 [eq:th_augmented_model] (unnumbered in the final draft): [p_hat; c_hat] = (A^T A)^-1 A^T d with A = [G N]."""
    A = np.column_stack([G, N])
    return np.linalg.solve(A.T @ A, A.T @ np.asarray(d, float))


def marginal_covariance(G: np.ndarray, N: np.ndarray, sigma: float) -> np.ndarray:
    """Thesis Eq. 3.23 [eq:th_marginal_covariance]: C = sigma^2 (G^T P_N^perp G)^-1, P_N^perp = I - N (N^T N)^-1 N^T.

    Equal to the top-left block of sigma^2 (A^T A)^-1 with A = [G N] (Schur complement); the price of fitting N.
    """
    G = np.asarray(G, float); N = np.asarray(N, float)
    NtN_inv_Nt = np.linalg.solve(N.T @ N, N.T)
    PG = G - N @ (NtN_inv_Nt @ G)
    return sigma ** 2 * np.linalg.inv(G.T @ PG)


def headings(sxyz: np.ndarray) -> np.ndarray:
    """Driving heading psi (rad, from +x toward +y) at each station by central differences of the xy path.

    Same arithmetic as ``analysis/a06_isolated_robustness.headings``.
    """
    s = sxyz[:, :2]
    d = np.gradient(s, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


def gradient_basis(sxyz: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Thesis Eq. 3.24 [eq:th_gradient_basis]: five station-major columns for a uniform field gradient.

    A field that is curl- and divergence-free has a symmetric, traceless gradient tensor T (5 degrees of freedom);
    the basis T1..T5 below is orthonormal in the Frobenius inner product, and column k is T_k (r - r_c) at each
    station, with r_c the survey centroid. Same arithmetic as ``analysis/a06_isolated_robustness.gradient_basis``.
    Returns (columns (3P x 5), [T1..T5]).
    """
    r = sxyz - sxyz.mean(0)
    E = []
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        T = np.zeros((3, 3)); T[i, j] = T[j, i] = 1.0 / np.sqrt(2); E.append(T)
    T = np.diag([1.0, -1.0, 0.0]) / np.sqrt(2); E.append(T)
    T = np.diag([1.0, 1.0, -2.0]) / np.sqrt(6); E.append(T)
    return np.column_stack([(r @ T.T).reshape(-1) for T in E]), E


# --------------------------------------------------------------------------------------------- attitude (tilt)
def heading_rotation(psi: float | np.ndarray) -> np.ndarray:
    """Thesis Eq. B.6 [eq:th_heading_rotation]: rotation by heading psi about z, (3, 3) or (n, 3, 3)."""
    psi = np.asarray(psi, float)
    c, s = np.cos(psi), np.sin(psi)
    R = np.zeros(psi.shape + (3, 3))
    R[..., 0, 0] = c; R[..., 0, 1] = -s; R[..., 1, 0] = s; R[..., 1, 1] = c; R[..., 2, 2] = 1.0
    return R


def skew(k: Sequence[float]) -> np.ndarray:
    """Cross-product matrix [k]_x, so that [k]_x v = k x v."""
    k = np.asarray(k, float)
    return np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])


def rodrigues(k: Sequence[float], angle_rad: float) -> np.ndarray:
    """Thesis Eq. B.8 [eq:th_rodrigues]: R_err = I + sin(a) [k]_x + (1 - cos(a)) [k]_x^2 for a unit axis k."""
    Kx = skew(k)
    return np.eye(3) + np.sin(angle_rad) * Kx + (1 - np.cos(angle_rad)) * Kx @ Kx


def tilt_error(sxyz: np.ndarray, Bamb: Sequence[float], dtheta_deg: float, axis_angle: float) -> np.ndarray:
    """Thesis Eq. B.7 [eq:th_tilt_error_exact]: e(psi) = R_psi (R_err - I) R_psi^T B_amb, station-major (nT).

    A constant sensor tilt of ``dtheta_deg`` about the horizontal rover-frame axis k = (cos a, sin a, 0)
    (a = ``axis_angle``), in a static ambient field ``Bamb`` (nT). Same arithmetic as
    ``analysis/a09_systematics.tilt_error``. R_err is the effective component-error rotation (review note Y6-19:
    if R_err instead denotes the physical sensor-axis orientation, the measured components use its transpose;
    with uniformly random tilt axes, as in every budget, the two are statistically identical).
    """
    psi = headings(sxyz)
    a = np.radians(dtheta_deg)
    k = np.array([np.cos(axis_angle), np.sin(axis_angle), 0.0])
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    Rerr = np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx
    c, s = np.cos(psi), np.sin(psi)
    Rpsi = np.zeros((len(psi), 3, 3)); Rpsi[:, 0, 0] = c; Rpsi[:, 0, 1] = -s; Rpsi[:, 1, 0] = s; Rpsi[:, 1, 1] = c; Rpsi[:, 2, 2] = 1
    M = np.einsum("nij,jk,nlk->nil", Rpsi, Rerr - np.eye(3), Rpsi)      # R_psi (R_err - I) R_psi^T
    return np.einsum("nij,j->ni", M, Bamb).reshape(-1)


def tilt_error_first_order(psi: np.ndarray, Bamb: Sequence[float], dtheta_deg: float,
                           axis_angle: float) -> np.ndarray:
    """Thesis Eqs. B.9-3.25: e(psi) ~ dtheta (R_psi k) x B_amb = cos(psi) a + sin(psi) b, station-major (nT).

    a = dtheta k x B_amb and b = dtheta (z x k) x B_amb (Eq. 3.25 [eq:th_tilt_cos_sin]).
    """
    a_, b_ = tilt_coefficients(Bamb, dtheta_deg, axis_angle)
    psi = np.asarray(psi, float)
    return (np.cos(psi)[:, None] * a_ + np.sin(psi)[:, None] * b_).reshape(-1)


def tilt_coefficients(Bamb: Sequence[float], dtheta_deg: float, axis_angle: float) -> tuple[np.ndarray, np.ndarray]:
    """The vectors (a, b) of thesis Eq. 3.25 [eq:th_tilt_cos_sin] for a tilt dtheta about k = (cos a, sin a, 0)."""
    th = np.radians(dtheta_deg)
    k = np.array([np.cos(axis_angle), np.sin(axis_angle), 0.0])
    B = np.asarray(Bamb, float)
    return th * np.cross(k, B), th * np.cross(np.cross([0.0, 0.0, 1.0], k), B)


def heading_columns(psi: np.ndarray) -> np.ndarray:
    """Thesis Eq. 3.26 [eq:th_tilt_columns]: H = [cos(psi_i) I3, sin(psi_i) I3] stacked over stations (3P x 6).

    Fitting H absorbs the first-order tilt error exactly (Eq. 3.25). Same columns, in the same order, as the
    ``fit_tilt`` branch of ``analysis/a09_systematics.budget``.
    """
    c, sn = np.cos(psi), np.sin(psi)
    cols = []
    for w in (c, sn):
        for j in range(3):
            v = np.zeros((len(psi), 3)); v[:, j] = w; cols.append(v.reshape(-1))
    return np.column_stack(cols)

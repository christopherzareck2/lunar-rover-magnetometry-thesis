"""Closed-form physics used to check and explain the simulations (pure NumPy, no SimPEG).

Every function names the thesis equation it implements. Equation numbers follow the renumbered Chapter 3 of the
2026-10-13 draft (labels in brackets are stable if chapters are renumbered again); ``docs/theory.md`` lists them all.

Units: SI throughout except fields, which are in nT (the unit of the simulated data). With mu0/(4 pi) =
1e-7 T m/A = 100 nT m/A, a magnetization of 1 A/m in a dimensionless geometric kernel gives 100 nT.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

MU0 = 4e-7 * np.pi            #: permeability of free space (H/m), as in engine/core.py
MU0_4PI_NT = 100.0            #: mu0 / (4 pi) in nT m / A


# --------------------------------------------------------------------------------------------- magnetization
def trm_magnetization(B_paleo_T: np.ndarray | float, chi_trm: float) -> np.ndarray | float:
    """Thesis Eq. 3.2 [eq:nrm]: M_NRM = (chi_TRM / mu0) B_paleo, in A/m for B_paleo in tesla."""
    return chi_trm * np.asarray(B_paleo_T, float) / MU0


def paleofield_from_magnetization(M: np.ndarray | float, chi_trm: float) -> np.ndarray | float:
    """Inverse of Eq. 3.2: B_paleo = mu0 M / chi_TRM (tesla). M_ref = 2.4e-3 A/m, chi = 1.58e-3 gives 1.909 uT."""
    return MU0 * np.asarray(M, float) / chi_trm


def induced_magnetization(B_site_nT: np.ndarray | float, chi: float) -> np.ndarray | float:
    """Thesis Eq. 3.3 [eq:induced]: magnetization induced by a weak static field, M_ind = chi B / mu0 (A/m, B in nT;
    isotropic, chi << 1).

    This is the expression used in ``analysis/a09_systematics.py`` (``chi * Bc * 1e-9 / MU0``).
    """
    return chi * np.asarray(B_site_nT, float) * 1e-9 / MU0


def unit_vector(inc_deg: float, dec_deg: float) -> np.ndarray:
    """Thesis Eq. 3.8 [eq:th_unit_vector]: u(I, D) = (cos I cos D, cos I sin D, -sin I).

    z is up and inclination is positive downward, so I = +90 deg is (0, 0, -1). D is measured from +x toward +y.
    """
    I, D = np.radians(inc_deg), np.radians(dec_deg)
    return np.array([np.cos(I) * np.cos(D), np.cos(I) * np.sin(D), -np.sin(I)])


def inc_dec(M: Sequence[float]) -> tuple[float, float]:
    """Inverse of Eq. 3.8: (I, D) in degrees, I = atan2(-Mz, sqrt(Mx^2 + My^2)), D = atan2(My, Mx)."""
    Mx, My, Mz = (float(v) for v in M)
    return float(np.degrees(np.arctan2(-Mz, np.hypot(Mx, My)))), float(np.degrees(np.arctan2(My, Mx)))


# --------------------------------------------------------------------------------------------- forward fields
def dipole_field(r_obs: np.ndarray, r_src: Sequence[float], m: Sequence[float]) -> np.ndarray:
    """Field of a point dipole, the far-field limit of thesis Eq. 3.6 [eq:forward_integral], in nT.

    B = (mu0 / 4 pi) [3 n (n . m) - m] / R^3, with n = (r - r_src)/R and m in A m^2. ``r_obs`` is (N, 3).
    """
    r = np.atleast_2d(np.asarray(r_obs, float)) - np.asarray(r_src, float)
    R = np.linalg.norm(r, axis=1, keepdims=True)
    n = r / R
    m = np.asarray(m, float)
    return MU0_4PI_NT * (3.0 * (n @ m)[:, None] * n - m) / R ** 3


def _log_plus(a: np.ndarray, R: np.ndarray, rest2: np.ndarray) -> np.ndarray:
    """ln(a + R) without cancellation: for a < 0 use ln(rest2) - ln(R - a), since (a + R)(R - a) = rest2."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a >= 0, np.log(a + R), np.log(rest2) - np.log(R - a))


def prism_tensor(bounds: Sequence[float], r_obs: np.ndarray) -> np.ndarray:
    """Closed-form geometric tensor U(r) of a uniformly magnetized rectangular prism, so that B = 100 U M nT.

    Thesis Eq. 3.6 [eq:forward_integral] integrated over the prism. Writing I(r) = int_V dV'/|r - r'|, the external
    field is B = (mu0 / 4 pi) (grad grad I) M, and with source-minus-observer corner coordinates (X, Y, Z),
    R = sqrt(X^2 + Y^2 + Z^2),

        U_xx = sum -atan(Y Z / (X R)),   U_xy = sum ln(Z + R),   U_xz = sum ln(Y + R),

    and cyclic permutations, where "sum" is the alternating sum over the 8 corners (+ for an upper bound, - for a
    lower bound in each coordinate) [Plouff 1976; Blakely 1996, Ch. 9]. Outside the prism tr U = 0.

    Valid for observation points strictly outside the prism and not on the planes of its faces (where the
    principal-value arctangent and the logarithms need limiting forms). Returns an (N, 3, 3) array.
    """
    x0, x1, y0, y1, z0, z1 = (float(b) for b in bounds)
    P = np.atleast_2d(np.asarray(r_obs, float))
    U = np.zeros((len(P), 3, 3))
    for i, xs in ((0, x0), (1, x1)):
        for j, ys in ((0, y0), (1, y1)):
            for k, zs in ((0, z0), (1, z1)):
                s = (1.0 if i else -1.0) * (1.0 if j else -1.0) * (1.0 if k else -1.0)   # + upper, - lower
                X, Y, Z = xs - P[:, 0], ys - P[:, 1], zs - P[:, 2]
                R = np.sqrt(X * X + Y * Y + Z * Z)
                U[:, 0, 0] += s * -np.arctan(Y * Z / (X * R))
                U[:, 1, 1] += s * -np.arctan(X * Z / (Y * R))
                U[:, 2, 2] += s * -np.arctan(X * Y / (Z * R))
                U[:, 0, 1] += s * _log_plus(Z, R, X * X + Y * Y)
                U[:, 0, 2] += s * _log_plus(Y, R, X * X + Z * Z)
                U[:, 1, 2] += s * _log_plus(X, R, Y * Y + Z * Z)
    U[:, 1, 0], U[:, 2, 0], U[:, 2, 1] = U[:, 0, 1], U[:, 0, 2], U[:, 1, 2]
    return U


def prism_field(bounds: Sequence[float], r_obs: np.ndarray, M: Sequence[float]) -> np.ndarray:
    """Field (N, 3) in nT of a uniformly magnetized prism, B = (mu0 / 4 pi) U M (see ``prism_tensor``)."""
    return MU0_4PI_NT * prism_tensor(bounds, r_obs) @ np.asarray(M, float)


def prism_field_quadrature(bounds: Sequence[float], r_obs: np.ndarray, M: Sequence[float],
                           n: int = 24) -> np.ndarray:
    """Thesis Eq. 3.6 evaluated by n^3-point Gauss-Legendre quadrature of the dipole kernel over the prism (nT).

    Independent of the closed form and of SimPEG; accurate to ~1e-9 relative at points a few cell sizes outside.
    """
    x0, x1, y0, y1, z0, z1 = (float(b) for b in bounds)
    t, w = np.polynomial.legendre.leggauss(n)

    def nodes(a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
        return 0.5 * (b - a) * t + 0.5 * (a + b), 0.5 * (b - a) * w

    (xs, wx), (ys, wy), (zs, wz) = nodes(x0, x1), nodes(y0, y1), nodes(z0, z1)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    W = (wx[:, None, None] * wy[None, :, None] * wz[None, None, :]).ravel()
    src = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    M = np.asarray(M, float)
    out = []
    for p in np.atleast_2d(np.asarray(r_obs, float)):
        r = p - src
        R = np.linalg.norm(r, axis=1)
        n_ = r / R[:, None]
        kern = (3.0 * (n_ @ M)[:, None] * n_ - M) / R[:, None] ** 3
        out.append(MU0_4PI_NT * (W[:, None] * kern).sum(0))
    return np.array(out)


# --------------------------------------------------------------------------------------------- error metrics
def direction_error_deg(M_rec: np.ndarray, M_true: Sequence[float]) -> np.ndarray:
    """Thesis Eq. 3.27 [eq:dir_error], vectorized over rows of ``M_rec``: arccos of the normalized dot product."""
    M_rec = np.atleast_2d(np.asarray(M_rec, float))
    M_true = np.asarray(M_true, float)
    c = M_rec @ M_true / (np.linalg.norm(M_rec, axis=1) * np.linalg.norm(M_true))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def intensity_error_pct(M_rec: np.ndarray, M_true: Sequence[float], signed: bool = False) -> np.ndarray:
    """Thesis Eq. 3.28 [eq:int_error]: 100 | |M_rec| - |M_true| | / |M_true| (``signed=True`` drops the outer |.|)."""
    M_rec = np.atleast_2d(np.asarray(M_rec, float))
    t = np.linalg.norm(np.asarray(M_true, float))
    d = 100.0 * (np.linalg.norm(M_rec, axis=1) - t) / t
    return d if signed else np.abs(d)


def residual_rms(d_obs: np.ndarray, d_pred: np.ndarray, sigma: np.ndarray | float) -> float:
    """Thesis Eq. 3.29 [eq:residual_rms]: sqrt(mean(((d_obs - d_pred) / sigma)^2)). Expected value ~ sqrt((Nd-q)/Nd)."""
    r = (np.asarray(d_obs, float) - np.asarray(d_pred, float)) / sigma
    return float(np.sqrt(np.mean(r * r)))


def white_asd(sigma: float, dx: float, v: float) -> float:
    """One-sided amplitude spectral density of white per-sample noise, sigma sqrt(2 dx / v) (thesis Eq. B.5 [eq:th_white_asd]).

    sigma = 0.50 nT per 1 cm sample at 1 cm/s gives 0.71 nT Hz^-1/2.
    """
    return float(sigma * np.sqrt(2.0 * dx / v))

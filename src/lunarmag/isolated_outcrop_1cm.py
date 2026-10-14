"""Isolated outcrop: the prism kernel per receiver and recovery with a mis-posed (imaged) outcrop.

The 4 x 4 x 1 m prism is centred at the origin (z = -0.5..0.5 m); the analysis scripts place the sensor at
z = 0, i.e. 0.5 m above the ground, 1 m outside the faces. warped_recovery rigidly rotates and translates the
body used by the inversion (pointing and range errors) while the data come from the true body.

Relocated from ``engine/isolated_outcrop_1cm.py`` (release v1.0.0); type hints and docstrings added, arithmetic
unchanged. ``engine/isolated_outcrop_1cm.py`` is an alias of this module.

Frame note: the code's prism z = -0.5..0.5 m with the sensor at z = 0 is the thesis's prism z = 0..1 m with the
sensor at z = 0.5 m, translated together; every field and recovery is identical.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import discretize

from .inversion_helpers import ls_operator, dir_err_deg, _sim_for, prism_bounds, PRISM_DIMS
from .core import generate_pose_errors

bnd = prism_bounds((0.0, 0.0, 0.0), PRISM_DIMS)


# --------------------------------------------------------------------------- #
# Pose helpers
# --------------------------------------------------------------------------- #
def per_receiver_kernel(bounds: Sequence[float], rx_locs: np.ndarray) -> np.ndarray:
    """A (nR,3,3): per-receiver 3x3 kernel of the single prism cell; A[r] maps (Mx,My,Mz)->(Bx,By,Bz).

    A[r] = (mu0/4pi) U(r) of the prism (thesis Eq. 3.6 integrated over the prism), in nT per A/m.
    """
    rx_locs = np.asarray(rx_locs, float)
    nR = len(rx_locs)
    sim_mesh = discretize.TensorMesh(
        [np.r_[bounds[1] - bounds[0]], np.r_[bounds[3] - bounds[2]], np.r_[bounds[5] - bounds[4]]],
        origin=[bounds[0], bounds[2], bounds[4]])
    sim = _sim_for(sim_mesh, rx_locs, 1)
    cols = [sim.dpred(np.eye(3)[k]).reshape(nR, 3) for k in range(3)]  # each (nR,3)
    return np.stack([np.column_stack([cols[0][r], cols[1][r], cols[2][r]]) for r in range(nR)], axis=0)


def imager_R(pointing_deg: Sequence[float]) -> np.ndarray:
    """Rigid rotation R = R_el @ R_az for (azimuth, elevation) pointing errors in degrees (matches core.pose_transform)."""
    az, el = np.deg2rad(pointing_deg[0]), np.deg2rad(pointing_deg[1])
    R_az = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1.0]])
    R_el = np.array([[np.cos(el), 0, np.sin(el)], [0, 1.0, 0], [-np.sin(el), 0, np.cos(el)]])
    return R_el @ R_az


def warped_recovery(bounds: Sequence[float], sxyz: np.ndarray, m_true: np.ndarray, sigma: float, n: int, seed: int,
                    pointing_max: float, range_max: float, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Recover M when the inversion body pose is mis-estimated by the imager (data = nominal geometry).

    mode='random' -> uniform within +/-bounds (typical); mode='worst' -> worst-magnitude/random-direction.
    The imaged-body operator is A_warp[r] = R A(R^T (s_r - T)) R^T (sensor positions mapped into the warped body frame).
    Returns the direction errors (deg) and signed intensity errors (%) of the n draws.
    """
    sxyz = np.asarray(sxyz, float)
    nR = len(sxyz)
    A_nom = per_receiver_kernel(bounds, sxyz)               # (nR,3,3)
    d_clean = (A_nom @ m_true).ravel()                      # station-major (nD,)
    rng = np.random.default_rng(seed)
    np.random.seed(int(seed) % (2 ** 32))                   # deterministic geometry stream
    de, me = np.zeros(n), np.zeros(n)
    for i in range(n):
        pdg, trans = generate_pose_errors(
            randomize=(mode == "random"),
            pointing_err_max_deg=(pointing_max, pointing_max), range_err_max_m=range_max)
        R = imager_R(pdg)
        T = np.asarray(trans, float)
        Q = (sxyz - T) @ R                                  # sensors in warped body-local frame
        A_q = per_receiver_kernel(bounds, Q)                # (nR,3,3)
        A_warp = np.einsum("ij,rjk,lk->ril", R, A_q, R)     # R @ A_q @ R^T per receiver
        A_inv = A_warp.reshape(3 * nR, 3)
        K = ls_operator(A_inv)
        noise = rng.normal(0.0, sigma, size=d_clean.size)
        m_rec = K @ (d_clean + noise)
        de[i] = dir_err_deg(m_rec, m_true)
        me[i] = 100.0 * (np.linalg.norm(m_rec) - np.linalg.norm(m_true)) / np.linalg.norm(m_true)
    return de, me

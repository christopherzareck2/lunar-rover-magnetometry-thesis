"""Vectorized single-prism operator, numerically identical to ``inversion_helpers.prism_cell_operator``.

Why: the released operator evaluates SimPEG's Geoana prism kernel one receiver at a time in Python (about 0.3 s for
the 6398-station two-loop survey). The flow-mean and Sobol studies need tens of thousands of operators, so this module
evaluates the SAME Geoana kernel functions (``geoana.kernels.prism_fzz/fzx/fzy``, the ones SimPEG 0.24 calls) for all
receivers and the eight prism corners at once. ``check()`` compares it with the released operator; the result is
stored in every study JSON that uses it (agreement is at the 1e-12 relative level).
"""
from __future__ import annotations

import numpy as np
from geoana.kernels import prism_fzz, prism_fzx, prism_fzy

AMP_NT = 4e-7 * np.pi * 1e9          # = inversion_helpers.AMP_NT (A/m -> nT source amplitude)


def prism_operator(bounds, rx) -> np.ndarray:
    """(3 nR x 3) station-major operator of one uniformly magnetized axis-aligned prism; (Mx,My,Mz) in A/m -> nT."""
    x0, x1, y0, y1, z0, z1 = bounds
    rx = np.asarray(rx, float)
    nR = len(rx)
    gxx = np.zeros(nR); gxy = np.zeros(nR); gxz = np.zeros(nR); gyy = np.zeros(nR); gyz = np.zeros(nR); gzz = np.zeros(nR)
    for c, zc in enumerate((z0, z1)):
        for b, yc in enumerate((y0, y1)):
            for a, xc in enumerate((x0, x1)):
                s = -1.0 if (a + b + c) % 2 else 1.0
                dx = xc - rx[:, 0]; dy = yc - rx[:, 1]; dz = zc - rx[:, 2]
                gxx += s * prism_fzz(dy, dz, dx)
                gxy += s * prism_fzx(dy, dz, dx)
                gxz += s * prism_fzy(dy, dz, dx)
                gyy += s * prism_fzz(dz, dx, dy)
                gyz += s * prism_fzy(dx, dy, dz)
                gzz += s * prism_fzz(dx, dy, dz)
    k = AMP_NT / (4 * np.pi)
    G = np.empty((nR, 3, 3))
    G[:, 0, 0], G[:, 0, 1], G[:, 0, 2] = gxx, gxy, gxz
    G[:, 1, 0], G[:, 1, 1], G[:, 1, 2] = gxy, gyy, gyz
    G[:, 2, 0], G[:, 2, 1], G[:, 2, 2] = gxz, gyz, gzz
    return (k * G).reshape(-1, 3)


def warped_operator(sx, R, T, bounds) -> np.ndarray:
    """As a09 warped_operator (body at the imaged pose), for any prism bounds, with the vectorized kernel."""
    A_q = prism_operator(bounds, (sx - T) @ R).reshape(-1, 3, 3)
    return np.einsum("ij,rjk,lk->ril", R, A_q, R).reshape(-1, 3)


def check(sx, bounds_list) -> dict:
    """Maximum relative difference from the released operators on survey points sx."""
    from inversion_helpers import prism_cell_operator
    import isolated_outcrop_1cm as ISO
    worst = 0.0
    for bnd in bounds_list:
        ref = prism_cell_operator(bnd, sx)
        worst = max(worst, float(np.abs(prism_operator(bnd, sx) - ref).max() / np.abs(ref).max()))
        R = ISO.imager_R((1.3, -0.7)); T = np.array([0.05, -0.03, 0.02])
        A_q = ISO.per_receiver_kernel(bnd, (sx - T) @ R)
        ref_w = np.einsum("ij,rjk,lk->ril", R, A_q, R).reshape(-1, 3)
        worst = max(worst, float(np.abs(warped_operator(sx, R, T, bnd) - ref_w).max() / np.abs(ref_w).max()))
    return {"max_rel_diff_vs_released_operator": worst, "n_bounds_checked": len(bounds_list), "n_rx": int(len(sx))}

"""Combined error budget for the isolated outcrop: a modified copy of ``analysis/a09_systematics.py::budget``.

What changed relative to a09 ``budget`` (release v1.0.0, commit 3498ead), and why
--------------------------------------------------------------------------------
1. ``budget_vectors`` returns the recovered vectors of every draw (noise-free ``Mnf`` and full ``Mrec``) instead of
   only their summary statistics, so that the angle of the mean vector, the flow-mean error and cross-outcrop
   statistics can be computed (Y6-02). With ``known_burial=False`` and the a09 seed label it consumes the random
   streams in exactly the order a09 does, so it reproduces the stored a09 numbers bit for bit (checked in
   ``y6_02_bias_decomposition.py``).
2. ``known_burial=True`` puts the drawn burial depth b into the inversion operator: the model body is the full
   buried prism (z from -0.5 - b to +0.5 in the engine frame) at the imaged (mis-posed) pose, so the depth is known
   exactly and only the pose, site field, tilt, drift, susceptibility and noise errors remain (Y6-03). The a09
   "burial known" rows used b = 0 instead.
3. ``outcrop_draw`` evaluates ONE outcrop draw from explicit inputs (burial, pose, site-field geometry, tilt axis,
   susceptibility error, drift seed, loop-offset azimuth) for several magnetization directions at once. It is used
   by the flow-mean and Sobol studies. Its physics is line-for-line that of a09 ``budget``; the only additions are
   (a) a loop offset in any azimuth (a09 always offsets along +x) and (b) inputs passed in instead of drawn inside.

4. Prism operators come from ``_fastprism`` (the same Geoana kernel functions, vectorized over receivers; agreement
   with the released ``prism_cell_operator`` / ``per_receiver_kernel`` is ~2e-16 relative, checked in each study).
   ``RELEASED_OPERATORS = True`` switches back to the released (200x slower) operators.

Nothing in ``engine/`` or ``analysis/`` is modified; the functions below call the released helpers.
"""
from __future__ import annotations

import numpy as np

from _common import A6, A9, ISO, CN, BND, MU0, M_NRM_MIN, SIGMA_B, prism_cell_operator, seed_for  # noqa: F401
from core import generate_pose_errors
import _fastprism as FP

RELEASED_OPERATORS = False


def prism_op(bounds, sx):
    return prism_cell_operator(bounds, sx) if RELEASED_OPERATORS else FP.prism_operator(bounds, sx)


# ------------------------------------------------------------------------------------------------ geometry helpers
def survey_points(dx: float, reverse: bool, offset_az: float = 0.0) -> np.ndarray:
    """a09 two_loops() (1 m and 3 m loops) offset by dx in azimuth offset_az, optionally each loop driven twice."""
    sx = A9.two_loops(dx=0.0)
    sx[:, 0] += dx * np.cos(offset_az)
    sx[:, 1] += dx * np.sin(offset_az)
    if reverse:
        n1 = len(A6.survey("drive_around"))
        l1, l3 = sx[:n1], sx[n1:]
        sx = np.vstack([l1, l1[::-1], l3, l3[::-1]])
    return sx


def nuisance(sx: np.ndarray, fit_tilt: bool) -> np.ndarray:
    """Site-field offsets and uniform gradient (8 columns), plus the six heading terms if fit_tilt (as a09)."""
    N = np.column_stack([A6.nuisance_columns(sx, "world_dc"), A6.gradient_basis(sx)[0]])
    if fit_tilt:
        psi = A6.headings(sx); c, sn = np.cos(psi), np.sin(psi)
        cols = []
        for w in (c, sn):
            for j in range(3):
                v = np.zeros((len(sx), 3)); v[:, j] = w; cols.append(v.reshape(-1))
        N = np.column_stack([N] + cols)
    return N


def body_bounds(b: float):
    """The exposed prism extended b metres below the ground surface (engine frame: ground at z = -0.5)."""
    return (BND[0], BND[1], BND[2], BND[3], BND[4] - b, BND[5])


def warped_operator_b(sx, R, T, bounds):
    """As a09 warped_operator, for any prism bounds: operator of the body at the imaged (wrong) pose."""
    if not RELEASED_OPERATORS:
        return FP.warped_operator(sx, R, T, bounds)
    Q = (sx - T) @ R
    A_q = ISO.per_receiver_kernel(bounds, Q)
    return np.einsum("ij,rjk,lk->ril", R, A_q, R).reshape(-1, 3)


# ------------------------------------------------------------------------------------------------ a09 budget, vectors kept
def budget_vectors(label: str, dx, R_site, pose, bmax, drift_rms, tilt_nT, chi, chi_err, n=150, B0=100.0,
                   root=0.0, fit_drift=False, dirs=None, fit_tilt=False, reverse=False, known_burial=False):
    """a09 ``budget`` with the per-draw vectors returned. ``label`` replaces a09's key in the seed labels:
    pass ``"a09:<key>"`` to reproduce a stored row, or a ``study:`` label for a new run."""
    assert not fit_drift, "fit_drift is not needed by these studies"
    sx = survey_points(dx, reverse)
    N = nuisance(sx, fit_tilt)
    dirs = A9.DIRS if dirs is None else dirs
    out = {}
    for dn, u in dirs.items():
        mt = M_NRM_MIN * u
        rng = np.random.default_rng(seed_for(f"{label}:{dn}"))
        np.random.seed(seed_for(f"{label}:{dn}:geom") % (2 ** 32))
        Mrec, Mnf = np.zeros((n, 3)), np.zeros((n, 3)); bs = np.zeros(n)
        for i in range(n):
            b = rng.uniform(0, bmax); bs[i] = b
            Gt = prism_op(body_bounds(b), sx)
            pdg, trans = generate_pose_errors(randomize=True, pointing_err_max_deg=(pose[0], pose[0]), range_err_max_m=pose[1])
            Rm, Tm = ISO.imager_R(pdg), np.asarray(trans, float)
            Gm = warped_operator_b(sx, Rm, Tm, body_bounds(b) if known_burial else BND)
            if root > 0:
                Gm = np.column_stack([Gm, warped_operator_b(sx, Rm, Tm, (BND[0], BND[1], BND[2], BND[3], BND[4] - root, BND[4]))])
            A = np.column_stack([Gm, N]); K = np.linalg.solve(A.T @ A, A.T)[:3]
            site, Bc = A9.dipole_field(sx, B0, R_site, rng)
            terr = A9.tilt_error(sx, Bc, np.degrees(tilt_nT / np.linalg.norm(Bc)), rng.uniform(0, 2 * np.pi))
            Mind = chi * Bc * 1e-9 / MU0
            chi_est = chi * np.exp(rng.normal(0.0, chi_err))
            corr = chi_est * Bc * 1e-9 / MU0
            dr = A9.ou_drift_exact(sx, drift_rms, rng)
            clean = Gt @ (mt + Mind) + site + terr
            Mnf[i] = K @ clean - corr
            Mrec[i] = K @ (clean + dr + rng.normal(0, SIGMA_B, clean.size)) - corr
        out[dn] = {"mt": mt, "Mnf": Mnf, "Mrec": Mrec, "b": bs}
    return out


# ------------------------------------------------------------------------------------------------ one outcrop, explicit inputs
class OutcropModel:
    """Evaluate one outcrop draw from explicit inputs. Caches the survey and nuisance columns per loop offset."""

    def __init__(self, dx, reverse, fit_tilt, R_site, B0, tilt_nT, chi, chi_err, drift_rms, dirs):
        self.dx, self.reverse, self.fit_tilt = dx, reverse, fit_tilt
        self.R_site, self.B0, self.tilt_nT, self.chi, self.chi_err, self.drift_rms = R_site, B0, tilt_nT, chi, chi_err, drift_rms
        self.dirs = dirs
        self.MT = np.array([M_NRM_MIN * u for u in dirs.values()])        # (nd, 3)
        self._cache = {}

    def geometry(self, offset_az):
        """Survey points for a loop offset of dx in azimuth offset_az. The nuisance columns (constant offsets, uniform
        gradient about the survey centroid, heading terms) are unchanged by a rigid translation, so they are built once."""
        if "base" not in self._cache:
            base = survey_points(0.0, self.reverse)
            self._cache["base"] = (base, nuisance(base, self.fit_tilt))
        base, N = self._cache["base"]
        return base + np.array([self.dx * np.cos(offset_az), self.dx * np.sin(offset_az), 0.0]), N

    def site_field(self, sx, az, mdir):
        """a09 dipole_field with its two random inputs given: source azimuth az and unit moment mdir."""
        el = np.radians(-30.0); c = sx.mean(0)
        src = c + self.R_site * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        m = mdir / np.linalg.norm(mdir)

        def B(p):
            r = p - src; d = np.linalg.norm(r, axis=-1, keepdims=True); u = r / d
            return A9.MU0_4PI * (3 * (u @ m)[..., None] * u - m) / d ** 3
        scale = self.B0 / np.linalg.norm(B(c[None])[0])
        return (B(sx) * scale).reshape(-1), B(c[None])[0] * scale

    def evaluate(self, b, az_deg, el_deg, trans, site_az, site_m, tilt_axis, eta, drift_seed, offset_az=0.0,
                 noise_seed=None, known_burial=False):
        """Recovered vectors (nd x 3) for every magnetization direction: noise-free (white noise and drift off),
        drift only (no white noise), and full (drift and white noise) if noise_seed is given.
        eta is the standard-normal susceptibility-error variate (log error = chi_err * eta)."""
        sx, N = self.geometry(offset_az)
        Gt = prism_op(body_bounds(b), sx)
        Rm, Tm = ISO.imager_R((az_deg, el_deg)), np.asarray(trans, float)
        Gm = warped_operator_b(sx, Rm, Tm, body_bounds(b) if known_burial else BND)
        A = np.column_stack([Gm, N]); K = np.linalg.solve(A.T @ A, A.T)[:3]
        site, Bc = self.site_field(sx, site_az, site_m)
        terr = A9.tilt_error(sx, Bc, np.degrees(self.tilt_nT / np.linalg.norm(Bc)), tilt_axis)
        Mind = self.chi * Bc * 1e-9 / MU0
        corr = self.chi * np.exp(self.chi_err * eta) * Bc * 1e-9 / MU0
        common = K @ (site + terr) + K @ (Gt @ Mind) - corr                 # direction-independent part
        sig = (K @ Gt) @ self.MT.T                                          # (3, nd)
        Mnf = (sig + common[:, None]).T
        out = {"Mnf": Mnf}
        if self.drift_rms > 0 or noise_seed is not None:
            dr = A9.ou_drift_exact(sx, self.drift_rms, np.random.default_rng(drift_seed)) if self.drift_rms > 0 else 0.0
            out["Mdrift"] = Mnf + (K @ dr if self.drift_rms > 0 else 0.0)
            if noise_seed is not None:
                e = np.random.default_rng(noise_seed).normal(0, SIGMA_B, sx.shape[0] * 3)
                out["Mrec"] = out["Mdrift"] + K @ e
        return out

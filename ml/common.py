"""Shared pieces of the amortized-inference study: geometry, a fast closed-form prism kernel, the
linear bases, the nuisance simulator and the classical (least-squares) estimators.

Everything physical is taken from the thesis code without modification:
  survey geometry      analysis/a09_systematics.two_loops (drive-around at 1 m + loop at 3 m, 1 cm spacing)
  offsets, gradient    analysis/a06_isolated_robustness.nuisance_columns('world_dc'), gradient_basis
  heading terms        the six cos/sin(psi) columns of a09.budget(fit_tilt=True)
  site field           a09.dipole_field (a dipole R m away, |B| = B0 at the survey centroid)
  tilt error           a09.tilt_error (full Rodrigues rotation in the rover frame)
  slow drift           engine/correlated_noise_sensitivity.make_noise (OU, L = 100 m), scaled to an exact rms about
                       the survey mean as a09.ou_drift_exact
  constants            engine/inversion_helpers: M_NRM_MIN (M_ref), SIGMA_B, seed_for
The only new physics code is `prism_kernel`, a vectorized closed form of the field of a uniformly magnetized
rectangular prism. `ml/validate_forward.py` checks it against the SimPEG integral operator that the thesis uses.

Units: magnetization in units of M_ref (= 2.4e-3 A/m) inside this package, fields in nT.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "engine"), os.path.join(ROOT, "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

_cwd = os.getcwd()
import a09_systematics as A9            # imports a06, which chdirs into engine/; restored below
import a06_isolated_robustness as A6
import correlated_noise_sensitivity as CN
from inversion_helpers import M_NRM_MIN, SIGMA_B, seed_for, prism_cell_operator
os.chdir(_cwd)

MREF = M_NRM_MIN                         # A/m
MU0_4PI = 100.0                          # nT per (A/m) for the dimensionless kernel below
BND = tuple(float(v) for v in A9.BND)    # exposed prism (-2, 2, -2, 2, -0.5, 0.5); ground surface at z = -0.5
DIRS3 = {k: np.asarray(v, float) for k, v in A9.DIRS.items()}
DIRS6 = {k: np.asarray(v, float) for k, v in A9.DIRS6.items()}

RESULTS = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figures")


# --------------------------------------------------------------------------- closed-form prism kernel
def _log_r_plus(a, r):
    """log(a + r) for r = |(.,.,.)| >= |a|, stable when a < 0 and a + r -> 0 (uses (r^2 - a^2)/(r - a))."""
    pos = a >= 0
    out = np.empty_like(r)
    out[pos] = np.log(a[pos] + r[pos])
    q = r[~pos] - a[~pos]
    out[~pos] = np.log((r[~pos] ** 2 - a[~pos] ** 2) / q)
    return out


def prism_kernel(bounds, rx):
    """Per-receiver 3x3 kernel A[r] (nT per unit M_ref) of a uniformly magnetized axis-aligned prism.

    B(r) = A[r] @ (M / M_ref). Closed form of the second derivatives of the Newtonian potential of a prism
    (Bhattacharyya 1964; Blakely 1995, Sec. 9.2): with x, y, z the corner coordinates relative to the receiver and
    s = (-1)^(i+j+k), N_xx = sum s atan(yz/(xR)), N_xy = -sum s ln(z+R), and cyclic permutations.
    Valid for receivers outside the prism. Overall sign and scale are fixed by `validate_forward.py` against SimPEG.
    """
    x0, x1, y0, y1, z0, z1 = bounds
    rx = np.asarray(rx, float)
    X = np.stack([x0 - rx[:, 0], x1 - rx[:, 0]], 1)        # (n, 2)
    Y = np.stack([y0 - rx[:, 1], y1 - rx[:, 1]], 1)
    Z = np.stack([z0 - rx[:, 2], z1 - rx[:, 2]], 1)
    n = len(rx)
    N = np.zeros((n, 3, 3))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = (-1.0) ** (i + j + k)
                x, y, z = X[:, i], Y[:, j], Z[:, k]
                R = np.sqrt(x * x + y * y + z * z)
                N[:, 0, 0] += s * np.arctan(y * z / (x * R))
                N[:, 1, 1] += s * np.arctan(x * z / (y * R))
                N[:, 2, 2] += s * np.arctan(x * y / (z * R))
                N[:, 0, 1] -= s * _log_r_plus(z, R)
                N[:, 0, 2] -= s * _log_r_plus(y, R)
                N[:, 1, 2] -= s * _log_r_plus(x, R)
    N[:, 1, 0] = N[:, 0, 1]; N[:, 2, 0] = N[:, 0, 2]; N[:, 2, 1] = N[:, 1, 2]
    return KERNEL_SIGN * MU0_4PI * MREF * N


KERNEL_SIGN = 1.0   # set by validate_forward.py (SimPEG convention); see ml/results/forward_validation.json


def operator(bounds, rx):
    """Station-major (3n x 3) operator, B = G @ (M / M_ref)."""
    return prism_kernel(bounds, rx).reshape(-1, 3)


def rot_az_el(az_deg, el_deg):
    """R = R_el @ R_az, identical to core.pose_transform / isolated_outcrop_1cm.imager_R."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    Ra = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1.0]])
    Re = np.array([[np.cos(el), 0, np.sin(el)], [0, 1.0, 0], [-np.sin(el), 0, np.cos(el)]])
    return Re @ Ra


def posed_operator(bounds, rx, R, T):
    """Operator of the prism after the rigid motion v -> R v + T (same algebra as a09.warped_operator)."""
    Q = (np.asarray(rx, float) - T) @ R
    A = prism_kernel(bounds, Q)
    return np.einsum("ij,rjk,lk->ril", R, A, R).reshape(-1, 3)


def buried_bounds(b):
    """The exposed prism extended b metres below the ground surface (z = -0.5)."""
    return (BND[0], BND[1], BND[2], BND[3], BND[4] - b, BND[5])


# --------------------------------------------------------------------------- survey and linear bases
class Survey:
    """Two-loop survey (thesis recommended design), loops offset dx from the outcrop centre, and every linear
    basis the estimators use. Column blocks (station-major, 3n rows):
      G    3   exposed prism, nominal pose
      E    3   constant site-field offsets (local frame)
      Ng   5   uniform curl- and divergence-free gradient
      H    6   heading terms cos(psi) I3, sin(psi) I3 (constant sensor tilt)
      S   30   ten 0.1 m slabs of buried root, 0-1 m below the surface (3 columns each)
      P   15   first-order pose derivatives d(G m)/d(az, el, tx, ty, tz) for m = e_x, e_y, e_z
      D    9   Legendre degree 1-3 in station order, per axis (slow drift)
    """
    def __init__(self, dx=0.5):
        self.dx = dx
        self.sx = A9.two_loops(dx=dx)
        self.n = len(self.sx)
        self.nd = 3 * self.n
        sx = self.sx
        self.G = operator(BND, sx)
        self.E = A6.nuisance_columns(sx, "world_dc")
        self.Ng = A6.gradient_basis(sx)[0]
        psi = A6.headings(sx); c, s = np.cos(psi), np.sin(psi)
        cols = []
        for w in (c, s):
            for j in range(3):
                v = np.zeros((self.n, 3)); v[:, j] = w; cols.append(v.reshape(-1))
        self.H = np.column_stack(cols)
        edges = np.round(np.arange(0.0, 1.0001, 0.1), 10)
        self.slab_edges = edges
        self.S = np.column_stack([operator((BND[0], BND[1], BND[2], BND[3], BND[4] - e1, BND[4] - e0), sx)
                                  for e0, e1 in zip(edges[:-1], edges[1:])])
        h = 1e-4
        P = []
        for k in range(5):
            dp = np.zeros(5); dp[k] = h
            Gp = self._posed(dp); Gm = self._posed(-dp)
            P.append((Gp - Gm) / (2 * h))           # (3n, 3): derivative for m = e_x, e_y, e_z
        self.P = np.column_stack(P)                   # units: nT per (deg or m) per M_ref
        t = np.linspace(-1.0, 1.0, self.n)
        cols = []
        for deg in (1, 2, 3):
            Lg = np.polynomial.legendre.Legendre.basis(deg)(t)
            for j in range(3):
                v = np.zeros((self.n, 3)); v[:, j] = Lg; cols.append(v.reshape(-1))
        self.D = np.column_stack(cols)
        self.N11 = np.column_stack([self.E, self.Ng])
        self.N17 = np.column_stack([self.E, self.Ng, self.H])
        # feature basis: orthonormal Q = [Q8, Qr] spanning [G E Ng H S P D]. Q8 spans the offsets and gradient
        # (8 linear nuisances with no useful prior); Qr spans the rest after projecting them out. The network sees
        # only Qr^T d, the maximal invariant to those 8 parameters (as are the LS11 and LS17 estimates).
        Q8, _ = np.linalg.qr(self.N11)
        A_rest = np.column_stack([self.G, self.H, self.S, self.P, self.D])
        A_rest = A_rest - Q8 @ (Q8.T @ A_rest)
        sc = np.linalg.norm(A_rest, axis=0)
        U, sv, _ = np.linalg.svd(A_rest / sc, full_matrices=False)
        keep = sv > sv[0] * 1e-10
        Ur = U[:, keep] - Q8 @ (Q8.T @ U[:, keep])  # re-orthogonalize (two-pass Gram-Schmidt)
        Ur, _ = np.linalg.qr(Ur)
        self.Q = np.column_stack([Q8, Ur])           # (3n, k)
        self.k = self.Q.shape[1]
        self.n_inv = 8                               # features are z[:, 8:]
        self.ext_singular_values = sv
        # least-squares operators (rows for M only), all computed from the full data
        self.est = {}
        self.cov = {}
        for name, A in self.designs().items():
            Kfull = np.linalg.solve(A.T @ A, A.T)
            self.est[name] = Kfull[:3]
            self.cov[name] = SIGMA_B ** 2 * np.linalg.inv(A.T @ A)[:3, :3]
        # every LS operator lies in span(Q), so it can be applied to the compressed data exactly
        self.est_z = {k: K @ self.Q for k, K in self.est.items()}
        for k, K in self.est.items():
            err = float(np.abs(self.est_z[k] @ self.Q.T - K).max() / np.abs(K).max())
            assert err < 1e-9, (k, err)
        # profile least squares over the buried extent: exact G(b) on a 0.05 m grid, 17 nuisance columns
        self.b_grid = np.round(np.arange(0.0, 1.0001, 0.05), 10)
        self.prof = []
        for b in self.b_grid:
            A = np.column_stack([operator(buried_bounds(b), sx), self.N17])
            Qb, _ = np.linalg.qr(A)
            self.prof.append((np.linalg.solve(A.T @ A, A.T)[:3], Qb))

    def _posed(self, p):
        return posed_operator(BND, self.sx, rot_az_el(p[0], p[1]), np.asarray(p[2:5], float))

    def designs(self):
        G = self.G
        root1 = self.S.reshape(self.nd, 10, 3).sum(1)
        return {"LS11": np.column_stack([G, self.N11]),
                "LS17": np.column_stack([G, self.N17]),
                "LS11_root1m": np.column_stack([G, root1, self.N11]),
                "LS17_root1m": np.column_stack([G, root1, self.N17])}

    def profile_ls(self, d):
        """Profile LS: for each b on the grid, LS with G(b) and 17 nuisance columns; keep the smallest residual."""
        best = None
        for (K, Qb), b in zip(self.prof, self.b_grid):
            r2 = d @ d - np.sum((Qb.T @ d) ** 2)
            if best is None or r2 < best[0]:
                best = (r2, K @ d, b)
        return best[1], best[2]


# --------------------------------------------------------------------------- nuisance simulator
PRIOR_FULL = dict(bmax=1.0, pose_deg=3.0, pose_m=0.15, drift_max=0.03, tilt_max=0.05,
                  site_B0=100.0, site_R=(530.0, 1060.0))
PRIOR_NONE = dict(bmax=0.0, pose_deg=0.0, pose_m=0.0, drift_max=0.0, tilt_max=0.0,
                  site_B0=0.0, site_R=(530.0, 1060.0))


def sample_m(rng, n):
    """Prior on M (units of M_ref): direction uniform on the sphere, ln|M| ~ N(ln 2, 0.35^2)."""
    u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u * np.exp(rng.normal(np.log(2.0), 0.35, size=(n, 1)))


def drift_batch(sx, rms, rng):
    """OU drift (L = 100 m) for each requested rms, minus its survey mean, scaled to that exact rms per axis
    (a09.ou_drift_exact, batched)."""
    rms = np.asarray(rms, float)
    d = CN.make_noise(sx, 100.0, len(rms), rng).reshape(len(rms), -1, 3)
    d = d - d.mean(1, keepdims=True)
    d *= (rms / np.sqrt(np.mean(d ** 2, axis=(1, 2))))[:, None, None]
    return d.reshape(len(rms), -1)


def simulate(sv, rng, n, prior, m=None, b=None, constant_site=False, return_full=False, profile=True):
    """n simulated surveys. Returns dict with true M (n,3) [M_ref units], nuisance draws, compressed data
    z = Q^T d (n,k), residual energy |(I - QQ^T) d|^2 / sigma^2, and all classical estimates.

    prior: dict as PRIOR_FULL. m: optional fixed (n,3) truth. b: optional fixed buried extent (scalar or (n,)).
    constant_site: site field = random constant + uniform gradient only (the linear-Gaussian 'noise-only' case).
    """
    sx = sv.sx
    M = sample_m(rng, n) if m is None else np.asarray(m, float)
    bb = rng.uniform(0.0, prior["bmax"], n) if b is None else np.broadcast_to(np.asarray(b, float), (n,)).copy()
    az = rng.uniform(-1, 1, n) * prior["pose_deg"]; el = rng.uniform(-1, 1, n) * prior["pose_deg"]
    T = rng.uniform(-1, 1, (n, 3)) * prior["pose_m"]
    fixed = prior.get("fixed", False)        # amplitudes at their maxima (thesis budgets) instead of U(0, max)
    drift_rms = (np.ones(n) if fixed else rng.uniform(0, 1, n)) * prior["drift_max"]
    tilt_nT = (np.ones(n) if fixed else rng.uniform(0, 1, n)) * prior["tilt_max"]
    Rsite = np.exp(rng.uniform(np.log(prior["site_R"][0]), np.log(prior["site_R"][1]), n))
    dr = drift_batch(sx, drift_rms, rng) if prior["drift_max"] > 0 else np.zeros((n, sv.nd))
    z = np.zeros((n, sv.k)); res = np.zeros(n)
    ests = {k: np.zeros((n, 3)) for k in list(sv.est) + ["profile"]}
    bhat = np.zeros(n)
    full = np.zeros((n, sv.nd)) if return_full else None
    for i in range(n):
        if prior["pose_deg"] > 0 or prior["pose_m"] > 0:
            Gt = posed_operator(buried_bounds(bb[i]), sx, rot_az_el(az[i], el[i]), T[i])
        else:
            Gt = operator(buried_bounds(bb[i]), sx) if bb[i] > 0 else sv.G
        d = Gt @ M[i]
        if constant_site:
            d = d + sv.E @ rng.normal(0, 50.0, 3) + sv.Ng @ rng.normal(0, 1.0, 5)
        elif prior["site_B0"] > 0:
            site, Bc = A9.dipole_field(sx, prior["site_B0"], Rsite[i], rng)
            d = d + site
            if tilt_nT[i] > 0:
                d = d + A9.tilt_error(sx, Bc, np.degrees(tilt_nT[i] / np.linalg.norm(Bc)), rng.uniform(0, 2 * np.pi))
        d = d + dr[i] + rng.normal(0.0, SIGMA_B, sv.nd)
        zi = sv.Q.T @ d
        z[i] = zi
        res[i] = (d @ d - zi @ zi) / SIGMA_B ** 2
        for k, K in sv.est.items():
            ests[k][i] = K @ d
        if profile:
            ests["profile"][i], bhat[i] = sv.profile_ls(d)
        if return_full:
            full[i] = d
    out = dict(M=M, b=bb, az=az, el=el, T=T, drift_rms=drift_rms, tilt_nT=tilt_nT, Rsite=Rsite,
               z=z, res=res, bhat_profile=bhat, **{f"est_{k}": v for k, v in ests.items()})
    if return_full:
        out["d"] = full
    return out


# --------------------------------------------------------------------------- metrics
def dir_err(Mhat, M):
    c = np.sum(Mhat * M, 1) / (np.linalg.norm(Mhat, axis=1) * np.linalg.norm(M, axis=1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def int_err_signed(Mhat, M):
    return 100.0 * (np.linalg.norm(Mhat, axis=1) / np.linalg.norm(M, axis=1) - 1.0)


def summary(a):
    a = np.asarray(a, float)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "std": float(a.std()), "sem": float(a.std() / np.sqrt(a.size)), "N": int(a.size)}

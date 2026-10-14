"""Y10: magnetization induced in the regolith by the site field, and its effect on the recovered outcrop magnetization.

Read-only use of the thesis repo engine (engine/, analysis/a06, a09). Everything is linear, so a regolith slab of
induced magnetization M_reg = chi_reg B_site / mu0 with a 4 x 4 m hole under the outcrop is the same as a "plug"
of -M_reg in the hole (y10_verify.py checks this against a +-1.2 km slab). Frame as the engine: ground surface at
z = -0.5 m, sensor at z = 0 (0.5 m above ground), outcrop prism x,y in +-2 m, z in -0.5..0.5 m.

Truth data  = outcrop prism (z from -0.5-b to 0.5) with M_nrm + M_b,  M_b = chi_b B/mu0, chi_b = 2e-3
            + plug in the hole (-M_reg)  + site field (dipole 530 m away at 100 nT, 918 m at 300 nT, scaled to |B| at the centroid; noise-free runs
              use only its uniform part)  + white noise 0.5 nT/axis (noisy runs).
Estimator   = recommended survey (loops 1 m + 3 m, sensor 0.5 m above ground, 1 cm spacing, loops centred or 0.5 m off
              centre), exposed 4 x 4 x 1 prism fitted, 3 world-frame offsets + 5 gradient terms fitted, M_b subtracted
              exactly (chi_b known; the thesis's own chi_b tolerance is a separate row of its budget).
Hole modes  S1     hole through the whole regolith thickness h (bedrock exposure); plug = -M_reg; b is the depth of
                   the NRM-carrying basalt (as the thesis), the rest of the hole is non-magnetic.
            S2     hole only as deep as the modelled body, min(b, h): the outcrop displaces regolith only to depth b.
            S1fill as S1 but the basalt that fills the hole below the NRM body has its induced term (chi_b).
Mitigations none | a  1 m root body fitted (as the thesis) | b  plug of known depth h along the fitted site-field
              direction, one scalar unknown | b_u20  as b with the site direction 20 deg off | b_hwrong  as b with a
              wrong assumed depth | c1.1 .. c2.0  plug subtracted with chi_reg known to a lognormal factor 1.1 / 1.2 / 1.4 / 2.0 (true site field vector, true depth).
"""
import os, sys, json, time
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "2")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
sys.path.insert(0, os.path.join(REPO, "analysis"))
import a09_systematics as A9                         # noqa  (imports a06, chdirs into engine/)
import a06_isolated_robustness as A6
from inversion_helpers import prism_cell_operator, seed_for, M_NRM_MIN, SIGMA_B

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.environ.get("Y10_OUT", "y10_results.json"))
SEEDTAG = os.environ.get("Y10_SEEDTAG", "")
MU0 = 4e-7 * np.pi
ZG = -0.5
BND = A9.BND
CHI_B = 2e-3
R_SITE = 530.0     # m at 100 nT; scaled as sqrt(B0/100) so the gradient-fit residual stays ~0.02 nT (thesis full tolerance)
N_DRAW = int(os.environ.get("Y10_N", "100"))
HS = (0.5, 1.0, 3.0, 5.0)
CHIS = tuple(float(v) for v in os.environ.get("Y10_CHIS", "0.01,0.02,0.04").split(","))
BS = tuple(float(v) for v in os.environ.get("Y10_BS", "100,300").split(","))
BURIAL = (0.0, 0.5, 1.0)
DIRS = A9.DIRS
MODES = tuple(os.environ.get("Y10_MODES", "S1,S2,S1fill").split(","))
HWRONG = {0.5: 1.0, 1.0: 3.0, 3.0: 1.0, 5.0: 3.0}


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def hole_op(sx, depth, top=ZG):
    return prism_cell_operator((-2.0, 2.0, -2.0, 2.0, top - depth, top), sx)


class Survey:
    """Operators for one loop placement."""

    def __init__(self, dx):
        self.dx = dx
        self.sx = sx = A9.two_loops(dx=dx)
        self.nD = 3 * len(sx)
        self.N = np.column_stack([A6.nuisance_columns(sx, "world_dc"), A6.gradient_basis(sx)[0]])     # 8 columns
        self.Gexp = prism_cell_operator(BND, sx)
        self.Groot = prism_cell_operator((BND[0], BND[1], BND[2], BND[3], BND[4] - 1.0, BND[4]), sx)
        self.Gt = {b: prism_cell_operator((BND[0], BND[1], BND[2], BND[3], BND[4] - b, BND[5]), sx) for b in BURIAL}
        depths = sorted(set(HS) | {0.5, 1.0})
        self.Gh = {d: hole_op(sx, d) for d in depths}
        # basalt filling the hole below the NRM body (S1fill): z from ZG-h to ZG-b
        self.Glow = {(h, b): prism_cell_operator((-2, 2, -2, 2, ZG - h, ZG - b), sx) for h in HS for b in BURIAL if h > b}
        self.BB = np.column_stack([self.Gexp, self.Groot, self.N])           # 14 columns: Gexp(0:3) Groot(3:6) N(6:14)
        self.Gram = self.BB.T @ self.BB
        self.cross = {d: self.BB.T @ G for d, G in self.Gh.items()}
        self.gg = {d: G.T @ G for d, G in self.Gh.items()}
        self.sel_base = list(range(3)) + list(range(6, 14))
        self.sel_a = list(range(14))
        self.rms_out_nrm = {dn: rms(self.Gexp @ (M_NRM_MIN * u)) for dn, u in DIRS.items()}


def solve_sel(S, Pd, sel):
    A = S.Gram[np.ix_(sel, sel)]
    x = np.linalg.solve(A, Pd[:, sel].T).T
    return x


def solve_b(S, Pd, Dx, depth, U):
    """Per draw: base columns + one plug column G_h u_i. Returns x (n x 12) with x[:, :3] = M, x[:, 3:6] offsets."""
    sel = S.sel_base
    n = Pd.shape[0]
    A0 = S.Gram[np.ix_(sel, sel)]
    cr = S.cross[depth][sel]                  # 11 x 3
    gg = S.gg[depth]
    X = np.zeros((n, len(sel) + 1))
    for i in range(n):
        u = U[i]
        cu = cr @ u
        A = np.zeros((len(sel) + 1, len(sel) + 1))
        A[:-1, :-1] = A0; A[:-1, -1] = cu; A[-1, :-1] = cu; A[-1, -1] = u @ gg @ u
        rhs = np.append(Pd[i, sel], Dx[i] @ u)
        X[i] = np.linalg.solve(A, rhs)
    return X


def rotate_off(u, deg, rng):
    """Rotate unit vector u by `deg` degrees about a random axis perpendicular to it."""
    a = rng.normal(size=3); a -= (a @ u) * u; a /= np.linalg.norm(a)
    t = np.radians(deg)
    return u * np.cos(t) + a * np.sin(t)


def mstats(Mrec, mt):
    m = A6.metrics(Mrec, mt)
    return {"dir_mean": m["dir_err_deg"]["mean"], "dir_p95": m["dir_err_deg"]["p95"], "dir_max": m["dir_err_deg"]["max"],
            "dM_mean": m["dM_signed_pct"]["mean"], "dM_std": m["dM_signed_pct"]["std"]}


def estimates(S, D, Bc, Mb, mt, Gh_true, depth_true, chi_reg, rng_c, hwrong_depth, rng_u, with_wrong=True, with_c=True):
    """All estimators for one data set D (nD x n). Returns dict mitigation -> Mrec (n x 3)."""
    Dt = D.T                                                  # n x nD
    Pd = Dt @ S.BB                                            # n x 14
    out = {}
    out["none"] = solve_sel(S, Pd, S.sel_base)[:, :3] - Mb
    out["a"] = solve_sel(S, Pd, S.sel_a)[:, :3] - Mb
    # b: plug of known depth along the fitted site-field direction
    x1 = solve_sel(S, Pd, S.sel_base)
    off = x1[:, 3:6]
    U = off / np.linalg.norm(off, axis=1, keepdims=True)
    Dx = Dt @ S.Gh[depth_true]
    xb = solve_b(S, Pd, Dx, depth_true, U)
    out["b"] = xb[:, :3] - Mb
    out["b_scalar"] = xb[:, -1]
    U20 = np.array([rotate_off(u, 20.0, rng_u) for u in U])
    out["b_u20"] = solve_b(S, Pd, Dx, depth_true, U20)[:, :3] - Mb
    if with_wrong:
        Dxw = Dt @ S.Gh[hwrong_depth]
        out["b_hwrong"] = solve_b(S, Pd, Dxw, hwrong_depth, U)[:, :3] - Mb
    return out, Pd


def run_group(S, B0, dn, b, rec, t0):
    mt = M_NRM_MIN * DIRS[dn]
    key = f"y10{SEEDTAG}:{S.dx}:{B0}:{dn}:{b}"
    rng = np.random.default_rng(seed_for(key))
    rng_c = np.random.default_rng(seed_for(key + ":c")); rng_u = np.random.default_rng(seed_for(key + ":u"))
    n = N_DRAW; nst = len(S.sx)
    site = np.zeros((S.nD, n)); Bc = np.zeros((n, 3))
    for i in range(n):
        f, bc = A9.dipole_field(S.sx, B0, R_SITE * np.sqrt(B0 / 100.0), rng)
        site[:, i] = f; Bc[i] = bc
    noise = rng.normal(0.0, SIGMA_B, size=(S.nD, n))
    zc = rng_c.normal(size=n)
    uni = np.tile(Bc, (1, nst)).reshape(n, -1).T            # uniform part of the site field, station-major
    Mb = CHI_B * Bc * 1e-9 / MU0                             # n x 3, A/m
    d_out = S.Gt[b] @ (mt[:, None] + Mb.T)                   # nD x n
    # regolith field rms vs outcrop (3 components, along the survey)
    for mode in MODES:
        for h in HS:
            if mode == "S2":
                depth = min(b, h)
                if depth == 0:
                    continue
            else:
                depth = h
            Gh = S.Gh[depth]
            hw = HWRONG[h] if mode == "S1" else None
            for chi in (0.0,) + CHIS:
                if chi == 0.0 and mode != "S1":
                    continue                                    # controls (no regolith effect) once per h
                Mreg = chi * Bc * 1e-9 / MU0                    # n x 3
                d_reg = Gh @ (-Mreg.T)                          # nD x n
                if mode == "S1fill" and (h, b) in S.Glow:
                    d_reg = d_reg + S.Glow[(h, b)] @ Mb.T
                for noisy in (False, True):
                    D = d_out + d_reg + (site + noise if noisy else uni)
                    est, Pd = estimates(S, D, Bc, Mb, mt, Gh, depth, chi, rng_c, hw, rng_u, with_wrong=(hw is not None))
                    # c: subtract the modelled plug with chi known to a lognormal factor
                    if chi > 0:
                        for tag, fac in (("c1.1", 1.1), ("c1.2", 1.2), ("c1.4", 1.4), ("c2.0", 2.0)):
                            chi_est = chi * np.exp(np.log(fac) * zc)
                            Dc = D + Gh @ (chi_est[None, :] * Bc.T * 1e-9 / MU0)
                            Pdc = Dc.T @ S.BB
                            est[tag] = solve_sel(S, Pdc, S.sel_base)[:, :3] - Mb
                    for mit, Mrec in est.items():
                        if mit == "b_scalar":
                            continue
                        row = {"dx": S.dx, "B": B0, "dir": dn, "b": b, "mode": mode, "h": h, "depth": depth, "chi_reg": chi,
                               "mit": mit, "noisy": noisy, **mstats(Mrec, mt)}
                        if mit == "b":
                            sc = est["b_scalar"]
                            row["plug_scalar_over_true"] = float(np.mean(-sc / np.linalg.norm(Mreg, axis=1))) if chi > 0 else None
                            row["plug_scalar_sd_A_per_m"] = float(np.std(sc))
                        rec.append(row)
                if chi > 0 and mode == "S1" and True:
                    rec.append({"dx": S.dx, "B": B0, "dir": dn, "b": b, "mode": "fieldrms", "h": h, "depth": depth, "chi_reg": chi,
                                "rms_reg_nT": rms(d_reg), "rms_out_nrm_nT": S.rms_out_nrm[dn], "rms_out_withb_nT": rms(S.Gt[b] @ mt),
                                "rms_out_induced_nT": rms(S.Gt[b] @ Mb.T), "Mreg_over_Mref": float(np.mean(np.linalg.norm(Mreg, axis=1)) / M_NRM_MIN)})
    print(f"[{time.time()-t0:6.0f}s] dx={S.dx} B={B0:.0f} {dn} b={b}: rows {len(rec)}", flush=True)


def main():
    t0 = time.time(); rec = []
    for dx in (0.0, 0.5):
        S = Survey(dx)
        print(f"operators built dx={dx} ({time.time()-t0:.0f}s)", flush=True)
        for B0 in BS:
            for dn in DIRS:
                for b in BURIAL:
                    run_group(S, B0, dn, b, rec, t0)
            json.dump({"meta": {"n_draw": N_DRAW, "chi_b": CHI_B, "R_site": R_SITE, "ZG": ZG, "partial": True}, "rows": rec}, open(OUT, "w"))
    json.dump({"meta": {"n_draw": N_DRAW, "chi_b": CHI_B, "R_site": R_SITE, "ZG": ZG, "partial": False,
                        "M_ref": M_NRM_MIN, "sigma_B": SIGMA_B}, "rows": rec}, open(OUT, "w"))
    print("done", len(rec), "rows", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

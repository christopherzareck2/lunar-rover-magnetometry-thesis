"""a09 - Isolated outcrop: systematic errors and combined error budgets (paper Sections 4.2 and 5.1-5.5, Table 5).

Same geometry, noise and solver as a02/a06 (prism 4 x 4 x 1 m, body z = -0.5..0.5, sensor at z = 0, i.e. 0.5 m above
the ground, 1 m standoff, 1 cm sampling, sigma_B = 0.50 nT per axis, closed-form weighted least squares). The
recommended survey is the drive-around at 1 m plus a second loop at 3 m, with local-frame (site-field) offsets and a
uniform gradient (5 terms) fitted.

Parts (run with --parts, one character each; results merge into results/analysis/a09_systematics.json)
  J  constant sensor tilt in a static ambient field (30 and 300 nT), loops centred.
  T  as J at sensor heights 0.25 and 1.0 m and with the loops 0.5 m off centre (300 nT).
  t  tilt with six heading-dependent terms fitted (cos/sin of heading times a vector), centred and off centre.
  K  realistic site field: a dipole 30 m-1 km away, 30-300 nT at the outcrop; residual after the gradient fit.
  M  heterogeneous ground below the outcrop vs depth to its top, single loop, no offsets fitted.
  2  as M with the recommended two-loop survey and offsets plus gradient fitted.
  N  magnetization induced by the site field (chi = 1-5e-3), noise-free bias if not removed.
  W  three-layer wall with a site field from a distant dipole, offsets and gradient fitted over both traverses.
  Budgets (function `budget`): white noise, site field (dipole, residual ~0.02 nT at full tolerance), pose error,
  buried extent, slow drift (OU, L = 100 m, scaled to an exact rms about the survey mean), tilt, induced magnetization
  removed with a susceptibility known to a lognormal factor.
  U, V, Y  every source at full, half and quarter tolerance, loops 0.5 m off centre.
  X  half tolerance, loops centred.
  G  U/V/Y-like budgets with a 1 m root body fitted.
  Z  each source alone at full tolerance, loops off centre.
  q  the requirement set of the paper's Conclusions, six magnetization directions (loops 0.1 or 0.5 m off centre).
  S  burial and pose bias vs loop offset from centre (0-0.5 m).
  R  burial with a root body fitted (1 m or 2 m), centred and off centre.
  D, d  drift alone at 0.01-0.03 nT, centred and off centre.  E  the same with a linear drift fitted.
  r  drift with each loop driven twice in opposite directions.
  i j  parts f and h with chi = 5e-3 SI (sensitivity to a higher mare-basalt susceptibility).
  f g h  the requirement set with both mitigations (each loop driven both ways, tilt heading terms fitted); Table 5 'Mitigated set' rows.

Run:  python analysis/a09_systematics.py --parts JTtKM2NWUVYXGZqSRDdErfghij
      (about 3 hours in total; A09_OUT=<path> writes a separate JSON for parallel runs)
Out:  results/analysis/a09_systematics.json
"""
import os, sys, json, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import a06_isolated_robustness as A6                       # chdirs into engine/ and sets sys.path
import discretize
import isolated_outcrop_1cm as ISO
import correlated_noise_sensitivity as CN
from inversion_helpers import prism_cell_operator, _sim_for, dir_err_deg, seed_for, M_NRM_MIN, SIGMA_B
from core import generate_pose_errors

OUT = os.environ.get("A09_OUT", os.path.join(ROOT, "results", "analysis", "a09_systematics.json"))
BND = A6.BND
MU0_4PI = 100.0          # nT m^3 / (A m^2): mu0/4pi = 1e-7 T m/A = 100 nT m / A
DIRS = {"-z": np.array([0, 0, -1.0]), "+x": np.array([1.0, 0, 0]), "inc45": np.array([0.7071068, 0, -0.7071068])}


def two_loops(height=0.5, dx=0.0):
    """Drive-around at 1 m + second loop at 3 m, loops centred at (dx, 0)."""
    l1 = A6.survey("drive_around", height=height); l3 = A6.survey("drive_around", height=height, standoff=3.0)
    s = np.vstack([l1, l3]); s[:, 0] += dx
    return s


def dipole_field(sxyz, B0, R, rng):
    """Field (station-major, nT) of a dipole R m away (30 deg below horizon, random azimuth, random moment
    direction), scaled so that |B| at the survey centroid is B0 nT."""
    az = rng.uniform(0, 2 * np.pi); el = np.radians(-30.0)
    c = sxyz.mean(0)
    src = c + R * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    m = rng.normal(size=3); m /= np.linalg.norm(m)
    def B(p):
        r = p - src; d = np.linalg.norm(r, axis=-1, keepdims=True); u = r / d
        return MU0_4PI * (3 * (u @ m)[..., None] * u - m) / d ** 3
    scale = B0 / np.linalg.norm(B(c[None])[0])
    return (B(sxyz) * scale).reshape(-1), B(c[None])[0] * scale


def design(sxyz, grad=True):
    G = prism_cell_operator(BND, sxyz)
    A, _ = A6.augmented(G, A6.nuisance_columns(sxyz, "world_dc"))
    if grad:
        A, _ = A6.augmented(A, A6.gradient_basis(sxyz)[0])
    return G, A6.lsK(A)[:3]


def tilt_error(sxyz, Bamb, dtheta_deg, axis_angle):
    """Station-major error field from a constant rover-frame tilt (about a horizontal rover axis)."""
    psi = A6.headings(sxyz)
    a = np.radians(dtheta_deg)
    k = np.array([np.cos(axis_angle), np.sin(axis_angle), 0.0])
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    Rerr = np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx
    c, s = np.cos(psi), np.sin(psi)
    Rpsi = np.zeros((len(psi), 3, 3)); Rpsi[:, 0, 0] = c; Rpsi[:, 0, 1] = -s; Rpsi[:, 1, 0] = s; Rpsi[:, 1, 1] = c; Rpsi[:, 2, 2] = 1
    M = np.einsum("nij,jk,nlk->nil", Rpsi, Rerr - np.eye(3), Rpsi)      # R_psi (R_err - I) R_psi^T
    return np.einsum("nij,j->ni", M, Bamb).reshape(-1)


def part_J(out):
    sx = two_loops(); G, K = design(sx, grad=True)
    d0 = G @ (M_NRM_MIN * DIRS["-z"])
    rows = []
    for Bmag in (30.0, 300.0):
        for dth in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
            rng = np.random.default_rng(seed_for(f"a09:J:{Bmag}:{dth}"))
            res = {}
            for dn, u in DIRS.items():
                mt = M_NRM_MIN * u; d0 = G @ mt
                n = 300; Mrec = np.zeros((n, 3))
                for i in range(n):
                    Bamb = rng.normal(size=3); Bamb *= Bmag / np.linalg.norm(Bamb)
                    e = tilt_error(sx, Bamb, dth, rng.uniform(0, 2 * np.pi))
                    Mrec[i] = K @ (d0 + e + rng.normal(0, SIGMA_B, d0.size))
                res[dn] = A6.metrics(Mrec, mt)
            rows.append({"B_amb_nT": Bmag, "tilt_deg": dth, "err_nT": Bmag * np.radians(dth), **res})
            print(f"[J] |B|={Bmag:.0f} nT tilt {dth} deg ({Bmag*np.radians(dth):.3f} nT): " +
                  "  ".join(f"{dn} {res[dn]['dir_err_deg']['mean']:.1f}" for dn in DIRS), flush=True)
    out["J_attitude_tilt"] = rows


def part_T(out):
    """J repeated where the loop symmetry is broken: sensor heights 0.25 and 1.0 m, loops 0.5 m off centre."""
    res = {}
    for cn, kw in {"h0.25": dict(height=0.25), "h1.0": dict(height=1.0), "offcentre0.5m": dict(dx=0.5)}.items():
        sx = two_loops(**kw); G, K = design(sx, grad=True)
        rows = []
        for dth in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
            rng = np.random.default_rng(seed_for(f"a09:T:{cn}:{dth}"))
            r = {"B_amb_nT": 300.0, "tilt_deg": dth}
            for dn, u in DIRS.items():
                mt = M_NRM_MIN * u; d0 = G @ mt; n = 300; Mrec = np.zeros((n, 3))
                for i in range(n):
                    Bamb = rng.normal(size=3); Bamb *= 300.0 / np.linalg.norm(Bamb)
                    Mrec[i] = K @ (d0 + tilt_error(sx, Bamb, dth, rng.uniform(0, 2 * np.pi)) + rng.normal(0, SIGMA_B, d0.size))
                r[dn] = A6.metrics(Mrec, mt)
            rows.append(r)
            print(f"[T {cn}] 300 nT tilt {dth} deg: " + "  ".join(f"{dn} {r[dn]['dir_err_deg']['mean']:.1f}" for dn in DIRS), flush=True)
        res[cn] = rows
    out["T_attitude_tilt_asymmetric"] = res


def part_K(out):
    res = {}
    cases = {"h0.5": dict(height=0.5, dx=0.0), "h0.25": dict(height=0.25, dx=0.0), "h1.0": dict(height=1.0, dx=0.0),
             "offcentre0.5m": dict(height=0.5, dx=0.5)}
    for cn, kw in cases.items():
        sx = two_loops(**kw)
        G, Kg = design(sx, grad=True); _, K0 = design(sx, grad=False)
        rows = []
        for B0 in (30.0, 100.0, 300.0):
            for R in (30.0, 100.0, 300.0, 1000.0):
                rng = np.random.default_rng(seed_for(f"a09:K:{cn}:{B0}:{R}"))
                r = {"B0_nT": B0, "R_m": R}
                for dn, u in DIRS.items():
                    mt = M_NRM_MIN * u; d0 = G @ mt; n = 200
                    Mg, M0 = np.zeros((n, 3)), np.zeros((n, 3)); curv = []
                    for i in range(n):
                        b, _ = dipole_field(sx, B0, R, rng)
                        d = d0 + b + rng.normal(0, SIGMA_B, d0.size)
                        Mg[i] = Kg @ d; M0[i] = K0 @ d
                        if dn == "-z":   # residual after the best offsets + gradient fit (curvature and higher)
                            Bg = A6.gradient_basis(sx)[0]
                            N = np.column_stack([A6.nuisance_columns(sx, "world_dc"), Bg])
                            cf, *_ = np.linalg.lstsq(N, b, rcond=None)
                            curv.append(float(np.sqrt(np.mean((b - N @ cf) ** 2))))
                    r[dn] = {"offsets_and_gradient": A6.metrics(Mg, mt), "offsets_only": A6.metrics(M0, mt)}
                    if curv:
                        r["residual_after_gradient_rms_nT"] = float(np.mean(curv))
                rows.append(r)
                print(f"[K {cn}] B0={B0:.0f} nT R={R:.0f} m (resid {r['residual_after_gradient_rms_nT']:.4f} nT): " +
                      "  ".join(f"{dn} {r[dn]['offsets_and_gradient']['dir_err_deg']['mean']:.1f}/{r[dn]['offsets_only']['dir_err_deg']['mean']:.1f}"
                                for dn in DIRS), flush=True)
        res[cn] = rows
    out["K_site_field_dipole"] = res


def warped_operator(sxyz, R, T):
    """Operator of the exposed prism at the imaged (wrong) pose, as ISO.warped_recovery."""
    Q = (sxyz - T) @ R
    A_q = ISO.per_receiver_kernel(BND, Q)
    return np.einsum("ij,rjk,lk->ril", R, A_q, R).reshape(-1, 3)


def part_M(out):
    sx = A6.survey("drive_around", spacing=0.05)
    sig = SIGMA_B * np.sqrt(0.01 / 0.05)
    G = prism_cell_operator(BND, sx); K = A6.lsK(G)
    mt = M_NRM_MIN * DIRS["-z"]; d0 = G @ mt
    res = {}
    for depth in (0.0, 1.0, 3.0):
        ztop = BND[4] - depth
        for block, nz in (("cell_1m", 5), ("block_2m", 1), ("block_5m", 1)):
            bs = 1.0 if block == "cell_1m" else float(block.split("_")[1][:-1])
            nxy = int(round(30 / bs))
            mesh = discretize.TensorMesh([np.full(nxy, bs), np.full(nxy, bs), np.full(nz, 5.0 / nz)],
                                         origin=[-15.0, -15.0, ztop - 5.0])
            nC = mesh.n_cells
            Gz = _sim_for(mesh, sx, nC).G[:, 2 * nC:3 * nC].copy()       # response to Mz of each cell
            rows = []
            for floor in (1.0, 0.8, 0.5, 0.2, 0.05, 0.01):
                rng = np.random.default_rng(seed_for(f"a09:M:{depth}:{block}:{floor}"))
                n = 400; th = np.zeros(n); sd = []; ratio = []
                for i in range(n):
                    I = 10 ** rng.uniform(np.log10(floor), 0.0, nC) if floor < 1 else np.ones(nC)
                    dI = I - I.mean(); sd.append(dI.std())
                    b = Gz @ (-M_NRM_MIN * dI)
                    ratio.append(np.sqrt(np.mean(b ** 2)) / np.sqrt(np.mean(d0 ** 2)))
                    th[i] = dir_err_deg(K @ (d0 + b + rng.normal(0, sig, d0.size)), mt)
                rows.append({"floor": floor, "cell_sd_over_Mref": float(np.mean(sd)), "bg_to_signal": float(np.mean(ratio)),
                             "dir_err_deg": A6.stats(th)})
                print(f"[M depth {depth} {block}] floor {floor}: sd {np.mean(sd):.3f} Mref, bg/signal {np.mean(ratio):.2f}, "
                      f"dir mean {th.mean():.1f} deg", flush=True)
            # interpolate the SD at which the mean error crosses 5 deg
            e = np.array([r["dir_err_deg"]["mean"] for r in rows]); s = np.array([r["cell_sd_over_Mref"] for r in rows])
            idx = np.where((e[:-1] < 5) & (e[1:] >= 5))[0]
            x = float(s[idx[0]] + (5 - e[idx[0]]) * (s[idx[0] + 1] - s[idx[0]]) / (e[idx[0] + 1] - e[idx[0]])) if len(idx) else None
            res[f"depth_{depth:g}m:{block}"] = {"rows": rows, "sd_at_5deg": x,
                                                  "status": "interpolated" if len(idx) else ("not_reached" if e.max() < 5 else "below_lowest")}
            print(f"[M depth {depth} {block}] 5 deg crossing at cell SD = {x}", flush=True)
    out["M_background_depth"] = res


# ---------------------------------------------------------------- combined budgets
MU0 = 4e-7 * np.pi


def ou_drift_exact(sx, rms, rng):
    """Slow drift (OU, L = 100 m) scaled so that its rms about the survey mean is exactly `rms` nT per axis."""
    d = CN.make_noise(sx, 100.0, 1, rng)[0].reshape(-1, 3)
    if rms == 0:
        return np.zeros(d.size)
    d = d - d.mean(0)
    return (d * (rms / np.sqrt(np.mean(d ** 2)))).reshape(-1)


def site_residual(sx, b):
    Nb = np.column_stack([A6.nuisance_columns(sx, "world_dc"), A6.gradient_basis(sx)[0]])
    cf, *_ = np.linalg.lstsq(Nb, b, rcond=None)
    return float(np.sqrt(np.mean((b - Nb @ cf) ** 2)))


def budget(out, key, dx, R_site, pose, bmax, drift_rms, tilt_nT, chi, chi_err, n=150, B0=100.0, root=0.0, fit_drift=False, dirs=None, fit_tilt=False, reverse=False):
    """Combined budget: exact drift rms about the survey mean, tilt given as tilt*|B| in nT,
    loops offset dx from the outcrop centre, and magnetization induced by the site field (volume susceptibility
    chi), removed using a susceptibility estimate that is wrong by a lognormal factor with sigma_ln = chi_err."""
    sx = two_loops(dx=dx)
    if reverse:               # each loop driven twice, the second time in the opposite direction
        n1 = len(A6.survey("drive_around"))
        l1, l3 = sx[:n1], sx[n1:]
        sx = np.vstack([l1, l1[::-1], l3, l3[::-1]])
    N = np.column_stack([A6.nuisance_columns(sx, "world_dc"), A6.gradient_basis(sx)[0]])
    if fit_drift:             # a linear drift in time per axis over the whole survey (stations in driving order)
        t = np.linspace(-1.0, 1.0, len(sx))
        N = np.column_stack([N] + [np.eye(3)[j][None, :].repeat(len(sx), 0).reshape(-1) * np.repeat(t, 3) for j in range(3)])
    if fit_tilt:              # heading-dependent terms cos(psi) a + sin(psi) b (6 columns) absorb a constant sensor tilt
        psi = A6.headings(sx); c, sn = np.cos(psi), np.sin(psi)
        cols = []
        for w in (c, sn):
            for j in range(3):
                v = np.zeros((len(sx), 3)); v[:, j] = w; cols.append(v.reshape(-1))
        N = np.column_stack([N] + cols)
    dirs = DIRS if dirs is None else dirs
    res = {"config": dict(reverse_loops=reverse, fit_heading_terms=fit_tilt, fit_linear_drift=fit_drift, dx_m=dx, site_B0_nT=B0, site_R_m=R_site, pose=pose, bmax_m=bmax, drift_rms_nT=drift_rms,
                          tilt_times_B_nT=tilt_nT, chi_SI=chi, chi_err_sigma_ln=chi_err, n=n, root_fit_m=root)}
    for dn, u in dirs.items():
        mt = M_NRM_MIN * u
        rng = np.random.default_rng(seed_for(f"a09:{key}:{dn}"))
        np.random.seed(seed_for(f"a09:{key}:{dn}:geom") % (2 ** 32))
        Mrec, Mnf = np.zeros((n, 3)), np.zeros((n, 3)); resid, drms, ind = [], [], []
        for i in range(n):
            b = rng.uniform(0, bmax)
            Gt = prism_cell_operator((BND[0], BND[1], BND[2], BND[3], BND[4] - b, BND[5]), sx)
            pdg, trans = generate_pose_errors(randomize=True, pointing_err_max_deg=(pose[0], pose[0]), range_err_max_m=pose[1])
            Gm = warped_operator(sx, ISO.imager_R(pdg), np.asarray(trans, float))
            if root > 0:          # buried part fitted as a separate body with its own magnetization, same imaged pose
                Rm, Tm = ISO.imager_R(pdg), np.asarray(trans, float)
                Aq = ISO.per_receiver_kernel((BND[0], BND[1], BND[2], BND[3], BND[4] - root, BND[4]), (sx - Tm) @ Rm)
                Gm = np.column_stack([Gm, np.einsum("ij,rjk,lk->ril", Rm, Aq, Rm).reshape(-1, 3)])
            A = np.column_stack([Gm, N]); K = np.linalg.solve(A.T @ A, A.T)[:3]
            site, Bc = dipole_field(sx, B0, R_site, rng)
            resid.append(site_residual(sx, site))
            terr = tilt_error(sx, Bc, np.degrees(tilt_nT / np.linalg.norm(Bc)), rng.uniform(0, 2 * np.pi))
            Mind = chi * Bc * 1e-9 / MU0                         # A/m, along the site field
            ind.append(float(np.linalg.norm(Mind) / M_NRM_MIN))
            chi_est = chi * np.exp(rng.normal(0.0, chi_err))
            corr = chi_est * Bc * 1e-9 / MU0
            dr = ou_drift_exact(sx, drift_rms, rng); drms.append(drift_rms)
            clean = Gt @ (mt + Mind) + site + terr
            Mnf[i] = K @ clean - corr
            Mrec[i] = K @ (clean + dr + rng.normal(0, SIGMA_B, clean.size)) - corr
        r = {"all_sources": A6.metrics(Mrec, mt), "noise_free_bias": A6.metrics(Mnf, mt),
             "site_residual_rms_nT": float(np.mean(resid)), "induced_over_Mref": float(np.mean(ind))}
        res[dn] = r
        print(f"[{key} {dn}] dir mean {r['all_sources']['dir_err_deg']['mean']:.2f} (p95 {r['all_sources']['dir_err_deg']['p95']:.1f}) "
              f"signed dM {r['all_sources']['dM_signed_pct']['mean']:+.0f}% | bias {r['noise_free_bias']['dir_err_deg']['mean']:.1f} | "
              f"site resid {r['site_residual_rms_nT']:.4f} nT, induced {r['induced_over_Mref']:.3f} Mref", flush=True)
    out[key] = res


CHI = 2e-3     # SI volume susceptibility of mare basalt, order of magnitude (Rochette et al. 2010)


def part_U(out):
    """Every source at its stated tolerance: loops 0.5 m off centre, site-field residual ~0.02 nT (100 nT from 530 m),
    pose 3 deg / 0.15 m, b <= 1 m, drift 0.03 nT rms, tilt*|B| = 0.05 nT, induced magnetization removed with chi known
    to a factor of 2."""
    budget(out, "U_budget_full_offcentre", 0.5, 530.0, (3.0, 0.15), 1.0, 0.03, 0.05, CHI, np.log(2.0))


def part_V(out):
    """Every source at half its tolerance (site residual ~0.01 nT from 750 m, 1.5 deg / 0.075 m, b <= 0.5 m,
    0.015 nT drift, 0.025 nT tilt, chi to a factor of 1.4), loops 0.5 m off centre."""
    budget(out, "V_budget_half_offcentre", 0.5, 750.0, (1.5, 0.075), 0.5, 0.015, 0.025, CHI, np.log(1.4))


def part_X(out):
    """As V with the loops centred (reference: the symmetric geometry)."""
    budget(out, "X_budget_half_centred", 0.0, 750.0, (1.5, 0.075), 0.5, 0.015, 0.025, CHI, np.log(1.4))


def part_Y(out):
    """As V with a quarter of each tolerance (site from 1060 m, 0.75 deg / 0.04 m, b <= 0.25 m, 0.0075 nT drift,
    0.0125 nT tilt, chi to a factor of 1.2)."""
    budget(out, "Y_budget_quarter_offcentre", 0.5, 1060.0, (0.75, 0.0375), 0.25, 0.0075, 0.0125, CHI, np.log(1.2))


def part_N(out):
    """Induced magnetization alone: bias of the recovered NRM when the magnetization induced by the site field is
    not removed, for |B_site| = 30, 100, 300 nT and chi = 1e-3, 2e-3, 5e-3 (noise-free, two loops, offsets+gradient)."""
    sx = two_loops(); G, K = design(sx, grad=True)
    rows = []
    rng = np.random.default_rng(seed_for("a09:N"))
    for B in (30.0, 100.0, 300.0):
        for chi in (1e-3, 2e-3, 5e-3):
            r = {"B_site_nT": B, "chi_SI": chi, "induced_over_Mref": chi * B * 1e-9 / MU0 / M_NRM_MIN}
            for dn, u in DIRS.items():
                mt = M_NRM_MIN * u; th = []
                for _ in range(200):
                    Bv = rng.normal(size=3); Bv *= B / np.linalg.norm(Bv)
                    th.append(dir_err_deg(K @ (G @ (mt + chi * Bv * 1e-9 / MU0)), mt))
                r[dn] = A6.stats(th)
            rows.append(r)
            print(f"[N] B={B:.0f} nT chi={chi}: induced {r['induced_over_Mref']:.3f} Mref; dir bias mean " +
                  "  ".join(f"{dn} {r[dn]['mean']:.1f}" for dn in DIRS), flush=True)
    out["N_induced_magnetization"] = rows


def part_M2(out):
    """As M with the recommended survey: loops at 1 and 3 m (5 cm spacing, fixed ASD) and site-field offsets plus
    gradient fitted."""
    l1 = A6.survey("drive_around", spacing=0.05); l3 = A6.survey("drive_around", standoff=3.0, spacing=0.05)
    sx = np.vstack([l1, l3]); sig = SIGMA_B * np.sqrt(0.01 / 0.05)
    G = prism_cell_operator(BND, sx)
    A, _ = A6.augmented(G, A6.nuisance_columns(sx, "world_dc")); A, _ = A6.augmented(A, A6.gradient_basis(sx)[0])
    K = A6.lsK(A)[:3]
    mt = M_NRM_MIN * DIRS["-z"]; d0 = G @ mt
    res = {}
    for depth in (0.0, 1.0, 3.0):
        ztop = BND[4] - depth
        for block, nz in (("cell_1m", 5), ("block_2m", 1), ("block_5m", 1)):
            bs = 1.0 if block == "cell_1m" else float(block.split("_")[1][:-1])
            nxy = int(round(30 / bs))
            mesh = discretize.TensorMesh([np.full(nxy, bs), np.full(nxy, bs), np.full(nz, 5.0 / nz)], origin=[-15.0, -15.0, ztop - 5.0])
            nC = mesh.n_cells
            Gz = _sim_for(mesh, sx, nC).G[:, 2 * nC:3 * nC].copy()
            rows = []
            for floor in (1.0, 0.8, 0.5, 0.2, 0.05, 0.01):
                rng = np.random.default_rng(seed_for(f"a09:M2:{depth}:{block}:{floor}"))
                th = np.zeros(400); sd = []
                for i in range(400):
                    I = 10 ** rng.uniform(np.log10(floor), 0.0, nC) if floor < 1 else np.ones(nC)
                    dI = I - I.mean(); sd.append(dI.std())
                    th[i] = dir_err_deg(K @ (d0 + Gz @ (-M_NRM_MIN * dI) + rng.normal(0, sig, d0.size)), mt)
                rows.append({"floor": floor, "cell_sd_over_Mref": float(np.mean(sd)), "dir_err_deg": A6.stats(th)})
            e = np.array([r["dir_err_deg"]["mean"] for r in rows]); s = np.array([r["cell_sd_over_Mref"] for r in rows])
            idx = np.where((e[:-1] < 5) & (e[1:] >= 5))[0]
            x = float(s[idx[0]] + (5 - e[idx[0]]) * (s[idx[0] + 1] - s[idx[0]]) / (e[idx[0] + 1] - e[idx[0]])) if len(idx) else None
            res[f"depth_{depth:g}m:{block}"] = {"rows": rows, "sd_at_5deg": x,
                                                  "status": "interpolated" if len(idx) else ("not_reached" if e.max() < 5 else "below_lowest")}
            print(f"[M2 depth {depth} {block}] errors " + " ".join(f"{v:.1f}" for v in e) + f" | 5 deg at SD {x}", flush=True)
    out["M2_background_depth_two_loops"] = res


def part_W(out):
    """Three-layer wall with a site field from a distant dipole (30 or 100 nT at 1-10 km), fitted as local offsets
    plus a uniform gradient over both traverses."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("a07", os.path.join(HERE, "a07_three_layer_wall_extent.py"))
    a07 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a07)
    m9 = np.concatenate([a07.MTRUE[:, 0], a07.MTRUE[:, 1], a07.MTRUE[:, 2]])
    res = {}
    for case in ("W50", "W50_ELEV_50", "END"):
        S = np.vstack([a07.traverse(case, z) for z in (0.5, 12.5)])
        G = a07.G_for(case, S)
        Nb = np.column_stack([A6.nuisance_columns(S, "world_dc"), A6.gradient_basis(S)[0]])
        A, kept = A6.augmented(G, Nb); K = np.linalg.solve(A.T @ A, A.T)[:9]
        rows = []
        for B0 in (30.0, 100.0):
            for R in (1000.0, 3000.0, 10000.0):
                rng = np.random.default_rng(seed_for(f"a09:W:{case}:{B0}:{R}"))
                n = 200; errs = np.zeros((n, 3)); resid = []
                for i in range(n):
                    b, _ = dipole_field(S, B0, R, rng)
                    cf, *_ = np.linalg.lstsq(Nb, b, rcond=None); resid.append(float(np.sqrt(np.mean((b - Nb @ cf) ** 2))))
                    m = K @ (G @ m9 + b + rng.normal(0, SIGMA_B, G.shape[0]))
                    for l in range(3):
                        errs[i, l] = dir_err_deg(m[[l, 3 + l, 6 + l]], a07.MTRUE[l])
                r = {"B0_nT": B0, "R_m": R, "residual_after_gradient_rms_nT": float(np.mean(resid)),
                     **{f"layer_{l+1}_dir_err_mean": float(errs[:, l].mean()) for l in range(3)}}
                rows.append(r)
                print(f"[W {case}] {B0:.0f} nT at {R/1000:.0f} km (resid {r['residual_after_gradient_rms_nT']:.3f} nT): " +
                      " | ".join(f"L{l+1} {errs[:, l].mean():.1f}" for l in range(3)), flush=True)
        res[case] = {"n_nuisance_kept": kept, "rows": rows}
    out["W_three_layer_site_field"] = res


def part_Z(out):
    """Decomposition of the full-tolerance budget U (loops 0.5 m off centre): each source alone at its tolerance."""
    base = dict(dx=0.5, R_site=1e6, pose=(0.0, 0.0), bmax=0.0, drift_rms=0.0, tilt_nT=0.0, chi=0.0, chi_err=0.0, n=80)
    one = {"site": dict(R_site=530.0), "pose": dict(pose=(3.0, 0.15)), "burial": dict(bmax=1.0),
           "drift": dict(drift_rms=0.03), "tilt": dict(tilt_nT=0.05), "induced": dict(chi=CHI, chi_err=np.log(2.0)),
           "none": {}}
    for nm, kw in one.items():
        cfg = dict(base); cfg.update(kw)
        budget(out, f"Z_single_{nm}", **cfg)


def part_R(out):
    """Unknown buried extent with the loops 0.5 m off centre: invert with the exposed prism only, or with the
    exposed prism plus a separate root prism (0-1 m or 0-2 m below the surface, its own magnetization fitted).
    True buried extent b ~ U(0, bmax); site-field offsets and gradient fitted; white noise."""
    res = {}
    for dx in (0.0, 0.5):
        sx = two_loops(dx=dx)
        N = np.column_stack([A6.nuisance_columns(sx, "world_dc"), A6.gradient_basis(sx)[0]])
        Ge = prism_cell_operator(BND, sx)
        models = {"exposed_only": Ge}
        for rd in (1.0, 2.0):
            Gr = prism_cell_operator((BND[0], BND[1], BND[2], BND[3], BND[4] - rd, BND[4]), sx)
            models[f"with_root_{rd:g}m"] = np.column_stack([Ge, Gr])
        Ks = {k: np.linalg.solve(np.column_stack([G, N]).T @ np.column_stack([G, N]), np.column_stack([G, N]).T)[:3] for k, G in models.items()}
        for bmax in (1.0, 2.0):
            for dn, u in DIRS.items():
                mt = M_NRM_MIN * u
                rng = np.random.default_rng(seed_for(f"a09:R:{dx}:{bmax}:{dn}"))
                n = 150
                rec = {k: np.zeros((n, 3)) for k in Ks}; nf = {k: np.zeros((n, 3)) for k in Ks}
                for i in range(n):
                    b = rng.uniform(0, bmax)
                    d = prism_cell_operator((BND[0], BND[1], BND[2], BND[3], BND[4] - b, BND[5]), sx) @ mt
                    e = rng.normal(0, SIGMA_B, d.size)
                    for k, K in Ks.items():
                        rec[k][i] = K @ (d + e); nf[k][i] = K @ d
                key = f"dx{dx:g}_bmax{bmax:g}_{dn}"
                res[key] = {k: {"all": A6.metrics(rec[k], mt), "noise_free": A6.metrics(nf[k], mt)} for k in Ks}
                print(f"[R dx={dx} b<={bmax} {dn}] " + " | ".join(
                    f"{k}: {res[key][k]['all']['dir_err_deg']['mean']:.1f} (bias {res[key][k]['noise_free']['dir_err_deg']['mean']:.1f}, "
                    f"dM {res[key][k]['all']['dM_signed_pct']['mean']:+.0f}%)" for k in Ks), flush=True)
    out["R_burial_root_fit"] = res


def part_S(out):
    """Noise-free bias from burial (b ~ U(0, 1 m)) and from pose error (+-3 deg, +-0.15 m) as a function of how far the
    two loops are off the outcrop centre (dx = 0-0.5 m). Site-field offsets and gradient fitted."""
    res = {}
    for dx in (0.0, 0.1, 0.25, 0.5):
        sx = two_loops(dx=dx)
        N = np.column_stack([A6.nuisance_columns(sx, "world_dc"), A6.gradient_basis(sx)[0]])
        Ge = prism_cell_operator(BND, sx)
        Ke = np.linalg.solve(np.column_stack([Ge, N]).T @ np.column_stack([Ge, N]), np.column_stack([Ge, N]).T)[:3]
        row = {}
        for dn, u in DIRS.items():
            mt = M_NRM_MIN * u
            rng = np.random.default_rng(seed_for(f"a09:S:{dx}:{dn}"))
            np.random.seed(seed_for(f"a09:S:{dx}:{dn}:geom") % (2 ** 32))
            bur, pos = [], []
            for i in range(60):
                b = rng.uniform(0, 1.0)
                d = prism_cell_operator((BND[0], BND[1], BND[2], BND[3], BND[4] - b, BND[5]), sx) @ mt
                bur.append(dir_err_deg(Ke @ d, mt))
                pdg, trans = generate_pose_errors(randomize=True, pointing_err_max_deg=(3.0, 3.0), range_err_max_m=0.15)
                Gm = warped_operator(sx, ISO.imager_R(pdg), np.asarray(trans, float))
                A = np.column_stack([Gm, N]); K = np.linalg.solve(A.T @ A, A.T)[:3]
                pos.append(dir_err_deg(K @ (Ge @ mt), mt))
            row[dn] = {"burial_bias_deg": A6.stats(bur), "pose_bias_deg": A6.stats(pos)}
        res[f"dx{dx:g}"] = row
        print(f"[S dx={dx}] burial bias " + " ".join(f"{d} {row[d]['burial_bias_deg']['mean']:.1f}" for d in DIRS) +
              " | pose bias " + " ".join(f"{d} {row[d]['pose_bias_deg']['mean']:.1f}" for d in DIRS), flush=True)
    out["S_centring"] = res


def part_D(out):
    """Slow drift alone with the recommended two-loop survey, loops centred or 0.5 m off centre, at 0.015 and 0.02 nT
    rms about the survey mean (complements Z_single_drift at 0.03 nT)."""
    base = dict(R_site=1e6, pose=(0.0, 0.0), bmax=0.0, tilt_nT=0.0, chi=0.0, chi_err=0.0, n=80)
    for dx in (0.0, 0.5):
        for rms in (0.015, 0.02, 0.03):
            budget(out, f"D_drift_{rms:g}nT_dx{dx:g}", dx=dx, drift_rms=rms, **base)


def part_F2(out):
    """Budgets with the buried part fitted as a 1 m root body, loops 0.5 m off centre: full, half and quarter tolerance."""
    budget(out, "F2_full_offcentre_root", 0.5, 530.0, (3.0, 0.15), 1.0, 0.03, 0.05, CHI, np.log(2.0), root=1.0)
    budget(out, "F2_half_offcentre_root", 0.5, 750.0, (1.5, 0.075), 0.5, 0.015, 0.025, CHI, np.log(1.4), root=1.0)
    budget(out, "F2_quarter_offcentre_root", 0.5, 1060.0, (0.75, 0.0375), 0.25, 0.0075, 0.0125, CHI, np.log(1.2), root=1.0)


def part_H2(out):
    """Slow drift with a linear drift term per axis fitted over both loops (centred and 0.5 m off centre)."""
    base = dict(R_site=1e6, pose=(0.0, 0.0), bmax=0.0, tilt_nT=0.0, chi=0.0, chi_err=0.0, n=80, fit_drift=True)
    for dx in (0.0, 0.5):
        for rms in (0.015, 0.03):
            budget(out, f"H2_drift_{rms:g}nT_dx{dx:g}_linfit", dx=dx, drift_rms=rms, **base)


DIRS6 = dict(DIRS, **{"+y": np.array([0, 1.0, 0]), "inc45y": np.array([0, 0.7071068, -0.7071068]),
                      "up45": np.array([0.7071068, 0, 0.7071068])})


def part_REQ(out):
    """The requirement set of the Conclusions run together, six magnetization directions:
    loops 0.1 m off centre with the buried extent unknown up to 0.5 m (a) or 1 m (b); loops 0.5 m off centre with the
    buried extent known (c). Site residual ~0.02 nT (100 nT from 530 m), pose +-1.5 deg / +-0.075 m, drift 0.01 nT rms,
    tilt*|B| 0.05 nT, chi known to a factor of 1.4."""
    budget(out, "REQ_a_dx0.1_b0.5", 0.1, 530.0, (1.5, 0.075), 0.5, 0.01, 0.05, CHI, np.log(1.4), n=120, dirs=DIRS6)
    budget(out, "REQ_b_dx0.1_b1", 0.1, 530.0, (1.5, 0.075), 1.0, 0.01, 0.05, CHI, np.log(1.4), n=120, dirs=DIRS6)
    budget(out, "REQ_c_dx0.5_b0", 0.5, 530.0, (1.5, 0.075), 0.0, 0.01, 0.05, CHI, np.log(1.4), n=120, dirs=DIRS6)


def part_TF(out):
    """Tilt alone (0.03 and 0.1 deg in 300 nT) with the six heading-dependent terms fitted, loops centred and 0.5 m off
    centre (the asymmetric case where an unfitted tilt matters)."""
    base = dict(R_site=1e6, pose=(0.0, 0.0), bmax=0.0, drift_rms=0.0, chi=0.0, chi_err=0.0, n=80, B0=300.0)
    for dx in (0.0, 0.5):
        for tdeg in (0.03, 0.1):
            tnT = 300.0 * np.radians(tdeg)
            budget(out, f"TF_tilt{tdeg:g}_dx{dx:g}_fit", dx=dx, tilt_nT=tnT, fit_tilt=True, **base)
            budget(out, f"TF_tilt{tdeg:g}_dx{dx:g}_nofit", dx=dx, tilt_nT=tnT, fit_tilt=False, **base)


def part_D01(out):
    """Drift of 0.01 nT alone, loops centred and 0.5 m off centre, six directions."""
    base = dict(R_site=1e6, pose=(0.0, 0.0), bmax=0.0, tilt_nT=0.0, chi=0.0, chi_err=0.0, n=80, dirs=DIRS6)
    for dx in (0.0, 0.5):
        budget(out, f"D_drift_0.01nT_dx{dx:g}", dx=dx, drift_rms=0.01, **base)


def part_RV(out):
    """Drift with each loop driven twice in opposite directions (doubles the survey time), loops centred and 0.5 m off
    centre; drift rms about the survey mean 0.01 and 0.03 nT. Also white noise alone with the reversed design."""
    base = dict(R_site=1e6, pose=(0.0, 0.0), bmax=0.0, tilt_nT=0.0, chi=0.0, chi_err=0.0, n=80, reverse=True)
    for dx in (0.0, 0.5):
        for rms in (0.0, 0.01, 0.03):
            budget(out, f"RV_drift_{rms:g}nT_dx{dx:g}", dx=dx, drift_rms=rms, **base)


def _req_full(out, key, dx, bmax):
    budget(out, key, dx, 530.0, (1.5, 0.075), bmax, 0.01, 0.05, CHI, np.log(1.4), n=120, dirs=DIRS6,
           fit_tilt=True, reverse=True)


def part_REQF_a(out):
    """Requirement set as part q, but with the two mitigations the text recommends: each loop driven twice in opposite
    directions (doubles the survey time) and the six heading-dependent tilt terms fitted. Case a: loops 0.1 m off
    centre, buried extent up to 0.5 m."""
    _req_full(out, "REQF_a_dx0.1_b0.5", 0.1, 0.5)


def part_REQF_b(out):
    """As part f, case b: loops 0.1 m off centre, buried extent up to 1 m."""
    _req_full(out, "REQF_b_dx0.1_b1", 0.1, 1.0)


def part_REQF_c(out):
    """As part f, case c: loops 0.5 m off centre, buried extent known."""
    _req_full(out, "REQF_c_dx0.5_b0", 0.5, 0.0)


def part_REQF_chi5_a(out):
    """Mitigated set (part f) with a higher susceptibility, chi = 5e-3 SI, still known to a factor of 1.4:
    loops 0.1 m off centre, buried extent up to 0.5 m."""
    budget(out, "REQF_chi5_a_dx0.1_b0.5", 0.1, 530.0, (1.5, 0.075), 0.5, 0.01, 0.05, 5e-3, np.log(1.4), n=120,
           dirs=DIRS6, fit_tilt=True, reverse=True)


def part_REQF_chi5_c(out):
    """As part i, loops 0.5 m off centre, buried extent known."""
    budget(out, "REQF_chi5_c_dx0.5_b0", 0.5, 530.0, (1.5, 0.075), 0.0, 0.01, 0.05, 5e-3, np.log(1.4), n=120,
           dirs=DIRS6, fit_tilt=True, reverse=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--parts", default="JTtKM2NWUVYXGZqSRDdErfghij"); a = ap.parse_args()
    t0 = time.time()
    # rerunning some parts only (--parts): keep the stored results of the others
    out = json.load(open(OUT)) if a.parts != "JTtKM2NWUVYXGZqSRDdErfghij" and os.path.exists(OUT) else {}
    out["meta"] = dict(script="analysis/a09_systematics.py", sigma_B_nT=SIGMA_B, geometry="as a02/a06",
                       survey="drive-around at 1 m + second loop at 3 m; M uses the single loop at 5 cm with the noise scaled to the same ASD")
    for p in a.parts:
        {"J": part_J, "K": part_K, "M": part_M, "T": part_T, "U": part_U, "V": part_V, "X": part_X, "Y": part_Y, "N": part_N, "2": part_M2, "W": part_W, "Z": part_Z, "R": part_R, "S": part_S, "D": part_D, "G": part_F2, "E": part_H2, "q": part_REQ, "t": part_TF, "d": part_D01, "r": part_RV, "f": part_REQF_a, "g": part_REQF_b, "h": part_REQF_c, "i": part_REQF_chi5_a, "j": part_REQF_chi5_c}[p](out)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(out, open(OUT, "w"), indent=1)
        print(f"part {p} done ({time.time()-t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

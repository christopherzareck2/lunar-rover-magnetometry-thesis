"""a07 - Three-layer wall: does the recovery depend on the wall ending at x = +-50 m? (paper Section 4.4)

The paper's wall is 100 m long (x = -50..50) and the traverse runs x = -120..120 m, so the traverse passes
both ends of every flow. A flow magnetized along strike (x) produces no external field if it is infinitely
long (a 2-D body magnetized along its axis has no external field), so the ends carry the information on
M_x. Here we change the wall geometry and keep everything else as in a03 (flows 5 m thick, 50 m deep into
the wall, z = 0-5/5-10/10-15 m; M = 2.4e-3 A/m along -z/+y/+z; traverse y = -1 m, z = 0.5 m and, for the
pair, z = 12.5 m; 1 cm sampling; sigma_B = 0.50 nT; closed-form weighted least squares, N = 500 draws
with the same noise per case).

Each flow is built from exact analytic prisms (engine prism_cell_operator), so a flow can be a union of
boxes (the 90-degree bend case).

Cases
  W50       x = -50..50 (paper geometry), traverse x = -120..120
  W100/W250/W1000   x = +-100, +-250, +-1000 m, same traverse (traverse never reaches an end for W250+)
  END       x = -1000..50: one end inside the traverse (flow truncated by a crater, fault or rille bend)
  END_deep / BEND_deep  as END / BEND with the flows extending 1000 m back into the wall instead of 50 m
  W1000_XPROF  W1000 plus a 100 m profile driven straight away from the wall at x = 0 (cross-strike)
  W50_ELEV_<h>  paper wall, ground traverse +-120 m, elevated traverse only over |x| <= h (on the wall face)
  Every case is also inverted with three local-frame (site-field) offsets fitted ("site_offsets_fitted").
  BEND      x = -1000..0 along the rille, then a 90-degree bend; the second limb runs along -y
            (rock at x = 0..50, y = -1000..50). Traverse: 120 m along each limb (1 m standoff).
For the straight long walls we also invert for the cross-strike components only (M_y, M_z per flow,
M_x fixed at zero, which is its true value here), which is all a 2-D section can constrain.

Run:  python analysis/a07_three_layer_wall_extent.py
Out:  results/analysis/a07_three_layer_wall_extent.json
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "engine")); os.chdir(os.path.join(ROOT, "engine"))
from inversion_helpers import prism_cell_operator, dir_err_deg, seed_for, M_NRM_MIN, SIGMA_B

N = 500
SP = 0.01
ZL = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]
MTRUE = M_NRM_MIN * np.array([[0, 0, -1.0], [0, 1.0, 0], [0, 0, 1.0]])


def line(p0, p1, z):
    n = int(round(np.linalg.norm(np.subtract(p1, p0)) / SP)) + 1
    t = np.linspace(0, 1, n)[:, None]
    xy = (1 - t) * np.asarray(p0, float) + t * np.asarray(p1, float)
    return np.c_[xy, np.full(n, z)]


def traverse(case, z):
    if case.startswith("W50_ELEV") and z > 1.0:
        half = float(case.split("_")[-1])                 # elevated traverse limited to |x| <= half (on the wall)
        return line((-half, -1.0), (half, -1.0), z)
    if case == "W1000_XPROF":
        a = line((-120.0, -1.0), (120.0, -1.0), z)
        b = line((0.0, -1.0 - SP), (0.0, -100.0), z)
        return np.vstack([a, b])
    if case.startswith("BEND"):
        a = line((-120.0, -1.0), (-1.0, -1.0), z)
        b = line((-1.0, -1.0 - SP), (-1.0, -120.0), z)
        return np.vstack([a, b])
    return line((-120.0, -1.0), (120.0, -1.0), z)


def boxes(case):
    D = 1000.0 if case.endswith("_deep") else 50.0
    if case == "W1000_XPROF":
        return [(-1000.0, 1000.0, 0.0, 50.0)]
    if case.startswith("W50_ELEV"):
        return [(-50.0, 50.0, 0.0, 50.0)]
    if case == "END_deep":
        return [(-1000.0, 50.0, 0.0, D)]
    if case == "BEND_deep":
        return [(-1000.0, 0.0, 0.0, D), (0.0, D, -1000.0, D)]
    if case.startswith("W"):
        h = float(case[1:]); return [(-h, h, 0.0, 50.0)]
    if case == "END":
        return [(-1000.0, 50.0, 0.0, 50.0)]
    if case == "BEND":
        return [(-1000.0, 0.0, 0.0, 50.0), (0.0, 50.0, -1000.0, 50.0)]


def G_for(case, sxyz):
    cols = [[], [], []]                                  # per component, per layer
    Gs = []
    for (z0, z1) in ZL:
        g = sum(prism_cell_operator((x0, x1, y0, y1, z0, z1), sxyz) for (x0, x1, y0, y1) in boxes(case))
        Gs.append(g)                                     # (nD, 3) for this layer
    # parameter order: [Mx L1..L3, My L1..L3, Mz L1..L3] as in the engine
    return np.column_stack([Gs[l][:, c] for c in range(3) for l in range(3)])


def per_layer(m9):
    return np.stack([m9[[i, 3 + i, 6 + i]] for i in range(3)])


def run(G, m9, noise, keep=None):
    Gi = G if keep is None else G[:, keep]
    GtG = Gi.T @ Gi
    cond = float(np.linalg.cond(GtG))
    K = np.linalg.solve(GtG, Gi.T)
    R = (G @ m9)[None] + noise
    Mrec_i = R @ K.T
    Mrec = np.zeros((len(R), 9))
    Mrec[:, keep if keep is not None else slice(None)] = Mrec_i
    out = {"cond_GtG": cond}
    for i in range(3):
        L = np.array([per_layer(r)[i] for r in Mrec])
        th = np.array([dir_err_deg(v, MTRUE[i]) for v in L])
        dm = 100 * (np.linalg.norm(L, axis=1) - M_NRM_MIN) / M_NRM_MIN
        sig = np.sqrt(np.diag(np.linalg.inv(GtG)))
        out[f"layer_{i+1}"] = {"dir_err_mean": float(th.mean()), "dir_err_median": float(np.median(th)),
                               "dM_abs_median": float(np.median(np.abs(dm)))}
    return out


def run_offsets(G, m9, noise):
    """As run(), with three local-frame (site-field) offsets fitted as nuisance parameters."""
    nD = G.shape[0]
    Off = np.zeros((nD, 3))
    for j in range(3):
        Off[j::3, j] = 1.0
    A = np.column_stack([G, Off])
    K = np.linalg.solve(A.T @ A, A.T)[:9]
    Mrec = ((G @ m9)[None] + noise) @ K.T
    out = {}
    for i in range(3):
        L = np.array([per_layer(r)[i] for r in Mrec])
        th = np.array([dir_err_deg(v, MTRUE[i]) for v in L])
        dm = 100 * (np.linalg.norm(L, axis=1) - M_NRM_MIN) / M_NRM_MIN
        out[f"layer_{i+1}"] = {"dir_err_mean": float(th.mean()), "dir_err_median": float(np.median(th)),
                               "dM_abs_median": float(np.median(np.abs(dm)))}
    return out


def main():
    t0 = time.time()
    m9 = np.concatenate([MTRUE[:, 0], MTRUE[:, 1], MTRUE[:, 2]])
    out = {"meta": dict(script="analysis/a07_three_layer_wall_extent.py", N=N, sigma_B_nT=SIGMA_B, spacing_m=SP,
                        M_true=MTRUE.tolist(), solver="closed-form weighted LS"), "cases": {}}
    import sys as _s
    cases = _s.argv[1:] or ["W50", "W100", "W250", "W1000", "END", "BEND", "END_deep", "BEND_deep", "W1000_XPROF",
                            "W50_ELEV_50", "W50_ELEV_25"]
    fn = os.path.join(ROOT, "results", "analysis", "a07_three_layer_wall_extent.json")
    if _s.argv[1:] and os.path.exists(fn):      # rerunning some cases only: keep the others
        out["cases"] = json.load(open(fn)).get("cases", {})
    for case in cases:
        res = {}
        for label, zs in (("ground", [0.5]), ("ground+12.5m", [0.5, 12.5])):
            S = np.vstack([traverse(case, z) for z in zs])
            G = G_for(case, S)
            rng = np.random.default_rng(seed_for(f"a07:{case}:{label}"))
            noise = rng.normal(0.0, SIGMA_B, size=(N, G.shape[0]))
            r = run(G, m9, noise)
            r["site_offsets_fitted"] = run_offsets(G, m9, noise)
            sig = [G[:, [l, 3 + l, 6 + l]] @ MTRUE[l] for l in range(3)]
            r["signal_rms_nT_all"] = [float(np.sqrt(np.mean(s ** 2))) for s in sig]
            mid = np.repeat((np.abs(S[:, 0]) <= 50) & (np.abs(S[:, 1] + 1) < 1e-6), 3) if not case.startswith("BEND") else np.ones(3 * len(S), bool)
            r["signal_rms_nT_absx_le_50"] = [float(np.sqrt(np.mean(s[mid] ** 2))) for s in sig]
            r["n_stations"] = int(len(S))
            if case.startswith("W") and case != "W50" or case == "W1000_XPROF":
                keep = [3, 4, 5, 6, 7, 8]
                r["cross_strike_only"] = run(G, m9, noise, keep=keep)
            res[label] = r
            msg = " | ".join(f"L{i+1} {r[f'layer_{i+1}']['dir_err_mean']:6.1f}" for i in range(3))
            if "cross_strike_only" in r:
                msg += "  || yz-only: " + " | ".join(f"L{i+1} {r['cross_strike_only'][f'layer_{i+1}']['dir_err_mean']:5.1f}" for i in range(3))
            msg += "  || offsets fitted: " + " | ".join(f"L{i+1} {r['site_offsets_fitted'][f'layer_{i+1}']['dir_err_mean']:5.1f}" for i in range(3))
            print(f"[{case:5s} {label:13s}] cond {r['cond_GtG']:.1e}  dir err (deg) {msg}  sig {['%.3f' % v for v in r['signal_rms_nT_absx_le_50']]}  ({time.time()-t0:.0f}s)", flush=True)
        out["cases"][case] = res
        json.dump(out, open(os.path.join(ROOT, "results", "analysis", "a07_three_layer_wall_extent.json"), "w"), indent=1)
    out["meta"]["wall_s"] = time.time() - t0
    json.dump(out, open(os.path.join(ROOT, "results", "analysis", "a07_three_layer_wall_extent.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

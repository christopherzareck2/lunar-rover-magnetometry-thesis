"""a08 - Skate park: neighbors magnetized in other directions, and a joint inversion (paper Section 4.3).

Same scene, mesh (0.25 m), survey (square loop, 1 m standoff, 0.5 m height, 1 cm) and noise
(0.50 nT) as the skate-park run of Figure 6 (engine/skatepark_1cm_dr_sweep.py, imported).
The data are linear in each body's uniform magnetization, so we build a 3-column operator for every
body (central + 4 neighbours) and form the data for any neighbour magnetization.

Neighbour configurations (central always -z, |M| equal for all bodies):
  A   all neighbours -z (configuration A)
  B   N/S -z, E/W +z (configuration B)
  HX  all neighbours +x (horizontal, along one survey side)
  HY  all neighbours +y
  RND each neighbour in an independent random direction, redrawn for every noise realization
Inversions:
  central  central body only (3 parameters), as in the paper
  joint    all five imaged bodies (15 parameters), shapes known
N = 250 per case at Delta_r = 1, 2, 5 m.

Run:  python analysis/a08_skatepark_neighbor_directions.py
Out:  results/analysis/a08_skatepark_neighbor_directions.json
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "engine")); os.chdir(os.path.join(ROOT, "engine"))
import skatepark_1cm_dr_sweep as SP
from inversion_helpers import seed_for

N = 250
DR = [1, 2, 5]


def body_operators(delx, survey):
    mesh = SP.build_mesh(delx)
    cc = mesh.cell_centers
    x, y, z = cc[:, 0], cc[:, 1], cc[:, 2]
    configs, _, _ = SP.outcrop_configs_for(delx, "A")
    ops, labels = [], []
    for cfg in configs:
        ocx, ocy, ocz = cfg["center"]
        mask = SP.create_shape_mask(x, y, z, ocx, ocy, ocz, cfg["shape"], cfg["params"],
                                    margin_frac=SP.MASK_MARGIN_FRAC if cfg["is_primary"] else 0.0,
                                    margin_z=SP.MASK_MARGIN_Z if cfg["is_primary"] else 0.0)
        mask &= (z >= 0.0)
        nC = int(mask.sum())
        sim = SP.sim_over(mesh, survey, mask)
        cols = []
        for k in range(3):
            e = np.zeros(3 * nC); e[k * nC:(k + 1) * nC] = 1.0
            cols.append(np.asarray(sim.dpred(e), float))
        ops.append(np.column_stack(cols)); labels.append(cfg["label"])
    return ops, labels


def rand_unit(rng, n):
    v = rng.normal(size=(n, 3)); return v / np.linalg.norm(v, axis=1, keepdims=True)


def main():
    t0 = time.time()
    sxyz = SP.build_survey_xyz(SP.SPACING)
    survey = SP.make_survey(sxyz)
    m_c = SP.mag_vector_true.copy()
    A = np.linalg.norm(m_c)
    down, up = np.array([0, 0, -A]), np.array([0, 0, A])
    fixed = {"A": [down] * 4, "B": [down, up, down, up],          # order N, E, S, W
             "HX": [np.array([A, 0, 0])] * 4, "HY": [np.array([0, A, 0])] * 4}
    out = {"meta": dict(script="analysis/a08_skatepark_neighbor_directions.py", N=N, delta_r_m=DR,
                        sigma_nT=SP.noise_floor, n_stations=int(len(sxyz)),
                        neighbour_order=["north cylinder", "east hemisphere", "south pyramid", "west sphere"]),
           "results": {}}
    for delx in DR:
        ops, labels = body_operators(delx, survey)
        Gc = ops[0]
        Gj = np.column_stack(ops)
        Kc = np.linalg.solve(Gc.T @ Gc, Gc.T)
        Kj = np.linalg.solve(Gj.T @ Gj, Gj.T)
        res = {"labels": labels, "cond_joint": float(np.linalg.cond(Gj.T @ Gj))}
        for case in ["A", "B", "HX", "HY", "RND"]:
            rng = np.random.default_rng(seed_for(f"a08:{delx}:{case}"))
            noise = rng.normal(0.0, SP.noise_floor, size=(N, Gc.shape[0]))
            if case == "RND":
                nb = rand_unit(rng, 4 * N).reshape(N, 4, 3) * A
            else:
                nb = np.broadcast_to(np.array(fixed[case]), (N, 4, 3))
            D = (Gc @ m_c)[None] + sum(nb[:, k, :] @ ops[k + 1].T for k in range(4)) + noise
            r = {}
            for inv, K in (("central", Kc), ("joint", Kj)):
                M = D @ K[:3].T if inv == "joint" else D @ K.T
                th = np.array([SP.dir_err_deg(v, m_c) for v in M])
                dm = 100 * (np.linalg.norm(M, axis=1) - A) / A
                r[inv] = {"dir_err_deg": SP.summarize(th), "dM_signed_pct": SP.summarize(dm),
                          "dM_abs_pct": SP.summarize(np.abs(dm))}
            res[case] = r
            print(f"[dr={delx} {case:3s}] central: dir med {r['central']['dir_err_deg']['median']:5.2f} "
                  f"p95 {r['central']['dir_err_deg']['p95']:5.1f}  dM med {r['central']['dM_signed_pct']['median']:+6.1f}% | "
                  f"joint: dir med {r['joint']['dir_err_deg']['median']:5.2f}  dM med {r['joint']['dM_signed_pct']['median']:+6.1f}% "
                  f"|dM| med {r['joint']['dM_abs_pct']['median']:5.1f}%  ({time.time()-t0:.0f}s)", flush=True)
        out["results"][str(delx)] = res
        json.dump(out, open(os.path.join(ROOT, "results", "analysis", "a08_skatepark_neighbor_directions.json"), "w"), indent=1)
    out["meta"]["wall_s"] = time.time() - t0
    json.dump(out, open(os.path.join(ROOT, "results", "analysis", "a08_skatepark_neighbor_directions.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

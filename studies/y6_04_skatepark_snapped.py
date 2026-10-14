"""Y6-04: skate-park central prism off its stated bounds. Rerun with the mesh origin snapped to the 0.25 m lattice.

Released code (engine/skatepark_1cm_dr_sweep.py::build_mesh) rounds the number of 0.25 m cells up but sets the mesh
origin from the unrounded scene width, so the cell faces sit at -region/2 + k*0.25 and the voxelized central body is
x, y in [-2.078, +1.922] m instead of [-2, +2] m at every Delta_r (review finding Y6-04, see docs/known-issues.md).

What this script changes (a modified copy of the sweep loop of engine/skatepark_1cm_dr_sweep.py::main and of
analysis/a08_skatepark_neighbor_directions.py::main; nothing in engine/ or analysis/ is edited):
  * ``snapped_build_mesh`` replaces SP.build_mesh: same cell size, padding and depth, but the horizontal extent is
    rounded up to a whole number of cells on EACH side of the origin, so the mesh is symmetric about x = y = 0 and
    every cell face lies on a multiple of 0.25 m. The central body then occupies exactly [-2, 2] x [-2, 2] x [0, 1] m
    (asserted below) and the neighbours are voxelized on the same lattice phase as the centre.
  * Everything else (scene, survey, noise, seeds, N = 250, closed-form solve) is unchanged. The noise seeds are the
    released ones (MASTER_SEED + 100*Delta_r + case for the sweep, seed_for("a08:<dr>:<case>") for a08), so the
    old -> new differences isolate the geometry fix; the released-mesh rows are recomputed first and must reproduce
    the stored JSON exactly. A replicate with fresh seed labels (study:y6-04:...) measures the Monte Carlo scatter.

Run:  python studies/y6_04_skatepark_snapped.py      (about 1 min)
Out:  studies/results/y6_04_skatepark_snapped.json
"""
from __future__ import annotations

import os
import time

import numpy as np

import _common as C                       # sets sys.path to engine/ and analysis/
_cwd = os.getcwd()
import skatepark_1cm_dr_sweep as SP       # noqa: E402
import a08_skatepark_neighbor_directions as A8   # noqa: E402  (chdirs into engine/)
os.chdir(_cwd)
from discretize import TensorMesh          # noqa: E402

RELEASED_BUILD_MESH = SP.build_mesh


def snapped_build_mesh(delx):
    """As SP.build_mesh, with the horizontal mesh symmetric about the origin and faces on the 0.25 m lattice."""
    dh = SP.dh_fine
    half_xy, z_top = SP.scene_bounds(delx)
    need = 2.0 * half_xy + 2.0 * SP.SP_MESH_PAD_XY_M
    region = max(16.0, need)
    n_half = int(np.ceil(region / 2.0 / dh))                 # cells on each side of x = 0 (and y = 0)
    max_depth, max_height = 4.0, max(2.0, z_top + SP.SP_MESH_PAD_Z_M)
    hx = [(dh, 2 * n_half)]; hy = [(dh, 2 * n_half)]
    hz = [(dh, int(np.ceil((max_height + max_depth) / dh)))]
    return TensorMesh([hx, hy, hz], x0=(-n_half * dh, -n_half * dh, -max_depth))


def central_geometry(delx):
    mesh = SP.build_mesh(delx)
    cc = mesh.cell_centers
    configs, _, _ = SP.outcrop_configs_for(delx, "A")
    out = {}
    for cfg in configs:
        ocx, ocy, ocz = cfg["center"]
        m = SP.create_shape_mask(cc[:, 0], cc[:, 1], cc[:, 2], ocx, ocy, ocz, cfg["shape"], cfg["params"],
                                 margin_frac=SP.MASK_MARGIN_FRAC if cfg["is_primary"] else 0.0,
                                 margin_z=SP.MASK_MARGIN_Z if cfg["is_primary"] else 0.0) & (cc[:, 2] >= 0.0)
        vol = float(m.sum() * SP.dh_fine ** 3)
        h = SP.dh_fine / 2
        bb = [float(cc[m, 0].min() - h), float(cc[m, 0].max() + h), float(cc[m, 1].min() - h),
              float(cc[m, 1].max() + h), float(cc[m, 2].min() - h), float(cc[m, 2].max() + h)]
        out[cfg["label"]] = {"volume_m3": vol, "bbox_m": bb, "centroid_m": [float(v) for v in cc[m].mean(0)]}
    return out


def sweep(seed_mode: str):
    """The released sweep loop (SP.main, without the one-off iterative check and file writes)."""
    sxyz = SP.build_survey_xyz(SP.SPACING); survey = SP.make_survey(sxyz); nD = int(survey.nD)
    m_true = SP.mag_vector_true.copy()
    res = {"A": {}, "B": {}}
    for delx in SP.DELTA_R:
        ops = SP.build_scene_operators(delx, survey)
        G = ops["G_cen"]; K = np.linalg.pinv(G)
        for case in ("A", "B"):
            seed = (SP.MASTER_SEED + 100 * delx + (0 if case == "A" else 1)) if seed_mode == "released" \
                else C.seed_for(f"study:y6-04:sweep:{delx}:{case}")
            rng = np.random.default_rng(seed)
            D = ops["d_clean"][case][None, :] + rng.normal(0.0, SP.noise_floor, size=(SP.N_MC, nD))
            M = D @ K.T
            rr = np.sqrt(np.mean(((M @ G.T - D) / SP.noise_floor) ** 2, axis=1))
            de = np.array([SP.dir_err_deg(v, m_true) for v in M])
            amp = np.linalg.norm(M, axis=1) / (SP.mu0 * 1e9)
            sgn = 100.0 * (amp - SP.strengthAm) / SP.strengthAm
            res[case][str(delx)] = {
                "seed": int(seed), "N": SP.N_MC, "n_cells_central": ops["nC_central"],
                "n_cells_union": ops["n_cells_union"][case],
                "dir_median_deg": float(np.median(de)), "dir_median_se": C.median_se(de, seed_label=f"study:y6-04:bs:{delx}:{case}"),
                "dir_mean_deg": float(de.mean()), "dir_mean_se": float(de.std(ddof=1) / np.sqrt(de.size)),
                "dM_signed_mean_pct": float(sgn.mean()), "dM_signed_mean_se": float(sgn.std(ddof=1) / np.sqrt(sgn.size)),
                "dM_signed_median_pct": float(np.median(sgn)),
                "dM_abs_median_pct": float(np.median(np.abs(sgn))),
                "dM_abs_median_se": C.median_se(np.abs(sgn), seed_label=f"study:y6-04:bsm:{delx}:{case}"),
                "residual_rms_median": float(np.median(rr)),
            }
    return res


def a08_runs(seed_mode: str):
    """The released a08 loop (central-only and joint K = 5 inversions, five neighbour configurations)."""
    sxyz = SP.build_survey_xyz(SP.SPACING); survey = SP.make_survey(sxyz)
    m_c = SP.mag_vector_true.copy(); Amp = np.linalg.norm(m_c)
    down, up = np.array([0, 0, -Amp]), np.array([0, 0, Amp])
    fixed = {"A": [down] * 4, "B": [down, up, down, up], "HX": [np.array([Amp, 0, 0])] * 4,
             "HY": [np.array([0, Amp, 0])] * 4}
    out = {}
    for delx in A8.DR:
        ops, labels = A8.body_operators(delx, survey)
        Gc = ops[0]; Gj = np.column_stack(ops)
        Kc = np.linalg.solve(Gc.T @ Gc, Gc.T); Kj = np.linalg.solve(Gj.T @ Gj, Gj.T)
        res = {}
        for case in ["A", "B", "HX", "HY", "RND"]:
            label = f"a08:{delx}:{case}" if seed_mode == "released" else f"study:y6-04:a08:{delx}:{case}"
            rng = np.random.default_rng(C.seed_for(label))
            noise = rng.normal(0.0, SP.noise_floor, size=(A8.N, Gc.shape[0]))
            nb = A8.rand_unit(rng, 4 * A8.N).reshape(A8.N, 4, 3) * Amp if case == "RND" else \
                np.broadcast_to(np.array(fixed[case]), (A8.N, 4, 3))
            D = (Gc @ m_c)[None] + sum(nb[:, k, :] @ ops[k + 1].T for k in range(4)) + noise
            r = {"seed_label": label}
            for inv, K in (("central", Kc), ("joint", Kj)):
                M = D @ K[:3].T if inv == "joint" else D @ K.T
                th = np.array([SP.dir_err_deg(v, m_c) for v in M])
                dm = 100 * (np.linalg.norm(M, axis=1) - Amp) / Amp
                r[inv] = {"dir_median_deg": float(np.median(th)),
                          "dir_median_se": C.median_se(th, seed_label=f"study:y6-04:bs8:{delx}:{case}:{inv}"),
                          "dir_mean_deg": float(th.mean()), "dir_mean_se": float(th.std(ddof=1) / np.sqrt(th.size)),
                          "dir_p95_deg": float(np.percentile(th, 95)),
                          "dM_signed_median_pct": float(np.median(dm)), "dM_abs_median_pct": float(np.median(np.abs(dm)))}
            res[case] = r
        out[str(delx)] = res
    return out


def main():
    t0 = time.time()
    stored_sw = C.stored("results/engine/skatepark_1cm_dr_sweep_results.json")["sweep"]
    stored_a8 = C.stored("results/analysis/a08_skatepark_neighbor_directions.json")["results"]

    SP.build_mesh = RELEASED_BUILD_MESH
    geom_old = {str(d): central_geometry(d) for d in SP.DELTA_R}
    old = sweep("released"); old8 = a08_runs("released")
    # reproduction check against the stored released results
    rep = []
    for case in ("A", "B"):
        for d in map(str, SP.DELTA_R):
            rep.append(abs(old[case][d]["dir_median_deg"] - stored_sw[case][d]["dir_err_deg"]["median"]))
            rep.append(abs(old[case][d]["dM_signed_mean_pct"] - stored_sw[case][d]["mag_err_pct_signed"]["mean"]))
    for d in old8:
        for case in old8[d]:
            for inv in ("central", "joint"):
                rep.append(abs(old8[d][case][inv]["dir_median_deg"] - stored_a8[d][case][inv]["dir_err_deg"]["median"]))

    SP.build_mesh = snapped_build_mesh
    geom_new = {str(d): central_geometry(d) for d in SP.DELTA_R}
    for d, g in geom_new.items():
        c = g["Primary (rect)"]
        assert np.allclose(c["bbox_m"], [-2, 2, -2, 2, 0, 1]) and abs(c["volume_m3"] - 16.0) < 1e-9, (d, c)
    new = sweep("released"); new8 = a08_runs("released")
    rep_new = sweep("fresh"); rep_new8 = a08_runs("fresh")
    SP.build_mesh = RELEASED_BUILD_MESH

    out = {"meta": C.provenance(__file__, {
        "finding": "Y6-04", "draws": "N = 250 per (Delta_r, case), as the thesis",
        "seeds": "old and new use the released seeds (paired, so old->new isolates the mesh fix); 'replicate_new' uses "
                 "fresh labels study:y6-04:... to show the Monte Carlo scatter",
        "max_abs_diff_reproducing_stored": float(max(rep)), "wall_s": None}),
        "geometry_released_mesh": geom_old, "geometry_snapped_mesh": geom_new,
        "sweep": {"old": old, "new": new, "replicate_new": rep_new},
        "a08": {"old": old8, "new": new8, "replicate_new": rep_new8}}
    out["meta"]["wall_s"] = time.time() - t0
    print("reproduction of stored (max abs diff):", max(rep))
    for case in ("A", "B"):
        for d in map(str, SP.DELTA_R):
            o, n = old[case][d], new[case][d]
            print(f"{case} dr={d:>2}: dir med {o['dir_median_deg']:.2f} -> {n['dir_median_deg']:.2f}  "
                  f"dM signed mean {o['dM_signed_mean_pct']:+.1f} -> {n['dM_signed_mean_pct']:+.1f}  "
                  f"|dM| med {o['dM_abs_median_pct']:.1f} -> {n['dM_abs_median_pct']:.1f}  rr {o['residual_rms_median']:.4f} -> {n['residual_rms_median']:.4f}")
    print(C.save("y6_04_skatepark_snapped.json", out))


if __name__ == "__main__":
    main()

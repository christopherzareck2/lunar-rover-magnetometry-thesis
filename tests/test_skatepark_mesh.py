"""Skate-park mesh (review finding Y6-04, fixed in v1.1): the central prism sits exactly where the text says, and the
stored skate-park results are what the fixed engine produces.

Release v1.0.0 set the mesh origin from the unrounded scene width, which put the central 4 x 4 x 1 m prism at
x, y in [-2.078, 1.922] m. ``build_mesh`` now makes the mesh symmetric about x = y = 0 on the 0.25 m lattice.
"""
import io
import json
import os
import sys
import contextlib

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RTOL = float(os.environ.get("LUNARMAG_RTOL", "0"))  # 0 = exact, as on the original machine
ATOL = float(os.environ.get("LUNARMAG_ATOL", "0"))


def _match(new, ref):
    """Exact equality by default; with LUNARMAG_RTOL set (CI on Linux), floats agree within RTOL/ATOL."""
    if not RTOL:
        return new == ref
    if isinstance(new, dict) and isinstance(ref, dict):
        return new.keys() == ref.keys() and all(_match(new[k], ref[k]) for k in new)
    if isinstance(new, list) and isinstance(ref, list):
        return len(new) == len(ref) and all(_match(a, b) for a, b in zip(new, ref))
    if isinstance(new, float) and isinstance(ref, float):
        return abs(new - ref) <= RTOL * max(abs(new), abs(ref)) + max(ATOL, 1e-12)
    return new == ref


@pytest.fixture(scope="module")
def SP():
    sys.path.insert(0, os.path.join(ROOT, "engine"))
    with contextlib.redirect_stdout(io.StringIO()):
        import skatepark_1cm_dr_sweep as SP
    return SP


def central_bounds(SP, mesh, delx):
    cc = mesh.cell_centers
    cfg = SP.outcrop_configs_for(delx, "A")[0][0]
    m = SP.create_shape_mask(cc[:, 0], cc[:, 1], cc[:, 2], *cfg["center"], cfg["shape"], cfg["params"],
                             margin_frac=SP.MASK_MARGIN_FRAC, margin_z=SP.MASK_MARGIN_Z) & (cc[:, 2] >= 0.0)
    h = SP.dh_fine / 2
    b = [cc[m, 0].min() - h, cc[m, 0].max() + h, cc[m, 1].min() - h, cc[m, 1].max() + h,
         cc[m, 2].min() - h, cc[m, 2].max() + h]
    return np.array(b), m.sum() * SP.dh_fine ** 3


def test_central_prism_is_exactly_at_plus_minus_two_metres(SP):
    for delx in SP.DELTA_R:
        b, vol = central_bounds(SP, SP.build_mesh(delx), delx)
        assert np.allclose(b, [-2, 2, -2, 2, 0, 1], atol=1e-12), (delx, b)
        assert vol == pytest.approx(16.0)
        mesh = SP.build_mesh(delx)
        assert np.allclose(np.mod(mesh.nodes_x / SP.dh_fine, 1.0) % 1.0, 0.0, atol=1e-9)   # faces on the lattice
        assert mesh.nodes_x[0] == -mesh.nodes_x[-1] and mesh.nodes_y[0] == -mesh.nodes_y[-1]


def test_released_mesh_had_the_offset(SP):
    b, vol = central_bounds(SP, SP.build_mesh_v1_0(1), 1)
    assert b[0] == pytest.approx(-2.0784271247, abs=1e-9) and b[1] == pytest.approx(1.9215728753, abs=1e-9)
    assert vol == pytest.approx(16.0)


@pytest.mark.parametrize("delx", [1, 10])
def test_stored_sweep_is_what_the_fixed_engine_gives(SP, delx):
    """Rerun the sweep loop of engine/skatepark_1cm_dr_sweep.main at one Delta_r; require the stored JSON exactly."""
    stored = json.load(open(os.path.join(ROOT, "results", "engine", "skatepark_1cm_dr_sweep_results.json")))["sweep"]
    survey = SP.make_survey(SP.build_survey_xyz(SP.SPACING))
    nD = int(survey.nD)
    ops = SP.build_scene_operators(delx, survey)
    G = ops["G_cen"]; K = np.linalg.pinv(G)
    m_true = SP.mag_vector_true.copy()
    for case in ("A", "B"):
        rng = np.random.default_rng(SP.MASTER_SEED + 100 * delx + (0 if case == "A" else 1))
        D = ops["d_clean"][case][None, :] + rng.normal(0.0, SP.noise_floor, size=(SP.N_MC, nD))
        M_rec = D @ K.T
        de = np.array([SP.dir_err_deg(M_rec[i], m_true) for i in range(SP.N_MC)])
        amp = np.linalg.norm(M_rec, axis=1) / (SP.mu0 * 1e9)
        me = 100.0 * (amp - SP.strengthAm) / SP.strengthAm
        ref = stored[case][str(delx)]
        assert _match(json.loads(json.dumps(SP.summarize(de))), ref["dir_err_deg"])
        assert _match(json.loads(json.dumps(SP.summarize(me))), ref["mag_err_pct_signed"])
        assert ref["n_cells_central"] == ops["nC_central"]


def test_stored_a08_is_what_the_fixed_engine_gives(tmp_path):
    """Run analysis/a08 with its output redirected to a scratch folder; require the stored JSON (except wall time)."""
    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    cwd = os.getcwd()
    try:
        import a08_skatepark_neighbor_directions as A8
    finally:
        os.chdir(cwd)
    (tmp_path / "results" / "analysis").mkdir(parents=True)
    root = A8.ROOT
    A8.ROOT = str(tmp_path)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            A8.main()
    finally:
        A8.ROOT = root
    new = json.load(open(tmp_path / "results" / "analysis" / "a08_skatepark_neighbor_directions.json"))
    ref = json.load(open(os.path.join(ROOT, "results", "analysis", "a08_skatepark_neighbor_directions.json")))
    new["meta"].pop("wall_s"); ref["meta"].pop("wall_s")
    assert _match(new, ref)

"""Analytic checks of the forward model (thesis Eqs. 3.2, 3.3, 3.6-3.8, 3.27-3.29) against independent calculations."""
import numpy as np
import pytest

from lunarmag import physics as P
from lunarmag.inversion_helpers import prism_cell_operator, dir_err_deg, M_NRM_MIN

BND = (-2.0, 2.0, -2.0, 2.0, -0.5, 0.5)          # the isolated outcrop in the code frame (sensor at z = 0)
PTS = np.array([[3.0, 0.3, 0.0], [3.0, -3.0, 0.0], [0.7, 0.2, 10.0], [-5.1, 2.3, -0.2], [1.3, -3.1, 0.4]])
MS = [np.array([0.0, 0.0, -M_NRM_MIN]), np.array([1e-3, 2e-3, -0.5e-3])]


@pytest.mark.parametrize("M", MS)
def test_closed_form_prism_matches_simpeg_operator(M):
    """Eq. 3.6 in closed form (Plouff/Blakely) vs the SimPEG operator every scenario uses: equal to ~1e-13."""
    G = prism_cell_operator(BND, PTS)
    simpeg = (G @ M).reshape(-1, 3)
    closed = P.prism_field(BND, PTS, M)
    assert np.max(np.abs(closed - simpeg)) <= 1e-12 * np.max(np.abs(simpeg))


@pytest.mark.parametrize("M", MS)
def test_closed_form_prism_matches_volume_quadrature(M):
    """Eq. 3.6 integrated numerically (24^3 Gauss-Legendre nodes) agrees with the closed form."""
    q = P.prism_field_quadrature(BND, PTS, M, n=24)
    closed = P.prism_field(BND, PTS, M)
    assert np.max(np.abs(closed - q)) <= 1e-8 * np.max(np.abs(q))


def test_prism_far_field_is_the_dipole_field():
    """Far from the prism its field tends to that of a dipole m = V M at its centre, with error ~ (a/R)^2."""
    M = np.array([0.4e-3, -1.1e-3, -2.0e-3])
    V = 4.0 * 4.0 * 1.0
    direction = np.array([0.8, 0.3, 0.52]) / np.linalg.norm([0.8, 0.3, 0.52])
    errs = []
    for R in (50.0, 100.0, 200.0, 400.0):
        p = (R * direction)[None, :]
        b = P.prism_field(BND, p, M)[0]
        d = P.dipole_field(p, [0, 0, 0], V * M)[0]
        errs.append(np.linalg.norm(b - d) / np.linalg.norm(d))
    assert errs[2] < 1e-3                              # 200 m: well inside 0.1 %
    ratios = np.array(errs[:-1]) / np.array(errs[1:])
    assert np.all((ratios > 3.5) & (ratios < 4.5))     # error falls as R^-2 (octupole term; quadrupole vanishes)


def test_prism_tensor_is_symmetric_and_traceless_outside():
    """Outside the source grad grad (1/R) integrates to a symmetric, traceless tensor (Laplace's equation)."""
    U = P.prism_tensor(BND, PTS)
    assert np.allclose(U, np.transpose(U, (0, 2, 1)))
    assert np.max(np.abs(np.trace(U, axis1=1, axis2=2))) < 1e-12


def test_field_signs_for_vertical_magnetization():
    """A downward M gives a downward field on the axis above and an upward field beside the outcrop (Y6 1.1)."""
    M = np.array([0.0, 0.0, -M_NRM_MIN])
    above = P.prism_field(BND, [[0.0, 0.0, 10.0]], M)[0]
    beside = P.prism_field(BND, [[3.0, 0.0, 0.0]], M)[0]
    assert above[2] < 0 and beside[2] > 0
    assert beside[2] == pytest.approx(0.245852, abs=2e-6)   # value quoted in the review, nT


def test_stored_peak_field_of_the_isolated_outcrop(stored):
    """The thesis's 0.25 nT peak (Section 3.2.1) recomputed in closed form along the stored drive-around path."""
    import a02_isolated_outcrop_paper_geometry as A02
    sx = A02.survey("drive_around")
    B = P.prism_field(BND, sx, [0.0, 0.0, -M_NRM_MIN])
    peak = float(np.max(np.linalg.norm(B, axis=1)))
    ref = stored("a02_isolated_outcrop_paper_geometry.json")["baseline"]["drive_around"]["peak_field_nT"]
    assert peak == pytest.approx(ref, rel=1e-10)
    assert round(peak, 2) == 0.25


def test_trm_and_induced_magnetization_conversions():
    """Eq. 3.2 at M_ref and chi_TRM = 1.58e-3 gives 1.909 uT; chi = 2e-3 in 100 nT induces 0.0663 M_ref."""
    assert P.paleofield_from_magnetization(M_NRM_MIN, 1.58e-3) * 1e6 == pytest.approx(1.908816, abs=1e-6)
    assert P.trm_magnetization(1.908816e-6, 1.58e-3) == pytest.approx(M_NRM_MIN, rel=1e-6)
    assert P.induced_magnetization(100.0, 2e-3) / M_NRM_MIN == pytest.approx(0.066315, abs=1e-6)
    assert P.induced_magnetization(300.0, 2e-3) / M_NRM_MIN == pytest.approx(0.198944, abs=1e-6)


@pytest.mark.parametrize("I,D", [(90, 0), (-90, 0), (0, 0), (45, 30), (-60, 225), (30, -135)])
def test_unit_vector_convention_round_trips(I, D):
    """Eq. 3.8: z up, inclination positive down, declination from +x toward +y."""
    u = P.unit_vector(I, D)
    assert np.linalg.norm(u) == pytest.approx(1.0)
    I2, D2 = P.inc_dec(u)
    assert I2 == pytest.approx(I, abs=1e-9)
    if abs(I) != 90:
        assert (D2 - D + 180) % 360 - 180 == pytest.approx(0.0, abs=1e-9)
    if I == 90:
        assert np.allclose(u, [0, 0, -1])


def test_error_metrics_match_the_engine():
    """Eqs. 3.27-3.28: the vectorized metrics equal the engine's per-vector versions."""
    rng = np.random.default_rng(1)
    mt = np.array([0.0, 0.0, -M_NRM_MIN])
    Mrec = mt + rng.normal(0, 2e-4, size=(50, 3))
    ours = P.direction_error_deg(Mrec, mt)
    ref = np.array([dir_err_deg(m, mt) for m in Mrec])
    assert np.allclose(ours, ref, rtol=1e-12, atol=1e-12)
    dm = P.intensity_error_pct(Mrec, mt, signed=True)
    assert np.allclose(dm, 100 * (np.linalg.norm(Mrec, axis=1) - M_NRM_MIN) / M_NRM_MIN)
    assert np.all(P.intensity_error_pct(Mrec, mt) >= 0)


def test_white_noise_asd():
    """Eq. B.5: 0.50 nT per 1 cm sample at 1 cm/s is 0.71 nT Hz^-1/2."""
    assert round(P.white_asd(0.5, 0.01, 0.01), 2) == 0.71


def test_closed_form_is_stable_on_edge_extensions():
    """Points on the extension of a prism edge (Z + R -> 0 cancellation) stay finite and match quadrature."""
    pts = np.array([[2 + 1e-9, 2 + 1e-9, 1.5], [-2 - 1e-9, 5.0, 0.5 + 1e-9], [0.3, 2.0 + 1e-9, -3.0]])
    M = [1e-3, 2e-3, -1e-3]
    closed = P.prism_field(BND, pts, M)
    assert np.all(np.isfinite(closed))
    q = P.prism_field_quadrature(BND, pts, M, n=24)
    assert np.max(np.abs(closed - q)) <= 1e-6 * np.max(np.abs(q))

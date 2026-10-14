"""Analytic checks of the estimator and nuisance designs (thesis Eqs. 3.11-3.26) and of their identity with the
frozen analysis scripts that produced the stored results."""
import numpy as np
import pytest

from lunarmag import estimators as E, spectral as S
from lunarmag.inversion_helpers import prism_cell_operator, ls_operator, M_NRM_MIN, SIGMA_B

BND = (-2.0, 2.0, -2.0, 2.0, -0.5, 0.5)
M_TRUE = np.array([0.0, 0.0, -M_NRM_MIN])


@pytest.fixture(scope="module")
def loop5(A6):
    """Drive-around at 1 m standoff sampled every 5 cm (480 stations): small enough for big Monte Carlo runs."""
    sx = A6.survey("drive_around", spacing=0.05)
    return sx, prism_cell_operator(BND, sx)


@pytest.fixture(scope="module")
def loop1(A6):
    sx = A6.survey("drive_around")
    return sx, prism_cell_operator(BND, sx)


def test_operator_matches_engine_and_is_unbiased(loop5):
    """Eq. 3.11 with equal weights is the engine's ls_operator, and K G = I (unbiased for a correct model)."""
    _, G = loop5
    K = E.least_squares_operator(G)
    assert np.allclose(K, ls_operator(G), rtol=1e-12, atol=1e-18)
    assert np.allclose(K @ G, np.eye(3), atol=1e-10)
    Kw = E.least_squares_operator(G, weights=np.full(G.shape[0], 1 / SIGMA_B))
    assert np.allclose(Kw, K, rtol=1e-10, atol=1e-18)            # equal weights cancel in K


def test_covariance_matches_monte_carlo_scatter(loop5):
    """Eq. 3.16: the sample covariance of 20000 recoveries agrees with sigma^2 (G^T G)^-1 within sampling error."""
    _, G = loop5
    C = E.estimator_covariance(G, SIGMA_B)
    K = E.least_squares_operator(G)
    rng = np.random.default_rng(2026)
    n = 20000
    chunks = []
    for _ in range(n // 2000):
        D = (G @ M_TRUE)[None, :] + rng.normal(0, SIGMA_B, size=(2000, G.shape[0]))
        chunks.append(D @ K.T)
    Mrec = np.vstack(chunks)
    C_mc = np.cov(Mrec.T)
    sd = np.sqrt(np.diag(C))
    assert np.all(np.abs(np.diag(C_mc) / np.diag(C) - 1) < 0.05)   # each variance within 5 % (1 sigma ~ 1 %)
    assert np.max(np.abs(C_mc - C) / np.outer(sd, sd)) < 0.05      # correlations within 0.05
    assert np.all(np.abs(Mrec.mean(0) - M_TRUE) < 4 * sd / np.sqrt(n))   # unbiased
    C_w = E.estimator_covariance(G, np.full(G.shape[0], SIGMA_B))  # Eq. 3.17 with equal sigma = Eq. 3.16
    scale = np.outer(sd, sd)                                       # compare relative to the covariance scale
    assert np.max(np.abs(C_w - C) / scale) < 1e-10
    assert np.max(np.abs(E.estimator_covariance(G, noise_cov=SIGMA_B ** 2 * np.eye(G.shape[0])) - C) / scale) < 1e-8


def test_covariance_predicts_the_stored_baseline(loop1, stored):
    """Eqs. 3.20-3.21: the covariance alone predicts the stored 1000-draw mean direction error (3.18 deg)."""
    _, G = loop1
    C = E.estimator_covariance(G, SIGMA_B)
    pred = E.predicted_mean_direction_error_deg(C, M_TRUE)
    mc = stored("a02_isolated_outcrop_paper_geometry.json")["baseline"]["drive_around"]["dir_err_deg"]["mean"]
    assert pred == pytest.approx(mc, rel=0.04)
    par, perp2 = E.small_error_moments(C, M_TRUE)
    std_signed = stored("a02_isolated_outcrop_paper_geometry.json")["baseline"]["drive_around"]["dM_signed_pct"]["std"]
    assert 100 * np.sqrt(par) / M_NRM_MIN == pytest.approx(std_signed, rel=0.06)   # intensity scatter, 5.5 %


def test_marginal_covariance_is_the_schur_block(loop5):
    """Eq. 3.23 equals the magnetization block of sigma^2 (A^T A)^-1 for A = [G E] (Eq. 3.22 and its least-squares solution)."""
    sx, G = loop5
    N = E.offset_columns(len(sx))
    A = np.column_stack([G, N])
    full = SIGMA_B ** 2 * np.linalg.inv(A.T @ A)
    Cm = E.marginal_covariance(G, N, SIGMA_B)
    s_ = np.sqrt(np.diag(full[:3, :3]))
    assert np.max(np.abs(Cm - full[:3, :3]) / np.outer(s_, s_)) < 1e-8
    rng = np.random.default_rng(3)
    c = rng.normal(size=3)
    d = G @ M_TRUE + N @ c
    sol = E.augmented_solution(G, N, d)
    assert np.allclose(sol[:3], M_TRUE, rtol=1e-8) and np.allclose(sol[3:], c, rtol=1e-8)


def test_gradient_basis_is_symmetric_traceless_orthonormal_and_identical_to_a06(loop1, A6):
    """Eq. 3.24: five symmetric, traceless, Frobenius-orthonormal tensors; columns T_k (r - r_c); bit-identical to a06."""
    sx, _ = loop1
    cols, T = E.gradient_basis(sx)
    for Tk in T:
        assert np.allclose(Tk, Tk.T) and abs(np.trace(Tk)) < 1e-15
    gram = np.array([[np.sum(a * b) for b in T] for a in T])
    assert np.allclose(gram, np.eye(5), atol=1e-15)
    r = sx - sx.mean(0)
    assert np.allclose(cols[:, 2], (r @ T[2].T).reshape(-1))
    ref_cols, _ = A6.gradient_basis(sx)
    assert np.array_equal(cols, ref_cols)
    # any uniform curl- and divergence-free gradient is reproduced exactly by the five columns
    rng = np.random.default_rng(4)
    S_ = rng.normal(size=(3, 3)); S_ = S_ + S_.T; S_ -= np.trace(S_) / 3 * np.eye(3)
    b = (r @ S_.T).reshape(-1)
    coef, *_ = np.linalg.lstsq(cols, b, rcond=None)
    assert np.allclose(cols @ coef, b, atol=1e-12)


def test_headings_match_a06(loop1, A6):
    sx, _ = loop1
    assert np.array_equal(E.headings(sx), A6.headings(sx))


def test_tilt_error_identical_to_a09_and_first_order_correct(A9):
    """Eqs. B.6-3.25: exact tilt error bit-identical to a09; first-order form error is O(dtheta^2)."""
    sx = A9.two_loops()
    rng = np.random.default_rng(5)
    B = rng.normal(size=3); B *= 300.0 / np.linalg.norm(B)
    psi = E.headings(sx)
    for dth in (0.01, 0.1, 1.0):
        a = rng.uniform(0, 2 * np.pi)
        exact = E.tilt_error(sx, B, dth, a)
        assert np.array_equal(exact, A9.tilt_error(sx, B, dth, a))
        first = E.tilt_error_first_order(psi, B, dth, a)
        rel = np.max(np.abs(exact - first)) / np.max(np.abs(first))
        assert rel < 0.6 * np.radians(dth)              # second-order remainder: ~ dtheta/2 relative
    R = E.rodrigues([np.cos(0.3), np.sin(0.3), 0.0], np.radians(2.0))
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
    k = np.array([np.cos(0.3), np.sin(0.3), 0.0])       # Eq. B.10: R_psi k = cos(psi) k + sin(psi) z x k
    for p in (0.0, 0.7, 2.5):
        assert np.allclose(E.heading_rotation(p) @ k, np.cos(p) * k + np.sin(p) * np.cross([0, 0, 1.0], k))


def test_heading_columns_absorb_first_order_tilt(A9):
    """Eq. 3.26: the six heading columns reproduce the first-order tilt error exactly, and the exact error to O(dtheta^2)."""
    sx = A9.two_loops()
    psi = E.headings(sx)
    H = E.heading_columns(psi)
    B = np.array([120.0, -40.0, 250.0])
    first = E.tilt_error_first_order(psi, B, 0.3, 1.1)
    coef, *_ = np.linalg.lstsq(H, first, rcond=None)
    assert np.max(np.abs(H @ coef - first)) < 1e-10 * np.max(np.abs(first))
    a_, b_ = E.tilt_coefficients(B, 0.3, 1.1)
    assert np.allclose(coef, np.r_[a_, b_])
    exact = E.tilt_error(sx, B, 0.3, 1.1)
    coef, *_ = np.linalg.lstsq(H, exact, rcond=None)
    resid = np.sqrt(np.mean((exact - H @ coef) ** 2)) / np.sqrt(np.mean(exact ** 2))
    assert resid < np.radians(0.3)


def test_spectral_band_legacy_reproduces_stored_and_corrected_matches_review(loop1, stored):
    """a06 part F (stored) used |rfft|^2 without one-sided doubling; the corrected weights give the review's
    numbers (Y6-01): DC 47.53 % (not 64.43 %), 94.98 % through two cycles per loop, 95 % first reached at bin 4."""
    sx, G = loop1
    K = ls_operator(G)
    n = len(sx)
    k, P_legacy = S.estimator_power(K, n, 0.01, one_sided=False)
    ref = stored("a06_isolated_robustness.json")["F_estimator_band"]["drive_around"]["quantiles_cyc_per_m"]
    assert S.band_quantiles(k, P_legacy) == ref
    assert S.dc_fraction(P_legacy) == pytest.approx(0.644304, abs=1e-6)
    k, P = S.estimator_power(K, n, 0.01)
    assert S.dc_fraction(P) == pytest.approx(0.475257, abs=1e-6)
    assert np.cumsum(P)[2] == pytest.approx(0.949778, abs=1e-6)
    assert int(np.searchsorted(np.cumsum(P), 0.95)) == 4
    _, Pd = S.estimator_power(K, n, 0.01, component="direction", M_true=M_TRUE)
    assert S.dc_fraction(Pd) == pytest.approx(0.159192, abs=1e-6)
    assert np.cumsum(Pd)[2] == pytest.approx(0.974685, abs=1e-6)

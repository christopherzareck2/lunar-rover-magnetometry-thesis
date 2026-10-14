# Theory notes: how the code implements the thesis equations

This page maps each equation of the thesis's modeling chapter (and the noise equations of Chapter 5) to the code
that computes it. The equation numbers are those of the thesis as submitted (read from the LaTeX labels). Numbers starting with B are in Appendix B,
where the final edit moved the tilt derivation and the noise model; "not numbered" marks a display that is no longer
numbered in the thesis. The LaTeX labels (the second column) do not change when chapters are renumbered, so look a label up in the
thesis PDF's source or its `.aux` file if the numbers have moved.

Four conventions hold throughout:

- Positions are in metres, magnetization in A/m and fields in nT.
- The frame is right-handed local Cartesian: x and y are horizontal and z points up. Inclination is positive
  downward, and declination is measured from +x toward +y.
- Data vectors are station-major: (bx, by, bz) at station 1, then station 2, and so on.
- The isolated-outcrop code places the prism at z = -0.5 to 0.5 m with the sensor at z = 0. This is the thesis
  geometry (prism z = 0 to 1 m, sensor at 0.5 m) with both shifted down by 0.5 m. Every field and every recovery is
  identical.

`tests/test_physics.py` and `tests/test_estimators.py` hold the checks quoted in the right-hand column.

## Signals and the magnetization (Section 3.2)

| Eq. | Label | Statement | Code | Checked by |
|---|---|---|---|---|
| 3.1 | `eq:field_decomposition` | B_meas = B_NRM + B_bg + B_IMF + B_sc + eps | The scenarios add an NRM field, a noise draw (`rng.normal(0, SIGMA_B)`; OU in `correlated_noise_sensitivity.make_noise`) and, in a09, a site field, tilt, drift and induced magnetization (`analysis/a09_systematics.budget`) | `test_regression.py` (exact rerun) |
| 3.2 | `eq:nrm` | M_NRM = (chi_TRM / mu0) B_paleo | `physics.trm_magnetization`, `physics.paleofield_from_magnetization` | M_ref = 2.4e-3 A/m with chi_TRM = 1.58e-3 gives 1.908816 uT |
| 3.3 | `eq:induced` | M_ind = chi B_site / mu0 | `physics.induced_magnetization`; in the budget, `a09.budget` (`chi * Bc * 1e-9 / MU0`, removed with a lognormal chi error) | chi = 2e-3 in 100 nT gives 0.0663 M_ref (300 nT: 0.199) |
| 3.4 | `eq:plug` | field of a regolith layer with a plug removed = field of the plug with -M_reg | `studies/regolith/y10_verify.py` checks that a slab with a hole equals the plug; `y10_regolith.py` runs the study | |
| 3.5 | `eq:induced_contrast` | induced field of a susceptibility contrast | no code in this repository yet (as Eq. 3.4) | |

## Forward model and coordinates (Section 3.3)

| Eq. | Label | Statement | Code | Checked by |
|---|---|---|---|---|
| 3.6 | `eq:forward_integral` | B(r) = (mu0/4pi) int_V [3(M.n)n - M]/R^3 dV' | `inversion_helpers.prism_cell_operator` (SimPEG; Geoana backend for these operators, Choclo for the skate park); closed form `physics.prism_tensor` / `prism_field`; quadrature `physics.prism_field_quadrature`; far field `physics.dipole_field` | The closed form matches SimPEG to 1e-12 relative and 24^3-point Gauss-Legendre quadrature to 1e-8. The far field converges to the dipole m = V M with error falling as R^-2. U is symmetric and traceless outside the source. The 0.25 nT peak along the drive-around (Section 3.2.1) is recomputed in closed form |
| 3.7 | `eq:forward_linear` | d = G m | `prism_cell_operator` (K = 1, columns Mx, My, Mz), `three_layer_G` (K = 3, SimPEG component-major columns), `isolated_outcrop_1cm.per_receiver_kernel` (3 x 3 per station) | as above |
| 3.8 | `eq:th_unit_vector` | u(I, D) = (cos I cos D, cos I sin D, -sin I) | `physics.unit_vector`, `physics.inc_dec`; `a06.dir_scan` | round trips; I = 90 gives -z |

The prism closed form uses the corner sums of Plouff (1976) and Blakely (1996). Write I(r) = int_V dV'/|r - r'| and
let (X, Y, Z) be the corner coordinates measured from the observer, with R the corner distance. Then
B = (mu0/4pi) (grad grad I) M, where

    U_xx = sum -atan(YZ/(XR)),   U_xy = sum ln(Z + R),   U_xz = sum ln(Y + R)   (and cyclic permutations)

Each sum alternates in sign over the eight corners. Because mu0/4pi = 100 nT m/A, a magnetization of 1 A/m acting
through the dimensionless U gives 100 U nT. SimPEG reaches the same scale with a source amplitude of
`AMP_NT = mu0 * 1e9`.

## Inversion and estimator (Sections 3.4 to 3.4.3)

| Eq. | Label | Statement | Code | Checked by |
|---|---|---|---|---|
| not numbered | `eq:voxel_objective` | voxel objective with smallness and smoothness | `engine/inversion_approach_comparison.py` (Cartesian and spherical voxel solves, SimPEG) | Table 4.1 results stored; rerun in `make rerun` |
| 3.9 | `eq:param_vector` | p = [Mx,1..K; My,1..K; Mz,1..K] (component-major) | `inversion_helpers.three_layer_G` column order; `layer_slices` | a03/a07 reruns in `make rerun` |
| 3.10 | `eq:objective` | Phi(p) = W_d(d - G_K p)^2 + beta p^2 | `inversion_helpers.iterative_recover_M`, `iterative_recover_M_ref` (SimPEG ProjectedGNCG, zero reference) | three-layer runs in `make rerun` |
| 3.11 | `eq:ls` | p_hat = (G^T W^2 G)^-1 G^T W^2 d | `inversion_helpers.ls_operator` (equal weights, so W cancels), `estimators.least_squares_operator` (any weights) | `K G = I`; equals the engine operator; weights cancel |
| 3.12 | `eq:damped` | p_beta = (A_d + beta I)^-1 G^T W^2 d | `inversion_helpers.iterative_recover_M_ref` (SimPEG, beta from `BetaEstimate_ByEig`); the damped 9-parameter solver of `engine/three_layer_wall.py` | Table 4.3 damped rows (stored) |
| 3.13 | `eq:damped_shrink` | p_beta = sum lambda_i/(lambda_i + beta) (v_i . p) v_i | analysis only; no separate function | |
| 3.14 | `eq:th_data_model` | d = G_K p + n | `inversion_helpers.linear_mc` | exact rerun of a02/a06/a09 parts |
| not numbered | `eq:th_estimator` | p_hat = K d, K = (G^T G)^-1 G^T | `inversion_helpers.ls_operator`, `estimators.least_squares_operator(G)` | `K G = I` |
| 3.15 | `eq:th_estimate_error` | p_hat = p + K n | `linear_mc`, `a06.lsK` | the sample mean of 20,000 recoveries is unbiased within 4 standard errors |
| 3.16 | `eq:th_covariance` | C_p = sigma^2 (G^T G)^-1 | `estimators.estimator_covariance(G, sigma)`; sampled in `a06.dir_scan` (Table 4.2) | the sample covariance of 20,000 recoveries agrees within 5% (variances) and 0.05 (correlations) |
| 3.17 | `eq:th_covariance_weighted` | C_p = K W^-2 K^T = (G^T W^2 G)^-1 | `estimator_covariance(G, sigma_array)`; general noise: `estimator_covariance(G, noise_cov=C_n)` = K C_n K^T | equal-sigma case equals Eq. 3.16 |
| 3.18 | `eq:th_covariance_correlated` | C_p = K C_n K^T | `estimators.estimator_covariance(G, noise_cov=C_n)` | equals Eq. 3.16 for C_n = sigma^2 I |
| 3.19 | `eq:th_gls_bound` | K C_n K^T = (G^T C_n^-1 G)^-1 + Delta C_n Delta^T | no dedicated function (the GLS term is `estimator_covariance(G, sigma_array)` for diagonal C_n) | |
| 3.20 | `eq:th_small_error_metrics` | dM/M ~ eps_par/M, theta ~ eps_perp/M | `estimators.small_error_moments`, `predicted_mean_direction_error_deg` | the covariance alone predicts the stored 1000-draw mean direction error (3.18 deg) to within 4% and the intensity scatter (5.5%) to within 6% |
| 3.21 | `eq:th_par_perp_variance` | sigma_par^2 = u^T C u, E eps_perp^2 = tr C - u^T C u | `small_error_moments` (E eps_perp^2 is a second moment, not Var theta) | as above |

## Nuisance terms: offsets, gradient and tilt (Sections 3.4.4 and 3.4.5)

| Eq. | Label | Statement | Code | Checked by |
|---|---|---|---|---|
| 3.22 | `eq:th_augmented_model` | d = G p + E c + n, E = [I3; ...; I3] | `estimators.offset_columns`; `a06.nuisance_columns(sx, "world_dc")`; rover-frame offsets rotate with heading (`"rover_dc"`) | recovers p and c exactly from noise-free data |
| not numbered | `eq:th_augmented_solution` | [p; c] = (A^T A)^-1 A^T d | `estimators.augmented_solution`; `a06.augmented` + `a06.lsK` (columns linearly dependent on earlier ones are dropped) | as above |
| 3.23 | `eq:th_marginal_covariance` | C = sigma^2 (G^T P_E G)^-1 | `estimators.marginal_covariance` | equals the Schur block of sigma^2 (A^T A)^-1 |
| 3.24 | `eq:th_gradient_basis` | T1..T5, symmetric and traceless | `estimators.gradient_basis` (same arithmetic as `a06.gradient_basis`) | symmetric, traceless, Frobenius-orthonormal; any uniform curl- and divergence-free gradient is reproduced exactly; bit-identical to a06 |
| B.6 | `eq:th_heading_rotation` | R_psi | `estimators.heading_rotation`; headings from `estimators.headings` (= `a06.headings`, central differences of the path) | bit-identical to a06 |
| B.7 | `eq:th_tilt_error_exact` | e(psi) = R_psi (R_err - I) R_psi^T B_amb | `estimators.tilt_error` (= `a09.tilt_error`) | bit-identical to a09 |
| B.8 | `eq:th_rodrigues` | R_err = I + sin(d) [k]x + (1 - cos d) [k]x^2 | `estimators.rodrigues` | orthogonal, det 1 |
| B.9 | `eq:th_tilt_error_first_order` | e ~ d (R_psi k) x B_amb | `estimators.tilt_error_first_order` | differs from the exact form by O(d^2) |
| B.10 | `eq:th_rotated_axis` | R_psi k = cos psi k + sin psi (z x k) | used in `tilt_error_first_order` | identity checked at three headings |
| 3.25 | `eq:th_tilt_cos_sin` | e ~ cos psi a + sin psi b | `estimators.tilt_coefficients` | the fitted coefficients equal (a, b) |
| 3.26 | `eq:th_tilt_columns` | H = [cos psi I3, sin psi I3] | `estimators.heading_columns` (same columns as the `fit_tilt` branch of `a09.budget`) | absorbs the first-order tilt error to 1e-10 and the exact error to O(d) |

## Recovery metrics (Section 3.5)

| Eq. | Label | Statement | Code |
|---|---|---|---|
| 3.27 | `eq:dir_error` | theta_err = arccos(M_true . M_rec / norms) | `inversion_helpers.dir_err_deg`, `physics.direction_error_deg` |
| 3.28 | `eq:int_error` | dM = abs(abs(M_rec) - abs(M_true)) / abs(M_true) x 100% | `physics.intensity_error_pct`. The engine's `mag_err_pct` is the signed version, and the analysis scripts store both `dM_signed_pct` and `dM_abs_pct` |
| 3.29 | `eq:residual_rms` | normalized residual rms | `physics.residual_rms`, `linear_mc`. For an undamped fit E[RMS^2] = (N_d - q)/N_d |

## Noise and drift (Section 5.1; the equations are in Appendix B)

| Eq. | Label | Statement | Code |
|---|---|---|---|
| B.3 | `eq:th_ou_noise` | n_k = w_k + c_k, with c_k an Ornstein-Uhlenbeck (OU) process | `correlated_noise_sensitivity.make_noise` (0.05 nT white plus OU, total 0.50 nT per axis); `effective_sample_count` |
| B.4 | `eq:th_space_time_map` | T = lambda/v, f = v k, tau = L/v | used in `a06.part_F` and the Fig. 5.1 axis (1 cm = 1 s at 1 cm/s) |
| B.5 | `eq:th_white_asd` | sqrt(S_t) = sigma sqrt(2 dx / v) = 0.71 nT Hz^-1/2 | `physics.white_asd` |
| not numbered | `eq:th_twenty_minutes` | T = 12 m / 1 cm/s = 20 min | `spectral.estimator_power`, `band_quantiles` (see the note below) |

Note on the spectral argument behind the "twenty minutes" statement (Section 5.1; derivation in Appendix Section B.3), from the review finding Y6-01, which this code verifies. Part F of
a06 summed |rfft|^2 without doubling the non-DC bins. This overweights the survey-mean bin: it gives a DC share of
64.43% where the correct share is 47.53%. `spectral.estimator_power(..., one_sided=True)` gives the correct
weights:

- Over the trace of the recovered-vector covariance, 94.98% of the weight lies below two cycles per loop, and 95%
  is first reached at bin 4, not bin 2.
- Projected onto the plane normal to M (the direction error), the DC share is 15.92% and the weight through bin 2
  is 97.47%.
- For the intensity alone, the DC share is 91.51%.

The legacy normalization is kept (`one_sided=False`) so that the stored numbers regenerate exactly.

## Chapter 5 and 7 equations

| Eq. | Label | Statement | Code |
|---|---|---|---|
| 5.1-5.3 | `eq:budget_data`, `eq:budget_estimate`, `eq:budget_msq` | the combined-budget data model, estimate and mean-square error | `analysis/a09_systematics.budget` (see the next section) |
| 5.4 | `eq:reg_error` | regolith error | regolith scripts (not yet in this repository) |
| 5.5 | `eq:sobol` | Sobol indices | `studies/sobol.py` (results `studies/results/sobol.json`, Fig. 5.3) |
| 5.6-5.7 | `eq:alpha95`, `eq:alpha95_kappa` | Fisher alpha95 | `studies/flow_mean.py` (Fig. 5.4) |
| 7.1-7.6 | `eq:ai_*` | linear model, Schur complement, scatter, loss of the learned estimator | `ml/` (`common.py`, `train.py`, `evaluate.py`; Fig. 7.1) |

## Error budget (Section 5.3, Table 5.1, Fig. 5.2)

`analysis/a09_systematics.budget` combines the error sources in data space and then evaluates the nonlinear
metrics. Each draw adds the following to the outcrop field `Gt @ (M + M_ind)`:

- a site field (`dipole_field`, a dipole 30 deg below the horizon at random azimuth);
- a tilt error (Eq. B.7);
- induced magnetization `chi B / mu0`, removed with a lognormal susceptibility error `chi exp(N(0, s^2))`;
- slow drift (`ou_drift_exact`, scaled to an exact pooled rms about the survey mean);
- white noise.

The draw is then inverted with the imaged (mis-posed, `warped_operator`) exposed prism, with offsets, the gradient
and optionally heading and drift columns fitted. The table's "noise-free bias" is the mean, over draws, of the
noise-free angular error. See `docs/known-issues.md` for what that statistic is and is not.

## Voxel inversions (moved from thesis Section 3.4.2)

The thesis body keeps one paragraph per inversion approach; the objective and solver settings it no longer prints are kept here.

Both voxel inversions minimize, in `SimPEG`,

    Phi(m) = || W_d (d_obs - G m) ||^2 + beta * sum_{j=1..3} ( alpha_s || W_s m_j ||^2 + sum_{k in x,y,z} alpha_k || W_k D_k m_j ||^2 )

where `W_d = diag(1/sigma_i)` weights each datum by its noise standard deviation, `m_j` holds one of the three model components for every cell, `D_k` is a first-difference operator along `k`, `alpha_s` and `alpha_k` set the balance of smallness and smoothness, and `W_s`, `W_k` include sensitivity-based cell weights that counteract the decay of the field with distance from the traverse (Li and Oldenburg 1996). The trade-off parameter `beta` starts at 10 times the ratio of the largest eigenvalues of the misfit and regularization terms and is reduced by a factor of 3 per iteration until the data misfit (the first term) reaches the number of data, or 25 iterations have run.

The spherical voxel inversion describes each cell by `(a, theta, phi)`: in `SimPEG`, `theta = arcsin(M_z / a)` is an elevation in radians, positive upward (so `theta = -I` in the thesis convention of Eq. `eq:th_unit_vector`), and `phi = atan2(M_y, M_x)` is the azimuth `D`. The model becomes `m = S(u)`, where `u` holds the amplitude and two angles of every cell and `S` converts them to Cartesian components; the regularization acts on `u`. It is minimized with a projected Gauss-Newton method that keeps the angles within their ranges, started from the Cartesian solution. The angle penalties are ordinary L2 penalties on `theta` and `phi`, so they depend on where the angles wrap and are not rotation invariant.

The comparison of Section 4.1 restricts all three inversions to the 512 cells of the test cube; implementation: `engine/inversion_approach_comparison.py`.

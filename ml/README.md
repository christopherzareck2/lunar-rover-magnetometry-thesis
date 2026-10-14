# ml/: a learned (amortized) estimator for the isolated outcrop

A small neural network, trained on simulated surveys, that estimates an outcrop's magnetization vector and its
uncertainty when the buried extent, the imaged pose, slow drift, sensor tilt and site-field curvature are all
uncertain. It is compared with the thesis's least-squares estimators and with a classical profile estimator that also
fits the buried extent, and checked against the Cramer-Rao and van Trees bounds.

Everything physical comes from the thesis code (`engine/`, `analysis/a06_*.py`, `analysis/a09_systematics.py`),
imported read-only. Nothing outside `ml/` is modified. No new dependencies: the network is plain numpy.

## Headline numbers (loops 0.5 m off centre, held-out simulations; `results/sbi_results.json`)

| | 11-column LS (thesis) | Profile LS over burial | Network ensemble |
|---|---|---|---|
| All errors at up to full tolerance, 20,000 surveys: mean / p95 direction error | 9.48 / 20.4 deg | 3.19 / 7.2 deg | **2.28 / 5.5 deg** |
| Same: mean signed intensity error | +23.8 % | +1.3 % | +0.5 % |
| Same: 95 % credible region covers the truth | 2.6 % (formal covariance) | n/a | 94.7 % |
| White noise only: MSE / Cramer-Rao bound | 0.998 +/- 0.017 | 1.62 | 1.013 +/- 0.017 (network trained on that case) |

The network does not beat the bound in the linear-Gaussian case (van Trees bound 0.998), loses to LS when there is no
burial, does not reduce drift error, and is overconfident outside its training ranges. Chapter 7
of the thesis gives the write-up and the threats to validity; the sections at the end of this file give the
equations and the training details.

## Files

| File | What it does |
|---|---|
| `common.py` | geometry, closed-form prism kernel, linear bases (offsets, gradient, heading, burial slabs, pose derivatives, drift), compression `z = Q^T d`, nuisance simulator, LS and profile-LS estimators, metrics |
| `validate_forward.py` | closed-form kernel vs the SimPEG operator (exposed, buried, slab, mis-posed prisms) -> `results/forward_validation.json` (worst relative error 7.4e-14) |
| `replicate_thesis.py` | LS11 on this simulator vs stored a09 `Z_single_*` results -> `results/replication_vs_a09.json` (18 of 18 cells within 1.5 standard errors) |
| `check_compression.py` | how much of the burial and pose signal lies outside the feature span -> `results/compression_check.json` |
| `nn.py` | Gaussian-output MLP (mean + full covariance), hand-written backprop, Adam; `python ml/nn.py` runs a finite-difference gradient check |
| `generate.py` | simulates every training, validation and test set in seeded chunks -> `data/*.npz` (not tracked; about 350 MB) |
| `train.py` | trains one network -> `models/<set>_s<seed>.npz`, `results/train_<set>_s<seed>.json` |
| `evaluate.py` | all estimators on all test sets, calibration, bounds, timing -> `results/sbi_results.json` |
| `fig_ml.py` | `figures/ml_fig1_burial.{pdf,png}`, `figures/ml_fig2_calib.{pdf,png}` (thesis house style via lunarmag.plotting, no transparency, PDF/A-1b ready; `make figures` copies them into the thesis as ch7_burial.pdf and ch7_calib.pdf) |
| `run_all.sh` | the whole study from scratch |

## Reproduce

```bash
conda activate lunar-mag
bash ml/run_all.sh          # about 30 min on a 14-core laptop
```

Seeds: every simulation chunk uses `seed_for("ml:<set>:<chunk>")` from `engine/inversion_helpers.py`, so results do
not depend on the number of worker processes. Network weights depend on BLAS threading at the last few digits;
the reported metrics are stable to about 0.01 deg across the four ensemble members.

## Conventions

- Magnetization is in units of M_ref = 2.4e-3 A/m inside this package; fields are in nT; sigma_B = 0.50 nT.
- The pose error moves the true body and keeps the estimator's model nominal (a09 does the reverse, moving the
  model). This keeps the estimator's model fixed, which an amortized estimator needs; `replicate_thesis.py` shows
  the two agree within Monte Carlo error.
- Induced magnetization is not simulated (it is removed by the same susceptibility correction for every estimator).

## Derivations and training details (moved from thesis Chapter 7)

The thesis chapter keeps the results and one-line statements; the detail lives here. Numbers trace to
`results/sbi_results.json` ([J]), `results/train_*.json`, `results/compression_check.json` and
`results/forward_validation.json`.

### Prior information (van Trees bound)

For any estimator, a prior pi that is continuously differentiable and vanishes on the boundary of its support, and a
regular likelihood, the van Trees inequality (Gill and Levit 1995) gives

    E[(theta_hat - theta)(theta_hat - theta)^T]  >=  ( E_pi[I] + J_pi )^-1,     J_pi = E_pi[ grad ln pi  grad ln pi^T ],

with the expectation over theta ~ pi and the data. Derivation: let psi = grad_theta ln[p(d|theta) pi(theta)].
Integrating by parts over theta (boundary terms zero) gives E[(theta_hat - theta) psi^T] = I_id; the cross terms of
E[psi psi^T] vanish because the score has zero mean given theta, so E[psi psi^T] = E_pi[I] + J_pi; the matrix
Cauchy-Schwarz inequality then gives the bound. Its leading 3x3 block bounds the magnetization.

The training prior draws the direction of m uniformly and ln r ~ N(ln 2, s^2), r = |m|, s = 0.35. Its density in 3-D
is pi(m) ~ r^-3 exp[-(ln r - ln 2)^2 / (2 s^2)], so grad ln pi = -(3 + w/s) m / r^2 with w = (ln r - ln 2)/s ~ N(0,1).
Averaging over directions (E[m m^T / r^2] = I/3) and then over w, with r = 2 e^{s w},

    J_pi,m = (1/12) E[(3 + w/s)^2 e^{-2 s w}] I = e^{2 s^2} (1 + s^-2) / 12 I = 0.976 I,

using E[e^{-2sw}] = e^{2s^2}, E[w e^{-2sw}] = -2s e^{2s^2}, E[w^2 e^{-2sw}] = (1 + 4s^2) e^{2s^2}.
[J] bounds.J_prior_M_per_component = 0.97560 (80-node Gauss-Hermite; the closed form agrees).

The data information on the evaluation survey (both loops 0.5 m off centre, offsets and gradient fitted, no priors) is
1/diag(CRLB) = 848, 550 and 338 per component (trace of the CRLB 0.0059533 M_ref^2, [J] bounds.crlb_trace). With
independent priors N(0, 50^2 nT^2) on each offset and N(0, 1) on each gradient coefficient as well, the trace of the
van Trees bound is 0.998 times the Cramer-Rao trace ([J] bounds.bcrb_over_crlb = 0.99780). So with white noise as the
only error, no estimator can have a prior-averaged MSE of the magnetization vector more than 0.22 % below least
squares. The statement is about the prior-averaged vector MSE, not the direction error of any one outcrop.
(Thesis Table 7.1's vertical information of 174 is the centred two-loop survey with offsets only; a different design.)

### Compression

The 19,194 data are projected onto an orthonormal basis Q (3P x 67) spanning: the magnetization columns; the offset,
gradient and heading columns; the fields of ten 0.1 m slabs of buried root; the 15 first-order derivatives of the
outcrop field with respect to the five pose parameters (taken at zero burial); Legendre polynomials of degree 1 to 3 in
station order for each axis. Features z = Q^T d. Because the noise is white, z has white noise of the same variance.
Every fixed least-squares estimator in the chapter is linear in z (the code checks K Q Q^T = K to 1e-9), and for any
linear model whose columns lie in the span, ||d - A theta||^2 = ||z - Q^T A theta||^2 + ||(I - Q Q^T) d||^2, so z is
sufficient for that model. For burial and pose the slabs and pose derivatives are a linear score compression (Heavens
et al. 2000; Alsing and Wandelt 2018), exact only near the model it was built around.

Sampled check (`check_compression.py`, 300 draws at |m| = 2): burial alone lies in the span to within 2e-8 of the change
it causes; pose alone leaves 3.7 % of the change outside (median; 8.1 % at p95), a median energy of 0.10 noise
variances (0.67 at p95); burial and pose together leave 0.29 noise variances at the median, 1.2 at p95 and 3.7 at most.
These are samples, not bounds. The network input drops the eight offset and gradient directions (invariance to those
parameters) and adds log of the normalized residual energy ||(I - Q Q^T) d||^2 / [sigma_B^2 (3P - 67)]: 60 inputs.

### Objective

The network returns mu(x) and a full covariance Sigma(x) through a Cholesky factor of the precision and minimizes the
Gaussian NLL E[ 1/2 (m - mu)^T Sigma^-1 (m - mu) + 1/2 ln|Sigma| ]. For fixed x, with conditional mean m_bar and
covariance C of m, the conditional expectation of the bracket is 1/2 ln|Sigma| + 1/2 tr(Sigma^-1 [C + (m_bar - mu)
(m_bar - mu)^T]). The trace is smallest at mu = m_bar for any Sigma, and setting the derivative with respect to
Sigma^-1 to zero gives Sigma = C. So at the population optimum the network mean is the posterior mean (the MMSE estimator
under the training prior) and the covariance the posterior covariance, both given the features. Neither makes the
posterior Gaussian, and both concern p(m|x), which equals p(m|d) only as far as x is sufficient; coverage has to be
measured (the thesis does).

### Network and training

Multilayer perceptron, three hidden layers of 256 units, SiLU activations (Elfwing et al. 2018), plus a linear path
from input to output mean initialized at the 11-column least-squares estimator; 149,696 parameters. Adam (Kingma and
Ba 2015), batch 512, 40 epochs on 400,000 surveys, keeping the epoch with the lowest validation loss on 20,000 more
(best epochs 20 to 27; `results/train_full_s*.json`). Four networks from different initial weights and data orders are
combined by matching moments (Lakshminarayanan et al. 2017): ensemble mean = average of the four means; ensemble
covariance = average of (Sigma_e + mu_e mu_e^T) minus the outer product of the ensemble mean. Plain numpy with
hand-written gradients, checked against finite differences to 1.4e-8 relative (`python ml/nn.py`).

### Cost

Per survey: ensemble 0.67 ms end to end, of which the projection onto Q is 0.42 ms; profile LS over 21 depths 3.1 ms
([J] timing_s_per_survey). Simulating the 400,000 training surveys took about 7.5 min on 7 cores and each network
trained in about 3.2 min, so a new survey geometry costs about 20 min. Desktop timings on a 14-core laptop.

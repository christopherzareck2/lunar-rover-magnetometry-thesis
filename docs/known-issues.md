# Known issues in the released simulations

The stored results in `results/` were produced by the v1.0.0 code, and this repository reproduces them exactly
(`make reproduce`). A physics review of Chapters 3 to 5 on 2026-10-13 (the "Y6" review) was done by an AI language model, not by a
human physicist, and a second AI model then checked each finding against the code and stored results. No human
reviewer is recorded (see [changes-from-paper-release.md](changes-from-paper-release.md)). The review found the core physics correct: the
field kernel and its sign and scale, the SI conversions, the least-squares estimator, and the error metrics. It also
found places where the code does something narrower than, or different from, what a
sentence of the thesis says.

The code-level findings are listed below. Each one has a status:

- **Verified**: rechecked against the code and stored results while building this repository.
- **Read**: confirmed by reading the code.
- **Not checked**: not independently confirmed here.

Except for Y6-04, which was fixed and rerun in v1.1, none of these findings changes a stored number. None of the
others was "fixed" by editing the frozen analysis scripts, because fixing them would change results the thesis
quotes, so each fix is a separate, deliberate rerun. Whether the text or the simulation changes is the author's
call.

| ID | Finding | Status | Where | What it means for the text |
|---|---|---|---|---|
| Y6-01 | The estimator wavenumber band (a06 part F) sums the `rfft` power without doubling the non-DC bins. Its DC share is 64.43%; the correct one-sided value is 47.53%. The share through bin 2 is 96.60% before the fix and 94.98% after, and 95% is first reached at bin 4, not bin 2. The direction-only share through bin 2 is 97.47%. | **Verified** (`tests/test_estimators.py::test_spectral_band_*`, `lunarmag.spectral`) | `analysis/a06_isolated_robustness.py` part F | Statements that rest on the median bin being 0, or on 95% at 12 m, need the corrected numbers or a "direction error" qualifier |
| Y6-02 | The budget's "noise-free bias" is the mean over draws of the angle between the noise-free recovery and M (`mean(angle(M_nf, M))`). It is not the angle of the mean vector, so random pose or tilt errors that cancel across outcrops still produce a positive value. | **Read** | `a09.budget` | Call it the "mean noise-free angular error" (white noise and drift removed). `lunarmag figures --review-labels` draws Fig. 5.2 with that wording |
| Y6-03 | "Burial known" cases use `bmax = 0`, which means no buried part at all. They don't model a known nonzero depth. | **Verified** (`REQ_c_dx0.5_b0` and `REQF_c_dx0.5_b0` store `bmax_m = 0.0`) | `a09` parts q, f, g, h | Relabel as "no buried part (b = 0)", or simulate known nonzero depths |
| Y6-04 | The skate-park mesh origin came from the unrounded scene width, so the central voxel prism spanned x, y = -2.0784 to +1.9216 m, offset by -0.078 m from the loop centre. | **Fixed in v1.1.** `build_mesh` is now symmetric about x = y = 0 on the 0.25 m lattice, and the central prism is exactly [-2, 2] m (`tests/test_skatepark_mesh.py`). `results/engine/skatepark_1cm_dr_sweep_results.json` and `results/analysis/a08_skatepark_neighbor_directions.json` were regenerated with the released seeds and N = 250; they match the rerun stored in `studies/results/y6_04_skatepark_snapped.json` exactly. The v1.0.0 mesh is kept as `build_mesh_v1_0` | `engine/skatepark_1cm_dr_sweep.py` | 21 Section 4.3 numbers move (listed in `notebook/paper_numbers.py`, `SUPERSEDED_Y6_04`); Fig. 4.3 was redrawn |
| Y6-05 | The shared prism operators use SimPEG's default Geoana backend; only the skate park uses Choclo. | **Verified** (`sim.engine == "geoana"`) | `inversion_helpers._sim_for` | Both are exact analytic prism kernels, so only the provenance sentence changes (Section 3.3.1 currently names Choclo for all) |
| Y6-06 | There are three angle conventions. SimPEG's spherical elevation is positive up (theta = -I) and in radians. The skate-park helper's declination is measured from +y. The skate-park solver works in nT-scaled model units. | **Read** | `engine/inversion_approach_comparison.py`, `engine/skatepark_1cm_dr_sweep.py` | Document the convention per solver |
| Y6-07 | 18 of the 20 spherical-voxel runs reach chi^2/N_d <= 1. Draws 2 and 7 end at 1.0267 and 1.0177. | **Verified** (stored `phi_d_over_nD`) | `results/engine/inversion_approach_comparison_results.json` | Say "20/20 Cartesian and 18/20 spherical reached the target" |
| Y6-08 | The uncertainty of the ratio of means treats the paired methods as independent. With the covariance included, 11.6 +- 1.9 becomes about +- 2.1. | Not checked | `inversion_approach_comparison.py` | |
| Y6-09 | The "inter-layer correlation" is the largest canonical correlation of the 3-D column spaces (cosine of the smallest principal angle), not the largest pairwise column correlation. | **Read** (now stated in the docstring of `operator_diagnostics`) | `inversion_helpers.operator_diagnostics` | Rename in the text |
| Y6-10 | The quoted cross-strike recovery (1.1, 3.7 and 1.5 deg) is the six-parameter solve with Mx = 0 imposed. | **Read** | `analysis/a07_three_layer_wall_extent.py` (`cross_strike_only`) | State the imposed prior |
| Y6-11 | The scripts form G^T G, which squares the condition number. The 2 km wall cases reach cond(G^T G) of about 1e16 to 1e17. | **Read** | a07, a09 | Say "effectively unresolvable at this noise". QR or SVD solves would be safer |
| Y6-12 | `ou_drift_exact` builds the drift from `make_noise`, which returns white plus OU noise. The "pure slow drift" therefore carries a small white part, which is then demeaned and normalized. | **Read** | `a09.ou_drift_exact`, `correlated_noise_sensitivity.make_noise` | Describe the process as tested, or rerun with sigma_w = 0 |
| Y6-13 | The drift rms is pooled over stations and the three axes; it is not enforced per axis. | **Read** | `a09.ou_drift_exact` | Say "pooled per-component rms" |
| Y6-17 | The susceptibility error is lognormal. Its median is unbiased, but its mean is high by exp(s^2/2), which is 1.27 for a factor-of-2 error. | **Read** | `a09.budget` (`chi * exp(N(0, s^2))`) | State the model |
| Y6-21 | Each 1 cm drive-around loop has 2399 stations with 3 duplicated corners (zero-length steps). The step is 0.0100167 m, and the loop ends at (-3, -2.98998) instead of (-3, -3). | **Verified** | `core.generate_traverse("full_orbit")` | Small effect. The text should not claim exactly four headings or uniform time steps |
| pose | `generate_pose_errors(randomize=True)` bounds each translation component separately, so the total translation can reach sqrt(3) times the bound. | **Read** (docstring corrected) | `core.generate_pose_errors` | "Range error bound" is a per-axis bound |

Two further items were fixed in this repository:

- **`paper_numbers` pairings.** Six rows of `notebook/paper_numbers.py` read a stored key that belongs to another
  quantity, although the values happened to round to the same printed number. They were corrected in v1.1, and
  `tests/test_paper_numbers.py` pins all six; the test fails on v1.0.0.
- **PDF/A.** The v1.0.0 figure PDFs had transparency and no CIDSets. The thesis figures in this repository have
  neither problem. Their CIDSets list every CID in each font's CIDToGIDMap, which passes veraPDF under both 1b and
  2b. A CIDSet that lists only the CIDs mapped to a non-zero glyph passes 1b but fails 2b (ISO 19005-2, clause
  6.2.11.4.2).

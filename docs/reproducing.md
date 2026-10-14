# Reproducing the thesis

Three different things can be meant by "reproducing", and they establish different amounts:

| Kind | Command | What it shows | What it does not show |
|---|---|---|---|
| Regenerate figures | `make figures` | Every figure is a function of the stored results: the same JSON gives the same PDF, byte for byte | That the stored results are right |
| Check against stored outputs | `make test`, `make reproduce` | The fast, seeded parts of the code, rerun with the same code and seeds, give the stored numbers bit for bit (about 16,000 numbers); the 374 numbers quoted from the companion paper are recomputed from the stored results and match the paper's sentences; the closed-form physics agrees with the SimPEG operator and with analytic limits | Independent replication. A bug shared by the stored run and the rerun would pass. "374 of 374" means the printed numbers match the stored results, not that someone else recomputed them |
| Fresh simulation rerun | `make rerun` | The 14 simulation scripts, rerun from their seeds in a scratch copy, reproduce `results/` (3.5 to 4 hours) | Not run on the thesis version. The error-budget parts behind Figure 5.2 were re-simulated on 2026-10-12 and all 3,224 compared numbers matched |

## 1. Set up (once)

```bash
conda env create -f environment.yml     # creates `lunar-mag` and installs this package in editable mode
conda activate lunar-mag
make info                               # versions, paths and the list of thesis figures
```

If the environment already exists, run `conda env update -f environment.yml` once (adds pytest and installs the
package in editable mode). Until then `make test` falls back to the base environment's pytest through a temporary
path shim and says so. You can also skip the install: every `make` target sets `PYTHONPATH=src`, and the scripts in
`engine/` and `analysis/` find the package on their own.

## 2. Check the code against the stored results (minutes)

| Command | Time | What it does |
|---|---|---|
| `make test` | about 40 s | 117 tests. Exact reruns of about 16,000 stored numbers (a06 parts A-I, a09 parts M and N, the notebook draw); analytic checks of Eqs. 3.6-3.29; the 374 quoted numbers and the six fixed key pairings; every figure's plotted values against the paper scripts; PDF/A-1b readiness; byte reproducibility; the manifest |
| `make test-all` | about 1.5 min | adds the slow tests: a09 tilt parts J and T (exact), and the notebook run with `RERUN = True` |
| `make reproduce` | about 20 s | the same checks as a single report, plus a redraw of every figure compared with the manifest |
| `make figures` | about 15 s | redraws Figs. 3.1-5.2 into `thesis/MIT-thesis-template/figures/` (PDF/A-1b ready), writes the plotted values to `figures/values/`, PNG previews to `figures/preview/`, and `figures/manifest.json` and `MANIFEST.md` |
| `lunarmag numbers 4.2` | seconds | prints the companion paper's quoted numbers for one section, recomputed |

## 3. Figures and PDF/A

`make figures` redraws Figs. 3.1 to 7.1 from the stored results into `thesis/MIT-thesis-template/figures/`. That
folder keeps the name it has in the thesis's LaTeX source, which is not part of this repository (see
[thesis/README.md](../thesis/README.md)). The command also reruns the Chapter 5 to 7 drivers
(`studies/fig_*.py`, `figures/ch6/make_ch6.py`, `ml/fig_ml.py`), copies their PDFs into that folder, makes every figure
PDF/A-1b ready, and lists all of them in `figures/MANIFEST.md`. To write the figures somewhere else, run
`python -m lunarmag figures --out DIR`.

MIT's thesis specifications ask for PDF/A-1. Two things in a pdfLaTeX thesis with matplotlib figures break
PDF/A-1b, and the figure code handles both:

1. **Transparency.** Matplotlib writes unused alpha-0 graphics states. The thesis style has no alpha, and
   `lunarmag.pdfa.no_zero_stroke_alpha` writes those unused states as 1.
2. **Missing CIDSet.** Matplotlib embeds TrueType fonts as CID fonts without a CIDSet. `lunarmag.pdfa.add_cidset`
   adds a complete one, listing every CID in the CIDToGIDMap. Complete sets pass both 1b and 2b; a set listing
   only the glyph-mapped CIDs fails 2b.

`python -m lunarmag pdfa FILE.pdf` checks any PDF for these two problems (and runs veraPDF if it is installed).

## 4. Rerun every simulation (hours)

`make rerun` copies the repository to `build/rerun/`, runs all 14 scripts in order under `nice`, logs each one's
wall time to `build/rerun/timing.txt`, and compares the regenerated JSON with `results/` file by file
(`lunarmag compare`). It never overwrites `results/`.

| Script | Feeds | Time (Apple M-series) |
|---|---|---|
| `notebook/part1_single_draw.py --write` | notebook Part 1 | < 1 min |
| `engine/inversion_approach_comparison.py` | Table 4.1 (paper Table 2) | about 1 min |
| `analysis/a02_isolated_outcrop_paper_geometry.py` | Section 4.2, Fig. 4.2 | about 15 min |
| `analysis/a02b_detection_sweep_N1000.py` | Fig. 4.1 | about 1 min |
| `analysis/a04_text_number_checks.py` | Sections 4.2, 4.4 | about 1 min |
| `analysis/a06_isolated_robustness.py` | Table 4.2, Fig. 5.1 | about 1 min |
| `engine/skatepark_1cm_dr_sweep.py` | Fig. 4.3 | about 1 min |
| `analysis/a08_skatepark_neighbor_directions.py` | Section 4.3 | about 1 min |
| `engine/three_layer_wall.py` | Table 4.3 | about 5 min |
| `analysis/a03_three_layer_direction_vs_distance.py` | Section 4.4 | about 1 min |
| `analysis/a07_three_layer_wall_extent.py` | Section 4.4 | about 2 min |
| `engine/scripts/elevated_traverse_height_sweep.py` | Fig. 4.4 | about 5 min |
| `analysis/a05_background_and_correlated_noise.py` | Sections 3.3, 5.1, 5.8 | about 5 min |
| `analysis/a09_systematics.py` | Sections 4.2, 5.1-5.3, 5.8, Table 5.1, Fig. 5.2 | about 3 h |

The total is 3.5 to 4 hours, almost all of it a09. The per-script times are the estimates in the notebook. To
rerun only part of a09, run `A09_OUT=/tmp/a09_U.json python analysis/a09_systematics.py --parts U`, then
`lunarmag compare` the file against `results/analysis/a09_systematics.json`. On macOS, give parts `G` and `g`
separate output files; the file system is case-insensitive. On 2026-10-12 a rerun of the ten a09
parts behind Fig. 5.2 matched the stored JSON: 3224 numbers, 0 differences.

## 5. What "reproduced" means here

- **Exact.** For every stored number the tests rerun, the rerun equals the stored value bit for bit, in the pinned
  environment on Apple silicon.
- **Figures.** Every plotted value equals what the companion-paper scripts plot from the same JSON (exact float
  equality). The thesis figures are a restyle, so their PDFs are new files. Their hashes are in
  `figures/manifest.json`.
- **Not covered.** Literature values, the laboratory experiment, and statements that interpret a
  number. `docs/known-issues.md` lists places where a sentence and the code disagree.

## 6. Figure by figure: script and stored result

Every figure is drawn from stored results (`results/`, `studies/results/`, `ml/results/`) and does not run a
simulation. `make figures` redraws all of them into `thesis/MIT-thesis-template/figures/`, writes the plotted values
to `figures/values/`, and records the script, the input files and the hash of each PDF in `figures/MANIFEST.md`.
Figure numbers are those of the thesis.

| Figure | Drawn by | Stored result it reads | Simulation that made the result |
|---|---|---|---|
| 1.1 The target (LROC NAC and Apollo 15 imagery) | `figures/scripts/fig_target_imagery.py` | `figures/inputs/nac_*.png`, `apollo15_*.png` (reduced public NASA/PDS images; see below) | `figures/scripts/prepare_image_inputs.py` |
| 3.1 Isolated outcrop geometry | `src/lunarmag/figures/geometry_figs.py` | scene constants | none |
| 3.2 Skate park geometry | `src/lunarmag/figures/geometry_figs.py` | scene definition in `engine/skatepark_1cm_dr_sweep.py` | none |
| 3.3 Three-layer wall geometry | `src/lunarmag/figures/geometry_figs.py` | `inversion_helpers.three_layer_mesh` | none |
| 4.1 Recovery against field strength | `src/lunarmag/figures/results_figs.py` | `results/analysis/a02b_detection_sweep_N1000.json` | `analysis/a02b_detection_sweep_N1000.py` |
| 4.2 Imaging mis-pose | `src/lunarmag/figures/results_figs.py` | `results/analysis/a02_isolated_outcrop_paper_geometry.json` | `analysis/a02_isolated_outcrop_paper_geometry.py` |
| 4.3 Skate park recovery | `src/lunarmag/figures/results_figs.py` | `results/engine/skatepark_1cm_dr_sweep_results.json` | `engine/skatepark_1cm_dr_sweep.py` |
| 4.4 Second traverse height | `src/lunarmag/figures/results_figs.py` | `results/engine/elevated_traverse_height_sweep_results.json` | `engine/scripts/elevated_traverse_height_sweep.py` |
| 5.1 Correlated noise and drift | `src/lunarmag/figures/results_figs.py` | `results/analysis/a06_isolated_robustness.json` | `analysis/a06_isolated_robustness.py` |
| 5.2 Combined error budget | `src/lunarmag/figures/results_figs.py` | `results/analysis/a09_systematics.json`, `studies/results/y6_03_known_burial.json` | `analysis/a09_systematics.py`, `studies/y6_03_known_burial.py` |
| 5.3 Sobol indices | `studies/fig_sobol.py` | `studies/results/sobol.json` | `studies/sobol.py` |
| 5.4 Flow-mean error | `studies/fig_flow_mean.py` | `studies/results/flow_mean.json` | `studies/flow_mean.py` |
| 6.1, 6.3 Mission architecture, Pareto | `figures/ch6/make_ch6.py` | `figures/ch6/ch6_data.py`, which reads the a06 and a09 results | |
| 6.2 Candidate sites on the nearside | `figures/scripts/fig_site_map.py` | `figures/inputs/lola_hillshade_nearside.png`, `site_coordinates.csv` (reduced public NASA/USGS data; see below) | `figures/scripts/prepare_image_inputs.py` |
| 7.1 Learned estimator against burial | `ml/fig_ml.py` | `ml/results/sbi_results.json` | `ml/run_all.sh` |

Figures 1.1 and 6.2 are images, not results. They read small reduced inputs that are committed in `figures/inputs/`
(a few MB). Those were made from public NASA, USGS and PDS downloads (LROC NAC EDR products, the LOLA 64 pixel/degree
elevation map, an Apollo 15 scan and the IAU Gazetteer center points) by `figures/scripts/prepare_image_inputs.py`.
The raw files are about 0.6 to 1 GB, are not in git, and live in `data/external/`; `data/external/SOURCES.md` lists
the URL, ID, size, retrieval date and credit of each. `make figures` needs only the committed inputs. Re-running the
preparation step needs the raw files: download them again from the URLs in `SOURCES.md`.

The Chapter 7 calibration figure also needs `ml/data/`, about 350 MB of simulated training and test sets that git
does not track. Without it `make figures` leaves the stored `ml/figures/ml_fig2_calib.pdf` as it is, and
`bash ml/run_all.sh` (30 minutes to an hour) regenerates the data. That figure is not printed in the thesis.

[docs/thesis-to-code.md](thesis-to-code.md) maps every table and each quoted number to its script and to the
key inside the stored JSON. `lunarmag numbers [section]` prints the 374 numbers that the thesis shares with the
companion paper, each recomputed next to its sentence.

## 7. The test suite

`make test` runs the 117 fast tests (about 40 s). `make test-all` adds the slow tests (a few minutes).

| File | Checks |
|---|---|
| `tests/test_regression.py` | Reruns seeded analysis parts and the notebook draw and requires the stored JSON exactly, about 16,000 numbers. Also checks the engine aliases and the seed derivation |
| `tests/test_physics.py` | Prism field: closed form against the SimPEG operator and volume quadrature; dipole far-field limit; tensor symmetry and trace; signs; TRM and induced conversions; angle conventions; error metrics |
| `tests/test_estimators.py` | Estimator covariance against the scatter of 20,000 Monte Carlo draws; the Schur block; gradient basis; tilt terms; the spectral band |
| `tests/test_paper_numbers.py` | 374 of 374 quoted numbers reproduce; the six wrong-key pairings of v1.0.0 stay fixed |
| `tests/test_figures.py` | Each thesis figure plots exactly the values of the companion-paper script, is PDF/A-1b ready and byte-reproducible, and matches the manifest |
| `tests/test_notebook_rerun.py` | The notebook's `RERUN` flag does what it says |
| `tests/test_skatepark_mesh.py` | The skate-park mesh fix: the central prism is exactly at x, y = +-2 m and the stored results are what the fixed engine gives |
| `tests/test_scripts_import.py` | Every simulation script still imports through the `engine/` aliases |

## 8. Troubleshooting

- **pytest missing from the `lunar-mag` environment.** `environment.yml` includes pytest. An environment created
  before that was added lacks it. Run `conda env update -f environment.yml` once. Until then `make test` prints a
  hint and borrows pytest from the base conda environment through a temporary path shim in `build/pytest-shim`;
  nothing is installed.
- **Environment not activated.** Without `conda activate lunar-mag`, the Makefile uses
  `~/miniforge3/envs/lunar-mag/bin/python` if it exists. Override with `make PYTHON=/path/to/python ...`.
- **Last-digit differences on other platforms.** The stored results were produced on Apple silicon. On Linux or Intel,
  set `LUNARMAG_RTOL=1e-3` for the tests (the CI workflow does). On Linux, a few Monte Carlo summary statistics (tail percentiles and minima of a few hundred draws) differ from the stored values by up to about 1 part in 10,000, because floating-point results differ slightly across processors; every number the thesis quotes agrees.
- **`ml/data` missing.** The Chapter 7 calibration figure needs about 350 MB of simulated data that git does not
  track. Without it `make figures` leaves the stored `ml/figures/ml_fig2_calib.pdf` as it is and prints a note.
  `bash ml/run_all.sh` regenerates the data (30 minutes to an hour). That figure is not printed in the thesis. The
  manifest then lacks the three `ml/data` input hashes for it.

# studies/

Targeted studies added after the v1.0.0 code release: reruns that address four findings (Y6-01 to Y6-04) of a
physics review of Chapters 3 to 5 that was done by an AI language model and checked by a second one; the findings
and their provenance are in `docs/known-issues.md`. The folder also holds a flow-mean simulation, a Sobol variance attribution of the combined error budget, a regolith study and a Fisher information
check. They import `engine/` and `analysis/` read-only. Where a study needs different behaviour, it uses a
modified copy kept in this folder, and the copy's header says what changed and why. Nothing in `engine/` or
`analysis/` is edited.

| Script | Question | Result file | Run time |
|---|---|---|---|
| `y6_01_spectral.py` | Wavenumber split of the estimator's noise sensitivity, with the one-sided FFT bins counted twice | `results/y6_01_spectral.json` | seconds |
| `y6_02_bias_decomposition.py` | For every row of the budget table: mean noise-free angular error (the table's "bias") vs the angle of the mean recovered vector | `results/y6_02_bias_decomposition.json` | ~4 min |
| `y6_03_known_burial.py` | "Burial known" rows rerun with known nonzero depths, b ~ U(0, 1 m) and U(0, 0.5 m), in the inversion operator | `results/y6_03_known_burial.json` | ~4 min |
| `y6_04_skatepark_snapped.py` | Skate park and a08 rerun with the mesh origin snapped so that the central prism is exactly +-2 m | `results/y6_04_skatepark_snapped.json` | ~1 min |
| `flow_mean.py` | Ten outcrops per flow, independent per-outcrop errors plus a shared frame rotation: flow-mean error and alpha95 coverage vs the Fisher-independence prediction | `results/flow_mean.json` | ~25 min |
| `flow_mean_bias_source.py` | Which source leaves a flow-shared bias when the loop offset is random (one source at a time, 2000 outcrops) | `results/flow_mean_bias_source.json` | ~3 min |
| `drift_reversal_check.py` | Drift-only vs white-only error, one-way vs each loop driven both ways, at a fixed drift rms (Y6-14) | `results/drift_reversal_check.json` | ~1 min |
| `regolith/y10_regolith.py` | Direction error caused by magnetization induced in the regolith around an outcrop, and the fits that remove it (see `regolith/README.md`) | `regolith/y10_results.json` | ~2 min |
| `fisher/fisher_check.py` | Schur complement and parallel-axis identities for the Fisher information of thesis Section 7.2, and what a second loop restores | `fisher/fisher_check.json` | seconds |
| `sobol.py` | First-order and total Sobol indices of the noise-free direction error, per error source, requirement-set tolerances | `results/sobol.json` (+ raw outputs `sobol_*_outputs.npy`) | ~30 min |
| `fig_flow_mean.py`, `fig_sobol.py` | Figures | `figures/*.pdf`, `figures/*.png` | seconds |

Shared modules:
- `_common.py`: path setup, seeds, statistics (mean with standard error, bootstrap standard errors), provenance block.
- `_budget.py`: a modified copy of `analysis/a09_systematics.py::budget` that keeps the per-draw vectors, can put a
  known nonzero burial depth into the operator, and evaluates one outcrop from explicit inputs (for the flow and
  Sobol studies). With the released seeds it reproduces every stored Table 5.2 mean to better than 1e-7 degrees.
- `_fastprism.py`: the same Geoana prism kernel that SimPEG uses, vectorized over receivers. It agrees with the
  released operator to about 2e-16 (relative), and it is about 200 times faster.

## Running

```bash
conda activate lunar-mag
nice python studies/y6_01_spectral.py
nice python studies/y6_04_skatepark_snapped.py
nice python studies/y6_02_bias_decomposition.py
nice python studies/y6_03_known_burial.py
nice python studies/flow_mean.py --flows 500
nice python studies/flow_mean_bias_source.py
nice python studies/drift_reversal_check.py
nice python studies/sobol.py --log2n 12 --workers 4
python studies/fig_flow_mean.py && python studies/fig_sobol.py && python studies/fig_sobol.py --set mitigated_set
python studies/fisher/fisher_check.py
```

Each script writes into `studies/results/`, which overwrites the stored file. To pin the imported engine and analysis
code to an exact commit, extract it with `git archive <commit> engine analysis results | tar -x -C <dir>` and set
`STUDY_CODE_ROOT=<dir>`; the result files record which code root and file hashes were used.

## Seeds and reproducibility

Every random stream comes from the repository's `seed_for(label)` (SHA-256 of `ACTIVE_SEED:label`). New runs use
labels that start with `study:`. Three scripts reuse released labels on purpose, and each JSON says so:
`y6_02` (the same draws as the stored table, so the stored means are reproduced), `y6_04` (the released skate-park
noise seeds, so the old-to-new change isolates the mesh fix; a fresh-label replicate is stored next to it), and the
reproduction checks. Each result file has a `meta` block with the repository HEAD, the code root, the SHA-256 of
each imported engine/analysis file, the library versions, the draw counts and the wall time.

The results were produced on 2026-10-13 with `STUDY_CODE_ROOT` set to a `git archive` of release v1.0.0 plus the
RERUN fix. The `repo_head` values in the result files are commits of the working history that this repository's
single commit replaces, so they do not exist here; `code_root` is a placeholder for the scratch copy that was used;
`code_sha256` is what identifies the code.

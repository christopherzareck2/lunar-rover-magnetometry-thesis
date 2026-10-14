# Changes from the companion paper's code release v1.0.0

The base of this repository is release v1.0.0 (commit `3498ead`, `christopherzareck2/lunar-bedrock-magnetometry`)
plus a fix to the notebook's `RERUN` flag. Everything else was added or changed for the thesis. Stored results are
byte for byte those of the release, except the two skate-park files.

## Corrections made in v1.1

- **Skate-park mesh (finding Y6-04).** The mesh origin came from the unrounded scene width, so the central prism
  spanned x, y = -2.0784 to +1.9216 m instead of +-2 m. `build_mesh` is now symmetric about the loop centre. The
  skate-park and a08 results were rerun and replace the v1.0.0 files; `tests/test_skatepark_mesh.py` pins the fix.
  `notebook/paper_numbers.py` lists the 21 Section 4.3 numbers that move (`SUPERSEDED_Y6_04`), each with the paper's
  value and the corrected one.
- **Six wrong-key pairings in `notebook/paper_numbers.py`.** In v1.0.0 six rows paired a sentence with a stored value
  that rounds to the same printed number but belongs to another quantity. Two now read the right key and four moved
  to the sentence their value belongs to. `tests/test_paper_numbers.py` pins all six. The count of quoted numbers
  stays 374 of 374.
- **Rerun studies.** Four studies in `studies/` rerun parts of the analysis to address findings Y6-01 to Y6-04
  (wavenumber normalization, what the "bias" column measures, "burial known" with nonzero depths, the mesh).

## Where the Y6 findings come from

The `Y6-nn` findings are from a physics and mathematics review of thesis Chapters 3 to 5 on 2026-10-13. The review
was done by an AI language model, not by a human physicist. A second AI model then checked each finding against the
code and the stored results and sorted them into the statuses used in [known-issues.md](known-issues.md). No human
reviewer is recorded for either step. The author decides which findings change the thesis text.

## What was added for the thesis

`studies/` (Y6 reruns, flow-mean simulation, Sobol attribution, regolith study, Fisher check), `ml/` (the learned
estimator), `src/lunarmag/` (the package, reference implementations of the equations and the figure code), `tests/`,
the thesis figures (`figures/`, `thesis/MIT-thesis-template/figures/`) and `docs/`.

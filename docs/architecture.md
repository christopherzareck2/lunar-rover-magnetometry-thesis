# Architecture

## Data flow

```
 src/lunarmag/ (engine)         engine/*.py, analysis/a0*.py          results/*.json
 forward operators, seeds,  ->  scenario scripts (seeded Monte   ->   stored results
 inversions, noise models       Carlo; hours in total)                (the thesis numbers)
                                                                           |
                       +---------------------------------------------------+------------------------+
                       v                                                   v                        v
     src/lunarmag/figures (thesis figures)          notebook/paper_numbers.py       tests/ (exact reruns of
     -> thesis/MIT-thesis-template/figures/*.pdf    (374 quoted numbers of the      the fast parts; analytic
     -> figures/manifest.json, values/, preview/    companion paper)                checks)
```

Nothing downstream of `results/` simulates anything. Figures and quoted numbers are pure functions of the stored
JSON, so they regenerate in seconds. The simulations reproduce the JSON from seeds, which takes 3.5 to 4 hours
for all of them (`make rerun`) and about 20 seconds for the parts the tests rerun.

## Layers

| Layer | What it is | Rule |
|---|---|---|
| `src/lunarmag/core.py`, `inversion_helpers.py`, `isolated_outcrop_1cm.py`, `correlated_noise_sensitivity.py` | The simulation engine of release v1.0.0, moved into a package, with type hints and docstrings added | Numerics frozen. `tests/test_regression.py` reruns about 16,000 stored numbers and requires every bit to match |
| `engine/<same names>.py` | Compatibility aliases. Each one replaces itself in `sys.modules` with the package module | Old scripts run unchanged: `import inversion_helpers` gives the very module object `lunarmag.inversion_helpers`, so module-level state such as `ACTIVE_SEED` is shared |
| `engine/*.py` (other), `engine/scripts/`, `analysis/a02`-`a09` | The scenario scripts that wrote `results/` | Frozen for provenance. They are the record of how each stored number was made. Changes are new reruns, not edits |
| `src/lunarmag/physics.py`, `estimators.py`, `spectral.py` | Documented, typed reference implementations of the thesis equations (see `theory.md`) | Where a function also exists in a frozen script, a test requires bit-identical output |
| `src/lunarmag/plotting.py`, `figures/` | The thesis figure style and one data/draw pair per figure | Values must equal the companion-paper scripts' values (`tests/test_figures.py`) |
| `src/lunarmag/pdfa.py`, `texbuild.py` | PDF/A-1b helpers for the figures and a LaTeX source converter | Changes no pixel |
| `src/lunarmag/manifest.py`, `reproduce.py`, `cli.py` | The manifest, fast reproduction and the `lunarmag` command | |
| `figures/scripts/` | The companion paper's figure scripts (Figures 1-8, `paperstyle.py`) | Kept unchanged: they reproduce the paper's figures in `figures/output/`, and the tests compare against them |
| `notebook/` | The paper's reproduction notebook and `paper_numbers.py` | Six key pairings fixed in v1.1 (see `known-issues.md`) |

## Reproducibility mechanics

- **Seeds.** `inversion_helpers.seed_for(label)` returns `int(sha256(f"{ACTIVE_SEED}:{label}")[:8], 16)`, with
  `ACTIVE_SEED = 20260624`. Every experiment draws from its own named stream, so results don't depend on run order
  or on which parts run together. The one exception is part1_single_draw, which uses seed 0 as the notebook does.
- **Environment.** Python 3.10, numpy 2.2, scipy 1.15, matplotlib 3.10, simpeg 0.24.0, discretize 0.11.3 and choclo
  0.3.2 (`environment.yml`). On Apple silicon with this environment the reruns match every bit. On other platforms
  last-digit differences are possible; set `LUNARMAG_RTOL=1e-3`, as CI does.
- **Figures.** These use only matplotlib's bundled fonts (Computer Modern `cmr10`/`cmb10`, with STIX filling a few
  missing glyphs) and contain no dates or software versions, so a figure PDF is byte-identical from run to run
  (tested). `figures/manifest.json` records the sha256 of each PDF, of its plotted values and of every stored
  result it reads.

## Why the frozen scripts were not refactored

The thesis quotes numbers that these scripts produced. Refactoring them, even with identical arithmetic intent,
risks a silent change in floating-point order, and with it in the stored numbers. The package therefore documents
and tests the frozen code rather than replacing it. The reference implementations in `physics.py` and
`estimators.py` are the readable versions, and the tests tie them to the frozen arithmetic.

#!/usr/bin/env bash
# Full amortized-inference study from scratch (about 1 h on a 14-core laptop under nice).
# Run from the repository root with the lunar-mag environment active:  bash ml/run_all.sh
set -euo pipefail
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python ml/nn.py                          # finite-difference gradient check of the network
python ml/validate_forward.py            # closed-form prism kernel vs SimPEG
python ml/replicate_thesis.py            # simulator vs stored a09 results
python ml/check_compression.py         # burial/pose signal outside the feature span
python ml/generate.py --workers 6        # simulated training, validation and test sets -> ml/data/
export OMP_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3
for s in 0 1 2 3; do python ml/train.py --set full --seed $s & done; wait
for s in 0 1; do python ml/train.py --set none --seed $s --epochs 25 & done; wait
python ml/evaluate.py                    # -> ml/results/sbi_results.json
python ml/fig_ml.py                      # -> ml/figures/

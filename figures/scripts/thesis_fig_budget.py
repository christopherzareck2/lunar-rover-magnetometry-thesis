"""Thesis Figure 5.2 - combined error budget (Table 5.1 as bars), from results/analysis/a09_systematics.json.

This was the one thesis figure without a script in the v1.0.0 release. The figure is drawn by
``lunarmag.figures.results_figs`` (``data_budget`` and ``draw_budget``) in the thesis style. This file is the
standalone entry point, kept next to the companion paper's figure scripts.

    python figures/scripts/thesis_fig_budget.py                  # writes thesis/MIT-thesis-template/figures/thesis_fig_budget.pdf
    python figures/scripts/thesis_fig_budget.py --out DIR        # anywhere else
    python figures/scripts/thesis_fig_budget.py --review-labels  # wording of the 2026-10-13 physics review

For each budget case the bar spans the min-max over magnetization directions of all_sources/dir_err_deg/mean,
and the dark bar spans the same for noise_free_bias/dir_err_deg/mean (plotted values: figures/values/).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from lunarmag.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["figures", "--only", "5.2", *sys.argv[1:]]))

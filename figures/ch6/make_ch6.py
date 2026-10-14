"""Regenerate every Chapter 6 figure (thesis/MIT-thesis-template/figures/ch6_*.pdf) from stored results and the
stated assumptions in ch6_data.py. Run from any directory:

    python figures/ch6/make_ch6.py

Each PDF is saved without timestamps, checked for transparency and given CIDSets for PDF/A-1b
(lunarmag.pdfa.make_pdfa1b_ready). 300 dpi PNG previews go to figures/ch6/png/.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ch6_architecture  # noqa: E402
import ch6_boom  # noqa: E402
import ch6_conops  # noqa: E402
import ch6_pareto  # noqa: E402
import ch6_surface_time  # noqa: E402

if __name__ == "__main__":
    for mod in (ch6_architecture, ch6_conops, ch6_surface_time, ch6_pareto, ch6_boom):
        mod.main()

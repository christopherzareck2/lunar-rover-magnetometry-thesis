"""Shared helpers for the Chapter 6 figures: thesis style, save, and the PDF/A-1b check."""
from __future__ import annotations

import ch6_data as D  # noqa: F401  (puts <repo>/src on sys.path)
from lunarmag import plotting as st
from lunarmag.pdfa import make_pdfa1b_ready


def finish(fig, name: str) -> None:
    """Lint the text for cmr10 glyph problems, save PDF (+PNG preview), and make the PDF PDF/A-1b ready."""
    st.lint_text(fig)
    paths = st.save(fig, D.OUT_DIR, name, png_dir=D.PNG_DIR)
    rep = make_pdfa1b_ready(paths[0])
    print(f"{name}: {paths[0]} {rep}")

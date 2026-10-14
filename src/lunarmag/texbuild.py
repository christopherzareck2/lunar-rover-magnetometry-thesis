"""Helpers for the thesis LaTeX build (a standalone helper; no make target uses it).

``pdfa1b_variant(tex)`` edits a COPY of a thesis source that declares PDF/A-2b so that it builds as PDF/A-1b, the
standard MIT's thesis specifications name. (The final thesis source already declares PDF/A-1b and carries both
changes.) The changes are:

1. in ``\\DocumentMetadata``: ``pdfversion = 1.4`` and ``pdfstandard = a-1b``;
2. before ``\\begin{document}``: redefine ``\\Gread@pdf`` so pdfTeX's graphics driver does not wrap every included
   PDF in a transparency group (PDF/A-1 forbids transparency groups, ISO 19005-1 clause 6.4).

The result of both changes passed veraPDF 1.30.0 on 2026-10-13. The function never modifies the source file.

    python -m lunarmag.texbuild pdfa1b path/to/copy/MIT-Thesis.tex
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

GREAD_PATCH = (
    "% PDF/A-1b: stop pdfTeX's graphics driver adding a transparency group to every included PDF\n"
    "\\makeatletter\n"
    "\\def\\Gread@pdf#1{\\let\\Gin@decode\\@empty\\Gin@interpolatefalse\\Gread@transgroupfalse\\Gread@pdftex{#1}}\n"
    "\\makeatother\n"
)


def pdfa1b_variant(tex_path: Path | str) -> None:
    p = Path(tex_path)
    s = p.read_text()
    s, n1 = re.subn(r"(?m)^(\s*)pdfversion\s*=\s*[0-9.]+\s*,", r"\1pdfversion  = 1.4,", s, count=1)
    s, n2 = re.subn(r"(?m)^(\s*)pdfstandard\s*=\s*[^,\n]+,", r"\1pdfstandard = a-1b,", s, count=1)
    if n1 != 1 or n2 != 1:
        raise SystemExit(f"{p}: could not find active pdfversion/pdfstandard lines in \\DocumentMetadata")
    if "\\Gread@transgroupfalse" not in s:
        s, n3 = re.subn(r"(?m)^\\begin\{document\}", lambda m: GREAD_PATCH + m.group(0), s, count=1)
        if n3 != 1:
            raise SystemExit(f"{p}: no \\begin{{document}} line")
    p.write_text(s)
    print(f"{p}: PDF/A-1b variant (pdfversion 1.4, pdfstandard a-1b, no transparency groups)")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "pdfa1b":
        raise SystemExit(__doc__)
    pdfa1b_variant(sys.argv[2])

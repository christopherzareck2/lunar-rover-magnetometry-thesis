"""``lunarmag`` command line: figures, fast reproduction, quoted numbers, PDF/A checks, environment info.

    lunarmag figures [--only 4.1 5.2] [--out DIR] [--no-pdfa] [--review-labels]
    lunarmag reproduce [--no-figures]          fast check from stored results (make reproduce)
    lunarmag compare NEW_RESULTS [REF]         compare a full rerun with the stored results (make rerun)
    lunarmag numbers [SECTION]                 the companion paper's quoted numbers, recomputed
    lunarmag pdfa FILE.pdf ...                 transparency and CIDSet check (+ veraPDF if installed)
    lunarmag info                              versions, paths and the figure list
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _figures(a) -> int:
    from . import figures as F, manifest
    from .paths import THESIS_FIGURES
    specs = [F.BY_NUMBER[n] for n in a.only] if a.only else F.FIGURES
    out = Path(a.out) if a.out else THESIS_FIGURES
    records = []
    for spec in specs:
        kw = {"review_labels": True} if (a.review_labels and spec.number == "5.2") else {}
        extra = {"png_dir": out, "values_dir": out} if a.out else {}      # never touch the canonical files
        rec = F.build(spec, out_dir=out, pdfa=not a.no_pdfa, **extra, **kw)
        records.append(rec)
        print(f"Fig. {spec.number:4s} {rec['file']}  sha256 {rec['pdf_sha256'][:12]}"
              + ("  PDF/A-1b ready" if not a.no_pdfa else ""))
    if not a.only and not a.out and not a.no_pdfa and not a.review_labels:
        from .figures import external as X
        if not a.skip_external:
            X.run_external()
        for r in X.postprocess_dir(THESIS_FIGURES):
            if r["cidsets_written"]:
                print(f"PDF/A pass: {r['file']}: CIDSet written for {r['cidsets_written']} font(s)")
        j, m = manifest.write(records, X.records())
        print(f"wrote {j.relative_to(F.ROOT)} and {m.relative_to(F.ROOT)}")
    return 0


def _pdfa_figures(a) -> int:
    """PDF/A-1b pass over every PDF in the thesis figures folder (a figure copied in by hand cannot break a PDF/A build)."""
    from .figures import external as X
    for r in X.postprocess_dir():
        print(f"{r['file']}: " + (f"CIDSet written for {r['cidsets_written']} font(s)" if r["cidsets_written"]
                                  else "PDF/A-1b ready"))
    return 0


def _reproduce(a) -> int:
    from . import reproduce
    return 0 if reproduce.fast(figures=not a.no_figures) else 1


def _compare(a) -> int:
    from . import reproduce
    return 0 if reproduce.compare(a.new, a.ref, rtol=a.rtol) else 1


def _numbers(a) -> int:
    from .paths import ROOT
    sys.path.insert(0, str(ROOT / "notebook"))
    import paper_numbers as P
    P.numbers(a.section)
    m, n = P.count()
    print(f"\n{m} of {n} numbers match")
    return 0 if m == n else 1


def _pdfa(a) -> int:
    from . import pdfa
    bad = 0
    vera = None if a.no_verapdf else shutil.which("verapdf")
    for f in a.files:
        tr, miss = pdfa.scan_transparency(f), pdfa.fonts_missing_cidset(f)
        line = f"{f}: transparency {tr or 'none'}; fonts without CIDSet {miss}"
        if vera:
            r = subprocess.run([vera, "--flavour", a.flavour, f], capture_output=True, text=True)
            comp = 'isCompliant="true"' in r.stdout
            line += f"; veraPDF {a.flavour}: {'PASS' if comp else 'FAIL'}"
            bad += not comp
        bad += (bool(tr) if a.flavour.startswith("1") else 0) + bool(miss)   # PDF/A-2 allows transparency
        print(line)
    return 1 if bad else 0


def _info(a) -> int:
    from . import figures as F, manifest
    from .paths import ROOT, THESIS_FIGURES
    for k, v in manifest.environment().items():
        print(f"{k:12s} {v}")
    print(f"{'repository':12s} {ROOT}\n{'figures to':12s} {THESIS_FIGURES}\n")
    for s in F.FIGURES:
        print(f"Fig. {s.number:4s} {s.stem:34s} {s.title}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lunarmag", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("figures", help="draw the thesis figures from the stored results")
    f.add_argument("--only", nargs="*", metavar="N", help="figure numbers, e.g. 4.1 5.2")
    f.add_argument("--out", help="output folder (default: thesis/MIT-thesis-template/figures)")
    f.add_argument("--no-pdfa", action="store_true", help="skip the PDF/A-1b post-processing (CIDSet)")
    f.add_argument("--review-labels", action="store_true",
                   help="Fig. 5.2 with the wording of the 2026-10-13 physics review (b = 0, 'noise-free mean error')")
    f.add_argument("--skip-external", action="store_true",
                   help="do not rerun the Chapter 5-7 drivers (studies/, figures/ch6/, ml/); still post-process them")
    f.set_defaults(fn=_figures)
    pf = sub.add_parser("pdfa-figures", help="make every PDF in the thesis figures folder PDF/A-1b ready")
    pf.set_defaults(fn=_pdfa_figures)
    r = sub.add_parser("reproduce", help="fast reproduction check from stored results")
    r.add_argument("--no-figures", action="store_true")
    r.set_defaults(fn=_reproduce)
    c = sub.add_parser("compare", help="compare rerun results with the stored results")
    c.add_argument("new"); c.add_argument("ref", nargs="?", default=None)
    c.add_argument("--rtol", type=float, default=0.0)
    c.set_defaults(fn=_compare)
    n = sub.add_parser("numbers", help="print the companion paper's quoted numbers, recomputed")
    n.add_argument("section", nargs="?", default=None)
    n.set_defaults(fn=_numbers)
    d = sub.add_parser("pdfa", help="check figure or thesis PDFs for PDF/A-1b problems")
    d.add_argument("files", nargs="+"); d.add_argument("--flavour", default="1b")
    d.add_argument("--no-verapdf", action="store_true",
                   help="only the transparency and CIDSet scan (for figure files, which are not PDF/A documents)")
    d.set_defaults(fn=_pdfa)
    i = sub.add_parser("info", help="versions, paths and the figure list")
    i.set_defaults(fn=_info)
    a = p.parse_args(argv)
    if getattr(a, "cmd", None) == "compare" and a.ref is None:
        from .paths import RESULTS
        a.ref = RESULTS
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

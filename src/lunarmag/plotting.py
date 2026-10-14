"""One publication style for every thesis figure.

* Sized for the MIT thesis text block (letter paper, 1 in margins: 6.5 in wide). Each figure is drawn at the width
  at which the thesis includes it (``WIDTH``), so 9 pt text in the figure prints at 9 pt; the canvas is saved
  without tight cropping so the size is exact.
* Computer Modern (matplotlib's bundled ``cmr10`` and mathtext ``cm``) to match the thesis body font, and
  because the bundled fonts make the PDFs byte-reproducible on any machine (no system fonts involved).
* Colour-blind-safe Okabe-Ito colours for categorical series, checked with the dataviz palette validator
  (vermillion/blue pass every CVD check; the three flow colours of Fig. 4.4 add markers and line styles as the
  secondary encoding the validator asks for). Magnetization maps use viridis (perceptually uniform, CVD-safe).
* No transparency anywhere (PDF/A-1 forbids it): opaque legend frames, pre-blended greys, and the PDF backend's
  unused stroke alpha 0 is written as 1 (``lunarmag.pdfa``).
* Semantics kept from the companion paper: vermillion circles = direction error, blue squares = intensity error,
  solid / dashed for the two cases of a comparison.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text  # noqa: E402,F401
from matplotlib import colors as mcolors  # noqa: E402

# ------------------------------------------------------------------------------------------------ geometry
TEXT_WIDTH_IN = 6.5                 #: \textwidth of mitthesis (8.5 in letter, 1 in margins)
WIDTH = {"full": TEXT_WIDTH_IN, "0.9": 0.9 * TEXT_WIDTH_IN, "0.7": 0.7 * TEXT_WIDTH_IN}

# ------------------------------------------------------------------------------------------------ colours
OKABE_ITO = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
             "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
DIRECTION = OKABE_ITO["vermillion"]     #: direction error (the paper's orange)
INTENSITY = OKABE_ITO["blue"]           #: intensity error
FLOWS = (OKABE_ITO["vermillion"], OKABE_ITO["green"], OKABE_ITO["purple"])
INK = "#1A1A1A"                         #: text and axes
MUTED = "#6E6E6E"                       #: reference lines, secondary text
GUIDE = "#9A9A9A"                       #: dashed goal / reference lines
GRID = "#E3E3E3"                        #: recessive grid
SHADE = "#E8E8E8"                       #: shaded regions (opaque, so no alpha is needed)
VIRIDIS = matplotlib.colormaps["viridis"]
NORM_M = mcolors.Normalize(0.0, 2.4)    #: magnetization colour scale, units of 1e-3 A/m
BG = VIRIDIS(0.0)                       #: zero magnetization
SRC = VIRIDIS(1.0)                      #: magnetization M_ref
TRAVERSE = OKABE_ITO["orange"]          #: rover paths in the geometry figures

# ------------------------------------------------------------------------------------------------ type
BASE_PT = 9.0
BOLD_FAMILY = ["cmb10", "STIXGeneral"]  #: Computer Modern bold (cmr10 has no bold weight)
RC = {
    "font.family": ["cmr10", "STIXGeneral"],    # STIX only fills glyphs cmr10 lacks (en dash, Delta)
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,     # cmr10 has no Unicode minus: route tick labels through mathtext
    "axes.unicode_minus": True,
    "font.size": BASE_PT,
    "axes.labelsize": BASE_PT,
    "axes.titlesize": BASE_PT,
    "xtick.labelsize": BASE_PT - 1,
    "ytick.labelsize": BASE_PT - 1,
    "legend.fontsize": BASE_PT - 1,
    "legend.frameon": False,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "#BBBBBB",
    "legend.handlelength": 2.2,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 4.0,
    "patch.linewidth": 0.6,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,                      # TrueType embedded (CIDSet added afterwards for PDF/A-1b)
    "ps.fonttype": 42,
    "pdf.compression": 6,
    "svg.fonttype": "none",
    "figure.dpi": 100,
}


@contextlib.contextmanager
def style() -> Iterator[None]:
    """Context manager applying the thesis style (restores the previous rcParams afterwards)."""
    with plt.rc_context(RC):
        yield


#: Characters that matplotlib's cmr10 maps to other glyphs (TeX OT1 encoding): outside $...$ they print wrong
#: (a degree sign becomes "fl", "|" an em dash). ``lint_text`` rejects them; write $^{\circ}$ and $|M|$ instead.
CMR10_UNSAFE = set('|<>{}_"\\') | {chr(c) for c in range(0xA0, 0xC5)}   # incl. degree, plus-minus, micro


def lint_text(fig: plt.Figure) -> None:
    """Raise if any text outside math mode uses a character that cmr10 renders as a different glyph."""
    import re
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        plain = re.sub(r"(?<!\\)\$.*?(?<!\\)\$", "", s)
        bad = set(plain) & CMR10_UNSAFE
        if bad:
            raise ValueError(f"text {s!r} uses {sorted(bad)} outside math mode; cmr10 would print other glyphs")


def figure(width: str | float, height: float, **kw) -> plt.Figure:
    """A figure of exact physical size; ``width`` is a key of ``WIDTH`` or inches."""
    w = WIDTH[width] if isinstance(width, str) else float(width)
    return plt.figure(figsize=(w, height), **kw)


def panel_label(ax, letter: str, x: float = -0.02, y: float = 1.02, dark: bool = False, **kw) -> None:
    """Bold panel letter (A, B, ...) at the top-left corner of an axes, in axes coordinates."""
    ax.text(x, y, letter, transform=ax.transAxes, ha="right" if x <= 0 else "left", va="bottom" if y >= 1 else "top",
            fontsize=BASE_PT + 1, family=BOLD_FAMILY, color="white" if dark else INK, zorder=30, **kw)


def goal_line(ax, value: float = 5.0, label: str | None = r"$5^{\circ}$ goal", x_text: float | None = None,
              orientation: str = "h") -> None:
    """Dashed grey 5-degree goal line with an optional small label."""
    if orientation == "h":
        ax.axhline(value, color=GUIDE, ls=(0, (4, 2.5)), lw=0.9, zorder=1)
        if label and x_text is not None:
            ax.text(x_text, value, label, ha="left", va="bottom", fontsize=BASE_PT - 1, color=MUTED)
    else:
        ax.axvline(value, color=GUIDE, ls=(0, (4, 2.5)), lw=0.9, zorder=1)


def save(fig: plt.Figure, out_dir: Path | str, name: str, png_dir: Path | str | None = None) -> list[Path]:
    """Save ``name``.pdf with no dates or software versions (byte-reproducible), at the exact figure size (no
    tight cropping) and with no PDF stroke alpha of 0 (see ``lunarmag.pdfa``); optionally a 300 dpi PNG preview
    in ``png_dir``. Returns [pdf] or [pdf, png]."""
    from .pdfa import no_zero_stroke_alpha
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with no_zero_stroke_alpha():
        pdf = out_dir / f"{name}.pdf"
        fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None, "Creator": None, "Producer": None})
    paths = [pdf]
    if png_dir is not None:
        os.makedirs(png_dir, exist_ok=True)
        png = Path(png_dir) / f"{name}.png"
        fig.savefig(png, metadata={"Software": None})
        paths.append(png)
    plt.close(fig)
    return paths

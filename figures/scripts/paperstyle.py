"""Shared style for the paper figures.

  * Helvetica, italic axis symbols, heavy black frame.
  * Geometry figures (1-3): viridis magnetization scale 0-2.4 x 1e-3 A/m in a colorbar on top, ground at the
    zero colour, sources at the top colour, x / o / arrows for magnetization into / out of the page / in the
    plane, traverses in orange, start/end as a black dot with a white edge.
  * Result figures (4-8): orange circles = direction error (left axis), blue squares = intensity error
    (right axis), solid/dashed for the two surveys.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "figures", "output")
RES_ANALYSIS = os.path.join(ROOT, "results", "analysis")
RES_ENGINE = os.path.join(ROOT, "results", "engine")

VIRIDIS = matplotlib.colormaps["viridis"]
NORM_M = mcolors.Normalize(0.0, 2.4)
BG = VIRIDIS(0.0)            # zero magnetization / subsurface
SRC = VIRIDIS(1.0)           # source at M_ref
TRAV_DARK = "#D9762B"        # traverse in legends on white
ORANGE = "#F28E2B"           # direction error
BLUE = "#3B75AF"             # intensity error
GREY = "#BDBDBD"

plt.rcParams.update({
    "font.family": "Arial",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "font.size": 15,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "axes.linewidth": 2.0,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#CCCCCC",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def panel_letter(ax, letter, dark=True, x=0.03, y=0.95, size=30):
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="top", fontsize=size,
            fontweight="bold", color="white" if dark else "black", zorder=20)


def m_colorbar(fig, rect, label_right=True, label=r"$M_{\rm NRM}$ ($\times10^{-3}$ A m$^{-1}$)"):
    """Horizontal magnetization colorbar in the paper's style. rect = [left, bottom, width, height]."""
    cax = fig.add_axes(rect)
    cb = fig.colorbar(cm.ScalarMappable(norm=NORM_M, cmap=VIRIDIS), cax=cax, orientation="horizontal",
                      ticks=[0, 0.6, 1.2, 1.8, 2.4])
    cb.ax.set_xticklabels(["0", "0.6", "1.2", "1.8", "2.4"])
    cb.outline.set_linewidth(1.6)
    if label_right:
        cax.text(1.04, 0.5, label, transform=cax.transAxes, ha="left", va="center", fontsize=17)
    else:
        cb.set_label(label, fontsize=17, labelpad=6)
        cb.ax.xaxis.set_label_position("top")
    return cb


def mark_into(ax, x, y, s=260, color="black", lw=2.2):
    ax.scatter(x, y, marker="x", s=s, c=color, linewidths=lw, zorder=6)


def mark_out(ax, x, y, s=260, color="black", lw=2.2):
    ax.scatter(x, y, marker="o", s=s, facecolors="none", edgecolors=color, linewidths=lw, zorder=6)


def arrow(ax, x, y, dx, dy, color="black", lw=2.0, head=0.18, ms=16):
    ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y), zorder=7,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=ms))


def start_dot(ax, x, y, s=90):
    ax.scatter([x], [y], s=s, c="black", edgecolors="white", linewidths=1.5, zorder=9)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        # no dates or software versions in the files, so rerunning a figure script gives identical bytes
        meta = {"CreationDate": None, "Creator": None, "Producer": None} if ext == "pdf" else {"Software": None}
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), metadata=meta)
    plt.close(fig)
    print(f"wrote figures/output/{name}.pdf/.png")


# ---- shared layout for the three scenario-geometry figures (Figures 1-3) ----
GEO_W = 13.0                   # figure width (in); all three are placed full width in the paper
GEO_TRAV_LW = 3.0              # traverse line width, same on dark plan views and white sections
# one style key for all three figures: solid = primary path, dashed = secondary path at another
# standoff or height, dash-dot = drive-by alternative, thin grey dotted = sensor-height guide in sections
LS_LOOP, LS_LOOP2, LS_BY, LS_ELEV = "-", (0, (5, 3)), (0, (6, 2, 1.5, 2)), (0, (5, 3))


def ground(ax, x0, x1, depth):
    """Non-magnetic ground below z = 0 in a vertical section."""
    from matplotlib.patches import Rectangle
    ax.set_facecolor("white")
    ax.add_patch(Rectangle((x0, -depth), x1 - x0, depth, color=BG, zorder=1))


def trav(ax, xs, ys, ls="-", label=None, zorder=6):
    ax.plot(xs, ys, color=TRAV_DARK, lw=GEO_TRAV_LW, ls=ls, zorder=zorder, label=label,
            solid_capstyle="butt", dash_capstyle="butt")


def start_legend_handle(ax):
    ax.scatter([], [], s=90, c="black", edgecolors="white", linewidths=1.5, label="Start/end")


def guide(ax, xs, z):
    """Thin grey dotted line at the sensor height in a vertical section (not a survey path)."""
    ax.plot(xs, [z, z], ls=(0, (1, 2)), color="#8C8C8C", lw=1.3, zorder=4)


def triad(ax, x0, y0, L, labels=("x", "y"), color="white", out_of_page=None, fs=14, Lz=None):
    """Small coordinate triad in data units: arrows along +first and +second axis; optional
    out-of-page axis label drawn as a dot in a circle at the origin."""
    Lz = L if Lz is None else Lz   # vertical arrow length in data units (shorter when the panel is exaggerated)
    for (dx, dy), lab in (((L, 0), labels[0]), ((0, Lz), labels[1])):
        ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0), zorder=12,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=14))
        ax.text(x0 + dx * 1.18 + (0.0 if dx else 0.0), y0 + dy * 1.18, f"${lab}$", color=color, fontsize=fs,
                ha="center", va="center", zorder=12)
    if out_of_page:
        ax.scatter([x0], [y0], s=110, facecolors="none", edgecolors=color, lw=1.5, zorder=12)
        ax.scatter([x0], [y0], s=10, c=color, zorder=12)
        ax.text(x0 - 0.18 * L, y0 - 0.22 * L, f"${out_of_page}$", color=color, fontsize=fs, ha="right", va="top", zorder=12)


# ---- layout for Figures 1-3: two columns of width COL_W centred on a fixed 13-in canvas. Plan views sit on top;
# each cross-section is a true-scale strip directly under the plan view it cuts, sharing its horizontal axis.
# The canvas is saved without cropping, so all three figures print at the same scale.
GRID_TOP = 1.15                 # space for the colorbar above the panels (in)
COL_W, COL_GAP, SHARE_GAP = 4.3, 0.95, 0.14
COL_X = [(GEO_W - (2 * COL_W + COL_GAP)) / 2 + i * (COL_W + COL_GAP) for i in (0, 1)]


class GeoGrid:
    """A figure on which axes are placed in inches measured from the top-left corner."""
    def __init__(self, height_in):
        self.H = height_in
        self.fig = plt.figure(figsize=(GEO_W, height_in))

    def axes(self, x, top, w, h, **kw):
        return self.fig.add_axes([x / GEO_W, 1 - (top + h) / self.H, w / GEO_W, h / self.H], **kw)


def h_true(xlim, ylim, ve=1.0):
    """Panel height (in) that draws a COL_W-wide panel at the given vertical exaggeration."""
    return COL_W * (ylim[1] - ylim[0]) * ve / (xlim[1] - xlim[0])


def colorbar_mid(g):
    cw, ch = 0.40 * GEO_W, 0.27
    return m_colorbar(g.fig, [(GEO_W - cw) / 2 / GEO_W, 1 - (0.28 + ch) / g.H, cw / GEO_W, ch / g.H], label_right=True)


def save_fixed(fig, name):
    """Save the whole canvas (no tight crop), so the three geometry figures share one scale on the page."""
    from matplotlib.transforms import Bbox
    os.makedirs(OUT, exist_ok=True)
    w, h = fig.get_size_inches()
    for ext in ("pdf", "png"):
        meta = {"CreationDate": None, "Creator": None, "Producer": None} if ext == "pdf" else {"Software": None}
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), metadata=meta, bbox_inches=Bbox.from_bounds(0, 0, w, h))
    plt.close(fig)
    print(f"wrote figures/output/{name}.pdf/.png")


LEG_H = 0.62        # band for the one-line legend at the bottom of every geometry figure (in)
LEG_DROP = 0.80     # distance from the bottom of the lowest panel to the top of the legend (clears the x label)


def bottom_legend(g, handles, labels, block_bottom):
    """The same legend in the same place on Figures 1-3: one line, centred on the canvas, under the panels."""
    return g.fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1 - (block_bottom + LEG_DROP) / g.H),
                        ncol=len(handles), fontsize=13, framealpha=0.95, handlelength=2.4, borderpad=0.6,
                        columnspacing=1.8)


def fig_height(block_bottom):
    return block_bottom + LEG_DROP + LEG_H

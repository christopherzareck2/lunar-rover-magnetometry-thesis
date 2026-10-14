"""Fig. 1.1 "The target": layered mare basalt as seen from orbit and from the surface.

    python figures/scripts/fig_target_imagery.py        (run by `make figures`; writes ch1_target_imagery.pdf)

Inputs (committed, reduced from public NASA/PDS products by figures/scripts/prepare_image_inputs.py; the raw
downloads are listed in data/external/SOURCES.md and are not in git):
    figures/inputs/nac_M113941548L.png  + .json   (a) LROC NAC, western rim of Hadley Rille, north up
    figures/inputs/nac_M1345757107L.png + .json   (b) LROC NAC, Mare Tranquillitatis pit, north up
    figures/inputs/apollo15_AS15-85-11451.png     (c) Apollo 15 Hasselblad frame, Irwin and the rover at Hadley Rille
Each NAC sidecar holds the metre-per-pixel scale of the saved crop and the sun azimuth in the north-up crop, so the
scale bar and the illumination arrow are drawn from the PDS label values, not estimated by eye.
The figure shows exposure only: it makes no statement about magnetization, rover access, or outcrop versus talus.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from lunarmag import plotting as st  # noqa: E402
from lunarmag.paths import THESIS_FIGURES  # noqa: E402
from lunarmag.pdfa import make_pdfa1b_ready  # noqa: E402

INPUTS = ROOT / "figures" / "inputs"
HALO = [pe.withStroke(linewidth=1.6, foreground="black")]


def _gray_rgb(path: Path) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("L"))
    return np.repeat(a[:, :, None], 3, axis=2)


def _outlined(ax, kind: str, xy, xytext, color: str, lw: float = 1.3, ms: float = 8) -> None:
    """Arrow with a black under-stroke so it reads on both bright and dark ground."""
    for c, w in (("black", lw + 1.6), (color, lw)):
        ax.annotate("", xy=xy, xytext=xytext, zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=w, shrinkA=0, shrinkB=0, mutation_scale=ms))


def _text(ax, x, y, s, color="white", ha="center", va="center", size=None, **kw):
    ax.text(x, y, s, color=color, ha=ha, va=va, fontsize=size or st.BASE_PT - 1, zorder=7, path_effects=HALO, **kw)


def _frame(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)
        sp.set_color(st.INK)


def nac_panel(ax, stem: str, label: str, letter: str) -> None:
    meta = json.loads((INPUTS / f"{stem}.json").read_text())
    rgb = _gray_rgb(INPUTS / f"{stem}.png")
    h, w = rgb.shape[:2]
    ax.imshow(rgb, interpolation="nearest", aspect="equal")
    _frame(ax)
    # scale bar
    L = meta["scale_bar_m"] / meta["m_per_px"]
    x0, y0 = 0.05 * w, 0.93 * h
    ax.plot([x0, x0 + L], [y0, y0], color="black", lw=3.4, solid_capstyle="butt", zorder=5)
    ax.plot([x0, x0 + L], [y0, y0], color="white", lw=1.8, solid_capstyle="butt", zorder=6)
    _text(ax, x0 + L / 2, y0 - 0.035 * h, f"{meta['scale_bar_m']} m", va="bottom")
    # north and the direction of the Sun, drawn from the PDS label angles (clockwise from up)
    ox, oy, ln = 0.64 * w, 0.21 * h, 0.115 * h
    for ang, color, txt in ((meta["north_deg_cw_from_up"], "white", "N"), (meta["sun_deg_cw_from_up"], "#FFD966", "Sun")):
        a = np.deg2rad(ang)
        tip = (ox + np.sin(a) * ln, oy - np.cos(a) * ln)
        _outlined(ax, "arrow", tip, (ox, oy), color)
        off = 0.045 * h
        _text(ax, tip[0] + np.sin(a) * off * (2.2 if txt == "Sun" else 1.0), tip[1] - np.cos(a) * off, txt, color=color)
    _text(ax, 0.97 * w, 0.965 * h, label, ha="right", va="bottom", size=st.BASE_PT - 1.5)
    st.panel_label(ax, letter, x=0.03, y=0.97, dark=True)


def apollo_panel(ax, letter: str) -> None:
    im = np.asarray(Image.open(INPUTS / "apollo15_AS15-85-11451.png").convert("L"))[28:, :]
    h, w = im.shape
    ax.imshow(np.repeat(im[:, :, None], 3, axis=2), interpolation="nearest", aspect="equal")
    _frame(ax)
    # arrow to the rover; the parked Lunar Roving Vehicle is about 3 m long
    rx, ry = 0.36 * w, 0.43 * h
    _outlined(ax, "arrow", (rx, ry), (0.55 * w, 0.16 * h), "white")
    _text(ax, 0.56 * w, 0.115 * h, "Irwin and rover", ha="left")
    _text(ax, 0.97 * w, 0.965 * h, "AS15-85-11451", ha="right", va="bottom", size=st.BASE_PT - 1.5)
    st.panel_label(ax, letter, x=0.03, y=0.97, dark=True)


def draw() -> plt.Figure:
    # Panel sizes in inches: two square NAC crops and the Apollo frame at the same height, 0.1 in apart.
    H, gap = 2.0, 0.1
    apollo = np.asarray(Image.open(INPUTS / "apollo15_AS15-85-11451.png"))
    wa = H * apollo.shape[1] / (apollo.shape[0] - 28)
    total = 2 * H + wa + 2 * gap
    fig = st.figure(total, H + 0.30)
    def axes_unused(x_in):
        return fig.add_axes([x_in / total, 0.0, 0.0, 0.0])
    xs = [0.0, H + gap, 2 * H + 2 * gap]
    widths = [H, H, wa]
    titles = ["Hadley Rille, western rim", "Mare Tranquillitatis pit", "Hadley Rille, Apollo 15"]
    axs = []
    for x, w, t in zip(xs, widths, titles):
        ax = fig.add_axes([x / total, 0.0, w / total, H / (H + 0.30)])
        ax.text(0.0, 1.03, t, transform=ax.transAxes, ha="left", va="bottom", fontsize=st.BASE_PT, color=st.INK)
        axs.append(ax)
    nac_panel(axs[0], "nac_M113941548L", "LROC NAC M113941548L", "a")
    nac_panel(axs[1], "nac_M1345757107L", "LROC NAC M1345757107L", "b")
    apollo_panel(axs[2], "c")
    return fig


def main() -> None:
    with st.style():
        fig = draw()
        st.lint_text(fig)
        paths = st.save(fig, THESIS_FIGURES, "ch1_target_imagery", png_dir=ROOT / "figures" / "preview")
    rep = make_pdfa1b_ready(paths[0])
    print(f"ch1_target_imagery: {paths[0]} {rep}")


if __name__ == "__main__":
    main()

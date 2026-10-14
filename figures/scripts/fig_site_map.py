"""Site map for Chapter 6: the four candidate rilles, Hadley Rille and the Blue Ghost Mission 1 site on a LOLA hillshade.

    python figures/scripts/fig_site_map.py        (run by `make figures`; writes ch6_site_map.pdf)

Inputs, both committed (the raw LOLA elevation map is ~530 MB and is not in git; see data/external/SOURCES.md and
figures/scripts/prepare_image_inputs.py for how they were made):
    figures/inputs/lola_hillshade_nearside.png   hillshade of LOLA LDEM_64, plate carree, 1950 x 1114 px = 300 dpi
    figures/inputs/site_coordinates.csv          IAU Gazetteer centre points and the Blue Ghost 1 landing region
The map carries locations only. It has no field contours: no crustal-field model with a checked open licence was
used, and a contour layer would be new analysis rather than context.
"""
from __future__ import annotations

import csv
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
LON0, LON1, LAT0, LAT1 = -65.0, 82.0, -42.0, 42.0       # window of the hillshade (see lola_hillshade_nearside.json)
KM_PER_DEG = 2 * np.pi * 1737.4 / 360.0                  # 30.32 km per degree of latitude (and of longitude at the equator)
HALO = [pe.withStroke(linewidth=1.8, foreground="white")]
#: maria named for orientation: (label shown, lon, lat of the label); the Gazetteer centre is moved where it would sit
#: under a marker or another label
MARIA_SHOWN = {"Mare Imbrium": ("Mare Imbrium", None, None), "Mare Serenitatis": ("Mare Serenitatis", None, None),
               "Mare Tranquillitatis": ("Mare Tranquillitatis", 26.5, 15.0), "Mare Crisium": ("Mare Crisium", 68.0, 6.0),
               "Mare Fecunditatis": ("Mare Fecunditatis", None, None), "Mare Nubium": ("Mare Nubium", None, None),
               "Mare Humorum": ("Mare Humorum", -35.5, -31.5), "Oceanus Procellarum": ("Oceanus Procellarum", -47.0, 12.0)}


def load_sites() -> dict[str, dict]:
    with open(INPUTS / "site_coordinates.csv") as fh:
        return {r["name"]: dict(r, lat=float(r["lat_deg_north"]), lon=float(r["lon_deg_east"])) for r in csv.DictReader(fh)}


def deg(v: float, pos: str = "") -> str:
    return rf"${abs(v):.0f}^{{\circ}}${pos}"


def draw() -> plt.Figure:
    sites = load_sites()
    img = np.asarray(Image.open(INPUTS / "lola_hillshade_nearside.png").convert("L"))
    rgb = np.repeat(img[:, :, None], 3, axis=2)          # opaque RGB, no alpha
    fig = st.figure("full", 3.80)
    ax = fig.add_axes([0.06, 0.079, 0.925, 0.904])
    ax.imshow(rgb, extent=(LON0, LON1, LAT0, LAT1), interpolation="nearest", aspect="equal", zorder=0)
    ax.set_xlim(LON0, LON1)
    ax.set_ylim(LAT0, LAT1)
    for sp in ax.spines.values():
        sp.set_visible(True)
    # graticule every 15 degrees, thin white dashes
    xt = np.arange(-60, 81, 20)
    yt = np.arange(-40, 41, 20)
    for x in xt:
        ax.axvline(x, color="white", lw=0.4, ls=(0, (3, 3)), zorder=1)
    for y in yt:
        ax.axhline(y, color="white", lw=0.4, ls=(0, (3, 3)), zorder=1)
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels([deg(v, "E" if v > 0 else ("W" if v < 0 else "")) for v in xt])
    ax.set_yticklabels([deg(v, "N" if v > 0 else ("S" if v < 0 else "")) for v in yt])
    ax.tick_params(length=2.5)

    # maria, for orientation only
    for n, (label, lon, lat) in MARIA_SHOWN.items():
        s = sites[n]
        ax.text(s["lon"] if lon is None else lon, s["lat"] if lat is None else lat, label, ha="center", va="center",
                fontsize=st.BASE_PT - 1, color="#2A2A2A", zorder=3, path_effects=HALO)

    # sites
    ms = dict(markeredgecolor="white", markeredgewidth=0.9, linestyle="none", zorder=6)
    cand = ["Rimae Doppelmayer", "Rima Cauchy", "Rima Oppolzer", "Rimae Hypatia"]
    short = {"Rimae Doppelmayer": "Rimae Doppelmayer", "Rima Cauchy": "Rima Cauchy", "Rima Oppolzer": "Rima Oppolzer",
             "Rimae Hypatia": "Rimae Hypatia"}
    off = {"Rimae Doppelmayer": (0, 3.6, "center"), "Rima Cauchy": (2.4, 0.0, "left"), "Rima Oppolzer": (0, -4.0, "center"),
           "Rimae Hypatia": (2.2, 2.6, "left")}
    for n in cand:
        s = sites[n]
        ax.plot(s["lon"], s["lat"], "o", color=st.DIRECTION, markersize=7.5, **ms)
        dx, dy, ha = off[n]
        ax.text(s["lon"] + dx, s["lat"] + dy, short[n], ha=ha, va="center", fontsize=st.BASE_PT, color=st.INK, zorder=7,
                path_effects=HALO)
    h = sites["Rima Hadley"]
    ax.plot(h["lon"], h["lat"], "s", color=st.INTENSITY, markersize=7, **ms)
    ax.text(h["lon"] - 2.2, h["lat"] + 0.3, "Hadley Rille (Apollo 15)", ha="right", va="center", fontsize=st.BASE_PT,
            color=st.INK, zorder=7, path_effects=HALO)
    b = sites["Blue Ghost Mission 1 (target landing region)"]
    ax.plot(b["lon"], b["lat"], "^", color=st.OKABE_ITO["green"], markersize=8, **ms)
    ax.text(b["lon"], b["lat"] + 4.0, "Blue Ghost 1", ha="center", va="center", fontsize=st.BASE_PT, color=st.INK,
            zorder=7, path_effects=HALO)

    # scale bar: 500 km at the equator (the map is plate carree, so the scale is exact only along the equator)
    L = 500.0 / KM_PER_DEG
    x0, y0 = LON0 + 3.0, LAT0 + 4.0
    ax.plot([x0, x0 + L], [y0, y0], color="white", lw=3.6, solid_capstyle="butt", zorder=8)
    ax.plot([x0, x0 + L], [y0, y0], color=st.INK, lw=1.6, solid_capstyle="butt", zorder=9)
    ax.text(x0 + L / 2, y0 + 1.7, r"500 km at the equator", ha="center", va="bottom", fontsize=st.BASE_PT - 1, color=st.INK,
            zorder=9, path_effects=HALO)
    return fig


def main() -> None:
    with st.style():
        fig = draw()
        st.lint_text(fig)
        paths = st.save(fig, THESIS_FIGURES, "ch6_site_map", png_dir=ROOT / "figures" / "ch6" / "png")
    rep = make_pdfa1b_ready(paths[0])
    print(f"ch6_site_map: {paths[0]} {rep}")


if __name__ == "__main__":
    main()

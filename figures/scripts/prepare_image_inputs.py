"""Reduce the public NASA/USGS/PDS downloads in ``data/external/`` to the small image inputs the two imagery
figures read (``figures/inputs/``). Not run by ``make figures``: the raw files are large and are not in git,
so the reduced products are committed instead (a few MB in all) and the figure scripts read only those.

    python figures/scripts/prepare_image_inputs.py map        # LOLA hillshade of the nearside (Fig. 6.x site map)
    python figures/scripts/prepare_image_inputs.py nac        # the two LROC NAC crops (Fig. 1.1 a, b)
    python figures/scripts/prepare_image_inputs.py apollo     # the Apollo 15 frame (Fig. 1.1 c)

What it needs, all listed with URL, size and licence in ``data/external/SOURCES.md``:
    data/external/lola/LDEM_64.IMG (+ .LBL)                       LOLA 64 pix/deg elevation map, PDS Geosciences Node
    data/external/lroc/M113941548LE.IMG, M1345757107LE.IMG         LROC NAC EDR products, PDS (public domain)
    data/external/apollo/a15_h_85_11451.gif                        NASA NSSDC scan of Apollo 15 frame AS15-85-11451
Reads PDS3 images with numpy only (no GDAL or rasterio), writes 8-bit PNGs and a JSON sidecar of the geometry.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "data" / "external"
OUT = ROOT / "figures" / "inputs"

# Map window of the site figure (degrees east, degrees north) and its output pixel grid. 13.3 pix/deg gives
# 1950 x 1114 pixels, which is 300 dpi at the 6.5 in text width.
MAP_LON = (-65.0, 82.0)
MAP_LAT = (-42.0, 42.0)
MAP_PX_PER_DEG = 1950.0 / (MAP_LON[1] - MAP_LON[0])
LIGHT_AZ_DEG, LIGHT_ALT_DEG, Z_EXAG = 315.0, 35.0, 3.0      # hillshade: light from the north-west, as in the USGS LOLA shade
R_MOON_M = 1_737_400.0


# ------------------------------------------------------------------------------------------------ LOLA hillshade
def lola_hillshade() -> dict:
    """Hillshade of LDEM_64 over the map window. Pixel-registered simple cylindrical map: column i is centred on
    longitude (i + 0.5)/64 degrees east (0 to 360), row j on latitude 90 - (j + 0.5)/64."""
    lbl = (EXT / "lola" / "LDEM_64.LBL").read_text()
    assert "LINE_SAMPLES          = 23040" in lbl and "SCALING_FACTOR        = 0.5" in lbl
    dem = np.memmap(EXT / "lola" / "LDEM_64.IMG", dtype="<i2", mode="r", shape=(11520, 23040))
    ppd = 64.0
    # Longitudes -65..82 east wrap across 0: take the 295..360 block and the 0..82 block.
    r0, r1 = int((90 - MAP_LAT[1]) * ppd), int((90 - MAP_LAT[0]) * ppd)
    west = np.asarray(dem[r0:r1, int((MAP_LON[0] + 360) * ppd):])
    east = np.asarray(dem[r0:r1, : int(MAP_LON[1] * ppd)])
    h = np.concatenate([west, east], axis=1).astype(np.float32) * 0.5          # metres
    n_lat, n_lon = h.shape
    lat = MAP_LAT[1] - (np.arange(n_lat) + 0.5) / ppd
    dy = np.deg2rad(1.0 / ppd) * R_MOON_M                                       # metres per row
    dx = (dy * np.cos(np.deg2rad(lat)))[:, None]                                # metres per column, by latitude
    gy, gx = np.gradient(h * Z_EXAG, dy, axis=0), np.gradient(h * Z_EXAG, axis=1) / dx   # rows go south, columns east
    # Surface normal (-dz/dx, +dz/dy_north, 1); gy above is dz/d(south), so north slope = -gy.
    nx, ny, nz = -gx, gy, np.ones_like(gx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    az, alt = np.deg2rad(LIGHT_AZ_DEG), np.deg2rad(LIGHT_ALT_DEG)
    lx, ly, lz = np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)     # towards the light (E, N, up)
    shade = (nx * lx + ny * ly + nz * lz) / norm
    shade = np.clip(shade, 0.0, 1.0).astype(np.float32)
    w, hgt = int(round((MAP_LON[1] - MAP_LON[0]) * MAP_PX_PER_DEG)), int(round((MAP_LAT[1] - MAP_LAT[0]) * MAP_PX_PER_DEG))
    img = Image.fromarray(shade).resize((w, hgt), Image.LANCZOS)
    s = np.asarray(img)
    lo, hi = np.percentile(s, [0.5, 99.5])
    g = np.clip((s - lo) / (hi - lo), 0, 1) ** 1.0
    out = (255 * g).astype(np.uint8)
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(OUT / "lola_hillshade_nearside.png", optimize=True)
    meta = {"source": "LOLA LDEM_64 (LRO-L-LOLA-4-GDR-V1.0, V3.1)", "window_lon_east_deg": MAP_LON, "window_lat_deg": MAP_LAT,
            "size_px": [w, hgt], "projection": "simple cylindrical (plate carree)", "light_azimuth_deg": LIGHT_AZ_DEG,
            "light_altitude_deg": LIGHT_ALT_DEG, "vertical_exaggeration": Z_EXAG, "stretch_percentiles": [0.5, 99.5]}
    (OUT / "lola_hillshade_nearside.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


# ------------------------------------------------------------------------------------------------ site coordinates
GAZETTEER_ZIP = EXT / "gazetteer" / "MOON_nomenclature_center_pts.zip"
#: (feature name in the IAU Gazetteer of Planetary Nomenclature, how the thesis calls it)
GAZETTEER_SITES = ["Rimae Doppelmayer", "Rima Cauchy", "Rima Oppolzer", "Rimae Hypatia", "Rima Hadley"]
MARIA = ["Mare Imbrium", "Mare Serenitatis", "Mare Tranquillitatis", "Mare Crisium", "Mare Fecunditatis", "Mare Nubium",
         "Mare Humorum", "Oceanus Procellarum", "Mare Vaporum", "Mare Insularum", "Mare Cognitum", "Mare Nectaris"]
#: Blue Ghost Mission 1 is not in the Gazetteer; the target landing region is the one on the LROC image page
BLUE_GHOST = ("Blue Ghost Mission 1 (target landing region)", 18.56, 61.81,
              "https://www.lroc.im-ldi.com/images/1400 (landing region, eastern Mare Crisium)")


def read_dbf(path: Path) -> list[dict]:
    import struct
    f = Path(path).read_bytes()
    n, hl, rl = struct.unpack("<xxxxIHH", f[:12])
    fields, o = [], 32
    while f[o] != 0x0D:
        fields.append((f[o:o + 11].split(b"\0")[0].decode(), f[o + 16]))
        o += 32
    rows = []
    for i in range(n):
        r, p, d = f[hl + i * rl: hl + (i + 1) * rl], 1, {}
        for name, ln in fields:
            d[name] = r[p:p + ln].decode("latin1").strip()
            p += ln
        rows.append(d)
    return rows


def site_coordinates() -> None:
    """Write figures/inputs/site_coordinates.csv: Gazetteer centre points (and the maria used as labels)."""
    import csv
    import io
    import zipfile
    z = zipfile.ZipFile(GAZETTEER_ZIP)
    dbf = [n for n in z.namelist() if n.endswith(".dbf")][0]
    tmp = EXT / "gazetteer" / "x"
    tmp.mkdir(exist_ok=True)
    z.extract(dbf, tmp)
    rows = {r["clean_name"]: r for r in read_dbf(tmp / dbf)}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "site_coordinates.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["kind", "name", "lat_deg_north", "lon_deg_east", "source"])
        for name in GAZETTEER_SITES:
            r = rows[name]
            lon = float(r["center_lon"])
            lon = lon - 360.0 if lon > 180 else lon
            w.writerow(["site", name, f"{float(r['center_lat']):.4f}", f"{lon:.4f}",
                        f"IAU Gazetteer of Planetary Nomenclature, {r['link']} (center point, MOON_nomenclature_center_pts)"])
        w.writerow(["site", BLUE_GHOST[0], f"{BLUE_GHOST[1]:.4f}", f"{BLUE_GHOST[2]:.4f}", BLUE_GHOST[3]])
        for name in MARIA:
            r = rows[name]
            lon = float(r["center_lon"])
            lon = lon - 360.0 if lon > 180 else lon
            w.writerow(["mare", name, f"{float(r['center_lat']):.4f}", f"{lon:.4f}",
                        f"IAU Gazetteer of Planetary Nomenclature, {r['link']}"])
    print((OUT / "site_coordinates.csv").read_text())


# ------------------------------------------------------------------------------------------------ Apollo frame
def apollo_frame() -> dict:
    """AS15-85-11451 as served by NASA NSSDC (648 x 590 GIF scan). Converted to an 8-bit grey PNG, unchanged otherwise."""
    im = Image.open(EXT / "apollo" / "a15_h_85_11451.gif").convert("L")
    OUT.mkdir(parents=True, exist_ok=True)
    im.save(OUT / "apollo15_AS15-85-11451.png", optimize=True)
    meta = {"frame": "AS15-85-11451", "size_px": list(im.size), "source": "https://nssdc.gsfc.nasa.gov/imgcat/hires/a15_h_85_11451.gif",
            "credit": "NASA (Apollo 15, photograph by D. R. Scott)"}
    (OUT / "apollo15_AS15-85-11451.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


# ------------------------------------------------------------------------------------------------ PDS3 label
def read_pds3(path: Path) -> tuple[dict, np.ndarray]:
    """Minimal PDS3 reader for a LROC NAC EDR: returns (label key/values, 8-bit array lines x samples)."""
    raw = Path(path).read_bytes()
    head = raw[: 2_000_000].decode("latin1")
    end = re.search(r"^END\s*$", head, re.M).end()
    lab = {}
    for m in re.finditer(r"^\s*([A-Z_^:a-z0-9]+)\s*=\s*(.+?)\s*$", head[:end], re.M):
        lab.setdefault(m.group(1), m.group(2))
    rec = int(lab["RECORD_BYTES"])
    lines, samples = int(lab["LINES"]), int(lab["LINE_SAMPLES"])
    bits = int(lab["SAMPLE_BITS"])
    # ^IMAGE = n gives the start record (1-based) when it is a bare integer
    ptr = lab.get("^IMAGE", "").split()[0].strip('"<>')
    offset = (int(ptr) - 1) * rec if ptr.isdigit() else int(lab["LABEL_RECORDS"]) * rec
    dt = {8: "u1", 16: ">u2"}[bits] if lab.get("SAMPLE_TYPE", "").upper().startswith("MSB") or bits == 8 else "<u2"
    arr = np.frombuffer(raw, dtype=dt, count=lines * samples, offset=offset).reshape(lines, samples)
    return lab, arr


def index_row(lbl: Path, row_file: Path) -> dict:
    """Named columns of one product's row of a LROC EDR INDEX.TAB (the label says which column is which)."""
    import csv
    import io
    names = re.findall(r'^\s*NAME\s*=\s*"?([A-Za-z0-9_:]+)"?', lbl.read_text(), re.M)[1:]
    row = next(csv.reader(io.StringIO(row_file.read_text().strip().split("\n")[0])))
    return {n: v.strip() for n, v in zip(names, row)}


#: The two NAC crops. ``center`` is (raw column, raw line) of the crop centre in the EDR file as stored (not mirrored),
#: ``field_m`` the side of the square crop on the ground, ``px`` the side of the output PNG (300 dpi at 2 in).
NAC_CROPS = {
    "M113941548L": dict(edr="M113941548LE.IMG", lbl="INDEX_0001.LBL", row="row_M113941548LE.txt", center=(2100, 5900),
                        field_m=500.0, px=600, scale_bar_m=100),
    "M1345757107L": dict(edr="M1345757107LE.IMG", lbl="INDEX_0043C.LBL", row="row_M1345757107LE.txt", center=(4008, 21856),
                         field_m=200.0, px=600, scale_bar_m=50, median_to=0.45),
}


def column_flatfield(arr: np.ndarray, line0: int = 0, line1: int | None = None) -> np.ndarray:
    """Remove the column-to-column pattern (vertical striping) of the raw EDR. The pattern is the per-column median
    over lines ``line0:line1`` minus its 101-column running mean, subtracted as an offset; broad brightness trends are
    kept. This is a display correction, not the LROC radiometric calibration (CDR)."""
    from scipy.ndimage import uniform_filter1d
    a = arr.astype(np.float32)
    med = np.median(a[line0:line1:2], axis=0)
    pattern = med - uniform_filter1d(med, 101, mode="nearest")
    pattern = np.where(med > 0, np.clip(pattern, -60.0, 60.0), 0.0)          # dead edge columns: no correction
    return a - pattern[None, :]


def nac_crop(pid: str) -> dict:
    cfg = NAC_CROPS[pid]
    lab, arr = read_pds3(EXT / "lroc" / cfg["edr"])
    ix = index_row(EXT / "lroc" / cfg["lbl"], EXT / "lroc" / cfg["row"])
    assert ix["PRODUCT_ID"] == cfg["edr"].replace(".IMG", "")
    pw, ph = float(ix["SCALED_PIXEL_WIDTH"]), float(ix["SCALED_PIXEL_HEIGHT"])   # m/pixel across and along track
    cx, cy = cfg["center"]
    hw, hh = int(round(cfg["field_m"] / pw / 2)), int(round(cfg["field_m"] / ph / 2))
    flat = column_flatfield(arr, cy - 5000, cy + 5000)    # local to the crop: the striping drifts along the strip
    crop = flat[cy - hh: cy + hh, cx - hw: cx + hw]
    crop = crop[:, ::-1]                      # the stored EDR is a mirror image of the surface seen from above (see below)
    lo, hi = np.percentile(crop, [0.5, 99.7])
    g = np.clip((crop - lo) / (hi - lo), 0, 1)
    gamma = cfg.get("gamma", 0.85)
    if "median_to" in cfg:                    # choose gamma so the median pixel lands on this grey level
        gamma = float(np.log(cfg["median_to"]) / np.log(np.median(g)))
    g = g ** gamma
    im = Image.fromarray((255 * g).astype(np.uint8)).resize((cfg["px"], cfg["px"]), Image.LANCZOS)
    OUT.mkdir(parents=True, exist_ok=True)
    im.save(OUT / f"nac_{pid}.png", optimize=True)
    # Label angles are clockwise from the image's right-hand axis, and describe the stored (mirrored) EDR: horizontal
    # mirroring maps an angle t to 180 - t. Checked against the physical bearing of the sub-solar point (see the
    # report): the stored image has east on the left, so the saved crop is flipped to the usual east-on-the-right.
    north_right, sun_right = 180.0 - float(ix["NORTH_AZIMUTH"]), 180.0 - float(ix["SUB_SOLAR_AZIMUTH"])
    meta = {"product_id": pid, "edr_file": cfg["edr"], "pds_path": ix["FILE_SPECIFICATION_NAME"], "start_time": ix["START_TIME"],
            "field_m": cfg["field_m"], "size_px": [cfg["px"], cfg["px"]], "m_per_px": cfg["field_m"] / cfg["px"],
            "native_m_per_px": [pw, ph], "resolution_m": float(ix["RESOLUTION"]), "incidence_deg": float(ix["INCIDENCE_ANGLE"]),
            "emission_deg": float(ix["EMISSION_ANGLE"]), "phase_deg": float(ix["PHASE_ANGLE"]),
            "scene_center_lat_lon": [float(ix["CENTER_LATITUDE"]), float(ix["CENTER_LONGITUDE"])],
            "crop_center_edr_column_line": list(cfg["center"]), "scale_bar_m": cfg["scale_bar_m"],
            "north_deg_cw_from_up": (north_right + 90.0 + 180.0) % 360.0 - 180.0,
            "sun_deg_cw_from_up": (sun_right + 90.0) % 360.0,
            "processing": "column-offset (destriping) correction of the raw EDR; horizontal mirror; resampled to square pixels (Lanczos); "
                          "linear stretch between the 0.5 and 99.7 percentiles, gamma (0.85; for the high-Sun pit image, chosen so the median pixel is mid-grey)"}
    (OUT / f"nac_{pid}.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main(argv: list[str]) -> int:
    what = argv[1] if len(argv) > 1 else "map"
    if what == "map":
        print(json.dumps(lola_hillshade(), indent=1))
        site_coordinates()
    elif what == "nac":
        for pid in NAC_CROPS:
            print(json.dumps(nac_crop(pid), indent=1))
    elif what == "apollo":
        print(json.dumps(apollo_frame(), indent=1))
    elif what == "coords":
        site_coordinates()
    else:
        print("not implemented yet:", what)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

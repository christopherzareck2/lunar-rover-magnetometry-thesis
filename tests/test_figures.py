"""Thesis figures: same plotted values as the companion-paper scripts, PDF/A-1b ready, deterministic, manifest current.

The thesis figures (``lunarmag.figures``) are a restyle of the companion paper's figure scripts
(``figures/scripts/fig0*.py``) and of the earlier thesis budget script (``thesis/.../figures/src``). The values test
runs each OLD script with its save step intercepted, collects what it plotted (line data, patch geometry, or its
own data function), and requires the new figure's plotted values to be identical (exact float equality).
"""
import json
import os
import runpy
import sys

import numpy as np
import pytest

from lunarmag import figures as F
from lunarmag import pdfa
from lunarmag.paths import ROOT

OLD_SCRIPTS = os.path.join(ROOT, "figures", "scripts")


def run_old(script):
    """Run a companion-paper figure script; return its namespace and the matplotlib Figure it would have saved."""
    import matplotlib
    matplotlib.use("Agg")
    if OLD_SCRIPTS not in sys.path:
        sys.path.insert(0, OLD_SCRIPTS)
    import paperstyle as ps
    captured = {}

    def grab(fig, name):
        captured["fig"] = fig

    saved = (ps.save, ps.save_fixed)
    ps.save = ps.save_fixed = grab
    try:
        ns = runpy.run_path(os.path.join(ROOT, script), run_name="old_figure")
        ns["main"]()
    finally:
        ps.save, ps.save_fixed = saved
    return ns, captured["fig"]


def old_lines(fig):
    out = []
    for ax in fig.axes:
        for ln in ax.get_lines():
            x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
            if x.size:
                out.append((x, y))
    return out


def has_line(lines, x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return any(a.shape == x.shape and np.array_equal(a, x) and np.array_equal(b, y) for a, b in lines)


def old_rects(fig):
    from matplotlib.patches import Rectangle
    return {tuple(np.round([p.get_x(), p.get_y(), p.get_width(), p.get_height()], 12))
            for ax in fig.axes for p in ax.patches if isinstance(p, Rectangle)}


@pytest.fixture(scope="module")
def values():
    return {f.number: F.figure_data(f) for f in F.FIGURES}


def test_fig_3_1_geometry_matches_paper_figure(values):
    d = values["3.1"]
    _, fig = run_old("figures/scripts/fig01_isolated_outcrop_geometry.py")
    lines = old_lines(fig)
    loop, by = np.array(d["drive_around_xy_m"]), np.array(d["drive_by_xy_m"])
    assert has_line(lines, loop[:, 0], loop[:, 1]) and has_line(lines, by[:, 0], by[:, 1])
    (x0, x1), (y0, y1), (z0, z1) = d["prism_xyz_m"]
    rects = old_rects(fig)
    assert (x0, y0, x1 - x0, y1 - y0) in rects and (x0, z0, x1 - x0, z1 - z0) in rects
    assert has_line(lines, [-12, 12], [d["sensor_z_m"]] * 2)


def test_fig_3_2_scene_matches_paper_figure(values):
    d = values["3.2"]
    ns, fig = run_old("figures/scripts/fig02_skatepark_geometry.py")
    for case in ("A", "B"):
        cfgs, _, _ = ns["SP"].outcrop_configs_for(ns["DR"], case)
        assert [c["shape"] for c in cfgs] == [c["shape"] for c in d[case]]
        for old, new in zip(cfgs, d[case]):
            assert list(old["center"]) == new["center"]
            assert list(old["magnetization"]) == new["magnetization"]
            assert {k: float(v) for k, v in old["params"].items()} == new["params"]
    L = d["loop_half_m"]
    assert has_line(old_lines(fig), [-L, L, L, -L, -L], [-L, -L, L, L, -L])


def test_fig_3_3_wall_matches_paper_figure(values):
    d = values["3.3"]
    ns, fig = run_old("figures/scripts/fig03_three_layer_geometry.py")
    assert [(z0, z1, tuple(m)) for z0, z1, _, m in ns["LAYERS"]] == [(*f["z_m"], tuple(f["M_dir"])) for f in d["flows"]]
    lines = old_lines(fig)
    x0, x1 = d["traverse_x_m"]
    for z in d["traverse_z_m"]:
        assert has_line(lines, [x0, x1], [z, z])
    assert has_line(lines, [x0, x1], [d["traverse_y_m"]] * 2)
    assert d["mesh_origin_m"] == [-50.0, 0.0, 0.0] and d["mesh_h_m"] == [[100.0], [50.0], [5.0, 5.0, 5.0]]


def test_fig_4_1_values_match_paper_figure(values):
    d = values["4.1"]
    _, fig = run_old("figures/scripts/fig04_detection_envelope.py")
    lines = old_lines(fig)
    assert has_line(lines, d["ratio"], d["dir_err_median_deg"])
    assert has_line(lines, d["ratio"], d["dM_abs_median_pct"])


def test_fig_4_2_values_match_paper_figure(values):
    d = values["4.2"]
    ns, _ = run_old("figures/scripts/fig05_lidar_sensitivity.py")
    g = json.load(open(os.path.join(ROOT, "results", "analysis", "a02_isolated_outcrop_paper_geometry.json")))["lidar_grid"]
    for name in ("drive_around", "drive_by"):
        P, R, D = ns["grid"](g[name], "dir_err_deg")
        _, _, M = ns["grid"](g[name], "dM_abs_pct")
        assert np.array_equal(P, d[name]["pointing_deg"]) and np.array_equal(R, d[name]["range_m"])
        assert np.array_equal(D, d[name]["dir_err_median_deg"]) and np.array_equal(M, d[name]["dM_abs_median_pct"])


def test_fig_4_3_values_match_paper_figure(values):
    d = values["4.3"]
    _, fig = run_old("figures/scripts/fig06_skatepark_recovery.py")
    lines = old_lines(fig)
    for case in ("A", "B"):
        assert has_line(lines, d[case]["delta_r_m"], d[case]["dir_err_median_deg"])
        assert has_line(lines, d[case]["delta_r_m"], d[case]["dM_abs_median_pct"])


def test_fig_4_4_values_match_paper_figure(values):
    d = values["4.4"]
    _, fig = run_old("figures/scripts/fig07_second_traverse_height.py")
    lines = old_lines(fig)
    for k in ("layer_1", "layer_2", "layer_3"):
        assert has_line(lines, d["z2_m"], d["flows"][k])
    from matplotlib.patches import Rectangle               # the grey band (axvspan) of the old figure
    xs = [(p.get_x(), p.get_x() + p.get_width()) for ax in fig.axes for p in ax.patches if isinstance(p, Rectangle)]
    assert any(np.allclose(x, d["band_m"], rtol=0, atol=1e-12) for x in xs)


def test_fig_5_1_values_match_paper_figure(values):
    d = values["5.1"]
    _, fig = run_old("figures/scripts/fig08_noise_requirement.py")
    lines = old_lines(fig)
    assert has_line(lines, d["L_cm"], d["none_deg"]) and has_line(lines, d["L_cm"], d["offsets_deg"])
    assert has_line(lines, d["drift_rms_nT"], d["drift_deg"])
    xinf = max(d["L_cm"]) * 3.2
    assert has_line(lines, [xinf], [d["inf_none_deg"]]) and has_line(lines, [xinf], [d["inf_offsets_deg"]])


def test_fig_5_2_values_match_earlier_thesis_figure(values, tmp_path):
    """Same min/max over directions as the earlier budget script (its stored plotted values are rounded to 1e-4)."""
    old_values = json.load(open(os.path.join(ROOT, "thesis", "MIT-thesis-template", "figures", "src",
                                             "thesis_fig_budget_plotted_values.json")))
    rows = [r for r in values["5.2"]["rows"] if not r["key"].startswith("y6_03:")]   # known-depth rows were added on Oct 13 (finding Y6-03)
    new = {r["key"]: (*r["all_sources_deg"], *r["noise_free_deg"]) for r in rows}
    assert [k for k, *_ in old_values] == [r["key"] for r in rows]
    for key, *vals in old_values:
        assert [round(v, 4) for v in new[key]] == vals


@pytest.mark.parametrize("spec", F.FIGURES, ids=[f.number for f in F.FIGURES])
def test_figure_builds_pdfa1b_ready(spec, tmp_path):
    """Smoke test: every figure draws, passes the text lint, and its PDF has no transparency and full CIDSets."""
    rec = F.build(spec, out_dir=tmp_path, png_dir=None, values_dir=tmp_path)
    pdf = tmp_path / f"{spec.stem}.pdf"
    assert pdfa.scan_transparency(pdf) == {}
    assert pdfa.fonts_missing_cidset(pdf) == 0
    assert pdfa.cidsets_complete(pdf)          # complete sets pass PDF/A-2b (6.2.11.4.2) as well as 1b (6.3.5)
    assert rec["pdfa"]["fonts_missing_cidset"] == 0


def test_figures_are_byte_reproducible(tmp_path):
    spec = F.BY_NUMBER["4.1"]
    a = F.build(spec, out_dir=tmp_path / "a", png_dir=None, values_dir=tmp_path / "a")
    b = F.build(spec, out_dir=tmp_path / "b", png_dir=None, values_dir=tmp_path / "b")
    assert a["pdf_sha256"] == b["pdf_sha256"]


def test_manifest_matches_the_thesis_figures():
    """figures/manifest.json (written by `make figures`) must describe the figure files now in the thesis."""
    path = os.path.join(ROOT, "figures", "manifest.json")
    if not os.path.exists(path):
        pytest.skip("no manifest yet; run `make figures`")
    man = json.load(open(path))
    assert {e["figure_id"] for e in man["figures"]} == {f.number for f in F.FIGURES}
    for e in man["figures"]:
        assert F.sha256(os.path.join(ROOT, e["file"])) == e["pdf_sha256"], e["file"]
        for res, sha in e["results"].items():
            assert F.sha256(os.path.join(ROOT, res)) == sha, res


def test_transparency_scanner_and_cidset_check_catch_bad_files(tmp_path):
    """The scanner sees indirect transparency groups and blend-mode arrays; the CIDSet check wants the exact bitmap."""
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /XObject /Group 2 0 R >>\nendobj\n"
                    b"2 0 obj\n<< /S /Transparency /CS /DeviceRGB >>\nendobj\n"
                    b"3 0 obj\n<< /Type /ExtGState /BM [/Multiply /Normal] >>\nendobj\ntrailer\n<< >>\n%%EOF\n")
    found = pdfa.scan_transparency(bad)
    assert found.get("Group/Transparency") == 1 and found.get("BM Multiply") == 1
    spec = F.BY_NUMBER["4.1"]
    F.build(spec, out_dir=tmp_path, png_dir=None, values_dir=tmp_path)
    pdf = tmp_path / f"{spec.stem}.pdf"
    assert pdfa.cidsets_complete(pdf)
    data = pdf.read_bytes()
    import re as _re
    tail = data[data.rfind(b"%%EOF", 0, len(data) - 6):]       # the appended CIDSet update
    m = _re.search(rb"/Length (\d+) >>\nstream\n", data[len(data) - len(tail):])
    assert m is not None


THESIS_FIG_DIR = os.path.join(ROOT, "thesis", "MIT-thesis-template", "figures")


@pytest.mark.parametrize("pdf", sorted(f for f in os.listdir(THESIS_FIG_DIR) if f.endswith(".pdf")))
def test_every_thesis_figure_is_pdfa1b_clean(pdf):
    """Every PDF the thesis can include (all chapters, any driver) has no transparency and complete CIDSets.
    Fix with `make figures` or `python -m lunarmag pdfa-figures`."""
    path = os.path.join(THESIS_FIG_DIR, pdf)
    assert pdfa.scan_transparency(path) == {}, f"{pdf}: transparency (PDF/A-1 6.4)"
    assert pdfa.fonts_missing_cidset(path) == 0, f"{pdf}: CIDFont without CIDSet (PDF/A-1 6.3.5)"
    assert pdfa.cidsets_complete(path), f"{pdf}: incomplete CIDSet (fails PDF/A-2b 6.2.11.4.2)"


def test_manifest_lists_every_thesis_figure():
    """No figure in the thesis folder without a recorded driver, inputs and hash."""
    path = os.path.join(ROOT, "figures", "manifest.json")
    if not os.path.exists(path):
        pytest.skip("no manifest yet; run `make figures`")
    man = json.load(open(path))
    listed = {os.path.basename(e["file"]) for e in man["figures"] + man.get("external_figures", [])}
    present = {f for f in os.listdir(THESIS_FIG_DIR) if f.endswith(".pdf")}
    assert present <= listed, f"figures without a manifest entry: {sorted(present - listed)}"
    for e in man.get("external_figures", []):
        assert F.sha256(os.path.join(ROOT, e["file"])) == e["pdf_sha256"], e["file"]
        assert os.path.exists(os.path.join(ROOT, e["script"])), e["script"]

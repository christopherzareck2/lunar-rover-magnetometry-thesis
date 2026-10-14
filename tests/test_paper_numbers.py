"""notebook/paper_numbers.py: every quoted number reproduces, and the six v1.0.0 wrong-key pairings stay fixed.

A wrong-key pairing is a row whose sentence describes one quantity while its value is read from another stored
key that happens to round to the same printed number. The checks below compare the row's value to the RIGHT key
with exact float equality, so a pairing that merely rounds the same way fails.
"""
import pytest

import paper_numbers as P


def rows_with(fragment, printed):
    hits = [r for r in P.ROWS if fragment in r[1] and r[2] == printed]
    assert hits, f"no row for {fragment!r} = {printed}"
    return hits


def test_all_quoted_numbers_reproduce():
    ok, total = P.count()
    assert (ok, total) == (374, 374)


def test_every_row_formats_to_its_printed_value():
    """Each row reproduces the paper's printed value, or its v1.1 corrected value if the skate-park fix moved it."""
    bad = [(s, paper, fmt.format(v)) for sec, s, paper, fmt, v in P.ROWS
           if fmt.format(v).lstrip("+") != P.expected(sec, s, paper).lstrip("+")]
    assert not bad, bad[:3]


def test_superseded_rows_are_exactly_the_skatepark_rows_that_moved():
    """The Y6-04 registry is not stale: every listed row differs from the paper and is a Section 4.3 number, and
    every other row still matches the paper exactly (21 rows moved; 353 of 374 match the paper as printed)."""
    keys = {(sec, s, paper) for sec, s, paper, _ in P.SUPERSEDED_Y6_04}
    assert len(keys) == len(P.SUPERSEDED_Y6_04) == 21
    for sec, s, paper, fmt, v in P.ROWS:
        moved = fmt.format(v).lstrip("+") != paper.lstrip("+")
        assert moved == ((sec, s, paper) in keys), (sec, paper, s[:60])
        if moved:
            assert sec == "4.3"
    assert P.count(strict=True) == (353, 374)


@pytest.mark.parametrize("fragment,printed,section,expected", [
    # 1. Section 4.1: the spherical-voxel mean intensity error comes from the approach comparison (Table 2),
    #    not from the skate-park sweep (v1.0.0 read min over skate-park case B; both round to -3.1).
    ("mean intensity errors are -0.3%, -8.9%, and -3.1%", "-3.1", "4.1",
     lambda: P.t2["spherical_voxel"]["mag_err_pct_mean"]),
    # 2. Abstract: "~30 deg" for the upper flows is the a03 baseline case, not the along-x test case (T1).
    ("but the upper two only to ∼30°", "30", "Abstract",
     lambda: P.a3["cases"]["T0_baseline"]["layer_2"]["dir_err_mean"]),
    # 3. Section 4.2: the 3.2 deg median at zero mis-pose belongs to the "perfect geometry" sentence.
    ("With perfect geometry the median direction error in this sweep is 3.2°", "3.2", "4.2",
     lambda: P.lg["drive_around"][(0.0, 0.0)]["dir_err_deg"]["median"]),
    # 4. Section 4.4: the damped top-flow intensity bias (20 %) belongs to the Table 4 sentence.
    ("biases their intensities low by 17% and 20%", "20", "4.4",
     lambda: -P.L("A_single_base_traverse", 3, "mag_err_pct")),
    # 5. Section 4.4: 6.1 deg is the middle flow of the wall that ends at x = 50 m.
    ("ends at x = 50 m, inside the traverse, gives 2.8°, 6.1°, and 2.9°", "6.1", "4.4",
     lambda: P.g7("END", "ground+12.5m", 2)),
    # 6. Section 4.4: 3.7 deg is the middle flow with the cross-strike profile.
    ("adding a 100 m cross-strike profile to both traverses gives direction errors of 1.1°, 3.7°, and 1.5°", "3.7", "4.4",
     lambda: P.a7["W1000_XPROF"]["ground+12.5m"]["cross_strike_only"]["layer_2"]["dir_err_mean"]),
])
def test_fixed_pairings_read_the_right_key(fragment, printed, section, expected):
    hits = rows_with(fragment, printed)
    assert len(hits) == 1
    sec, _, _, _, value = hits[0]
    assert sec == section
    assert value == expected()


@pytest.mark.parametrize("fragment,printed,expected", [
    # the sentences the four moved rows used to sit under are still checked, against their own keys
    ("with a mean direction error of 3.2° and a 95th percentile of 6.1°", "3.2", lambda: P.da["dir_err_deg"]["mean"]),
    ("with a mean direction error of 3.2° and a 95th percentile of 6.1°", "6.1", lambda: P.da["dir_err_deg"]["p95"]),
    ("by +16–20% at b = 0.25 m", "20", lambda: P.C["drive_around"][0.25]["+x"]["dM_signed_pct"]["mean"]),
    ("leaves the direction error of a vertical magnetization at 3.7°", "3.7",
     lambda: P.H1r["drive_around"][(-90, 0)]["dir_err_mean"]),
])
def test_original_sentences_keep_only_their_own_key(fragment, printed, expected):
    for _, _, _, _, value in rows_with(fragment, printed):
        assert value == expected()

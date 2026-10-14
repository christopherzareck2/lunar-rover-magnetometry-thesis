"""Regression: rerun seeded parts of the analysis from scratch and require the stored results EXACTLY.

Every random draw is seeded (``inversion_helpers.seed_for``), so with the pinned environment the rerun must
reproduce the stored JSON bit for bit (the comparison is ``==`` on every number, not a tolerance; set
``LUNARMAG_RTOL`` to allow last-digit differences on another platform). These parts
cover about 16,000 stored numbers of Table 4.2 and Sections 4.2, 5.1-5.3 and 5.8 and take about 20 s together; the
attitude-tilt parts J and T of a09 (about 50 s) are marked slow. This extends ``tests/reproduce_check.py``
(a06 E and a09 M at rtol 1e-6), which still runs on its own.
"""
import contextlib
import io
import math
import os
import sys

import pytest

A6_PARTS = {"A": "A_dc_offsets", "B": "B_direction_height_scan", "C": "C_buried_extent", "D": "D_heterogeneity",
            "E": "E_sampling_vs_ASD", "F": "F_estimator_band", "G": "G_gradient_and_drift",
            "H": "H_offsets_and_burial", "I": "I_survey_designs_for_offsets"}
A9_FAST = {"M": "M_background_depth", "N": "N_induced_magnetization"}
A9_SLOW = {"J": "J_attitude_tilt", "T": "T_attitude_tilt_asymmetric"}


def leaves(o, p=""):
    """Every leaf value AND every container's type and length, so structure differences are caught too."""
    if isinstance(o, dict):
        yield f"{p}<dict>", len(o)
        for k, v in o.items():
            yield from leaves(v, f"{p}/{k}")
    elif isinstance(o, (list, tuple)):
        yield f"{p}<list>", len(o)
        for i, v in enumerate(o):
            yield from leaves(v, f"{p}[{i}]")
    elif isinstance(o, bool) or o is None or isinstance(o, str):
        yield p, o
    else:
        yield p, float(o)


def run_part(mod, part):
    import json
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        getattr(mod, "part_" + part)(out)
    return json.loads(json.dumps(out))          # same float/str normalization as the stored JSON


# Exact by default (the pinned environment on Apple silicon reproduces every bit). Other platforms or BLAS builds
# can differ in the last bits of a matrix product; CI on Linux sets LUNARMAG_RTOL=1e-3.
RTOL = float(os.environ.get("LUNARMAG_RTOL", "0"))
ATOL = float(os.environ.get("LUNARMAG_ATOL", "0"))  # absolute floor for values stored as exactly 0


def _same(x, y):
    if isinstance(x, float) and isinstance(y, float) and not (math.isfinite(x) and math.isfinite(y)):
        return x == y or (math.isnan(x) and math.isnan(y))
    if RTOL and isinstance(x, float) and isinstance(y, float):
        return abs(x - y) <= RTOL * max(abs(x), abs(y)) + max(ATOL, 1e-12)
    return x == y


def assert_exact(new, ref, label):
    a, b = dict(leaves(new)), dict(leaves(ref))
    assert set(a) == set(b), f"{label}: key sets differ: {sorted(set(a) ^ set(b))[:5]}"
    bad = [k for k in a if not _same(a[k], b[k])]
    assert not bad, f"{label}: {len(bad)} of {len(a)} values differ, e.g. {bad[0]}: {a[bad[0]]!r} vs {b[bad[0]]!r}"
    return len(a)


@pytest.mark.parametrize("part", sorted(A6_PARTS))
def test_a06_part_reproduces_exactly(A6, stored, part):
    key = A6_PARTS[part]
    n = assert_exact(run_part(A6, part)[key], stored("a06_isolated_robustness.json")[key], f"a06 {part}")
    assert n > 30


@pytest.mark.parametrize("part", sorted(A9_FAST))
def test_a09_fast_part_reproduces_exactly(A9, stored, part):
    key = A9_FAST[part]
    assert_exact(run_part(A9, part)[key], stored("a09_systematics.json")[key], f"a09 {part}")


@pytest.mark.slow
@pytest.mark.parametrize("part", sorted(A9_SLOW))
def test_a09_tilt_part_reproduces_exactly(A9, stored, part):
    key = A9_SLOW[part]
    assert_exact(run_part(A9, part)[key], stored("a09_systematics.json")[key], f"a09 {part}")


def test_notebook_single_draw_matches_stored(stored):
    import part1_single_draw as P1
    assert P1.check(P1.compute(), stored("part1_single_draw.json"), rtol=max(1e-6, RTOL))


def test_engine_aliases_are_the_package_modules():
    """engine/<name>.py must hand back the lunarmag module itself, so module-level state is shared."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "engine"))
    try:
        import core, inversion_helpers, isolated_outcrop_1cm, correlated_noise_sensitivity  # noqa: E401
        import lunarmag.core, lunarmag.inversion_helpers, lunarmag.isolated_outcrop_1cm  # noqa: E401
        import lunarmag.correlated_noise_sensitivity
        assert core is lunarmag.core
        assert inversion_helpers is lunarmag.inversion_helpers
        assert isolated_outcrop_1cm is lunarmag.isolated_outcrop_1cm
        assert correlated_noise_sensitivity is lunarmag.correlated_noise_sensitivity
        assert inversion_helpers.seed_for("a09:U_budget_full_offcentre:-z") == lunarmag.inversion_helpers.seed_for(
            "a09:U_budget_full_offcentre:-z")
    finally:
        sys.path.remove(os.path.join(root, "engine"))


def test_seed_derivation_is_stable():
    """seed_for is SHA-256 of '<master seed>:<label>'; changing it would silently change every result."""
    from lunarmag.inversion_helpers import seed_for, MASTER_SEED, ACTIVE_SEED
    assert MASTER_SEED == ACTIVE_SEED == 20260624
    assert seed_for("a06:A1:drive_around") == int(
        __import__("hashlib").sha256(b"20260624:a06:A1:drive_around").hexdigest()[:8], 16)

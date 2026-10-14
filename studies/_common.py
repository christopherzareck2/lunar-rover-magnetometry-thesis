"""Shared setup for the studies in this folder (imports, seeds, statistics, provenance).

The studies import the released simulation code (``engine/`` and ``analysis/``) read-only. By default that is the
code in this checkout. Set ``STUDY_CODE_ROOT`` to a directory that holds an ``engine/`` and ``analysis/`` tree (for
example ``git archive 3498ead engine analysis results``) to pin the studies to an exact commit; every result file records which code root and which file hashes were used.

Seeds: every random stream is drawn from the repository's ``seed_for(label)`` (SHA-256 of ``ACTIVE_SEED:label``).
Study labels start with ``study:`` so they never coincide with a label used by the released analysis scripts,
except where a study deliberately reuses a released label to reproduce a stored number (marked in its JSON).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

STUDIES = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(STUDIES)
CODE_ROOT = os.path.abspath(os.environ.get("STUDY_CODE_ROOT", REPO))
RESULTS = os.path.join(STUDIES, "results")
FIGURES = os.path.join(STUDIES, "figures")

_cwd = os.getcwd()
for sub in ("analysis", "engine"):
    p = os.path.join(CODE_ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)
import a06_isolated_robustness as A6          # noqa: E402  (chdirs into engine/; restored below)
import a09_systematics as A9                  # noqa: E402
import isolated_outcrop_1cm as ISO            # noqa: E402
import correlated_noise_sensitivity as CN     # noqa: E402
from inversion_helpers import (prism_cell_operator, dir_err_deg, seed_for, M_NRM_MIN, SIGMA_B)  # noqa: E402,F401
os.chdir(_cwd)

MU0 = 4e-7 * np.pi
BND = A9.BND
DIRS3 = A9.DIRS
DIRS6 = A9.DIRS6


def rng_for(label: str) -> np.random.Generator:
    return np.random.default_rng(seed_for(label))


# ------------------------------------------------------------------------------------------------ statistics
def mean_se(a) -> dict:
    a = np.asarray(a, float)
    return {"mean": float(a.mean()), "se": float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else None,
            "median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "N": int(a.size)}


def median_se(a, n_boot: int = 2000, seed_label: str = "study:bootstrap") -> float:
    """Bootstrap standard error of the median."""
    a = np.asarray(a, float)
    r = rng_for(seed_label)
    idx = r.integers(0, a.size, size=(n_boot, a.size))
    return float(np.median(a[idx], axis=1).std(ddof=1))


def angle_deg(a, b) -> float:
    return dir_err_deg(a, b)


def angle_of_mean(V, mt, n_boot: int = 2000, seed_label: str = "study:bootstrap:aom") -> dict:
    """Angle between the mean of the vectors V (n x 3) and the truth, with a bootstrap standard error."""
    V = np.asarray(V, float)
    val = angle_deg(V.mean(0), mt)
    r = rng_for(seed_label)
    idx = r.integers(0, len(V), size=(n_boot, len(V)))
    bs = np.array([angle_deg(V[i].mean(0), mt) for i in idx])
    return {"value": float(val), "se_bootstrap": float(bs.std(ddof=1)), "N": int(len(V))}


# ------------------------------------------------------------------------------------------------ provenance
def _sha(path: str) -> str | None:
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()
    except OSError:
        return None


def provenance(script: str, extra: dict | None = None) -> dict:
    def git(*a):
        try:
            return subprocess.check_output(["git", "-C", REPO, *a], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    files = ["engine/core.py", "engine/inversion_helpers.py", "engine/isolated_outcrop_1cm.py",
             "engine/correlated_noise_sensitivity.py", "engine/skatepark_1cm_dr_sweep.py",
             "analysis/a06_isolated_robustness.py", "analysis/a09_systematics.py",
             "analysis/a08_skatepark_neighbor_directions.py"]
    import inversion_helpers as IH
    import simpeg
    d = {"script": os.path.relpath(script, REPO), "repo_head": git("rev-parse", "HEAD"),
         "code_root": CODE_ROOT if CODE_ROOT != REPO else "this checkout",
         "code_sha256": {f: _sha(os.path.join(CODE_ROOT, f)) for f in files},
         "ACTIVE_SEED": int(IH.ACTIVE_SEED), "python": platform.python_version(), "numpy": np.__version__,
         "simpeg": simpeg.__version__, "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if extra:
        d.update(extra)
    return d


def save(name: str, obj: dict) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    os.replace(tmp, path)
    return path


def stored(relpath: str) -> dict:
    """A stored result of the released code (from the pinned code root, so it matches the code used)."""
    p = os.path.join(CODE_ROOT, relpath)
    if not os.path.exists(p):
        p = os.path.join(REPO, relpath)
    return json.load(open(p))

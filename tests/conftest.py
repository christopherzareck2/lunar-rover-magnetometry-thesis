"""Shared pytest setup: make the package, the analysis scripts and the notebook helpers importable.

Run from the repository root:  make test      (fast suite, about a minute)
                               make test-all  (adds the slow tests, several minutes)
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("src", "analysis", "notebook"):
    p = os.path.join(ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("MPLBACKEND", "Agg")

SLOW_NAMES = {"test_rerun_true_redraws_and_matches_stored"}


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.name in SLOW_NAMES:
            item.add_marker(pytest.mark.slow)


def _import_analysis(name):
    """Import a frozen analysis script; they chdir into engine/ at import, so restore the working directory."""
    cwd = os.getcwd()
    try:
        return __import__(name)
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="session")
def A6():
    return _import_analysis("a06_isolated_robustness")


@pytest.fixture(scope="session")
def A9():
    return _import_analysis("a09_systematics")


@pytest.fixture(scope="session")
def stored():
    def load(name):
        for sub in ("analysis", "engine", "notebook"):
            p = os.path.join(ROOT, "results", sub, name)
            if os.path.exists(p):
                with open(p) as f:
                    return json.load(f)
        raise FileNotFoundError(name)
    return load

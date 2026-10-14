"""Compatibility alias: this module moved to ``src/lunarmag/isolated_outcrop_1cm.py`` (package ``lunarmag``).

Scripts that put ``engine/`` on sys.path and run ``import isolated_outcrop_1cm`` keep working unchanged: this file replaces
itself in ``sys.modules`` with ``lunarmag.isolated_outcrop_1cm``, so both names refer to ONE module object (module-level settings,
e.g. ``inversion_helpers.ACTIVE_SEED``, stay shared) and the numerics are those of release v1.0.0.
"""
import os as _os
import sys as _sys

_SRC = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)          # use this checkout's package even when lunarmag is not pip-installed

import lunarmag.isolated_outcrop_1cm as _impl            # noqa: E402

_sys.modules[__name__] = _impl

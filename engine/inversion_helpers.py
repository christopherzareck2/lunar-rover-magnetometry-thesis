"""Compatibility alias: this module moved to ``src/lunarmag/inversion_helpers.py`` (package ``lunarmag``).

Scripts that put ``engine/`` on sys.path and run ``import inversion_helpers`` keep working unchanged: this file replaces
itself in ``sys.modules`` with ``lunarmag.inversion_helpers``, so both names refer to ONE module object (module-level settings,
e.g. ``inversion_helpers.ACTIVE_SEED``, stay shared) and the numerics are those of release v1.0.0.
"""
import os as _os
import sys as _sys

_SRC = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)          # use this checkout's package even when lunarmag is not pip-installed

import lunarmag.inversion_helpers as _impl            # noqa: E402

_sys.modules[__name__] = _impl

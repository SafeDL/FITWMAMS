"""Vendored, minimally patched TrafficBots V1.5 implementation.

The upstream code uses ``models`` and ``utils`` absolute imports.  Registering
this directory once preserves those imports without changing its hypothesis.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

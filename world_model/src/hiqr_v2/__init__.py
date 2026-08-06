"""HiQR-v2: prior-driven hierarchical interaction world model.

This package deliberately has an independent checkpoint contract from the
original :mod:`world_model.src.hiqr` implementation.  V1 remains a read-only
experiment baseline.
"""

from .config import HiQRV2Config
from .environment import (
    BatchedHiQRV2WorldModelEnvironment,
    BatchedHiQRV2WorldSnapshot,
    HiQRV2WorldModelEnvironment,
    HiQRV2WorldSnapshot,
)
from .model import HiQRV2WorldModel

__all__ = (
    "HiQRV2Config",
    "HiQRV2WorldModel",
    "BatchedHiQRV2WorldModelEnvironment",
    "BatchedHiQRV2WorldSnapshot",
    "HiQRV2WorldModelEnvironment",
    "HiQRV2WorldSnapshot",
)

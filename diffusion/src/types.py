"""Shared dataclasses and constants for action diffusion."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np


class EventType(str, Enum):
    FOLLOWING = "following"
    CUT_IN = "cut_in"


STATE_FEATURES: Tuple[str, ...] = ("x", "y", "vx", "vy", "ax", "ay")
NUM_ACTORS = 2
NUM_STATE_FEATURES = len(STATE_FEATURES)

FOLLOWING_ACCEL_ACTION_KEYS: Tuple[str, ...] = ("ax",)
FOLLOWING_JERK_ACTION_KEYS: Tuple[str, ...] = ("jx",)

FOLLOWING_SCENARIO_CONDITION_KEYS: Tuple[str, ...] = (
    "ego_vx_0",
    "initial_gap",
    "initial_delta_v",
    "lead_ax_0",
    "lead_speed_change",
    "lead_min_ax",
    "lead_braking_duration",
)

CUTIN_ACCEL_ACTION_KEYS: Tuple[str, ...] = ("ax", "ay")
CUTIN_SCENARIO_CONDITION_KEYS: Tuple[str, ...] = (
    "ego_vx_0",
    "initial_gap",
    "initial_lateral_offset",
    "initial_delta_vx",
    "target_ax_0",
    "target_vy_0",
    "target_ay_0",
    "final_lateral_offset",
    "time_to_cross",
    "target_speed_change",
)


@dataclass
class VehicleState:
    """Minimal closed-loop vehicle state in the local simulation frame."""

    x: float
    y: float
    vx: float
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    lane_id: int | None = None

    def as_feature(self) -> np.ndarray:
        return np.asarray([self.x, self.y, self.vx, self.vy, self.ax, self.ay], dtype=np.float32)

"""Shared dataclasses and constants for action diffusion."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple

import numpy as np


class EventType(str, Enum):
    FOLLOWING = "following"
    CUT_IN = "cut_in"


STATE_FEATURES: Tuple[str, ...] = ("x", "y", "vx", "vy", "ax", "ay")
NUM_ACTORS = 2
NUM_STATE_FEATURES = len(STATE_FEATURES)

FOLLOWING_ACCEL_ACTION_KEYS: Tuple[str, ...] = ("ax",)
FOLLOWING_JERK_ACTION_KEYS: Tuple[str, ...] = ("jx",)
FOLLOWING_ACTION_KEYS: Tuple[str, ...] = FOLLOWING_ACCEL_ACTION_KEYS
CUTIN_ACTION_KEYS: Tuple[str, ...] = ("ax", "yaw_rate")

FOLLOWING_RELATIVE_HISTORY_KEYS: Tuple[str, ...] = (
    "gap",
    "lateral_offset",
    "delta_v",
    "delta_a",
    "ttc",
    "thw",
)


@dataclass(frozen=True)
class VehicleBox:
    length: float = 4.8
    width: float = 1.8


@dataclass
class VehicleState:
    """Minimal closed-loop vehicle state in the local simulation frame."""

    x: float
    y: float
    vx: float
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    yaw: float = 0.0
    box: VehicleBox = field(default_factory=VehicleBox)
    lane_id: int | None = None

    def as_feature(self) -> np.ndarray:
        return np.asarray([self.x, self.y, self.vx, self.vy, self.ax, self.ay], dtype=np.float32)


@dataclass
class ClosedLoopContext:
    """Current ADS-centred interaction state for rolling generation.

    ``history`` is in ego-current coordinates with shape
    ``[history_steps, actors, state_features]`` and actor order
    ``0=ego/ADS, 1=adversarial vehicle``.
    """

    history: np.ndarray
    ego: VehicleState
    adversary: VehicleState
    event_type: EventType | str
    lane_width: float = 3.75
    target_final_y: float | None = None
    extra: Dict[str, float] | None = None


@dataclass
class CandidatePlan:
    actions: np.ndarray
    trajectory: np.ndarray
    risk: float
    naturalness_cost: float
    violation_cost: float
    score: float
    is_valid: bool

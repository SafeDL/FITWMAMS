"""Shared schema for START/ROLL background traffic world-model data."""
from __future__ import annotations

from dataclasses import dataclass

from normalizing_flow.src.features import (
    DEFAULT_EGO_LENGTH_M,
    DEFAULT_EGO_WIDTH_M,
    DEFAULT_LANE_WIDTH_M,
    DEFAULT_OTHER_LENGTH_M,
    DEFAULT_OTHER_WIDTH_M,
    SLOT_NAMES,
    TRAJECTORY_FEATURES,
)


MODE_NAMES: tuple[str, ...] = ("START", "ROLL")
START_MODE_INDEX = 0
ROLL_MODE_INDEX = 1

EGO_AGENT_NAME = "ego"
AGENT_NAMES: tuple[str, ...] = (EGO_AGENT_NAME, *SLOT_NAMES)

AGENT_STATE_FEATURES: tuple[str, ...] = (
    "x_m",
    "y_left_m",
    "vx_mps",
    "vy_left_mps",
    "ax_mps2",
    "ay_left_mps2",
)

ACTION_FEATURES: tuple[str, ...] = (
    "ax_mps2",
    "ay_left_mps2",
)

FLOW_ACTION_SUMMARY_FEATURES: tuple[str, ...] = TRAJECTORY_FEATURES

RELATION_FEATURES: tuple[str, ...] = (
    "signed_rel_x_m",
    "abs_longitudinal_gap_m",
    "rel_y_left_m",
    "rel_vx_mps",
    "rel_vy_left_mps",
    "closing_speed_mps",
    "ttc_s_clipped",
    "drac_mps2",
    "primary_slot_flag",
    "slot_valid_flag",
)

DEFAULT_FPS = 25.0
DEFAULT_HISTORY_STEPS = 25
DEFAULT_START_HORIZON_STEPS = 25
DEFAULT_ROLL_HORIZON_STEPS = 25


@dataclass(frozen=True)
class WorldModelSchema:
    """Fixed dimensions and feature names stored in dataset_schema.json."""

    fps: float = DEFAULT_FPS
    history_steps: int = DEFAULT_HISTORY_STEPS
    horizon_steps: int = DEFAULT_START_HORIZON_STEPS
    agent_names: tuple[str, ...] = AGENT_NAMES
    slot_names: tuple[str, ...] = SLOT_NAMES
    state_features: tuple[str, ...] = AGENT_STATE_FEATURES
    action_features: tuple[str, ...] = ACTION_FEATURES
    flow_action_summary_features: tuple[str, ...] = FLOW_ACTION_SUMMARY_FEATURES
    relation_features: tuple[str, ...] = RELATION_FEATURES

    @property
    def num_agents(self) -> int:
        return len(self.agent_names)

    @property
    def num_slots(self) -> int:
        return len(self.slot_names)

    @property
    def state_dim(self) -> int:
        return len(self.state_features)

    @property
    def action_dim(self) -> int:
        return len(self.action_features)


def mode_index(mode: str) -> int:
    value = str(mode).upper()
    if value not in MODE_NAMES:
        raise ValueError(f"Unknown world-model mode={mode!r}; expected {MODE_NAMES}")
    return MODE_NAMES.index(value)


def mode_name(index: int) -> str:
    return MODE_NAMES[int(index)]


def slot_index(slot_name: str) -> int:
    if slot_name not in SLOT_NAMES:
        raise ValueError(f"Unknown slot={slot_name!r}; expected {SLOT_NAMES}")
    return SLOT_NAMES.index(slot_name)


def primary_slot_index(primary_slot: str | int | None, slot_mask=None) -> int:
    if primary_slot is None or str(primary_slot) == "none":
        if slot_mask is not None:
            active = [idx for idx, value in enumerate(slot_mask) if bool(value)]
            if active:
                return int(active[0])
        return 0
    if isinstance(primary_slot, int):
        value = int(primary_slot)
        if value < 0 or value >= len(SLOT_NAMES):
            return primary_slot_index(None, slot_mask=slot_mask)
        return value
    value = str(primary_slot)
    if value not in SLOT_NAMES:
        return primary_slot_index(None, slot_mask=slot_mask)
    return SLOT_NAMES.index(value)


PHYSICAL_CONSTANTS = {
    "ego_length_m": DEFAULT_EGO_LENGTH_M,
    "ego_width_m": DEFAULT_EGO_WIDTH_M,
    "other_length_m": DEFAULT_OTHER_LENGTH_M,
    "other_width_m": DEFAULT_OTHER_WIDTH_M,
    "lane_width_m": DEFAULT_LANE_WIDTH_M,
}

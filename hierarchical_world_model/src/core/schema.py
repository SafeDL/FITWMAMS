"""Actor slots and state features in the shared highD sequence cache."""

from __future__ import annotations

from normalizing_flow.src.features import SLOT_NAMES

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


def slot_index(slot_name: str) -> int:
    if slot_name not in SLOT_NAMES:
        raise ValueError(f"Unknown slot={slot_name!r}; expected {SLOT_NAMES}")
    return SLOT_NAMES.index(slot_name)

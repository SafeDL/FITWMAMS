"""Shared relation features derived from a current traffic scene."""
from __future__ import annotations

import numpy as np

from normalizing_flow.src.features import SLOT_NAMES

from .schema import (
    DEFAULT_EGO_LENGTH_M,
    DEFAULT_OTHER_LENGTH_M,
    RELATION_FEATURES,
)


def build_relation_features_from_current(
    current_states: np.ndarray,
    current_valid: np.ndarray,
    *,
    primary_slot_index: int,
    ttc_clip_s: float = 10.0,
    drac_clip_mps2: float = 12.0,
) -> np.ndarray:
    """Compute fixed-slot interaction features without depending on a model variant."""
    current = np.asarray(current_states, dtype=np.float32)
    valid = np.asarray(current_valid, dtype=bool)
    out = np.zeros((len(SLOT_NAMES), len(RELATION_FEATURES)), dtype=np.float32)
    if current.shape[0] < 1 + len(SLOT_NAMES) or not bool(valid[0]):
        return out
    ego = current[0]
    slots = current[1:]
    slot_valid = valid[1:]
    rel_x = slots[:, 0] - ego[0]
    rel_y = slots[:, 1] - ego[1]
    rel_vx = slots[:, 2] - ego[2]
    rel_vy = slots[:, 3] - ego[3]
    abs_gap = np.maximum(
        np.abs(rel_x) - 0.5 * (DEFAULT_EGO_LENGTH_M + DEFAULT_OTHER_LENGTH_M),
        0.0,
    )
    closing_speed = np.maximum(np.where(rel_x >= 0.0, -rel_vx, rel_vx), 0.0)
    ttc = np.full(len(SLOT_NAMES), float(ttc_clip_s), dtype=np.float32)
    closing = closing_speed > 1.0e-3
    ttc[closing] = abs_gap[closing] / np.maximum(closing_speed[closing], 1.0e-3)
    ttc = np.clip(ttc, 0.0, float(ttc_clip_s))
    drac = np.zeros(len(SLOT_NAMES), dtype=np.float32)
    drac[closing] = (closing_speed[closing] ** 2) / np.maximum(2.0 * abs_gap[closing], 1.0e-3)
    drac = np.clip(drac, 0.0, float(drac_clip_mps2))
    primary = np.zeros(len(SLOT_NAMES), dtype=np.float32)
    if 0 <= int(primary_slot_index) < len(SLOT_NAMES):
        primary[int(primary_slot_index)] = 1.0
    out[:, 0] = rel_x
    out[:, 1] = abs_gap
    out[:, 2] = rel_y
    out[:, 3] = rel_vx
    out[:, 4] = rel_vy
    out[:, 5] = closing_speed
    out[:, 6] = ttc
    out[:, 7] = drac
    out[:, 8] = primary
    out[:, 9] = slot_valid.astype(np.float32)
    out[~slot_valid] = 0.0
    return out.astype(np.float32)

"""Physical validity checks shared by generated-scene consumers."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .features import SLOT_NAMES, feature_index


def physical_validity_flags(
    features: np.ndarray,
    slot_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, int], dict[str, np.ndarray]]:
    """Flag non-finite, kinematically implausible and slot-invalid C0 rows."""
    values = np.asarray(features, np.float32)
    slots = np.asarray(slot_mask, bool)
    invalid = ~np.isfinite(values).all(axis=1)
    overlap = np.zeros(len(values), bool)
    negative_gap = np.zeros(len(values), bool)
    semantic = np.zeros(len(values), bool)
    reasons: Counter[str] = Counter()
    ego_vx = values[:, feature_index(None, "ego_vx_mps")]
    ego_vy = values[:, feature_index(None, "ego_vy_left_mps")]
    ego_ax = values[:, feature_index(None, "ego_ax_mps2")]
    ego_ay = values[:, feature_index(None, "ego_ay_left_mps2")]
    bad_ego = (
        (ego_vx < -5)
        | (ego_vx > 70)
        | (np.abs(ego_vy) > 10)
        | (np.abs(ego_ax) > 10)
        | (np.abs(ego_ay) > 5)
    )
    invalid |= bad_ego
    reasons["ego_kinematics_out_of_range"] = int(bad_ego.sum())
    for index, name in enumerate(SLOT_NAMES):
        active = slots[:, index]
        dx = values[:, feature_index(name, "rel_x_m")]
        dy = values[:, feature_index(name, "rel_y_left_m")]
        dvx = values[:, feature_index(name, "rel_vx_mps")]
        ax = values[:, feature_index(name, "other_ax_mps2")]
        ay = values[:, feature_index(name, "other_ay_left_mps2")]
        bad_motion = active & (
            (ego_vx + dvx < -10)
            | (ego_vx + dvx > 75)
            | (np.abs(ax) > 10)
            | (np.abs(ay) > 5)
        )
        bad_semantic = active & (
            (("front" in name) & (dx <= 0))
            | (("rear" in name) & (dx >= 0))
            | (name.startswith("left") & (dy <= 0))
            | (name.startswith("right") & (dy >= 0))
        )
        gap = active & (np.abs(dx) <= 4.8)
        box_overlap = gap & (np.abs(dy) <= 1.9)
        invalid |= bad_motion | bad_semantic | box_overlap
        semantic |= bad_semantic
        negative_gap |= gap
        overlap |= box_overlap
        reasons["slot_kinematics_out_of_range"] += int(bad_motion.sum())
        reasons["slot_semantic_invalid"] += int(bad_semantic.sum())
        reasons["vehicle_box_overlap"] += int(box_overlap.sum())
    return invalid, dict(reasons), {
        "overlap": overlap,
        "negative_longitudinal_gap": negative_gap,
        "slot_semantic_invalid": semantic,
    }

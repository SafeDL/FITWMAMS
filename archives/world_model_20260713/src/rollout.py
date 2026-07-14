"""Action integration and Flow START adapters for background traffic rollout."""
from __future__ import annotations

from typing import Any

import numpy as np

from normalizing_flow.src.features import (
    EGO_FEATURES,
    SLOT_NAMES,
    slot_feature_index,
    trajectory_feature_index,
)
from normalizing_flow.src.metrics import physical_validity_flags

from .schema import (
    AGENT_STATE_FEATURES,
    DEFAULT_EGO_LENGTH_M,
    DEFAULT_OTHER_LENGTH_M,
    FLOW_ACTION_SUMMARY_FEATURES,
    RELATION_FEATURES,
    START_MODE_INDEX,
)
from .utils import normalize_with_mask


def _normalizer(schema: dict[str, Any], name: str) -> tuple[np.ndarray, np.ndarray]:
    norm = schema["normalization"][name]
    return np.asarray(norm["mean"], dtype=np.float32), np.asarray(norm["std"], dtype=np.float32)


def unnormalize_actions(actions_normalized: np.ndarray, schema: dict[str, Any]) -> np.ndarray:
    mean, std = _normalizer(schema, "action")
    return (np.asarray(actions_normalized, dtype=np.float32) * std + mean).astype(np.float32)


def normalize_states(states: np.ndarray, valid: np.ndarray, schema: dict[str, Any]) -> np.ndarray:
    mean, std = _normalizer(schema, "state")
    mask = np.broadcast_to(np.asarray(valid, dtype=bool)[..., None], np.shape(states))
    return normalize_with_mask(states, mask, mean, std)


def normalize_flow_summary(summary: np.ndarray, valid: np.ndarray, schema: dict[str, Any]) -> np.ndarray:
    mean, std = _normalizer(schema, "flow_action_summary")
    return normalize_with_mask(summary, valid, mean, std)


def normalize_relation_features(features: np.ndarray, valid: np.ndarray, schema: dict[str, Any]) -> np.ndarray:
    mean, std = _normalizer(schema, "relation_features")
    mask = np.broadcast_to(np.asarray(valid, dtype=bool)[..., None], np.shape(features))
    return normalize_with_mask(features, mask, mean, std)


def build_relation_features_from_current(
    current_states: np.ndarray,
    current_valid: np.ndarray,
    *,
    primary_slot_index: int,
    ttc_clip_s: float = 10.0,
    drac_clip_mps2: float = 12.0,
) -> np.ndarray:
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


def integrate_background_actions(
    current_states: np.ndarray,
    current_valid: np.ndarray,
    actions: np.ndarray,
    *,
    dt: float,
    ax_min: float = -8.0,
    ax_max: float = 4.0,
    ay_abs_max: float = 4.0,
    speed_min_mps: float = 0.0,
    speed_max_mps: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate `[ax, ay_left]` background actions with conservative kinematics.

    Args:
        current_states: `[7, 6]` raw states, ego plus six slots.
        current_valid: `[7]` mask.
        actions: `[K, 6, 2]` raw actions for background slots.

    Returns:
        future background states `[K, 6, 6]` and valid mask `[K, 6]`.
    """
    states, valid = integrate_background_actions_batch(
        np.asarray(current_states, dtype=np.float32)[None],
        np.asarray(current_valid, dtype=bool)[None],
        np.asarray(actions, dtype=np.float32)[None],
        dt=dt,
        ax_min=ax_min,
        ax_max=ax_max,
        ay_abs_max=ay_abs_max,
        speed_min_mps=speed_min_mps,
        speed_max_mps=speed_max_mps,
    )
    return states[0], valid[0]


def integrate_background_actions_batch(
    current_states: np.ndarray,
    current_valid: np.ndarray,
    actions: np.ndarray,
    *,
    dt: float,
    ax_min: float = -8.0,
    ax_max: float = 4.0,
    ay_abs_max: float = 4.0,
    speed_min_mps: float = 0.0,
    speed_max_mps: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized batch version of `integrate_background_actions`.

    Args:
        current_states: `[N, 7, 6]`.
        current_valid: `[N, 7]`.
        actions: `[N, K, 6, 2]`.
    """
    current_states = np.asarray(current_states, dtype=np.float32)
    current_valid = np.asarray(current_valid, dtype=bool)
    actions = np.asarray(actions, dtype=np.float32)
    n = int(actions.shape[0])
    k = int(actions.shape[1])
    states = np.zeros((n, k, len(SLOT_NAMES), len(AGENT_STATE_FEATURES)), dtype=np.float32)
    valid = np.repeat(current_valid[:, None, 1:], k, axis=1).astype(bool)
    bg = current_states[:, 1:, :].copy()
    dt = max(float(dt), 1.0e-6)
    for step in range(k):
        ax = np.clip(actions[:, step, :, 0], float(ax_min), float(ax_max))
        ay = np.clip(actions[:, step, :, 1], -float(ay_abs_max), float(ay_abs_max))
        bg[:, :, 0] = bg[:, :, 0] + bg[:, :, 2] * dt + 0.5 * ax * dt * dt
        bg[:, :, 1] = bg[:, :, 1] + bg[:, :, 3] * dt + 0.5 * ay * dt * dt
        bg[:, :, 2] = np.clip(bg[:, :, 2] + ax * dt, float(speed_min_mps), float(speed_max_mps))
        bg[:, :, 3] = bg[:, :, 3] + ay * dt
        bg[:, :, 4] = ax
        bg[:, :, 5] = ay
        states[:, step] = bg
    states[~valid] = 0.0
    return states.astype(np.float32), valid


def build_start_condition_from_flow_feature(
    feature_row: np.ndarray,
    slot_mask_row: np.ndarray,
    *,
    primary_slot_index: int,
    schema: dict[str, Any],
) -> dict[str, np.ndarray]:
    """将一个完整的 Flow 场景样本转换为模型的 START 张量。

    连续 76 维特征、slot mask 和主交互车辆槽位共同构成 Flow 的场景样本。
    强制传入离散主槽位，使 START 条件完全由 Flow 样本唯一确定。
    """
    feature_row = np.asarray(feature_row, dtype=np.float32)
    slot_mask_row = np.asarray(slot_mask_row, dtype=bool)
    history_steps = int(schema["history_steps"])
    state_dim = len(AGENT_STATE_FEATURES)
    current = np.zeros((1 + len(SLOT_NAMES), state_dim), dtype=np.float32)
    valid = np.zeros(1 + len(SLOT_NAMES), dtype=bool)
    ego = {
        name: float(feature_row[EGO_FEATURES.index(name)])
        for name in EGO_FEATURES
    }
    current[0] = np.asarray(
        [
            0.0,
            0.0,
            ego["ego_vx_mps"],
            ego["ego_vy_left_mps"],
            ego["ego_ax_mps2"],
            ego["ego_ay_left_mps2"],
        ],
        dtype=np.float32,
    )
    valid[0] = True
    flow_summary = np.zeros((len(SLOT_NAMES), len(FLOW_ACTION_SUMMARY_FEATURES)), dtype=np.float32)
    flow_summary_valid = np.zeros_like(flow_summary, dtype=bool)
    for slot_idx, slot_name in enumerate(SLOT_NAMES):
        if not bool(slot_mask_row[slot_idx]):
            continue
        rel_x = float(feature_row[slot_feature_index(slot_name, "rel_x_m")])
        rel_y = float(feature_row[slot_feature_index(slot_name, "rel_y_left_m")])
        rel_vx = float(feature_row[slot_feature_index(slot_name, "rel_vx_mps")])
        rel_vy = float(feature_row[slot_feature_index(slot_name, "rel_vy_left_mps")])
        ax = float(feature_row[slot_feature_index(slot_name, "other_ax_mps2")])
        ay = float(feature_row[slot_feature_index(slot_name, "other_ay_left_mps2")])
        current[slot_idx + 1] = np.asarray(
            [
                rel_x,
                rel_y,
                ego["ego_vx_mps"] + rel_vx,
                ego["ego_vy_left_mps"] + rel_vy,
                ax,
                ay,
            ],
            dtype=np.float32,
        )
        valid[slot_idx + 1] = True
        for feat_idx, feat_name in enumerate(FLOW_ACTION_SUMMARY_FEATURES):
            flow_summary[slot_idx, feat_idx] = float(
                feature_row[trajectory_feature_index(slot_name, feat_name)]
            )
        flow_summary_valid[slot_idx, :] = True

    history = np.zeros((history_steps, 1 + len(SLOT_NAMES), state_dim), dtype=np.float32)
    history_valid = np.zeros((history_steps, 1 + len(SLOT_NAMES)), dtype=bool)
    history[-1] = current
    history_valid[-1] = valid
    relation = build_relation_features_from_current(
        current,
        valid,
        primary_slot_index=int(primary_slot_index),
    )
    relation_valid = valid[1:]

    return {
        "history_states": history,
        "history_valid": history_valid,
        "history_states_normalized": normalize_states(history, history_valid, schema),
        "current_states": current,
        "current_valid": valid,
        "current_states_normalized": normalize_states(current, valid, schema),
        "mode_index": np.asarray(START_MODE_INDEX, dtype=np.int64),
        "primary_slot_index": np.asarray(int(primary_slot_index), dtype=np.int64),
        "flow_action_summary": flow_summary,
        "flow_action_summary_valid": flow_summary_valid,
        "flow_action_summary_normalized": normalize_flow_summary(
            flow_summary,
            flow_summary_valid,
            schema,
        ),
        "relation_features": relation,
        "relation_features_normalized": normalize_relation_features(
            relation,
            relation_valid,
            schema,
        ),
        "slot_mask": slot_mask_row.astype(bool),
    }


def c0_is_physically_valid(feature_row: np.ndarray, slot_mask_row: np.ndarray) -> tuple[bool, dict[str, int]]:
    invalid, reasons, _detail = physical_validity_flags(
        np.asarray(feature_row, dtype=np.float32).reshape(1, -1),
        np.asarray(slot_mask_row, dtype=bool).reshape(1, -1),
    )
    return (not bool(invalid[0])), reasons

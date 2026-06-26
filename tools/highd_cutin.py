"""Shared highD cut-in reconstruction and event-level risk scoring."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diffusion.src.data import (
    _build_world_states,
    _vehicle_length_from_meta,
)
from diffusion.src.features import extract_scenario_condition
from diffusion.src.scenario_frame import compute_ego_frame, world_to_ego_states
from tools.risk import (
    longitudinal_proxy_from_series,
    longitudinal_series_from_states,
    resolve_risk_scoring,
    softmax_pool_np,
)

from .highd_longitudinal import (
    DEFAULT_HIGHD_LONGITUDINAL_CONFIG,
    PASSENGER_CAR_CLASS,
    _optional_frame_value,
    _safe_frame_value,
)


logger = logging.getLogger(__name__)


HIGHD_CUTIN_SCORE_KEYS = (
    "recording_id",
    "event_id",
    "ego_id",
    "target_id",
    "start_frame",
    "end_frame",
    "anchor_frame",
    "context_end_frame",
    "context_length",
    "context_horizon_steps",
    "pre_cross_steps",
    "post_cross_steps",
    "risk_start_frame",
    "risk_start_index",
    "cross_frame",
    "cutin_start_frame",
    "cutin_end_frame",
    "source_lane",
    "target_lane",
    "ego_length",
    "adv_length",
    "initial_gap",
    "initial_closing_speed",
    "recorded_min_gap",
    "recorded_min_ttc",
    "completion_gap",
    "post_cutin_min_gap",
    "post_cutin_min_ttc",
    "cutin_gap",
    "cutin_ttc",
    "cutin_time_headway",
    "cutin_lateral_time_gap",
    "max_post_cutin_drac",
    "cutin_duration_seconds",
    "cross_lateral_offset",
    "min_abs_lateral_offset",
    "max_abs_lateral_velocity",
    "max_lateral_approach_speed",
    "final_abs_lateral_offset",
    "cutin_safety_risk_score",
    "post_longitudinal_risk_score",
    "ltg_risk_score",
    "y_long",
    "is_cutin",
    "is_front_cutin",
    "collision",
    "near_collision",
    "y_cutin",
)


DEFAULT_HIGHD_CUTIN_CONFIG: dict[str, Any] = {
    **DEFAULT_HIGHD_LONGITUDINAL_CONFIG,
    "history_steps": 25,
    "context_horizon_steps": 100,
    "context_pre_cross_steps": (15, 20, 25, 30, 35, 45, 50),
    "min_post_cross_steps": 50,
    "lateral_overlap_threshold": 1.0,
    "cutin_lateral_offset": 1.0,
    "min_lateral_approach_speed": 0.05,
    "semantic_window_seconds": 6.0,
    "ltg_window_steps": 5,
    "ltg_weight": 0.2,
    "ltg_scale": 1.0,
    "ltg_eps": 0.2,
    "require_front_at_cutin": True,
    "min_cutin_front_gap": 0.0,
    "require_cutin_for_failure": True,
    "non_cutin_y_cutin": 0.0,
}


def highd_cutin_options(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    options = dict(DEFAULT_HIGHD_CUTIN_CONFIG)
    if overrides:
        options.update(overrides)
    return options


def highd_cutin_options_from_config(
    config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = highd_cutin_options(overrides)
    sampling = config.get("sampling", {})
    filters = config.get("filters", {})
    cutin_event = config.get("cutin", {})
    cutin_risk = config.get("cutin_risk", {})
    for source, mapping in (
        (sampling, {"target_fps": "target_fps"}),
        (
            cutin_event,
            {
                "context_history_steps": "history_steps",
                "context_horizon_steps": "context_horizon_steps",
                "context_pre_cross_steps": "context_pre_cross_steps",
                "min_future_steps": "min_future_steps",
                "min_post_cutin_duration_steps": "min_post_cross_steps",
            },
        ),
        (
            filters,
            {
                "max_abs_accel": "max_abs_accel",
                "max_abs_jerk": "max_abs_jerk",
                "max_position_jump": "max_position_jump",
                "min_vehicle_speed": "min_vehicle_speed",
                "require_passenger_car_ego": "require_passenger_car_ego",
                "require_passenger_car_lead": "require_passenger_car_lead",
            },
        ),
        (
            cutin_risk,
            {
                "lateral_overlap_threshold": "lateral_overlap_threshold",
                "cutin_lateral_offset": "cutin_lateral_offset",
                "min_lateral_approach_speed": "min_lateral_approach_speed",
                "semantic_window_seconds": "semantic_window_seconds",
                "ltg_window_steps": "ltg_window_steps",
                "ltg_weight": "ltg_weight",
                "ltg_scale": "ltg_scale",
                "ltg_eps": "ltg_eps",
                "require_front_at_cutin": "require_front_at_cutin",
                "min_cutin_front_gap": "min_cutin_front_gap",
                "require_cutin_for_failure": "require_cutin_for_failure",
                "non_cutin_y_cutin": "non_cutin_y_cutin",
            },
        ),
    ):
        for source_key, target_key in mapping.items():
            if source_key in source:
                options[target_key] = source[source_key]
    if "min_post_cutin_duration_seconds" in cutin_event:
        fps = max(float(options.get("target_fps", 25)), 1.0e-6)
        options["min_post_cross_steps"] = int(
            np.ceil(float(cutin_event["min_post_cutin_duration_seconds"]) * fps)
        )
    return options


def _completion_frame(row: pd.Series) -> int:
    anchor = _safe_frame_value(row, "anchor_frame", 0)
    cross = _safe_frame_value(row, "cross_frame", anchor)
    return _safe_frame_value(row, "cutin_end_frame", cross)


def _normalize_pre_cross_steps(value: Any) -> tuple[int, ...]:
    raw = value if isinstance(value, (list, tuple)) else [value]
    offsets = tuple(sorted({int(item) for item in raw}))
    if not offsets or offsets[0] < 0:
        raise ValueError("cut-in pre-cross steps must contain non-negative integers")
    return offsets


def _choose_cutin_fixed_horizon_bounds(
    row: pd.Series,
    *,
    pre_cross_steps: int | list[int] | tuple[int, ...],
    horizon_steps: int,
    min_post_cross_steps: int,
) -> tuple[int, int, int] | None:
    """Choose fixed-horizon cut-in bounds aligned to diffusion windows."""
    event_start = _safe_frame_value(row, "start_frame", 0)
    event_end = _safe_frame_value(row, "end_frame", event_start)
    cross = _optional_frame_value(row, "cross_frame")
    if cross is None:
        return None
    horizon = int(horizon_steps)
    if horizon <= 0:
        return None
    required_end = int(cross) + max(int(min_post_cross_steps), 0)
    anchors: list[int] = []
    for offset in _normalize_pre_cross_steps(pre_cross_steps):
        anchor = int(cross) - int(offset)
        context_end = int(anchor) + horizon
        if anchor < event_start or context_end > event_end:
            continue
        if anchor > int(cross) or context_end < required_end:
            continue
        anchors.append(anchor)
    if not anchors:
        return None
    event_id = str(row.get("event_id", ""))
    digest = hashlib.sha256(event_id.encode("utf-8")).digest()
    anchor = anchors[int.from_bytes(digest[:8], "little") % len(anchors)]
    context_end = int(anchor) + horizon
    return int(anchor), int(context_end), int(cross) - int(anchor)


def _cutin_fixed_horizon_context(
    recording: Any,
    row: pd.Series,
    pre_cross_steps: int | list[int] | tuple[int, ...],
    horizon_steps: int,
    min_post_cross_steps: int,
) -> dict[str, Any] | None:
    """Build a fixed-horizon cut-in context for EVT scoring and tail selection."""
    cross = _optional_frame_value(row, "cross_frame")
    cutin_start = _optional_frame_value(row, "cutin_start_frame")
    cutin_end = _optional_frame_value(row, "cutin_end_frame")
    if cross is None or cutin_start is None or cutin_end is None:
        return None

    chosen = _choose_cutin_fixed_horizon_bounds(
        row,
        pre_cross_steps=pre_cross_steps,
        horizon_steps=horizon_steps,
        min_post_cross_steps=min_post_cross_steps,
    )
    if chosen is None:
        return None
    anchor, context_end, selected_pre_cross_steps = chosen
    frames = np.arange(anchor, context_end + 1, dtype=np.int64)
    if len(frames) <= 1:
        return None
    states = _build_world_states(recording, row, frames)
    if states is None:
        return None

    ego_len = _vehicle_length_from_meta(recording.tracks_meta, int(row["ego_id"]))
    adv_len = _vehicle_length_from_meta(recording.tracks_meta, int(row["target_id"]))
    ego_frame = compute_ego_frame(states[0, 0])
    local = world_to_ego_states(states, ego_frame).astype(np.float32)
    initial_states = local[0]
    future_states = local[1:]
    metadata = {
        "anchor_frame": int(anchor),
        "cross_frame": int(cross),
        "cutin_start_frame": int(cutin_start),
        "cutin_end_frame": int(cutin_end),
    }
    scenario_conditions, _keys = extract_scenario_condition(
        initial_states,
        future_states,
        ego_len,
        adv_len,
        event_type="cut_in",
        dt=1.0 / max(float(recording.recording_meta.get("frameRate", 25)), 1.0),
        metadata=metadata,
    )
    return {
        "scenario_conditions": scenario_conditions.astype(np.float32),
        "initial_states": initial_states.astype(np.float32),
        "future_states": future_states.astype(np.float32),
        "local_states": local.astype(np.float32),
        "world_states": states.astype(np.float32),
        "frames": frames.astype(np.int64),
        "ego_length": float(ego_len),
        "adv_length": float(adv_len),
        "anchor_frame": int(anchor),
        "context_end_frame": int(context_end),
        "context_length": int(len(frames)),
        "context_horizon_steps": int(horizon_steps),
        "pre_cross_steps": int(selected_pre_cross_steps),
        "post_cross_steps": int(context_end) - int(cross),
        "risk_start_frame": int(cross),
        "risk_start_index": max(0, int(cross) - int(anchor) - 1),
    }


def _cutin_raw_motion_metrics(
    recording: Any,
    row: pd.Series,
    *,
    fps: float,
    ego_length: float,
    adv_length: float,
) -> dict[str, float]:
    ego_id = int(row["ego_id"])
    target_id = int(row["target_id"])
    anchor = _safe_frame_value(row, "anchor_frame", 0)
    start = _safe_frame_value(row, "cutin_start_frame", anchor)
    end = _safe_frame_value(row, "cutin_end_frame", anchor)
    cross = _safe_frame_value(row, "cross_frame", anchor)
    frames = np.arange(min(start, end), max(start, end) + 1, dtype=np.int64)
    try:
        ego = recording.get_vehicle_track(ego_id)
        target = recording.get_vehicle_track(target_id)
        common = [
            int(frame)
            for frame in frames
            if frame in ego.index and frame in target.index
        ]
    except KeyError:
        common = []
    if common:
        ego_sub = ego.loc[common]
        target_sub = target.loc[common]
        lateral = target_sub["y"].astype(float).to_numpy() - ego_sub["y"].astype(float).to_numpy()
        if "yVelocity" in target_sub.columns:
            rel_vy = target_sub["yVelocity"].astype(float).to_numpy()
            if "yVelocity" in ego_sub.columns:
                rel_vy = rel_vy - ego_sub["yVelocity"].astype(float).to_numpy()
        else:
            rel_vy = np.zeros(len(common), dtype=np.float64)
        min_abs_lateral = float(np.min(np.abs(lateral)))
        max_abs_vy = float(np.max(np.abs(rel_vy)))
    else:
        min_abs_lateral = float("nan")
        max_abs_vy = float("nan")

    try:
        ego_cross = recording.get_vehicle_track(ego_id).loc[cross]
        target_cross = recording.get_vehicle_track(target_id).loc[cross]
        cross_lateral = float(target_cross["y"] - ego_cross["y"])
    except KeyError:
        cross_lateral = float("nan")

    completion = _completion_frame(row)
    try:
        ego_end = recording.get_vehicle_track(ego_id).loc[completion]
        target_end = recording.get_vehicle_track(target_id).loc[completion]
        completion_gap = float(target_end["x"] - ego_end["x"]) - 0.5 * (
            float(ego_length) + float(adv_length)
        )
    except KeyError:
        completion_gap = float("nan")

    duration_steps = max(0, int(end) - int(start))
    duration_seconds = float(duration_steps / max(float(fps), 1.0e-6))
    return {
        "completion_gap": completion_gap,
        "cutin_duration_seconds": duration_seconds,
        "cross_lateral_offset": cross_lateral,
        "min_abs_lateral_offset": min_abs_lateral,
        "max_abs_lateral_velocity": max_abs_vy,
    }


def cutin_risk_from_series(
    *,
    series: dict[str, np.ndarray],
    lateral_offset: np.ndarray,
    config: dict[str, Any],
    scoring_section: str,
    dt: float,
    min_ego_accel: float | None = None,
    risk_start_index: int | None = None,
) -> dict[str, float]:
    """Compute semantic cut-in risk from aligned longitudinal/lateral series.

    The lateral trajectory only gates the event and locates the cut-in moment.
    Longitudinal risk is scored with the shared following formula from the
    cut-in moment through the end of the event context. Lateral LTG risk is
    scored only over a short post cut-in window.
    """
    cfg = config.get("closed_loop_risk", {})
    cutin_cfg = config.get("cutin_risk", {})
    lateral = np.asarray(lateral_offset, dtype=np.float64)
    if lateral.size == 0:
        abs_lateral = np.asarray([], dtype=np.float64)
        lateral_speed = np.asarray([], dtype=np.float64)
        approach_speed = np.asarray([], dtype=np.float64)
    else:
        abs_lateral = np.abs(lateral)
        lateral_speed = np.diff(lateral, prepend=lateral[0]) / max(float(dt), 1.0e-6)
        abs_lateral_speed = (
            np.diff(abs_lateral, prepend=abs_lateral[0]) / max(float(dt), 1.0e-6)
        )
        approach_speed = np.maximum(-abs_lateral_speed, 0.0)

    overlap_threshold = max(
        float(
            cutin_cfg.get(
                "lateral_overlap_threshold",
                cutin_cfg.get("cutin_lateral_offset", 1.0),
            )
        ),
        1.0e-6,
    )
    cutin_lateral_offset = max(
        float(cutin_cfg.get("cutin_lateral_offset", overlap_threshold)),
        1.0e-6,
    )
    min_approach_speed = max(
        float(cutin_cfg.get("min_lateral_approach_speed", 0.05)),
        0.0,
    )
    lengths = [
        len(np.asarray(value))
        for value in series.values()
        if hasattr(value, "__len__")
    ]
    n = min([int(abs_lateral.size), *lengths]) if lengths else int(abs_lateral.size)
    if n <= 0:
        n = 0
    abs_lateral = abs_lateral[:n]
    lateral_speed = lateral_speed[:n]
    approach_speed = approach_speed[:n]
    min_abs_lateral = (
        float(np.min(abs_lateral)) if abs_lateral.size else float("inf")
    )
    final_abs_lateral = (
        float(abs_lateral[-1]) if abs_lateral.size else float("inf")
    )
    max_abs_lateral_velocity = (
        float(np.max(np.abs(lateral_speed))) if lateral_speed.size else 0.0
    )
    max_lateral_approach_speed = (
        float(np.max(approach_speed)) if approach_speed.size else 0.0
    )
    initial_abs_lateral = (
        float(abs_lateral[0]) if abs_lateral.size else float("inf")
    )
    overlap_mask = abs_lateral <= overlap_threshold
    has_overlap = bool(np.any(overlap_mask))
    if has_overlap:
        semantic_index = int(np.flatnonzero(overlap_mask)[0])
    elif n > 0:
        semantic_index = int(np.argmin(abs_lateral))
    else:
        semantic_index = 0
    if risk_start_index is None:
        cutin_index = semantic_index
    elif n > 0:
        cutin_index = int(np.clip(int(risk_start_index), 0, n - 1))
    else:
        cutin_index = 0

    semantic_steps = max(
        1,
        int(
            np.ceil(
                float(cutin_cfg.get("semantic_window_seconds", 6.0))
                / max(float(dt), 1.0e-6)
            )
        ),
    )
    semantic_end = min(n, semantic_index + semantic_steps)
    if n > 0 and semantic_end > semantic_index:
        post_cutin_max_abs_lateral = float(
            np.max(abs_lateral[semantic_index:semantic_end])
        )
    else:
        post_cutin_max_abs_lateral = float("inf")
    aligned_series = {
        key: np.asarray(value, dtype=np.float64)[:n]
        for key, value in series.items()
        if hasattr(value, "__len__")
    }

    def _series_value(key: str, default: float) -> np.ndarray:
        value = aligned_series.get(key)
        if value is None or value.size == 0:
            return np.full(n, default, dtype=np.float64)
        return np.asarray(value, dtype=np.float64)

    gap_all = _series_value("gap", float("inf"))
    cutin_gap_for_gate = float(gap_all[cutin_index]) if n else float("inf")
    requires_front = bool(cutin_cfg.get("require_front_at_cutin", True))
    is_front_cutin = (
        cutin_gap_for_gate >= float(cutin_cfg.get("min_cutin_front_gap", 0.0))
    )
    if initial_abs_lateral > cutin_lateral_offset:
        approach_end = semantic_index + 1 if has_overlap else n
        approach_window = approach_speed[:approach_end]
        has_approach = bool(
            approach_window.size
            and float(np.max(approach_window)) >= min_approach_speed
        )
    else:
        # The rollout may start at the lane-crossing instant. In that case the
        # pre-cross approach is in the history, so require the vehicle to remain
        # in the target lane instead of moving back out.
        has_approach = True
    remains_in_target_lane = post_cutin_max_abs_lateral <= cutin_lateral_offset
    is_cutin = bool(
        has_overlap
        and has_approach
        and remains_in_target_lane
        and (is_front_cutin or not requires_front)
    )

    ttc_all = _series_value("ttc", 1000.0)
    thw_all = _series_value("thw", 1000.0)
    drac_all = _series_value("drac", 0.0)
    ego_accel_all = _series_value("ego_accel", 0.0)

    post_slice = slice(cutin_index, n)
    post_gap = gap_all[post_slice] if n else np.asarray([], dtype=np.float64)
    post_ttc = ttc_all[post_slice] if n else np.asarray([], dtype=np.float64)
    post_thw = thw_all[post_slice] if n else np.asarray([], dtype=np.float64)
    post_drac = drac_all[post_slice] if n else np.asarray([], dtype=np.float64)
    post_accel = ego_accel_all[post_slice] if n else np.asarray([], dtype=np.float64)
    min_gap = float(np.min(post_gap)) if post_gap.size else float("inf")
    min_post_ttc = float(np.min(post_ttc)) if post_ttc.size else 1000.0
    max_post_drac = float(np.max(post_drac)) if post_drac.size else 0.0
    collision = float(post_gap.size > 0 and min_gap <= 0.0)
    near_gap_threshold = float(cfg.get("near_collision_gap", 2.0))
    near_collision = float(post_gap.size > 0 and min_gap < near_gap_threshold)
    if post_accel.size:
        min_ego_accel_value = float(np.min(post_accel))
    elif min_ego_accel is not None:
        min_ego_accel_value = float(min_ego_accel)
    else:
        min_ego_accel_value = 0.0
    hard_brake_threshold = float(cfg.get("hard_brake_threshold", -4.0))
    hard_brake = max(
        0.0,
        hard_brake_threshold - min_ego_accel_value,
    ) / max(abs(hard_brake_threshold), 1.0e-6)
    collision_score = float(cfg.get("collision_bonus", 5.0)) * collision
    near_score = float(cfg.get("near_collision_weight", 1.0)) * near_collision
    hard_score = float(cfg.get("hard_brake_weight", 1.0)) * hard_brake

    scoring = resolve_risk_scoring(config, scoring_section)
    longitudinal_proxy = longitudinal_proxy_from_series(
        {
            "gap": post_gap,
            "ttc": post_ttc,
            "thw": post_thw,
            "drac": post_drac,
        },
        config,
        scoring_section=scoring_section,
    )
    cutin_gap = float(gap_all[cutin_index]) if n else float("inf")
    cutin_ttc = float(ttc_all[cutin_index]) if n else 1000.0
    cutin_thw = float(thw_all[cutin_index]) if n else 1000.0
    ltg_window_steps = max(1, int(cutin_cfg.get("ltg_window_steps", 5)))
    ltg_end = min(n, cutin_index + ltg_window_steps)
    ltg_lateral = (
        abs_lateral[cutin_index:ltg_end]
        if n and ltg_end > cutin_index
        else np.asarray([], dtype=np.float64)
    )
    ltg_approach = (
        approach_speed[cutin_index:ltg_end]
        if approach_speed.size and ltg_end > cutin_index
        else np.asarray([], dtype=np.float64)
    )
    ltg_denom = np.maximum(
        ltg_approach,
        max(float(cutin_cfg.get("min_lateral_approach_speed", 0.05)), 1.0e-6),
    )
    if ltg_lateral.size and ltg_denom.size:
        ltg = ltg_lateral / ltg_denom
        ltg_inverse = 1.0 / np.maximum(
            ltg,
            max(float(cutin_cfg.get("ltg_eps", 0.2)), 1.0e-6),
        )
        ltg_objective = softmax_pool_np(ltg_inverse, scoring["pool_beta"]) / max(
            float(cutin_cfg.get("ltg_scale", 1.0)),
            1.0e-6,
        )
        cutin_ltg = float(np.min(ltg))
    else:
        ltg_objective = 0.0
        cutin_ltg = float("inf")

    ttc_objective = float(longitudinal_proxy["ttc_objective"])
    thw_objective = float(longitudinal_proxy["thw_objective"])
    gap_objective = float(longitudinal_proxy["gap_objective"])
    drac_objective = float(longitudinal_proxy["drac_objective"])
    ttc_score = float(longitudinal_proxy["ttc_score"])
    thw_score = float(longitudinal_proxy["thw_score"])
    gap_score = float(longitudinal_proxy["gap_score"])
    drac_score = float(longitudinal_proxy["drac_score"])
    longitudinal_score = float(longitudinal_proxy["proxy_risk_score"])
    ltg_score = float(cutin_cfg.get("ltg_weight", 1.0)) * ltg_objective
    safety_score = longitudinal_score + ltg_score
    y_long = collision_score + near_score + hard_score + longitudinal_score
    risk_value = y_long + ltg_score
    require_cutin = bool(
        cutin_cfg.get("require_cutin_for_failure", True)
    )
    y_cutin = (
        risk_value
        if (is_cutin or not require_cutin)
        else float(cutin_cfg.get("non_cutin_y_cutin", 0.0))
    )
    return {
        "y_cutin": float(y_cutin),
        "y_long": float(y_long),
        "cutin_safety_risk_score": float(safety_score),
        "post_longitudinal_risk_score": float(longitudinal_score),
        "collision": collision,
        "near_collision": near_collision,
        "collision_risk_score": float(collision_score),
        "near_collision_risk_score": float(near_score),
        "hard_brake_severity": float(hard_brake),
        "hard_brake_risk_score": float(hard_score),
        "ttc_objective": float(ttc_objective),
        "thw_objective": float(thw_objective),
        "drac_objective": float(drac_objective),
        "gap_objective": float(gap_objective),
        "ltg_objective": float(ltg_objective),
        "ttc_risk_score": float(ttc_score),
        "thw_risk_score": float(thw_score),
        "drac_risk_score": float(drac_score),
        "gap_risk_score": float(gap_score),
        "ltg_risk_score": float(ltg_score),
        "cutin_gap": float(cutin_gap),
        "cutin_ttc": float(cutin_ttc),
        "cutin_time_headway": float(cutin_thw),
        "cutin_lateral_time_gap": float(cutin_ltg),
        "min_post_cutin_gap": float(min_gap),
        "min_post_cutin_ttc": float(min_post_ttc),
        "max_post_cutin_drac": float(max_post_drac),
        "min_abs_lateral_offset": float(min_abs_lateral),
        "final_abs_lateral_offset": float(final_abs_lateral),
        "max_abs_lateral_velocity": float(max_abs_lateral_velocity),
        "max_lateral_approach_speed": float(max_lateral_approach_speed),
        "is_cutin": float(is_cutin),
        "is_front_cutin": float(is_front_cutin),
        "require_front_at_cutin": float(requires_front),
        "min_cutin_front_gap": float(cutin_cfg.get("min_cutin_front_gap", 0.0)),
        "lateral_overlap_fraction": float(np.mean(overlap_mask)) if n else 0.0,
        "lateral_overlap_threshold": float(overlap_threshold),
        "risk_start_index": float(cutin_index),
        "semantic_cutin_index": float(semantic_index),
    }


def _cutin_metrics(
    recording: Any,
    row: pd.Series,
    context: np.ndarray,
    future: np.ndarray,
    ego_length: float,
    adv_length: float,
    options: dict[str, Any],
) -> dict[str, Any]:
    series = longitudinal_series_from_states(future, ego_length, adv_length)
    gap = series["gap"]
    ttc = series["ttc"]
    thw = series["thw"]
    drac = series["drac"]
    initial_ego = context[0]
    initial_target = context[1]
    initial_gap = initial_target[0] - initial_ego[0] - 0.5 * (
        float(ego_length) + float(adv_length)
    )
    raw = _cutin_raw_motion_metrics(
        recording,
        row,
        fps=float(options["target_fps"]),
        ego_length=ego_length,
        adv_length=adv_length,
    )
    lateral_offset = (
        np.asarray(future[:, 1, 1], dtype=np.float32)
        - np.asarray(future[:, 0, 1], dtype=np.float32)
    )
    return {
        "initial_gap": float(initial_gap),
        "initial_closing_speed": float(initial_ego[2] - initial_target[2]),
        "recorded_min_gap": float(np.min(gap)),
        "recorded_min_ttc": float(np.min(np.clip(ttc, 0.0, 1000.0))),
        "post_cutin_min_gap": float(np.min(gap)),
        "post_cutin_min_ttc": float(np.min(np.clip(ttc, 0.0, 1000.0))),
        "_gap_series": gap.astype(np.float32),
        "_ttc_series": np.clip(ttc, 0.0, 1000.0).astype(np.float32),
        "_thw_series": np.clip(thw, 0.0, 1000.0).astype(np.float32),
        "_drac_series": np.clip(drac, 0.0, 1000.0).astype(np.float32),
        "_ego_speed_series": series["ego_speed"].astype(np.float32),
        "_lead_speed_series": series["lead_speed"].astype(np.float32),
        "_ego_accel_series": series["ego_accel"].astype(np.float32),
        "_lateral_offset_series": lateral_offset.astype(np.float32),
        **raw,
    }


def build_highd_cutin_event_rows_from_recording(
    recording: Any,
    events: pd.DataFrame,
    *,
    options: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    opts = highd_cutin_options(options)
    pre_cross_steps = opts.get("context_pre_cross_steps", opts["history_steps"])
    horizon_steps = int(opts["context_horizon_steps"])
    min_post_cross_steps = int(opts["min_post_cross_steps"])
    rows: list[dict[str, Any]] = []
    skipped = 0
    for _, row in events.iterrows():
        ego_id = int(row["ego_id"])
        target_id = int(row["target_id"])
        ego_meta = recording.tracks_meta.loc[ego_id]
        target_meta = recording.tracks_meta.loc[target_id]
        ego_class = str(ego_meta.get("class", "")).strip().lower()
        target_class = str(target_meta.get("class", "")).strip().lower()
        if bool(opts.get("require_passenger_car_ego", True)) and ego_class != PASSENGER_CAR_CLASS:
            skipped += 1
            continue
        if bool(opts.get("require_passenger_car_lead", True)) and target_class != PASSENGER_CAR_CLASS:
            skipped += 1
            continue
        cross_frame = _optional_frame_value(row, "cross_frame")
        cutin_start_frame = _optional_frame_value(row, "cutin_start_frame")
        cutin_end_frame = _optional_frame_value(row, "cutin_end_frame")
        if (
            cross_frame is None
            or cutin_start_frame is None
            or cutin_end_frame is None
        ):
            skipped += 1
            continue
        item = _cutin_fixed_horizon_context(
            recording,
            row,
            pre_cross_steps,
            horizon_steps,
            min_post_cross_steps,
        )
        if item is None:
            skipped += 1
            continue
        metrics = _cutin_metrics(
            recording,
            row,
            item["initial_states"],
            item["future_states"],
            float(item["ego_length"]),
            float(item["adv_length"]),
            opts,
        )
        rows.append(
            {
                "recording_id": int(row["recording_id"]),
                "event_id": str(row["event_id"]),
                "ego_id": ego_id,
                "target_id": target_id,
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "anchor_frame": int(item["anchor_frame"]),
                "context_end_frame": int(item["context_end_frame"]),
                "context_length": int(item["context_length"]),
                "context_horizon_steps": int(item["context_horizon_steps"]),
                "pre_cross_steps": int(item["pre_cross_steps"]),
                "post_cross_steps": int(item["post_cross_steps"]),
                "risk_start_frame": int(item["risk_start_frame"]),
                "risk_start_index": int(item["risk_start_index"]),
                "cross_frame": cross_frame,
                "cutin_start_frame": cutin_start_frame,
                "cutin_end_frame": cutin_end_frame,
                "source_lane": int(row.get("source_lane", -1)),
                "target_lane": int(row.get("target_lane", -1)),
                "scenario_conditions": item["scenario_conditions"],
                "initial_states": item["initial_states"],
                "future_states": item["future_states"],
                "local_states": item["local_states"],
                "world_states": item["world_states"],
                "frames": item["frames"],
                "ego_length": float(item["ego_length"]),
                "adv_length": float(item["adv_length"]),
                **metrics,
            }
        )
    return rows, skipped


def score_highd_cutin_event_rows(
    rows: list[dict[str, Any]],
    *,
    options: dict[str, Any] | None = None,
) -> None:
    opts = highd_cutin_options(options)
    cfg = {
        "closed_loop_risk": {
            "collision_bonus": float(opts["collision_bonus"]),
            "near_collision_weight": float(opts["near_collision_weight"]),
            "hard_brake_weight": float(opts["hard_brake_weight"]),
            "hard_brake_threshold": float(opts["hard_brake_threshold"]),
        },
        "longitudinal_risk_scoring": {
            "ttc_weight": float(opts["w_ttc"]),
            "thw_weight": float(opts["w_thw"]),
            "gap_weight": float(opts["w_gap"]),
            "drac_weight": float(opts["w_drac"]),
            "ttc_scale": float(opts["ttc_scale"]),
            "thw_scale": float(opts["thw_scale"]),
            "gap_scale": float(opts["gap_scale"]),
            "drac_scale": float(opts["drac_scale"]),
            "ttc_eps": float(opts["ttc_eps"]),
            "thw_eps": float(opts["thw_eps"]),
            "gap_eps": float(opts["gap_eps"]),
            "pool_beta": float(opts["pool_beta"]),
        },
        "cutin_risk": {
            "lateral_overlap_threshold": float(
                opts["lateral_overlap_threshold"]
            ),
            "cutin_lateral_offset": float(opts["cutin_lateral_offset"]),
            "min_lateral_approach_speed": float(
                opts["min_lateral_approach_speed"]
            ),
            "semantic_window_seconds": float(opts["semantic_window_seconds"]),
            "ltg_window_steps": int(opts["ltg_window_steps"]),
            "ltg_weight": float(opts["ltg_weight"]),
            "ltg_scale": float(opts["ltg_scale"]),
            "ltg_eps": float(opts["ltg_eps"]),
            "require_front_at_cutin": bool(opts["require_front_at_cutin"]),
            "min_cutin_front_gap": float(opts["min_cutin_front_gap"]),
            "require_cutin_for_failure": bool(
                opts["require_cutin_for_failure"]
            ),
            "non_cutin_y_cutin": float(opts["non_cutin_y_cutin"]),
        },
    }

    for row in rows:
        series = {
            "gap": row["_gap_series"],
            "ttc": row["_ttc_series"],
            "thw": row["_thw_series"],
            "drac": row["_drac_series"],
            "ego_speed": row["_ego_speed_series"],
            "lead_speed": row["_lead_speed_series"],
            "ego_accel": row["_ego_accel_series"],
        }
        risk = cutin_risk_from_series(
            series=series,
            lateral_offset=row["_lateral_offset_series"],
            config=cfg,
            scoring_section="longitudinal_risk_scoring",
            dt=1.0 / max(float(opts["target_fps"]), 1.0e-6),
            risk_start_index=int(row.get("risk_start_index", 0)),
        )
        row["collision"] = float(risk["collision"])
        row["near_collision"] = float(risk["near_collision"])
        row["y_cutin"] = float(risk["y_cutin"])
        row["y_long"] = float(risk["y_long"])
        row["cutin_safety_risk_score"] = float(
            risk["cutin_safety_risk_score"]
        )
        row["post_longitudinal_risk_score"] = float(
            risk["post_longitudinal_risk_score"]
        )
        row["ltg_risk_score"] = float(risk["ltg_risk_score"])
        row["cutin_gap"] = float(risk["cutin_gap"])
        row["cutin_ttc"] = float(risk["cutin_ttc"])
        row["cutin_time_headway"] = float(risk["cutin_time_headway"])
        row["cutin_lateral_time_gap"] = float(risk["cutin_lateral_time_gap"])
        row["post_cutin_min_gap"] = float(risk["min_post_cutin_gap"])
        row["post_cutin_min_ttc"] = float(risk["min_post_cutin_ttc"])
        row["max_post_cutin_drac"] = float(risk["max_post_cutin_drac"])
        row["min_abs_lateral_offset"] = float(
            risk["min_abs_lateral_offset"]
        )
        row["max_abs_lateral_velocity"] = float(
            risk["max_abs_lateral_velocity"]
        )
        row["final_abs_lateral_offset"] = float(
            risk["final_abs_lateral_offset"]
        )
        row["max_lateral_approach_speed"] = float(
            risk["max_lateral_approach_speed"]
        )
        row["is_cutin"] = float(risk["is_cutin"])
        row["is_front_cutin"] = float(risk["is_front_cutin"])


def highd_cutin_score_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: row[key] for key in HIGHD_CUTIN_SCORE_KEYS if key in row} for row in rows]


def filter_semantic_cutin_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep rows that passed the semantic cut-in gate."""
    return [row for row in rows if float(row.get("is_cutin", 0.0)) >= 0.5]


def save_highd_cutin_event_context_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lengths = np.asarray([len(row["frames"]) for row in rows], dtype=np.int64)
    offsets = np.zeros(len(rows), dtype=np.int64)
    if len(rows) > 1:
        offsets[1:] = np.cumsum(lengths[:-1], dtype=np.int64)
    payload: dict[str, np.ndarray] = {
        "scenario_conditions": np.asarray(
            [row["scenario_conditions"] for row in rows],
            dtype=np.float32,
        ),
        "initial_states": np.asarray(
            [row["initial_states"] for row in rows],
            dtype=np.float32,
        ),
        "offset": offsets,
        "length": lengths,
        "frames": np.concatenate(
            [row["frames"] for row in rows],
            axis=0,
        ).astype(np.int64),
        "local_states": np.concatenate(
            [row["local_states"] for row in rows],
            axis=0,
        ).astype(np.float32),
        "world_states": np.concatenate(
            [row["world_states"] for row in rows],
            axis=0,
        ).astype(np.float32),
    }
    for key in HIGHD_CUTIN_SCORE_KEYS:
        if key in rows[0]:
            payload[key] = np.asarray([row[key] for row in rows])
    np.savez_compressed(path, **payload)


def load_highd_cutin_event_context_cache(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"highD cut-in event context cache not found: {path}")
    data = np.load(path, allow_pickle=True)
    if "scenario_conditions" not in data.files or "initial_states" not in data.files:
        raise KeyError(f"{path} is missing scenario_conditions/initial_states")
    arrays = {
        key: data[key]
        for key in ("scenario_conditions", "initial_states", *HIGHD_CUTIN_SCORE_KEYS)
        if key in data.files
    }
    has_packed_states = all(
        key in data.files for key in ("offset", "length", "frames", "local_states")
    )
    if has_packed_states:
        packed_offset = data["offset"]
        packed_length = data["length"]
        packed_frames = data["frames"]
        packed_local = data["local_states"]
        packed_world = data["world_states"] if "world_states" in data.files else None
    else:
        packed_offset = packed_length = packed_frames = packed_local = packed_world = None
    count = int(arrays["scenario_conditions"].shape[0])
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        row: dict[str, Any] = {
            "scenario_conditions": arrays["scenario_conditions"][idx].astype(np.float32),
            "initial_states": arrays["initial_states"][idx].astype(np.float32),
        }
        if has_packed_states:
            offset = int(packed_offset[idx])
            length = int(packed_length[idx])
            stop = offset + length
            local_states = packed_local[offset:stop].astype(np.float32)
            row["frames"] = packed_frames[offset:stop].astype(np.int64)
            row["local_states"] = local_states
            row["future_states"] = local_states[1:]
            if packed_world is not None:
                row["world_states"] = packed_world[offset:stop].astype(np.float32)
        for key in HIGHD_CUTIN_SCORE_KEYS:
            if key not in arrays:
                continue
            value = arrays[key][idx]
            if isinstance(value, np.generic):
                value = value.item()
            row[key] = value
        rows.append(row)
    return rows

"""Safety-envelope intrusion risk for highD natural driving segments.

Pairwise safety distances define a dynamic safety ellipse.  Positive ellipse
intrusion is kept as raw severity, and a trajectory score combines peak
severity with linear exposure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SEI_COMPONENT_NAMES = (
    "sei_instant",
    "sei_raw_margin",
    "sei_ellipse_intrusion",
    "sei_longitudinal_deficit",
    "sei_lateral_deficit",
    "sei_bbox_overlap",
)


@dataclass(frozen=True)
class SafetyEnvelopeRiskOptions:
    prediction_horizon_seconds: float = 1.5
    prediction_dt_seconds: float = 0.2
    prediction_use_acceleration: bool = False
    acceleration_clip_mps2: float = 3.0
    longitudinal_mode: str = "calibrated_headway"
    longitudinal_time_gap_seconds: float = 0.7
    longitudinal_min_margin_m: float = 1.0
    longitudinal_brake_mps2: float = 4.0
    rss_response_time_seconds: float = 0.75
    rss_accel_max_mps2: float = 2.0
    rss_brake_min_mps2: float = 4.0
    rss_brake_max_mps2: float = 8.0
    lateral_min_margin_m: float = 0.20
    lateral_response_time_seconds: float = 1.0
    lateral_brake_mps2: float = 0.8
    pair_smooth_beta: float = 8.0
    horizon_smooth_beta: float = 8.0
    horizon_time_discount: float = 0.10
    exposure_weight: float = 0.15

    @classmethod
    def from_config(
        cls,
        risk_config: dict[str, Any],
        *,
        fps: float,
    ) -> "SafetyEnvelopeRiskOptions":
        cfg = dict(risk_config.get("safety_envelope", {}))
        default_dt = max(1.0 / float(fps), 0.04)
        return cls(
            prediction_horizon_seconds=float(
                cfg.get("prediction_horizon_seconds", 1.5)
            ),
            prediction_dt_seconds=float(cfg.get("prediction_dt_seconds", default_dt)),
            prediction_use_acceleration=bool(
                cfg.get("prediction_use_acceleration", False)
            ),
            acceleration_clip_mps2=float(cfg.get("acceleration_clip_mps2", 3.0)),
            longitudinal_mode=str(
                cfg.get("longitudinal_mode", "calibrated_headway")
            ).strip().lower(),
            longitudinal_time_gap_seconds=float(
                cfg.get("longitudinal_time_gap_seconds", 0.7)
            ),
            longitudinal_min_margin_m=float(
                cfg.get("longitudinal_min_margin_m", 1.0)
            ),
            longitudinal_brake_mps2=float(cfg.get("longitudinal_brake_mps2", 4.0)),
            rss_response_time_seconds=float(
                cfg.get("rss_response_time_seconds", 0.75)
            ),
            rss_accel_max_mps2=float(cfg.get("rss_accel_max_mps2", 2.0)),
            rss_brake_min_mps2=float(cfg.get("rss_brake_min_mps2", 4.0)),
            rss_brake_max_mps2=float(cfg.get("rss_brake_max_mps2", 8.0)),
            lateral_min_margin_m=float(cfg.get("lateral_min_margin_m", 0.20)),
            lateral_response_time_seconds=float(
                cfg.get("lateral_response_time_seconds", 1.0)
            ),
            lateral_brake_mps2=float(cfg.get("lateral_brake_mps2", 0.8)),
            pair_smooth_beta=float(cfg.get("pair_smooth_beta", 8.0)),
            horizon_smooth_beta=float(cfg.get("horizon_smooth_beta", 8.0)),
            horizon_time_discount=float(cfg.get("horizon_time_discount", 0.10)),
            exposure_weight=float(cfg.get("exposure_weight", 0.15)),
        )


def _positive(value: float, floor: float = 1.0e-6) -> float:
    return max(float(value), float(floor))


def _prediction_offsets(options: SafetyEnvelopeRiskOptions) -> np.ndarray:
    horizon = max(float(options.prediction_horizon_seconds), 0.0)
    step = _positive(options.prediction_dt_seconds)
    count = max(1, int(np.floor(horizon / step)) + 1)
    offsets = np.arange(count, dtype=np.float32) * float(step)
    if offsets[-1] < horizon - 1.0e-6:
        offsets = np.append(offsets, np.float32(horizon))
    return offsets


def smoothmax_axis(
    values: np.ndarray,
    beta: float,
    *,
    axis: int,
    empty_value: float = 0.0,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    count = np.sum(finite, axis=axis)
    if arr.size == 0:
        shape = list(arr.shape)
        del shape[axis]
        return np.full(shape, float(empty_value), dtype=np.float32)
    beta_value = _positive(beta)
    scaled = np.where(finite, beta_value * arr, -np.inf)
    max_scaled = np.max(scaled, axis=axis, keepdims=True)
    all_empty = ~np.isfinite(max_scaled)
    safe_max_scaled = np.where(np.isfinite(max_scaled), max_scaled, 0.0)
    stable = np.where(finite, scaled - safe_max_scaled, -np.inf)
    total = np.sum(np.where(finite, np.exp(stable), 0.0), axis=axis)
    max_flat = np.squeeze(max_scaled, axis=axis)
    out = (max_flat + np.log(np.maximum(total, 1.0e-300)) - np.log(np.maximum(count, 1))) / beta_value
    out = np.where(count > 0, out, float(empty_value))
    return out.astype(np.float32)


def _rss_braking_clearance(
    *,
    follower_speed: np.ndarray,
    leader_speed: np.ndarray,
    options: SafetyEnvelopeRiskOptions,
) -> np.ndarray:
    rho = max(float(options.rss_response_time_seconds), 0.0)
    accel = max(float(options.rss_accel_max_mps2), 0.0)
    brake_min = _positive(options.rss_brake_min_mps2)
    brake_max = _positive(options.rss_brake_max_mps2)
    follower = np.maximum(np.asarray(follower_speed, dtype=np.float64), 0.0)
    leader = np.maximum(np.asarray(leader_speed, dtype=np.float64), 0.0)
    follower_after_response = follower + accel * rho
    follower_distance = (
        follower * rho
        + 0.5 * accel * rho * rho
        + np.square(follower_after_response) / (2.0 * brake_min)
    )
    leader_brake_distance = np.square(leader) / (2.0 * brake_max)
    return np.maximum(follower_distance - leader_brake_distance, 0.0)


def _longitudinal_safe_clearance(
    *,
    dx_pred: np.ndarray,
    ego_vx_pred: np.ndarray,
    other_vx_pred: np.ndarray,
    rel_vx_pred: np.ndarray,
    options: SafetyEnvelopeRiskOptions,
) -> np.ndarray:
    other_in_front = dx_pred >= 0.0
    follower_speed = np.where(other_in_front, ego_vx_pred, other_vx_pred)
    leader_speed = np.where(other_in_front, other_vx_pred, ego_vx_pred)
    follower_speed = np.maximum(follower_speed, 0.0)
    leader_speed = np.maximum(leader_speed, 0.0)
    sign_dx = np.where(dx_pred >= 0.0, 1.0, -1.0)
    closing = np.maximum(-sign_dx * rel_vx_pred, 0.0)
    mode = str(options.longitudinal_mode).strip().lower()
    if mode in {"rss", "rss_braking", "braking"}:
        dynamic = _rss_braking_clearance(
            follower_speed=follower_speed,
            leader_speed=leader_speed,
            options=options,
        )
    elif mode in {"approach", "relative"}:
        brake = _positive(options.longitudinal_brake_mps2)
        dynamic = (
            max(float(options.rss_response_time_seconds), 0.0) * closing
            + np.square(closing) / (2.0 * brake)
        )
    elif mode in {"calibrated_headway", "headway", "rss_headway"}:
        brake = _positive(options.longitudinal_brake_mps2)
        dynamic = (
            max(float(options.longitudinal_time_gap_seconds), 0.0) * follower_speed
            + np.square(closing) / (2.0 * brake)
        )
    else:
        raise ValueError(f"unknown safety_envelope.longitudinal_mode: {mode!r}")
    return dynamic + max(float(options.longitudinal_min_margin_m), 0.0)


def _lateral_safe_clearance(
    *,
    dy_pred: np.ndarray,
    rel_vy_pred: np.ndarray,
    options: SafetyEnvelopeRiskOptions,
) -> np.ndarray:
    sign_dy = np.where(dy_pred >= 0.0, 1.0, -1.0)
    lateral_closing = np.maximum(-sign_dy * rel_vy_pred, 0.0)
    brake = _positive(options.lateral_brake_mps2)
    dynamic = (
        max(float(options.lateral_response_time_seconds), 0.0) * lateral_closing
        + np.square(lateral_closing) / (2.0 * brake)
    )
    return dynamic + max(float(options.lateral_min_margin_m), 0.0)


def pairwise_safety_envelope_intrusion(
    *,
    ego_x: np.ndarray,
    ego_y: np.ndarray,
    ego_vx: np.ndarray,
    ego_vy: np.ndarray,
    ego_ax: np.ndarray,
    ego_ay: np.ndarray,
    other_x: np.ndarray,
    other_y: np.ndarray,
    other_vx: np.ndarray,
    other_vy: np.ndarray,
    other_ax: np.ndarray,
    other_ay: np.ndarray,
    ego_length: float,
    ego_width: float,
    other_length: float,
    other_width: float,
    valid: np.ndarray,
    options: SafetyEnvelopeRiskOptions,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(np.asarray(valid).size)
    pair_score = np.zeros(n, dtype=np.float32)
    components = np.zeros((n, len(SEI_COMPONENT_NAMES)), dtype=np.float32)
    valid_mask = np.asarray(valid, dtype=bool)
    if n == 0 or not np.any(valid_mask):
        return pair_score, components

    tau = _prediction_offsets(options).astype(np.float64)
    dx = np.asarray(other_x, dtype=np.float64) - np.asarray(ego_x, dtype=np.float64)
    dy = np.asarray(other_y, dtype=np.float64) - np.asarray(ego_y, dtype=np.float64)
    rel_vx = np.asarray(other_vx, dtype=np.float64) - np.asarray(ego_vx, dtype=np.float64)
    rel_vy = np.asarray(other_vy, dtype=np.float64) - np.asarray(ego_vy, dtype=np.float64)

    if bool(options.prediction_use_acceleration):
        clip = _positive(options.acceleration_clip_mps2)
        ego_ax_arr = np.clip(np.asarray(ego_ax, dtype=np.float64), -clip, clip)
        ego_ay_arr = np.clip(np.asarray(ego_ay, dtype=np.float64), -clip, clip)
        other_ax_arr = np.clip(np.asarray(other_ax, dtype=np.float64), -clip, clip)
        other_ay_arr = np.clip(np.asarray(other_ay, dtype=np.float64), -clip, clip)
        rel_ax = other_ax_arr - ego_ax_arr
        rel_ay = other_ay_arr - ego_ay_arr
    else:
        ego_ax_arr = np.zeros(n, dtype=np.float64)
        ego_ay_arr = np.zeros(n, dtype=np.float64)
        other_ax_arr = np.zeros(n, dtype=np.float64)
        other_ay_arr = np.zeros(n, dtype=np.float64)
        rel_ax = np.zeros(n, dtype=np.float64)
        rel_ay = np.zeros(n, dtype=np.float64)

    tau_2d = tau.reshape(1, -1)
    dx_pred = dx.reshape(-1, 1) + rel_vx.reshape(-1, 1) * tau_2d + 0.5 * rel_ax.reshape(-1, 1) * tau_2d * tau_2d
    dy_pred = dy.reshape(-1, 1) + rel_vy.reshape(-1, 1) * tau_2d + 0.5 * rel_ay.reshape(-1, 1) * tau_2d * tau_2d
    rel_vx_pred = rel_vx.reshape(-1, 1) + rel_ax.reshape(-1, 1) * tau_2d
    rel_vy_pred = rel_vy.reshape(-1, 1) + rel_ay.reshape(-1, 1) * tau_2d
    ego_vx_pred = np.asarray(ego_vx, dtype=np.float64).reshape(-1, 1) + ego_ax_arr.reshape(-1, 1) * tau_2d
    other_vx_pred = np.asarray(other_vx, dtype=np.float64).reshape(-1, 1) + other_ax_arr.reshape(-1, 1) * tau_2d

    half_length = 0.5 * (float(ego_length) + float(other_length))
    half_width = 0.5 * (float(ego_width) + float(other_width))
    long_safe = _longitudinal_safe_clearance(
        dx_pred=dx_pred,
        ego_vx_pred=ego_vx_pred,
        other_vx_pred=other_vx_pred,
        rel_vx_pred=rel_vx_pred,
        options=options,
    )
    lat_safe = _lateral_safe_clearance(
        dy_pred=dy_pred,
        rel_vy_pred=rel_vy_pred,
        options=options,
    )
    axis_x = np.maximum(half_length + long_safe, 1.0e-3)
    axis_y = np.maximum(half_width + lat_safe, 1.0e-3)
    ellipse_margin = np.square(dx_pred / axis_x) + np.square(dy_pred / axis_y) - 1.0
    raw_margin = -ellipse_margin - float(options.horizon_time_discount) * tau_2d
    raw_margin[~valid_mask, :] = -np.inf

    raw_pair = smoothmax_axis(
        raw_margin,
        float(options.horizon_smooth_beta),
        axis=1,
        empty_value=-np.inf,
    ).astype(np.float64)
    score = np.maximum(raw_pair, 0.0).astype(np.float32)
    score[~valid_mask] = 0.0
    pair_score[:] = score

    long_clearance = np.abs(dx_pred) - half_length
    lat_clearance = np.abs(dy_pred) - half_width
    long_deficit = np.maximum(long_safe - np.maximum(long_clearance, 0.0), 0.0)
    lat_deficit = np.maximum(lat_safe - np.maximum(lat_clearance, 0.0), 0.0)
    long_norm = long_deficit / np.maximum(long_safe + 1.0, 1.0)
    lat_norm = lat_deficit / np.maximum(lat_safe + 0.25, 0.25)
    overlap = np.maximum(-long_clearance, 0.0) * np.maximum(-lat_clearance, 0.0)
    long_norm[~valid_mask, :] = -np.inf
    lat_norm[~valid_mask, :] = -np.inf
    overlap[~valid_mask, :] = -np.inf

    components[:, 0] = pair_score
    components[:, 1] = np.maximum(raw_pair, 0.0).astype(np.float32)
    components[:, 2] = np.maximum(raw_pair, 0.0).astype(np.float32)
    components[:, 3] = smoothmax_axis(
        long_norm,
        float(options.horizon_smooth_beta),
        axis=1,
        empty_value=0.0,
    )
    components[:, 4] = smoothmax_axis(
        lat_norm,
        float(options.horizon_smooth_beta),
        axis=1,
        empty_value=0.0,
    )
    components[:, 5] = smoothmax_axis(
        overlap,
        float(options.horizon_smooth_beta),
        axis=1,
        empty_value=0.0,
    )
    components[~valid_mask, :] = 0.0
    return pair_score, components


def trajectory_safety_envelope_risk_trace(
    instant_risk: np.ndarray,
    *,
    options: SafetyEnvelopeRiskOptions,
    dt_seconds: float,
) -> np.ndarray:
    instant = np.maximum(np.asarray(instant_risk, dtype=np.float64), 0.0)
    if instant.size == 0:
        return np.zeros(0, dtype=np.float32)
    prefix_peak = np.maximum.accumulate(instant)
    exposure = (
        float(options.exposure_weight)
        * max(float(dt_seconds), 0.0)
        * np.cumsum(instant)
    )
    trace = prefix_peak + exposure
    trace = np.maximum.accumulate(np.maximum(trace, 0.0))
    return trace.astype(np.float32)

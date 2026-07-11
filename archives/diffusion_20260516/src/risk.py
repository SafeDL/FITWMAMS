"""Future-window risk scoring for generated or observed adversarial actions."""
from __future__ import annotations

from typing import Dict

import numpy as np

from process_highd.src.risk_metrics import (
    compute_drac,
    compute_gap,
    compute_instant_risk,
    compute_thw,
    compute_trajectory_risk,
    compute_ttc,
)

from .types import EventType


def _event_value(event_type: EventType | str) -> str:
    return event_type.value if isinstance(event_type, EventType) else str(event_type)


def following_risk(
    ego_future: np.ndarray,
    adv_future: np.ndarray,
    ego_length: float,
    adv_length: float,
    cfg: Dict[str, float] | None = None,
) -> float:
    cfg = cfg or {}
    eps = float(cfg.get("epsilon", 1e-6))
    n = min(len(ego_future), len(adv_future))
    if n == 0:
        return 0.0
    ego = np.asarray(ego_future[:n], dtype=np.float64)
    adv = np.asarray(adv_future[:n], dtype=np.float64)
    gap = compute_gap(ego[:, 0], adv[:, 0], ego_length, adv_length)
    mask = gap > eps
    if not np.any(mask):
        return float(cfg.get("collision_risk", 100.0))
    ttc = compute_ttc(gap, ego[:, 2], adv[:, 2], float(cfg.get("max_ttc_clip", 1000.0)), eps)
    thw = compute_thw(gap, ego[:, 2], float(cfg.get("max_thw_clip", 200.0)), eps)
    drac = compute_drac(gap, ego[:, 2], adv[:, 2], eps)
    instant = compute_instant_risk(ttc[mask], thw[mask], drac[mask], cfg, eps)
    return compute_trajectory_risk(instant, float(cfg.get("softmax_lambda", 10.0)))


def cutin_risk(
    ego_future: np.ndarray,
    adv_future: np.ndarray,
    ego_length: float,
    adv_length: float,
    lane_width: float,
    cfg: Dict[str, float] | None = None,
) -> float:
    cfg = cfg or {}
    base = following_risk(ego_future, adv_future, ego_length, adv_length, cfg)
    n = min(len(ego_future), len(adv_future))
    if n == 0:
        return base
    ego = np.asarray(ego_future[:n], dtype=np.float64)
    adv = np.asarray(adv_future[:n], dtype=np.float64)
    lateral_offset = np.abs(adv[:, 1] - ego[:, 1])
    lane_w = max(float(lane_width), 1e-6)
    intrusion = np.clip(1.0 - lateral_offset / lane_w, 0.0, 1.0)
    lat_speed = np.abs(adv[:, 3])
    intrusion_term = float(np.max(intrusion) * cfg.get("intrusion_weight", 0.4))
    lat_term = float(np.max(lat_speed) * cfg.get("lateral_velocity_weight", 0.1))
    return float(base + intrusion_term + lat_term)


def score_future_risk(
    event_type: EventType | str,
    ego_future: np.ndarray,
    adv_future: np.ndarray,
    ego_length: float,
    adv_length: float,
    lane_width: float = 3.75,
    cfg: Dict[str, float] | None = None,
) -> float:
    if _event_value(event_type) == EventType.FOLLOWING.value:
        return following_risk(ego_future, adv_future, ego_length, adv_length, cfg)
    if _event_value(event_type) == EventType.CUT_IN.value:
        return cutin_risk(ego_future, adv_future, ego_length, adv_length, lane_width, cfg)
    raise ValueError(f"Unsupported event_type: {event_type}")


def constant_velocity_rollout(current: np.ndarray, horizon: int, dt: float) -> np.ndarray:
    """Roll an ego state ``[x, y, vx, vy, ax, ay]`` forward as a fallback ADS prediction."""
    s = np.asarray(current, dtype=np.float32).copy()
    out = np.zeros((int(horizon) + 1, 6), dtype=np.float32)
    out[0] = s
    for i in range(int(horizon)):
        s[0] = s[0] + s[2] * dt
        s[1] = s[1] + s[3] * dt
        s[4] = 0.0
        s[5] = 0.0
        out[i + 1] = s
    return out


def constant_acceleration_rollout(current: np.ndarray, horizon: int, dt: float) -> np.ndarray:
    """Roll a state ``[x, y, vx, vy, ax, ay]`` forward with fixed acceleration."""
    s = np.asarray(current, dtype=np.float32).copy()
    out = np.zeros((int(horizon) + 1, 6), dtype=np.float32)
    out[0] = s
    ax = float(s[4])
    ay = float(s[5])
    for i in range(int(horizon)):
        s[0] = s[0] + s[2] * dt + 0.5 * ax * dt * dt
        s[1] = s[1] + s[3] * dt + 0.5 * ay * dt * dt
        s[2] = max(float(s[2]) + ax * dt, 0.0)
        s[3] = float(s[3]) + ay * dt
        s[4] = ax
        s[5] = ay
        out[i + 1] = s
    return out

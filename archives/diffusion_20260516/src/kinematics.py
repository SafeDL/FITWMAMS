"""Action integration and physical feasibility checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .types import EventType, VehicleState


def _event_value(event_type: EventType | str) -> str:
    return event_type.value if isinstance(event_type, EventType) else str(event_type)


@dataclass(frozen=True)
class ConstraintConfig:
    ax_min: float = -8.0
    ax_max: float = 4.0
    jerk_abs_max: float = 12.0
    lateral_velocity_abs_max: float = 3.0
    yaw_rate_abs_max: float = 0.6
    lane_margin: float = 0.4
    min_initial_gap: float = 0.2


def integrate_following_actions(
    initial: VehicleState,
    actions: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Integrate lead-car longitudinal acceleration.

    Returns states with shape ``[H + 1, 6]`` in the same local frame as
    ``initial``.
    """
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim == 2:
        ax_seq = a[:, 0]
    else:
        ax_seq = a.reshape(-1)
    states = np.zeros((len(ax_seq) + 1, 6), dtype=np.float32)
    states[0] = initial.as_feature()
    x = float(initial.x)
    y = float(initial.y)
    vx = max(float(initial.vx), 0.0)
    vy = float(initial.vy)
    ay = float(initial.ay)
    for i, ax in enumerate(ax_seq):
        ax_f = float(ax)
        x = x + vx * dt + 0.5 * ax_f * dt * dt
        vx = max(vx + ax_f * dt, 0.0)
        states[i + 1] = np.asarray([x, y, vx, vy, ax_f, ay], dtype=np.float32)
    return states


def integrate_cutin_actions(
    initial: VehicleState,
    actions: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Integrate cut-in actions ``[ax, yaw_rate]`` with a simple yaw-velocity model."""
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] < 2:
        raise ValueError(f"cut-in actions must be [H, 2], got {a.shape}")
    states = np.zeros((a.shape[0] + 1, 6), dtype=np.float32)
    states[0] = initial.as_feature()
    x = float(initial.x)
    y = float(initial.y)
    speed = max(float(np.hypot(initial.vx, initial.vy)), 0.0)
    yaw = float(initial.yaw)
    prev_vx = float(initial.vx)
    prev_vy = float(initial.vy)
    for i, (ax, yaw_rate) in enumerate(a):
        yaw = yaw + float(yaw_rate) * dt
        speed = max(speed + float(ax) * dt, 0.0)
        vx = speed * np.cos(yaw)
        vy = speed * np.sin(yaw)
        x = x + vx * dt
        y = y + vy * dt
        ay = (vy - prev_vy) / dt
        states[i + 1] = np.asarray([x, y, vx, vy, float(ax), ay], dtype=np.float32)
        prev_vx = vx
        prev_vy = vy
    return states


def integrate_actions(
    event_type: EventType | str,
    initial: VehicleState,
    actions: np.ndarray,
    dt: float,
) -> np.ndarray:
    if _event_value(event_type) == EventType.FOLLOWING.value:
        return integrate_following_actions(initial, actions, dt)
    if _event_value(event_type) == EventType.CUT_IN.value:
        return integrate_cutin_actions(initial, actions, dt)
    raise ValueError(f"Unsupported event_type: {event_type}")


def naturalness_cost(actions: np.ndarray, dt: float, cfg: ConstraintConfig) -> Tuple[float, Dict[str, float]]:
    a = np.asarray(actions, dtype=np.float32)
    ax = a[:, 0]
    jerk = np.diff(ax, prepend=ax[0]) / max(dt, 1e-6)
    violations = {
        "ax_low": float(np.maximum(cfg.ax_min - ax, 0.0).sum()),
        "ax_high": float(np.maximum(ax - cfg.ax_max, 0.0).sum()),
        "jerk": float(np.maximum(np.abs(jerk) - cfg.jerk_abs_max, 0.0).sum()),
    }
    if a.ndim == 2 and a.shape[1] > 1:
        yaw_rate = a[:, 1]
        violations["yaw_rate"] = float(np.maximum(np.abs(yaw_rate) - cfg.yaw_rate_abs_max, 0.0).sum())
    cost = float(sum(violations.values()))
    return cost, violations


def feasibility_cost(
    event_type: EventType | str,
    trajectory: np.ndarray,
    ego_future: np.ndarray,
    actions: np.ndarray,
    dt: float,
    lane_width: float,
    cfg: ConstraintConfig,
) -> Tuple[float, Dict[str, float]]:
    nat_cost, parts = naturalness_cost(actions, dt, cfg)
    ego = np.asarray(ego_future, dtype=np.float32)
    adv = np.asarray(trajectory, dtype=np.float32)
    n = min(len(ego), len(adv))
    if n > 0:
        gap0 = adv[0, 0] - ego[0, 0]
        parts["initial_gap"] = float(max(cfg.min_initial_gap - gap0, 0.0))
    if _event_value(event_type) == EventType.CUT_IN.value and n > 0:
        lateral_v = adv[:n, 3]
        parts["lateral_velocity"] = float(np.maximum(np.abs(lateral_v) - cfg.lateral_velocity_abs_max, 0.0).sum())
        lane_half = 0.5 * max(float(lane_width), 1e-6)
        parts["lane_boundary"] = float(np.maximum(np.abs(adv[:n, 1]) - (lane_half + cfg.lane_margin), 0.0).sum())
    cost = nat_cost + float(sum(v for k, v in parts.items() if k not in {"ax_low", "ax_high", "jerk", "yaw_rate"}))
    return float(cost), parts

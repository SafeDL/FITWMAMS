"""Observation adapter from TREAD closed-loop vehicles to PPO input."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.kinematics import Vehicle


def _normalise(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    scaled = -1.0 + 2.0 * (float(value) - float(lower)) / (float(upper) - float(lower))
    return float(np.clip(scaled, -1.0, 1.0))


def _lane_width(vehicle: Any) -> float:
    width = float(getattr(AbstractLane, "DEFAULT_WIDTH", 4.0))
    try:
        return float(vehicle.road.network.get_lane(vehicle.lane_index).width)
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        return width


def _kinematic_row(
    vehicle: Any,
    *,
    origin: Any | None,
    lanes_count: int,
) -> list[float]:
    if origin is None:
        x = float(vehicle.position[0])
        y = float(vehicle.position[1])
        vx = float(vehicle.velocity[0])
        vy = float(vehicle.velocity[1])
    else:
        x = float(vehicle.position[0] - origin.position[0])
        y = float(vehicle.position[1] - origin.position[1])
        vx = float(vehicle.velocity[0] - origin.velocity[0])
        vy = float(vehicle.velocity[1] - origin.velocity[1])
    max_speed = float(getattr(Vehicle, "MAX_SPEED", 40.0))
    lane_width = _lane_width(vehicle)
    return [
        1.0,
        _normalise(x, -5.0 * max_speed, 5.0 * max_speed),
        _normalise(
            y,
            -lane_width * max(1, int(lanes_count)),
            lane_width * max(1, int(lanes_count)),
        ),
        _normalise(vx, -2.0 * max_speed, 2.0 * max_speed),
        _normalise(vy, -2.0 * max_speed, 2.0 * max_speed),
    ]


def _nearest_vehicles(
    ego_vehicle: Any,
    vehicles: Iterable[Any],
    *,
    max_count: int,
) -> list[Any]:
    candidates = [vehicle for vehicle in vehicles if vehicle is not ego_vehicle]
    candidates.sort(
        key=lambda vehicle: float(
            np.linalg.norm(np.asarray(vehicle.position) - np.asarray(ego_vehicle.position))
        )
    )
    return candidates[:max_count]


def convert_tread_obs_to_ppo_obs(
    tread_obs: Any,
    env: Any = None,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Convert TREAD/highway-env state to PPO's 5x5 kinematics observation."""

    cfg = dict(config or {})
    vehicles_count = int(cfg.get("vehicles_count", 5))
    features_count = int(cfg.get("features_count", 5))
    if vehicles_count != 5 or features_count != 5:
        raise ValueError(
            "PPO checkpoint expects a 5x5 kinematics observation; got "
            f"{vehicles_count}x{features_count}"
        )

    if isinstance(tread_obs, np.ndarray):
        obs = np.asarray(tread_obs, dtype=np.float32)
        if obs.shape == (vehicles_count, features_count):
            return obs
        if obs.size == vehicles_count * features_count:
            return obs.reshape(vehicles_count, features_count)
        raise ValueError(f"Cannot reshape observation {obs.shape} into 5x5")

    ego_vehicle = tread_obs
    lanes_count = int(cfg.get("lanes_count", 1))
    rows = [_kinematic_row(ego_vehicle, origin=None, lanes_count=lanes_count)]

    target_vehicle = cfg.get("target_vehicle")
    if target_vehicle is not None:
        rows.append(
            _kinematic_row(target_vehicle, origin=ego_vehicle, lanes_count=lanes_count)
        )

    road = getattr(ego_vehicle, "road", None)
    if road is not None and len(rows) < vehicles_count:
        for vehicle in _nearest_vehicles(
            ego_vehicle,
            getattr(road, "vehicles", []),
            max_count=vehicles_count - 1,
        ):
            if vehicle is target_vehicle:
                continue
            rows.append(_kinematic_row(vehicle, origin=ego_vehicle, lanes_count=lanes_count))
            if len(rows) >= vehicles_count:
                break

    while len(rows) < vehicles_count:
        rows.append([0.0] * features_count)
    return np.asarray(rows[:vehicles_count], dtype=np.float32)

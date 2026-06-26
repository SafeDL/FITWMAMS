"""highway-env IDM ego rollout helpers for generated highD scenarios."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


IDM_PARAMETER_KEYS = (
    "ACC_MAX",
    "COMFORT_ACC_MAX",
    "COMFORT_ACC_MIN",
    "DISTANCE_WANTED",
    "TIME_WANTED",
    "DELTA",
    "POLITENESS",
    "LANE_CHANGE_MIN_ACC_GAIN",
    "LANE_CHANGE_MAX_BRAKING_IMPOSED",
    "LANE_CHANGE_DELAY",
)


def load_idm_ego_config(
    path: str | Path,
    *,
    event_type: str | None = None,
) -> dict[str, Any]:
    """Load shared highway-env IDM ego playback settings."""
    cfg_path = Path(path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if "base" not in raw:
        return dict(raw)
    out = dict(raw.get("base") or {})
    if event_type is not None:
        overrides = raw.get("scenario_overrides", {}) or {}
        out.update(dict(overrides.get(str(event_type), {}) or {}))
    return out


def _load_highway_env_classes():
    root = Path(__file__).resolve().parents[2]
    highway_root = root / "HighwayEnv"
    highway_package = highway_root / "highway_env"
    if not highway_package.is_dir():
        raise FileNotFoundError(
            f"Required local highway-env package not found: {highway_package}"
        )
    if str(highway_root) not in sys.path:
        sys.path.insert(0, str(highway_root))
    try:
        from highway_env.road.road import Road, RoadNetwork
        from highway_env.vehicle.behavior import IDMVehicle
        from highway_env.vehicle.kinematics import Vehicle
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import the required local highway-env package. "
            "Install HighwayEnv dependencies, including gymnasium, in the active environment."
        ) from exc
    return Road, RoadNetwork, IDMVehicle, Vehicle


def _vehicle_state(vehicle: Any) -> np.ndarray:
    if isinstance(vehicle.action, dict):
        acceleration = float(vehicle.action.get("acceleration", 0.0))
        steering = float(vehicle.action.get("steering", 0.0))
    else:
        acceleration = 0.0
        steering = 0.0
    del steering
    vx = float(vehicle.speed) * float(np.cos(vehicle.heading))
    vy = float(vehicle.speed) * float(np.sin(vehicle.heading))
    ax = acceleration * float(np.cos(vehicle.heading))
    ay = acceleration * float(np.sin(vehicle.heading))
    return np.asarray(
        [vehicle.position[0], vehicle.position[1], vx, vy, ax, ay],
        dtype=np.float32,
    )


def _speed_and_heading(state: np.ndarray, *, keep_lateral_heading: bool) -> tuple[float, float]:
    vx = float(state[2])
    vy = float(state[3])
    speed = float(np.hypot(vx, vy))
    if keep_lateral_heading and speed > 1.0e-6:
        return speed, float(np.arctan2(vy, max(vx, 1.0e-6)))
    return speed, 0.0


def rollout_idm_ego_trajectory(
    initial_states: np.ndarray,
    adversary_trajectory: np.ndarray,
    ego_length: np.ndarray,
    adv_length: np.ndarray,
    *,
    dt: float,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Replay generated adversary trajectories while ego responds with IDM.

    Parameters
    ----------
    initial_states:
        ``[B, 2, 6]`` local-frame initial ego/adversary states.
    adversary_trajectory:
        ``[B, H, 6]`` generated lead/target future states.
    ego_length, adv_length:
        Vehicle lengths for collision/gap-aware highway-env dynamics.
    dt:
        Simulation timestep.
    config:
        Optional policy config. Supported keys: ``target_speed``, ``lanes_count``,
        ``speed_limit``, ``enable_lane_change``, ``keep_lateral_heading`` and
        IDM class parameters such as ``COMFORT_ACC_MAX``.
    """
    Road, RoadNetwork, IDMVehicle, Vehicle = _load_highway_env_classes()

    class ScriptedAdversaryVehicle(Vehicle):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.forced_position: np.ndarray | None = None
            self.forced_heading: float | None = None
            self.forced_speed: float | None = None

        def set_forced_state(
            self,
            position: np.ndarray,
            heading: float,
            speed: float,
        ) -> None:
            self.forced_position = np.asarray(position, dtype=np.float64)
            self.forced_heading = float(heading)
            self.forced_speed = float(speed)

        def act(self, action: dict | str = None) -> None:
            Vehicle.act(self, {"steering": 0.0, "acceleration": 0.0})

        def step(self, dt: float) -> None:
            del dt
            if self.forced_position is None:
                return
            self.position = self.forced_position
            self.heading = float(self.forced_heading)
            self.speed = float(self.forced_speed)
            self.forced_position = None
            self.forced_heading = None
            self.forced_speed = None
            self.on_state_update()

    cfg = dict(config or {})
    init = np.asarray(initial_states, dtype=np.float32)
    adv = np.asarray(adversary_trajectory, dtype=np.float32)
    if init.ndim != 3 or init.shape[1:] != (2, 6):
        raise ValueError(f"initial_states must have shape [B, 2, 6], got {init.shape}")
    if adv.ndim != 3 or adv.shape[0] != init.shape[0] or adv.shape[2] != 6:
        raise ValueError(
            "adversary_trajectory must have shape [B, H, 6] matching initial_states, "
            f"got {adv.shape}"
        )

    batch, horizon = int(adv.shape[0]), int(adv.shape[1])
    out = np.zeros((batch, horizon, 6), dtype=np.float32)
    lanes_count = int(cfg.get("lanes_count", 3))
    speed_limit = float(cfg.get("speed_limit", 50.0))
    enable_lane_change = bool(cfg.get("enable_lane_change", False))
    keep_lateral_heading = bool(cfg.get("keep_lateral_heading", True))
    target_speed_cfg = cfg.get("target_speed", "initial")
    seed = int(cfg.get("seed", 42))

    idm_params = {
        key: float(cfg[key])
        for key in IDM_PARAMETER_KEYS
        if key in cfg
    }

    for idx in range(batch):
        road = Road(
            network=RoadNetwork.straight_road_network(
                lanes_count,
                speed_limit=speed_limit,
            ),
            np_random=np.random.RandomState(seed + idx),
            record_history=False,
        )
        ego0 = init[idx, 0]
        adv0 = init[idx, 1]
        ego_speed, ego_heading = _speed_and_heading(
            ego0,
            keep_lateral_heading=False,
        )
        adv_speed, adv_heading = _speed_and_heading(
            adv0,
            keep_lateral_heading=keep_lateral_heading,
        )
        if target_speed_cfg is None or str(target_speed_cfg).lower() in {
            "initial",
            "context",
        }:
            target_speed = ego_speed
        else:
            target_speed = float(target_speed_cfg)

        ego = IDMVehicle(
            road,
            position=np.asarray([ego0[0], ego0[1]], dtype=np.float64),
            heading=ego_heading,
            speed=ego_speed,
            target_speed=target_speed,
            enable_lane_change=enable_lane_change,
        )
        adversary = ScriptedAdversaryVehicle(
            road,
            position=np.asarray([adv0[0], adv0[1]], dtype=np.float64),
            heading=adv_heading,
            speed=adv_speed,
        )
        for key, value in idm_params.items():
            setattr(ego, key, value)
        ego.LENGTH = float(np.asarray(ego_length, dtype=np.float64)[idx])
        adversary.LENGTH = float(np.asarray(adv_length, dtype=np.float64)[idx])
        if hasattr(ego, "diagonal"):
            ego.diagonal = float(np.sqrt(ego.LENGTH**2 + ego.WIDTH**2))
        if hasattr(adversary, "diagonal"):
            adversary.diagonal = float(
                np.sqrt(adversary.LENGTH**2 + adversary.WIDTH**2)
            )
        road.vehicles.extend([ego, adversary])
        if hasattr(ego, "front_vehicle"):
            ego.front_vehicle = adversary

        for step in range(horizon):
            adv_state = adv[idx, step]
            adv_speed_next, adv_heading_next = _speed_and_heading(
                adv_state,
                keep_lateral_heading=keep_lateral_heading,
            )
            adversary.set_forced_state(
                np.asarray([adv_state[0], adv_state[1]], dtype=np.float64),
                adv_heading_next,
                adv_speed_next,
            )
            road.act()
            road.step(float(dt))
            out[idx, step] = _vehicle_state(ego)
    return out

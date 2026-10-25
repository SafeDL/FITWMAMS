"""Execute background actions on the local HighwayEnv road backend."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.idm_ego import IDM_PARAMETER_KEYS

from .config import WorldModelConfig
from .core.dynamics import HIGHD_INTEGRATION_SPEED_MAX_MPS
from .influence_graph import build_dynamic_interaction_relations
from .reaction_controller import ReactionControllerContext
from .scene_geometry import SceneGeometry, VehicleDimensions
from .tracking import TrackingConfig, tracking_actions, tracking_reference

HIGHWAY_ENV_DYNAMICS_CONTRACT = "kinematic_unicycle_on_highwayenv_road"


def _highway_classes() -> tuple[Any, Any, Any, Any, Any]:
    root = Path(__file__).resolve().parents[2] / "HighwayEnv"
    if not (root / "highway_env").is_dir():
        raise FileNotFoundError(f"local HighwayEnv package is missing: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from highway_env.road.lane import StraightLane
    from highway_env.road.road import Road, RoadNetwork
    from highway_env.vehicle.behavior import IDMVehicle
    from highway_env.vehicle.kinematics import Vehicle

    return StraightLane, Road, RoadNetwork, IDMVehicle, Vehicle


def yaw_rate_to_steering(
    yaw_rate_rps: float,
    speed_mps: float,
    vehicle_length_m: float,
    max_steering_rad: float,
) -> float:
    """Invert HighwayEnv's modified-bicycle yaw-rate relation."""
    speed = max(abs(float(speed_mps)), 1.0e-4)
    sine_beta = (
        float(yaw_rate_rps) * max(float(vehicle_length_m), 1.0e-4) / (2.0 * speed)
    )
    beta = np.arcsin(np.clip(sine_beta, -0.999, 0.999))
    steering = np.arctan(2.0 * np.tan(beta))
    return float(np.clip(steering, -max_steering_rad, max_steering_rad))


def steering_to_yaw_rate(
    steering_rad: float,
    speed_mps: float,
    vehicle_length_m: float,
) -> float:
    """Convert one HighwayEnv low-level steering command to yaw rate."""
    beta = np.arctan(0.5 * np.tan(float(steering_rad)))
    return float(
        float(speed_mps) * np.sin(beta) / max(float(vehicle_length_m) / 2.0, 1.0e-4)
    )


class ControlledBackgroundVehicleMixin:
    """Keeps a learned background action while HighwayEnv advances physics."""

    def set_control(
        self,
        action: dict[str, float],
        control: np.ndarray,
    ) -> None:
        self._external_action = dict(action)
        self._external_control = np.asarray(control, np.float32).copy()
        # The formal runner calls ``Road.step`` after controller inference instead
        # of ``Road.act``. Commit the command immediately so the next physics
        # step executes the learned background action.
        super().act(self._external_action)

    def act(self, action: dict | str | None = None) -> None:
        del action
        super().act(self._external_action)


class UnicycleDynamicsVehicleMixin:
    """Apply the the ``[acceleration, yaw_rate]`` contract on a Highway road.

    Uses the same acceleration-aware unicycle as the vectorized tracker checks,
    while HighwayEnv owns the road, collisions and IDM ego.
    """

    def step(self, dt: float) -> None:
        if self.impact is not None:
            self.position += self.impact
            self.impact = None
            self.crashed = True
        if self.crashed:
            control = np.asarray((-float(self.speed), 0.0), np.float32)
            self.action = {"steering": 0.0, "acceleration": float(control[0])}
        else:
            control = getattr(self, "_external_control", None)
        if control is None:
            control = np.asarray(
                (
                    float(self.action.get("acceleration", 0.0)),
                    steering_to_yaw_rate(
                        float(self.action.get("steering", 0.0)),
                        float(self.speed),
                        float(self.LENGTH),
                    ),
                ),
                np.float32,
            )
        acceleration = float(np.clip(control[0], -8.0, 4.0))
        yaw_rate = float(control[1])
        speed = float(self.speed)
        heading = float(self.heading)
        duration = float(dt)
        acceleration_time = duration
        if acceleration < 0:
            acceleration_time = min(duration, speed / -acceleration)
        elif acceleration > 0:
            acceleration_time = min(
                duration, max(HIGHD_INTEGRATION_SPEED_MAX_MPS - speed, 0) / acceleration
            )
        next_speed = float(
            np.clip(
                speed + acceleration * duration, 0.0, HIGHD_INTEGRATION_SPEED_MAX_MPS
            )
        )
        distance = speed * acceleration_time + 0.5 * acceleration * acceleration_time**2
        distance += next_speed * (duration - acceleration_time)
        self.position += distance * np.asarray((np.cos(heading), np.sin(heading)))
        self.heading = heading + yaw_rate * duration
        self.speed = next_speed
        realized_acceleration = acceleration if acceleration_time == duration else 0.0
        self._external_control = np.asarray(
            (realized_acceleration, yaw_rate), np.float32
        )
        self.on_state_update()


@dataclass(frozen=True)
class HighwayEnvSnapshot:
    states: np.ndarray
    actions: np.ndarray
    highway_actions: np.ndarray
    crashed: np.ndarray
    ego_timer: float | None
    positions: np.ndarray
    headings: np.ndarray
    speeds: np.ndarray
    road_ego_y: float
    impacts: np.ndarray
    pending_impacts: np.ndarray


@dataclass(frozen=True)
class HighwayEnvStep:
    states: np.ndarray
    ego_action: np.ndarray
    background_actions: np.ndarray
    collision: bool
    crashed: np.ndarray
    collision_pairs: np.ndarray
    offroad: np.ndarray


@dataclass(frozen=True)
class HighwayEnvWorldSnapshot:
    states: torch.Tensor
    history: torch.Tensor
    history_valid: torch.Tensor
    reference_index: int
    previous_background_actions: torch.Tensor | None
    controller_runtime_state: dict[str, torch.Tensor] | None
    traffic: tuple[HighwayEnvSnapshot, ...]


class HighwayEnvTraffic:
    """One fixed-slot traffic world advanced by HighwayEnv at 25 Hz.

    The ego is a real ``IDMVehicle`` when ``idm_config`` is supplied.  Each
    valid background slot is a plain HighwayEnv vehicle whose action comes from
    the controller. The adapter has no learned dynamics of its own.
    """

    def __init__(
        self,
        *,
        dt_s: float = 0.04,
        lane_width_m: float = 3.6,
        lanes_count: int = 8,
        speed_limit_mps: float = HIGHD_INTEGRATION_SPEED_MAX_MPS,
        vehicle_length_m: float = 4.8,
        vehicle_width_m: float = 1.8,
        seed: int = 0,
    ) -> None:
        if dt_s <= 0.0 or lane_width_m <= 0.0 or lanes_count < 1:
            raise ValueError("dt_s, lane_width_m and lanes_count must be positive")
        self.dt_s = float(dt_s)
        self.lane_width_m = float(lane_width_m)
        self.lanes_count = int(lanes_count)
        self.speed_limit_mps = float(speed_limit_mps)
        self.vehicle_length_m = float(vehicle_length_m)
        self.vehicle_width_m = float(vehicle_width_m)
        self.seed = int(seed)
        self.road: Any | None = None
        self.ego: Any | None = None
        self.background: dict[int, Any] = {}
        self.valid: np.ndarray | None = None
        self.idm_config: dict[str, Any] | None = None
        self.road_ego_y: float | None = None
        self.scene_geometry: SceneGeometry | None = None
        self.vehicle_dimensions: VehicleDimensions | None = None

    @staticmethod
    def _heading(state: np.ndarray) -> float:
        speed = float(np.hypot(state[2], state[3]))
        return 0.0 if speed < 1.0e-6 else float(np.arctan2(state[3], state[2]))

    @staticmethod
    def _speed(state: np.ndarray) -> float:
        return float(np.hypot(state[2], state[3]))

    def _make_road(self, geometry: SceneGeometry) -> Any:
        StraightLane, Road, RoadNetwork, _, _ = _highway_classes()
        network = RoadNetwork()
        for lane_id, centerline, width in zip(
            geometry.lane_ids, geometry.centerlines, geometry.lane_widths
        ):
            direction = centerline[-1] - centerline[0]
            norm = float(np.linalg.norm(direction))
            if norm < 1.0e-6:
                # Minimal synthetic tests and legacy generated maps sometimes
                # encode only the lateral centre.  The maintained highD road
                # frame is longitudinal x, so complete that explicit straight
                # template without shifting its supplied centre.
                direction = np.asarray((1.0, 0.0), dtype=np.float64)
            else:
                direction /= norm
            midpoint = 0.5 * (centerline[0] + centerline[-1])
            start = midpoint - 10_000.0 * direction
            end = midpoint + 10_000.0 * direction
            network.add_lane(
                "0",
                "1",
                StraightLane(
                    start,
                    end,
                    width=float(width),
                    speed_limit=self.speed_limit_mps,
                ),
            )
        return Road(
            network=network,
            np_random=np.random.RandomState(self.seed),
            record_history=False,
        )

    def _make_vehicle(self, cls: Any, state: np.ndarray, slot: int) -> Any:
        assert self.road is not None
        assert self.vehicle_dimensions is not None
        vehicle = cls(
            self.road,
            position=np.asarray(state[:2], np.float64),
            heading=self._heading(state),
            speed=self._speed(state),
        )
        vehicle.LENGTH = float(self.vehicle_dimensions.length_m[slot])
        vehicle.WIDTH = float(self.vehicle_dimensions.width_m[slot])
        vehicle.diagonal = float(np.hypot(vehicle.LENGTH, vehicle.WIDTH))
        return vehicle

    def reset(
        self,
        initial_states: np.ndarray,
        valid: np.ndarray,
        *,
        idm_config: dict[str, Any] | None = None,
        scene_geometry: SceneGeometry | None = None,
        vehicle_dimensions: VehicleDimensions | np.ndarray | None = None,
    ) -> np.ndarray:
        """Create the HighwayEnv road from a `[7, 6]` physical world state."""
        states = np.asarray(initial_states, np.float64)
        present = np.asarray(valid, bool)
        if states.shape != (7, 6) or present.shape != (7,) or not present[0]:
            raise ValueError(
                "initial state contract is states [7,6], valid [7] with ego present"
            )
        _, _, _, IDMVehicle, Vehicle = _highway_classes()
        self.valid = present.copy()
        self.idm_config = None if idm_config is None else dict(idm_config)
        self.road_ego_y = float(states[0, 1])
        if scene_geometry is None:
            offset = -(self.lanes_count // 2 - 1)
            centers = (
                self.road_ego_y
                + (offset + np.arange(self.lanes_count, dtype=np.float64))
                * self.lane_width_m
            )
            scene_geometry = SceneGeometry.straight_template(
                lane_centers_y=centers, lane_widths=self.lane_width_m
            )
        self.scene_geometry = scene_geometry
        if vehicle_dimensions is None:
            dimensions = VehicleDimensions.constant(
                7, length_m=self.vehicle_length_m, width_m=self.vehicle_width_m
            )
        elif isinstance(vehicle_dimensions, VehicleDimensions):
            dimensions = vehicle_dimensions
        else:
            dimensions = VehicleDimensions.from_array(vehicle_dimensions)
        if len(dimensions.length_m) != 7:
            raise ValueError("vehicle_dimensions must describe all seven slots")
        self.vehicle_dimensions = dimensions
        self.road = self._make_road(scene_geometry)
        if self.idm_config is None:
            self.ego = self._make_vehicle(Vehicle, states[0], 0)
        else:
            target_speed = float(
                self.idm_config.get("target_speed", self._speed(states[0]))
            )
            self.ego = IDMVehicle(
                self.road,
                position=np.asarray(states[0, :2], np.float64),
                heading=self._heading(states[0]),
                speed=self._speed(states[0]),
                target_speed=target_speed,
                enable_lane_change=bool(
                    self.idm_config.get("enable_lane_change", False)
                ),
            )
            self.ego.LENGTH = float(dimensions.length_m[0])
            self.ego.WIDTH = float(dimensions.width_m[0])
            self.ego.diagonal = float(np.hypot(self.ego.LENGTH, self.ego.WIDTH))
            self.ego.MAX_SPEED = HIGHD_INTEGRATION_SPEED_MAX_MPS
            for name in IDM_PARAMETER_KEYS:
                if name in self.idm_config:
                    setattr(self.ego, name, float(self.idm_config[name]))
        unicycle_type = type(
            "UnicycleVehicle", (UnicycleDynamicsVehicleMixin, Vehicle), {}
        )
        background_type = type(
            "ControlledBackgroundVehicle",
            (ControlledBackgroundVehicleMixin, UnicycleDynamicsVehicleMixin, Vehicle),
            {},
        )
        self.background = {}
        if self.idm_config is None:
            self.ego = self._make_vehicle(unicycle_type, states[0], 0)
        vehicles = [self.ego]
        for slot in np.flatnonzero(present[1:]):
            vehicle = self._make_vehicle(
                background_type, states[slot + 1], int(slot + 1)
            )
            vehicle.set_control(
                {"acceleration": 0.0, "steering": 0.0},
                np.zeros(2, np.float32),
            )
            self.background[int(slot)] = vehicle
            vehicles.append(vehicle)
        self.road.vehicles = vehicles
        for slot, vehicle in self._slot_vehicles().items():
            control = self._control_from_state(states[slot])
            action = self._action_dict(control, vehicle)
            if hasattr(vehicle, "set_control"):
                vehicle.set_control(action, control)
            else:
                Vehicle.act(vehicle, action)
        return self.states()

    def _require(self) -> tuple[Any, Any, np.ndarray]:
        if self.road is None or self.ego is None or self.valid is None:
            raise RuntimeError("reset the HighwayEnv traffic world before use")
        return self.road, self.ego, self.valid

    def _control_from_vehicle(self, vehicle: Any) -> np.ndarray:
        if hasattr(vehicle, "_external_control"):
            return np.asarray(vehicle._external_control, np.float32).copy()
        action = vehicle.action if isinstance(vehicle.action, dict) else {}
        acceleration = float(action.get("acceleration", 0.0))
        yaw_rate = steering_to_yaw_rate(
            float(action.get("steering", 0.0)),
            float(vehicle.speed),
            float(vehicle.LENGTH),
        )
        return np.asarray([acceleration, yaw_rate], np.float32)

    @staticmethod
    def _control_from_state(state: np.ndarray) -> np.ndarray:
        speed = float(np.hypot(state[2], state[3]))
        heading = HighwayEnvTraffic._heading(state)
        longitudinal = float(state[4] * np.cos(heading) + state[5] * np.sin(heading))
        lateral = float(-state[4] * np.sin(heading) + state[5] * np.cos(heading))
        return np.asarray((longitudinal, lateral / max(speed, 0.5)), np.float32)

    @staticmethod
    def _action_dict(control: np.ndarray, vehicle: Any) -> dict[str, float]:
        return {
            "acceleration": float(control[0]),
            "steering": yaw_rate_to_steering(
                float(control[1]),
                float(vehicle.speed),
                float(vehicle.LENGTH),
                float(getattr(vehicle, "MAX_STEERING_ANGLE", np.pi / 3.0)),
            ),
        }

    def _slot_vehicles(self) -> dict[int, Any]:
        _, ego, _ = self._require()
        return {
            0: ego,
            **{slot + 1: vehicle for slot, vehicle in self.background.items()},
        }

    def _state_from_vehicle(self, vehicle: Any) -> np.ndarray:
        control = self._control_from_vehicle(vehicle)
        heading = float(vehicle.heading)
        speed = float(vehicle.speed)
        acceleration, yaw_rate = (float(value) for value in control)
        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)
        ax = acceleration * np.cos(heading) - speed * yaw_rate * np.sin(heading)
        ay = acceleration * np.sin(heading) + speed * yaw_rate * np.cos(heading)
        return np.asarray(
            [vehicle.position[0], vehicle.position[1], vx, vy, ax, ay], np.float32
        )

    def states(self) -> np.ndarray:
        _, ego, valid = self._require()
        states = np.zeros((7, 6), np.float32)
        states[0] = self._state_from_vehicle(ego)
        for slot, vehicle in self.background.items():
            states[slot + 1] = self._state_from_vehicle(vehicle)
        return states * valid[:, None]

    def idm_action(self) -> np.ndarray:
        """Ask the HighwayEnv IDM ego for its current low-level command."""
        _, ego, _ = self._require()
        if self.idm_config is None:
            raise RuntimeError("idm_action requires an IDM-configured ego vehicle")
        ego.act()
        return self._control_from_vehicle(ego)

    def step(
        self,
        background_actions: np.ndarray,
        *,
        ego_action: np.ndarray | None = None,
    ) -> HighwayEnvStep:
        """Apply NPC background controls and advance all vehicles in HighwayEnv."""
        road, ego, _ = self._require()
        actions = np.asarray(background_actions, np.float32)
        if actions.shape != (6, 2):
            raise ValueError("background_actions must have shape [6,2]")
        if self.idm_config is None:
            if ego_action is None:
                raise ValueError(
                    "a non-IDM ego requires ego_action [acceleration,yaw_rate]"
                )
            control = np.asarray(ego_action, np.float32)
            if control.shape != (2,):
                raise ValueError("ego_action must have shape [2]")
            ego.act(self._action_dict(control, ego))
            if isinstance(ego, UnicycleDynamicsVehicleMixin):
                ego._external_control = control.copy()
        elif ego_action is None:
            self.idm_action()
        for slot, vehicle in self.background.items():
            control = actions[slot]
            vehicle.set_control(self._action_dict(control, vehicle), control)
        road.step(self.dt_s)
        ego_control = self._control_from_vehicle(ego)
        executed_background = np.zeros((6, 2), np.float32)
        for slot, vehicle in self.background.items():
            executed_background[slot] = self._control_from_vehicle(vehicle)
        crashed = np.zeros(7, dtype=bool)
        vehicles = self._slot_vehicles()
        for slot, vehicle in vehicles.items():
            crashed[slot] = bool(vehicle.crashed)
        pairs = np.zeros((7, 7), dtype=bool)
        items = list(vehicles.items())
        for index, (left_slot, left) in enumerate(items):
            for right_slot, right in items[index + 1 :]:
                intersecting, _, _ = left._is_colliding(right, 0.0)
                pairs[left_slot, right_slot] = pairs[right_slot, left_slot] = bool(
                    intersecting
                )
        assert self.scene_geometry is not None
        assert self.vehicle_dimensions is not None
        offroad = self.scene_geometry.offroad(
            self.states()[:, :2],
            vehicle_width_m=self.vehicle_dimensions.width_m,
        )
        return HighwayEnvStep(
            states=self.states(),
            ego_action=ego_control,
            background_actions=executed_background,
            collision=bool(crashed.any()),
            crashed=crashed,
            collision_pairs=pairs,
            offroad=offroad,
        )

    def snapshot(self) -> HighwayEnvSnapshot:
        _, ego, _ = self._require()
        actions = np.zeros((7, 2), np.float32)
        highway_actions = np.zeros((7, 2), np.float64)
        actions[0] = self._control_from_vehicle(ego)
        crashed = np.zeros(7, bool)
        impacts = np.zeros((7, 2), np.float64)
        pending_impacts = np.zeros(7, bool)
        crashed[0] = bool(ego.crashed)
        for slot, vehicle in self.background.items():
            actions[slot + 1] = self._control_from_vehicle(vehicle)
            crashed[slot + 1] = bool(vehicle.crashed)
        for slot, vehicle in self._slot_vehicles().items():
            highway_actions[slot] = (
                float(vehicle.action.get("acceleration", 0.0)),
                float(vehicle.action.get("steering", 0.0)),
            )
            if vehicle.impact is not None:
                impacts[slot] = vehicle.impact
                pending_impacts[slot] = True
        return HighwayEnvSnapshot(
            states=self.states(),
            actions=actions,
            highway_actions=highway_actions,
            crashed=crashed,
            ego_timer=(None if self.idm_config is None else float(ego.timer)),
            positions=np.stack(
                [self._slot_vehicles().get(slot, ego).position for slot in range(7)]
            ).astype(np.float64),
            headings=np.asarray(
                [self._slot_vehicles().get(slot, ego).heading for slot in range(7)],
                np.float64,
            ),
            speeds=np.asarray(
                [self._slot_vehicles().get(slot, ego).speed for slot in range(7)],
                np.float64,
            ),
            road_ego_y=float(self.road_ego_y),
            impacts=impacts,
            pending_impacts=pending_impacts,
        )

    def restore(self, snapshot: HighwayEnvSnapshot) -> np.ndarray:
        """Restore a deterministic, non-random HighwayEnv traffic snapshot."""
        if self.valid is None:
            raise RuntimeError("reset the HighwayEnv traffic world before restore")
        raw = np.asarray(snapshot.states, np.float64).copy()
        raw[:, :2] = snapshot.positions
        raw[:, 2] = snapshot.speeds * np.cos(snapshot.headings)
        raw[:, 3] = snapshot.speeds * np.sin(snapshot.headings)
        self.reset(
            raw,
            self.valid,
            idm_config=self.idm_config,
            scene_geometry=self.scene_geometry,
            vehicle_dimensions=self.vehicle_dimensions,
        )
        _, ego, _ = self._require()
        for slot, vehicle in self._slot_vehicles().items():
            vehicle.position = snapshot.positions[slot].copy()
            vehicle.heading = float(snapshot.headings[slot])
            vehicle.speed = float(snapshot.speeds[slot])
            vehicle.on_state_update()
            _, _, _, _, Vehicle = _highway_classes()
            Vehicle.act(
                vehicle,
                {
                    "acceleration": float(snapshot.highway_actions[slot, 0]),
                    "steering": float(snapshot.highway_actions[slot, 1]),
                },
            )
            vehicle.crashed = bool(snapshot.crashed[slot])
            vehicle._external_control = snapshot.actions[slot].copy()
            vehicle.impact = (
                snapshot.impacts[slot].copy()
                if snapshot.pending_impacts[slot]
                else None
            )
        if snapshot.ego_timer is not None:
            ego.timer = float(snapshot.ego_timer)
        return self.states()


class HighwayEnvClosedLoopWorld:
    """Track a stored plan and execute irreversible NPC handovers at 25 Hz."""

    def __init__(
        self,
        *,
        device="cpu",
        idm_config=None,
        controller=None,
        tracking_config=None,
        cfg=None,
    ):
        self.device = torch.device(device)
        self.cfg = cfg or WorldModelConfig()
        self.tracking_config = tracking_config or TrackingConfig()
        self.idm_config = None if idm_config is None else dict(idm_config)
        self.controller = None if controller is None else controller.to(self.device)
        self.traffic = []
        self.states = None
        self.valid = None
        self.reference_index = 0

    def _require(self):
        if self.states is None or self.valid is None:
            raise RuntimeError("reset the world before execution")
        return self.states, self.valid

    @staticmethod
    def _clone(value):
        return None if value is None else value.detach().clone()

    @torch.no_grad()
    def reset(
        self,
        initial_states,
        valid,
        soft_reference,
        map_polylines,
        map_polyline_valid,
        *,
        exogenous_state,
        initial_history=None,
        initial_history_valid=None,
        vehicle_dimensions=None,
    ):
        states = torch.as_tensor(
            initial_states, dtype=torch.float32, device=self.device
        )
        present = torch.as_tensor(valid, dtype=torch.bool, device=self.device)
        reference = torch.as_tensor(
            soft_reference, dtype=torch.float32, device=self.device
        )
        if (
            states.ndim != 3
            or states.shape[1:] != (7, 6)
            or present.shape != states.shape[:2]
        ):
            raise ValueError("initial state contract is [batch,7,6]/[batch,7]")
        if (
            reference.ndim != 4
            or reference.shape[:1] != states.shape[:1]
            or reference.shape[2:] != (6, 2)
        ):
            raise ValueError("soft_reference must be [batch,frames,6,2]")
        maps = torch.as_tensor(map_polylines, dtype=torch.float32, device=self.device)
        map_valid = torch.as_tensor(
            map_polyline_valid, dtype=torch.bool, device=self.device
        )
        if maps.ndim == 3:
            maps = maps[None].expand(states.shape[0], -1, -1, -1)
            map_valid = map_valid[None].expand(states.shape[0], -1, -1)
        if maps.ndim != 4 or maps.shape[0] != states.shape[0] or maps.shape[-1] < 2:
            raise ValueError("map_polylines must be [batch,lanes,points,features]")
        if map_valid.shape != maps.shape[:3]:
            raise ValueError("map_polyline_valid must match map_polylines[:3]")
        dimensions = (
            torch.tensor((4.8, 1.8), dtype=states.dtype, device=self.device)
            .view(1, 1, 2)
            .expand(states.shape[0], 7, 2)
            .clone()
            if vehicle_dimensions is None
            else torch.as_tensor(
                vehicle_dimensions, dtype=states.dtype, device=self.device
            )
        )
        if dimensions.ndim == 2:
            dimensions = dimensions[None].expand(states.shape[0], -1, -1)
        if dimensions.shape != (states.shape[0], 7, 2):
            raise ValueError("vehicle_dimensions must be [batch,7,2] or [7,2]")
        if not bool(torch.isfinite(dimensions).all()) or bool(
            (dimensions <= 0.0).any()
        ):
            raise ValueError("vehicle_dimensions must be finite and positive")
        exogenous_state.validate()
        if exogenous_state.batch_size != states.shape[0]:
            raise ValueError(
                "exogenous state and initial states have different batch sizes"
            )
        if (initial_history is None) != (initial_history_valid is None):
            raise ValueError(
                "initial history and its validity mask must be provided together"
            )
        history = None
        history_valid = None
        if initial_history is not None and initial_history_valid is not None:
            history = torch.as_tensor(
                initial_history, dtype=states.dtype, device=self.device
            )
            history_valid = torch.as_tensor(
                initial_history_valid, dtype=torch.bool, device=self.device
            )
            if (
                history.ndim != 4
                or history.shape[0] != states.shape[0]
                or history.shape[2:] != (7, 6)
                or history_valid.shape != history.shape[:3]
                or not 1 <= history.shape[1] <= self.cfg.history_frames
            ):
                raise ValueError(
                    "initial history must be [batch,1..history_frames,7,6] "
                    "with a matching validity mask"
                )
        # Retain HighwayEnv objects while using their exact physical state as
        # the model's first realized history frame.
        self.traffic = []
        self.scene_geometries = tuple(
            SceneGeometry.from_map_polylines(
                maps[index].detach().cpu().numpy(),
                map_valid[index].detach().cpu().numpy(),
            )
            for index in range(states.shape[0])
        )
        self.vehicle_dimensions = dimensions.clone()
        initial_realized: list[np.ndarray] = []
        for index in range(states.shape[0]):
            traffic = HighwayEnvTraffic(dt_s=self.cfg.dt_s, seed=index)
            initial_realized.append(
                traffic.reset(
                    states[index].detach().cpu().numpy(),
                    present[index].detach().cpu().numpy(),
                    idm_config=self.idm_config,
                    scene_geometry=self.scene_geometries[index],
                    vehicle_dimensions=dimensions[index].detach().cpu().numpy(),
                )
            )
            self.traffic.append(traffic)
        self.states = torch.from_numpy(np.stack(initial_realized)).to(self.device)
        self.valid = present.clone()
        self.history = (
            self.states[:, None].clone() if history is None else history.clone()
        )
        self.history_valid = (
            present[:, None].clone() if history_valid is None else history_valid.clone()
        )
        self.reference = reference.clone()
        self.reference_base = self.states[:, 1:, :2].clone()
        self.tracking_reference = torch.as_tensor(
            tracking_reference(
                self.reference_base.cpu().numpy(),
                reference.cpu().numpy(),
                dt=self.cfg.dt_s,
                window=self.tracking_config.derivative_window,
            ),
            device=self.device,
        )
        self.reference_index = 0
        self.map_polylines = maps.clone()
        self.map_polyline_valid = map_valid.clone()
        self.policy_response_innovations = torch.as_tensor(
            exogenous_state.policy_response_innovations,
            dtype=states.dtype,
            device=self.device,
        )
        self.policy_initial_innovations = torch.as_tensor(
            exogenous_state.policy_calibration_innovations[:, 0],
            dtype=states.dtype,
            device=self.device,
        )
        self.previous_background_actions = None
        if self.controller is not None:
            self.controller.restore_runtime({})
        return self.observe()

    def observe(self):
        states, valid = self._require()
        return {
            "agent_states": states.detach().clone(),
            "agent_valid": valid.detach().clone(),
            "map_polylines": self.map_polylines.clone(),
            "map_polyline_valid": self.map_polyline_valid.clone(),
            "reference_index": self.reference_index,
        }

    def _controller_context(self, actions):
        index = min(self.reference_index, self.reference.shape[1] - 1)
        return ReactionControllerContext(
            history=self.history,
            history_valid=self.history_valid,
            current=self.states,
            current_valid=self.valid,
            base_actions=actions,
            cfg=self.cfg,
            previous_background_actions=self.previous_background_actions,
            policy_standard_normal=self.policy_response_innovations[
                :, self.reference_index
            ],
            policy_initial_standard_normal=self.policy_initial_innovations,
            planned_current_xy=(
                self.reference_base if index == 0 else self.reference[:, index - 1]
            ),
            planned_path_xy=self.reference,
            map_polylines=self.map_polylines,
            map_polyline_valid=self.map_polyline_valid,
            vehicle_dimensions=self.vehicle_dimensions,
            interaction_relations=build_dynamic_interaction_relations(
                self.states,
                self.valid,
                self.map_polylines,
                self.map_polyline_valid,
                vehicle_lengths_m=self.vehicle_dimensions[..., 0],
                prediction_horizon_s=self.controller.prediction_horizon_s,
            ),
            response_index=self.reference_index,
        )

    def idm_actions(self):
        if self.idm_config is None:
            raise RuntimeError("idm_actions requires an IDM-configured world")
        return torch.from_numpy(
            np.stack([traffic.idm_action() for traffic in self.traffic])
        ).to(self.device)

    @torch.no_grad()
    def advance_response(self, ego_actions, *, background_action_delta=None):
        states, valid = self._require()
        action = torch.as_tensor(ego_actions, dtype=states.dtype, device=self.device)
        if action.ndim == 2:
            action = action[:, None]
        if tuple(action.shape) != (len(states), 1, 2):
            raise ValueError("ego_actions must have shape [batch,1,2] or [batch,2]")
        if self.reference_index >= self.policy_response_innovations.shape[1]:
            raise RuntimeError("world policy innovations are exhausted")
        index = min(self.reference_index, self.reference.shape[1] - 1)
        base_actions = (
            tracking_actions(
                states[:, 1:],
                self.tracking_reference[:, index],
                self.tracking_config,
                min_acceleration=self.cfg.min_acceleration_mps2,
                max_acceleration=self.cfg.max_acceleration_mps2,
                max_yaw_rate=self.cfg.max_yaw_rate_rps,
            )[:, None]
            * valid[:, None, 1:, None]
        )
        output = (
            None
            if self.controller is None
            else self.controller(self._controller_context(base_actions))
        )
        actions = base_actions if output is None else output.actions
        if background_action_delta is not None:
            # External mechanism experiment, applied after the controller so
            # the bridge cannot observe an intervention before it is realised.
            delta = torch.as_tensor(background_action_delta, device=self.device)
            if delta.shape != actions.shape:
                raise ValueError("background_action_delta must match [batch,1,6,2]")
            actions = actions + delta
            actions[..., 0] = actions[..., 0].clamp(
                self.cfg.min_acceleration_mps2, self.cfg.max_acceleration_mps2
            )
            actions[..., 1] = actions[..., 1].clamp(
                -self.cfg.max_yaw_rate_rps, self.cfg.max_yaw_rate_rps
            )
            actions = actions * valid[:, None, 1:, None]
        steps = [
            traffic.step(
                actions[i, 0].cpu().numpy(), ego_action=action[i, 0].cpu().numpy()
            )
            for i, traffic in enumerate(self.traffic)
        ]

        def tensor(name):
            return torch.from_numpy(
                np.stack([getattr(step, name) for step in steps])
            ).to(self.device)

        self.states = tensor("states")
        self.previous_background_actions = tensor("background_actions")
        self.history = torch.cat((self.history, self.states[:, None]), 1)[
            :, -self.cfg.history_frames :
        ]
        self.history_valid = torch.cat((self.history_valid, valid[:, None]), 1)[
            :, -self.cfg.history_frames :
        ]
        self.reference_index += 1
        result = self.observe()
        result.update(
            agent_state_frames=self.states[:, None],
            background_actions=self.previous_background_actions[:, None],
            base_background_actions=base_actions.clone(),
            response_background_actions=actions.clone(),
            ego_actions=tensor("ego_action")[:, None],
            collision=tensor("collision"),
            crashed=tensor("crashed"),
            collision_pairs=tensor("collision_pairs"),
            offroad=tensor("offroad"),
        )
        for name in (
            "alpha",
            "delta_ax",
            "active",
            "rule_action_ax",
            "autonomous_lane_active",
            "mobil_incentive_mps2",
            "realized_leader_index",
            "takeover_started",
            "takeover_owned",
            "takeover_reason",
            "takeover_motion_changed",
            "takeover_score",
            "takeover_conflict_time_s",
            "takeover_influencer",
            "takeover_interaction_score",
            "takeover_following_score_s",
            "takeover_release_score_per_s",
            "takeover_interaction_time_s",
        ):
            key = name if name.startswith("takeover_") else "controller_" + name
            result[key] = None if output is None else self._clone(getattr(output, name))
        return result

    def snapshot(self):
        return HighwayEnvWorldSnapshot(
            states=self.states.clone(),
            history=self.history.clone(),
            history_valid=self.history_valid.clone(),
            reference_index=self.reference_index,
            previous_background_actions=self._clone(self.previous_background_actions),
            controller_runtime_state=(
                None if self.controller is None else self.controller.snapshot_runtime()
            ),
            traffic=tuple(t.snapshot() for t in self.traffic),
        )

    def restore(self, snapshot):
        if len(snapshot.traffic) != len(self.traffic):
            raise ValueError(
                "snapshot traffic batch size differs from the active world"
            )
        for traffic, saved in zip(self.traffic, snapshot.traffic):
            traffic.restore(saved)
        self.states = snapshot.states.clone().to(self.device)
        self.history = snapshot.history.clone().to(self.device)
        self.history_valid = snapshot.history_valid.clone().to(self.device)
        self.reference_index = snapshot.reference_index
        self.previous_background_actions = self._clone(
            snapshot.previous_background_actions
        )
        if self.controller is not None:
            self.controller.restore_runtime(snapshot.controller_runtime_state or {})
        return self.observe()

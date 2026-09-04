"""Execute HiQR background actions on the local HighwayEnv road backend."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.idm_ego import IDM_PARAMETER_KEYS
from world_model.src.hiqr.filter import FilterState

from .model import DiffusionGuidedHiQR
from .influence_graph import CausalInfluenceGraph, InfluenceGraphState
from .reaction_controller import (
    ReactionController,
    ReactionControllerContext,
    controller_features,
    make_reaction_controller,
)


HIGHWAY_ENV_HIQR_DYNAMICS_CONTRACT = "kinematic_unicycle_on_highwayenv_road"


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
    sine_beta = float(yaw_rate_rps) * max(float(vehicle_length_m), 1.0e-4) / (2.0 * speed)
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


class HiQRBackgroundVehicleMixin:
    """Keeps a learned background action while HighwayEnv advances physics."""

    def set_hiqr_action(
        self,
        action: dict[str, float],
        control: np.ndarray,
    ) -> None:
        self._hiqr_action = dict(action)
        self._hiqr_control = np.asarray(control, np.float32).copy()
        # The formal runner calls ``Road.step`` after HiQR inference instead
        # of ``Road.act``. Commit the command immediately so the next physics
        # step executes the learned background action.
        super().act(self._hiqr_action)

    def act(self, action: dict | str | None = None) -> None:
        del action
        super().act(self._hiqr_action)


class UnicycleDynamicsVehicleMixin:
    """Apply the HiQR ``[acceleration, yaw_rate]`` contract on a Highway road.

    HighwayEnv's stock ``Vehicle`` uses a sideslip bicycle model.  HiQR is
    trained and factually evaluated with ``KinematicTrafficDynamics``: an
    acceleration-aware unicycle.  Retaining the HighwayEnv road, collisions
    and IDM while matching that action contract prevents a hidden plant change
    at the model/environment interface.
    """

    def step(self, dt: float) -> None:
        if self.crashed:
            self.action = {"steering": 0.0, "acceleration": -float(self.speed)}
        control = getattr(self, "_hiqr_control", None)
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
        self.position += np.asarray(
            (
                speed * np.cos(heading) + 0.5 * acceleration * np.cos(heading) * duration,
                speed * np.sin(heading) + 0.5 * acceleration * np.sin(heading) * duration,
            )
        ) * duration
        self.heading = heading + yaw_rate * duration
        self.speed = float(np.clip(speed + acceleration * duration, 0.0, 50.0))
        self._hiqr_control = np.asarray((acceleration, yaw_rate), np.float32)
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


@dataclass(frozen=True)
class HighwayEnvStep:
    states: np.ndarray
    ego_action: np.ndarray
    background_actions: np.ndarray
    collision: bool
    crashed: np.ndarray


@dataclass(frozen=True)
class HighwayEnvWorldSnapshot:
    states: torch.Tensor
    history: torch.Tensor
    history_valid: torch.Tensor
    reference_index: int
    filter_global: torch.Tensor | None
    filter_agents: torch.Tensor | None
    slow_scene: torch.Tensor | None
    slow_scene_noise: torch.Tensor | None
    agent_noise_state: torch.Tensor | None
    agent_style_state: torch.Tensor | None
    previous_current: torch.Tensor | None
    committed_ego_controls: torch.Tensor
    intervention_memory: torch.Tensor | None
    lateral_intervention_memory: torch.Tensor | None
    reaction_enabled: torch.Tensor | None
    reaction_phase: torch.Tensor | None
    reaction_age_frames: torch.Tensor | None
    reaction_recovery_remaining: torch.Tensor | None
    previous_background_actions: torch.Tensor | None
    intervention_armed: torch.Tensor | None
    influence_state: InfluenceGraphState | None
    traffic: tuple[HighwayEnvSnapshot, ...]


class HighwayEnvTraffic:
    """One fixed-slot traffic world advanced by HighwayEnv at 25 Hz.

    The ego is a real ``IDMVehicle`` when ``idm_config`` is supplied.  Each
    valid background slot is a plain HighwayEnv vehicle whose action comes from
    HiQR.  The adapter deliberately has no learned dynamics of its own.
    """

    def __init__(
        self,
        *,
        dt_s: float = 0.04,
        lane_width_m: float = 3.6,
        lanes_count: int = 8,
        speed_limit_mps: float = 50.0,
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

    @staticmethod
    def _heading(state: np.ndarray) -> float:
        speed = float(np.hypot(state[2], state[3]))
        return 0.0 if speed < 1.0e-6 else float(np.arctan2(state[3], state[2]))

    @staticmethod
    def _speed(state: np.ndarray) -> float:
        return float(np.hypot(state[2], state[3]))

    def _make_road(self, ego_y: float) -> Any:
        StraightLane, Road, RoadNetwork, _, _ = _highway_classes()
        network = RoadNetwork()
        offset = -(self.lanes_count // 2 - 1)
        for lane in range(self.lanes_count):
            center = float(ego_y + (offset + lane) * self.lane_width_m)
            network.add_lane(
                "0",
                "1",
                StraightLane(
                    np.asarray([-10_000.0, center]),
                    np.asarray([10_000.0, center]),
                    width=self.lane_width_m,
                    speed_limit=self.speed_limit_mps,
                ),
            )
        return Road(
            network=network,
            np_random=np.random.RandomState(self.seed),
            record_history=False,
        )

    def _make_vehicle(self, cls: Any, state: np.ndarray) -> Any:
        assert self.road is not None
        vehicle = cls(
            self.road,
            position=np.asarray(state[:2], np.float64),
            heading=self._heading(state),
            speed=self._speed(state),
        )
        vehicle.LENGTH = self.vehicle_length_m
        vehicle.WIDTH = self.vehicle_width_m
        vehicle.diagonal = float(np.hypot(vehicle.LENGTH, vehicle.WIDTH))
        return vehicle

    def reset(
        self,
        initial_states: np.ndarray,
        valid: np.ndarray,
        *,
        idm_config: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Create the HighwayEnv road from a `[7, 6]` physical world state."""
        states = np.asarray(initial_states, np.float64)
        present = np.asarray(valid, bool)
        if states.shape != (7, 6) or present.shape != (7,) or not present[0]:
            raise ValueError("initial state contract is states [7,6], valid [7] with ego present")
        _, _, _, IDMVehicle, Vehicle = _highway_classes()
        self.valid = present.copy()
        self.idm_config = None if idm_config is None else dict(idm_config)
        self.road_ego_y = float(states[0, 1])
        self.road = self._make_road(self.road_ego_y)
        if self.idm_config is None:
            self.ego = self._make_vehicle(Vehicle, states[0])
        else:
            target_speed = float(self.idm_config.get("target_speed", self._speed(states[0])))
            self.ego = IDMVehicle(
                self.road,
                position=np.asarray(states[0, :2], np.float64),
                heading=self._heading(states[0]),
                speed=self._speed(states[0]),
                target_speed=target_speed,
                enable_lane_change=bool(self.idm_config.get("enable_lane_change", False)),
            )
            self.ego.LENGTH = self.vehicle_length_m
            self.ego.WIDTH = self.vehicle_width_m
            self.ego.diagonal = float(np.hypot(self.ego.LENGTH, self.ego.WIDTH))
            for name in IDM_PARAMETER_KEYS:
                if name in self.idm_config:
                    setattr(self.ego, name, float(self.idm_config[name]))
        unicycle_type = type("HiQRUnicycleVehicle", (UnicycleDynamicsVehicleMixin, Vehicle), {})
        background_type = type(
            "HiQRBackgroundVehicle",
            (HiQRBackgroundVehicleMixin, UnicycleDynamicsVehicleMixin, Vehicle),
            {},
        )
        self.background = {}
        if self.idm_config is None:
            self.ego = self._make_vehicle(unicycle_type, states[0])
        vehicles = [self.ego]
        for slot in np.flatnonzero(present[1:]):
            vehicle = self._make_vehicle(background_type, states[slot + 1])
            vehicle.set_hiqr_action(
                {"acceleration": 0.0, "steering": 0.0},
                np.zeros(2, np.float32),
            )
            self.background[int(slot)] = vehicle
            vehicles.append(vehicle)
        self.road.vehicles = vehicles
        for slot, vehicle in self._slot_vehicles().items():
            control = self._control_from_state(states[slot])
            action = self._action_dict(control, vehicle)
            if hasattr(vehicle, "set_hiqr_action"):
                vehicle.set_hiqr_action(action, control)
            else:
                Vehicle.act(vehicle, action)
        return self.states()

    def _require(self) -> tuple[Any, Any, np.ndarray]:
        if self.road is None or self.ego is None or self.valid is None:
            raise RuntimeError("reset the HighwayEnv traffic world before use")
        return self.road, self.ego, self.valid

    def _control_from_vehicle(self, vehicle: Any) -> np.ndarray:
        if hasattr(vehicle, "_hiqr_control"):
            return np.asarray(vehicle._hiqr_control, np.float32).copy()
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
        return {0: ego, **{slot + 1: vehicle for slot, vehicle in self.background.items()}}

    def _state_from_vehicle(self, vehicle: Any) -> np.ndarray:
        control = self._control_from_vehicle(vehicle)
        heading = float(vehicle.heading)
        speed = float(vehicle.speed)
        acceleration, yaw_rate = (float(value) for value in control)
        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)
        ax = acceleration * np.cos(heading) - speed * yaw_rate * np.sin(heading)
        ay = acceleration * np.sin(heading) + speed * yaw_rate * np.cos(heading)
        return np.asarray([vehicle.position[0], vehicle.position[1], vx, vy, ax, ay], np.float32)

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
        """Apply HiQR background controls and advance all vehicles in HighwayEnv."""
        road, ego, _ = self._require()
        actions = np.asarray(background_actions, np.float32)
        if actions.shape != (6, 2):
            raise ValueError("background_actions must have shape [6,2]")
        if self.idm_config is None:
            if ego_action is None:
                raise ValueError("a non-IDM ego requires ego_action [acceleration,yaw_rate]")
            control = np.asarray(ego_action, np.float32)
            if control.shape != (2,):
                raise ValueError("ego_action must have shape [2]")
            ego.act(self._action_dict(control, ego))
            if isinstance(ego, UnicycleDynamicsVehicleMixin):
                ego._hiqr_control = control.copy()
        elif ego_action is None:
            self.idm_action()
        for slot, vehicle in self.background.items():
            control = actions[slot]
            vehicle.set_hiqr_action(self._action_dict(control, vehicle), control)
        ego_control = self._control_from_vehicle(ego)
        executed_background = np.zeros((6, 2), np.float32)
        for slot, vehicle in self.background.items():
            executed_background[slot] = self._control_from_vehicle(vehicle)
        road.step(self.dt_s)
        crashed = np.zeros(7, dtype=bool)
        for slot, vehicle in self._slot_vehicles().items():
            crashed[slot] = bool(vehicle.crashed)
        return HighwayEnvStep(
            states=self.states(),
            ego_action=ego_control,
            background_actions=executed_background,
            collision=bool(crashed.any()),
            crashed=crashed,
        )

    def snapshot(self) -> HighwayEnvSnapshot:
        _, ego, _ = self._require()
        actions = np.zeros((7, 2), np.float32)
        highway_actions = np.zeros((7, 2), np.float64)
        actions[0] = self._control_from_vehicle(ego)
        crashed = np.zeros(7, bool)
        crashed[0] = bool(ego.crashed)
        for slot, vehicle in self.background.items():
            actions[slot + 1] = self._control_from_vehicle(vehicle)
            crashed[slot + 1] = bool(vehicle.crashed)
        for slot, vehicle in self._slot_vehicles().items():
            highway_actions[slot] = (
                float(vehicle.action.get("acceleration", 0.0)),
                float(vehicle.action.get("steering", 0.0)),
            )
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
        )

    def restore(self, snapshot: HighwayEnvSnapshot) -> np.ndarray:
        """Restore a deterministic, non-random HighwayEnv traffic snapshot."""
        if self.valid is None:
            raise RuntimeError("reset the HighwayEnv traffic world before restore")
        raw = np.asarray(snapshot.states, np.float64).copy()
        raw[:, :2] = snapshot.positions
        raw[:, 2] = snapshot.speeds * np.cos(snapshot.headings)
        raw[:, 3] = snapshot.speeds * np.sin(snapshot.headings)
        raw[0, 1] = float(snapshot.road_ego_y)
        self.reset(raw, self.valid, idm_config=self.idm_config)
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
        if snapshot.ego_timer is not None:
            ego.timer = float(snapshot.ego_timer)
        return self.states()


class HighwayEnvClosedLoopWorld:
    """Run batched HiQR inference while HighwayEnv advances each traffic scene.

    HiQR remains vectorized on the selected torch device.  HighwayEnv owns the
    physical state of each scene, so its vehicle states are synchronized into a
    single tensor before every response call and after every road step.
    """

    def __init__(
        self,
        model: DiffusionGuidedHiQR,
        *,
        device: str | torch.device = "cpu",
        idm_config: dict[str, Any] | None = None,
        controller: ReactionController | str | None = None,
        reaction_min_frames: int = 5,
        reaction_max_frames: int = 75,
        reaction_recovery_frames: int = 15,
        reaction_safety_ttc_s: float = 2.0,
        reaction_release_ttc_s: float = 4.0,
        influence_radius_m: float = 50.0,
        influence_secondary_radius_m: float = 35.0,
        influence_prediction_horizon_s: float = 1.5,
        influence_stable_release_frames: int = 13,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.idm_config = None if idm_config is None else dict(idm_config)
        # ``None`` preserves historical callers.  An explicit controller mode
        # always requests decoder base actions and applies exactly one hook.
        if isinstance(controller, str):
            controller = make_reaction_controller(
                controller, adapter_logit=self.model.decoder.intervention_logit
            )
        self.controller = None if controller is None else controller.to(self.device)
        if not (0 <= reaction_min_frames <= reaction_max_frames):
            raise ValueError("reaction authority frames must satisfy 0 <= min <= max")
        if reaction_recovery_frames < 1 or reaction_safety_ttc_s <= 0.0 or reaction_release_ttc_s < reaction_safety_ttc_s:
            raise ValueError("reaction recovery frames must be positive and release TTC must be >= safety TTC")
        self.reaction_min_frames = int(reaction_min_frames)
        self.reaction_max_frames = int(reaction_max_frames)
        self.reaction_recovery_frames = int(reaction_recovery_frames)
        self.reaction_safety_ttc_s = float(reaction_safety_ttc_s)
        self.reaction_release_ttc_s = float(reaction_release_ttc_s)
        self.influence_graph = CausalInfluenceGraph(
            radius_m=influence_radius_m,
            secondary_radius_m=influence_secondary_radius_m,
            prediction_horizon_s=influence_prediction_horizon_s,
            release_ttc_s=reaction_release_ttc_s,
            stable_release_frames=influence_stable_release_frames,
            recovery_frames=reaction_recovery_frames,
        )
        self.traffic: list[HighwayEnvTraffic] = []
        self.states: torch.Tensor | None = None
        self.valid: torch.Tensor | None = None
        self.history: torch.Tensor | None = None
        self.history_valid: torch.Tensor | None = None
        self.reference: torch.Tensor | None = None
        self.reference_base: torch.Tensor | None = None
        self.reference_index = 0
        self.map_polylines: torch.Tensor | None = None
        self.map_polyline_valid: torch.Tensor | None = None
        self.filter_state: FilterState | None = None
        self.slow_scene: torch.Tensor | None = None
        self.slow_scene_noise: torch.Tensor | None = None
        self.agent_noise_state: torch.Tensor | None = None
        self.agent_style_state: torch.Tensor | None = None
        self.previous_current: torch.Tensor | None = None
        self.committed_ego_controls: torch.Tensor | None = None
        self.intervention_memory: torch.Tensor | None = None
        self.lateral_intervention_memory: torch.Tensor | None = None
        self.response_innovations: torch.Tensor | None = None
        self.response_agent_innovations: torch.Tensor | None = None
        self.deterministic_response = False
        self.reaction_enabled: torch.Tensor | None = None
        # 0=inactive, 1=engaged (response/safety), 2=recovery (return to plan).
        self.reaction_phase: torch.Tensor | None = None
        self.reaction_age_frames: torch.Tensor | None = None
        self.reaction_recovery_remaining: torch.Tensor | None = None
        self.previous_background_actions: torch.Tensor | None = None
        self.intervention_armed: torch.Tensor | None = None
        self.influence_state: InfluenceGraphState | None = None

    def set_reaction_enabled(self, enabled: torch.Tensor | np.ndarray | bool) -> None:
        """Legacy manual authority override; PPO uses causal registration below."""
        states, _ = self._require()
        value = torch.as_tensor(enabled, dtype=torch.bool, device=self.device)
        if value.ndim == 0:
            value = value.expand(states.shape[0])
        if tuple(value.shape) != (states.shape[0],):
            raise ValueError("reaction enable mask must be scalar or [batch]")
        self.intervention_armed = value.clone()
        expanded = value[:, None].expand(-1, 6)
        self.reaction_enabled = expanded.clone()
        self.reaction_phase = expanded.long()
        self.reaction_age_frames = torch.zeros_like(expanded, dtype=torch.long)
        self.reaction_recovery_remaining = torch.zeros_like(expanded, dtype=torch.long)

    def register_executed_ego_intervention(
        self, executed: torch.Tensor | np.ndarray | bool,
    ) -> None:
        """Arm dynamic influence monitoring after an ADS action has executed."""
        states, _ = self._require()
        event = torch.as_tensor(executed, dtype=torch.bool, device=self.device)
        if event.ndim == 0:
            event = event.expand(states.shape[0])
        if tuple(event.shape) != (states.shape[0],):
            raise ValueError("executed intervention mask must be scalar or [batch]")
        if self.intervention_armed is None:
            self.intervention_armed = torch.zeros_like(event)
        # This method is called after ``advance_response``.  Therefore this
        # latch cannot expose the command to the background action that ran in
        # the same tick; the graph first sees it at the next response boundary.
        self.intervention_armed = self.intervention_armed | event

    def _controller_context(self, response) -> ReactionControllerContext:
        assert self.history is not None and self.history_valid is not None
        assert self.states is not None and self.valid is not None
        assert self.committed_ego_controls is not None
        assert self.intervention_armed is not None
        self.influence_state = self.influence_graph.update(
            self.states,
            self.valid,
            self.history,
            self.intervention_armed,
            self.influence_state,
            self.previous_background_actions,
        )
        self.reaction_enabled = self.influence_state.authority > 0.0
        self.reaction_phase = self.influence_state.phase
        self.reaction_age_frames = self.influence_state.age_frames
        self.reaction_recovery_remaining = self.influence_state.recovery_remaining
        return ReactionControllerContext(
            history=self.history, history_valid=self.history_valid,
            current=self.states, current_valid=self.valid,
            committed_ego_controls=self.committed_ego_controls,
            base_actions=response.actions, reference_actions=response.reference_actions,
            intervention_trigger=response.intervention_trigger,
            intervention_memory=response.intervention_memory,
            lateral_intervention_memory=response.lateral_intervention_memory,
            agent_style_state=response.agent_style_state,
            response_field_gain=response.response_field_gain,
            response_sensitivity_bounds=self.model.response_sensitivity_bounds,
            adapter_gain=torch.sigmoid(self.model.decoder.intervention_logit), cfg=self.model.cfg,
            reaction_enabled=self.reaction_enabled,
            reaction_phase=self.reaction_phase,
            reaction_age_frames=self.reaction_age_frames,
            reaction_max_frames=self.reaction_max_frames,
            reaction_release_ttc_s=self.reaction_release_ttc_s,
            reaction_recovery_remaining=self.reaction_recovery_remaining,
            reaction_recovery_frames=self.reaction_recovery_frames,
            previous_background_actions=self.previous_background_actions,
            influence_authority=self.influence_state.authority,
            influence_role=self.influence_state.role,
            influence_parent=self.influence_state.parent,
            influence_direct=self.influence_state.direct,
            influence_secondary=self.influence_state.secondary,
            influence_predicted_ttc_s=self.influence_state.predicted_ttc_s,
            influence_predicted_min_gap_m=self.influence_state.predicted_min_gap_m,
        )

    def _require(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.states is None or self.valid is None:
            raise RuntimeError("reset the HighwayEnv closed-loop world before stepping")
        return self.states, self.valid

    @staticmethod
    def _clone(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else value.detach().clone()

    @torch.no_grad()
    def reset(
        self,
        initial_states: torch.Tensor,
        valid: torch.Tensor,
        soft_reference: torch.Tensor,
        map_polylines: torch.Tensor,
        map_polyline_valid: torch.Tensor,
        *,
        exogenous_state: Any,
        initial_history: torch.Tensor | None = None,
        initial_history_valid: torch.Tensor | None = None,
        committed_ego_controls: torch.Tensor | None = None,
        deterministic_response: bool = False,
    ) -> dict[str, torch.Tensor | int]:
        states = torch.as_tensor(initial_states, dtype=torch.float32, device=self.device)
        present = torch.as_tensor(valid, dtype=torch.bool, device=self.device)
        reference = torch.as_tensor(soft_reference, dtype=torch.float32, device=self.device)
        if states.ndim != 3 or states.shape[1:] != (7, 6) or present.shape != states.shape[:2]:
            raise ValueError("initial state contract is [batch,7,6]/[batch,7]")
        if reference.ndim != 4 or reference.shape[:1] != states.shape[:1] or reference.shape[2:] != (6, 2):
            raise ValueError("soft_reference must be [batch,frames,6,2]")
        exogenous_state.validate(
            response_steps=exogenous_state.response_steps,
            scene_dim=self.model.cfg.scene_latent_dim,
            agent_dim=self.model.cfg.agent_latent_dim,
        )
        if exogenous_state.batch_size != states.shape[0]:
            raise ValueError("exogenous state and initial states have different batch sizes")
        if (initial_history is None) != (initial_history_valid is None):
            raise ValueError("initial history and its validity mask must be provided together")
        history = None
        history_valid = None
        if initial_history is not None and initial_history_valid is not None:
            history = torch.as_tensor(initial_history, dtype=states.dtype, device=self.device)
            history_valid = torch.as_tensor(
                initial_history_valid, dtype=torch.bool, device=self.device
            )
            if (
                history.ndim != 4
                or history.shape[0] != states.shape[0]
                or history.shape[2:] != (7, 6)
                or history_valid.shape != history.shape[:3]
                or not 1 <= history.shape[1] <= self.model.cfg.history_frames
            ):
                raise ValueError(
                    "initial history must be [batch,1..history_frames,7,6] "
                    "with a matching validity mask"
                )
        controls = None
        if committed_ego_controls is not None:
            controls = torch.as_tensor(
                committed_ego_controls, dtype=states.dtype, device=self.device
            )
            if controls.ndim != 3 or controls.shape[:1] != states.shape[:1] or controls.shape[-1] != 2:
                raise ValueError("committed ego controls must be [batch,past_frames,2]")
        # Retain HighwayEnv objects while using their exact physical state as
        # the model's first realized history frame.
        self.traffic = []
        initial_realized: list[np.ndarray] = []
        for index in range(states.shape[0]):
            traffic = HighwayEnvTraffic(dt_s=self.model.cfg.dt_s, seed=index)
            initial_realized.append(
                traffic.reset(
                    states[index].detach().cpu().numpy(),
                    present[index].detach().cpu().numpy(),
                    idm_config=self.idm_config,
                )
            )
            self.traffic.append(traffic)
        self.states = torch.from_numpy(np.stack(initial_realized)).to(self.device)
        self.valid = present.clone()
        self.history = self.states[:, None].clone() if history is None else history.clone()
        self.history_valid = (
            present[:, None].clone() if history_valid is None else history_valid.clone()
        )
        self.reference = reference.clone()
        self.reference_base = self.states[:, 1:, :2].clone()
        self.reference_index = 0
        self.map_polylines = torch.as_tensor(map_polylines, dtype=torch.float32, device=self.device)
        self.map_polyline_valid = torch.as_tensor(map_polyline_valid, dtype=torch.bool, device=self.device)
        if self.map_polylines.ndim == 3:
            self.map_polylines = self.map_polylines[None]
            self.map_polyline_valid = self.map_polyline_valid[None]
        self.filter_state = None
        self.slow_scene = None
        self.slow_scene_noise = None
        self.agent_noise_state = None
        self.agent_style_state = None
        self.previous_current = None
        self.committed_ego_controls = (
            torch.zeros((len(self.traffic), 1, 2), dtype=states.dtype, device=self.device)
            if controls is None
            else controls.clone()
        )
        self.intervention_memory = None
        self.lateral_intervention_memory = None
        self.response_innovations = torch.as_tensor(
            exogenous_state.scene_innovations,
            dtype=states.dtype,
            device=self.device,
        )
        self.response_agent_innovations = torch.as_tensor(
            exogenous_state.agent_response_innovations,
            dtype=states.dtype,
            device=self.device,
        )
        self.deterministic_response = bool(deterministic_response)
        self.reaction_enabled = torch.zeros((len(self.traffic), 6), dtype=torch.bool, device=self.device)
        self.reaction_phase = torch.zeros((len(self.traffic), 6), dtype=torch.long, device=self.device)
        self.reaction_age_frames = torch.zeros((len(self.traffic), 6), dtype=torch.long, device=self.device)
        self.reaction_recovery_remaining = torch.zeros((len(self.traffic), 6), dtype=torch.long, device=self.device)
        self.previous_background_actions = None
        self.intervention_armed = torch.zeros(len(self.traffic), dtype=torch.bool, device=self.device)
        self.influence_state = InfluenceGraphState.empty(len(self.traffic), device=self.device)
        return self.observe()

    def observe(self) -> dict[str, torch.Tensor | int]:
        states, valid = self._require()
        return {
            "agent_states": states.detach().clone(),
            "agent_valid": valid.detach().clone(),
            "reference_index": self.reference_index,
        }

    def _preview(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.reference is not None and self.reference_base is not None
        start = self.reference_index
        preview = self.reference[:, start : start + self.model.cfg.preview_frames]
        if preview.shape[1] < self.model.cfg.preview_frames:
            preview = torch.cat(
                (
                    preview,
                    preview[:, -1:].expand(-1, self.model.cfg.preview_frames - preview.shape[1], -1, -1),
                ),
                dim=1,
            )
        base = self.reference_base if start == 0 else self.reference[:, start - 1]
        return preview, base

    def idm_actions(self) -> torch.Tensor:
        if self.idm_config is None:
            raise RuntimeError("idm_actions requires an IDM-configured HighwayEnv world")
        return torch.from_numpy(np.stack([traffic.idm_action() for traffic in self.traffic])).to(self.device)

    @torch.no_grad()
    def advance_response(self, ego_actions: torch.Tensor) -> dict[str, torch.Tensor | int]:
        states, valid = self._require()
        assert self.history is not None and self.history_valid is not None
        assert self.committed_ego_controls is not None
        action = torch.as_tensor(ego_actions, dtype=states.dtype, device=self.device)
        if action.ndim == 2:
            action = action[:, None]
        expected = (states.shape[0], self.model.cfg.execute_frames, 2)
        if tuple(action.shape) != expected:
            raise ValueError(f"ego_actions must have shape {expected}")
        if self.reference_index >= self.response_agent_innovations.shape[1]:
            raise RuntimeError("world exogenous agent innovations are exhausted")
        assert self.response_innovations is not None and self.response_agent_innovations is not None
        preview, base = self._preview()
        refresh = self.reference_index % self.model.cfg.scene_refresh_responses == 0
        scene_noise = (
            self.response_innovations[:, self.reference_index // self.model.cfg.scene_refresh_responses]
            if refresh
            else torch.zeros(
                (states.shape[0], self.model.cfg.scene_latent_dim),
                dtype=states.dtype,
                device=self.device,
            )
        )
        agent_noise = self.response_agent_innovations[:, self.reference_index]
        response = self.model(
            self.history,
            self.history_valid,
            states,
            valid,
            preview,
            base,
            self.map_polylines,
            self.map_polyline_valid,
            filter_state=self.filter_state,
            previous_current=self.previous_current,
            slow_scene=self.slow_scene,
            slow_scene_noise=self.slow_scene_noise,
            agent_noise_state=self.agent_noise_state,
            agent_style_state=self.agent_style_state,
            committed_ego_controls=self.committed_ego_controls,
            intervention_memory=self.intervention_memory,
            lateral_intervention_memory=self.lateral_intervention_memory,
            response_index=self.reference_index,
            scene_standard_normal=scene_noise,
            agent_standard_normal=agent_noise,
            deterministic=self.deterministic_response,
            apply_intervention_adapter=self.controller is None,
            # The learned frozen HiQR ego-response actuator is part of the
            # factual checkpoint.  Only the migrated handcrafted adapter is
            # mutually exclusive with a post-HiQR controller.
            apply_explicit_ego_response=True,
        )
        self.filter_state = response.filter_state
        self.slow_scene = response.slow_scene
        self.slow_scene_noise = response.slow_scene_noise
        self.agent_noise_state = response.agent_noise_state
        self.agent_style_state = response.agent_style_state
        self.intervention_memory = response.intervention_memory
        self.lateral_intervention_memory = response.lateral_intervention_memory
        self.previous_current = states.detach().clone()
        base_actions = response.actions
        controller_output = None
        if self.controller is not None:
            controller_context = self._controller_context(response)
            controller_output = self.controller(controller_context, deterministic=self.deterministic_response)
            response_actions = controller_output.actions
        else:
            response_actions = base_actions
        realized: list[np.ndarray] = []
        executed_ego: list[np.ndarray] = []
        executed_background: list[np.ndarray] = []
        collisions: list[bool] = []
        crashed: list[np.ndarray] = []
        for index, traffic in enumerate(self.traffic):
            step = traffic.step(
                response_actions[index, 0].detach().cpu().numpy(),
                ego_action=action[index, 0].detach().cpu().numpy(),
            )
            realized.append(step.states)
            executed_ego.append(step.ego_action)
            executed_background.append(step.background_actions)
            collisions.append(bool(step.collision))
            crashed.append(step.crashed)
        self.states = torch.from_numpy(np.stack(realized)).to(self.device)
        self.previous_background_actions = torch.from_numpy(np.stack(executed_background)).to(self.device)
        self.committed_ego_controls = torch.cat((self.committed_ego_controls, action), dim=1)[
            :, -self.model.cfg.intervention_trigger_history_frames - 1 :
        ]
        self.history = torch.cat((self.history, self.states[:, None]), dim=1)[
            :, -self.model.cfg.history_frames :
        ]
        self.history_valid = torch.cat((self.history_valid, valid[:, None]), dim=1)[
            :, -self.model.cfg.history_frames :
        ]
        self.reference_index += 1
        result = self.observe()
        result.update(
            {
                "agent_state_frames": self.states[:, None],
                "background_actions": torch.from_numpy(np.stack(executed_background)).to(self.device)[:, None],
                "base_background_actions": base_actions.detach().clone(),
                "ego_actions": torch.from_numpy(np.stack(executed_ego)).to(self.device)[:, None],
                "response_mean": response.mean,
                "response_std": response.std,
                "rebased_preview": response.rebased_preview,
                "intervention_trigger": response.intervention_trigger.detach().clone(),
                "intervention_memory": response.intervention_memory.detach().clone(),
                "collision": torch.as_tensor(collisions, device=self.device, dtype=torch.bool),
                "crashed": torch.from_numpy(np.stack(crashed)).to(self.device),
                "controller_alpha": None if controller_output is None else controller_output.alpha.detach().clone(),
                "controller_delta_ax": None if controller_output is None else controller_output.delta_ax.detach().clone(),
                "controller_active": None if controller_output is None else controller_output.active.detach().clone(),
                "controller_phase": self.reaction_phase.detach().clone(),
                "controller_age_frames": self.reaction_age_frames.detach().clone(),
                "controller_features": None if controller_output is None else (
                    controller_features(controller_context) if controller_output.policy_features is None else controller_output.policy_features
                ).detach().clone(),
                "controller_raw_action": None if controller_output is None or controller_output.raw_action is None else controller_output.raw_action.detach().clone(),
                "controller_log_prob": None if controller_output is None or controller_output.log_prob is None else controller_output.log_prob.detach().clone(),
                "controller_entropy": None if controller_output is None or controller_output.entropy is None else controller_output.entropy.detach().clone(),
                "controller_value": None if controller_output is None or controller_output.value is None else controller_output.value.detach().clone(),
                "controller_rule_action_ax": None if controller_output is None or controller_output.rule_action_ax is None else controller_output.rule_action_ax.detach().clone(),
                "controller_natural_kl": None if controller_output is None or controller_output.natural_kl is None else controller_output.natural_kl.detach().clone(),
                "controller_desired_action_ax": None if controller_output is None or controller_output.desired_action_ax is None else controller_output.desired_action_ax.detach().clone(),
                "influence_authority": None if self.influence_state is None else self.influence_state.authority.detach().clone(),
                "influence_role": None if self.influence_state is None else self.influence_state.role.detach().clone(),
                "influence_parent": None if self.influence_state is None else self.influence_state.parent.detach().clone(),
                "influence_direct": None if self.influence_state is None else self.influence_state.direct.detach().clone(),
                "influence_secondary": None if self.influence_state is None else self.influence_state.secondary.detach().clone(),
                "influence_predicted_ttc_s": None if self.influence_state is None else self.influence_state.predicted_ttc_s.detach().clone(),
                "influence_predicted_min_gap_m": None if self.influence_state is None else self.influence_state.predicted_min_gap_m.detach().clone(),
            }
        )
        return result

    def snapshot(self) -> HighwayEnvWorldSnapshot:
        states, _ = self._require()
        assert self.history is not None and self.history_valid is not None
        assert self.committed_ego_controls is not None
        return HighwayEnvWorldSnapshot(
            states=states.detach().clone(),
            history=self.history.detach().clone(),
            history_valid=self.history_valid.detach().clone(),
            reference_index=self.reference_index,
            filter_global=None if self.filter_state is None else self.filter_state.global_hidden.detach().clone(),
            filter_agents=None if self.filter_state is None else self.filter_state.agent_hidden.detach().clone(),
            slow_scene=self._clone(self.slow_scene),
            slow_scene_noise=self._clone(self.slow_scene_noise),
            agent_noise_state=self._clone(self.agent_noise_state),
            agent_style_state=self._clone(self.agent_style_state),
            previous_current=self._clone(self.previous_current),
            committed_ego_controls=self.committed_ego_controls.detach().clone(),
            intervention_memory=self._clone(self.intervention_memory),
            lateral_intervention_memory=self._clone(self.lateral_intervention_memory),
            reaction_enabled=self._clone(self.reaction_enabled),
            reaction_phase=self._clone(self.reaction_phase),
            reaction_age_frames=self._clone(self.reaction_age_frames),
            reaction_recovery_remaining=self._clone(self.reaction_recovery_remaining),
            previous_background_actions=self._clone(self.previous_background_actions),
            intervention_armed=self._clone(self.intervention_armed),
            influence_state=self.influence_state,
            traffic=tuple(traffic.snapshot() for traffic in self.traffic),
        )

    def restore(self, snapshot: HighwayEnvWorldSnapshot) -> dict[str, torch.Tensor | int]:
        if len(snapshot.traffic) != len(self.traffic):
            raise ValueError("snapshot traffic batch size differs from the active world")
        for traffic, traffic_snapshot in zip(self.traffic, snapshot.traffic):
            traffic.restore(traffic_snapshot)
        self.states = snapshot.states.detach().clone().to(self.device)
        self.history = snapshot.history.detach().clone().to(self.device)
        self.history_valid = snapshot.history_valid.detach().clone().to(self.device)
        self.reference_index = int(snapshot.reference_index)
        self.filter_state = None if snapshot.filter_global is None else FilterState(
            snapshot.filter_global.detach().clone().to(self.device),
            snapshot.filter_agents.detach().clone().to(self.device),
        )
        self.slow_scene = self._clone(snapshot.slow_scene)
        self.slow_scene_noise = self._clone(snapshot.slow_scene_noise)
        self.agent_noise_state = self._clone(snapshot.agent_noise_state)
        self.agent_style_state = self._clone(snapshot.agent_style_state)
        self.previous_current = self._clone(snapshot.previous_current)
        self.committed_ego_controls = snapshot.committed_ego_controls.detach().clone().to(self.device)
        self.intervention_memory = self._clone(snapshot.intervention_memory)
        self.lateral_intervention_memory = self._clone(snapshot.lateral_intervention_memory)
        self.reaction_enabled = self._clone(snapshot.reaction_enabled)
        self.reaction_phase = self._clone(snapshot.reaction_phase)
        self.reaction_age_frames = self._clone(snapshot.reaction_age_frames)
        self.reaction_recovery_remaining = self._clone(snapshot.reaction_recovery_remaining)
        self.previous_background_actions = self._clone(snapshot.previous_background_actions)
        self.intervention_armed = self._clone(snapshot.intervention_armed)
        self.influence_state = snapshot.influence_state
        return self.observe()

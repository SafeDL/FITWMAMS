"""One physical execution loop with model-specific observation/action interfaces."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from ads.common.control import (
    ACTION_NAMES,
    ContinuousController,
    NuplanTrajectoryController,
    SemanticEgoController,
    TrajectoryController,
)
from ads.common.observations import (
    CausalScene,
    carl_constraints,
    carl_observation,
    plant_observation,
)
from ads.common.safety import initial_safety
from ads.common.world import WorldEvent
from ads.tasks import NATIVE_PLANNERS, POLICY_SUFFIX
from hierarchical_world_model.src.highway import yaw_rate_to_steering
from hierarchical_world_model.src.scene_geometry import SceneGeometry

LONGITUDINAL_RESPONSE_THRESHOLD_MPS2 = 0.2
MANEUVER_RESPONSE_THRESHOLD_RPS = 0.01


def build_ads_observation(
    states: np.ndarray,
    valid: np.ndarray,
    geometry: SceneGeometry,
    controller: SemanticEgoController,
    *,
    reference_index: int,
    horizon: int,
) -> np.ndarray:
    """Build the public causal 49-D ADS observation for one world."""
    states = np.asarray(states)
    valid = np.asarray(valid, bool)
    ego = states[0]
    ego_speed = float(np.linalg.norm(ego[2:4]))
    ego_heading = float(np.arctan2(ego[3], max(abs(ego[2]), 1.0e-6)))
    lane_id = geometry.nearest_lane(ego[:2])
    lane_y = geometry.center_y(lane_id)
    target_lane_delta = controller.target_y - lane_y
    head = np.asarray(
        (
            ego_speed / 45.0,
            float(ego[4]) / 8.0,
            ego_heading / np.pi,
            (float(ego[1]) - lane_y) / 4.0,
            (controller.target_speed_mps - ego_speed) / 15.0,
            target_lane_delta / 4.0,
            min(float(reference_index) / max(int(horizon), 1), 1.0),
        ),
        dtype=np.float32,
    )
    neighbors: list[float] = []
    for slot in range(1, 7):
        if valid[slot]:
            other = states[slot]
            other_lane = geometry.nearest_lane(other[:2])
            neighbors.extend(
                (
                    (other[0] - ego[0]) / 80.0,
                    (other[1] - ego[1]) / 8.0,
                    (other[2] - ego[2]) / 20.0,
                    (other[3] - ego[3]) / 10.0,
                    1.0,
                    float(other_lane == lane_id),
                    float(geometry.relative_lane(lane_id, other_lane)) / 3.0,
                )
            )
        else:
            neighbors.extend((0.0,) * 7)
    return np.clip(
        np.concatenate((head, np.asarray(neighbors, np.float32))), -20.0, 20.0
    ).astype(np.float32)


class ADSHighwayEnv(gym.Env[np.ndarray, int]):
    """Five-action ADS task with 5 Hz decisions and 25 Hz world execution."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        events: Sequence[WorldEvent],
        world_factory: Callable[[WorldEvent], Any],
        *,
        physics_steps_per_action: int = 5,
        max_decisions: int | None = None,
        random_events: bool = True,
        seed: int = 0,
        record_trace: bool = False,
    ) -> None:
        super().__init__()
        if not events:
            raise ValueError("events must not be empty")
        if int(physics_steps_per_action) < 1:
            raise ValueError("physics_steps_per_action must be positive")
        self.events = tuple(events)
        self.world_factory = world_factory
        self.physics_steps_per_action = int(physics_steps_per_action)
        self.max_decisions = max_decisions
        self.random_events = bool(random_events)
        self._rng = np.random.default_rng(int(seed))
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(
            low=-20.0, high=20.0, shape=(49,), dtype=np.float32
        )
        self.world: Any | None = None
        self.event: WorldEvent | None = None
        self.controller: SemanticEgoController | None = None
        self.decision_count = 0
        self._next_event = 0
        self._last_ego_action = np.zeros(2, np.float32)
        self._npc_response_mode_counts = np.zeros(3, np.int64)
        self._npc_action_delta_sum = np.zeros(2, np.float64)
        self._npc_action_delta_count = 0
        self.record_trace = bool(record_trace)
        self.physics_states: list[np.ndarray] = []
        self.physics_controls: list[np.ndarray] = []
        self.physics_diagnostics: list[dict[str, np.ndarray]] = []

    def _make_controller(self, event: WorldEvent):
        ego = np.asarray(event.initial_states[0], np.float64)
        return SemanticEgoController(
            event.geometry, ego[:2], float(np.linalg.norm(ego[2:4]))
        )

    def _action_label(self, action):
        return ACTION_NAMES[int(action)]

    def _select_event(self, options: dict[str, Any] | None) -> WorldEvent:
        if options and "event_index" in options:
            return self.events[int(options["event_index"]) % len(self.events)]
        if options and "event_id" in options:
            wanted = str(options["event_id"])
            for event in self.events:
                if event.event_id == wanted:
                    return event
            raise KeyError(f"unknown event id {wanted!r}")
        if self.random_events:
            return self.events[int(self._rng.integers(len(self.events)))]
        event = self.events[self._next_event % len(self.events)]
        self._next_event += 1
        return event

    def _states(self) -> np.ndarray:
        if self.world is None:
            raise RuntimeError("reset the environment before use")
        observation = self.world.observe()
        value = observation["agent_states"]
        return np.asarray(value.detach().cpu() if torch.is_tensor(value) else value)[0]

    def _observation(self) -> np.ndarray:
        assert self.event is not None and self.controller is not None
        return build_ads_observation(
            self._states(),
            self.event.valid,
            self.event.geometry,
            self.controller,
            reference_index=int(getattr(self.world, "reference_index", 0)),
            horizon=len(self.event.soft_reference),
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self.event = self._select_event(options)
        self.world = self.world_factory(self.event)
        history = self.event.initial_history
        history_valid = self.event.initial_history_valid
        if history is not None:
            frames = self.world.cfg.history_frames
            history = history[-frames:]
            history_valid = history_valid[-frames:]
        self.world.reset(
            torch.as_tensor(self.event.initial_states)[None],
            torch.as_tensor(self.event.valid)[None],
            torch.as_tensor(self.event.soft_reference)[None],
            torch.as_tensor(self.event.map_polylines)[None],
            torch.as_tensor(self.event.map_polyline_valid)[None],
            exogenous_state=self.event.exogenous_state,
            vehicle_dimensions=(
                None
                if self.event.vehicle_dimensions is None
                else torch.as_tensor(self.event.vehicle_dimensions)[None]
            ),
            initial_history=(
                None if history is None else torch.as_tensor(history)[None]
            ),
            initial_history_valid=(
                None if history_valid is None else torch.as_tensor(history_valid)[None]
            ),
        )
        self.controller = self._make_controller(self.event)
        self.decision_count = 0
        self._last_ego_action = np.zeros(2, np.float32)
        self._npc_response_mode_counts[:] = 0
        self._npc_action_delta_sum[:] = 0.0
        self._npc_action_delta_count = 0
        self.physics_states = [self._states().copy()] if self.record_trace else []
        self.physics_controls = []
        self.physics_diagnostics = []
        return self._observation(), {
            "event_id": self.event.event_id,
            "source_event_id": self.event.source_event_id,
            "split": self.event.split,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.world is None or self.event is None or self.controller is None:
            raise RuntimeError("reset the environment before stepping")
        structurally_valid = self.controller.update_target(action)
        start_x = float(self._states()[0, 0])
        collision = False
        any_collision = False
        npc_collision = False
        offroad = False
        executed = 0
        action_cost = 0.0
        reference_exhausted = False
        for _ in range(self.physics_steps_per_action):
            if int(self.world.reference_index) >= len(self.event.soft_reference):
                reference_exhausted = True
                break
            ego_action = self.controller.control(self._states()[0])
            action_cost += float(np.square(ego_action - self._last_ego_action).sum())
            self._last_ego_action = ego_action
            result = self.world.advance_response(torch.as_tensor(ego_action)[None])
            if self.record_trace:
                self.physics_states.append(self._states().copy())
                self.physics_controls.append(np.asarray(ego_action).copy())
                self.physics_diagnostics.append(
                    {
                        name: np.asarray(
                            value.detach().cpu() if torch.is_tensor(value) else value
                        )[0].copy()
                        for name, value in result.items()
                        if value is not None
                        and (
                            name.startswith("takeover_")
                            or name in ("crashed", "offroad", "collision_pairs")
                        )
                    }
                )
            if "background_actions" in result and "base_background_actions" in result:
                background = np.asarray(result["background_actions"].detach().cpu())[
                    0, 0
                ]
                base_background = np.asarray(
                    result["base_background_actions"].detach().cpu()
                )[0, 0]
                delta_background = background - base_background
                valid_npc = np.asarray(self.event.valid[1:], bool)
                mode = np.where(
                    np.abs(delta_background[:, 1]) > MANEUVER_RESPONSE_THRESHOLD_RPS,
                    2,
                    np.where(
                        np.abs(delta_background[:, 0])
                        > LONGITUDINAL_RESPONSE_THRESHOLD_MPS2,
                        1,
                        0,
                    ),
                )
                self._npc_response_mode_counts += np.bincount(
                    mode[valid_npc].astype(np.int64), minlength=3
                )
                self._npc_action_delta_sum += np.abs(delta_background[valid_npc]).sum(
                    axis=0
                )
                self._npc_action_delta_count += int(valid_npc.sum())
            step_any_collision = bool(np.asarray(result["collision"].detach().cpu())[0])
            crashed_value = result.get("crashed")
            if crashed_value is None:
                step_ego_collision = step_any_collision
                step_npc_collision = False
            else:
                crashed = np.asarray(crashed_value.detach().cpu(), bool)[0]
                step_ego_collision = bool(crashed[0])
                pairs = result.get("collision_pairs")
                step_npc_collision = (
                    bool(np.asarray(pairs.detach().cpu(), bool)[0, 1:, 1:].any())
                    if pairs is not None
                    else bool(crashed[1:].any() and not step_ego_collision)
                )
            collision = collision or step_ego_collision
            any_collision = any_collision or step_any_collision
            npc_collision = npc_collision or step_npc_collision
            offroad_value = np.asarray(result["offroad"].detach().cpu())[0]
            offroad = offroad or bool(offroad_value[0])
            executed += 1
            if collision or offroad:
                break
        self.decision_count += 1
        states = self._states()
        ego = states[0]
        progress = float(ego[0]) - start_x
        speed = float(np.linalg.norm(ego[2:4]))
        speed_score = 1.0 - min(
            abs(speed - self.controller.target_speed_mps)
            / max(self.controller.target_speed_mps, 5.0),
            1.0,
        )
        reward = 0.08 * progress + 0.1 * speed_score - 0.002 * action_cost
        if not structurally_valid:
            reward -= 0.02
        if collision:
            reward -= 10.0
        if offroad:
            reward -= 5.0
        terminated = bool(collision or offroad)
        horizon_limit = self.max_decisions or int(
            np.ceil(len(self.event.soft_reference) / self.physics_steps_per_action)
        )
        truncated = bool(
            not terminated
            and (
                reference_exhausted
                or int(self.world.reference_index) >= len(self.event.soft_reference)
                or self.decision_count >= horizon_limit
            )
        )
        info = {
            "event_id": self.event.event_id,
            "source_event_id": self.event.source_event_id,
            "executed_physics_steps": executed,
            "collision": collision,
            "any_vehicle_collision": any_collision,
            "npc_collision": npc_collision,
            "offroad": offroad,
            "progress_m": float(ego[0] - self.event.initial_states[0, 0]),
            "speed_mps": speed,
            "semantic_action": self._action_label(action),
            "structurally_valid_action": structurally_valid,
            "TimeLimit.truncated": truncated,
            "stalled": bool(truncated and speed < 2.0),
            "safety_failure": bool(collision or offroad),
            "npc_track_count": int(self._npc_response_mode_counts[0]),
            "npc_longitudinal_count": int(self._npc_response_mode_counts[1]),
            "npc_maneuver_count": int(self._npc_response_mode_counts[2]),
            "npc_mean_abs_delta_ax": (
                0.0
                if self._npc_action_delta_count == 0
                else self._npc_action_delta_sum[0] / self._npc_action_delta_count
            ),
            "npc_mean_abs_delta_yaw": (
                0.0
                if self._npc_action_delta_count == 0
                else self._npc_action_delta_sum[1] / self._npc_action_delta_count
            ),
        }
        return self._observation(), float(reward), terminated, truncated, info


class PolicyEnvironment(ADSHighwayEnv):
    def __init__(
        self,
        algorithm,
        instances,
        world_factory,
        *,
        purpose="repair",
        ledger=None,
        **kwargs,
    ):
        if algorithm not in POLICY_SUFFIX:
            raise ValueError("unknown ADS algorithm")
        self.algorithm = algorithm
        self.instances = tuple(instances)
        for instance in self.instances:
            instance.validate()
            if purpose != "final_test" and instance.purpose == "final_test":
                raise ValueError(
                    "final-test data cannot enter an ADS training environment"
                )
            if ledger is not None:
                ledger.register(instance, purpose)
        super().__init__(
            [i.event for i in self.instances],
            world_factory,
            record_trace=True,
            **kwargs,
        )
        if algorithm == "carl":
            self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
            self.observation_space = spaces.Dict(
                {
                    "bev_semantics": spaces.Box(0, 255, (9, 256, 256), dtype=np.uint8),
                    "measurements": spaces.Box(
                        -np.inf, np.inf, (10,), dtype=np.float32
                    ),
                    "value_measurements": spaces.Box(
                        -np.inf, np.inf, (4,), dtype=np.float32
                    ),
                }
            )
        elif algorithm == "plant2":
            self.action_space = spaces.Tuple(
                (
                    spaces.Box(-np.inf, np.inf, (20, 2), dtype=np.float32),
                    spaces.Box(-np.inf, np.inf, (8, 2), dtype=np.float32),
                )
            )
            self.observation_space = spaces.Dict(
                {
                    "objects": spaces.Sequence(
                        spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32), stack=True
                    ),
                    "object_slots": spaces.Sequence(spaces.Discrete(7), stack=True),
                    "route_original": spaces.Box(
                        -np.inf, np.inf, (20, 2), dtype=np.float32
                    ),
                    "speed_limit": spaces.Discrete(4),
                    "input_ego_speed": spaces.Box(0, np.inf, (), dtype=np.float32),
                    "BEV": spaces.Box(0, 1, (3, 256, 256), dtype=np.float32),
                }
            )
        elif algorithm in NATIVE_PLANNERS:
            from ads.nuplan.features import PlannerObservationSpace

            self.action_space = spaces.Box(-np.inf, np.inf, (80, 3), dtype=np.float32)
            self.observation_space = PlannerObservationSpace()
        self.previous_native_action = np.zeros(2, np.float32)

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        collision, offroad, npc = initial_safety(self.event)
        self.initial_failure = collision or offroad
        self.initial_info = {
            **info,
            "collision": collision,
            "offroad": offroad,
            "npc_collision": npc,
            "safety_failure": self.initial_failure,
            "stalled": False,
            "progress_m": 0.0,
            "executed_physics_steps": 0,
            "speed_mps": float(np.linalg.norm(self.event.initial_states[0, 2:4])),
            "initial_safety_failure": self.initial_failure,
        }
        return observation, self.initial_info.copy()

    def _make_controller(self, event):
        self.previous_native_action = np.zeros(2, np.float32)
        if self.algorithm == "carl":
            return ContinuousController(event)
        if self.algorithm == "plant2":
            return TrajectoryController(event)
        if self.algorithm in NATIVE_PLANNERS:
            return NuplanTrajectoryController(event)
        return super()._make_controller(event)

    def _action_label(self, action):
        return (
            super()._action_label(action)
            if self.algorithm in ("dqn", "ppo")
            else self.algorithm
        )

    def causal_scene(self):
        states = self._states().copy()
        heading = float(np.arctan2(states[0, 3], states[0, 2]))
        speed = max(float(np.linalg.norm(states[0, 2:4])), 0.5)
        yaw = float(
            (-states[0, 4] * np.sin(heading) + states[0, 5] * np.cos(heading)) / speed
        )
        previous_yaw = yaw
        previous_speed = speed
        if len(self.physics_states) > 1:
            last = self.physics_states[-2][0]
            previous_speed = max(float(np.linalg.norm(last[2:4])), 0.5)
            h = np.arctan2(last[3], last[2])
            previous_yaw = float(
                (-last[4] * np.sin(h) + last[5] * np.cos(h))
                / max(np.linalg.norm(last[2:4]), 0.5)
            )
        length = float(self.event.vehicle_dimensions[0, 0])
        steering = yaw_rate_to_steering(yaw, speed, length, 0.838)
        previous_steering = yaw_rate_to_steering(
            previous_yaw, previous_speed, length, 0.838
        )
        steering_rate = (steering - previous_steering) / 0.04
        if self.algorithm == "carl":
            steering = self.controller.steering
            steering_rate = self.controller.steering_rate
        elif self.algorithm in NATIVE_PLANNERS:
            wheelbase = length * 3.089 / 5.176
            steering = float(np.arctan(yaw * wheelbase / speed))
            previous_steering = float(
                np.arctan(previous_yaw * wheelbase / previous_speed)
            )
            steering_rate = (steering - previous_steering) / 0.04
        return CausalScene(
            states,
            self.event.valid.copy(),
            self.event.vehicle_dimensions.copy(),
            self.event.geometry,
            int(self.world.reference_index) * 0.04,
            len(self.event.soft_reference) * 0.04,
            self.previous_native_action.copy(),
            yaw,
            (yaw - previous_yaw) / 0.04,
            steering,
            steering_rate,
            progress_fraction=max(
                0, float(states[0, 0] - self.event.initial_states[0, 0])
            )
            / max(
                float(np.linalg.norm(self.event.initial_states[0, 2:4]))
                * len(self.event.soft_reference)
                * 0.04,
                1,
            ),
        )

    def _observation(self):
        if self.algorithm in ("dqn", "ppo"):
            return super()._observation()
        if self.algorithm == "carl":
            return carl_observation(self.causal_scene())
        if self.algorithm in NATIVE_PLANNERS:
            from ads.nuplan.features import build_features, sample_history

            scene = self.causal_scene()
            history, valid = sample_history(
                self.physics_states,
                len(self.physics_states) - 1,
                initial_history=self.event.initial_history,
                initial_history_valid=self.event.initial_history_valid,
            )
            valid &= self.event.valid[None]
            features, slots = build_features(
                scene, history, valid, algorithm=self.algorithm
            )
            return {
                "features": features,
                "scene": scene,
                "slots": slots,
                "history": history,
                "history_valid": valid,
            }
        return plant_observation(self.causal_scene())

    def step(self, action):
        if self.initial_failure:
            return self._observation(), -10.0, True, False, self.initial_info.copy()
        if self.algorithm == "plant2" or self.algorithm in NATIVE_PLANNERS:
            self.controller.current_ego = self._states()[0].copy()
        start_x = float(self._states()[0, 0])
        observation, reward, terminated, truncated, info = super().step(action)
        if self.algorithm == "carl":
            self.previous_native_action = np.asarray(action, np.float32).copy()
            observation = self._observation()
            initial_speed = max(
                float(np.linalg.norm(self.event.initial_states[0, 2:4])), 2
            )
            duration = len(self.event.soft_reference) * 0.04
            progress = max(float(self._states()[0, 0]) - start_x, 0) / (
                initial_speed * duration
            )
            lane_factor, speed_factor, comfort_factor, ttc_factor = carl_constraints(
                self.causal_scene()
            )
            reward = (
                100
                * (
                    0.4
                    * progress
                    * lane_factor
                    * speed_factor
                    * comfort_factor
                    * ttc_factor
                    + 0.6 * info["executed_physics_steps"] * 0.04 / duration
                )
                * (not terminated)
            )
        info["task_reward"] = float(reward)
        return observation, reward, terminated, truncated, info

"""Closed-loop highway-env rollouts for subset rolling plans."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HIGHWAY_ROOT = ROOT / "HighwayEnv"
HIGHWAY_PACKAGE = HIGHWAY_ROOT / "highway_env"
if not HIGHWAY_PACKAGE.is_dir():
    raise FileNotFoundError(
        f"Required local highway-env package not found: {HIGHWAY_PACKAGE}"
    )
if str(HIGHWAY_ROOT) not in sys.path:
    sys.path.insert(0, str(HIGHWAY_ROOT))

from tools.idm_ego import IDM_PARAMETER_KEYS
from tools.highd_cutin import cutin_risk_from_series
from tools.risk import (
    apply_closed_loop_risk,
    evt_model_from_config,
    longitudinal_series_from_arrays,
)

from .frozen_diffusion_sampler import FrozenDiffusionSampler

try:
    from highway_env.road.road import Road, RoadNetwork
    from highway_env.vehicle.behavior import IDMVehicle
    from highway_env.vehicle.kinematics import Vehicle
except ImportError as exc:
    py_version = (
        f"{sys.version_info.major}."
        f"{sys.version_info.minor}."
        f"{sys.version_info.micro}"
    )
    raise RuntimeError(
        "Failed to import the required local highway-env package. "
        f"Package path: {HIGHWAY_PACKAGE}. "
        "Install dependencies from HighwayEnv/pyproject.toml, including "
        f"gymnasium. Current Python: {py_version}."
    ) from exc


@dataclass
class RolloutResult:
    risk_score: float
    metrics: dict[str, float]
    num_generated_plans: int
    trace: list[dict[str, float]] = field(default_factory=list)
    actions: np.ndarray | None = None


class ScriptedLeadVehicle(Vehicle):
    """A highway-env vehicle whose acceleration and steering are scripted."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.commanded_acceleration = 0.0
        self.commanded_steering = 0.0
        self.forced_position: np.ndarray | None = None
        self.forced_heading: float | None = None
        self.forced_speed: float | None = None

    def set_control(self, acceleration: float, steering: float = 0.0) -> None:
        self.commanded_acceleration = float(acceleration)
        self.commanded_steering = float(steering)

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
        Vehicle.act(
            self,
            {
                "steering": self.commanded_steering,
                "acceleration": self.commanded_acceleration,
            },
        )

    def step(self, dt: float) -> None:
        if self.forced_position is None:
            Vehicle.step(self, dt)
            return
        self.position = self.forced_position
        self.heading = float(self.forced_heading)
        self.speed = float(self.forced_speed)
        self.forced_position = None
        self.forced_heading = None
        self.forced_speed = None
        self.on_state_update()


def _speed_and_yaw(state: np.ndarray) -> tuple[float, float]:
    # highD following contexts are stored in an ego-current road-aligned frame.
    # At very low speeds, tiny lateral velocity noise makes atan2(vy, vx)
    # produce a near-sideways vehicle heading, which corrupts rollouts.
    speed = float(np.hypot(float(state[2]), float(state[3])))
    return speed, 0.0


def _accel_bounds_for_speed(
    speed: float,
    dt: float,
    ax_min: float,
    ax_max: float,
    speed_min: float,
    speed_max: float,
) -> tuple[float, float]:
    lower = float(ax_min)
    upper = float(ax_max)
    if dt > 0.0:
        lower = max(lower, (speed_min - speed) / dt)
        upper = min(upper, (speed_max - speed) / dt)
    if lower > upper:
        return float(ax_min), float(ax_max)
    return lower, upper


def _bound_residual(value: float, lower: float, upper: float) -> float:
    return max(0.0, lower - value) ** 2 + max(0.0, value - upper) ** 2


class ClosedLoopFollowingRunner:
    """Roll a generated lead plan on a highway-env car-following road."""

    def __init__(
        self,
        sampler: FrozenDiffusionSampler,
        config: dict[str, Any],
    ) -> None:
        self.sampler = sampler
        self.config = config
        env_cfg = config.get("env", {})
        prior_cfg = sampler.prior.model.denoiser.cfg
        target_fps = float(
            sampler.prior.config.get("sampling", {}).get(
                "target_fps",
                25.0,
            )
        )
        self.dt = float(env_cfg.get("dt", 1.0 / max(target_fps, 1.0)))
        self.episode_steps = int(
            env_cfg.get("episode_steps", min(25, prior_cfg.horizon_steps))
        )
        self.lanes_count = int(env_cfg.get("lanes_count", 1))
        self.speed_limit = float(env_cfg.get("speed_limit", 40.0))
        self.idm_ego_config = dict(config.get("idm_ego", {}) or {})
        ego_target_speed = env_cfg.get("ego_target_speed", None)
        if ego_target_speed is None or str(ego_target_speed).lower() in {
            "context",
            "initial",
        }:
            self.ego_target_speed: float | None = None
        else:
            self.ego_target_speed = float(ego_target_speed)
        self.initial_gap_min = float(env_cfg.get("initial_gap_min", 0.1))
        self.allow_initial_nonpositive_gap = bool(
            env_cfg.get("allow_initial_nonpositive_gap", False)
        )
        self.initial_nonpositive_gap_min_lateral_separation = float(
            env_cfg.get("initial_nonpositive_gap_min_lateral_separation", 1.0)
        )
        self.dynamics_model = str(
            config.get("dynamics", {}).get("model", "longitudinal")
        ).lower()
        if self.dynamics_model not in {
            "longitudinal",
            "kinematic_bicycle",
            "point_mass",
        }:
            raise ValueError(f"Unknown dynamics.model: {self.dynamics_model}")

    def _make_road(self) -> Any:
        return Road(
            network=RoadNetwork.straight_road_network(
                self.lanes_count,
                speed_limit=self.speed_limit,
            ),
            np_random=np.random.RandomState(
                int(self.config.get("training", {}).get("seed", 42))
            ),
            record_history=False,
        )

    def _closed_loop_risk(
        self,
        metrics: dict[str, float],
        trace: list[dict[str, float]],
    ) -> float:
        return apply_closed_loop_risk(
            metrics,
            trace,
            self.config,
            scoring_section="closed_loop_risk_scoring",
        )

    def rollout(
        self,
        initial_context: dict[str, Any],
        *,
        fixed_plan: np.ndarray,
        episode_steps: int | None = None,
    ) -> RolloutResult:
        initial_states = np.asarray(initial_context["initial_states"], dtype=np.float32)
        if initial_states.shape != (2, 6):
            raise ValueError(
                "initial_context['initial_states'] must have shape [2, 6], "
                f"got {tuple(initial_states.shape)}"
            )
        ego_length = float(initial_context.get("ego_length", 4.8))
        lead_length = float(
            initial_context.get(
                "adv_length",
                initial_context.get("lead_length", 4.8),
            )
        )
        ego0 = initial_states[0]
        lead0 = initial_states[1]
        initial_gap = float(lead0[0] - ego0[0] - 0.5 * (ego_length + lead_length))
        initial_lateral_offset = float(abs(lead0[1] - ego0[1]))
        allow_cutin_start_gap = (
            self.allow_initial_nonpositive_gap
            and initial_lateral_offset
            >= self.initial_nonpositive_gap_min_lateral_separation
        )
        if initial_gap <= self.initial_gap_min and not allow_cutin_start_gap:
            raise RuntimeError(
                "Invalid initial context: gap "
                f"{initial_gap:.3f} <= {self.initial_gap_min:.3f}, "
                f"lateral_offset={initial_lateral_offset:.3f}"
            )
        road = self._make_road()
        ego_speed, ego_yaw = _speed_and_yaw(ego0)
        lead_speed, lead_yaw = _speed_and_yaw(lead0)
        ego = IDMVehicle(
            road,
            position=np.asarray([ego0[0], ego0[1]], dtype=np.float64),
            heading=ego_yaw,
            speed=ego_speed,
            target_speed=(
                ego_speed if self.ego_target_speed is None else self.ego_target_speed
            ),
            enable_lane_change=bool(
                self.config.get("ego_response", {}).get(
                    "enable_lane_change",
                    False,
                )
            ),
        )
        for key in IDM_PARAMETER_KEYS:
            if key in self.idm_ego_config:
                setattr(ego, key, float(self.idm_ego_config[key]))
        lead = ScriptedLeadVehicle(
            road,
            position=np.asarray([lead0[0], lead0[1]], dtype=np.float64),
            heading=lead_yaw,
            speed=lead_speed,
        )
        ego.LENGTH = ego_length
        lead.LENGTH = lead_length
        if hasattr(ego, "diagonal"):
            ego.diagonal = float(np.sqrt(ego.LENGTH**2 + ego.WIDTH**2))
        if hasattr(lead, "diagonal"):
            lead.diagonal = float(np.sqrt(lead.LENGTH**2 + lead.WIDTH**2))
        setattr(road, "vehicles", [ego, lead])
        if hasattr(ego, "front_vehicle"):
            ego.front_vehicle = lead

        num_generated_plans = 1
        plan = np.asarray(fixed_plan, dtype=np.float32)
        if plan.ndim != 2 or int(plan.shape[0]) <= 0:
            raise ValueError(
                "fixed_plan must have shape [steps, action_dim], "
                f"got {tuple(plan.shape)}"
            )
        plan_cursor = 0
        lead_accel = float(lead0[4])
        prev_lead_accel = lead_accel
        lead_lateral_accel = float(lead0[5])
        prev_lead_lateral_accel = lead_lateral_accel
        min_ttc = 1000.0
        min_gap = float("inf")
        min_ego_accel = 0.0
        lead_physics_penalty = 0.0
        action_clip_count = 0
        jerk_violation_count = 0
        lateral_jerk_violation_count = 0
        speed_negative_count = 0
        speed_violation_count = 0
        lead_accel_values: list[float] = []
        lead_jerk_values: list[float] = []
        lead_lateral_accel_values: list[float] = []
        lead_lateral_jerk_values: list[float] = []
        lead_speed_values: list[float] = []
        trace: list[dict[str, float]] = []
        action_cfg = self.config.get("physics", {})
        dyn_cfg = self.config.get("dynamics", {})
        ax_min = float(action_cfg.get("ax_min", -8.0))
        ax_max = float(action_cfg.get("ax_max", 4.0))
        jerk_abs_max = float(action_cfg.get("jerk_abs_max", 12.0))
        ay_abs_max = float(action_cfg.get("ay_abs_max", dyn_cfg.get("ay_abs_max", 4.0)))
        lateral_jerk_abs_max = float(
            action_cfg.get(
                "lateral_jerk_abs_max",
                dyn_cfg.get("lateral_jerk_abs_max", 8.0),
            )
        )
        lateral_speed_abs_max = float(dyn_cfg.get("lateral_speed_abs_max", 5.0))
        wheelbase = max(float(dyn_cfg.get("wheelbase", 5.0)), 1e-6)
        speed_min = float(dyn_cfg.get("speed_min", 0.0))
        speed_max = float(dyn_cfg.get("speed_max", self.speed_limit))
        initial_accel_lower, initial_accel_upper = _accel_bounds_for_speed(
            max(float(lead.speed), 0.0),
            self.dt,
            ax_min,
            ax_max,
            speed_min,
            speed_max,
        )
        lead_accel = float(
            np.clip(lead_accel, initial_accel_lower, initial_accel_upper)
        )
        prev_lead_accel = lead_accel
        lead_steering = 0.0
        schema_rep = self.sampler.prior.schema.get("action_representation")
        config_rep = self.sampler.prior.config.get("action", {}).get(
            "representation",
            "jerk",
        )
        rep = str(schema_rep or config_rep).lower()
        uses_lateral_accel = rep in {"ax_ay", "acceleration"}

        total_steps = (
            self.episode_steps if episode_steps is None else int(episode_steps)
        )
        if total_steps <= 0:
            raise ValueError(f"episode_steps must be positive, got {total_steps}")
        executed_actions: list[np.ndarray] = []

        for step in range(total_steps):
            if plan_cursor >= len(plan):
                break

            cursor = plan_cursor
            raw_action_row = np.asarray(plan[cursor], dtype=np.float32)
            action_row = raw_action_row.copy()
            plan_cursor += 1
            speed_before = max(float(lead.speed), 0.0)
            if rep == "jerk":
                raw_jerk = float(raw_action_row[0])
                jerk = float(
                    np.clip(
                        raw_jerk,
                        -jerk_abs_max,
                        jerk_abs_max,
                    )
                )
                proposed_accel = prev_lead_accel + jerk * self.dt
                accel_lower, accel_upper = _accel_bounds_for_speed(
                    speed_before,
                    self.dt,
                    ax_min,
                    ax_max,
                    speed_min,
                    speed_max,
                )
                jerk_lower = prev_lead_accel - jerk_abs_max * self.dt
                jerk_upper = prev_lead_accel + jerk_abs_max * self.dt
                accel_lower = max(accel_lower, jerk_lower)
                accel_upper = min(accel_upper, jerk_upper)
                if accel_lower > accel_upper:
                    accel_lower = max(ax_min, jerk_lower)
                    accel_upper = min(ax_max, jerk_upper)
                lead_accel = float(np.clip(proposed_accel, accel_lower, accel_upper))
                jerk = (lead_accel - prev_lead_accel) / max(self.dt, 1e-6)
                action_row[0] = jerk
            else:
                raw_accel = float(raw_action_row[0])
                accel_lower, accel_upper = _accel_bounds_for_speed(
                    speed_before,
                    self.dt,
                    ax_min,
                    ax_max,
                    speed_min,
                    speed_max,
                )
                accel_lower = max(
                    accel_lower,
                    prev_lead_accel - jerk_abs_max * self.dt,
                )
                accel_upper = min(
                    accel_upper,
                    prev_lead_accel + jerk_abs_max * self.dt,
                )
                if accel_lower > accel_upper:
                    accel_lower = max(
                        ax_min,
                        prev_lead_accel - jerk_abs_max * self.dt,
                    )
                    accel_upper = min(
                        ax_max,
                        prev_lead_accel + jerk_abs_max * self.dt,
                    )
                lead_accel = float(np.clip(raw_accel, accel_lower, accel_upper))
                jerk = (lead_accel - prev_lead_accel) / max(self.dt, 1e-6)
                action_row[0] = lead_accel
            lateral_jerk = 0.0
            if uses_lateral_accel:
                raw_lateral_accel = (
                    float(raw_action_row[1]) if raw_action_row.size > 1 else 0.0
                )
                lateral_lower = max(
                    -ay_abs_max,
                    prev_lead_lateral_accel - lateral_jerk_abs_max * self.dt,
                )
                lateral_upper = min(
                    ay_abs_max,
                    prev_lead_lateral_accel + lateral_jerk_abs_max * self.dt,
                )
                if lateral_lower > lateral_upper:
                    lateral_lower = -ay_abs_max
                    lateral_upper = ay_abs_max
                lead_lateral_accel = float(
                    np.clip(raw_lateral_accel, lateral_lower, lateral_upper)
                )
                lateral_jerk = (
                    lead_lateral_accel - prev_lead_lateral_accel
                ) / max(self.dt, 1e-6)
                if action_row.size > 1:
                    action_row[1] = lead_lateral_accel
                lead_steering = 0.0
            else:
                lead_steering = 0.0
                lead_lateral_accel = 0.0
            lead_physics_penalty += _bound_residual(
                lead_accel,
                ax_min,
                ax_max,
            )
            lead_physics_penalty += _bound_residual(
                jerk,
                -jerk_abs_max,
                jerk_abs_max,
            )
            lead_physics_penalty += _bound_residual(
                lead_lateral_accel,
                -ay_abs_max,
                ay_abs_max,
            )
            lead_physics_penalty += _bound_residual(
                lateral_jerk,
                -lateral_jerk_abs_max,
                lateral_jerk_abs_max,
            )
            action_clip_count += int(np.max(np.abs(raw_action_row - action_row)) > 1e-6)
            jerk_violation_count += int(abs(jerk) > jerk_abs_max + 1e-6)
            lateral_jerk_violation_count += int(
                abs(lateral_jerk) > lateral_jerk_abs_max + 1e-6
            )
            executed_actions.append(action_row.copy())
            prev_lead_accel = lead_accel
            prev_lead_lateral_accel = lead_lateral_accel
            lead_accel_values.append(float(lead_accel))
            lead_jerk_values.append(float(jerk))
            lead_lateral_accel_values.append(float(lead_lateral_accel))
            lead_lateral_jerk_values.append(float(lateral_jerk))
            lead.set_control(lead_accel, lead_steering)
            if uses_lateral_accel or self.dynamics_model == "point_mass":
                lead_vx_before = float(lead.speed) * float(np.cos(lead.heading))
                lead_vy_before = float(lead.speed) * float(np.sin(lead.heading))
                lead_position_next = np.asarray(
                    [
                        float(lead.position[0])
                        + lead_vx_before * self.dt
                        + 0.5 * lead_accel * self.dt * self.dt,
                        float(lead.position[1])
                        + lead_vy_before * self.dt
                        + 0.5 * lead_lateral_accel * self.dt * self.dt,
                    ],
                    dtype=np.float64,
                )
                lead_vx_next = float(
                    np.clip(
                        lead_vx_before + lead_accel * self.dt,
                        speed_min,
                        speed_max,
                    )
                )
                lead_vy_next = float(
                    np.clip(
                        lead_vy_before + lead_lateral_accel * self.dt,
                        -lateral_speed_abs_max,
                        lateral_speed_abs_max,
                    )
                )
                lead_speed_next = float(np.hypot(lead_vx_next, lead_vy_next))
                if lead_speed_next > speed_max and lead_speed_next > 1e-6:
                    scale = speed_max / lead_speed_next
                    lead_vx_next *= scale
                    lead_vy_next *= scale
                    lead_speed_next = speed_max
                lead_heading_next = float(
                    np.arctan2(lead_vy_next, max(lead_vx_next, 1e-6))
                )
                lead.set_forced_state(
                    lead_position_next,
                    lead_heading_next,
                    lead_speed_next,
                )
            elif self.dynamics_model == "kinematic_bicycle":
                lead_speed_before = max(float(lead.speed), 0.0)
                lead_position_next = np.asarray(
                    [
                        float(lead.position[0])
                        + lead_speed_before * float(np.cos(lead.heading)) * self.dt,
                        float(lead.position[1])
                        + lead_speed_before * float(np.sin(lead.heading)) * self.dt,
                    ],
                    dtype=np.float64,
                )
                lead_heading_next = float(
                    lead.heading
                    + lead_speed_before
                    / wheelbase
                    * float(np.tan(lead_steering))
                    * self.dt
                )
                lead_speed_next = float(
                    np.clip(
                        lead_speed_before + lead_accel * self.dt,
                        speed_min,
                        speed_max,
                    )
                )
                lead.set_forced_state(
                    lead_position_next,
                    lead_heading_next,
                    lead_speed_next,
                )

            road.act()
            road.step(self.dt)
            speed_negative_count += int(
                float(lead.speed) < float(action_cfg.get("speed_min", 0.0))
            )
            speed_violation_count += int(
                float(lead.speed) < speed_min - 1e-6
                or float(lead.speed) > speed_max + 1e-6
            )
            gap = float(
                lead.position[0] - ego.position[0] - 0.5 * (ego_length + lead_length)
            )
            closing = float(ego.speed - lead.speed)
            ttc = gap / max(closing, 1e-6) if closing > 1e-6 else 1000.0
            ego_accel = float(ego.action.get("acceleration", 0.0))
            min_gap = min(min_gap, gap)
            min_ttc = min(min_ttc, ttc)
            min_ego_accel = min(min_ego_accel, ego_accel)
            lead_speed_values.append(float(lead.speed))
            trace.append(
                {
                    "step": float(step),
                    "gap": gap,
                    "closing_speed": closing,
                    "ttc": float(ttc),
                    "collision": float(ego.crashed),
                    "ego_accel": ego_accel,
                    "ego_speed": float(ego.speed),
                    "ego_position": float(ego.position[0]),
                    "ego_y": float(ego.position[1]),
                    "ego_yaw": float(ego.heading),
                    "ego_action_accel": ego_accel,
                    "ego_action_steering": float(ego.action.get("steering", 0.0)),
                    "lead_speed": float(lead.speed),
                    "lead_position": float(lead.position[0]),
                    "lead_y": float(lead.position[1]),
                    "lead_yaw": float(lead.heading),
                    "lead_accel": float(lead_accel),
                    "lead_jerk": float(jerk),
                    "lead_lateral_accel": float(lead_lateral_accel),
                    "lead_lateral_jerk": float(lateral_jerk),
                    "lead_steering": float(lead_steering),
                }
            )
            if ego.crashed:
                break

        collision = bool(ego.crashed)
        risk_cfg = self.config.get("closed_loop_risk", {})
        near_gap = float(risk_cfg.get("near_collision_gap", 2.0))
        hard_brake_threshold = float(risk_cfg.get("hard_brake_threshold", -4.0))
        physics_penalty_mean = float(lead_physics_penalty / max(len(trace), 1))
        physical_feasible = bool(
            physics_penalty_mean <= 1e-8
            and jerk_violation_count == 0
            and lateral_jerk_violation_count == 0
            and speed_negative_count == 0
            and speed_violation_count == 0
        )
        metrics = {
            "collision": float(collision),
            "invalid_initial_context": 0.0,
            "initial_gap": float(initial_gap),
            "initial_lateral_offset": float(initial_lateral_offset),
            "ego_target_speed": float(ego.target_speed),
            "min_ttc": float(min_ttc),
            "min_gap": float(min_gap),
            "final_gap": float(trace[-1]["gap"]) if trace else float(min_gap),
            "min_ego_accel": float(min_ego_accel),
            "near_collision": float(min_gap < near_gap),
            "hard_brake": float(min_ego_accel <= hard_brake_threshold),
            "lead_physics_penalty": physics_penalty_mean,
            "physical_feasible": float(physical_feasible),
            "lead_accel_mean": (
                float(np.mean(lead_accel_values)) if lead_accel_values else 0.0
            ),
            "lead_accel_std": (
                float(np.std(lead_accel_values)) if lead_accel_values else 0.0
            ),
            "lead_accel_min": (
                float(np.min(lead_accel_values)) if lead_accel_values else 0.0
            ),
            "lead_accel_max": (
                float(np.max(lead_accel_values)) if lead_accel_values else 0.0
            ),
            "lead_jerk_mean": (
                float(np.mean(lead_jerk_values)) if lead_jerk_values else 0.0
            ),
            "lead_jerk_std": (
                float(np.std(lead_jerk_values)) if lead_jerk_values else 0.0
            ),
            "lead_jerk_min": (
                float(np.min(lead_jerk_values)) if lead_jerk_values else 0.0
            ),
            "lead_jerk_max": (
                float(np.max(lead_jerk_values)) if lead_jerk_values else 0.0
            ),
            "lead_jerk_abs_mean": (
                float(np.mean(np.abs(lead_jerk_values))) if lead_jerk_values else 0.0
            ),
            "lead_jerk_abs_max": (
                float(np.max(np.abs(lead_jerk_values))) if lead_jerk_values else 0.0
            ),
            "lead_lateral_accel_mean": (
                float(np.mean(lead_lateral_accel_values))
                if lead_lateral_accel_values
                else 0.0
            ),
            "lead_lateral_accel_std": (
                float(np.std(lead_lateral_accel_values))
                if lead_lateral_accel_values
                else 0.0
            ),
            "lead_lateral_accel_min": (
                float(np.min(lead_lateral_accel_values))
                if lead_lateral_accel_values
                else 0.0
            ),
            "lead_lateral_accel_max": (
                float(np.max(lead_lateral_accel_values))
                if lead_lateral_accel_values
                else 0.0
            ),
            "lead_lateral_jerk_abs_max": (
                float(np.max(np.abs(lead_lateral_jerk_values)))
                if lead_lateral_jerk_values
                else 0.0
            ),
            "lead_speed_mean": (
                float(np.mean(lead_speed_values)) if lead_speed_values else 0.0
            ),
            "lead_speed_std": (
                float(np.std(lead_speed_values)) if lead_speed_values else 0.0
            ),
            "lead_speed_min": (
                float(np.min(lead_speed_values)) if lead_speed_values else 0.0
            ),
            "lead_speed_max": (
                float(np.max(lead_speed_values)) if lead_speed_values else 0.0
            ),
            "action_clip_rate": float(action_clip_count / max(len(trace), 1)),
            "jerk_violation_rate": float(jerk_violation_count / max(len(trace), 1)),
            "lateral_jerk_violation_rate": float(
                lateral_jerk_violation_count / max(len(trace), 1)
            ),
            "speed_negative_rate": float(speed_negative_count / max(len(trace), 1)),
            "speed_violation_rate": float(speed_violation_count / max(len(trace), 1)),
            "num_generated_plans": float(num_generated_plans),
            "steps": float(len(trace)),
        }
        risk_start_index = initial_context.get("risk_start_index", np.nan)
        try:
            risk_start_index = float(risk_start_index)
        except (TypeError, ValueError):
            risk_start_index = float("nan")
        if np.isfinite(risk_start_index):
            metrics["risk_start_index"] = float(risk_start_index)
        return RolloutResult(
            risk_score=self._closed_loop_risk(metrics, trace),
            metrics=metrics,
            num_generated_plans=num_generated_plans,
            trace=trace,
            actions=(
                np.asarray(executed_actions, dtype=np.float32)
                if executed_actions
                else np.zeros((0, 1), dtype=np.float32)
            ),
        )

    def rollout_pre_sampled_plan(
        self,
        initial_context: dict[str, Any],
        plan: np.ndarray,
        *,
        episode_steps: int | None = None,
    ) -> RolloutResult:
        return self.rollout(
            initial_context,
            fixed_plan=plan,
            episode_steps=episode_steps,
        )


class ClosedLoopCutInRunner(ClosedLoopFollowingRunner):
    """Roll a generated cut-in target plan and score event-level cut-in risk."""

    def __init__(
        self,
        sampler: FrozenDiffusionSampler,
        config: dict[str, Any],
    ) -> None:
        config.setdefault("env", {})
        config["env"]["lanes_count"] = max(int(config["env"].get("lanes_count", 2)), 2)
        super().__init__(sampler, config)

    def _closed_loop_risk(
        self,
        metrics: dict[str, float],
        trace: list[dict[str, float]],
    ) -> float:
        cfg = self.config.get("closed_loop_risk", {})

        if trace:
            gap = np.asarray([row["gap"] for row in trace], dtype=np.float64)
            ego_speed = np.asarray(
                [row["ego_speed"] for row in trace],
                dtype=np.float64,
            )
            lead_speed = np.asarray(
                [row["lead_speed"] for row in trace],
                dtype=np.float64,
            )
            ego_accel = np.asarray(
                [
                    row.get("ego_accel", row.get("ego_action_accel", 0.0))
                    for row in trace
                ],
                dtype=np.float64,
            )
            series = longitudinal_series_from_arrays(
                gap=gap,
                ego_speed=ego_speed,
                lead_speed=lead_speed,
                ego_accel=ego_accel,
            )
            lateral = np.asarray(
                [row["lead_y"] - row["ego_y"] for row in trace],
                dtype=np.float64,
            )
        else:
            series = {}
            lateral = np.asarray([], dtype=np.float64)
        risk = cutin_risk_from_series(
            series=series,
            lateral_offset=lateral,
            config=self.config,
            scoring_section="closed_loop_risk_scoring",
            dt=self.dt,
            min_ego_accel=float(metrics["min_ego_accel"]),
            risk_start_index=(
                int(metrics["risk_start_index"])
                if np.isfinite(float(metrics.get("risk_start_index", np.nan)))
                else None
            ),
        )

        y_cutin = float(risk["y_cutin"])
        risk_score = y_cutin
        evt_tail_probability = float("nan")
        evt_return_level_target = float("nan")
        evt_failure_threshold = float("nan")
        evt_model, evt_path = evt_model_from_config(self.config)
        if evt_model is not None:
            risk_score = float(evt_model.score(y_cutin))
            evt_tail_probability = float(evt_model.survival(y_cutin))
            evt_cfg = dict(self.config.get("evt", {}))
            if str(evt_cfg.get("target_mode", "return_period")) == "collision_critical_level":
                if "collision_critical_level" not in evt_cfg:
                    raise KeyError(
                        "evt.collision_critical_level must be resolved before "
                        "closed-loop cut-in evaluation"
                    )
                evt_return_level_target = float(
                    evt_cfg["collision_critical_level"]
                )
            else:
                return_period = int(evt_cfg.get("return_period", 100))
                evt_return_level_target = float(evt_model.return_level(return_period))
            evt_failure_threshold = float(evt_model.score(evt_return_level_target))

        physics_penalty_score = -float(cfg.get("lead_physics_weight", 0.1)) * float(
            metrics["lead_physics_penalty"]
        )
        metrics.update(
            {
                "risk_score": float(risk_score),
                "y_cutin": float(y_cutin),
                "y_long": float(risk["y_long"]),
                "proxy_risk_score": float(y_cutin),
                "post_longitudinal_risk_score": float(
                    risk["post_longitudinal_risk_score"]
                ),
                "cutin_safety_risk_score": float(
                    risk["cutin_safety_risk_score"]
                ),
                "collision_risk_score": float(risk["collision_risk_score"]),
                "near_collision_risk_score": float(
                    risk["near_collision_risk_score"]
                ),
                "ttc_objective": risk["ttc_objective"],
                "thw_objective": risk["thw_objective"],
                "drac_objective": risk["drac_objective"],
                "gap_objective": risk["gap_objective"],
                "ltg_objective": risk["ltg_objective"],
                "ttc_risk_score": risk["ttc_risk_score"],
                "thw_risk_score": risk["thw_risk_score"],
                "drac_risk_score": risk["drac_risk_score"],
                "gap_risk_score": risk["gap_risk_score"],
                "ltg_risk_score": risk["ltg_risk_score"],
                "hard_brake_severity": float(risk["hard_brake_severity"]),
                "hard_brake_risk_score": float(risk["hard_brake_risk_score"]),
                "cutin_gap": float(risk["cutin_gap"]),
                "cutin_ttc": float(risk["cutin_ttc"]),
                "cutin_time_headway": float(risk["cutin_time_headway"]),
                "cutin_lateral_time_gap": float(
                    risk["cutin_lateral_time_gap"]
                ),
                "min_post_cutin_gap": float(risk["min_post_cutin_gap"]),
                "min_post_cutin_ttc": float(risk["min_post_cutin_ttc"]),
                "max_post_cutin_drac": float(risk["max_post_cutin_drac"]),
                "min_abs_lateral_offset": float(risk["min_abs_lateral_offset"]),
                "final_abs_lateral_offset": float(
                    risk["final_abs_lateral_offset"]
                ),
                "max_abs_lateral_velocity": float(
                    risk["max_abs_lateral_velocity"]
                ),
                "max_lateral_approach_speed": float(
                    risk["max_lateral_approach_speed"]
                ),
                "is_cutin": float(risk["is_cutin"]),
                "is_front_cutin": float(risk["is_front_cutin"]),
                "risk_start_index": float(risk["risk_start_index"]),
                "lateral_overlap_fraction": float(
                    risk["lateral_overlap_fraction"]
                ),
                "evt_tail_probability": evt_tail_probability,
                "evt_return_level_target": evt_return_level_target,
                "evt_failure_threshold": evt_failure_threshold,
                "evt_model_path": evt_path or "",
                "physics_penalty_score": float(physics_penalty_score),
                "validity_penalized_score": float(risk_score + physics_penalty_score),
            }
        )
        return float(risk_score)

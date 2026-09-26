"""Closed-loop ADS interventions with complete longitudinal/lateral semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


def _wrap_angle(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


@dataclass(frozen=True)
class OnlineAccelerationWindowPolicy:
    """Apply one ADS acceleration setpoint without a prerecorded action trace."""

    base_policy: Callable[[dict[str, torch.Tensor | int]], torch.Tensor]
    acceleration_mps2: float
    start_frame: int = 25
    stop_frame: int = 50

    def __post_init__(self) -> None:
        if not -8.0 <= self.acceleration_mps2 <= 4.0:
            raise ValueError("ADS acceleration must be in [-8, 4] m/s²")
        if not 0 <= self.start_frame < self.stop_frame:
            raise ValueError("invalid intervention window")

    def __call__(self, context: dict[str, torch.Tensor | int]) -> torch.Tensor:
        action = self.base_policy(context).clone()
        if self.start_frame <= int(context["reference_index"]) < self.stop_frame:
            action[:, 0] = self.acceleration_mps2
        return action


class OnlineSemanticLaneChangePolicy:
    """Feedback lane change from realized ADS state, with no nominal branch."""

    def __init__(
        self,
        base_policy: Callable[[dict[str, torch.Tensor | int]], torch.Tensor],
        *,
        direction: str,
        start_frame: int = 25,
        lane_width_m: float = 3.6,
        lateral_gain_per_s: float = 0.9,
        heading_gain_per_s: float = 6.0,
        maximum_lateral_speed_mps: float = 2.0,
        maximum_yaw_rate_rps: float = 0.35,
    ) -> None:
        if direction not in {"left", "right"}:
            raise ValueError("direction must be left or right")
        if min(lane_width_m, lateral_gain_per_s, heading_gain_per_s,
               maximum_lateral_speed_mps, maximum_yaw_rate_rps) <= 0:
            raise ValueError("lane-change parameters must be positive")
        self.base_policy = base_policy
        self.direction = direction
        self.start_frame = int(start_frame)
        self.lane_width_m = float(lane_width_m)
        self.lateral_gain_per_s = float(lateral_gain_per_s)
        self.heading_gain_per_s = float(heading_gain_per_s)
        self.maximum_lateral_speed_mps = float(maximum_lateral_speed_mps)
        self.maximum_yaw_rate_rps = float(maximum_yaw_rate_rps)
        self.target_y: torch.Tensor | None = None

    def __call__(self, context: dict[str, torch.Tensor | int]) -> torch.Tensor:
        action = self.base_policy(context).clone()
        current = context["agent_states"]
        if not isinstance(current, torch.Tensor):
            raise TypeError("agent_states must be a torch tensor")
        index = int(context["reference_index"])
        if index == 0:
            self.target_y = None
        if index < self.start_frame:
            return action
        ego = current[:, 0]
        if self.target_y is None:
            sign = 1.0 if self.direction == "left" else -1.0
            lanes = context.get("map_polylines")
            lane_valid = context.get("map_polyline_valid")
            if isinstance(lanes, torch.Tensor) and isinstance(lane_valid, torch.Tensor):
                lane_y = lanes[..., 0, 1]
                lane_present = lane_valid.any(dim=-1)
                distance = torch.where(
                    lane_present, (lane_y - ego[:, None, 1]).abs(),
                    torch.full_like(lane_y, float("inf")),
                )
                if not bool(lane_present.any(dim=1).all()):
                    raise ValueError("ADS world has no mapped source lane")
                source_y = torch.gather(lane_y, 1, distance.argmin(dim=1, keepdim=True)).squeeze(1)
                displacement = (lane_y - source_y[:, None]) * sign
                adjacent = lane_present & displacement.gt(1.8) & displacement.lt(5.4)
                available = adjacent.any(dim=1)
                if not bool(available.all()):
                    self.target_y = None
                    raise ValueError("requested ADS lane change has no mapped target lane")
                target_index = torch.where(
                    adjacent, displacement,
                    torch.full_like(displacement, float("inf")),
                ).argmin(dim=1, keepdim=True)
                self.target_y = torch.gather(lane_y, 1, target_index).squeeze(1).detach().clone()
            else:
                self.target_y = ego[:, 1].detach().clone() + sign * self.lane_width_m
        error_y = self.target_y.to(ego) - ego[:, 1]
        speed = torch.linalg.vector_norm(ego[:, 2:4], dim=-1).clamp_min(0.5)
        heading = torch.atan2(ego[:, 3], ego[:, 2])
        desired_vy = self.maximum_lateral_speed_mps * torch.tanh(
            self.lateral_gain_per_s * error_y / self.maximum_lateral_speed_mps
        )
        desired_heading = torch.asin((desired_vy / speed).clamp(-0.95, 0.95))
        yaw_rate = self.heading_gain_per_s * _wrap_angle(desired_heading - heading)
        action[:, 1] = yaw_rate.clamp(
            -self.maximum_yaw_rate_rps, self.maximum_yaw_rate_rps
        )
        return action


@dataclass(frozen=True)
class AbsoluteAccelerationPolicy:
    """Override ADS acceleration with an absolute physical setpoint in a window."""

    baseline_actions: np.ndarray
    acceleration_mps2: float
    start_frame: int = 25
    stop_frame: int = 50
    minimum_acceleration_mps2: float = -8.0
    maximum_acceleration_mps2: float = 4.0

    def __post_init__(self) -> None:
        values = np.asarray(self.baseline_actions, np.float32)
        if values.ndim != 3 or values.shape[-1] != 2:
            raise ValueError("baseline_actions must have shape [batch,time,2]")
        if not 0 <= self.start_frame < self.stop_frame <= values.shape[1]:
            raise ValueError("intervention window must lie inside baseline_actions")
        if not self.minimum_acceleration_mps2 <= self.acceleration_mps2 <= self.maximum_acceleration_mps2:
            raise ValueError("absolute acceleration setpoint is outside physical bounds")
        object.__setattr__(self, "baseline_actions", values.copy())

    def __call__(self, context: dict[str, torch.Tensor | int]) -> torch.Tensor:
        index = int(context["reference_index"])
        current = context["agent_states"]
        if not isinstance(current, torch.Tensor):
            raise TypeError("agent_states must be a torch tensor")
        action = torch.as_tensor(
            self.baseline_actions[:, index], device=current.device, dtype=current.dtype
        ).clone()
        if self.start_frame <= index < self.stop_frame:
            action[:, 0] = float(self.acceleration_mps2)
        return action


class SemanticLaneChangePolicy:
    """Complete one lane change and settle at the target-lane centre.

    A raw yaw-rate pulse has no terminal lateral state and can keep rotating the
    ADS out of the road.  This policy instead defines a target lane centre and
    applies closed-loop heading feedback until both lateral and heading errors
    settle.  Longitudinal control continues to use the matched baseline ADS
    command, so the intervention isolates the lane-change semantics.
    """

    def __init__(
        self,
        baseline_actions: np.ndarray,
        initial_states: np.ndarray,
        *,
        map_polylines: np.ndarray | None = None,
        map_polyline_valid: np.ndarray | None = None,
        direction: str = "left",
        start_frame: int = 25,
        lane_width_m: float = 3.6,
        lateral_gain_per_s: float = 0.9,
        heading_gain_per_s: float = 6.0,
        maximum_lateral_speed_mps: float = 2.0,
        maximum_yaw_rate_rps: float = 0.35,
        lateral_tolerance_m: float = 0.15,
        heading_tolerance_rad: float = 0.02,
    ) -> None:
        actions = np.asarray(baseline_actions, np.float32)
        states = np.asarray(initial_states, np.float32)
        if actions.ndim != 3 or actions.shape[-1] != 2:
            raise ValueError("baseline_actions must have shape [batch,time,2]")
        if states.shape != (len(actions), 7, 6):
            raise ValueError("initial_states must have shape [batch,7,6]")
        if direction not in {"left", "right"}:
            raise ValueError("direction must be 'left' or 'right'")
        if not 0 <= start_frame < actions.shape[1]:
            raise ValueError("start_frame must lie inside baseline_actions")
        for name, value in (
            ("lane_width_m", lane_width_m),
            ("lateral_gain_per_s", lateral_gain_per_s),
            ("heading_gain_per_s", heading_gain_per_s),
            ("maximum_lateral_speed_mps", maximum_lateral_speed_mps),
            ("maximum_yaw_rate_rps", maximum_yaw_rate_rps),
            ("lateral_tolerance_m", lateral_tolerance_m),
            ("heading_tolerance_rad", heading_tolerance_rad),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        sign = 1.0 if direction == "left" else -1.0
        self.baseline_actions = actions.copy()
        self.start_frame = int(start_frame)
        self.lane_width_m = float(lane_width_m)
        self.lateral_gain_per_s = float(lateral_gain_per_s)
        self.heading_gain_per_s = float(heading_gain_per_s)
        self.maximum_lateral_speed_mps = float(maximum_lateral_speed_mps)
        self.maximum_yaw_rate_rps = float(maximum_yaw_rate_rps)
        self.lateral_tolerance_m = float(lateral_tolerance_m)
        self.heading_tolerance_rad = float(heading_tolerance_rad)
        self.source_y = states[:, 0, 1].copy()
        self.target_y = self.source_y + sign * self.lane_width_m
        if (map_polylines is None) != (map_polyline_valid is None):
            raise ValueError("map_polylines and map_polyline_valid must be supplied together")
        if map_polylines is not None:
            lane_map = np.asarray(map_polylines, np.float32)
            lane_valid = np.asarray(map_polyline_valid, bool)
            if lane_map.ndim != 4 or lane_map.shape[0] != len(states):
                raise ValueError("map_polylines must have shape [batch,lane,point,feature]")
            if lane_valid.shape != lane_map.shape[:3]:
                raise ValueError("map_polyline_valid shape does not match map_polylines")
            lane_y = lane_map[..., 0, 1]
            present = lane_valid.any(axis=-1)
            source_distance = np.where(
                present, np.abs(lane_y - self.source_y[:, None]), np.inf
            )
            if not present.any(axis=1).all():
                raise ValueError("ADS world has no mapped source lane")
            source_index = source_distance.argmin(axis=1)
            self.source_y = lane_y[np.arange(len(states)), source_index].copy()
            displacement = (lane_y - self.source_y[:, None]) * sign
            adjacent = present & (displacement > 1.8) & (displacement < 5.4)
            if not adjacent.any(axis=1).all():
                raise ValueError("requested ADS lane change has no mapped target lane")
            target_index = np.where(adjacent, displacement, np.inf).argmin(axis=1)
            self.target_y = lane_y[np.arange(len(states)), target_index].copy()

    def __call__(self, context: dict[str, torch.Tensor | int]) -> torch.Tensor:
        index = int(context["reference_index"])
        current = context["agent_states"]
        if not isinstance(current, torch.Tensor):
            raise TypeError("agent_states must be a torch tensor")
        action = torch.as_tensor(
            self.baseline_actions[:, index], device=current.device, dtype=current.dtype
        ).clone()
        if index < self.start_frame:
            return action

        ego = current[:, 0]
        target_y = torch.as_tensor(self.target_y, device=ego.device, dtype=ego.dtype)
        error_y = target_y - ego[:, 1]
        speed = torch.linalg.vector_norm(ego[:, 2:4], dim=-1).clamp_min(0.5)
        heading = torch.atan2(ego[:, 3], ego[:, 2])
        desired_vy = self.maximum_lateral_speed_mps * torch.tanh(
            self.lateral_gain_per_s * error_y / self.maximum_lateral_speed_mps
        )
        desired_heading = torch.asin(
            (desired_vy / speed).clamp(-0.95, 0.95)
        )
        yaw_rate = self.heading_gain_per_s * _wrap_angle(desired_heading - heading)
        yaw_rate = yaw_rate.clamp(
            -self.maximum_yaw_rate_rps, self.maximum_yaw_rate_rps
        )
        # Keep the feedback active after first entering the tolerance band.
        # Zeroing yaw rate at first entry leaves residual lateral velocity and
        # can drift the vehicle back outside the target-lane tolerance.
        action[:, 1] = yaw_rate
        return action

    def diagnostics(self, states: np.ndarray) -> dict[str, np.ndarray | float]:
        values = np.asarray(states, np.float32)
        if values.ndim != 4 or values.shape[0] != len(self.target_y) or values.shape[2:] != (7, 6):
            raise ValueError("states must have shape [batch,time,7,6]")
        final = values[:, -1, 0]
        final_heading = np.arctan2(final[:, 3], final[:, 2])
        final_error = final[:, 1] - self.target_y
        completed = (np.abs(final_error) <= self.lateral_tolerance_m) & (
            np.abs(final_heading) <= self.heading_tolerance_rad
        )
        direction = np.sign(self.target_y - self.source_y)
        signed_progress = (
            values[:, :, 0, 1] - self.source_y[:, None]
        ) * direction[:, None]
        lane_distance = np.abs(self.target_y - self.source_y)
        overshoot = np.maximum(signed_progress - lane_distance[:, None], 0.0)
        return {
            "target_y_m": self.target_y.copy(),
            "final_lateral_error_m": final_error,
            "final_heading_error_rad": final_heading,
            "completed": completed,
            "maximum_target_overshoot_m": overshoot.max(axis=1),
            "completion_rate": float(completed.mean()),
        }

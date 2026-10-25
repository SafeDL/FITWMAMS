"""Closed-loop ADS interventions with complete longitudinal/lateral semantics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .scene_geometry import (
    torch_adjacent_lane_index,
    torch_map_lane_geometry,
    torch_nearest_lane_geometry,
)


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
        if (
            min(
                lane_width_m,
                lateral_gain_per_s,
                heading_gain_per_s,
                maximum_lateral_speed_mps,
                maximum_yaw_rate_rps,
            )
            <= 0
        ):
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
                lane_y, lane_present, lane_width = torch_map_lane_geometry(
                    lanes, lane_valid
                )
                source_index, _, _, source_available = torch_nearest_lane_geometry(
                    ego[:, 1], lane_y, lane_present, lane_width
                )
                if not bool(source_available.all()):
                    raise ValueError("ADS world has no mapped source lane")
                target_index, available = torch_adjacent_lane_index(
                    source_index,
                    lane_y,
                    lane_present,
                    lane_width,
                    side=1 if sign > 0 else -1,
                )
                if not bool(available.all()):
                    self.target_y = None
                    raise ValueError(
                        "requested ADS lane change has no mapped target lane"
                    )
                self.target_y = (
                    torch.gather(lane_y, 1, target_index[:, None])
                    .squeeze(1)
                    .detach()
                    .clone()
                )
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

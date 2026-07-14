"""Differentiable kinematic dynamics and highD compatibility conversion."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DynamicsConfig:
    speed_min_mps: float = 0.0
    speed_max_mps: float = 50.0
    acceleration_min_mps2: float = -8.0
    acceleration_max_mps2: float = 4.0


class KinematicTrafficDynamics:
    """Unicycle/single-track compatibility dynamics using `[a, yaw_rate]`."""

    version = "kinematic_unicycle_v1"

    def __init__(self, cfg: DynamicsConfig | None = None) -> None:
        self.cfg = cfg or DynamicsConfig()

    @staticmethod
    def _safe_heading(vy: torch.Tensor, vx: torch.Tensor) -> torch.Tensor:
        # atan2(0, 0) has an undefined backward derivative.  Invalid/padded
        # agents are masked after integration, but must still be numerically
        # safe while autograd evaluates the vectorized operation.
        safe_vx = torch.where(vx.abs() < 1.0e-4, torch.full_like(vx, 1.0e-4), vx)
        return torch.atan2(vy, safe_vx)

    def step(self, states: torch.Tensor, controls: torch.Tensor, valid: torch.Tensor, dt: float) -> torch.Tensor:
        x, y, vx, vy = (states[..., index] for index in range(4))
        speed = torch.sqrt(vx.square() + vy.square() + 1.0e-8)
        heading = self._safe_heading(vy, vx)
        accel = controls[..., 0].clamp(self.cfg.acceleration_min_mps2, self.cfg.acceleration_max_mps2)
        yaw_rate = controls[..., 1]
        next_speed = (speed + accel * float(dt)).clamp(self.cfg.speed_min_mps, self.cfg.speed_max_mps)
        next_heading = heading + yaw_rate * float(dt)
        next_x = x + speed * torch.cos(heading) * float(dt) + 0.5 * accel * torch.cos(heading) * float(dt) ** 2
        next_y = y + speed * torch.sin(heading) * float(dt) + 0.5 * accel * torch.sin(heading) * float(dt) ** 2
        next_vx = next_speed * torch.cos(next_heading)
        next_vy = next_speed * torch.sin(next_heading)
        ax = accel * torch.cos(next_heading) - next_speed * yaw_rate * torch.sin(next_heading)
        ay = accel * torch.sin(next_heading) + next_speed * yaw_rate * torch.cos(next_heading)
        output = torch.stack((next_x, next_y, next_vx, next_vy, ax, ay), dim=-1)
        return output * valid[..., None].float()

    @staticmethod
    def highd_actions(controls: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Convert universal `[a,yaw_rate]` to highD `[ax,ay_left]`."""
        speed = torch.linalg.vector_norm(states[..., 2:4], dim=-1)
        heading = KinematicTrafficDynamics._safe_heading(states[..., 3], states[..., 2])
        accel, yaw_rate = controls[..., 0], controls[..., 1]
        return torch.stack((
            accel * torch.cos(heading) - speed * yaw_rate * torch.sin(heading),
            accel * torch.sin(heading) + speed * yaw_rate * torch.cos(heading),
        ), dim=-1)

    @staticmethod
    def controls_from_highd_actions(actions: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Project legacy Cartesian acceleration labels to `[a,yaw_rate]`."""
        speed = torch.linalg.vector_norm(states[..., 2:4], dim=-1).clamp_min(0.5)
        heading = KinematicTrafficDynamics._safe_heading(states[..., 3], states[..., 2])
        ax, ay = actions[..., 0], actions[..., 1]
        longitudinal = ax * torch.cos(heading) + ay * torch.sin(heading)
        lateral = -ax * torch.sin(heading) + ay * torch.cos(heading)
        return torch.stack((longitudinal, lateral / speed), dim=-1)

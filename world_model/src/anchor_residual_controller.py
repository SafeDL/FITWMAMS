"""Bounded graph-conditioned residual controls for the first second only."""
from __future__ import annotations

import torch
import torch.nn as nn


class AnchorResidualController(nn.Module):
    """Predict a small physics-rate residual around a deterministic B0 plan."""

    def __init__(self, hidden_dim: int, physics_steps: int, residual_accel_limit: float = 2.0, residual_yaw_limit: float = 0.20) -> None:
        super().__init__()
        self.physics_steps = int(physics_steps)
        self.residual_accel_limit = float(residual_accel_limit)
        self.residual_yaw_limit = float(residual_yaw_limit)
        self.network = nn.Sequential(
            # Keep target, realized prefix and their difference in the same
            # frozen-Flow coordinate system.  Feeding only a difference made
            # it too easy to accidentally mix physical units and normalized
            # coordinates at this boundary.
            nn.Linear(hidden_dim * 3 + 6 * 3 + 2 + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, self.physics_steps * 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        agent_context: torch.Tensor,
        scene_context: torch.Tensor,
        latent_context: torch.Tensor,
        target_anchor_std: torch.Tensor,
        realized_anchor_std: torch.Tensor,
        remaining_anchor_std: torch.Tensor,
        start_controls: torch.Tensor,
        seconds_remaining: float,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return `[B, physics_steps, agents, 2]` residual universal controls."""
        b, n, _ = agent_context.shape
        scene = scene_context[:, None].expand(-1, n, -1)
        latent = latent_context[:, None].expand(-1, n, -1)
        # Ego does not own a Flow slot; give it a zero condition and mask it.
        target = torch.zeros((b, n, 6), dtype=agent_context.dtype, device=agent_context.device)
        realized = torch.zeros_like(target)
        remaining = torch.zeros_like(target)
        target[:, 1:] = target_anchor_std
        realized[:, 1:] = realized_anchor_std
        remaining[:, 1:] = remaining_anchor_std
        start = start_controls.mean(dim=1)
        time = torch.full((b, n, 1), float(max(seconds_remaining, 0.0)), dtype=agent_context.dtype, device=agent_context.device)
        output = self.network(torch.cat((agent_context, scene, latent, target, realized, remaining, start, time), dim=-1))
        output = output.reshape(b, n, self.physics_steps, 2).permute(0, 2, 1, 3)
        bounded = torch.stack((
            torch.tanh(output[..., 0]) * self.residual_accel_limit,
            torch.tanh(output[..., 1]) * self.residual_yaw_limit,
        ), dim=-1)
        return bounded * valid[:, None, :, None].to(dtype=bounded.dtype)

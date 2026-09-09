"""Convex longitudinal cost bases for nominal-preserving decisions.

The bases are represented as affine residuals in the 25-frame action vector.
Their positive-part squares are encoded with non-negative QP slack variables;
no future vehicle state is an input to these functions.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DecisionCostInputs:
    gap_m: torch.Tensor
    speed_mps: torch.Tensor
    leader_speed_mps: torch.Tensor
    leader_acceleration_mps2: torch.Tensor
    reference_speed_mps: torch.Tensor


class PositiveCostWeights(nn.Module):
    """At-most-two-layer causal feature head with a positive output floor."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64, layers: int = 2, floor: float = 1e-4) -> None:
        super().__init__()
        if not 1 <= layers <= 2 or hidden_dim > 64:
            raise ValueError("head must use one/two layers with hidden_dim <= 64")
        body: list[nn.Module] = [nn.Linear(feature_dim, hidden_dim), nn.SiLU()]
        if layers == 2:
            body.extend((nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
        body.append(nn.Linear(hidden_dim, 3))
        self.net = nn.Sequential(*body)
        self.floor = float(floor)
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight); nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.net(features)) + self.floor


def affine_cost_residuals(
    values: DecisionCostInputs, horizon: int, *, dt_s: float = 0.04,
    d0_m: float = 2.0, time_headway_s: float = 1.5,
    gap_scale_m: float = 5.0, speed_scale_mps: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``M,b`` where three residual families equal ``M @ a + b``.

    Each row is [gap-deficit, positive-closing, reference-speed] at a future
    control frame.  Leader motion is constant acceleration extrapolated only
    from its realized boundary history.
    """
    batch = values.gap_m.shape[0]
    for field in values.__dict__.values():
        if field.shape != (batch,):
            raise ValueError("cost inputs must all be [batch]")
    dtype, device = values.gap_m.dtype, values.gap_m.device
    cumsum = torch.tril(torch.ones((horizon, horizon), dtype=dtype, device=device))
    position = torch.tril(cumsum) * float(dt_s * dt_s)
    frames = torch.arange(1, horizon + 1, dtype=dtype, device=device) * float(dt_s)
    speed_map = float(dt_s) * cumsum
    # d0 + T*v - gap.  Own acceleration lowers predicted gap and raises speed.
    gap_m = (float(time_headway_s) * speed_map + position) / float(gap_scale_m)
    closing_m = speed_map / float(speed_scale_mps)
    speed_m = speed_map / float(speed_scale_mps)
    matrix = torch.stack((gap_m, closing_m, speed_m), dim=1).expand(batch, -1, -1, -1)
    lead_position = values.gap_m[:, None] + frames * (values.leader_speed_mps - values.speed_mps)[:, None] + 0.5 * frames.square()[None] * values.leader_acceleration_mps2[:, None]
    lead_speed = values.leader_speed_mps[:, None] + frames[None] * values.leader_acceleration_mps2[:, None]
    b_gap = (float(d0_m) + float(time_headway_s) * values.speed_mps[:, None] - lead_position) / float(gap_scale_m)
    b_closing = (values.speed_mps[:, None] - lead_speed) / float(speed_scale_mps)
    b_speed = (values.speed_mps[:, None] - values.reference_speed_mps[:, None]).expand(-1, horizon) / float(speed_scale_mps)
    return matrix, torch.stack((b_gap, b_closing, b_speed), dim=-1)

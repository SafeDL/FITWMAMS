"""Shared continuous memory for closed-loop traffic world models."""

from __future__ import annotations

import torch
import torch.nn as nn


class ContinuousTrafficMemory(nn.Module):
    """Update a continuous scene state from generated traffic history."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 12, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(
        self,
        scene: torch.Tensor,
        agents: torch.Tensor,
        previous_plan: torch.Tensor | None,
        state_delta: torch.Tensor,
        memory: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, hidden = scene.shape
        pooled_agents = agents.mean(dim=1)
        if previous_plan is None:
            plan_summary = scene.new_zeros((batch, hidden))
        else:
            plan_summary = torch.nn.functional.pad(
                previous_plan.mean(dim=(1, 2)), (0, max(0, hidden - 2))
            )[:, :hidden]
        delta = torch.nn.functional.pad(state_delta.mean(dim=1), (0, 6))[:, :12]
        value = self.input(torch.cat((scene, pooled_agents, plan_summary, delta), dim=-1))
        return self.gru(value, scene.new_zeros((batch, hidden)) if memory is None else memory)

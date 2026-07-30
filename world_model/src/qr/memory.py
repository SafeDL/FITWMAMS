"""Single persistent scene memory used by QR-WM."""

from __future__ import annotations

import torch
import torch.nn as nn


class PersistentSceneMemory(nn.Module):
    """Carry traffic interaction, prior background actions, and state change.

    The state follows ``m_t=f(m_{t-1}, S_t, U_{t-1}, ΔS_t)``.  Ego motion is
    represented by its observed state change, never by an ADS action input.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.plan_summary = nn.Sequential(nn.Linear(4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.input = nn.Sequential(nn.Linear(hidden_dim * 3 + 6, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.update = nn.GRUCell(hidden_dim, hidden_dim)

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
            plan = scene.new_zeros((batch, hidden))
        else:
            # Mean and absolute variation distinguish a calm carried buffer
            # from a braking/turning response without keeping another memory.
            summary = torch.cat(
                (previous_plan.mean(dim=(1, 2)), previous_plan.abs().mean(dim=(1, 2))), dim=-1
            )
            plan = self.plan_summary(summary)
        delta = state_delta.mean(dim=1)
        value = self.input(torch.cat((scene, pooled_agents, plan, delta), dim=-1))
        initial = scene.new_zeros((batch, hidden)) if memory is None else memory
        return self.update(value, initial)

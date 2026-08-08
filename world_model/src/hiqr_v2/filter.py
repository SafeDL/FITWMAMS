"""Observation-only filter and transient two-time-scale innovations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import HiQRV2Config


def masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.float()
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


@dataclass(frozen=True)
class FilterState:
    """Only observed traffic state; it deliberately contains no latent intent."""

    global_hidden: torch.Tensor
    agent_hidden: torch.Tensor

    def detach(self) -> "FilterState":
        return FilterState(self.global_hidden.detach(), self.agent_hidden.detach())


class ObservedHierarchicalInteractionFilter(nn.Module):
    """Filter observed traffic before drawing transient scene/agent intentions."""

    def __init__(self, cfg: HiQRV2Config) -> None:
        super().__init__()
        self.cfg = cfg
        h, g, z = (
            int(cfg.hidden_dim),
            int(cfg.scene_latent_dim),
            int(cfg.agent_residual_dim),
        )
        self.b0_agent = nn.Sequential(nn.Linear(6, h), nn.SiLU(), nn.Linear(h, h))
        self.agent_initializer = nn.Sequential(
            nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, h)
        )
        self.global_initializer = nn.Sequential(
            nn.Linear(3 * h, h), nn.SiLU(), nn.Linear(h, h)
        )
        self.global_delta = nn.Sequential(nn.Linear(6, h), nn.SiLU(), nn.Linear(h, h))
        self.agent_delta = nn.Sequential(nn.Linear(6, h), nn.SiLU(), nn.Linear(h, h))
        self.global_observer = nn.GRUCell(2 * h, h)
        self.agent_observer = nn.GRUCell(2 * h, h)
        self.scene_prior = nn.Sequential(
            nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, g)
        )
        self.agent_prior = nn.Sequential(
            nn.Linear(3 * h + g, h), nn.SiLU(), nn.Linear(h, z)
        )

    @staticmethod
    def _pool(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.float()[..., None]
        return (value * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def initialize(
        self,
        scene: torch.Tensor,
        agents: torch.Tensor,
        raw_b0: torch.Tensor,
        b0_valid: torch.Tensor,
    ) -> FilterState:
        if raw_b0.shape[1:] != (6, 6) or b0_valid.shape != raw_b0.shape[:2]:
            raise ValueError(
                "HiQR-v2 B0 must be [batch, 6, 6] with [batch, 6] validity"
            )
        b0 = self.b0_agent(raw_b0) * b0_valid[..., None].float()
        background = self.agent_initializer(torch.cat((agents[:, 1:], b0), dim=-1))
        agent_hidden = torch.zeros_like(agents)
        agent_hidden[:, 1:] = background * b0_valid[..., None].float()
        pooled_b0 = self._pool(b0, b0_valid)
        pooled_agents = self._pool(background, b0_valid)
        global_hidden = self.global_initializer(
            torch.cat((scene, pooled_b0, pooled_agents), dim=-1)
        )
        return FilterState(global_hidden, agent_hidden)

    def observe(
        self,
        state: FilterState,
        agents: torch.Tensor,
        scene: torch.Tensor,
        current: torch.Tensor,
        previous_current: torch.Tensor | None,
        current_valid: torch.Tensor,
    ) -> FilterState:
        """Update only from observed/generated physical state, never from g/z."""
        delta = (
            torch.zeros_like(current)
            if previous_current is None
            else current - previous_current
        )
        global_delta = self.global_delta(self._pool(delta, current_valid))
        global_hidden = self.global_observer(
            torch.cat((scene, global_delta), dim=-1), state.global_hidden
        )
        agent_delta = self.agent_delta(delta)
        agent_input = torch.cat((agents, agent_delta), dim=-1).reshape(
            -1, 2 * agents.shape[-1]
        )
        agent_hidden = self.agent_observer(
            agent_input, state.agent_hidden.reshape(-1, agents.shape[-1])
        ).reshape_as(state.agent_hidden)
        return FilterState(
            global_hidden, agent_hidden * current_valid[..., None].float()
        )

    def prior_scene(self, state: FilterState, scene: torch.Tensor) -> torch.Tensor:
        return self.scene_prior(torch.cat((scene, state.global_hidden), dim=-1))

    def prior_agents(
        self, state: FilterState, agents: torch.Tensor, scene_latent: torch.Tensor
    ) -> torch.Tensor:
        shared = torch.cat((state.global_hidden, scene_latent), dim=-1)[:, None].expand(
            -1, agents.shape[1], -1
        )
        return self.agent_prior(torch.cat((agents, state.agent_hidden, shared), dim=-1))

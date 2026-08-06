"""Observation-only persistent filter and local hierarchical posterior for V2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional

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
            nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, 2 * g)
        )
        self.agent_prior = nn.Sequential(
            nn.Linear(3 * h + g, h), nn.SiLU(), nn.Linear(h, 2 * z)
        )
        self.future = nn.Sequential(nn.Linear(12, h), nn.SiLU(), nn.Linear(h, h))
        self.scene_posterior = nn.Sequential(
            nn.Linear(3 * h, h), nn.SiLU(), nn.Linear(h, 2 * g)
        )
        self.agent_posterior = nn.Sequential(
            nn.Linear(4 * h + g, h), nn.SiLU(), nn.Linear(h, 2 * z)
        )

    @staticmethod
    def distribution_parameters(
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw = value.chunk(2, dim=-1)
        return mean, -3.0 + 2.5 * torch.sigmoid(raw)

    @staticmethod
    def _pool(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.float()[..., None]
        return (value * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _future_features(
        current: torch.Tensor, future: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        weights = valid.float()[..., None]
        count = weights.sum(dim=1).clamp_min(1.0)
        delta = ((future - current[:, None]) * weights).sum(dim=1) / count
        last_index = valid.long().sum(dim=1).sub(1).clamp_min(0)
        last = future.gather(
            1, last_index[:, None, :, None].expand(-1, 1, -1, future.shape[-1])
        ).squeeze(1)
        return torch.cat((delta[..., :6], (last - current)[..., :6]), dim=-1)

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
        if self.cfg.filter_update_mode == "stateless":
            return FilterState(
                torch.zeros_like(scene),
                torch.zeros_like(agents),
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
        if self.cfg.filter_update_mode == "stateless":
            return FilterState(
                torch.zeros_like(state.global_hidden),
                torch.zeros_like(state.agent_hidden),
            )
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

    def prior_scene(
        self, state: FilterState, scene: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.distribution_parameters(
            self.scene_prior(torch.cat((scene, state.global_hidden), dim=-1))
        )

    def prior_agents(
        self, state: FilterState, agents: torch.Tensor, scene_latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = torch.cat((state.global_hidden, scene_latent), dim=-1)[:, None].expand(
            -1, agents.shape[1], -1
        )
        return self.distribution_parameters(
            self.agent_prior(torch.cat((agents, state.agent_hidden, shared), dim=-1))
        )

    def posterior(
        self,
        state: FilterState,
        agents: torch.Tensor,
        scene: torch.Tensor,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        future: torch.Tensor,
        future_valid: torch.Tensor,
        prior_scene: tuple[torch.Tensor, torch.Tensor],
        fixed_scene_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return a local posterior; callers must not put it back into FilterState."""
        future = future.clone()
        valid = future_valid.clone()
        future[:, :, 0] = current[:, None, 0]
        valid[:, :, 0] = current_valid[:, None, 0]
        future_agent = self.future(self._future_features(current, future, valid))
        background = current_valid.clone()
        background[:, 0] = False
        pooled = self._pool(future_agent, background)
        inferred_g, inferred_g_log = self.distribution_parameters(
            self.scene_posterior(
                torch.cat((scene, state.global_hidden, pooled), dim=-1)
            )
        )
        scene_refresh = fixed_scene_latent is None
        mean_g = inferred_g if scene_refresh else fixed_scene_latent
        log_g = inferred_g_log
        shared = torch.cat((state.global_hidden, mean_g), dim=-1)[:, None].expand(
            -1, agents.shape[1], -1
        )
        mean_z, log_z = self.distribution_parameters(
            self.agent_posterior(
                torch.cat((agents, state.agent_hidden, future_agent, shared), dim=-1)
            )
        )
        prior_g, prior_g_log = prior_scene
        prior_z, prior_z_log = self.prior_agents(state, agents, mean_g)
        scene_ratio = torch.exp(2.0 * (log_g - prior_g_log))
        scene_kl = (
            prior_g_log
            - log_g
            + 0.5
            * (
                scene_ratio
                + (mean_g - prior_g).square() * torch.exp(-2.0 * prior_g_log)
                - 1.0
            )
        ).mean()
        if not scene_refresh:
            scene_kl = scene_kl.new_zeros(())
        agent_ratio = torch.exp(2.0 * (log_z - prior_z_log))
        agent_kl = (
            prior_z_log
            - log_z
            + 0.5
            * (
                agent_ratio
                + (mean_z - prior_z).square() * torch.exp(-2.0 * prior_z_log)
                - 1.0
            )
        )
        agent_kl = masked_mean(agent_kl.mean(dim=-1), background)
        scene_distillation = (
            (mean_g - prior_g).square().mean()
            if scene_refresh
            else mean_g.new_zeros(())
        )
        distillation = scene_distillation + masked_mean(
            (mean_z - prior_z).square().mean(dim=-1), background
        )
        scene_diversity = (
            functional.relu(0.12 - torch.exp(log_g).mean())
            if scene_refresh
            else mean_g.new_zeros(())
        )
        diversity = scene_diversity + functional.relu(0.08 - torch.exp(log_z).mean())
        mean_z = mean_z * background[..., None].float()
        return (
            mean_g,
            mean_z,
            {
                "scene_kl": scene_kl,
                "agent_kl": agent_kl,
                "prior_distillation": distillation,
                "diversity_floor": diversity,
                "posterior_scene_std": (
                    torch.exp(log_g).mean() if scene_refresh else mean_g.new_zeros(())
                ),
                "posterior_agent_std": masked_mean(
                    torch.exp(log_z).mean(dim=-1), background
                ),
            },
        )

"""Hierarchical stochastic interaction state for HiQR-WM."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import HiQRWorldModelConfig


def masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.float()
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


class HierarchicalStochasticInteractionState(nn.Module):
    """Shared persistent state plus scene and agent-level latent variables."""

    def __init__(self, cfg: HiQRWorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h, g, z = (
            int(cfg.hidden_dim),
            int(cfg.scene_latent_dim),
            int(cfg.agent_residual_dim),
        )
        self.event = nn.Sequential(nn.Linear(6, h), nn.SiLU(), nn.Linear(h, h))
        self.b0 = nn.Sequential(nn.Linear(36, h), nn.SiLU(), nn.Linear(h, h))
        self.initializer = nn.Sequential(
            nn.Linear(h * 3, h), nn.SiLU(), nn.Linear(h, h)
        )
        self.scene_prior = nn.Sequential(
            nn.Linear(h * 2, h), nn.SiLU(), nn.Linear(h, g * 2)
        )
        self.agent_prior = nn.Sequential(
            nn.Linear(h * 3 + g, h), nn.SiLU(), nn.Linear(h, z * 2)
        )
        self.future = nn.Sequential(nn.Linear(12, h), nn.SiLU(), nn.Linear(h, h))
        self.scene_posterior = nn.Sequential(
            nn.Linear(h * 3, h), nn.SiLU(), nn.Linear(h, g * 2)
        )
        self.agent_posterior = nn.Sequential(
            nn.Linear(h * 4 + g, h), nn.SiLU(), nn.Linear(h, z * 2)
        )
        self.delta = nn.Sequential(nn.Linear(6, h), nn.SiLU(), nn.Linear(h, h))
        self.transition = nn.GRUCell(h * 2 + g + z, h)

    @staticmethod
    def event_features(slot_valid: torch.Tensor) -> torch.Tensor:
        """Encode only the t0-observable Flow slot mask."""
        if slot_valid.ndim != 2 or slot_valid.shape[1] != 6:
            raise ValueError("slot_valid must have shape [batch, 6]")
        return slot_valid.float()

    @staticmethod
    def distribution_parameters(
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw = value.chunk(2, dim=-1)
        return mean, -3.0 + 2.5 * torch.sigmoid(raw)

    @staticmethod
    def _future_features(
        current: torch.Tensor, future: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Future-only training features; caller has already neutralized ego."""
        weight = valid.float()[..., None]
        count = weight.sum(dim=1).clamp_min(1.0)
        delta = ((future - current[:, None]) * weight).sum(dim=1) / count
        last_index = valid.long().sum(dim=1).sub(1).clamp_min(0)
        last = future.gather(
            1, last_index[:, None, :, None].expand(-1, 1, -1, future.shape[-1])
        ).squeeze(1)
        return torch.cat((delta[..., :6], (last - current)[..., :6]), dim=-1)

    def initialize(
        self,
        scene: torch.Tensor,
        raw_b0: torch.Tensor,
        behavior_anchor_valid: torch.Tensor,
        event_slot_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Use B0 exactly once, as a Flow-conditioned interaction-state seed."""
        if raw_b0.ndim != 3 or raw_b0.shape[1:] != (6, 6):
            raise ValueError("raw_b0 must have shape [batch, 6, 6]")
        if behavior_anchor_valid.shape != raw_b0.shape[:2]:
            raise ValueError("behavior_anchor_valid must have shape [batch, 6]")
        if event_slot_valid.shape != raw_b0.shape[:2]:
            raise ValueError("event_slot_valid must have shape [batch, 6]")
        masked_b0 = raw_b0 * behavior_anchor_valid[..., None].float()
        return self.initializer(
            torch.cat(
                (
                    scene,
                    self.b0(masked_b0.flatten(1)),
                    self.event(self.event_features(event_slot_valid)),
                ),
                dim=-1,
            )
        )

    def prior_scene(
        self, scene: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.distribution_parameters(
            self.scene_prior(torch.cat((scene, hidden), dim=-1))
        )

    def prior_agents(
        self,
        agents: torch.Tensor,
        scene: torch.Tensor,
        hidden: torch.Tensor,
        scene_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = torch.cat((scene, hidden, scene_latent), dim=-1)[:, None].expand(
            -1, agents.shape[1], -1
        )
        return self.distribution_parameters(
            self.agent_prior(torch.cat((agents, shared), dim=-1))
        )

    def sample(
        self,
        agents: torch.Tensor,
        scene: torch.Tensor,
        hidden: torch.Tensor,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        *,
        deterministic: bool,
        use_posterior: bool = False,
        posterior_future: torch.Tensor | None = None,
        posterior_future_valid: torch.Tensor | None = None,
        scene_standard_normal: torch.Tensor | None = None,
        agent_standard_normal: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Sample ``g`` then all per-agent residuals, once per response."""
        scene_mean, scene_log = self.prior_scene(scene, hidden)
        agent_mean_prior: torch.Tensor | None = None
        agent_log_prior: torch.Tensor | None = None
        terms = {
            "scene_kl": current.new_zeros(()),
            "agent_kl": current.new_zeros(()),
            "diversity_floor": current.new_zeros(()),
        }
        mean_g, log_g = scene_mean, scene_log
        future_agent = None
        if use_posterior:
            if posterior_future is None or posterior_future_valid is None:
                raise ValueError(
                    "hierarchical posterior training requires "
                    "response-local future states"
                )
            future = posterior_future.clone()
            valid = posterior_future_valid.clone()
            # Future ego is training supervision, never a condition.
            future[:, :, 0] = current[:, None, 0]
            valid[:, :, 0] = current_valid[:, None, 0]
            future_agent = self.future(self._future_features(current, future, valid))
            background = current_valid.clone()
            background[:, 0] = False
            pooled = (future_agent * background[..., None].float()).sum(
                dim=1
            ) / background.float().sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_g, log_g = self.distribution_parameters(
                self.scene_posterior(torch.cat((scene, hidden, pooled), dim=-1))
            )
            ratio = torch.exp(2.0 * (log_g - scene_log))
            terms["scene_kl"] = (
                scene_log
                - log_g
                + 0.5
                * (
                    ratio
                    + (mean_g - scene_mean).square() * torch.exp(-2.0 * scene_log)
                    - 1.0
                )
            ).mean()
        if deterministic:
            if scene_standard_normal is not None or agent_standard_normal is not None:
                raise ValueError(
                    "deterministic HiQR rollout must not receive latent noise"
                )
            g = mean_g
        else:
            if scene_standard_normal is None:
                scene_noise = torch.randn_like(mean_g)
            else:
                if scene_standard_normal.shape != mean_g.shape:
                    raise ValueError(
                        "scene_standard_normal must be [batch, scene_latent_dim]"
                    )
                scene_noise = scene_standard_normal.to(
                    device=mean_g.device, dtype=mean_g.dtype
                )
            g = mean_g + scene_noise * torch.exp(log_g)
        agent_mean_prior, agent_log_prior = self.prior_agents(agents, scene, hidden, g)
        mean_z, log_z = agent_mean_prior, agent_log_prior
        if use_posterior:
            assert future_agent is not None
            shared = torch.cat((scene, hidden, g), dim=-1)[:, None].expand(
                -1, agents.shape[1], -1
            )
            mean_z, log_z = self.distribution_parameters(
                self.agent_posterior(torch.cat((agents, future_agent, shared), dim=-1))
            )
            ratio = torch.exp(2.0 * (log_z - agent_log_prior))
            kl = (
                agent_log_prior
                - log_z
                + 0.5
                * (
                    ratio
                    + (mean_z - agent_mean_prior).square()
                    * torch.exp(-2.0 * agent_log_prior)
                    - 1.0
                )
            )
            background = current_valid.clone()
            background[:, 0] = False
            terms["agent_kl"] = masked_mean(kl.mean(dim=-1), background)
        if deterministic:
            z = mean_z
        else:
            if agent_standard_normal is None:
                agent_noise = torch.randn_like(mean_z)
            elif agent_standard_normal.shape != mean_z.shape:
                raise ValueError(
                    "agent_standard_normal must be [batch, agents, agent_residual_dim]"
                )
            else:
                agent_noise = agent_standard_normal.to(
                    device=mean_z.device, dtype=mean_z.dtype
                )
            z = mean_z + agent_noise * torch.exp(log_z)
        background_valid = current_valid.clone()
        background_valid[:, 0] = False
        z = z * background_valid[..., None].float()
        terms["diversity_floor"] = functional.relu(
            0.12 - torch.exp(log_g).mean()
        ) + functional.relu(0.12 - torch.exp(log_z).mean())
        return g, z, terms

    def update(
        self,
        hidden: torch.Tensor,
        scene: torch.Tensor,
        scene_latent: torch.Tensor,
        agent_residual: torch.Tensor,
        delta_state: torch.Tensor,
        current_valid: torch.Tensor,
    ) -> torch.Tensor:
        background = current_valid.clone()
        background[:, 0] = False
        pooled_residual = (agent_residual * background[..., None].float()).sum(
            dim=1
        ) / background.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_delta = (delta_state * current_valid[..., None].float()).sum(
            dim=1
        ) / current_valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        value = torch.cat(
            (scene, self.delta(pooled_delta), scene_latent, pooled_residual), dim=-1
        )
        return self.transition(value, hidden)

"""Causal, slot-aware joint jerk centre for FIRM-WM's rolling plan."""

from __future__ import annotations

import torch
import torch.nn as nn


class RelationalPlanField(nn.Module):
    """Generate one coordinated 1 s raw-jerk field, not Top-K candidates.

    The field retains each background vehicle's relation token instead of
    collapsing the scene before controls are decoded.  At every 25 Hz plan
    frame, a self-attention pass coordinates all valid background slots.  Its
    output is a *raw* jerk centre; the caller applies tanh/physical limits and
    uses the same raw centre in the residual-flow likelihood.
    """

    def __init__(
        self,
        hidden_dim: int,
        world_latent_dim: int,
        frames: int,
        *,
        agents: int = 6,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.frames = int(frames)
        self.agents = int(agents)
        self.previous = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.agent = nn.Sequential(
            nn.Linear(hidden_dim * 5 + world_latent_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.slot_embedding = nn.Parameter(torch.empty(agents, hidden_dim))
        self.frame_embedding = nn.Parameter(torch.empty(frames, hidden_dim))
        nn.init.normal_(self.slot_embedding, std=0.02)
        nn.init.normal_(self.frame_embedding, std=0.02)
        self.coordination = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2)
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
    def forward(
        self,
        agents: torch.Tensor,
        scene: torch.Tensor,
        memory: torch.Tensor,
        world_latent: torch.Tensor,
        flow_embedding: torch.Tensor,
        current_valid: torch.Tensor,
        previous_plan: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return raw jerk centre with shape ``[B, plan_frames, 6, 2]``."""
        batch = agents.shape[0]
        background = agents[:, 1:]
        if previous_plan is None:
            previous = background.new_zeros((batch, self.agents, 4))
        else:
            # The per-slot mean and final applied plan controls make previous
            # plan intent available without accessing an unexecuted future.
            previous = torch.cat((previous_plan.mean(dim=1), previous_plan[:, -1]), dim=-1)
        shared = torch.cat((scene, memory, world_latent, flow_embedding), dim=-1)
        shared = shared[:, None].expand(-1, self.agents, -1)
        token = self.agent(torch.cat((background, shared, self.previous(previous)), dim=-1))
        token = token + self.slot_embedding[None]
        frame_token = token[:, None] + self.frame_embedding[None, :, None]
        frame_token = frame_token.reshape(batch * self.frames, self.agents, self.hidden_dim)
        key_padding = (~current_valid[:, 1:]).bool()
        key_padding = key_padding[:, None].expand(-1, self.frames, -1).reshape(
            batch * self.frames, self.agents
        )
        key_padding = torch.where(
            key_padding.all(dim=1, keepdim=True), torch.zeros_like(key_padding), key_padding
        )
        coordinated, _ = self.coordination(
            frame_token, frame_token, frame_token, key_padding_mask=key_padding, need_weights=False
        )
        raw = self.output(coordinated).reshape(batch, self.frames, self.agents, 2)
        return raw * current_valid[:, None, 1:, None].float()

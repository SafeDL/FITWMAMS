"""Deterministic joint agent-time refinement for QR-WM."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import safe_key_padding_mask


class _AgentTimeBlock(nn.Module):
    """Temporal, inter-agent, then scene/map cross-attention in one block."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.temporal = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.agents = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.scene_cross = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.agent_norm = nn.LayerNorm(hidden_dim)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor, context: torch.Tensor, context_valid: torch.Tensor) -> torch.Tensor:
        batch, frames, agents, hidden = tokens.shape
        temporal_tokens = tokens.permute(0, 2, 1, 3).reshape(batch * agents, frames, hidden)
        temporal_valid = valid.permute(0, 2, 1).reshape(batch * agents, frames)
        temporal, _ = self.temporal(
            temporal_tokens, temporal_tokens, temporal_tokens,
            key_padding_mask=safe_key_padding_mask(temporal_valid), need_weights=False,
        )
        temporal = self.temporal_norm(temporal_tokens + self.dropout(temporal))
        tokens = temporal.reshape(batch, agents, frames, hidden).permute(0, 2, 1, 3)

        agent_tokens = tokens.reshape(batch * frames, agents, hidden)
        agent_valid = valid.reshape(batch * frames, agents)
        cross_agents, _ = self.agents(
            agent_tokens, agent_tokens, agent_tokens,
            key_padding_mask=safe_key_padding_mask(agent_valid), need_weights=False,
        )
        tokens = self.agent_norm(agent_tokens + self.dropout(cross_agents)).reshape(batch, frames, agents, hidden)

        query = tokens.reshape(batch, frames * agents, hidden)
        cross, _ = self.scene_cross(
            query, context, context, key_padding_mask=safe_key_padding_mask(context_valid), need_weights=False
        )
        tokens = self.cross_norm(query + self.dropout(cross)).reshape(batch, frames, agents, hidden)
        tokens = self.feed_forward_norm(tokens + self.dropout(self.feed_forward(tokens)))
        return tokens * valid[..., None].float()


class JointAgentTimeRefiner(nn.Module):
    """Generate and refine one joint background future-action sequence."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        behavior_latent_dim: int,
        plan_frames: int,
        attention_layers: int,
        num_heads: int,
        dropout: float,
        min_acceleration: float,
        max_acceleration: float,
        max_yaw_rate: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads for QR-WM attention")
        self.plan_frames = int(plan_frames)
        self.min_acceleration = float(min_acceleration)
        self.max_acceleration = float(max_acceleration)
        self.max_yaw_rate = float(max_yaw_rate)
        self.register_buffer("action_scale", torch.tensor((1.5, 0.15)))
        self.time_embedding = nn.Parameter(torch.randn(self.plan_frames, hidden_dim) * 0.02)
        self.agent_embedding = nn.Parameter(torch.randn(6, hidden_dim) * 0.02)
        self.seed = nn.Sequential(
            nn.Linear(hidden_dim * 2 + behavior_latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.seed_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2))
        self.behavior_embedding = nn.Sequential(
            nn.Linear(behavior_latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.action_embedding = nn.Sequential(nn.Linear(2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.state_embedding = nn.Sequential(nn.Linear(6, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.scene_embedding = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.memory_embedding = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.map_embedding = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList(
            [_AgentTimeBlock(hidden_dim, num_heads, dropout) for _ in range(max(1, int(attention_layers)))]
        )
        self.residual_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2))

    def _clamp(self, raw: torch.Tensor) -> torch.Tensor:
        acceleration = torch.tanh(raw[..., 0]) * max(abs(self.min_acceleration), abs(self.max_acceleration))
        yaw_rate = torch.tanh(raw[..., 1]) * self.max_yaw_rate
        return torch.stack(
            (acceleration.clamp(self.min_acceleration, self.max_acceleration), yaw_rate), dim=-1
        )

    def fresh_plan(
        self,
        agents: torch.Tensor,
        scene_memory: torch.Tensor,
        behavior: torch.Tensor,
    ) -> torch.Tensor:
        """Create an unrefined background future-action sequence."""
        background = agents[:, 1:]
        count = background.shape[1]
        if count > self.agent_embedding.shape[0]:
            raise ValueError("QR-WM joint refiner supports at most six background slots")
        base = self.seed(torch.cat((background, scene_memory[:, None].expand(-1, count, -1), behavior[:, 1:]), dim=-1))
        token = (
            base[:, None]
            + self.time_embedding[None, :, None]
            + self.agent_embedding[None, None, :count]
        )
        return self._clamp(self.seed_head(token))

    def residual(
        self,
        actions: torch.Tensor,
        plan_states: torch.Tensor,
        agents: torch.Tensor,
        scene: torch.Tensor,
        scene_memory: torch.Tensor,
        behavior: torch.Tensor,
        map_tokens: torch.Tensor,
        map_valid: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return a masked residual for a background future-action sequence."""
        batch, frames, count, _ = actions.shape
        relative = plan_states.clone()
        relative[..., :2] = relative[..., :2] - plan_states[:, :1, :, :2]
        background_tokens = (
            self.action_embedding(actions / self.action_scale)
            + self.state_embedding(relative)
            + agents[:, None, 1:]
            + self.behavior_embedding(behavior[:, None, 1:])
            + self.time_embedding[None, :frames, None]
            + self.agent_embedding[None, None, :count]
        )
        # Repeat only the observed ego context over the planning horizon.  It
        # participates in attention but receives no background action residual.
        ego_token = (
            agents[:, None, :1] + self.time_embedding[None, :frames, None]
        )
        tokens = torch.cat((ego_token, background_tokens), dim=2)
        token_valid = torch.cat((torch.ones((batch, frames, 1), dtype=torch.bool, device=actions.device), valid), dim=2)
        context = torch.cat(
            (self.scene_embedding(scene)[:, None], self.memory_embedding(scene_memory)[:, None], self.map_embedding(map_tokens)), dim=1
        )
        context_valid = torch.cat((torch.ones((batch, 2), dtype=torch.bool, device=actions.device), map_valid.bool()), dim=1)
        for block in self.blocks:
            tokens = block(tokens, token_valid, context, context_valid)
        raw = self.residual_head(tokens[:, :, 1:])
        return torch.tanh(raw) * self.action_scale * valid[..., None].float()

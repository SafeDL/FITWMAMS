"""Causal, map-free dynamic relation encoder used by FIRM-WM."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class CausalRelationEncoder(nn.Module):
    """Encode traffic states with only current-state lane-frame relations.

    highD's cached straight-line map is deliberately not an input here.  Lane
    relations are deterministic functions of the current vehicle states and
    the ego-centred lane frame, so no generated future or map token can leak
    into the encoder.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        dropout: float = 0.1,
        lane_width_m: float = 3.6,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.lane_width_m = float(lane_width_m)
        self.state_mlp = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temporal = nn.GRUCell(hidden_dim, hidden_dim)
        self.relation_mlp = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scene = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )

    @staticmethod
    def _state_features(
        states: torch.Tensor, valid: torch.Tensor, ego_mask: torch.Tensor
    ) -> torch.Tensor:
        x, y, vx, vy, ax, ay = (states[..., index] for index in range(6))
        heading = torch.atan2(vy, vx.abs().clamp_min(1.0e-4))
        feature = torch.stack(
            (
                x / 100.0,
                y / 12.0,
                vx / 50.0,
                vy / 10.0,
                ax / 8.0,
                ay / 4.0,
                torch.sin(heading),
                torch.cos(heading),
                ego_mask.float().expand_as(vx),
                valid.float(),
            ),
            dim=-1,
        )
        return feature * valid.unsqueeze(-1).float()

    def _temporal_context(
        self,
        history: torch.Tensor,
        history_valid: torch.Tensor,
        ego_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, frames, agents, _ = history.shape
        value = history.new_zeros((batch, agents, self.hidden_dim))
        for frame in range(frames):
            encoded = self.state_mlp(
                self._state_features(
                    history[:, frame], history_valid[:, frame], ego_mask
                )
            )
            proposal = self.temporal(
                encoded.reshape(batch * agents, -1), value.reshape(batch * agents, -1)
            ).reshape(batch, agents, -1)
            value = torch.where(history_valid[:, frame, :, None], proposal, value)
        return value

    def _relation_features(self, states: torch.Tensor) -> torch.Tensor:
        position, velocity = states[..., :2], states[..., 2:4]
        delta_position = position[:, None] - position[:, :, None]
        delta_velocity = velocity[:, None] - velocity[:, :, None]
        ego_y = states[:, :1, 1:2]
        lane_index = torch.round((states[..., 1:2] - ego_y) / self.lane_width_m)
        lane_delta = lane_index[:, None] - lane_index[:, :, None]
        same_lane = (lane_delta.abs() < 0.5).float()
        adjacent_lane = ((lane_delta.abs() >= 0.5) & (lane_delta.abs() < 1.5)).float()
        heading = torch.atan2(states[..., 3], states[..., 2].abs().clamp_min(1.0e-4))
        heading_delta = heading[:, None] - heading[:, :, None]
        longitudinal_gap = delta_position[..., 0].abs().clamp_min(0.5)
        closing = (
            -(delta_velocity * delta_position).sum(dim=-1) / longitudinal_gap
        ).clamp_min(0.0)
        ttc = torch.where(
            closing > 1.0e-3,
            longitudinal_gap / closing.clamp_min(1.0e-3),
            longitudinal_gap.new_full(longitudinal_gap.shape, 10.0),
        ).clamp_max(10.0)
        drac = torch.where(
            closing > 1.0e-3,
            closing.square() / (2.0 * longitudinal_gap),
            torch.zeros_like(closing),
        ).clamp_max(20.0)
        return torch.stack(
            (
                delta_position[..., 0] / 100.0,
                delta_position[..., 1] / 12.0,
                delta_velocity[..., 0] / 50.0,
                delta_velocity[..., 1] / 10.0,
                torch.sin(heading_delta),
                torch.cos(heading_delta),
                lane_delta.squeeze(-1).clamp(-3.0, 3.0) / 3.0,
                same_lane.squeeze(-1),
                adjacent_lane.squeeze(-1),
                closing / 20.0,
                ttc / 10.0,
                drac / 20.0,
            ),
            dim=-1,
        )

    def forward(
        self,
        history: torch.Tensor,
        history_valid: torch.Tensor,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        ego_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-agent and scene contexts without a map input."""
        temporal = self._temporal_context(history, history_valid, ego_mask)
        current_token = self.state_mlp(
            self._state_features(current, current_valid, ego_mask)
        )
        relation = self.relation_mlp(self._relation_features(current))
        query = self.query(temporal + current_token).unsqueeze(2)
        key = self.key(temporal + current_token).unsqueeze(1)
        value = self.value(temporal + current_token).unsqueeze(1)
        agents = current.shape[1]
        diagonal = torch.eye(agents, device=current.device, dtype=torch.bool)[None]
        relation_valid = (
            current_valid[:, :, None] & current_valid[:, None, :] & ~diagonal
        )
        score = (query * (key + relation)).sum(-1) / (self.hidden_dim**0.5)
        score = score.masked_fill(~relation_valid, -1.0e4)
        weights = functional.softmax(score, dim=-1) * relation_valid.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1.0e-6)
        relation_context = (weights.unsqueeze(-1) * (value + relation)).sum(2)
        tokens = self.output(temporal + current_token + relation_context)
        tokens = tokens * current_valid.unsqueeze(-1).float()
        denominator = current_valid.float().sum(1, keepdim=True).clamp_min(1.0)
        scene = self.scene((tokens * current_valid.unsqueeze(-1).float()).sum(1) / denominator)
        return tokens, scene

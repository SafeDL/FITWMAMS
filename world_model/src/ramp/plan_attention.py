"""Relation-conditioned coordination in the candidate plan space."""

from __future__ import annotations

import torch
import torch.nn as nn


class PlanRelationAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # dx/dy, dvx/dvy, gap, closing, TTC, DRAC, same/adjacent lane,
        # and each participant's observed planar position.
        self.state = nn.Linear(14, hidden_dim)
        self.relation_bias = nn.Sequential(
            nn.Linear(14, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.jerk = nn.Linear(2, hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )
        self.output = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, states: torch.Tensor, jerk: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Coordinate `[B,M,C,6,2]` jerk values using only current relations."""
        b, m, controls, agents, _ = jerk.shape
        position, velocity = states[:, 1:, :2], states[:, 1:, 2:4]
        delta_pos = position[:, :, None] - position[:, None, :]
        delta_vel = velocity[:, :, None] - velocity[:, None, :]
        gap = torch.linalg.vector_norm(delta_pos, dim=-1, keepdim=True)
        closing = (
            -(delta_vel * delta_pos).sum(dim=-1, keepdim=True) / gap.clamp_min(1.0)
        ).clamp_min(0.0)
        ttc = torch.where(
            closing > 1.0e-3,
            gap / closing.clamp_min(1.0e-3),
            gap.new_full(gap.shape, 10.0),
        ).clamp_max(10.0)
        drac = torch.where(
            closing > 1.0e-3,
            closing.square() / (2.0 * gap.clamp_min(1.0e-3)),
            torch.zeros_like(closing),
        )
        lateral = delta_pos[..., 1:2].abs()
        same_lane = (lateral < 1.8).float()
        adjacent_lane = ((lateral >= 1.8) & (lateral < 5.4)).float()
        own = states[:, 1:, None, :2].expand(-1, -1, states.shape[1] - 1, -1)
        other = states[:, None, 1:, :2].expand(-1, states.shape[1] - 1, -1, -1)
        relation = torch.cat(
            (
                delta_pos,
                delta_vel,
                gap,
                closing,
                ttc,
                drac,
                same_lane,
                adjacent_lane,
                own,
                other,
            ),
            dim=-1,
        )
        relation_summary = relation.mean(dim=2)  # [B,6,14]
        token = self.state(relation_summary)[:, None, None] + self.jerk(jerk)
        token = token.reshape(b * m * controls, agents, -1)
        key_padding = (
            (~valid[:, 1:])
            .bool()[:, None, None]
            .expand(b, m, controls, agents)
            .reshape(b * m * controls, agents)
        )
        # MultiheadAttention produces NaNs for an all-masked row.  Padded
        # scenes are allowed by the fixed six-slot protocol, so keep one
        # harmless zero token visible; its output is masked below.
        key_padding = torch.where(
            key_padding.all(dim=1, keepdim=True),
            torch.zeros_like(key_padding),
            key_padding,
        )
        pair_bias = (
            self.relation_bias(relation)
            .squeeze(-1)[:, None, None, None]
            .expand(b, m, controls, 4, agents, agents)
            .reshape(b * m * controls * 4, agents, agents)
        )
        coordinated, _ = self.attention(
            token,
            token,
            token,
            key_padding_mask=key_padding,
            attn_mask=pair_bias,
            need_weights=False,
        )
        delta = self.output(coordinated).reshape(b, m, controls, agents, 2)
        return jerk + delta * valid[:, None, None, 1:, None].float()

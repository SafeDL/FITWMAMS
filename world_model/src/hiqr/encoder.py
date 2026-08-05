"""Unified START/ROLL relational query encoder for HiQR-WM."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .config import HiQRWorldModelConfig


def _safe_padding_mask(valid: torch.Tensor) -> torch.Tensor:
    """Return a Transformer padding mask without all-masked rows."""
    mask = ~valid.bool()
    empty = mask.all(dim=-1)
    if empty.any():
        mask = mask.clone()
        mask[empty, 0] = False
    return mask


def _heading(vx: torch.Tensor, vy: torch.Tensor) -> torch.Tensor:
    """Return a signed velocity heading while keeping zero speed differentiable."""
    safe_vx = torch.where(
        vx.abs() < 1.0e-4,
        torch.where(
            vx < 0.0, -torch.full_like(vx, 1.0e-4), torch.full_like(vx, 1.0e-4)
        ),
        vx,
    )
    return torch.atan2(vy, safe_vx)


class _RelationAttention(nn.Module):
    """Agent attention with relation features used only as attention bias."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = int(heads)
        self.bias = nn.Sequential(
            nn.Linear(9, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, heads)
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self, tokens: torch.Tensor, relation: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        batch, agents = tokens.shape[:2]
        bias = (
            self.bias(relation)
            .permute(0, 3, 1, 2)
            .reshape(batch * self.heads, agents, agents)
        )
        key_padding = _safe_padding_mask(valid)
        key_padding_bias = torch.zeros(
            key_padding.shape, dtype=tokens.dtype, device=tokens.device
        ).masked_fill(key_padding, float("-inf"))
        attended, _ = self.attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=key_padding_bias,
            attn_mask=bias,
            need_weights=False,
        )
        value = self.norm1(tokens + attended)
        value = self.norm2(value + self.ff(value))
        return value * valid[..., None].float()


class UnifiedRelationalQueryEncoder(nn.Module):
    """One relation-query space for Flow STARTs and realized ROLL histories.

    ``mode`` changes only the temporal source token and a mode embedding.  All
    state, relation and lane computations are shared; there is no ``encode_start``
    network or a fabricated 25-frame START history.
    """

    relation_feature_dim = 9
    lane_feature_dim = 6

    def __init__(self, cfg: HiQRWorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = int(cfg.hidden_dim)
        self.state_mlp = nn.Sequential(nn.Linear(10, h), nn.SiLU(), nn.Linear(h, h))
        self.lane_mlp = nn.Sequential(
            nn.Linear(self.lane_feature_dim, h), nn.SiLU(), nn.Linear(h, h)
        )
        self.temporal_position = nn.Parameter(torch.zeros(25, h))
        self.start_token = nn.Parameter(torch.zeros(1, 1, h))
        self.mode_embedding = nn.Embedding(2, h)
        layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=int(cfg.num_heads),
            dim_feedforward=h * 2,
            dropout=float(cfg.dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal_layers = nn.TransformerEncoder(
            layer, num_layers=max(1, int(cfg.temporal_layers))
        )
        self.relation_layers = nn.ModuleList(
            [
                _RelationAttention(h, int(cfg.num_heads), float(cfg.dropout))
                for _ in range(max(1, int(cfg.attention_layers)))
            ]
        )
        self.merge_norm = nn.LayerNorm(h)
        self.scene = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.LayerNorm(h))
        nn.init.normal_(self.start_token, std=0.02)
        nn.init.normal_(self.temporal_position, std=0.02)

    @staticmethod
    def _state_features(
        states: torch.Tensor, valid: torch.Tensor, ego_mask: torch.Tensor
    ) -> torch.Tensor:
        x, y, vx, vy, ax, ay = (states[..., index] for index in range(6))
        heading = _heading(vx, vy)
        features = torch.stack(
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
        return features * valid[..., None].float()

    def _temporal_tokens(
        self,
        history: torch.Tensor | None,
        history_valid: torch.Tensor | None,
        current_valid: torch.Tensor,
        ego_mask: torch.Tensor,
        mode: Literal["start", "roll"],
    ) -> torch.Tensor:
        batch, agents = current_valid.shape
        mode_index = 0 if mode == "start" else 1
        if mode == "start":
            values = self.start_token.expand(batch * agents, 1, -1)
            valid = current_valid.reshape(batch * agents, 1)
        else:
            if history is None or history_valid is None or history.ndim != 4:
                raise ValueError(
                    "ROLL encoding requires actual history [batch, frames, agents, 6]"
                )
            frames = int(history.shape[1])
            if not 1 <= frames <= 25:
                raise ValueError("HiQR-WM accepts one to 25 realized history frames")
            features = self._state_features(history, history_valid, ego_mask[:, None])
            values = (
                self.state_mlp(features)
                .permute(0, 2, 1, 3)
                .reshape(batch * agents, frames, -1)
            )
            valid = history_valid.permute(0, 2, 1).reshape(batch * agents, frames)
        values = (
            values
            + self.temporal_position[None, : values.shape[1]]
            + self.mode_embedding.weight[mode_index]
        )
        encoded = self.temporal_layers(
            values, src_key_padding_mask=_safe_padding_mask(valid)
        )
        last = valid.long().sum(dim=1).sub(1).clamp_min(0)
        token = encoded.gather(
            1, last[:, None, None].expand(-1, 1, encoded.shape[-1])
        ).squeeze(1)
        return token.reshape(batch, agents, -1) * current_valid[..., None].float()

    def _compact_lane_context(
        self,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        map_polylines: torch.Tensor,
        map_polyline_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Associate each vehicle with its nearest valid centreline point.

        Only local geometry is consumed.  Lane graph edges deliberately never
        enter this method, so highD defaults avoid map message passing.
        """
        batch, agents = current.shape[:2]
        if (
            map_polylines.ndim != 4
            or map_polylines.shape[0] != batch
            or map_polylines.shape[-1] != 6
        ):
            raise ValueError("map_polylines must be [batch, polylines, points, 6]")
        if map_polyline_valid.shape != map_polylines.shape[:3]:
            raise ValueError("map_polyline_valid must match map_polylines[:3]")
        maps, points = map_polylines.shape[1:3]
        if maps == 0 or points == 0:
            empty = current.new_zeros((batch, agents, self.lane_feature_dim))
            return empty, torch.zeros(
                (batch, agents), dtype=torch.bool, device=current.device
            )
        geometry = map_polylines[..., :2].reshape(batch, maps * points, 2)
        point_valid = map_polyline_valid.reshape(batch, maps * points).bool()
        delta = current[:, :, None, :2] - geometry[:, None]
        squared = (
            delta.square().sum(dim=-1).masked_fill(~point_valid[:, None], float("inf"))
        )
        distance, index = squared.min(dim=-1)
        found = torch.isfinite(distance) & current_valid
        gather = index[..., None, None].expand(-1, -1, 1, 6)
        source = map_polylines.reshape(batch, maps * points, 6)[:, None].expand(
            -1, agents, -1, -1
        )
        nearest = source.gather(2, gather).squeeze(2)
        tangent = nearest[..., 2:4]
        tangent = tangent / torch.linalg.vector_norm(
            tangent, dim=-1, keepdim=True
        ).clamp_min(1.0e-4)
        lateral_normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)
        local_delta = current[..., :2] - nearest[..., :2]
        width = nearest[..., 4].abs().clamp_min(1.0)
        lateral = (local_delta * lateral_normal).sum(dim=-1) / width
        heading = _heading(current[..., 2], current[..., 3])
        lane_heading = torch.atan2(tangent[..., 1], tangent[..., 0])
        difference = heading - lane_heading
        features = torch.stack(
            (
                lateral.clamp(-4.0, 4.0) / 4.0,
                torch.sin(difference),
                torch.cos(difference),
                (width / 6.0).clamp_max(2.0),
                distance.sqrt().clamp_max(100.0) / 100.0,
                found.float(),
            ),
            dim=-1,
        )
        return features * found[..., None].float(), found

    def _relation_features(
        self, states: torch.Tensor, lane: torch.Tensor, lane_valid: torch.Tensor
    ) -> torch.Tensor:
        position, velocity = states[..., :2], states[..., 2:4]
        delta_position = position[:, None] - position[:, :, None]
        delta_velocity = velocity[:, None] - velocity[:, :, None]
        heading = _heading(states[..., 2], states[..., 3])
        heading_delta = heading[:, None] - heading[:, :, None]
        lane_pair_valid = lane_valid[:, None, :] & lane_valid[:, :, None]
        lane_delta = (
            lane[:, None, :, 0] - lane[:, :, None, 0]
        ) * lane_pair_valid.float()
        return torch.stack(
            (
                delta_position[..., 0] / 100.0,
                delta_position[..., 1] / 12.0,
                delta_velocity[..., 0] / 50.0,
                delta_velocity[..., 1] / 10.0,
                torch.sin(heading_delta),
                torch.cos(heading_delta),
                lane_delta.clamp(-2.0, 2.0) / 2.0,
                ((lane_delta.abs() < 0.25) & lane_pair_valid).float(),
                ((lane_delta.abs() < 1.25) & lane_pair_valid).float(),
            ),
            dim=-1,
        )

    def forward(
        self,
        history: torch.Tensor | None,
        history_valid: torch.Tensor | None,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        ego_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_polyline_valid: torch.Tensor,
        *,
        mode: Literal["start", "roll"],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            current.ndim != 3
            or current.shape[-1] != 6
            or current_valid.shape != current.shape[:2]
        ):
            raise ValueError(
                "current/current_valid must be [batch, agents, 6]/[batch, agents]"
            )
        temporal = self._temporal_tokens(
            history, history_valid, current_valid, ego_mask, mode
        )
        lane, lane_valid = self._compact_lane_context(
            current, current_valid, map_polylines, map_polyline_valid
        )
        base = self.state_mlp(self._state_features(current, current_valid, ego_mask))
        tokens = (
            self.merge_norm(base + temporal + self.lane_mlp(lane))
            * current_valid[..., None].float()
        )
        relation = self._relation_features(current, lane, lane_valid)
        for layer in self.relation_layers:
            tokens = layer(tokens, relation, current_valid)
        denominator = current_valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        scene = self.scene(
            (tokens * current_valid[..., None].float()).sum(dim=1) / denominator
        )
        return tokens, scene, lane, lane_valid

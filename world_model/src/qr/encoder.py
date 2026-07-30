"""Lightweight relation-aware multi-head scene encoder for QR-WM."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import QRWorldModelConfig


def _safe_padding(valid: torch.Tensor) -> torch.Tensor:
    padding = ~valid.bool()
    empty = padding.all(dim=1)
    if empty.any():
        padding = padding.clone()
        padding[empty, 0] = False
    return padding


class _RelationAwareAgentBlock(nn.Module):
    """A small self-attention Transformer block with learned pairwise bias."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.relation_bias = nn.Linear(12, self.num_heads, bias=False)
        self.attention = nn.MultiheadAttention(hidden_dim, self.num_heads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, relation: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, agents, _ = tokens.shape
        bias = self.relation_bias(relation).permute(0, 3, 1, 2).reshape(batch * self.num_heads, agents, agents)
        attended, _ = self.attention(
            tokens, tokens, tokens, key_padding_mask=_safe_padding(valid), attn_mask=bias, need_weights=False
        )
        tokens = self.attention_norm(tokens + self.dropout(attended))
        tokens = self.feed_forward_norm(tokens + self.dropout(self.feed_forward(tokens)))
        return tokens * valid[..., None].float()


class QueryRelationalSceneEncoder(nn.Module):
    """Temporal, relation-aware multi-head, and map cross-attentive scene encoding."""

    def __init__(self, cfg: QRWorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = int(cfg.hidden_dim)
        if h % int(cfg.num_heads):
            raise ValueError("hidden_dim must be divisible by num_heads for QR-WM attention")
        self.state_mlp = nn.Sequential(nn.Linear(10, h), nn.SiLU(), nn.Linear(h, h))
        self.current_mlp = nn.Sequential(nn.Linear(10, h), nn.SiLU(), nn.Linear(h, h))
        self.temporal_position = nn.Parameter(torch.randn(25, h) * 0.02)
        self.temporal_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=h, nhead=int(cfg.num_heads), dim_feedforward=h * 2, dropout=cfg.dropout,
                    activation="gelu", batch_first=True, norm_first=True,
                )
                for _ in range(max(1, int(cfg.temporal_layers)))
            ]
        )
        self.relation_mlp = nn.Sequential(nn.Linear(12, h), nn.SiLU(), nn.Linear(h, h))
        self.relation_blocks = nn.ModuleList(
            [_RelationAwareAgentBlock(h, int(cfg.num_heads), cfg.dropout) for _ in range(max(1, int(cfg.attention_layers)))]
        )
        self.map_point_mlp = nn.Sequential(nn.Linear(6, h), nn.SiLU(), nn.Linear(h, h))
        self.map_topology_mlp = nn.Sequential(nn.Linear(3, h), nn.SiLU(), nn.Linear(h, h))
        self.cross_attention = nn.MultiheadAttention(h, int(cfg.num_heads), dropout=cfg.dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(h)
        self.scene = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.LayerNorm(h))

    @staticmethod
    def _state_features(states: torch.Tensor, valid: torch.Tensor, ego_mask: torch.Tensor) -> torch.Tensor:
        x, y, vx, vy, ax, ay = (states[..., index] for index in range(6))
        heading = torch.atan2(vy, vx.abs().clamp_min(1.0e-4))
        values = torch.stack(
            (x / 100.0, y / 12.0, vx / 50.0, vy / 10.0, ax / 8.0, ay / 4.0,
             torch.sin(heading), torch.cos(heading), ego_mask.float().expand_as(vx), valid.float()), dim=-1
        )
        return values * valid[..., None].float()

    def _temporal_tokens(self, history: torch.Tensor, history_valid: torch.Tensor, ego_mask: torch.Tensor) -> torch.Tensor:
        batch, frames, agents, _ = history.shape
        if frames > self.temporal_position.shape[0]:
            raise ValueError("QR-WM accepts at most 25 history frames")
        features = self._state_features(history, history_valid, ego_mask[:, None])
        encoded = self.state_mlp(features).permute(0, 2, 1, 3).reshape(batch * agents, frames, -1)
        encoded = encoded + self.temporal_position[None, :frames]
        valid = history_valid.permute(0, 2, 1).reshape(batch * agents, frames)
        for block in self.temporal_blocks:
            encoded = block(encoded, src_key_padding_mask=_safe_padding(valid))
        last = valid.long().sum(dim=1).sub(1).clamp_min(0)
        token = encoded.gather(1, last[:, None, None].expand(-1, 1, encoded.shape[-1])).squeeze(1)
        return token.reshape(batch, agents, -1) * history_valid.any(dim=1)[..., None].float()

    def _relation_features(self, states: torch.Tensor) -> torch.Tensor:
        position, velocity = states[..., :2], states[..., 2:4]
        delta_position = position[:, None] - position[:, :, None]
        delta_velocity = velocity[:, None] - velocity[:, :, None]
        lane = torch.round((states[..., 1] - states[:, :1, 1]) / self.cfg.lane_width_m)
        lane_delta = lane[:, None] - lane[:, :, None]
        heading = torch.atan2(states[..., 3], states[..., 2].abs().clamp_min(1.0e-4))
        heading_delta = heading[:, None] - heading[:, :, None]
        gap = delta_position[..., 0].abs().clamp_min(0.5)
        closing = (-(delta_velocity * delta_position).sum(dim=-1) / gap).clamp_min(0.0)
        ttc = torch.where(closing > 1.0e-3, gap / closing.clamp_min(1.0e-3), gap.new_full(gap.shape, 10.0)).clamp_max(10.0)
        drac = torch.where(closing > 1.0e-3, closing.square() / (2.0 * gap), torch.zeros_like(gap)).clamp_max(20.0)
        return torch.stack(
            (delta_position[..., 0] / 100.0, delta_position[..., 1] / 12.0,
             delta_velocity[..., 0] / 50.0, delta_velocity[..., 1] / 10.0,
             torch.sin(heading_delta), torch.cos(heading_delta), lane_delta.clamp(-3.0, 3.0) / 3.0,
             (lane_delta.abs() < 0.5).float(), (lane_delta.abs() < 1.5).float(), closing / 20.0,
             ttc / 10.0, drac / 20.0), dim=-1
        )

    def _map_tokens(
        self, map_polylines: torch.Tensor, map_polyline_valid: torch.Tensor, lane_graph_edges: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        point_valid = map_polyline_valid.bool()
        weight = point_valid[..., None].float()
        token = (self.map_point_mlp(map_polylines) * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1.0)
        valid = point_valid.any(dim=-1)
        if lane_graph_edges is not None and lane_graph_edges.numel():
            batch, maps = valid.shape
            source, destination, kind = lane_graph_edges.long().unbind(dim=-1)
            edge_valid = (source >= 0) & (destination >= 0) & (source < maps) & (destination < maps) & (kind >= 0)
            safe_source, safe_destination = source.clamp(0, maps - 1), destination.clamp(0, maps - 1)
            source_token = token.gather(1, safe_source[..., None].expand(-1, -1, token.shape[-1]))
            destination_token = token.gather(1, safe_destination[..., None].expand(-1, -1, token.shape[-1]))
            edge_feature = torch.stack((kind.float().clamp(0.0, 4.0) / 4.0, edge_valid.float(), (source != destination).float()), dim=-1)
            update = self.map_topology_mlp(edge_feature) + 0.5 * (source_token + destination_token)
            aggregate, count = token.new_zeros(token.shape), token.new_zeros((*token.shape[:2], 1))
            aggregate.scatter_add_(1, safe_source[..., None].expand_as(update), update * edge_valid[..., None].float())
            aggregate.scatter_add_(1, safe_destination[..., None].expand_as(update), update * edge_valid[..., None].float())
            count.scatter_add_(1, safe_source[..., None], edge_valid[..., None].float())
            count.scatter_add_(1, safe_destination[..., None], edge_valid[..., None].float())
            token = token + aggregate / count.clamp_min(1.0)
        return token * valid[..., None].float(), valid

    def forward(
        self,
        history: torch.Tensor,
        history_valid: torch.Tensor,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        ego_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_polyline_valid: torch.Tensor,
        lane_graph_edges: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        temporal = self._temporal_tokens(history, history_valid, ego_mask)
        return self._encode_current(current, current_valid, ego_mask, map_polylines, map_polyline_valid, lane_graph_edges, temporal)

    def encode_start(
        self,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        ego_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_polyline_valid: torch.Tensor,
        lane_graph_edges: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a Flow START from C0 and map tokens without synthetic history."""
        return self._encode_current(
            current, current_valid, ego_mask, map_polylines, map_polyline_valid, lane_graph_edges,
            torch.zeros_like(self.current_mlp(self._state_features(current, current_valid, ego_mask))),
        )

    def _encode_current(
        self,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        ego_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_polyline_valid: torch.Tensor,
        lane_graph_edges: torch.Tensor | None,
        temporal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base = (temporal + self.current_mlp(self._state_features(current, current_valid, ego_mask))) * current_valid[..., None].float()
        relation = self._relation_features(current)
        tokens = base
        for block in self.relation_blocks:
            tokens = block(tokens, relation, current_valid)
        # Keep a learned relation value path in addition to relation-aware attention bias.
        tokens = tokens + self.relation_mlp(relation).mean(dim=2) * current_valid[..., None].float()
        map_tokens, map_valid = self._map_tokens(map_polylines, map_polyline_valid, lane_graph_edges)
        context = torch.cat((tokens, map_tokens), dim=1)
        context_valid = torch.cat((current_valid, map_valid), dim=1)
        queried, _ = self.cross_attention(tokens, context, context, key_padding_mask=_safe_padding(context_valid), need_weights=False)
        tokens = self.cross_norm(tokens + queried) * current_valid[..., None].float()
        denominator = current_valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        scene = self.scene((tokens * current_valid[..., None].float()).sum(dim=1) / denominator)
        return tokens, scene, map_tokens, map_valid

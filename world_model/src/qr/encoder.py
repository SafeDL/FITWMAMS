"""Query-centric relational scene encoder used exclusively by QR-WM."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import QRWorldModelConfig


def _masked_softmax(scores: torch.Tensor, valid: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically safe masked softmax, including scenes without map tokens."""
    masked = scores.masked_fill(~valid, -1.0e4)
    weights = functional.softmax(masked, dim=dim) * valid.float()
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1.0e-6)


class QueryRelationalSceneEncoder(nn.Module):
    """Agent-query, temporal, relational, and map-topology scene encoder.

    Agent histories are encoded temporally.  Current agent tokens then attend
    over relation-aware agent context and learned map-polyline context through
    an explicit independent query for each agent.  No traffic signal feature is
    accepted or synthesized by this module.
    """

    def __init__(self, cfg: QRWorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = int(cfg.hidden_dim)
        self.state_mlp = nn.Sequential(nn.Linear(10, h), nn.SiLU(), nn.Linear(h, h))
        self.temporal = nn.GRU(
            h, h, num_layers=max(1, int(cfg.temporal_layers)), batch_first=True
        )
        self.current_mlp = nn.Sequential(nn.Linear(10, h), nn.SiLU(), nn.Linear(h, h))
        self.relation_mlp = nn.Sequential(nn.Linear(12, h), nn.SiLU(), nn.Linear(h, h))
        self.map_point_mlp = nn.Sequential(nn.Linear(6, h), nn.SiLU(), nn.Linear(h, h))
        self.map_topology_mlp = nn.Sequential(nn.Linear(3, h), nn.SiLU(), nn.Linear(h, h))
        self.agent_query = nn.Linear(h, h, bias=False)
        self.context_key = nn.Linear(h, h, bias=False)
        self.context_value = nn.Linear(h, h, bias=False)
        self.self_query = nn.Linear(h, h, bias=False)
        self.self_key = nn.Linear(h, h, bias=False)
        self.self_value = nn.Linear(h, h, bias=False)
        self.self_output = nn.Sequential(nn.LayerNorm(h), nn.SiLU(), nn.Linear(h, h))
        self.cross_output = nn.Sequential(nn.LayerNorm(h), nn.SiLU(), nn.Dropout(cfg.dropout), nn.Linear(h, h))
        self.scene = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.LayerNorm(h))

    @staticmethod
    def _state_features(
        states: torch.Tensor, valid: torch.Tensor, ego_mask: torch.Tensor
    ) -> torch.Tensor:
        x, y, vx, vy, ax, ay = (states[..., index] for index in range(6))
        heading = torch.atan2(vy, vx.abs().clamp_min(1.0e-4))
        values = torch.stack(
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
        return values * valid[..., None].float()

    def _temporal_tokens(
        self, history: torch.Tensor, history_valid: torch.Tensor, ego_mask: torch.Tensor
    ) -> torch.Tensor:
        batch, frames, agents, _ = history.shape
        features = self._state_features(history, history_valid, ego_mask[:, None])
        encoded = self.state_mlp(features).permute(0, 2, 1, 3).reshape(batch * agents, frames, -1)
        values, _ = self.temporal(encoded)
        token = values[:, -1].reshape(batch, agents, -1)
        last_valid = history_valid.any(dim=1)
        return token * last_valid[..., None].float()

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
            (
                delta_position[..., 0] / 100.0,
                delta_position[..., 1] / 12.0,
                delta_velocity[..., 0] / 50.0,
                delta_velocity[..., 1] / 10.0,
                torch.sin(heading_delta),
                torch.cos(heading_delta),
                lane_delta.clamp(-3.0, 3.0) / 3.0,
                (lane_delta.abs() < 0.5).float(),
                (lane_delta.abs() < 1.5).float(),
                closing / 20.0,
                ttc / 10.0,
                drac / 20.0,
            ),
            dim=-1,
        )

    def _map_tokens(
        self,
        map_polylines: torch.Tensor,
        map_polyline_valid: torch.Tensor,
        lane_graph_edges: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        point_valid = map_polyline_valid.bool()
        weight = point_valid[..., None].float()
        points = self.map_point_mlp(map_polylines)
        token = (points * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1.0)
        valid = point_valid.any(dim=-1)
        if lane_graph_edges is not None and lane_graph_edges.numel():
            batch, maps = valid.shape
            edge = lane_graph_edges.long()
            source, destination, kind = edge.unbind(dim=-1)
            edge_valid = (source >= 0) & (destination >= 0) & (source < maps) & (destination < maps) & (kind >= 0)
            safe_source, safe_destination = source.clamp(0, maps - 1), destination.clamp(0, maps - 1)
            source_token = token.gather(1, safe_source[..., None].expand(-1, -1, token.shape[-1]))
            destination_token = token.gather(1, safe_destination[..., None].expand(-1, -1, token.shape[-1]))
            edge_feature = torch.stack(
                (
                    kind.float().clamp(0.0, 4.0) / 4.0,
                    edge_valid.float(),
                    (source != destination).float(),
                ),
                dim=-1,
            )
            update = self.map_topology_mlp(edge_feature) + 0.5 * (source_token + destination_token)
            aggregate = token.new_zeros(token.shape)
            count = token.new_zeros((*token.shape[:2], 1))
            aggregate.scatter_add_(1, safe_source[..., None].expand_as(update), update * edge_valid[..., None].float())
            aggregate.scatter_add_(1, safe_destination[..., None].expand_as(update), update * edge_valid[..., None].float())
            edge_count = edge_valid[..., None].float()
            count.scatter_add_(1, safe_source[..., None], edge_count)
            count.scatter_add_(1, safe_destination[..., None], edge_count)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temporal = self._temporal_tokens(history, history_valid, ego_mask)
        current_token = self.current_mlp(self._state_features(current, current_valid, ego_mask))
        base = (temporal + current_token) * current_valid[..., None].float()
        relation = self.relation_mlp(self._relation_features(current))
        agents = current.shape[1]
        diagonal = torch.eye(agents, dtype=torch.bool, device=current.device)[None]
        pair_valid = current_valid[:, :, None] & current_valid[:, None, :] & ~diagonal
        self_scores = (self.self_query(base).unsqueeze(2) * (self.self_key(base).unsqueeze(1) + relation)).sum(-1)
        self_scores = self_scores / math.sqrt(float(base.shape[-1]))
        self_weights = _masked_softmax(self_scores, pair_valid, dim=-1)
        relational = (self_weights[..., None] * (self.self_value(base).unsqueeze(1) + relation)).sum(dim=2)
        agent_context = self.self_output(base + relational) * current_valid[..., None].float()

        map_tokens, map_valid = self._map_tokens(map_polylines, map_polyline_valid, lane_graph_edges)
        context = torch.cat((agent_context, map_tokens), dim=1)
        context_valid = torch.cat((current_valid, map_valid), dim=1)
        # q_i = W_q h_i and z_i = Attention(q_i, C, C): each participant has
        # a separately computed query against both dynamic and map context.
        scores = (self.agent_query(agent_context).unsqueeze(2) * self.context_key(context).unsqueeze(1)).sum(-1)
        scores = scores / math.sqrt(float(agent_context.shape[-1]))
        cross_valid = current_valid[:, :, None] & context_valid[:, None, :]
        cross_weights = _masked_softmax(scores, cross_valid, dim=-1)
        queried = (cross_weights[..., None] * self.context_value(context).unsqueeze(1)).sum(dim=2)
        tokens = self.cross_output(agent_context + queried) * current_valid[..., None].float()
        denominator = current_valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        scene = self.scene((tokens * current_valid[..., None].float()).sum(dim=1) / denominator)
        return tokens, scene

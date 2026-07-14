"""Lane-topology-gated pairwise-relative heterogeneous attention encoder."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RelationalEncoderConfig:
    hidden_dim: int = 128
    temporal_layers: int = 1
    dropout: float = 0.1
    lane_width_m: float = 3.6
    use_conflict_zones: bool = False
    include_ego_relative_position: bool = False


class RelationalTrafficEncoder(nn.Module):
    """Permutation-equivariant agent encoder with cached-map compatible input.

    Participant attention has no absolute participant index embedding.  It is
    therefore invariant to padded ordering and uses only physical/road
    relations. ``ego_mask`` identifies the externally supplied physical ego,
    not an ADS implementation or identity.
    """

    def __init__(self, cfg: RelationalEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = int(cfg.hidden_dim)
        # Position enters the default graph through AA/AL relative edges.  The
        # optional ego-relative pair feature lets the temporal encoder retain
        # its evolution across the history too, without introducing a global
        # recording coordinate or a participant-slot identity.
        state_dim = 12 if cfg.include_ego_relative_position else 10
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, h), nn.SiLU(), nn.Linear(h, h))
        self.temporal = nn.GRU(h, h, num_layers=max(1, int(cfg.temporal_layers)), batch_first=True)
        self.map_mlp = nn.Sequential(nn.Linear(6, h), nn.SiLU(), nn.Linear(h, h))
        self.map_q = nn.Linear(h, h, bias=False)
        self.map_k = nn.Linear(h, h, bias=False)
        self.map_v = nn.Linear(h, h, bias=False)
        self.al_edge = nn.Sequential(nn.Linear(4, h), nn.SiLU(), nn.Linear(h, h))
        if cfg.use_conflict_zones:
            self.conflict_mlp = nn.Sequential(nn.Linear(4, h), nn.SiLU(), nn.Linear(h, h))
            self.conflict_q = nn.Linear(h, h, bias=False)
            self.conflict_k = nn.Linear(h, h, bias=False)
            self.conflict_v = nn.Linear(h, h, bias=False)
            self.ac_edge = nn.Sequential(nn.Linear(4, h), nn.SiLU(), nn.Linear(h, h))
        self.q = nn.Linear(h, h, bias=False)
        self.k = nn.Linear(h, h, bias=False)
        self.v = nn.Linear(h, h, bias=False)
        self.aa_edge = nn.Sequential(nn.Linear(8, h), nn.SiLU(), nn.Linear(h, h))
        self.output = nn.Sequential(nn.LayerNorm(h), nn.SiLU(), nn.Dropout(cfg.dropout), nn.Linear(h, h))
        self.scene = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.LayerNorm(h))

    @staticmethod
    def _featureize(
        states: torch.Tensor,
        valid: torch.Tensor,
        ego_mask: torch.Tensor,
        *,
        include_ego_relative_position: bool,
    ) -> torch.Tensor:
        """Create local agent features from `[x,y,vx,vy,ax,ay]`."""
        vx, vy, ax, ay = (states[..., index] for index in (2, 3, 4, 5))
        heading = torch.atan2(vy, vx.abs().clamp_min(1.0e-4))
        valid_f = valid.float()
        # Geometry/type are physical compatibility defaults; adapters can later
        # replace them with observed vehicle geometry without model changes.
        length = torch.full_like(vx, 4.8)
        width = torch.full_like(vx, 1.9)
        ego_feature = ego_mask.float().expand_as(vx)
        features = [vx, vy, ax, ay, torch.sin(heading), torch.cos(heading), length, width, ego_feature, valid_f]
        if include_ego_relative_position:
            # ``ego_mask`` carries a single externally replayed ego marker,
            # never an ADS network or identity.  The subtraction yields a
            # translation-invariant AA relation that is valid for highways,
            # roundabouts and future map adapters alike.
            ego_position = (states[..., :2] * ego_mask.float().unsqueeze(-1)).sum(dim=-2, keepdim=True)
            relative = states[..., :2] - ego_position
            features.extend((relative[..., 0], relative[..., 1]))
        return torch.stack(features, dim=-1) * valid_f.unsqueeze(-1)

    @staticmethod
    def _candidate_points(
        states: torch.Tensor,
        lane_points: torch.Tensor,
        lane_valid: torch.Tensor,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the nearest valid point of each candidate lane.

        A lane's mean ``y`` works only on a straight highway.  Selecting the
        nearest polyline point preserves curved roundabout geometry and gives
        the decoder a local tangent at the agent's actual road position.
        """
        b, n, _ = states.shape
        m, p = lane_points.shape[1:3]
        r = candidates.shape[-1]
        if m == 0 or p == 0 or r == 0:
            return (
                lane_points.new_zeros((b, n, r, lane_points.shape[-1])),
                torch.zeros((b, n, r), dtype=torch.bool, device=states.device),
            )
        safe = candidates.clamp(0, m - 1)
        lanes = lane_points[:, None].expand(b, n, m, p, lane_points.shape[-1]).gather(
            2, safe[..., None, None].expand(b, n, r, p, lane_points.shape[-1])
        )
        valid = lane_valid[:, None].expand(b, n, m, p).gather(
            2, safe[..., None].expand(b, n, r, p)
        ) & (candidates[..., None] >= 0)
        delta = lanes[..., :2] - states[:, :, None, None, :2]
        distance = delta.square().sum(dim=-1).masked_fill(~valid, float("inf"))
        nearest = distance.argmin(dim=-1)
        point = lanes.gather(3, nearest[..., None, None].expand(b, n, r, 1, lanes.shape[-1])).squeeze(3)
        point_valid = valid.gather(3, nearest[..., None]).squeeze(3)
        return point, point_valid

    @classmethod
    def _agent_lane_edges(
        cls,
        states: torch.Tensor,
        lane_points: torch.Tensor,
        lane_valid: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        points, point_valid = cls._candidate_points(states, lane_points, lane_valid, candidates)
        dx = states[:, :, None, 0] - points[..., 0]
        dy = states[:, :, None, 1] - points[..., 1]
        lane_heading = torch.atan2(points[..., 3], points[..., 2].abs().clamp_min(1.0e-4))
        own_heading = torch.atan2(states[..., 3], states[..., 2].abs().clamp_min(1.0e-4)).unsqueeze(-1)
        lateral = -dx * torch.sin(lane_heading) + dy * torch.cos(lane_heading)
        longitudinal = dx * torch.cos(lane_heading) + dy * torch.sin(lane_heading)
        return torch.stack((lateral, own_heading - lane_heading, longitudinal, point_valid.float()), dim=-1)

    @staticmethod
    def _pair_edges(
        states: torch.Tensor,
        lane_candidates: torch.Tensor,
        lane_graph_edges: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = states[..., :2]
        vel = states[..., 2:4]
        dxdy = pos[:, None, :, :] - pos[:, :, None, :]
        dvd = vel[:, None, :, :] - vel[:, :, None, :]
        heading = torch.atan2(states[..., 3], states[..., 2].abs().clamp_min(1e-4))
        dpsi = heading[:, None, :] - heading[:, :, None]
        lane = lane_candidates[..., 0]
        li, lj = lane[:, :, None], lane[:, None, :]
        same = (li == lj) & (li >= 0)
        if lane_graph_edges is None or lane_graph_edges.numel() == 0:
            adjacent = (li >= 0) & (lj >= 0) & ((li - lj).abs() == 1)
            relation = torch.where(same, torch.zeros_like(li + lj), torch.where(adjacent, torch.ones_like(li + lj), torch.full_like(li + lj, 5)))
        else:
            edge = lane_graph_edges.long()
            source, destination, kind = edge[..., 0], edge[..., 1], edge[..., 2]
            valid_edge = (source >= 0) & (destination >= 0) & (kind >= 0) & (kind <= 4)
            connected = (
                (li[..., None] == source[:, None, None, :])
                & (lj[..., None] == destination[:, None, None, :])
                & valid_edge[:, None, None, :]
            ) | (
                (li[..., None] == destination[:, None, None, :])
                & (lj[..., None] == source[:, None, None, :])
                & valid_edge[:, None, None, :]
            )
            # Relation indices match graph_schema.RELATION_TYPES:
            # successor=0 is a continuous same traffic stream, then adjacent,
            # merge, diverge, cross, and unrelated=5.  The `where` sequence
            # handles padded map edges and preserves explicit non-highway
            # topology in the learned pairwise-relative attention.
            relation = torch.full_like(li + lj, 5)
            for map_kind, relation_index in ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4)):
                relation = torch.where((connected & (kind[:, None, None, :] == map_kind)).any(dim=-1), torch.full_like(relation, relation_index), relation)
            relation = torch.where(same, torch.zeros_like(relation), relation)
        edge = torch.cat((dxdy, dvd, torch.sin(dpsi).unsqueeze(-1), torch.cos(dpsi).unsqueeze(-1),
                          relation.unsqueeze(-1).float(), torch.ones_like(relation).unsqueeze(-1).float()), dim=-1)
        topology = relation != 5
        return edge, topology

    @staticmethod
    def _agent_conflict_edges(states: torch.Tensor, conflict_zones: torch.Tensor) -> torch.Tensor:
        """Dynamic agent--conflict-zone edge features [dx, dy, clearance, priority]."""
        delta = states[:, :, None, :2] - conflict_zones[:, None, :, :2]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        clearance = distance - conflict_zones[:, None, :, 2].clamp_min(0.0)
        priority = conflict_zones[:, None, :, 3].expand_as(clearance)
        return torch.cat((delta, clearance.unsqueeze(-1), priority.unsqueeze(-1)), dim=-1)

    def forward(
        self,
        history_states: torch.Tensor,
        history_valid: torch.Tensor,
        current_states: torch.Tensor,
        current_valid: torch.Tensor,
        ego_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_polyline_valid: torch.Tensor,
        lane_candidates: torch.Tensor,
        lane_graph_edges: torch.Tensor | None = None,
        conflict_zone_features: torch.Tensor | None = None,
        conflict_zone_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return participant and pooled scene contexts.

        All tensors are padded dense batch tensors.  Dynamic membership is
        always controlled through the validity masks.
        """
        b, hist, n, _ = history_states.shape
        history_features = self._featureize(
            history_states, history_valid, ego_mask[:, None, :],
            include_ego_relative_position=self.cfg.include_ego_relative_position,
        )
        temporal_in = self.state_mlp(history_features).permute(0, 2, 1, 3).reshape(b * n, hist, -1)
        temporal_out, _ = self.temporal(temporal_in)
        temporal = temporal_out[:, -1].reshape(b, n, -1)
        current = self.state_mlp(self._featureize(
            current_states, current_valid, ego_mask,
            include_ego_relative_position=self.cfg.include_ego_relative_position,
        ))

        map_valid_f = map_polyline_valid.float().unsqueeze(-1)
        map_denom = map_valid_f.sum(dim=2).clamp_min(1.0)
        map_tokens = self.map_mlp((map_polylines * map_valid_f).sum(dim=2) / map_denom)
        m = map_tokens.shape[1]
        safe_lane = lane_candidates.clamp(0, max(m - 1, 0))
        gathered = map_tokens[:, None].expand(b, n, m, -1).gather(
            2, safe_lane[..., None].expand(-1, -1, -1, map_tokens.shape[-1])
        )
        al = self.al_edge(self._agent_lane_edges(current_states, map_polylines, map_polyline_valid, lane_candidates))
        score = (self.map_q(current).unsqueeze(2) * (self.map_k(gathered) + al)).sum(-1) / (current.shape[-1] ** 0.5)
        lane_valid = lane_candidates >= 0
        score = score.masked_fill(~lane_valid, -1.0e4)
        weights = F.softmax(score, dim=-1) * lane_valid.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        map_context = (weights.unsqueeze(-1) * (self.map_v(gathered) + al)).sum(dim=2)

        conflict_context = torch.zeros_like(map_context)
        if self.cfg.use_conflict_zones and conflict_zone_features is not None:
            zones = conflict_zone_features
            zone_valid = (
                conflict_zone_valid.bool()
                if conflict_zone_valid is not None
                else torch.isfinite(zones).all(dim=-1)
            )
            if zones.shape[1] > 0:
                conflict_tokens = self.conflict_mlp(zones)
                ac = self.ac_edge(self._agent_conflict_edges(current_states, zones))
                gathered_conflicts = conflict_tokens[:, None].expand(b, n, zones.shape[1], -1)
                score = (self.conflict_q(current).unsqueeze(2) * (self.conflict_k(gathered_conflicts) + ac)).sum(-1)
                score = score / (current.shape[-1] ** 0.5)
                # A dynamic AC edge exists only near a conflict region.
                ac_valid = current_valid[:, :, None] & zone_valid[:, None, :] & (ac[..., 2] <= 35.0)
                score = score.masked_fill(~ac_valid, -1.0e4)
                weight = F.softmax(score, dim=-1) * ac_valid.float()
                weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
                conflict_context = (weight.unsqueeze(-1) * (self.conflict_v(gathered_conflicts) + ac)).sum(dim=2)

        edges, topology = self._pair_edges(current_states, lane_candidates, lane_graph_edges)
        pair_valid = current_valid[:, :, None] & current_valid[:, None, :] & topology
        diagonal = torch.eye(n, device=current_states.device, dtype=torch.bool).unsqueeze(0)
        pair_valid = pair_valid & ~diagonal
        q = self.q(temporal + current).unsqueeze(2)
        k = self.k(temporal + current).unsqueeze(1)
        edge_context = self.aa_edge(edges)
        pair_score = (q * (k + edge_context)).sum(-1) / (current.shape[-1] ** 0.5)
        pair_score = pair_score.masked_fill(~pair_valid, -1.0e4)
        pair_weight = F.softmax(pair_score, dim=-1) * pair_valid.float()
        pair_weight = pair_weight / pair_weight.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        pair_context = (pair_weight.unsqueeze(-1) * (self.v(temporal + current).unsqueeze(1) + edge_context)).sum(dim=2)

        tokens = self.output(temporal + current + map_context + conflict_context + pair_context) * current_valid.unsqueeze(-1).float()
        denom = current_valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        scene = self.scene((tokens * current_valid.unsqueeze(-1).float()).sum(dim=1) / denom)
        return tokens, scene

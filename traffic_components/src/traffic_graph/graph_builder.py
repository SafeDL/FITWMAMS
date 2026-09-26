"""Construction of sparse, dataset-neutral dynamic traffic graphs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph_schema import (
    AC_EDGE_DIM,
    AA_EDGE_DIM,
    AL_EDGE_DIM,
    DynamicTrafficGraph,
    DynamicTrafficSequence,
    RELATION_TO_INDEX,
)


@dataclass(frozen=True)
class GraphBuilderConfig:
    lane_width_m: float = 3.6
    top_r_lanes: int = 3
    neighbour_distance_m: float = 90.0
    max_lanes: int = 8
    conflict_association_distance_m: float = 35.0


class DynamicTrafficGraphBuilder:
    """Build lane-aware sparse graphs without referring to slot names.

    The highD adapter uses straight lane polylines, while the same graph format
    accepts supplied map polylines. Agent identifiers are stable
    across a sequence but their membership is allowed to change at every step.
    """

    def __init__(self, cfg: GraphBuilderConfig | None = None) -> None:
        self.cfg = cfg or GraphBuilderConfig()

    def lane_candidates(self, states: np.ndarray, valid: np.ndarray, lane_centers: np.ndarray) -> np.ndarray:
        n = int(len(states))
        requested_r = int(self.cfg.top_r_lanes)
        available_r = min(requested_r, int(len(lane_centers)))
        out = np.full((n, requested_r), -1, dtype=np.int64)
        if available_r == 0:
            return out
        distances = np.abs(np.asarray(states, np.float32)[:, 1:2] - lane_centers.reshape(1, -1))
        order = np.argsort(distances, axis=1)[:, :available_r]
        out[np.asarray(valid, bool), :available_r] = order[np.asarray(valid, bool)]
        return out

    def lane_candidates_from_polylines(
        self,
        states: np.ndarray,
        valid: np.ndarray,
        map_polylines: np.ndarray,
        map_polyline_valid: np.ndarray,
    ) -> np.ndarray:
        """Associate agents with nearest valid polyline geometry.

        The fixed centerline-y helper remains for compatibility with
        callers. Supplied map geometry is never overwritten by this fallback.
        """
        states = np.asarray(states, np.float32)
        active = np.asarray(valid, bool)
        polylines = np.asarray(map_polylines, np.float32)
        point_valid = np.asarray(map_polyline_valid, bool)
        n, requested_r = len(states), int(self.cfg.top_r_lanes)
        out = np.full((n, requested_r), -1, dtype=np.int64)
        if not len(polylines):
            return out
        delta = states[:, None, None, :2] - polylines[None, :, :, :2]
        distance = np.sum(delta * delta, axis=-1)
        distance = np.where(point_valid[None], distance, np.inf)
        lane_distance = distance.min(axis=-1)
        available_r = min(requested_r, polylines.shape[0])
        order = np.argsort(lane_distance, axis=1)[:, :available_r]
        lane_exists = point_valid.any(axis=1)
        for index in np.flatnonzero(active):
            choices = order[index]
            out[index, :available_r] = np.where(lane_exists[choices], choices, -1)
        return out

    def _agent_edges(
        self,
        states: np.ndarray,
        valid: np.ndarray,
        lane_candidates: np.ndarray,
        lane_graph_edges: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        states = np.asarray(states, np.float32)
        valid = np.asarray(valid, bool)
        pairs: list[tuple[int, int]] = []
        features: list[np.ndarray] = []
        headings = np.arctan2(states[:, 3], np.maximum(np.abs(states[:, 2]), 1.0e-5))
        primary_lane = lane_candidates[:, 0] if lane_candidates.shape[1] else np.full(len(states), -1)
        topology = np.asarray(lane_graph_edges if lane_graph_edges is not None else np.zeros((0, 3)), np.int64).reshape(-1, 3)
        # 0=successor (same traffic stream), 1=adjacent, 2=merge,
        # 3=diverge, and 4=cross.  Keep the non-highway relations intact;
        # collapsing them to adjacent lane would discard meaningful map
        # topology before the encoder sees it.
        relation_for_kind = {0: "same_lane", 1: "adjacent_lane", 2: "merge", 3: "diverge", 4: "cross"}
        direct_topology = {
            (int(source), int(destination)): int(kind)
            for source, destination, kind in topology
            if source >= 0 and destination >= 0 and int(kind) in relation_for_kind
        }
        for i in np.flatnonzero(valid):
            for j in np.flatnonzero(valid):
                if i == j:
                    continue
                dx, dy = states[j, :2] - states[i, :2]
                if float(np.hypot(dx, dy)) > self.cfg.neighbour_distance_m:
                    continue
                lane_i, lane_j = int(primary_lane[i]), int(primary_lane[j])
                if lane_i < 0 or lane_j < 0:
                    relation = "unrelated"
                elif lane_i == lane_j:
                    relation = "same_lane"
                else:
                    kind = direct_topology.get((lane_i, lane_j), direct_topology.get((lane_j, lane_i), -1))
                    relation = relation_for_kind.get(kind, "unrelated")
                    # The geometry-only fallback preserves useful highD
                    # behaviour for callers that intentionally omit a map
                    # topology, but never overwrites an explicit map.
                    if relation == "unrelated" and not len(direct_topology) and abs(lane_i - lane_j) == 1:
                        relation = "adjacent_lane"
                # Keep only topology-relevant graph edges; unrelated agents do
                # not become dense attention neighbours.
                if relation == "unrelated":
                    continue
                dpsi = headings[j] - headings[i]
                pairs.append((int(i), int(j)))
                features.append(np.asarray([
                    dx,
                    dy,
                    states[j, 2] - states[i, 2],
                    states[j, 3] - states[i, 3],
                    np.sin(dpsi),
                    np.cos(dpsi),
                    float(RELATION_TO_INDEX[relation]),
                    1.0,
                ], dtype=np.float32))
        if not pairs:
            return np.zeros((2, 0), np.int64), np.zeros((0, AA_EDGE_DIM), np.float32)
        return np.asarray(pairs, np.int64).T, np.stack(features).astype(np.float32)

    def _lane_edges(
        self,
        states: np.ndarray,
        valid: np.ndarray,
        candidates: np.ndarray,
        lane_centers: np.ndarray,
    ) -> np.ndarray:
        n, r = candidates.shape
        out = np.zeros((n, r, AL_EDGE_DIM), dtype=np.float32)
        headings = np.arctan2(states[:, 3], np.maximum(np.abs(states[:, 2]), 1.0e-5))
        for i in range(n):
            if not valid[i]:
                continue
            for q, lane in enumerate(candidates[i]):
                if lane < 0:
                    continue
                lateral = states[i, 1] - lane_centers[int(lane)]
                confidence = float(np.exp(-0.5 * (lateral / max(self.cfg.lane_width_m, 1.0e-3)) ** 2))
                out[i, q] = (lateral, headings[i], states[i, 0], confidence)
        return out

    def _agent_conflict_edges(
        self,
        states: np.ndarray,
        valid: np.ndarray,
        conflict_zone_features: np.ndarray,
        conflict_zone_valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Associate active agents with nearby merge/crossing conflict zones."""
        zones = np.asarray(conflict_zone_features, np.float32)
        zone_valid = np.asarray(conflict_zone_valid, bool)
        states = np.asarray(states, np.float32)
        active = np.asarray(valid, bool)
        pairs: list[tuple[int, int]] = []
        features: list[np.ndarray] = []
        for agent in np.flatnonzero(active):
            for zone in np.flatnonzero(zone_valid):
                delta = states[agent, :2] - zones[zone, :2]
                distance = float(np.hypot(delta[0], delta[1]))
                clearance = distance - float(max(zones[zone, 2], 0.0))
                if clearance > float(self.cfg.conflict_association_distance_m):
                    continue
                pairs.append((int(agent), int(zone)))
                features.append(np.asarray((delta[0], delta[1], clearance, zones[zone, 3]), np.float32))
        if not pairs:
            return np.zeros((2, 0), np.int64), np.zeros((0, AC_EDGE_DIM), np.float32)
        return np.asarray(pairs, np.int64).T, np.stack(features).astype(np.float32)

    def graph_at(
        self,
        *,
        timestamp: float,
        agent_ids: np.ndarray,
        states: np.ndarray,
        valid: np.ndarray,
        ego_index: int,
        primary_agent_index: int,
        map_polylines: np.ndarray,
        map_polyline_valid: np.ndarray,
        lane_graph_edges: np.ndarray,
        conflict_zone_features: np.ndarray | None = None,
        conflict_zone_valid: np.ndarray | None = None,
    ) -> DynamicTrafficGraph:
        centers = np.asarray(map_polylines, np.float32)[:, :, 1]
        lane_centers = np.asarray([
            np.mean(row[np.asarray(map_polyline_valid[idx], bool)]) if np.any(map_polyline_valid[idx]) else 0.0
            for idx, row in enumerate(centers)
        ], dtype=np.float32)
        candidates = self.lane_candidates_from_polylines(states, valid, map_polylines, map_polyline_valid)
        aa_index, aa_features = self._agent_edges(states, valid, candidates, lane_graph_edges)
        zones = np.zeros((0, 4), np.float32) if conflict_zone_features is None else np.asarray(conflict_zone_features, np.float32)
        zone_valid = np.zeros((0,), bool) if conflict_zone_valid is None else np.asarray(conflict_zone_valid, bool)
        ac_index, ac_features = self._agent_conflict_edges(states, valid, zones, zone_valid)
        return DynamicTrafficGraph(
            timestamp=float(timestamp), agent_ids=np.asarray(agent_ids, np.int64),
            agent_states=np.asarray(states, np.float32), agent_valid=np.asarray(valid, bool),
            ego_index=int(ego_index), primary_agent_index=int(primary_agent_index),
            map_polylines=np.asarray(map_polylines, np.float32),
            map_polyline_valid=np.asarray(map_polyline_valid, bool),
            lane_graph_edges=np.asarray(lane_graph_edges, np.int64), aa_edge_index=aa_index,
            aa_edge_features=aa_features, al_lane_indices=candidates,
            al_edge_features=self._lane_edges(states, valid, candidates, lane_centers),
            conflict_zone_features=zones, conflict_zone_valid=zone_valid,
            ac_edge_index=ac_index, ac_edge_features=ac_features,
        )

    def straight_lane_map(self, states: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make a lightweight highD lane graph in road-local coordinates."""
        y = np.asarray(states, np.float32)[np.asarray(valid, bool), 1]
        center = float(np.median(y)) if len(y) else 0.0
        observed_ids = np.unique(np.round((y - center) / self.cfg.lane_width_m).astype(np.int64)) if len(y) else np.array([0])
        # A sparse scene can contain only same-lane agents, but the highD map
        # still has adjacent lanes.  Retain a local three-lane context rather
        # than collapsing the static lane graph to one occupied polyline.
        lane_ids = np.unique(np.concatenate((observed_ids, np.asarray([-1, 0, 1], dtype=np.int64))))
        lane_ids = lane_ids[: self.cfg.max_lanes]
        if not len(lane_ids):
            lane_ids = np.array([0])
        x_valid = np.asarray(states, np.float32)[np.asarray(valid, bool), 0]
        x0 = float(np.min(x_valid) - 120.0) if len(x_valid) else -120.0
        x1 = float(np.max(x_valid) + 120.0) if len(x_valid) else 120.0
        p = 8
        map_polylines = np.zeros((len(lane_ids), p, 6), dtype=np.float32)
        map_polylines[:, :, 0] = np.linspace(x0, x1, p, dtype=np.float32)
        for lane_index, lane_id in enumerate(lane_ids):
            map_polylines[lane_index, :, 1] = center + lane_id * self.cfg.lane_width_m
            map_polylines[lane_index, :, 2] = 1.0  # tangent x
            map_polylines[lane_index, :, 4] = self.cfg.lane_width_m
            map_polylines[lane_index, :, 5] = 1.0  # normal road priority
        valid_map = np.ones((len(lane_ids), p), dtype=bool)
        edges: list[tuple[int, int, int]] = []
        for lane_index in range(len(lane_ids) - 1):
            edges.extend(((lane_index, lane_index + 1, 1), (lane_index + 1, lane_index, 1)))
        return map_polylines, valid_map, np.asarray(edges, np.int64).reshape(-1, 3)

    def sequence_from_dense_slots(
        self,
        *,
        sequence_id: str,
        recording_id: str,
        ego_id: str,
        timestamps: np.ndarray,
        states: np.ndarray,
        valid: np.ndarray,
        primary_slot_index: int,
        split: str,
        is_evt_tail: bool,
    ) -> DynamicTrafficSequence:
        """Convert fixed slots into a variable-agent sequence once.

        Slot labels are intentionally discarded.  The six background identities
        only provide stable source ids for this highD migration adapter.
        """
        states = np.asarray(states, np.float32)
        valid = np.asarray(valid, bool)
        if states.ndim != 3 or states.shape[-1] != 6:
            raise ValueError("states must be [T, N, 6]")
        n = states.shape[1]
        agent_ids = np.arange(n, dtype=np.int64)
        # START samples in the frozen migration cache contain only the final
        # observed frame in their history.  Build the static lane map from the
        # first frame that actually has agents instead of an all-padding row.
        # Use all valid observations to retain lanes that are only occupied
        # later in the sequence (for example after a lane change).
        map_polylines, map_valid, lane_edges = self.straight_lane_map(
            states.reshape(-1, states.shape[-1]), valid.reshape(-1)
        )
        candidates = np.stack([
            self.lane_candidates_from_polylines(step, mask, map_polylines, map_valid)
            for step, mask in zip(states, valid)
        ])
        return DynamicTrafficSequence(
            sequence_id=str(sequence_id), recording_id=str(recording_id), ego_id=str(ego_id),
            timestamps=np.asarray(timestamps, np.float32), agent_ids=agent_ids,
            agent_states=states, agent_valid=valid, ego_index=0,
            primary_agent_index=(int(primary_slot_index) + 1 if int(primary_slot_index) >= 0 else -1),
            map_polylines=map_polylines, map_polyline_valid=map_valid,
            lane_graph_edges=lane_edges, agent_lane_candidates=candidates,
            split=str(split), is_evt_tail=bool(is_evt_tail),
        )

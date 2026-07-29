"""highD migration adapter for the semi-Markov relational model."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .graph_builder import DynamicTrafficGraphBuilder, GraphBuilderConfig
from .graph_schema import DynamicTrafficSequence


class HighDGraphAdapter:
    """Convert highD physical states to the common dynamic graph schema.

    No slot name is exposed outside this adapter.  The current highD cache has
    a bounded number of agents, represented by a validity mask.
    """

    version = "highd_straight_lane_v1"

    def __init__(self, lane_width_m: float = 3.6, top_r_lanes: int = 3) -> None:
        self.builder = DynamicTrafficGraphBuilder(GraphBuilderConfig(
            lane_width_m=float(lane_width_m), top_r_lanes=int(top_r_lanes),
        ))

    @staticmethod
    def map_from_recording_metadata(
        recording_meta: dict,
        *,
        ego_global_y_m: float,
        lateral_sign: float = 1.0,
        x_center_m: float = 0.0,
        x_span_m: float = 240.0,
        max_lanes: int = 8,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert highD recording lane markings into local straight polylines.

        This is intentionally an adapter concern, not a model feature.  All
        markings are converted to the ego-relative y coordinate of the
        sequence's initial physical state; lane edges connect neighbours only
        within each carriageway direction.
        """
        from process_highD.src.lane_utils import parse_lane_markings

        lane_info = parse_lane_markings(recording_meta)
        lanes = lane_info["lanes"]
        ordered_ids = list(lane_info["direction_1_lanes"]) + list(lane_info["direction_2_lanes"])
        ordered_ids = ordered_ids[: int(max_lanes)]
        points = 8
        polylines = np.zeros((len(ordered_ids), points, 6), dtype=np.float32)
        valid = np.ones((len(ordered_ids), points), dtype=bool)
        x = np.linspace(float(x_center_m) - float(x_span_m) / 2.0, float(x_center_m) + float(x_span_m) / 2.0, points)
        index_by_id = {lane_id: index for index, lane_id in enumerate(ordered_ids)}
        for index, lane_id in enumerate(ordered_ids):
            item = lanes[lane_id]
            polylines[index, :, 0] = x
            polylines[index, :, 1] = float(lateral_sign) * float(item["center"]) - float(ego_global_y_m)
            polylines[index, :, 2] = 1.0
            polylines[index, :, 4] = float(item["width"])
            polylines[index, :, 5] = 1.0
        edge_rows: list[tuple[int, int, int]] = []
        for direction_ids in (lane_info["direction_1_lanes"], lane_info["direction_2_lanes"]):
            visible = [lane_id for lane_id in direction_ids if lane_id in index_by_id]
            for left, right in zip(visible, visible[1:]):
                edge_rows.extend(((index_by_id[left], index_by_id[right], 1), (index_by_id[right], index_by_id[left], 1)))
        return polylines, valid, np.asarray(edge_rows, np.int64).reshape(-1, 3)

    def adapt(
        self,
        *,
        sequence_id: str,
        recording_id: str,
        ego_id: str,
        timestamps: np.ndarray,
        agent_states: np.ndarray,
        agent_valid: np.ndarray,
        primary_agent_index: int,
        split: str,
        is_evt_tail: bool = False,
        map_override: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> DynamicTrafficSequence:
        sequence = self.builder.sequence_from_dense_slots(
            sequence_id=sequence_id, recording_id=recording_id, ego_id=ego_id,
            timestamps=timestamps, states=agent_states, valid=agent_valid,
            primary_slot_index=int(primary_agent_index), split=split,
            is_evt_tail=bool(is_evt_tail),
        )
        if map_override is None:
            return sequence
        polylines, polyline_valid, lane_edges = map_override
        candidates = np.stack([
            self.builder.lane_candidates_from_polylines(step, mask, polylines, polyline_valid)
            for step, mask in zip(sequence.agent_states, sequence.agent_valid)
        ])
        return replace(
            sequence, map_polylines=np.asarray(polylines, np.float32),
            map_polyline_valid=np.asarray(polyline_valid, bool), lane_graph_edges=np.asarray(lane_edges, np.int64),
            agent_lane_candidates=candidates,
        )

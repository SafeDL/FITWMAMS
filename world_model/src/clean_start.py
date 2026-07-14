"""Bridge the clean Flow START state to a dynamic traffic graph.

The Semi-Markov model must receive a physical graph at time zero, not the
legacy Flow's future-action summary.  This module is deliberately small and
highD-specific only at its boundary: the returned object is the common
``DynamicTrafficGraph`` used by every adapter.
"""
from __future__ import annotations

from typing import Final

import numpy as np

from .graph_builder import DynamicTrafficGraphBuilder, GraphBuilderConfig
from .graph_schema import DynamicTrafficGraph


CLEAN_START_SLOT_NAMES: Final[tuple[str, ...]] = (
    "same_front", "same_rear", "left_front", "left_rear", "right_front", "right_rear",
)
EGO_FEATURE_COUNT: Final[int] = 4
SLOT_FEATURE_COUNT: Final[int] = 6
CLEAN_START_FEATURE_COUNT: Final[int] = EGO_FEATURE_COUNT + len(CLEAN_START_SLOT_NAMES) * SLOT_FEATURE_COUNT
CLEAN_START_ADAPTER_VERSION: Final[str] = "highd_clean_start_to_dynamic_graph_v1"


def graph_from_clean_start(
    feature_row: np.ndarray,
    slot_mask: np.ndarray,
    *,
    primary_slot_index: int | None = None,
    timestamp: float = 0.0,
    lane_width_m: float = 3.6,
    top_r_lanes: int = 3,
    map_polylines: np.ndarray | None = None,
    map_polyline_valid: np.ndarray | None = None,
    lane_graph_edges: np.ndarray | None = None,
) -> DynamicTrafficGraph:
    """Create ``G0`` from one 40-D Flow sample and its event structure.

    The accepted vector is exactly the ``clean_start`` schema:
    ego ``[vx, vy, ax, ay]`` followed by each slot's current relative
    ``[x, y, vx, vy, ax, ay]``.  A 76-D legacy Flow sample is rejected rather
    than silently allowing a future one-second action summary into the world
    model.
    """
    row = np.asarray(feature_row, dtype=np.float32).reshape(-1)
    if row.shape != (CLEAN_START_FEATURE_COUNT,):
        raise ValueError(
            f"clean_start feature_row must have {CLEAN_START_FEATURE_COUNT} current-state values; "
            f"got shape {row.shape}"
        )
    if not np.isfinite(row).all():
        raise ValueError("clean_start feature_row must contain only finite physical values")
    active_slots = np.asarray(slot_mask, dtype=bool).reshape(-1)
    if active_slots.shape != (len(CLEAN_START_SLOT_NAMES),):
        raise ValueError(f"slot_mask must have shape ({len(CLEAN_START_SLOT_NAMES)},)")
    if primary_slot_index is not None and int(primary_slot_index) >= 0:
        if int(primary_slot_index) >= len(CLEAN_START_SLOT_NAMES) or not active_slots[int(primary_slot_index)]:
            raise ValueError("primary_slot_index must identify an active clean-start slot")

    states = np.zeros((1 + len(CLEAN_START_SLOT_NAMES), 6), dtype=np.float32)
    # Flow coordinates are ego-centric, hence ego position is the origin.
    states[0, 2:] = row[:EGO_FEATURE_COUNT]
    for slot_index in range(len(CLEAN_START_SLOT_NAMES)):
        start = EGO_FEATURE_COUNT + slot_index * SLOT_FEATURE_COUNT
        relative = row[start : start + SLOT_FEATURE_COUNT]
        states[slot_index + 1] = (
            relative[0],
            relative[1],
            row[0] + relative[2],
            row[1] + relative[3],
            relative[4],
            relative[5],
        )
    valid = np.concatenate((np.asarray([True]), active_slots))
    builder = DynamicTrafficGraphBuilder(GraphBuilderConfig(
        lane_width_m=float(lane_width_m), top_r_lanes=int(top_r_lanes),
    ))
    supplied_map = (map_polylines, map_polyline_valid, lane_graph_edges)
    if any(value is not None for value in supplied_map):
        if not all(value is not None for value in supplied_map):
            raise ValueError("map_polylines, map_polyline_valid, and lane_graph_edges must be supplied together")
        polylines = np.asarray(map_polylines, dtype=np.float32)
        polyline_valid = np.asarray(map_polyline_valid, dtype=bool)
        edges = np.asarray(lane_graph_edges, dtype=np.int64).reshape(-1, 3)
    else:
        polylines, polyline_valid, edges = builder.straight_lane_map(states, valid)
    return builder.graph_at(
        timestamp=float(timestamp),
        agent_ids=np.arange(len(states), dtype=np.int64), states=states, valid=valid,
        ego_index=0,
        primary_agent_index=(-1 if primary_slot_index is None or int(primary_slot_index) < 0 else int(primary_slot_index) + 1),
        map_polylines=polylines, map_polyline_valid=polyline_valid, lane_graph_edges=edges,
    )

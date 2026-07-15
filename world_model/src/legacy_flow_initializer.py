"""Atomic construction of a graph and B0 from one frozen legacy Flow row."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from normalizing_flow.src.features import EGO_FEATURES, SLOT_NAMES, slot_feature_index

from .graph_builder import DynamicTrafficGraphBuilder
from .graph_schema import DynamicTrafficGraph
from .initial_behavior_anchor import FrozenLegacyFlowSchema, behavior_anchor_from_flow_feature


@dataclass(frozen=True)
class FlowInitializedScene:
    graph: DynamicTrafficGraph
    behavior_anchor_raw: np.ndarray
    behavior_anchor_std: np.ndarray
    behavior_anchor_valid: np.ndarray
    slot_to_agent_index: np.ndarray
    primary_agent_index: int
    flow_schema_sha256: str
    map_adapter_version: str
    map_context_sha256: str


def _hash_map_context(context: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(str(context.get("map_adapter_version", "straight_lane_v1")).encode())
    for key in ("map_polylines", "map_polyline_valid", "lane_graph_edges", "conflict_zone_features", "conflict_zone_valid"):
        if key in context and context[key] is not None:
            value = np.ascontiguousarray(np.asarray(context[key]))
            digest.update(key.encode()); digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def _schema(context: dict[str, Any]) -> FrozenLegacyFlowSchema:
    supplied = context.get("frozen_flow_schema")
    if isinstance(supplied, FrozenLegacyFlowSchema):
        return supplied
    source = context.get("flow_schema_path")
    if source is None:
        raise ValueError("map_context must provide frozen_flow_schema or flow_schema_path")
    return FrozenLegacyFlowSchema.load(Path(source))


def graph_and_anchor_from_legacy_flow(
    feature_row: np.ndarray,
    slot_mask: np.ndarray,
    primary_slot_index: int,
    map_context: dict[str, Any],
) -> FlowInitializedScene:
    """Build an inseparable `(graph, B0)` scene from a raw 76-D Flow sample."""
    feature = np.asarray(feature_row, np.float32).reshape(-1)
    mask = np.asarray(slot_mask, bool).reshape(-1)
    if feature.shape != (76,) or mask.shape != (len(SLOT_NAMES),):
        raise ValueError("legacy Flow initialization requires one raw 76-D row and six-slot mask")
    if not np.isfinite(feature[:4]).all():
        raise ValueError("legacy Flow ego features must be finite")
    if not 0 <= int(primary_slot_index) < len(SLOT_NAMES) or not mask[int(primary_slot_index)]:
        raise ValueError("primary_slot_index must refer to an active legacy Flow slot")
    contract = _schema(map_context)
    expected_schema_hash = map_context.get("flow_schema_sha256")
    if expected_schema_hash is not None and contract.schema_sha256 != str(expected_schema_hash):
        raise ValueError("map_context Flow schema hash does not match its frozen contract")
    for slot, active in zip(SLOT_NAMES, mask):
        if active:
            indices = [slot_feature_index(slot, name) for name in ("rel_x_m", "rel_y_left_m", "rel_vx_mps", "rel_vy_left_mps", "other_ax_mps2", "other_ay_left_mps2")]
            if not np.isfinite(feature[indices]).all():
                raise ValueError(f"active Flow slot {slot} contains non-finite S0 features")
    ego = {name: feature[EGO_FEATURES.index(name)] for name in EGO_FEATURES}
    states = np.zeros((1 + len(SLOT_NAMES), 6), np.float32)
    valid = np.zeros((1 + len(SLOT_NAMES),), bool)
    states[0] = (0.0, 0.0, ego["ego_vx_mps"], ego["ego_vy_left_mps"], ego["ego_ax_mps2"], ego["ego_ay_left_mps2"])
    valid[0] = True
    for index, slot in enumerate(SLOT_NAMES):
        if not mask[index]:
            continue
        states[index + 1] = (
            feature[slot_feature_index(slot, "rel_x_m")], feature[slot_feature_index(slot, "rel_y_left_m")],
            ego["ego_vx_mps"] + feature[slot_feature_index(slot, "rel_vx_mps")],
            ego["ego_vy_left_mps"] + feature[slot_feature_index(slot, "rel_vy_left_mps")],
            feature[slot_feature_index(slot, "other_ax_mps2")], feature[slot_feature_index(slot, "other_ay_left_mps2")],
        )
        valid[index + 1] = True
    anchor_raw, anchor_valid = behavior_anchor_from_flow_feature(feature, mask)
    anchor_std = contract.standardize(torch.from_numpy(anchor_raw)[None], torch.from_numpy(anchor_valid)[None]).squeeze(0).numpy()
    builder = map_context.get("graph_builder") or DynamicTrafficGraphBuilder()
    if all(key in map_context for key in ("map_polylines", "map_polyline_valid", "lane_graph_edges")):
        lanes = np.asarray(map_context["map_polylines"], np.float32)
        lane_valid = np.asarray(map_context["map_polyline_valid"], bool)
        lane_edges = np.asarray(map_context["lane_graph_edges"], np.int64)
    else:
        lanes, lane_valid, lane_edges = builder.straight_lane_map(states, valid)
    graph = builder.graph_at(
        timestamp=float(map_context.get("timestamp", 0.0)), agent_ids=np.asarray(map_context.get("agent_ids", np.arange(len(states))), np.int64),
        states=states, valid=valid, ego_index=0, primary_agent_index=int(primary_slot_index) + 1,
        map_polylines=lanes, map_polyline_valid=lane_valid, lane_graph_edges=lane_edges,
        conflict_zone_features=map_context.get("conflict_zone_features"), conflict_zone_valid=map_context.get("conflict_zone_valid"),
    )
    mapping = np.arange(1, 1 + len(SLOT_NAMES), dtype=np.int64)
    if not np.array_equal(graph.agent_valid[1:], mask) or graph.primary_agent_index != int(primary_slot_index) + 1:
        raise RuntimeError("atomic Flow graph construction violated the fixed slot-to-agent mapping")
    return FlowInitializedScene(
        graph, anchor_raw, anchor_std, anchor_valid, mapping, int(primary_slot_index) + 1,
        contract.schema_sha256, str(map_context.get("map_adapter_version", "straight_lane_v1")), _hash_map_context(map_context),
    )

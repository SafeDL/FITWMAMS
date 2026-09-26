"""Dataset-neutral dynamic traffic graph schema.

The sequence pipeline uses this schema instead of highD's fixed-slot
six-slot layout.  Arrays are padded only at batching time; ``agent_valid`` and
``agent_ids`` preserve the variable participant set of every scene.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np


AGENT_STATE_DIM: Final[int] = 6
AGENT_FEATURE_DIM: Final[int] = 10
MAP_POINT_DIM: Final[int] = 6
AA_EDGE_DIM: Final[int] = 8
AL_EDGE_DIM: Final[int] = 4
CONFLICT_ZONE_DIM: Final[int] = 4  # [x, y, radius, priority]
AC_EDGE_DIM: Final[int] = 4        # [dx, dy, clearance, priority]

RELATION_TYPES: Final[tuple[str, ...]] = (
    "same_lane", "adjacent_lane", "merge", "diverge", "cross", "unrelated",
)
RELATION_TO_INDEX: Final[dict[str, int]] = {
    name: index for index, name in enumerate(RELATION_TYPES)
}


@dataclass(frozen=True)
class DynamicTrafficGraph:
    """A single graph at one physical time.

    ``agent_states`` uses the common compatibility representation
    ``[x, y, vx, vy, ax, ay]``.  The encoder derives local velocity, heading,
    geometry and validity features from it so it is not tied to a road-global
    coordinate system.
    """

    timestamp: float
    agent_ids: np.ndarray                 # [N]
    agent_states: np.ndarray              # [N, 6]
    agent_valid: np.ndarray               # [N]
    ego_index: int
    primary_agent_index: int = -1
    map_polylines: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, MAP_POINT_DIM), np.float32))
    map_polyline_valid: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), bool))
    lane_graph_edges: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int64))
    aa_edge_index: np.ndarray = field(default_factory=lambda: np.zeros((2, 0), np.int64))
    aa_edge_features: np.ndarray = field(default_factory=lambda: np.zeros((0, AA_EDGE_DIM), np.float32))
    al_lane_indices: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.int64))
    al_edge_features: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, AL_EDGE_DIM), np.float32))
    conflict_zone_features: np.ndarray = field(default_factory=lambda: np.zeros((0, CONFLICT_ZONE_DIM), np.float32))
    conflict_zone_valid: np.ndarray = field(default_factory=lambda: np.zeros((0,), bool))
    ac_edge_index: np.ndarray = field(default_factory=lambda: np.zeros((2, 0), np.int64))
    ac_edge_features: np.ndarray = field(default_factory=lambda: np.zeros((0, AC_EDGE_DIM), np.float32))

    def __post_init__(self) -> None:
        n = int(np.asarray(self.agent_ids).shape[0])
        if np.asarray(self.agent_states).shape != (n, AGENT_STATE_DIM):
            raise ValueError("agent_states must have shape [N, 6]")
        if np.asarray(self.agent_valid).shape != (n,):
            raise ValueError("agent_valid must have shape [N]")
        if not 0 <= int(self.ego_index) < n:
            raise ValueError("ego_index must refer to an agent")
        zones = np.asarray(self.conflict_zone_features)
        zone_valid = np.asarray(self.conflict_zone_valid)
        if zones.ndim != 2 or zones.shape[1:] != (CONFLICT_ZONE_DIM,) or zone_valid.shape != (zones.shape[0],):
            raise ValueError("conflict zones must have shapes [C, 4] and [C]")
        ac_index = np.asarray(self.ac_edge_index)
        ac_features = np.asarray(self.ac_edge_features)
        if ac_index.ndim != 2 or ac_index.shape[0] != 2 or ac_features.shape != (ac_index.shape[1], AC_EDGE_DIM):
            raise ValueError("agent-conflict edges must have shapes [2, E] and [E, 4]")


@dataclass(frozen=True)
class DynamicTrafficSequence:
    """The sequence-level cache unit used by semi-Markov training."""

    sequence_id: str
    recording_id: str
    ego_id: str
    timestamps: np.ndarray                # [T]
    agent_ids: np.ndarray                 # [N]
    agent_states: np.ndarray              # [T, N, 6]
    agent_valid: np.ndarray               # [T, N]
    ego_index: int
    primary_agent_index: int
    map_polylines: np.ndarray             # [M, P, 6]
    map_polyline_valid: np.ndarray        # [M, P]
    lane_graph_edges: np.ndarray          # [E, 3]
    agent_lane_candidates: np.ndarray     # [T, N, R]
    split: str
    is_evt_tail: bool = False
    conflict_zone_features: np.ndarray = field(default_factory=lambda: np.zeros((0, CONFLICT_ZONE_DIM), np.float32))
    conflict_zone_valid: np.ndarray = field(default_factory=lambda: np.zeros((0,), bool))

    def __post_init__(self) -> None:
        t, n, d = np.asarray(self.agent_states).shape
        if d != AGENT_STATE_DIM or np.asarray(self.agent_valid).shape != (t, n):
            raise ValueError("invalid sequence agent state or validity shape")
        if np.asarray(self.timestamps).shape != (t,) or np.asarray(self.agent_ids).shape != (n,):
            raise ValueError("timestamps and agent_ids must align with agent_states")
        zones = np.asarray(self.conflict_zone_features)
        if zones.ndim != 2 or zones.shape[1:] != (CONFLICT_ZONE_DIM,) or np.asarray(self.conflict_zone_valid).shape != (zones.shape[0],):
            raise ValueError("invalid conflict-zone sequence arrays")


def empty_edges() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((2, 0), np.int64), np.zeros((0, AA_EDGE_DIM), np.float32)

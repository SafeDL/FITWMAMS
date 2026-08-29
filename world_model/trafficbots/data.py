"""Cache-only highD to TrafficBots V1.5 schema adaptation."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from world_model.src.core.sequential_dataset import load_sequential_dataset
from world_model.src.core.evaluation_scope import scoped_canonical_trajectory

ANCHOR_INDEX = 24
STATE_POINTS = 150
ROLL_STEPS = 149
REAL_AGENTS, PADDED_AGENTS = 7, 8
REAL_MAPS, PADDED_MAPS, MAP_NODES = 8, 16, 8
PADDED_TLS = 8
FREEWAY_INDEX = 0
DT_S = 0.04
# Below this speed highD finite-difference velocity is not a reliable vehicle
# heading (a stopped car can otherwise appear to point at pi because vx is a
# tiny negative rounding residual).  The specified causal fallback is the
# latest realised yaw, or zero at S0.
YAW_SPEED_EPS_MPS = 1.0


def split_rows(arrays: dict[str, np.ndarray], split: str, *, seed: int = 0, maximum: int = 0) -> np.ndarray:
    index = {"train": 0, "val": 1, "test": 2}[split]
    rows = np.flatnonzero(np.asarray(arrays["split_index"]) == index)
    np.random.default_rng(seed).shuffle(rows)
    return rows[:maximum] if maximum else rows


def sequence_hashes(arrays: dict[str, np.ndarray]) -> dict[str, str]:
    identifiers = np.asarray(arrays["sequence_id"]).astype(str)
    return {
        split: hashlib.sha256("\n".join(sorted(identifiers[split_rows(arrays, split)])) .encode()).hexdigest()
        for split in ("train", "val", "test")
    }


def _wrap(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def states_to_motion(states: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert canonical states to pose and motion without peeking beyond S0."""
    states = np.asarray(states, np.float32)
    valid = np.asarray(valid, bool)
    velocity = states[..., 2:4]
    speed = np.linalg.norm(velocity, axis=-1)
    raw_yaw = np.arctan2(velocity[..., 1], velocity[..., 0])
    yaw = np.zeros_like(speed, np.float32)
    for time in range(states.shape[-2]):
        observable = valid[..., time] & (speed[..., time] >= YAW_SPEED_EPS_MPS)
        if time:
            yaw[..., time] = yaw[..., time - 1]
        yaw[..., time][observable] = raw_yaw[..., time][observable]
    acceleration = states[..., 4] * np.cos(yaw) + states[..., 5] * np.sin(yaw)
    yaw_rate = np.zeros_like(speed, np.float32)
    if states.shape[-2] > 1:
        yaw_rate[..., 1:] = _wrap(yaw[..., 1:] - yaw[..., :-1]) / DT_S
    yaw_rate[~valid] = 0.0
    acceleration[~valid] = 0.0
    pose = np.concatenate((states[..., :2], yaw[..., None]), axis=-1)
    motion = np.stack((speed, acceleration, yaw_rate), axis=-1)
    pose[~valid] = 0.0
    motion[~valid] = 0.0
    return pose.astype(np.float32), motion.astype(np.float32)


def _map_pose(polylines: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create map poses from geometric tangents rather than cache feature columns."""
    points = np.asarray(polylines[..., :2], np.float32)
    point_valid = np.asarray(valid, bool)
    delta = np.zeros_like(points)
    delta[..., :-1, :] = points[..., 1:, :] - points[..., :-1, :]
    delta[..., -1, :] = delta[..., -2, :]
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    direction = np.divide(delta, np.maximum(norm, 1.0e-6), out=np.zeros_like(delta), where=norm > 1.0e-6)
    yaw = np.arctan2(direction[..., 1], direction[..., 0])
    pose = np.concatenate((points, yaw[..., None]), axis=-1)
    pose[~point_valid] = 0.0
    return pose.astype(np.float32), direction.astype(np.float32)


def _destination_contract(
    pose: np.ndarray, valid: np.ndarray, map_pose: np.ndarray, map_valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-agent directional lane candidates and final-lane GT targets."""
    batch, agents, steps, _ = pose.shape
    maps = map_pose.shape[1]
    candidates = np.zeros((batch, agents, maps), bool)
    destination = np.zeros((batch, agents), np.int64)
    lane_valid = map_valid.any(-1)
    for item in range(batch):
        for agent in range(agents):
            if not valid[item, agent, 0]:
                continue
            start = pose[item, agent, 0, :2]
            heading = np.array((np.cos(pose[item, agent, 0, 2]), np.sin(pose[item, agent, 0, 2])), np.float32)
            distance = np.sum((map_pose[item, :, :, :2] - start) ** 2, axis=-1)
            distance[~map_valid[item]] = np.inf
            nearest_node = np.argmin(distance, axis=-1)
            tangent = map_pose[item, np.arange(maps), nearest_node, 2]
            compatible = np.cos(tangent) * heading[0] + np.sin(tangent) * heading[1] >= 0.0
            candidates[item, agent] = lane_valid[item] & compatible
            if not candidates[item, agent].any():
                raise ValueError("S0-valid agent has no direction-compatible FREEWAY destination")
            last = np.flatnonzero(valid[item, agent])[-1]
            final_xy = pose[item, agent, last, :2]
            distance = np.sum((map_pose[item, :, :, :2] - final_xy) ** 2, axis=-1)
            distance[~map_valid[item]] = np.inf
            lane_distance = distance.min(-1)
            lane_distance[~candidates[item, agent]] = np.inf
            destination[item, agent] = int(np.argmin(lane_distance))
    return candidates, destination


def adapt_highd_batch(
    states: np.ndarray, valid: np.ndarray, polylines: np.ndarray, polyline_valid: np.ndarray
) -> dict[str, np.ndarray]:
    """Build the fixed-shape TrafficBots input without any non-cache data."""
    states = np.asarray(states, np.float32)
    valid = np.asarray(valid, bool)
    if states.ndim == 3:
        states, valid, polylines, polyline_valid = (value[None] for value in (states, valid, polylines, polyline_valid))
    if states.shape[1:] != (STATE_POINTS, REAL_AGENTS, 6):
        raise ValueError(f"expected [B,{STATE_POINTS},{REAL_AGENTS},6] canonical states")
    batch = states.shape[0]
    agent_states = np.swapaxes(states, 1, 2)
    agent_valid = np.swapaxes(valid, 1, 2)
    pose, motion = states_to_motion(agent_states, agent_valid)
    padded_valid = np.zeros((batch, PADDED_AGENTS, STATE_POINTS), bool)
    padded_pose = np.zeros((batch, PADDED_AGENTS, STATE_POINTS, 3), np.float32)
    padded_motion = np.zeros((batch, PADDED_AGENTS, STATE_POINTS, 3), np.float32)
    padded_valid[:, :REAL_AGENTS], padded_pose[:, :REAL_AGENTS], padded_motion[:, :REAL_AGENTS] = agent_valid, pose, motion
    map_valid = np.zeros((batch, PADDED_MAPS, MAP_NODES), bool)
    map_valid[:, :REAL_MAPS] = polyline_valid
    map_pose = np.zeros((batch, PADDED_MAPS, MAP_NODES, 3), np.float32)
    real_pose, _ = _map_pose(polylines, polyline_valid)
    map_pose[:, :REAL_MAPS] = real_pose
    map_type = np.zeros((batch, PADDED_MAPS, 11), bool)
    map_type[:, :REAL_MAPS, FREEWAY_INDEX] = polyline_valid.any(-1)
    candidates, destination = _destination_contract(padded_pose, padded_valid, map_pose, map_valid)
    agent_type = np.zeros((batch, PADDED_AGENTS, 3), bool)
    # Test-time identity and physical attributes are an S0 observation.  Using
    # ``any(time)`` here would reveal a future spawn even if the released highD
    # cache currently contains no spawning agents.
    agent_type[:, :REAL_AGENTS, 0] = agent_valid[..., 0]
    agent_size = np.zeros((batch, PADDED_AGENTS, 3), np.float32)
    # A canonical slot is not necessarily an agent.  In particular, do not
    # give padded slots a physical footprint merely because they are within
    # the seven cache slots.
    agent_size[agent_type.any(-1)] = np.asarray((4.8, 1.8, 1.5), np.float32)
    agent_role = np.zeros((batch, PADDED_AGENTS, 3), bool)
    agent_role[:, 0, 0] = agent_valid[:, 0, 0]
    agent_role[:, :REAL_AGENTS, 2] = agent_valid[..., 0]
    map_xy = map_pose[..., :2][map_valid]
    boundary = np.zeros((batch, 4), np.float32)
    for item in range(batch):
        xy = map_pose[item, ..., :2][map_valid[item]]
        if len(xy):
            boundary[item] = (xy[:, 0].min() - 20, xy[:, 0].max() + 20, xy[:, 1].min() - 20, xy[:, 1].max() + 20)
    tl_valid = np.zeros((batch, PADDED_TLS, STATE_POINTS), bool)
    tl_state = np.zeros((batch, PADDED_TLS, STATE_POINTS, 5), bool)
    tl_pos = np.zeros((batch, PADDED_TLS, 3), np.float32)
    tl_dir = np.zeros((batch, PADDED_TLS, 3), np.float32); tl_dir[..., 0] = 1.0
    return {
        "agent/valid": padded_valid, "agent/pos": np.concatenate((padded_pose[..., :2], np.zeros((*padded_pose.shape[:-1], 1), np.float32)), -1),
        "agent/vel": np.stack((padded_motion[..., 0] * np.cos(padded_pose[..., 2]), padded_motion[..., 0] * np.sin(padded_pose[..., 2])), -1),
        "agent/spd": padded_motion[..., :1], "agent/acc": padded_motion[..., 1:2], "agent/yaw_bbox": padded_pose[..., 2:3], "agent/yaw_rate": padded_motion[..., 2:3],
        "agent/type": agent_type, "agent/role": agent_role, "agent/size": agent_size, "agent/dest": destination,
        "destination_candidate_valid": candidates, "map/valid": map_valid, "map/type": map_type,
        "map/pos": np.concatenate((map_pose[..., :2], np.zeros((*map_pose.shape[:-1], 1), np.float32)), -1),
        "map/dir": np.concatenate((np.cos(map_pose[..., 2:3]), np.sin(map_pose[..., 2:3]), np.zeros((*map_pose.shape[:-1], 1), np.float32)), -1),
        "map/boundary": boundary, "tl_stop/valid": tl_valid, "tl_stop/state": tl_state, "tl_stop/pos": tl_pos, "tl_stop/dir": tl_dir,
    }


class TrafficBotsHighDDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        *,
        seed: int = 0,
        maximum: int = 0,
        evaluation_scope: bool = False,
    ) -> None:
        self.arrays, self.manifest = load_sequential_dataset(cache_dir)
        self.rows = split_rows(self.arrays, split, seed=seed, maximum=maximum)
        self.evaluation_scope = bool(evaluation_scope)

    def __len__(self) -> int: return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = int(self.rows[index])
        states = np.asarray(self.arrays["agent_states"][row, ANCHOR_INDEX:ANCHOR_INDEX + STATE_POINTS], np.float32)
        valid = np.asarray(self.arrays["agent_valid"][row, ANCHOR_INDEX:ANCHOR_INDEX + STATE_POINTS], bool)
        full_states = np.asarray(self.arrays["agent_states"][row], np.float32)
        full_valid = np.asarray(self.arrays["agent_valid"][row], bool)
        if self.evaluation_scope:
            states, valid = scoped_canonical_trajectory(states, valid)
            full_states, full_valid = scoped_canonical_trajectory(
                full_states, full_valid
            )
        batch = adapt_highd_batch(states, valid, self.arrays["map_polylines"][row], self.arrays["map_polyline_valid"][row])
        output = {key: torch.from_numpy(value[0].copy()) for key, value in batch.items()}
        output["canonical/states"] = torch.from_numpy(states.copy())
        output["canonical/valid"] = torch.from_numpy(valid.copy())
        output["canonical/actions_highd"] = torch.from_numpy(
            np.asarray(self.arrays["actions_highd"][row], np.float32).copy()
        )
        output["canonical/full_states"] = torch.from_numpy(
            np.asarray(full_states, np.float32).copy()
        )
        output["canonical/full_valid"] = torch.from_numpy(
            np.asarray(full_valid, bool).copy()
        )
        output["is_evt_tail"] = torch.tensor(bool(self.arrays["is_evt_tail"][row]))
        output["sequence_id"] = str(self.arrays["sequence_id"][row]); output["row_index"] = torch.tensor(row)
        return output


def make_loader(
    dataset: Dataset, *, batch_size: int, shuffle: bool, workers: int = 0,
    seed: int | None = None,
) -> DataLoader:
    generator = None if seed is None else torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        generator=generator,
    )

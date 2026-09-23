"""PyTorch views of the causal highD cache."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .behavior_condition import append_behavior_condition
from .causal_cache import load_causal_cache
from .manifest import load_benchmark_config

DECISION_OFFSETS = np.arange(0, 149, 5, dtype=np.int64)


def aggregate_decision_controls(controls: np.ndarray) -> np.ndarray:
    """Aggregate 149 native controls into 30 causal 5 Hz decisions."""
    values = np.asarray(controls, np.float32)
    output = []
    for start in DECISION_OFFSETS:
        output.append(values[..., start : min(start + 5, 149), :, :].mean(axis=-3))
    return np.stack(output, axis=-3)


def policy_features(
    history: np.ndarray,
    valid: np.ndarray,
    lengths_m: np.ndarray,
    widths_m: np.ndarray,
    map_polylines: np.ndarray,
    map_valid: np.ndarray,
) -> np.ndarray:
    """Create ego-relative temporal features without future information."""
    states = np.asarray(history, np.float32)
    present = np.asarray(valid, bool)
    current_ego = states[-1, 0]
    position = states[..., :2] - current_ego[None, None, :2]
    relative_velocity = states[..., 2:4] - current_ego[None, None, 2:4]
    size = np.stack((lengths_m, widths_m), axis=-1).astype(np.float32)
    size = np.broadcast_to(size[None], (*states.shape[:2], 2))
    lane_mask = np.asarray(map_valid, bool).any(-1)
    if lane_mask.any():
        lines = np.asarray(map_polylines, np.float32)[lane_mask]
        line_valid = np.asarray(map_valid, bool)[lane_mask]
        centers = (lines[..., 1] * line_valid).sum(-1) / line_valid.sum(-1).clip(1)
        lane_widths = (lines[..., 4] * line_valid).sum(-1) / line_valid.sum(-1).clip(1)
        nearest = np.abs(states[..., 1, None] - centers).argmin(-1)
        lane = np.stack(
            (states[..., 1] - centers[nearest], lane_widths[nearest]), axis=-1
        )
    else:
        lane = np.zeros((*states.shape[:2], 2), np.float32)
    feature = np.concatenate(
        (position, states[..., 2:6], relative_velocity, size, lane), axis=-1
    )
    return (feature * present[..., None]).astype(np.float32, copy=False)


class CausalBCDataset(Dataset):
    """One scenario per item, with a deterministic offset varying by epoch."""

    def __init__(
        self,
        config_path: str | Path,
        split: str,
        *,
        seed: int,
        all_offsets: bool = False,
    ) -> None:
        self.arrays, self.metadata, self.cache_manifest = load_causal_cache(config_path)
        split_value = {"train": 0, "validation": 1, "val": 1, "test": 2}[split]
        self.rows = np.flatnonzero(self.metadata["split_index"] == split_value)
        self.seed = int(seed)
        self.epoch = 0
        self.all_offsets = bool(all_offsets)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows) * (len(DECISION_OFFSETS) if self.all_offsets else 1)

    def _row_offset(self, index: int) -> tuple[int, int]:
        if self.all_offsets:
            return int(self.rows[index // len(DECISION_OFFSETS)]), int(
                index % len(DECISION_OFFSETS)
            )
        row = int(self.rows[index])
        # Coprime multipliers give repeatable but changing coverage per epoch.
        offset_index = (row * 17 + self.epoch * 13 + self.seed) % len(DECISION_OFFSETS)
        return row, int(offset_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row, offset_index = self._row_offset(int(index))
        offset = int(DECISION_OFFSETS[offset_index])
        decision_state = 24 + offset
        history = np.asarray(
            self.arrays["agent_states"][row, decision_state - 24 : decision_state + 1]
        ).copy()
        valid = np.asarray(
            self.arrays["agent_valid"][row, decision_state - 24 : decision_state + 1]
        ).copy()
        features = policy_features(
            history,
            valid,
            self.metadata["lengths_m"][row],
            self.metadata["widths_m"][row],
            self.arrays["map_polylines"][row],
            self.arrays["map_polyline_valid"][row],
        )
        start = offset
        target = (
            np.asarray(self.arrays["target_controls"][row, start : min(start + 5, 149)])
            .mean(0)
            .copy()
        )
        return {
            "features": torch.from_numpy(features),
            "history_valid": torch.from_numpy(valid),
            "target": torch.from_numpy(target),
            "target_valid": torch.from_numpy(valid[-1, 1:].copy()),
            "scenario_row": torch.tensor(row, dtype=torch.long),
            "decision_offset": torch.tensor(offset, dtype=torch.long),
        }


def _controls_from_state_acceleration(states: np.ndarray) -> np.ndarray:
    """Project canonical Cartesian acceleration to ``[a, yaw_rate]``."""
    values = np.asarray(states, np.float32)
    speed = np.linalg.norm(values[..., 2:4], axis=-1).clip(0.5)
    heading = np.arctan2(
        values[..., 3],
        np.where(np.abs(values[..., 2]) < 1.0e-4, 1.0e-4, values[..., 2]),
    )
    ax, ay = values[..., 4], values[..., 5]
    longitudinal = ax * np.cos(heading) + ay * np.sin(heading)
    lateral = -ax * np.sin(heading) + ay * np.cos(heading)
    return np.stack((longitudinal, lateral / speed), axis=-1).astype(np.float32)


class EgoBCDataset(CausalBCDataset):
    """Causal structured histories with logged ego-control targets."""

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        base = super().__getitem__(index)
        row = int(base["scenario_row"])
        offset = int(base["decision_offset"])
        start = 24 + offset
        stop = min(start + 5, 173)
        ego_states = np.asarray(self.arrays["agent_states"][row, start:stop, 0])
        base["ego_target"] = torch.from_numpy(
            _controls_from_state_acceleration(ego_states).mean(0)
        )
        return base


def estimate_ego_action_statistics(dataset: CausalBCDataset) -> dict[str, np.ndarray]:
    total = np.zeros(2, np.float64)
    square = np.zeros(2, np.float64)
    count = 0
    for start in range(0, len(dataset.rows), 512):
        rows = dataset.rows[start : start + 512]
        states = np.asarray(dataset.arrays["agent_states"][rows, 24:-1, 0], np.float32)
        controls = _controls_from_state_acceleration(states).astype(np.float64)
        total += controls.sum((0, 1))
        square += np.square(controls).sum((0, 1))
        count += controls.shape[0] * controls.shape[1]
    mean = total / max(count, 1)
    std = np.sqrt(np.maximum(square / max(count, 1) - np.square(mean), 1.0e-6))
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "count": np.asarray(count),
    }


class CausalActionBlockDataset(CausalBCDataset):
    """Prefix-conditioned 15-decision action blocks for Idea B."""

    block_decisions = 15

    def _row_offset(self, index: int) -> tuple[int, int]:
        if self.all_offsets:
            return super()._row_offset(index)
        row = int(self.rows[index])
        # The final offset has only four native transitions; B's block
        # training uses complete five-frame decisions only.
        offset_index = (row * 17 + self.epoch * 13 + self.seed) % (
            len(DECISION_OFFSETS) - 1
        )
        return row, int(offset_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        base = super().__getitem__(index)
        row = int(base["scenario_row"])
        offset = int(base["decision_offset"])
        offset_index = int(np.searchsorted(DECISION_OFFSETS, offset))
        active = np.asarray(self.arrays["agent_valid"][row, 24 + offset, 1:]).copy()
        block = np.zeros((self.block_decisions, 6, 2), np.float32)
        block_valid = np.zeros((self.block_decisions, 6), bool)
        future_states = np.zeros((self.block_decisions, 6, 6), np.float32)
        for local in range(self.block_decisions):
            decision = offset_index + local
            if decision >= len(DECISION_OFFSETS):
                break
            start = int(DECISION_OFFSETS[decision])
            stop = min(start + 5, 149)
            if stop - start < 5:
                break
            block[local] = np.asarray(
                self.arrays["target_controls"][row, start:stop]
            ).mean(0)
            endpoint = min(24 + stop, 173)
            future_states[local] = np.asarray(
                self.arrays["agent_states"][row, endpoint, 1:]
            )
            block_valid[local] = active
        base.update(
            {
                "action_block": torch.from_numpy(block),
                "action_block_valid": torch.from_numpy(block_valid),
                "current_background_state": torch.from_numpy(
                    np.asarray(self.arrays["agent_states"][row, 24 + offset, 1:]).copy()
                ),
                "future_background_states": torch.from_numpy(future_states),
            }
        )
        return base


class CausalActionSelfRolloutDataset(CausalActionBlockDataset):
    """B2 view adding causal state needed for one self-unfolded decision."""

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        base = super().__getitem__(index)
        row = int(base["scenario_row"])
        offset = int(base["decision_offset"])
        decision_state = 24 + offset
        base.update(
            {
                "raw_history": torch.from_numpy(
                    np.asarray(
                        self.arrays["agent_states"][
                            row, decision_state - 24 : decision_state + 1
                        ]
                    ).copy()
                ),
                "next_ego_states": torch.from_numpy(
                    np.asarray(
                        self.arrays["agent_states"][
                            row, decision_state + 1 : decision_state + 6, 0
                        ]
                    ).copy()
                ),
                "lengths": torch.from_numpy(self.metadata["lengths_m"][row].copy()),
                "widths": torch.from_numpy(self.metadata["widths_m"][row].copy()),
                "map_polylines": torch.from_numpy(
                    np.asarray(self.arrays["map_polylines"][row]).copy()
                ),
                "map_valid": torch.from_numpy(
                    np.asarray(self.arrays["map_polyline_valid"][row]).copy()
                ),
            }
        )
        return base


class EventAnchoredBCDataset(Dataset):
    """D1 event-onset histories with causal five-frame action targets."""

    def __init__(
        self, config_path: str | Path, split: str, *, seed: int, fraction: float = 1.0
    ) -> None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("event fraction must be in (0, 1]")
        self.arrays, self.metadata, self.cache_manifest = load_causal_cache(config_path)
        config, _ = load_benchmark_config(config_path)
        split_value = {"train": 0, "validation": 1, "val": 1, "test": 2}[split]
        events = pd.read_csv(Path(config["paths"]["output_dir"]) / "event_manifest.csv")
        selected = events[events.split_index == split_value].copy()
        if fraction < 1.0:
            rng = np.random.default_rng(int(seed))
            indices = []
            for _, family in selected.groupby("event_type", sort=True):
                count = max(1, int(round(len(family) * float(fraction))))
                indices.extend(
                    rng.choice(
                        family.index.to_numpy(), size=count, replace=False
                    ).tolist()
                )
            selected = selected.loc[sorted(indices)]
        self.events = selected.reset_index(drop=True)
        self.fraction = float(fraction)
        if len(self.events):
            onset = self.events.local_onset_frame.to_numpy(np.int64)
            if (onset < 24).any() or (
                onset + 75 >= self.arrays["agent_states"].shape[1]
            ).any():
                raise ValueError(
                    "D1 event anchor does not provide the required 25-frame history and 75-frame future"
                )

    def set_epoch(self, epoch: int) -> None:
        del epoch

    def __len__(self) -> int:
        return len(self.events)

    def _base(self, index: int) -> tuple[dict[str, torch.Tensor], int, int]:
        event = self.events.iloc[int(index)]
        row, onset = int(event.scenario_row), int(event.local_onset_frame)
        history = np.asarray(
            self.arrays["agent_states"][row, onset - 24 : onset + 1]
        ).copy()
        valid = np.asarray(
            self.arrays["agent_valid"][row, onset - 24 : onset + 1]
        ).copy()
        features = policy_features(
            history,
            valid,
            self.metadata["lengths_m"][row],
            self.metadata["widths_m"][row],
            self.arrays["map_polylines"][row],
            self.arrays["map_polyline_valid"][row],
        )
        start = onset - 24
        target = (
            np.asarray(self.arrays["target_controls"][row, start : start + 5])
            .mean(0)
            .copy()
        )
        base = {
            "features": torch.from_numpy(features),
            "history_valid": torch.from_numpy(valid),
            "target": torch.from_numpy(target),
            "target_valid": torch.from_numpy(valid[-1, 1:].copy()),
            "scenario_row": torch.tensor(row, dtype=torch.long),
            "decision_offset": torch.tensor(start, dtype=torch.long),
        }
        return base, row, onset

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        base, _, _ = self._base(int(index))
        return base


class EventAnchoredActionBlockDataset(EventAnchoredBCDataset):
    """D1 event-onset view with the same 15-decision target as Idea B."""

    block_decisions = 15

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        base, row, onset = self._base(int(index))
        active = np.asarray(self.arrays["agent_valid"][row, onset, 1:]).copy()
        block = np.zeros((self.block_decisions, 6, 2), np.float32)
        block_valid = np.zeros((self.block_decisions, 6), bool)
        future_states = np.zeros((self.block_decisions, 6, 6), np.float32)
        control_start = onset - 24
        for local in range(self.block_decisions):
            start = control_start + local * 5
            block[local] = np.asarray(
                self.arrays["target_controls"][row, start : start + 5]
            ).mean(0)
            future_states[local] = np.asarray(
                self.arrays["agent_states"][row, onset + (local + 1) * 5, 1:]
            )
            block_valid[local] = active
        base.update(
            {
                "action_block": torch.from_numpy(block),
                "action_block_valid": torch.from_numpy(block_valid),
                "current_background_state": torch.from_numpy(
                    np.asarray(self.arrays["agent_states"][row, onset, 1:]).copy()
                ),
                "future_background_states": torch.from_numpy(future_states),
            }
        )
        return base


class EventAnchoredActionSelfRolloutDataset(EventAnchoredActionBlockDataset):
    """D1 event-onset action blocks with state needed by B2 self-unfolding."""

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        base = super().__getitem__(index)
        row = int(base["scenario_row"])
        onset = 24 + int(base["decision_offset"])
        base.update(
            {
                "raw_history": torch.from_numpy(
                    np.asarray(
                        self.arrays["agent_states"][row, onset - 24 : onset + 1]
                    ).copy()
                ),
                "next_ego_states": torch.from_numpy(
                    np.asarray(
                        self.arrays["agent_states"][row, onset + 1 : onset + 6, 0]
                    ).copy()
                ),
                "lengths": torch.from_numpy(self.metadata["lengths_m"][row].copy()),
                "widths": torch.from_numpy(self.metadata["widths_m"][row].copy()),
                "map_polylines": torch.from_numpy(
                    np.asarray(self.arrays["map_polylines"][row]).copy()
                ),
                "map_valid": torch.from_numpy(
                    np.asarray(self.arrays["map_polyline_valid"][row]).copy()
                ),
            }
        )
        return base


def event_behavior_condition(event: pd.Series) -> dict[str, int | str] | None:
    """Map a train-only D1 annotation to a requested T4a behavior.

    The condition is a coarse semantic request, not a future trajectory.  It
    is deliberately absent when the annotated role is ego-only, because T4a
    controls NPCs rather than leaking a goal through the ego slot.
    """
    event_type = str(event.event_type)
    response = int(event.response_agent_index)
    stimulus = int(event.stimulus_agent_index)
    if event_type == "lane_change" and response > 0:
        direction = str(event.direction)
        if direction in {"left", "right"}:
            return {"target_agent_index": response, "mode": f"lane_{direction}"}
    if event_type == "cut_out" and response > 0:
        return {"target_agent_index": response, "mode": "recover"}
    if event_type in {"cut_in", "following_brake"}:
        target = stimulus if stimulus > 0 else response
        if target > 0:
            return {"target_agent_index": target, "mode": "brake"}
    return None


class ConditionedActionBlockDataset(CausalActionBlockDataset):
    """D0 natural action blocks with an explicit all-zero T4a condition."""

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        base = super().__getitem__(index)
        base["features"] = torch.from_numpy(
            append_behavior_condition(base["features"].numpy(), None)
        )
        base["condition_available"] = torch.tensor(False)
        # ``ConditionedEventActionBlockDataset`` carries this bookkeeping
        # field for every D1 example.  D0 examples intentionally have no
        # requested behavior, but ConcatDataset still requires an identical
        # mapping schema for the default DataLoader collate function.
        base["condition_target_agent_index"] = torch.tensor(-1, dtype=torch.long)
        return base


class ConditionedEventActionBlockDataset(EventAnchoredActionBlockDataset):
    """D1 action blocks paired with a train-only semantic behavior request."""

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        base = super().__getitem__(index)
        condition = event_behavior_condition(self.events.iloc[int(index)])
        base["features"] = torch.from_numpy(
            append_behavior_condition(base["features"].numpy(), condition)
        )
        base["condition_available"] = torch.tensor(condition is not None)
        base["condition_target_agent_index"] = torch.tensor(
            -1 if condition is None else condition["target_agent_index"],
            dtype=torch.long,
        )
        return base


def estimate_action_statistics(dataset: CausalBCDataset) -> dict[str, np.ndarray]:
    total = np.zeros(2, np.float64)
    square = np.zeros(2, np.float64)
    count = 0
    for start in range(0, len(dataset.rows), 512):
        rows = dataset.rows[start : start + 512]
        controls = np.asarray(dataset.arrays["target_controls"][rows], np.float64)
        valid = np.asarray(dataset.arrays["agent_valid"][rows, 24:-1, 1:])
        values = controls[valid]
        total += values.sum(0)
        square += np.square(values).sum(0)
        count += len(values)
    mean = total / max(count, 1)
    std = np.sqrt(np.maximum(square / max(count, 1) - np.square(mean), 1.0e-6))
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "count": np.asarray(count),
    }


def estimate_position_block_statistics(
    dataset: CausalActionBlockDataset,
) -> dict[str, np.ndarray]:
    """Fit train-only per-horizon relative-position normalization for B3."""
    horizon = dataset.block_decisions
    total = np.zeros((horizon, 2), np.float64)
    square = np.zeros_like(total)
    count = np.zeros(horizon, np.int64)
    rows = np.asarray(dataset.rows, np.int64)
    for begin in range(0, len(rows), 512):
        take = rows[begin : begin + 512]
        offset_index = (take * 17 + dataset.seed) % (len(DECISION_OFFSETS) - 1)
        start_frames = DECISION_OFFSETS[offset_index]
        current_index = 24 + start_frames
        current = np.asarray(
            dataset.arrays["agent_states"][take, current_index, 1:, :2], np.float64
        )
        active = np.asarray(
            dataset.arrays["agent_valid"][take, current_index, 1:], bool
        )
        for local in range(horizon):
            decision = offset_index + local
            available = decision < len(DECISION_OFFSETS) - 1
            if not available.any():
                continue
            selected_rows = take[available]
            endpoint = 24 + DECISION_OFFSETS[decision[available]] + 5
            future = np.asarray(
                dataset.arrays["agent_states"][selected_rows, endpoint, 1:, :2],
                np.float64,
            )
            values = future - current[available]
            mask = active[available]
            selected = values[mask]
            total[local] += selected.sum(0)
            square[local] += np.square(selected).sum(0)
            count[local] += len(selected)
    mean = total / np.maximum(count[:, None], 1)
    std = np.sqrt(
        np.maximum(square / np.maximum(count[:, None], 1) - np.square(mean), 1.0e-6)
    )
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "count": count,
    }


def estimate_feature_statistics(
    dataset: CausalBCDataset, maximum_rows: int = 0
) -> dict[str, np.ndarray]:
    """Fit normalization on train prefixes only, in streaming float64."""
    rows = dataset.rows[: int(maximum_rows)] if maximum_rows else dataset.rows
    total = np.zeros(12, np.float64)
    square = np.zeros(12, np.float64)
    count = 0
    for row in rows:
        history = np.asarray(dataset.arrays["agent_states"][row, :25])
        valid = np.asarray(dataset.arrays["agent_valid"][row, :25])
        values = policy_features(
            history,
            valid,
            dataset.metadata["lengths_m"][row],
            dataset.metadata["widths_m"][row],
            dataset.arrays["map_polylines"][row],
            dataset.arrays["map_polyline_valid"][row],
        )[valid]
        total += values.sum(0)
        square += np.square(values).sum(0)
        count += len(values)
    mean = total / max(count, 1)
    std = np.sqrt(np.maximum(square / max(count, 1) - np.square(mean), 1.0e-6))
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "count": np.asarray(count),
    }

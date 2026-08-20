"""Canonical highD response-boundary data for the hierarchical model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from diffusion.src.data import (
    ANCHOR_INDEX,
    DataBundle,
    load_data_bundle,
    pilot_rows,
    split_rows,
)
from world_model.src.core.dynamics import KinematicTrafficDynamics


def ego_controls(source: np.ndarray, target: np.ndarray, dt_s: float) -> np.ndarray:
    """Recover realized `[a, yaw_rate]` controls from adjacent ego states."""
    source_speed = np.linalg.norm(source[..., 2:4], axis=-1)
    target_speed = np.linalg.norm(target[..., 2:4], axis=-1)
    acceleration = (target_speed - source_speed) / float(dt_s)
    source_heading = np.arctan2(source[..., 3], np.maximum(source[..., 2], 1.0e-4))
    target_heading = np.arctan2(target[..., 3], np.maximum(target[..., 2], 1.0e-4))
    difference = np.arctan2(
        np.sin(target_heading - source_heading),
        np.cos(target_heading - source_heading),
    )
    return np.stack((acceleration, difference / float(dt_s)), axis=-1).astype(
        np.float32
    )


@dataclass(frozen=True)
class ExperimentData:
    bundle: DataBundle
    train_rows: np.ndarray
    validation_rows: np.ndarray
    test_rows: np.ndarray
    diffusion_contract: dict[str, Any]


class ResponseDataset(Dataset):
    """One causal 25 Hz response boundary sampled from each sequence."""

    def __init__(
        self,
        bundle: DataBundle,
        rows: np.ndarray,
        *,
        training: bool,
        seed: int,
        history_choices: tuple[int, ...] = (5, 10, 15, 25),
        preview_frames: int = 25,
        execute_frames: int = 1,
        soft_plans: np.ndarray | None = None,
        response_calibrator: Any | None = None,
    ) -> None:
        self.bundle = bundle
        self.rows = np.asarray(rows, np.int64)
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0
        self.history_choices = tuple(int(value) for value in history_choices)
        self.preview_frames = int(preview_frames)
        self.execute_frames = int(execute_frames)
        if self.execute_frames != 1:
            raise ValueError("the maintained response dataset samples one 25 Hz frame")
        self.soft_plans = (
            None if soft_plans is None else np.asarray(soft_plans, np.float32)
        )
        self.response_calibrator = response_calibrator
        if self.soft_plans is not None and len(self.soft_plans) != len(self.rows):
            raise ValueError("soft_plans must align one-to-one with rows")
        self.responses = 149
        self.closed_loop_frames = 25

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _selection(self, item: int) -> tuple[int, int, bool]:
        sequence = np.random.SeedSequence((self.seed, self.epoch, int(item)))
        generator = np.random.default_rng(sequence)
        response = (
            int(generator.integers(self.responses - self.closed_loop_frames + 1))
            if self.training
            else int(item) % (self.responses - self.closed_loop_frames + 1)
        )
        history = (
            int(generator.choice(self.history_choices))
            if self.training
            else max(self.history_choices)
        )
        mask_older_history = bool(generator.random() < 0.5) if self.training else False
        return response, history, mask_older_history

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = int(self.rows[int(item)])
        response, history_count, mask_older_history = self._selection(int(item))
        start = response * self.execute_frames
        current_index = ANCHOR_INDEX + start
        arrays = self.bundle.arrays
        states = np.asarray(arrays["agent_states"][row], np.float32)
        valid = np.asarray(arrays["agent_valid"][row], bool)
        history_start = max(0, current_index - history_count + 1)
        history = states[history_start : current_index + 1].copy()
        history_valid = valid[history_start : current_index + 1].copy()
        committed_ego_controls = ego_controls(
            states[history_start:current_index, 0],
            states[history_start + 1 : current_index + 1, 0],
            dt_s=0.04,
        )
        if mask_older_history and len(history) > 5:
            history[:-5] = 0.0
            history_valid[:-5] = False
            committed_ego_controls[:-4] = 0.0
        current = states[current_index].copy()
        current_valid = valid[current_index].copy()
        target_states = states[
            current_index + 1 : current_index + 1 + self.execute_frames, 1:
        ].copy()
        source_background = states[
            current_index : current_index + self.execute_frames, 1:
        ].copy()
        target_highd = np.asarray(
            arrays["actions_highd"][
                row, start : start + self.execute_frames
            ],
            np.float32,
        ).copy()
        target_actions = KinematicTrafficDynamics.controls_from_highd_actions(
            torch.from_numpy(target_highd), torch.from_numpy(source_background)
        ).numpy()
        closed_stop = current_index + 1 + self.closed_loop_frames
        closed_target_states = states[current_index + 1 : closed_stop, 1:].copy()
        closed_source = states[current_index : closed_stop - 1, 1:].copy()
        closed_highd = np.asarray(
            arrays["actions_highd"][row, start : start + self.closed_loop_frames],
            np.float32,
        ).copy()
        closed_target_actions = KinematicTrafficDynamics.controls_from_highd_actions(
            torch.from_numpy(closed_highd), torch.from_numpy(closed_source)
        ).numpy()
        closed_ego_actions = ego_controls(
            states[current_index : closed_stop - 1, 0],
            states[current_index + 1 : closed_stop, 0],
            dt_s=0.04,
        )
        previous_actions = KinematicTrafficDynamics.controls_from_highd_actions(
            torch.from_numpy(current[1:, 4:6]), torch.from_numpy(current[1:])
        ).numpy()
        full_background = states[ANCHOR_INDEX:174, 1:]
        if self.soft_plans is None:
            raise RuntimeError("training requires frozen diffusion soft previews")
        reference = self.soft_plans[int(item)]
        preview = reference[start : start + self.preview_frames]
        if len(preview) < self.preview_frames:
            preview = np.concatenate(
                (
                    preview,
                    np.repeat(preview[-1:], self.preview_frames - len(preview), axis=0),
                ),
                axis=0,
            )
        reference_base = (
            full_background[0, :, :2] if start == 0 else reference[start - 1]
        )
        response_bounds = (
            np.zeros((2, 6, 2), np.float32)
            if self.response_calibrator is None
            else self.response_calibrator.bounds_for(current, current_valid)
        )
        return {
            "history": torch.from_numpy(history),
            "history_valid": torch.from_numpy(history_valid),
            "committed_ego_controls": torch.from_numpy(committed_ego_controls),
            "current": torch.from_numpy(current),
            "current_valid": torch.from_numpy(current_valid),
            "target_actions": torch.from_numpy(target_actions),
            "target_states": torch.from_numpy(target_states),
            "closed_target_states": torch.from_numpy(closed_target_states),
            "closed_target_actions": torch.from_numpy(closed_target_actions),
            "closed_ego_actions": torch.from_numpy(closed_ego_actions),
            "previous_actions": torch.from_numpy(previous_actions),
            "soft_reference": torch.from_numpy(preview.copy()),
            "reference_base": torch.from_numpy(reference_base.copy()),
            "map_polylines": torch.from_numpy(
                np.asarray(arrays["map_polylines"][row], np.float32).copy()
            ),
            "map_polyline_valid": torch.from_numpy(
                np.asarray(arrays["map_polyline_valid"][row], bool).copy()
            ),
            "natural_response_bounds": torch.from_numpy(response_bounds),
            "row_index": torch.tensor(row),
            "response_index": torch.tensor(response),
        }


def response_collate(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Left-pad variable realized histories without fabricating observations."""
    frames = max(int(row["history"].shape[0]) for row in rows)
    histories: list[torch.Tensor] = []
    validity: list[torch.Tensor] = []
    for row in rows:
        count = int(row["history"].shape[0])
        pad = frames - count
        histories.append(torch.cat((torch.zeros(pad, 7, 6), row["history"]), dim=0))
        validity.append(
            torch.cat(
                (torch.zeros(pad, 7, dtype=torch.bool), row["history_valid"]), dim=0
            )
        )
    output = {
        name: torch.stack([row[name] for row in rows])
        for name in rows[0]
        if name not in {"history", "history_valid", "committed_ego_controls"}
    }
    output["history"] = torch.stack(histories)
    output["history_valid"] = torch.stack(validity)
    control_frames = max(frames - 1, 1)
    controls: list[torch.Tensor] = []
    for row in rows:
        value = row["committed_ego_controls"]
        controls.append(torch.cat((torch.zeros(control_frames - len(value), 2), value)))
    output["committed_ego_controls"] = torch.stack(controls)
    return output


def response_loader(
    dataset: ResponseDataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator if shuffle else None,
        num_workers=max(0, int(workers)),
        persistent_workers=int(workers) > 0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=response_collate,
    )


def prepare_experiment_data(config: dict[str, Any], config_dir: Path) -> ExperimentData:
    bundle = load_data_bundle(config, config_dir)
    training = config["training"]
    seed = int(training["seed"])
    split_names = {"train": "train", "validation": "val", "test": "test"}
    scope = str(training.get("experiment_scope", "full"))
    if scope == "full":
        rows = {
            split: split_rows(bundle.arrays, source, seed=seed)
            for split, source in split_names.items()
        }
    elif scope == "pilot":
        limits = {
            "train": int(training.get("pilot_train_sequences", 4_096)),
            "validation": int(training.get("pilot_validation_sequences", 1_024)),
            "test": int(training.get("pilot_test_sequences", 1_024)),
        }
        rows = {
            split: pilot_rows(bundle, source, maximum=limits[split], seed=seed)
            for split, source in split_names.items()
        }
    else:
        raise ValueError("experiment_scope must be 'full' or 'pilot'")
    return ExperimentData(
        bundle=bundle,
        train_rows=rows["train"],
        validation_rows=rows["validation"],
        test_rows=rows["test"],
        diffusion_contract={},
    )

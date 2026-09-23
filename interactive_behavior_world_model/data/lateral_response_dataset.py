"""Train/validation pairs for lateral response-sensitivity preservation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .causal_cache import load_causal_cache
from .manifest import load_benchmark_config


def _advance_stimulus(
    state: np.ndarray,
    target_y_m: float,
    controller_rate_rps: float,
    frames: int,
    *,
    intervene: bool,
) -> np.ndarray:
    """NumPy equivalent of the shared unicycle lateral-stimulus rollout."""
    current = np.asarray(state, np.float32).copy()
    output = np.empty((frames, 6), np.float32)
    for frame in range(frames):
        x, y, vx, vy = (float(current[index]) for index in range(4))
        speed = max(float(np.hypot(vx, vy)), 1.0e-4)
        safe_vx = 1.0e-4 if abs(vx) < 1.0e-4 else vx
        heading = float(np.arctan2(vy, safe_vx))
        yaw_rate = 0.0
        if intervene:
            control_speed = max(speed, 1.0)
            yaw_rate = (
                controller_rate_rps**2 * (target_y_m - y) / control_speed
                - 2.0 * controller_rate_rps * heading
            )
            yaw_rate = float(
                np.clip(
                    yaw_rate,
                    -min(0.6, 4.0 / control_speed),
                    min(0.6, 4.0 / control_speed),
                )
            )
        next_heading = heading + yaw_rate * 0.04
        current = np.asarray(
            [
                x + speed * np.cos(heading) * 0.04,
                y + speed * np.sin(heading) * 0.04,
                speed * np.cos(next_heading),
                speed * np.sin(next_heading),
                -speed * yaw_rate * np.sin(next_heading),
                speed * yaw_rate * np.cos(next_heading),
            ],
            np.float32,
        )
        output[frame] = current
    return output


class LateralResponsePairDataset(Dataset):
    """D3 lateral histories with matched natural/intervention branches.

    Each anchor is exposed once per epoch.  Dose and observation delay rotate
    deterministically, so train sampling covers the Cartesian product without
    tripling the number of optimizer steps.  Validation is fixed and repeatable.
    """

    doses = (0.6, 0.9, 1.2)
    observation_frames = (5, 10, 15, 20)

    def __init__(
        self, config_path: str | Path, split: str, *, seed: int = 20260919
    ) -> None:
        config, _ = load_benchmark_config(config_path)
        self.arrays, self.metadata, _ = load_causal_cache(config_path)
        split_index = {"train": 0, "validation": 1, "val": 1}[split]
        manifest = pd.read_csv(
            Path(config["paths"]["output_dir"]) / "lateral_probe_manifest.csv"
        )
        self.probes = manifest[manifest.split_index == split_index].reset_index(
            drop=True
        )
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.probes)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        probe = self.probes.iloc[int(index)]
        # Coprime rotations avoid coupling family/recording order to condition.
        dose_index = (int(index) * 5 + self.epoch + self.seed) % len(self.doses)
        delay_index = (int(index) * 7 + self.epoch * 3 + self.seed) % len(
            self.observation_frames
        )
        dose = float(self.doses[dose_index])
        frames = int(self.observation_frames[delay_index])
        row = int(probe.scenario_row)
        stimulus = int(probe.stimulus_agent_index)
        response = int(probe.response_agent_index)
        original_history = np.asarray(self.arrays["agent_states"][row, :25]).copy()
        active = np.asarray(self.arrays["agent_valid"][row, 24]).copy()
        logged_tail = np.asarray(
            self.arrays["agent_states"][row, 25 : 25 + frames]
        ).copy()
        baseline_stimulus = _advance_stimulus(
            original_history[-1, stimulus],
            float(probe.target_lane_center_y_m),
            dose,
            frames,
            intervene=False,
        )
        changed_stimulus = _advance_stimulus(
            original_history[-1, stimulus],
            float(probe.target_lane_center_y_m),
            dose,
            frames,
            intervene=True,
        )
        baseline_tail = logged_tail.copy()
        baseline_tail[:, stimulus] = baseline_stimulus
        changed_tail = logged_tail.copy()
        changed_tail[:, stimulus] = changed_stimulus
        baseline_history = np.concatenate(
            (original_history[frames:], baseline_tail), axis=0
        )
        changed_history = np.concatenate(
            (original_history[frames:], changed_tail), axis=0
        )
        valid_history = np.concatenate(
            (
                np.asarray(self.arrays["agent_valid"][row, frames:25]).copy(),
                np.broadcast_to(active, (frames, 7)).copy(),
            ),
            axis=0,
        )
        side = float(
            np.sign(
                original_history[-1, stimulus, 1] - original_history[-1, response, 1]
            )
        )
        avoidance_sign = -side if side else 1.0
        family = 0 if str(probe.stimulus_family) == "merge" else 1
        values = {
            "baseline_history": baseline_history,
            "changed_history": changed_history,
            "history_valid": valid_history,
            "response_index": np.asarray(response, np.int64),
            "stimulus_index": np.asarray(stimulus, np.int64),
            "family": np.asarray(family, np.int64),
            "avoidance_sign": np.asarray(avoidance_sign, np.float32),
            "dose": np.asarray(dose, np.float32),
            "observation_frames": np.asarray(frames, np.int64),
            "lengths": self.metadata["lengths_m"][row].copy(),
            "widths": self.metadata["widths_m"][row].copy(),
            "map_polylines": np.asarray(self.arrays["map_polylines"][row]).copy(),
            "map_valid": np.asarray(self.arrays["map_polyline_valid"][row]).copy(),
        }
        return {key: torch.from_numpy(value) for key, value in values.items()}

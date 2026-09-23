"""Dataset view of cached MA-IDM response pairs."""

from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from .causal_cache import load_causal_cache
from .manifest import load_benchmark_config


class ResponsePairDataset(Dataset):
    def __init__(self, config_path: str | Path, split: str):
        config, _ = load_benchmark_config(config_path)
        self.arrays, self.metadata, _ = load_causal_cache(config_path)
        pair_path = (
            Path(config["paths"]["output_dir"])
            / "data/response_pairs_ma_idm_train_v3.npz"
        )
        with np.load(pair_path, allow_pickle=False) as data:
            self.pairs = {k: np.asarray(data[k]) for k in data.files}
        validation = np.asarray(
            [
                int.from_bytes(hashlib.sha256(x.encode()).digest()[:2], "little") % 10
                == 0
                for x in self.pairs["event_id"].astype(str)
            ]
        )
        self.indices = np.flatnonzero(
            validation if split in {"validation", "val"} else ~validation
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        pair = int(self.indices[index])
        row = int(self.pairs["scenario_row"][pair])
        onset = int(self.pairs["local_onset_frame"][pair])
        values = {
            "history": np.asarray(
                self.arrays["agent_states"][row, onset - 24 : onset + 1]
            ).copy(),
            "history_valid": np.asarray(
                self.arrays["agent_valid"][row, onset - 24 : onset + 1]
            ).copy(),
            "logged_future": np.asarray(
                self.arrays["agent_states"][row, onset + 1 : onset + 76]
            ).copy(),
            "intervention_stimulus": self.pairs["intervention_stimulus_states"][
                pair
            ].copy(),
            "stimulus_index": np.asarray(self.pairs["stimulus_agent_index"][pair]),
            "response_index": np.asarray(self.pairs["response_agent_index"][pair]),
            "lengths": self.metadata["lengths_m"][row].copy(),
            "widths": self.metadata["widths_m"][row].copy(),
            "map_polylines": np.asarray(self.arrays["map_polylines"][row]).copy(),
            "map_valid": np.asarray(self.arrays["map_polyline_valid"][row]).copy(),
            "teacher_delta": self.pairs["teacher_response_delta"][pair].copy(),
            "teacher_reactive": self.pairs["teacher_reactive_features"][pair].copy(),
        }
        return {key: torch.from_numpy(value) for key, value in values.items()}

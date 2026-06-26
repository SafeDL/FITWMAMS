"""Array normalization helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, axis=None, eps: float = 1e-6) -> "Normalizer":
        mean = np.mean(x, axis=axis, keepdims=False).astype(np.float32)
        std = np.std(x, axis=axis, keepdims=False).astype(np.float32)
        std = np.where(std < eps, 1.0, std).astype(np.float32)
        return cls(mean=mean, std=std)

    def encode(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict[str, list]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, list]) -> "Normalizer":
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
        )


def fit_dataset_normalizers(
    scenario_conditions: np.ndarray,
    actions: np.ndarray,
    train_mask: np.ndarray,
) -> dict[str, dict[str, list]]:
    idx = np.asarray(train_mask, dtype=bool)
    if not np.any(idx):
        raise RuntimeError("Cannot fit diffusion normalizers without train split samples")
    return {
        "scenario_conditions": Normalizer.fit(scenario_conditions[idx], axis=0).to_dict(),
        "actions": Normalizer.fit(actions[idx], axis=(0, 1)).to_dict(),
    }


def apply_normalizers(
    arrays: dict[str, np.ndarray],
    stats: dict[str, dict[str, list]],
) -> dict[str, np.ndarray]:
    required_arrays = ("scenario_conditions", "actions")
    missing_arrays = [key for key in required_arrays if key not in arrays]
    if missing_arrays:
        raise KeyError(f"Diffusion dataset is missing arrays: {missing_arrays}")
    missing_stats = [key for key in required_arrays if key not in stats]
    if missing_stats:
        raise KeyError(f"Diffusion normalization stats are missing keys: {missing_stats}")
    out = dict(arrays)
    out["scenario_conditions"] = Normalizer.from_dict(
        stats["scenario_conditions"]
    ).encode(out["scenario_conditions"])
    out["actions"] = Normalizer.from_dict(stats["actions"]).encode(out["actions"])
    return out

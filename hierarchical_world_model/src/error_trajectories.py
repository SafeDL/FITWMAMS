"""Frozen whole-window pose errors and deterministic two-vehicle quadrature."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

GROUPS = (
    "history",
    "history_braking",
    "history_lateral",
    "history_braking_lateral",
    "cold",
    "cold_lateral",
    "tracking",
    "tracking_lane_change",
)
DT = 0.04
HORIZON = 1.5
STEPS = int(HORIZON / DT)
SAMPLES = 16
REPRESENTATIVES = 32
MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "results/hierarchical_world_model/model_assets/interaction_errors.npz"
)


class ErrorTrajectoryBank(nn.Module):
    def __init__(self, trajectories, weights, points, *, dt=DT):
        super().__init__()
        errors = torch.as_tensor(trajectories, dtype=torch.float32)
        weights = torch.as_tensor(weights, dtype=torch.float32)
        points = torch.as_tensor(points, dtype=torch.float32)
        if (
            errors.ndim != 4
            or errors.shape[:2] != (8, 32)
            or errors.shape[-1] != 3
            or weights.shape != (8, 32)
            or points.shape != (16, 2)
            or dt != DT
        ):
            raise ValueError("Invalid empirical pose-trajectory bank")
        if (
            not torch.isfinite(errors).all()
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
            or not torch.allclose(weights.sum(-1), torch.ones(8))
        ):
            raise ValueError("Invalid representative errors or sample weights")
        cumulative = weights.cumsum(-1).contiguous()
        index = torch.searchsorted(
            cumulative, points.T[None].expand(8, -1, -1).reshape(8, -1).contiguous()
        ).clamp_max(31)
        selected = errors.gather(
            1, index[..., None, None].expand(-1, -1, errors.shape[2], 3)
        )
        self.register_buffer("samples", selected.reshape(8, 2, 16, errors.shape[2], 3))
        self.dt = dt

    @classmethod
    def nominal(cls, steps=STEPS):
        return cls(
            np.zeros((8, 32, steps, 3), np.float32),
            np.full((8, 32), 1 / 32, np.float32),
            np.column_stack((np.arange(16) / 16, (np.arange(16) + 0.5) / 16)),
        )

    @classmethod
    def load(cls, path):
        data = _read_bank(str(Path(path).resolve()))
        return cls(*data[:3], dt=data[3])


@lru_cache(maxsize=4)
def _read_bank(path):
    with np.load(path, allow_pickle=False) as data:
        if str(data["protocol"]) != "observed_lateral_trend_direction_normalized":
            raise ValueError("Empirical errors use a different motion forecast")
        if data["groups"].tolist() != list(GROUPS):
            raise ValueError("Empirical trajectory groups do not match execution")
        return (
            data["trajectories"],
            data["weights"],
            data["points"],
            float(data["dt_s"]),
        )

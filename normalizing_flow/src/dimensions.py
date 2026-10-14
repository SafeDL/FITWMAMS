"""Recorded vehicle sizes and their Diffusion normalization."""

from __future__ import annotations

import numpy as np

DIMENSION_FEATURES = ("length_m", "width_m")
DIMENSION_DIM = 14


def dimensions_valid(slot_mask: np.ndarray) -> np.ndarray:
    slots = np.asarray(slot_mask, bool)
    agents = np.concatenate((np.ones(slots.shape[:-1] + (1,), bool), slots), axis=-1)
    return np.broadcast_to(agents[..., None], agents.shape + (2,))


def fit_dimension_contract(
    dimensions: np.ndarray, slot_mask: np.ndarray, split_index: np.ndarray
) -> dict[str, object]:
    """Fit the Diffusion size-normalization contract on training recordings."""
    values = np.asarray(dimensions, np.float32)
    valid = dimensions_valid(slot_mask)
    selected = valid & (np.asarray(split_index) == 0)[:, None, None]
    if not np.isfinite(values[valid]).all() or (values[valid] <= 0).any():
        raise ValueError("recorded vehicle dimensions must be finite and positive")
    mean = np.zeros((7, 2), np.float32)
    std = np.ones((7, 2), np.float32)
    lower = np.zeros((7, 2), np.float32)
    upper = np.zeros((7, 2), np.float32)
    for agent in range(7):
        observed = values[selected[:, agent, 0], agent]
        mean[agent] = np.log(observed).mean(axis=0)
        std[agent] = np.maximum(np.log(observed).std(axis=0), 0.01)
        lower[agent] = observed.min(axis=0)
        upper[agent] = observed.max(axis=0)
    return {
        "shape_per_sequence": [7, 2],
        "feature_names": list(DIMENSION_FEATURES),
        "transform": "log",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "minimum_m": lower.tolist(),
        "maximum_m": upper.tolist(),
        "fit_split": "train",
    }


def normalize_dimensions(dimensions, slot_mask, contract):
    valid = dimensions_valid(slot_mask)
    values = np.asarray(dimensions, np.float32)
    if not np.isfinite(values[valid]).all() or (values[valid] <= 0).any():
        raise ValueError("active vehicle dimensions must be finite and positive")
    logged = np.log(np.where(valid, values, 1.0))
    normalized = (logged - np.asarray(contract["mean"])) / np.asarray(contract["std"])
    return np.where(valid, normalized, 0.0).astype(np.float32)

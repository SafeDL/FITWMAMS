"""Public sampling boundary for the hierarchical highD scenario density."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .data import load_natural_dataset
from .transforms import inverse_transform_model_features, transform_features_for_model


def inverse_normalize_features(x_norm: np.ndarray, schema: dict[str, Any]) -> np.ndarray:
    norm = schema["normalization"]
    model = np.asarray(x_norm, np.float32) * np.asarray(norm["std"], np.float32)
    model += np.asarray(norm["mean"], np.float32)
    return inverse_transform_model_features(
        model,
        list(schema["feature_names"]),
        list(schema.get("model_feature_transforms", [])) or None,
    )


def normalize_features(raw: np.ndarray, valid: np.ndarray, schema: dict[str, Any]) -> np.ndarray:
    norm = schema["normalization"]
    mean = np.asarray(norm["mean"], np.float32)
    std = np.asarray(norm["std"], np.float32)
    model = transform_features_for_model(
        np.asarray(raw, np.float32),
        np.asarray(valid, bool),
        list(schema["feature_names"]),
        list(schema.get("model_feature_transforms", [])) or None,
    )
    output = np.zeros_like(model, dtype=np.float32)
    present = np.asarray(valid, bool)
    output[present] = ((model - mean) / std)[present]
    return output


def load_checkpoint_and_dataset(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    repo_root: str | Path,
    device: str | Any = "cpu",
):
    from .scenario import load_checkpoint

    model, payload = load_checkpoint(
        checkpoint, repo_root=repo_root, device=device
    )
    arrays, schema = load_natural_dataset(output_dir)
    return model, arrays, schema, payload


def sample_scenarios(
    model,
    n: int,
    scenario_seed: int,
):
    """Sample the complete scenario condition used by trajectory diffusion."""
    return model.sample_scenarios(int(n), int(scenario_seed))


def sample_constraints(
    model,
    c0: np.ndarray,
    slot_mask: np.ndarray,
    n: int,
    scenario_seed: int,
):
    """Sample multiple state constraints while holding an initial state fixed."""
    return model.sample_constraints(
        c0,
        slot_mask,
        int(n),
        int(scenario_seed),
    )


def log_prob(model, c0, slot_mask, k) -> dict[str, np.ndarray]:
    """Return all three probability factors and their exact sum."""
    return model.log_prob(c0, slot_mask, k)

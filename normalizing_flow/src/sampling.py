"""Feature normalization and joint loading for scenario Flow evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .data import load_scenario_dataset
from .transforms import inverse_transform_model_features, transform_features_for_model


def inverse_normalize_features(
    x_norm: np.ndarray, schema: dict[str, Any]
) -> np.ndarray:
    norm = schema["normalization"]
    model = np.asarray(x_norm, np.float32) * np.asarray(norm["std"], np.float32)
    model += np.asarray(norm["mean"], np.float32)
    return inverse_transform_model_features(
        model,
        list(schema["model_feature_transforms"]),
    )


def normalize_features(
    raw: np.ndarray, valid: np.ndarray, schema: dict[str, Any]
) -> np.ndarray:
    norm = schema["normalization"]
    mean = np.asarray(norm["mean"], np.float32)
    std = np.asarray(norm["std"], np.float32)
    model = transform_features_for_model(
        np.asarray(raw, np.float32),
        np.asarray(valid, bool),
        list(schema["model_feature_transforms"]),
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

    model, payload = load_checkpoint(checkpoint, repo_root=repo_root, device=device)
    arrays, schema = load_scenario_dataset(output_dir)
    return model, arrays, schema, payload

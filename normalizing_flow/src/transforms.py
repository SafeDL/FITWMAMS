"""Feature-coordinate transforms used by the natural-driving density model."""

from __future__ import annotations

from typing import Sequence

import numpy as np

_EPS = 1.0e-4


def feature_transform_kinds(
    feature_names: Sequence[str],
) -> tuple[str, ...]:
    kinds: list[str] = []
    for name in feature_names:
        if name.endswith("_rel_x_m") and "_front_" in name:
            kinds.append("positive_log")
        elif name.endswith("_rel_x_m") and "_rear_" in name:
            kinds.append("negative_log")
        elif name.endswith("_rel_y_left_m") and name.startswith("left_"):
            kinds.append("positive_log")
        elif name.endswith("_rel_y_left_m") and name.startswith("right_"):
            kinds.append("negative_log")
        else:
            kinds.append("identity")
    return tuple(kinds)


def transform_features_for_model(
    raw_features: np.ndarray,
    feature_valid: np.ndarray,
    transform_kinds: Sequence[str],
) -> np.ndarray:
    """Map raw auditable features into unconstrained model coordinates."""

    raw = np.asarray(raw_features, dtype=np.float32)
    valid = np.asarray(feature_valid, dtype=bool)
    out = raw.copy()
    kinds = list(transform_kinds)
    for idx, kind in enumerate(kinds):
        rows = valid[:, idx] & np.isfinite(raw[:, idx])
        if kind == "positive_log":
            out[rows, idx] = np.log(np.maximum(raw[rows, idx], _EPS))
        elif kind == "negative_log":
            out[rows, idx] = np.log(np.maximum(-raw[rows, idx], _EPS))
    return out


def inverse_transform_model_features(
    model_features: np.ndarray,
    transform_kinds: Sequence[str],
) -> np.ndarray:
    """Map unconstrained model-coordinate features back to raw audit units."""

    model = np.asarray(model_features, dtype=np.float32)
    out = model.copy()
    kinds = list(transform_kinds)
    for idx, kind in enumerate(kinds):
        if kind == "positive_log":
            out[:, idx] = np.exp(np.clip(model[:, idx], -12.0, 12.0))
        elif kind == "negative_log":
            out[:, idx] = -np.exp(np.clip(model[:, idx], -12.0, 12.0))
    return out.astype(np.float32)

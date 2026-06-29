"""Feature-coordinate transforms used by the tail-event density model."""
from __future__ import annotations

from typing import Sequence

import numpy as np


_EPS = 1.0e-4


def feature_transform_kinds(
    feature_names: Sequence[str],
) -> tuple[str, ...]:
    kinds: list[str] = []
    for name in feature_names:
        if name.endswith("_min_ax_1s_mps2"):
            kinds.append("positive_mean_minus_min_ax_softplus")
        else:
            kinds.append("identity")
    return tuple(kinds)


def _softplus_inverse(x: np.ndarray) -> np.ndarray:
    positive = np.maximum(np.asarray(x, dtype=np.float32), _EPS)
    out = np.empty_like(positive, dtype=np.float32)
    large = positive > 20.0
    out[large] = positive[large]
    out[~large] = np.log(np.expm1(positive[~large]) + _EPS)
    return out


def _softplus(x: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(
        np.asarray(x, dtype=np.float32),
        nan=0.0,
        posinf=20.0,
        neginf=-20.0,
    )
    return np.logaddexp(finite, 0.0).astype(np.float32)


def _matching_mean_ax_name(min_ax_name: str) -> str:
    return min_ax_name.replace("_min_ax_1s_mps2", "_mean_ax_1s_mps2")


def transform_features_for_model(
    raw_features: np.ndarray,
    feature_valid: np.ndarray,
    feature_names: Sequence[str],
    transform_kinds: Sequence[str] | None = None,
) -> np.ndarray:
    """Map raw auditable features into unconstrained model coordinates."""

    raw = np.asarray(raw_features, dtype=np.float32)
    valid = np.asarray(feature_valid, dtype=bool)
    out = raw.copy()
    names = list(feature_names)
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    kinds = list(transform_kinds or feature_transform_kinds(names))
    for idx, kind in enumerate(kinds):
        if kind == "positive_mean_minus_min_ax_softplus":
            mean_idx = name_to_idx[_matching_mean_ax_name(names[idx])]
            rows = (
                valid[:, idx]
                & valid[:, mean_idx]
                & np.isfinite(raw[:, idx])
                & np.isfinite(raw[:, mean_idx])
            )
            gap = raw[rows, mean_idx] - raw[rows, idx]
            out[rows, idx] = _softplus_inverse(gap).astype(np.float32)
    return out


def inverse_transform_model_features(
    model_features: np.ndarray,
    feature_names: Sequence[str],
    transform_kinds: Sequence[str] | None = None,
) -> np.ndarray:
    """Map unconstrained model-coordinate features back to raw audit units."""

    model = np.asarray(model_features, dtype=np.float32)
    out = model.copy()
    names = list(feature_names)
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    kinds = list(transform_kinds or feature_transform_kinds(names))
    for idx, kind in enumerate(kinds):
        if kind == "positive_mean_minus_min_ax_softplus":
            mean_idx = name_to_idx[_matching_mean_ax_name(names[idx])]
            gap = _softplus(model[:, idx])
            out[:, idx] = (out[:, mean_idx] - gap).astype(np.float32)
    return out.astype(np.float32)

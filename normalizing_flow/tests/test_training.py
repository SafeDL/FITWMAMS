import numpy as np
import torch

from normalizing_flow.src.features import build_feature_schema
from normalizing_flow.src.train import _fold_coordinates, _mask_log_prob


def test_mask_density_uses_tail_support_and_selected_counts() -> None:
    arrays = {
        "mask_pattern": np.asarray([1, 1, 2, 3, 3, 4]),
        "is_evt_tail": np.asarray([True, True, True, True, True, False]),
    }
    value = _mask_log_prob(arrays, np.asarray([0, 2, 3]))
    probability = value.exp().numpy()
    assert np.isclose(probability[[1, 2, 3]].sum(), 1.0)
    assert torch.isneginf(value[4])


def test_mask_density_changes_when_selected_tail_counts_change() -> None:
    arrays = {
        "mask_pattern": np.asarray([1, 1, 2, 2]),
        "is_evt_tail": np.ones(4, bool),
    }
    balanced = _mask_log_prob(arrays, np.asarray([0, 2]))
    skewed = _mask_log_prob(arrays, np.asarray([0, 1, 2]))
    assert skewed[1] > balanced[1]
    assert skewed[2] < balanced[2]


def test_cross_validation_normalization_uses_only_fit_rows() -> None:
    names = build_feature_schema().feature_names
    features = np.zeros((3, len(names)), np.float32)
    features[:, 0] = (1.0, 3.0, 100.0)
    valid = np.zeros_like(features, bool)
    valid[:, :2] = True
    arrays = {
        "features": features,
        "feature_valid": valid,
        "trajectory_constraint": np.zeros((3, 6, 6), np.float32),
        "trajectory_constraint_valid": np.zeros((3, 6, 6), bool),
    }
    schema = {
        "feature_names": list(names),
        "normalization": {},
        "long_horizon_constraint": {},
    }
    transformed, fold_schema = _fold_coordinates(arrays, schema, np.asarray([0, 1]))
    assert fold_schema["normalization"]["mean"][0] == 2.0
    assert transformed["features_normalized"][2, 0] == 98.0

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from normalizing_flow.src import train
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


def test_natural_training_honors_configured_epochs(tmp_path, monkeypatch):
    features = np.random.default_rng(7).uniform(0.1, 1.0, (4, 26)).astype(np.float32)
    arrays = {
        "features": features,
        "feature_valid": np.ones_like(features, bool),
        "trajectory_constraint": np.zeros((4, 6, 6), np.float32),
        "trajectory_constraint_valid": np.ones((4, 6, 6), bool),
        "slot_mask": np.ones((4, 6), bool),
        "mask_pattern": np.full(4, 63),
        "vehicle_dimensions": np.full((4, 7, 2), (4.8, 1.8), np.float32),
        "split_index": np.array([0, 0, 1, 2]),
        "segment_id": np.array(["train_a", "train_b", "val", "test"]),
        "recording_id": np.array([1, 1, 2, 3]),
    }
    schema = {
        "feature_names": list(build_feature_schema().feature_names),
        "normalization": {},
        "long_horizon_constraint": {},
    }
    monkeypatch.setattr(train, "load_scenario_dataset", lambda path: (arrays, schema))
    flow_config = {
        "num_layers": 1,
        "hidden_features": 4,
        "num_bins": 2,
        "tail_bound": 4.0,
        "num_blocks": 1,
        "dropout_probability": 0.0,
    }
    config = {
        "seed": 7,
        "device": "cpu",
        "paths": {"dataset_dir": ".", "output_dir": "output"},
        "model": {"c0_flow": flow_config, "constraint_flow": flow_config},
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "grad_clip": 5.0,
        },
    }
    train.train_natural_flow(
        config, config_dir=tmp_path, repo_root=Path(__file__).resolve().parents[2]
    )
    output = tmp_path / "output"
    assert pd.read_csv(output / "training_history.csv").epoch.tolist() == [1, 2]
    checkpoint = torch.load(output / "checkpoints/scenario_flow.pt", weights_only=False)
    assert checkpoint["metrics"]["epochs"] == 2
    assert checkpoint["metrics"]["num_fit_events"] == 3
    assert checkpoint["schema"]["fit_event_ids"] == ["train_a", "train_b", "val"]

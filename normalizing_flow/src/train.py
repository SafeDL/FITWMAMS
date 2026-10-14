"""Training and recording-level cross-validation for the tail-scene Flow."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    _normalize_masked_array,
    add_inactive_reference_noise,
    apply_feature_normalizer,
    dataset_dir_from_config,
    evt_tail_indices,
    fit_feature_normalizer,
    load_scenario_dataset,
    output_dir_from_config,
)
from .constraints import constraint_to_model_coordinates
from .scenario import build_direct_model, save_checkpoint
from .utils import save_json, select_device, set_seed

LOGGER = logging.getLogger(__name__)


def _fold_coordinates(arrays, schema, fit_rows):
    """Fit preprocessing on a CV training fold and apply it to all rows."""
    fold_arrays = dict(arrays)
    names = tuple(schema["feature_names"])
    feature_norm = fit_feature_normalizer(
        arrays["features"], arrays["feature_valid"], fit_rows, names
    )
    fold_arrays["features_normalized"] = add_inactive_reference_noise(
        apply_feature_normalizer(
            arrays["features"], arrays["feature_valid"], feature_norm, names
        ),
        arrays["feature_valid"],
    )
    constraint_model = constraint_to_model_coordinates(
        arrays["trajectory_constraint"], arrays["features"]
    )
    constraint_norm, constraint_contract = _normalize_masked_array(
        constraint_model, arrays["trajectory_constraint_valid"], fit_rows
    )
    fold_arrays["trajectory_constraint_normalized"] = constraint_norm
    fold_schema = deepcopy(schema)
    fold_schema["normalization"].update(
        {
            "mean": np.asarray(feature_norm["mean"], float).tolist(),
            "std": np.asarray(feature_norm["std"], float).tolist(),
            "valid_train_count": np.asarray(
                feature_norm["valid_train_count"], int
            ).tolist(),
            "fit_population": "cross_validation_fit_fold",
        }
    )
    fold_schema["long_horizon_constraint"]["normalization"] = constraint_contract
    return fold_arrays, fold_schema


def _mask_log_prob(arrays: dict[str, np.ndarray], rows: np.ndarray):
    import torch

    counts = np.bincount(arrays["mask_pattern"][rows], minlength=64).astype(np.float64)
    tail = np.asarray(arrays["is_evt_tail"], bool)
    support = np.bincount(arrays["mask_pattern"][tail], minlength=64) > 0
    smoothed = counts + 0.5 * support
    probability = np.zeros(64, np.float64)
    probability[support] = smoothed[support] / smoothed[support].sum()
    log_probability = np.full(64, -np.inf, np.float32)
    log_probability[support] = np.log(probability[support]).astype(np.float32)
    return torch.from_numpy(log_probability)


def _loader(arrays, rows: np.ndarray, *, shuffle: bool, batch_size: int):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(
        torch.from_numpy(arrays["features_normalized"][rows]).float(),
        torch.from_numpy(arrays["slot_mask"][rows]).bool(),
        torch.from_numpy(arrays["trajectory_constraint_normalized"][rows]).float(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


def _epoch(model, loader, device, *, optimizer=None, grad_clip: float = 0.0):
    import torch

    training = optimizer is not None
    model.train(training)
    totals = {name: 0.0 for name in ("joint", "mask", "c0", "k")}
    count = 0
    for c0, slots, constraint in loader:
        c0 = c0.to(device, non_blocking=True)
        slots = slots.to(device, non_blocking=True)
        constraint = constraint.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            terms = model.log_prob_tensors(
                c0_normalized=c0,
                slot_mask=slots,
                constraint_normalized=constraint,
            )
            loss = -terms["joint_log_prob"].mean()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        count += len(c0)
        for short, key in (
            ("joint", "joint_log_prob"),
            ("mask", "mask_log_prob"),
            ("c0", "c0_log_prob"),
            ("k", "k_log_prob"),
        ):
            totals[short] += float((-terms[key]).sum().detach().cpu())
    return {f"{name}_nll": value / max(count, 1) for name, value in totals.items()}


def _build_model(arrays, schema, rows, model_cfg, repo_root, device):
    import torch

    return build_direct_model(
        schema,
        model_cfg,
        repo_root,
        _mask_log_prob(arrays, rows),
        torch.from_numpy(arrays["vehicle_dimensions"][rows]).float(),
        torch.from_numpy(arrays["mask_pattern"][rows]).long(),
    ).to(device)


def _fit(
    arrays,
    schema,
    rows,
    model_cfg,
    training,
    repo_root,
    device,
    *,
    seed: int,
):
    import torch

    set_seed(seed)
    model = _build_model(arrays, schema, rows, model_cfg, repo_root, device)
    loader = _loader(
        arrays,
        rows,
        shuffle=True,
        batch_size=int(training["batch_size"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    history = []
    for epoch in range(1, int(training["epochs"]) + 1):
        metrics = _epoch(
            model,
            loader,
            device,
            optimizer=optimizer,
            grad_clip=float(training["grad_clip"]),
        )
        history.append({"epoch": epoch, **metrics})
    model.eval()
    return model, history


def _baseline_nll(arrays, train_rows, test_rows) -> dict[str, float]:
    from sklearn.mixture import GaussianMixture

    keys = ("features_normalized", "trajectory_constraint_normalized")
    train = np.concatenate(
        [
            np.asarray(arrays[key][train_rows]).reshape(len(train_rows), -1)
            for key in keys
        ],
        axis=1,
    ).astype(np.float64)
    test = np.concatenate(
        [
            np.asarray(arrays[key][test_rows]).reshape(len(test_rows), -1)
            for key in keys
        ],
        axis=1,
    ).astype(np.float64)
    mean = train.mean(axis=0)
    variance = np.maximum(train.var(axis=0), 1.0e-5)
    gaussian = 0.5 * np.sum(
        np.log(2.0 * np.pi * variance) + np.square(test - mean) / variance,
        axis=1,
    )
    gmm = GaussianMixture(
        n_components=8,
        covariance_type="diag",
        reg_covar=1.0e-5,
        max_iter=200,
        random_state=42,
    ).fit(train)
    return {
        "diagonal_gaussian_continuous_nll": float(gaussian.mean()),
        "diagonal_gmm8_continuous_nll": float(-gmm.score(test)),
    }


def _cross_validate(
    arrays,
    schema,
    rows,
    model_cfg,
    training,
    repo_root,
    device,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    from sklearn.model_selection import GroupKFold

    records = np.asarray(arrays["recording_id"])[rows]
    reports = []
    splitter = GroupKFold(n_splits=int(folds))
    for fold, (fit_index, held_out_index) in enumerate(
        splitter.split(rows, groups=records), start=1
    ):
        fit_rows = rows[fit_index]
        held_out_rows = rows[held_out_index]
        fold_arrays, fold_schema = _fold_coordinates(arrays, schema, fit_rows)
        model, _ = _fit(
            fold_arrays,
            fold_schema,
            fit_rows,
            model_cfg,
            training,
            repo_root,
            device,
            seed=seed + fold,
        )
        held_out = _epoch(
            model,
            _loader(
                fold_arrays,
                held_out_rows,
                shuffle=False,
                batch_size=int(training["batch_size"]),
            ),
            device,
        )
        baselines = _baseline_nll(fold_arrays, fit_rows, held_out_rows)
        reports.append(
            {
                "fold": fold,
                "fit_events": int(len(fit_rows)),
                "held_out_events": int(len(held_out_rows)),
                "fit_recordings": int(len(np.unique(arrays["recording_id"][fit_rows]))),
                "held_out_recordings": int(
                    len(np.unique(arrays["recording_id"][held_out_rows]))
                ),
                **held_out,
                **baselines,
            }
        )
        del model
    metric_names = [
        "joint_nll",
        "mask_nll",
        "c0_nll",
        "k_nll",
        "diagonal_gaussian_continuous_nll",
        "diagonal_gmm8_continuous_nll",
    ]
    return {
        "protocol": "recording_group_5_fold_fixed_epochs_train_fold_normalization",
        "folds": reports,
        "mean": {
            name: float(np.mean([item[name] for item in reports]))
            for name in metric_names
        },
        "std": {
            name: float(np.std([item[name] for item in reports]))
            for name in metric_names
        },
    }


def train_tail_flow(
    config: dict[str, Any],
    *,
    config_dir: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Cross-validate on recordings, then fit the released model on every tail event."""
    config_dir = Path(config_dir).resolve()
    repo_root = Path(repo_root).resolve()
    output = output_dir_from_config(config, config_dir)
    dataset_dir = dataset_dir_from_config(config, config_dir)
    arrays, schema = load_scenario_dataset(dataset_dir)
    rows = evt_tail_indices(arrays)
    if len(rows) != int(schema["evt_reference"]["num_evt_tail"]):
        raise RuntimeError("EVT-tail selection and dataset schema disagree")
    device = select_device(str(config["device"]))
    seed = int(config["seed"])
    model_cfg = dict(config["model"])
    training = dict(config["training"])
    cross_validation = _cross_validate(
        arrays,
        schema,
        rows,
        model_cfg,
        training,
        repo_root,
        device,
        folds=int(config["evaluation"]["cross_validation_folds"]),
        seed=seed,
    )
    save_json(cross_validation, output / "cross_validation.json")
    model, history = _fit(
        arrays,
        schema,
        rows,
        model_cfg,
        training,
        repo_root,
        device,
        seed=seed,
    )
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    checkpoint = output / "checkpoints/scenario_flow.pt"
    metrics = {
        "population": "evt_tail",
        "fit_scope": "all_evt_tail_events",
        "num_fit_events": int(len(rows)),
        "epochs": int(training["epochs"]),
        "cross_validation": "results/flow/cross_validation.json",
    }
    save_checkpoint(checkpoint, model, model_cfg, metrics)
    summary = {
        "model": "tail_scenario_flow",
        "checkpoint": "results/flow/checkpoints/scenario_flow.pt",
        "dataset": str(Path(schema["dataset_npz"]).resolve().relative_to(repo_root)),
        "evt_tail_events": int(len(rows)),
        "fit_events": int(len(rows)),
        "fit_recordings": int(len(np.unique(arrays["recording_id"][rows]))),
        "final_epoch_nll": history[-1],
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        **metrics,
    }
    save_json(summary, output / "training_summary.json")
    LOGGER.info("Wrote all-tail Flow checkpoint: %s", checkpoint)
    return summary

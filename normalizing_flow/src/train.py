"""Training for the single maintained hierarchical scenario Flow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    build_natural_flow_dataset,
    dataset_schema_is_current,
    feature_mode_from_config,
    load_natural_dataset,
    output_dir_from_config,
    split_indices,
)
from .scenario import build_direct_model, save_checkpoint
from .utils import repo_root_from_file, save_json, select_device, set_seed

LOGGER = logging.getLogger(__name__)


def _mask_log_prob(arrays):
    import torch

    train = arrays["split_index"] == 0
    counts = np.bincount(arrays["mask_pattern"][train], minlength=64).astype(np.float64)
    probability = (counts + 1.0e-6) / (counts.sum() + 64.0e-6)
    return torch.from_numpy(np.log(probability).astype(np.float32))


def _loader(arrays, split: str, batch_size: int):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    rows = split_indices(arrays, split)
    dataset = TensorDataset(
        torch.from_numpy(arrays["features_normalized"][rows]).float(),
        torch.from_numpy(arrays["slot_mask"][rows]).bool(),
        torch.from_numpy(arrays["trajectory_constraint_normalized"][rows]).float(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        pin_memory=torch.cuda.is_available(),
    )


def _epoch(model, loader, device, *, optimizer=None, grad_clip: float = 0.0):
    import torch

    training = optimizer is not None
    model.train(training)
    totals = {
        name: 0.0
        for name in ("joint", "mask", "c0", "k")
    }
    count = 0
    for c0, slots, constraint in loader:
        c0, slots, constraint = (
            value.to(device, non_blocking=True)
            for value in (c0, slots, constraint)
        )
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


def train_natural_flow(
    config: dict[str, Any],
    *,
    config_dir: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Train the three-factor density on the recording-level train split."""
    import torch

    if config.get("model", {}).get("type") != "direct_scenario_condition_flow":
        raise ValueError("only model.type=direct_scenario_condition_flow is maintained")
    config_dir = Path(config_dir).resolve()
    repo_root = (
        Path(repo_root).resolve() if repo_root else repo_root_from_file(config_dir)
    )
    output = output_dir_from_config(config, config_dir)
    set_seed(int(config.get("seed", 42)))
    rebuild = bool(config.get("dataset", {}).get("rebuild", False))
    current = dataset_schema_is_current(
        output, feature_mode=feature_mode_from_config(config)
    )
    if rebuild or not current:
        build_natural_flow_dataset(config, config_dir=config_dir)
    arrays, schema = load_natural_dataset(output)
    device = select_device(str(config.get("device", "auto")))
    model_cfg = dict(config["model"])
    model = build_direct_model(
        schema, model_cfg, repo_root, _mask_log_prob(arrays)
    ).to(device)
    training = dict(config["training"]["defaults"])
    loaders = {
        split: _loader(arrays, split, int(training["batch_size"]))
        for split in ("train", "val")
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    history, best, best_epoch, best_state, stale = [], float("inf"), 0, None, 0
    for epoch in range(1, int(training["max_epochs"]) + 1):
        train = _epoch(
            model,
            loaders["train"],
            device,
            optimizer=optimizer,
            grad_clip=float(training["grad_clip"]),
        )
        val = _epoch(model, loaders["val"], device)
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train.items()},
            **{f"val_{k}": v for k, v in val.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        LOGGER.info(
            "Direct flow epoch %03d joint train=%.4f val=%.4f",
            epoch,
            train["joint_nll"],
            val["joint_nll"],
        )
        if val["joint_nll"] < best - float(training["minimum_delta"]):
            best = val["joint_nll"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(training["patience"]):
                break
    if best_state is None:
        raise RuntimeError("training produced no finite checkpoint")
    model.load_state_dict(best_state)
    checkpoint = output / "checkpoints/best_scenario_condition_flow.pt"
    metrics = {
        "best_val_joint_nll": best,
        "best_epoch": best_epoch,
        "epochs": len(history),
    }
    save_checkpoint(checkpoint, model, model_cfg, metrics)
    summary = {
        "model": "direct_scenario_condition_flow",
        "checkpoint": str(checkpoint),
        "dataset": schema["dataset_npz"],
        "num_samples": len(arrays["features"]),
        "split_summary": schema["split_summary"],
        **metrics,
    }
    save_json(summary, output / "training_summary.json")
    return summary

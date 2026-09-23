"""Train the B0/R0 prefix-only joint action diffusion prior."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from interactive_behavior_world_model.data.dataset import (
    CausalActionBlockDataset,
    ConditionedActionBlockDataset,
    ConditionedEventActionBlockDataset,
    EventAnchoredActionBlockDataset,
    estimate_action_statistics,
    estimate_feature_statistics,
)
from interactive_behavior_world_model.data.manifest import load_benchmark_config
from interactive_behavior_world_model.policies.rolling_action_policy import (
    ActionDiffusionConfig,
    RollingActionDiffusion,
)


def data_augmented_method_id(base: str, include_d1: bool, fraction: float = 1.0) -> str:
    """Name a D1 dose without changing already-qualified method identifiers."""
    if not include_d1 or "_d0d1" in base:
        return base
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("D1 training fraction must be in (0, 1]")
    suffix = (
        "_d0d1"
        if float(fraction) == 1.0
        else f"_d0d1f{int(round(float(fraction) * 100)):03d}"
    )
    return base + suffix


def _move(batch, device, feature_mean, feature_std):
    output = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    output["features"] = (output["features"] - feature_mean) / feature_std
    return output


def _ema_update(ema, model, decay: float = 0.999):
    with torch.no_grad():
        for target, source in zip(ema.parameters(), model.parameters()):
            target.mul_(decay).add_(source, alpha=1.0 - decay)
        for target, source in zip(ema.buffers(), model.buffers()):
            target.copy_(source)


@torch.no_grad()
def _validate(model, loader, device, feature_mean, feature_std):
    model.eval()
    torch.manual_seed(12345)
    totals: dict[str, float] = {}
    count = 0
    for raw in loader:
        batch = _move(raw, device, feature_mean, feature_std)
        losses = model.loss(
            batch["action_block"],
            batch["features"],
            batch["history_valid"],
            batch["action_block_valid"],
            batch["current_background_state"],
            batch["future_background_states"],
        )
        size = len(batch["features"])
        count += size
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value) * size
    return {key: value / max(count, 1) for key, value in totals.items()}


def train_action(args: argparse.Namespace) -> dict:
    config, _ = load_benchmark_config(args.config)
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    controlled_t4a = bool(getattr(args, "controlled_t4a", False))
    d0_dataset = (
        ConditionedActionBlockDataset if controlled_t4a else CausalActionBlockDataset
    )
    d1_dataset = (
        ConditionedEventActionBlockDataset
        if controlled_t4a
        else EventAnchoredActionBlockDataset
    )
    d0_train = d0_dataset(args.config, "train", seed=seed)
    d0_validation = d0_dataset(args.config, "validation", seed=seed)
    include_d1 = bool(getattr(args, "include_d1", False))
    if controlled_t4a and not include_d1:
        raise ValueError(
            "--controlled-t4a requires --include-d1 for supervised goal examples"
        )
    d1_fraction = float(getattr(args, "d1_fraction", 1.0))
    d1_train = (
        d1_dataset(args.config, "train", seed=seed, fraction=d1_fraction)
        if include_d1
        else None
    )
    d1_validation = (
        d1_dataset(args.config, "validation", seed=seed) if include_d1 else None
    )
    train = ConcatDataset((d0_train, d1_train)) if d1_train is not None else d0_train
    validation = (
        ConcatDataset((d0_validation, d1_validation))
        if d1_validation is not None
        else d0_validation
    )
    normalization_name = (
        "action_controlled_t4a_training_normalization.npz"
        if controlled_t4a
        else "action_training_normalization.npz"
    )
    normalization_path = (
        Path(config["paths"]["output_dir"]) / "data" / normalization_name
    )
    if normalization_path.exists() and not args.max_train:
        with np.load(normalization_path, allow_pickle=False) as stored:
            feature_stats = {
                "mean": stored["feature_mean"],
                "std": stored["feature_std"],
                "count": stored["feature_count"],
            }
            action_stats = {
                "mean": stored["action_mean"],
                "std": stored["action_std"],
                "count": stored["action_count"],
            }
    else:
        # Condition channels are deliberate categorical inputs, so retain the
        # natural zero origin and unit scale instead of fitting a split-
        # dependent prevalence normalization.
        # Statistics of the physical 12-D state are always fitted on D0;
        # conditioned datasets expose 16-D features and must not change that
        # training-only normalization contract.
        stats_dataset = CausalActionBlockDataset(args.config, "train", seed=seed)
        original_stats = estimate_feature_statistics(
            stats_dataset, maximum_rows=args.max_train
        )
        feature_stats = original_stats
        if controlled_t4a:
            feature_stats = {
                "mean": np.concatenate(
                    (original_stats["mean"], np.zeros(4, np.float32))
                ),
                "std": np.concatenate((original_stats["std"], np.ones(4, np.float32))),
                "count": original_stats["count"],
            }
        action_stats = estimate_action_statistics(d0_train)
        if not args.max_train:
            normalization_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                normalization_path,
                feature_mean=feature_stats["mean"],
                feature_std=feature_stats["std"],
                feature_count=feature_stats["count"],
                action_mean=action_stats["mean"],
                action_std=action_stats["std"],
                action_count=action_stats["count"],
            )
    feature_mean = torch.from_numpy(feature_stats["mean"]).to(device)
    feature_std = torch.from_numpy(feature_stats["std"]).to(device)
    training_objective = str(getattr(args, "objective", "diffusion"))
    model_config = ActionDiffusionConfig(
        feature_dim=16 if controlled_t4a else 12, training_objective=training_objective
    )
    model = RollingActionDiffusion(
        model_config, action_stats["mean"], action_stats["std"]
    ).to(device)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate * 0.1
    )
    # Full precision avoids v-prediction/kinematic-gradient overflow observed
    # with the legacy torch 1.13 AMP stack in the tread environment.
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    batch_size = args.batch_size or 128
    workers = (
        args.num_workers
        if args.num_workers is not None
        else int(config["training"]["num_workers"])
    )
    train_rows = range(min(args.max_train, len(train))) if args.max_train else None
    val_rows = (
        range(min(args.max_validation, len(validation)))
        if args.max_validation
        else None
    )
    train_view = (
        torch.utils.data.Subset(train, train_rows) if train_rows is not None else train
    )
    val_view = (
        torch.utils.data.Subset(validation, val_rows)
        if val_rows is not None
        else validation
    )
    make_loader = lambda data, shuffle: DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    train_loader, val_loader = make_loader(train_view, True), make_loader(
        val_view, False
    )
    method_id = (
        "rolling_action_controlled_b0"
        if controlled_t4a
        else (
            "rolling_action_flow_b0"
            if training_objective == "flow_matching"
            else "rolling_action_b0"
        )
    )
    method_id = data_augmented_method_id(method_id, include_d1, d1_fraction)
    output = Path(config["paths"]["output_dir"]) / f"methods/{method_id}" / str(seed)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "best.pt"
    history = []
    best = float("inf")
    stale = 0
    started = time.time()
    for epoch in range(args.epochs):
        d0_train.set_epoch(epoch)
        model.train()
        totals: dict[str, float] = {}
        count = 0
        epoch_start = time.time()
        for raw in train_loader:
            batch = _move(raw, device, feature_mean, feature_std)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=False):
                losses = model.loss(
                    batch["action_block"],
                    batch["features"],
                    batch["history_valid"],
                    batch["action_block_valid"],
                    batch["current_background_state"],
                    batch["future_background_states"],
                )
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            _ema_update(ema, model)
            size = len(batch["features"])
            count += size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * size
        scheduler.step()
        train_metrics = {key: value / count for key, value in totals.items()}
        val_metrics = _validate(ema, val_loader, device, feature_mean, feature_std)
        row = {
            "epoch": epoch + 1,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"validation_{k}": v for k, v in val_metrics.items()},
            "seconds": time.time() - epoch_start,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["loss"] < best - 1.0e-5:
            best = val_metrics["loss"]
            stale = 0
            torch.save(
                {
                    "model_state": ema.state_dict(),
                    "model_config": asdict(model_config),
                    "feature_mean": feature_stats["mean"],
                    "feature_std": feature_stats["std"],
                    "action_mean": action_stats["mean"],
                    "action_std": action_stats["std"],
                    "fit_seed": seed,
                    "method_id": method_id,
                    "training_data": "D0+D1" if include_d1 else "D0",
                    "d1_training_fraction": d1_fraction if include_d1 else 0.0,
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= int(config["training"]["validation_patience"]):
                break
    report = {
        "method_id": method_id,
        "ablation_id": "B0-flow" if training_objective == "flow_matching" else "B0/R0",
        "fit_seed": seed,
        "training_objective": training_objective,
        "training_data": "D0+D1" if include_d1 else "D0",
        "normalization_scope": "D0_train",
        "t4a_behavior_condition": (
            "four_mode_explicit_goal" if controlled_t4a else "none"
        ),
        "d1_training_fraction": d1_fraction if include_d1 else 0.0,
        "d0_training_rows": len(d0_train),
        "d1_training_rows": len(d1_train) if d1_train is not None else 0,
        "training_rows": len(train_view),
        "validation_rows": len(val_view),
        "epochs_completed": len(history),
        "best_validation_loss": best,
        "wall_time_seconds": time.time() - started,
        "history": history,
    }
    (output / "training_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/benchmark_v1.yaml",
    )
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument(
        "--objective", choices=("diffusion", "flow_matching"), default="diffusion"
    )
    parser.add_argument("--include-d1", action="store_true")
    parser.add_argument(
        "--controlled-t4a",
        action="store_true",
        help="train 16-feature explicit-goal T4a policy on D0+D1",
    )
    parser.add_argument("--d1-fraction", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-validation", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(train_action(args), indent=2))


if __name__ == "__main__":
    main()

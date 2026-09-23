"""Train B3 position-block diffusion with the B0 network and information."""

from __future__ import annotations

import argparse, copy, json, random, time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from interactive_behavior_world_model.data.dataset import (
    CausalActionBlockDataset,
    estimate_position_block_statistics,
)
from interactive_behavior_world_model.data.manifest import load_benchmark_config
from interactive_behavior_world_model.policies.rolling_action_policy import (
    ActionDiffusionConfig,
    RollingActionDiffusion,
)
from interactive_behavior_world_model.scripts.train_action import _ema_update


def _move(raw, device, feature_mean, feature_std):
    batch = {
        key: value.to(device, non_blocking=True)
        for key, value in raw.items()
        if isinstance(value, torch.Tensor)
    }
    batch["features"] = (batch["features"] - feature_mean) / feature_std
    batch["position_block"] = (
        batch["future_background_states"][..., :2]
        - batch["current_background_state"][:, None, :, :2]
    )
    return batch


@torch.no_grad()
def _validate(model, loader, device, feature_mean, feature_std):
    model.eval()
    torch.manual_seed(12345)
    totals = {}
    count = 0
    for raw in loader:
        batch = _move(raw, device, feature_mean, feature_std)
        losses = model.loss(
            batch["position_block"],
            batch["features"],
            batch["history_valid"],
            batch["action_block_valid"],
            batch["current_background_state"],
            batch["future_background_states"],
            trajectory_weight=0.0,
        )
        size = len(batch["features"])
        count += size
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value) * size
    return {key: value / max(count, 1) for key, value in totals.items()}


def train(args):
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
    train = CausalActionBlockDataset(args.config, "train", seed=seed)
    validation = CausalActionBlockDataset(args.config, "validation", seed=seed)
    norm_path = (
        Path(config["paths"]["output_dir"])
        / "data/position_b3_horizon_normalization.npz"
    )
    feature_path = (
        Path(config["paths"]["output_dir"]) / "data/action_training_normalization.npz"
    )
    with np.load(feature_path, allow_pickle=False) as stored:
        feature_stats = {
            "mean": stored["feature_mean"],
            "std": stored["feature_std"],
            "count": stored["feature_count"],
        }
    if norm_path.exists():
        with np.load(norm_path, allow_pickle=False) as stored:
            target_stats = {
                "mean": stored["position_mean"],
                "std": stored["position_std"],
                "count": stored["position_count"],
            }
    else:
        target_stats = estimate_position_block_statistics(train)
        np.savez_compressed(
            norm_path,
            position_mean=target_stats["mean"],
            position_std=target_stats["std"],
            position_count=target_stats["count"],
        )
    feature_mean = torch.from_numpy(feature_stats["mean"]).to(device)
    feature_std = torch.from_numpy(feature_stats["std"]).to(device)
    model_config = ActionDiffusionConfig()
    model = RollingActionDiffusion(
        model_config, target_stats["mean"], target_stats["std"]
    ).to(device)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate * 0.1
    )
    train_view = (
        torch.utils.data.Subset(train, range(min(args.max_train, len(train))))
        if args.max_train
        else train
    )
    val_view = (
        torch.utils.data.Subset(
            validation, range(min(args.max_validation, len(validation)))
        )
        if args.max_validation
        else validation
    )
    make = lambda data, shuffle: DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_loader, val_loader = make(train_view, True), make(val_view, False)
    output = (
        Path(config["paths"]["output_dir"]) / "methods/position_track_b3" / str(seed)
    )
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "best.pt"
    history = []
    best = float("inf")
    stale = 0
    started = time.time()
    for epoch in range(args.epochs):
        train.set_epoch(epoch)
        model.train()
        totals = {}
        count = 0
        tick = time.time()
        for raw in train_loader:
            batch = _move(raw, device, feature_mean, feature_std)
            optimizer.zero_grad(set_to_none=True)
            losses = model.loss(
                batch["position_block"],
                batch["features"],
                batch["history_valid"],
                batch["action_block_valid"],
                batch["current_background_state"],
                batch["future_background_states"],
                trajectory_weight=0.0,
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
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
            "seconds": time.time() - tick,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["loss"] < best - 1e-5:
            best = val_metrics["loss"]
            stale = 0
            torch.save(
                {
                    "model_state": ema.state_dict(),
                    "model_config": asdict(model_config),
                    "feature_mean": feature_stats["mean"],
                    "feature_std": feature_stats["std"],
                    "action_mean": target_stats["mean"],
                    "action_std": target_stats["std"],
                    "fit_seed": seed,
                    "method_id": "position_track_b3",
                    "target_semantics": "relative_position_endpoint_m",
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= int(config["training"]["validation_patience"]):
                break
    report = {
        "method_id": "position_track_b3",
        "ablation_id": "B3",
        "fit_seed": seed,
        "training_rows": len(train_view),
        "validation_rows": len(val_view),
        "epochs_completed": len(history),
        "best_validation_loss": best,
        "wall_time_seconds": time.time() - started,
        "history": history,
    }
    (output / "training_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/benchmark_v1.yaml",
    )
    p.add_argument("--seed", type=int, default=20260919)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-validation", type=int, default=0)
    args = p.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()

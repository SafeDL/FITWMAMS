"""Train the single frozen structured-state behavior-cloning PNC for T3."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from interactive_behavior_world_model.data.dataset import (
    EgoBCDataset,
    estimate_ego_action_statistics,
    estimate_feature_statistics,
)
from interactive_behavior_world_model.data.manifest import load_benchmark_config
from interactive_behavior_world_model.policies.learned_pnc import StructuredBCPNC


def _loader(dataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )


def _loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.smooth_l1_loss(prediction, target, beta=0.5)


@torch.no_grad()
def _evaluate(
    model, loader, feature_mean, feature_std, action_mean, action_std, device
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        features = (batch["features"].to(device) - feature_mean) / feature_std
        valid = batch["history_valid"].to(device)
        target = (batch["ego_target"].to(device) - action_mean) / action_std
        loss = _loss(model(features, valid), target)
        total += float(loss) * len(features)
        count += len(features)
    return total / max(count, 1)


def train(args: argparse.Namespace) -> dict:
    config, _ = load_benchmark_config(args.config)
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_data = EgoBCDataset(args.config, "train", seed=seed)
    validation_data = EgoBCDataset(args.config, "validation", seed=seed)
    feature_stats = estimate_feature_statistics(train_data)
    action_stats = estimate_ego_action_statistics(train_data)
    action_scale = action_stats["std"].copy()
    action_scale[1] = max(float(action_scale[1]), float(args.yaw_scale_floor_rps))
    feature_mean = torch.from_numpy(feature_stats["mean"]).to(device)
    feature_std = torch.from_numpy(feature_stats["std"]).to(device)
    action_mean = torch.from_numpy(action_stats["mean"]).to(device)
    action_std = torch.from_numpy(action_scale).to(device)
    model_config = {"feature_dim": 12, "hidden_dim": 96, "heads": 4, "layers": 2}
    model = StructuredBCPNC(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.1
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    train_loader = _loader(train_data, args.batch_size, args.num_workers, True)
    validation_loader = _loader(
        validation_data, args.batch_size, args.num_workers, False
    )
    output = Path(config["paths"]["output_dir"]) / "pnc" / args.method_id / str(seed)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "best.pt"
    best = float("inf")
    stale = 0
    history = []
    started = time.time()
    for epoch in range(args.epochs):
        train_data.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        count = 0
        epoch_start = time.time()
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            valid = batch["history_valid"].to(device, non_blocking=True)
            target = batch["ego_target"].to(device, non_blocking=True)
            if args.lateral_augmentation_m > 0.0:
                lateral_offset = torch.empty((len(features),), device=device).uniform_(
                    -args.lateral_augmentation_m,
                    args.lateral_augmentation_m,
                )
                features = features.clone()
                features[:, :, 1:, 1] -= lateral_offset[:, None, None]
                features[:, :, 0, 10] += lateral_offset[:, None]
                target = target.clone()
                target[:, 1] += torch.atan(-lateral_offset / 12.0) / 0.8
            features = (features - feature_mean) / feature_std
            target = (target - action_mean) / action_std
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = _loss(model(features, valid), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(features)
            count += len(features)
        scheduler.step()
        validation_loss = _evaluate(
            model,
            validation_loader,
            feature_mean,
            feature_std,
            action_mean,
            action_std,
            device,
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(count, 1),
            "validation_loss": validation_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": time.time() - epoch_start,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_loss < best - 1.0e-5:
            best = validation_loss
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "feature_mean": feature_stats["mean"],
                    "feature_std": feature_stats["std"],
                    "action_mean": action_stats["mean"],
                    "action_std": action_scale,
                    "fit_seed": seed,
                    "method_id": args.method_id,
                    "lateral_augmentation_m": args.lateral_augmentation_m,
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    report = {
        "method_id": args.method_id,
        "fit_seed": seed,
        "training_rows": len(train_data),
        "validation_rows": len(validation_data),
        "best_validation_loss": best,
        "epochs_completed": len(history),
        "wall_time_seconds": time.time() - started,
        "lateral_augmentation_m": args.lateral_augmentation_m,
        "yaw_scale_floor_rps": args.yaw_scale_floor_rps,
        "feature_statistics_count": int(feature_stats["count"]),
        "action_statistics_count": int(action_stats["count"]),
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--method-id", default="structured_bc_pnc")
    parser.add_argument("--lateral-augmentation-m", type=float, default=0.0)
    parser.add_argument("--yaw-scale-floor-rps", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    print(json.dumps(train(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()

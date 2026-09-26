"""Train benchmark policies; currently provides the full shared-BC/A0 baseline."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from npc_behavior_benchmark.data.dataset import (
    CausalBCDataset,
    EventAnchoredBCDataset,
    estimate_feature_statistics,
)
from npc_behavior_benchmark.data.manifest import load_benchmark_config
from npc_behavior_benchmark.policies.shared_bc import SharedBCModel, gaussian_nll


def _loader(dataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        # Workers are recreated each epoch so the dataset's deterministic
        # epoch-dependent decision offset is propagated without shared state.
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )


@torch.no_grad()
def _evaluate(model, loader, mean, std, device) -> float:
    model.eval()
    total, weight = 0.0, 0
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        features = (features - mean) / std
        history_valid = batch["history_valid"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        valid = batch["target_valid"].to(device, non_blocking=True)
        style = (
            torch.zeros(
                (len(features), features.shape[2], model.style_dim), device=device
            )
            if model.style_dim
            else None
        )
        pred, log_std = model(features, history_valid, style)
        loss = gaussian_nll(pred, log_std, target, valid)
        total += float(loss) * len(features)
        weight += len(features)
    return total / max(weight, 1)


def train_shared_bc(args: argparse.Namespace) -> dict:
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
    d0_train = CausalBCDataset(args.config, "train", seed=seed)
    d0_validation = CausalBCDataset(args.config, "validation", seed=seed)
    d1_train = (
        EventAnchoredBCDataset(
            args.config, "train", seed=seed, fraction=args.d1_fraction
        )
        if args.include_d1
        else None
    )
    d1_validation = (
        EventAnchoredBCDataset(args.config, "validation", seed=seed)
        if args.include_d1
        else None
    )
    train_data = (
        ConcatDataset((d0_train, d1_train)) if d1_train is not None else d0_train
    )
    validation_data = (
        ConcatDataset((d0_validation, d1_validation))
        if d1_validation is not None
        else d0_validation
    )
    if args.max_train:
        train_data = Subset(train_data, range(min(args.max_train, len(train_data))))
    if args.max_validation:
        validation_data = Subset(
            validation_data, range(min(args.max_validation, len(validation_data)))
        )
    # Keep D0 normalization fixed in the data-attribution track so the only
    # intended change is the addition of event-anchored D1 supervision.
    statistics = estimate_feature_statistics(d0_train, maximum_rows=args.max_train)
    feature_mean = torch.from_numpy(statistics["mean"]).to(device)
    feature_std = torch.from_numpy(statistics["std"]).to(device)
    model = SharedBCModel(style_dim=args.style_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.1
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    batch_size = args.batch_size or int(config["training"]["batch_size"])
    workers = (
        args.num_workers
        if args.num_workers is not None
        else int(config["training"]["num_workers"])
    )
    train_loader = _loader(train_data, batch_size, workers, True)
    validation_loader = _loader(validation_data, batch_size, workers, False)
    patience = int(config["training"]["validation_patience"])
    method_id = "shared_bc_style" if args.style_dim else "shared_bc"
    if args.include_d1:
        method_id += (
            "_d0d1"
            if args.d1_fraction == 1.0
            else f"_d0d1f{int(round(args.d1_fraction * 100)):03d}"
        )
    output = Path(config["paths"]["output_dir"]) / f"methods/{method_id}" / str(seed)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "best.pt"
    history, best, stale = [], float("inf"), 0
    started = time.time()

    for epoch in range(args.epochs):
        d0_train.set_epoch(epoch)
        model.train()
        loss_sum, count = 0.0, 0
        epoch_start = time.time()
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            features = (features - feature_mean) / feature_std
            history_valid = batch["history_valid"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            valid = batch["target_valid"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                style = (
                    torch.randn(
                        (len(features), features.shape[2], model.style_dim),
                        device=device,
                    )
                    if model.style_dim
                    else None
                )
                pred, log_std = model(features, history_valid, style)
                loss = gaussian_nll(pred, log_std, target, valid)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(features)
            count += len(features)
        scheduler.step()
        val_loss = _evaluate(
            model, validation_loader, feature_mean, feature_std, device
        )
        row = {
            "epoch": epoch + 1,
            "train_nll": loss_sum / max(count, 1),
            "validation_nll": val_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": time.time() - epoch_start,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best - 1.0e-5:
            best, stale = val_loss, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_mean": statistics["mean"],
                    "feature_std": statistics["std"],
                    "fit_seed": seed,
                    "method_id": method_id,
                    "style_resampling": (
                        args.style_resampling if args.style_dim else "none"
                    ),
                    "training_data": "D0+D1" if args.include_d1 else "D0",
                    "d1_training_fraction": (
                        args.d1_fraction if args.include_d1 else 0.0
                    ),
                    "model_config": {
                        "feature_dim": 12,
                        "hidden_dim": 96,
                        "heads": 4,
                        "layers": 2,
                        "style_dim": args.style_dim,
                    },
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= patience:
                break
    report = {
        "method_id": method_id,
        "ablation_id": "A0_style" if args.style_dim else "A0",
        "fit_seed": seed,
        "style_dim": args.style_dim,
        "style_resampling": args.style_resampling if args.style_dim else "none",
        "training_data": "D0+D1" if args.include_d1 else "D0",
        "normalization_scope": "D0_train",
        "d1_training_fraction": args.d1_fraction if args.include_d1 else 0.0,
        "d0_training_rows": len(d0_train),
        "d1_training_rows": len(d1_train) if d1_train is not None else 0,
        "device": str(device),
        "training_rows": len(train_data),
        "validation_rows": len(validation_data),
        "epochs_completed": len(history),
        "best_validation_nll": best,
        "wall_time_seconds": time.time() - started,
        "feature_statistics_count": int(statistics["count"]),
        "history": history,
    }
    (output / "training_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("a0", "shared_bc", "b0"), default="a0")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/benchmark_v1.yaml",
    )
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument(
        "--objective", choices=("diffusion", "flow_matching"), default="diffusion"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-validation", type=int, default=0)
    parser.add_argument("--style-dim", type=int, default=0)
    parser.add_argument(
        "--style-resampling",
        choices=("persistent", "per_decision"),
        default="persistent",
    )
    parser.add_argument("--include-d1", action="store_true")
    parser.add_argument("--d1-fraction", type=float, default=1.0)
    args = parser.parse_args()
    if args.method == "b0":
        from npc_behavior_benchmark.scripts.train_action import train_action

        print(json.dumps(train_action(args), indent=2))
    else:
        print(json.dumps(train_shared_bc(args), indent=2))


if __name__ == "__main__":
    main()

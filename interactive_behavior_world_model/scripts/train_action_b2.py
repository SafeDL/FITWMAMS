"""Fine-tune B1 with one-decision self-unfolded action-block training (B2)."""

from __future__ import annotations

import argparse, copy, json, random, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from world_model.src.core.dynamics import KinematicTrafficDynamics

from interactive_behavior_world_model.data.dataset import (
    CausalActionSelfRolloutDataset,
    EventAnchoredActionSelfRolloutDataset,
)
from interactive_behavior_world_model.data.manifest import load_benchmark_config
from interactive_behavior_world_model.evaluation.rollout import _torch_features
from interactive_behavior_world_model.policies.rolling_action_policy import (
    ActionDiffusionConfig,
    RollingActionDiffusion,
)
from interactive_behavior_world_model.scripts.train_action import (
    _ema_update,
    _move,
    _validate,
    data_augmented_method_id,
)


def _self_unfold(
    model,
    checkpoint,
    batch,
    feature_mean,
    feature_std,
    *,
    trajectory_weight: float = 0.02,
    lateral_trajectory_multiplier: float = 1.0,
):
    raw = batch["raw_history"]
    valid = batch["history_valid"]
    active = valid[:, -1]
    lengths = batch["lengths"]
    widths = batch["widths"]
    lines = batch["map_polylines"]
    line_valid = batch["map_valid"]
    lane_valid = line_valid.any(-1)
    centers = (lines[..., 1] * line_valid).sum(-1) / line_valid.sum(-1).clamp_min(1)
    lane_width = (lines[..., 4] * line_valid).sum(-1) / line_valid.sum(-1).clamp_min(1)
    mask = batch["action_block_valid"]
    noise = torch.randn_like(batch["action_block"])
    with torch.no_grad():
        block = model.sample(batch["features"], valid, mask, noise, inference_steps=4)
        action = block[:, 0].clone()
        action[..., 0].clamp_(-8, 4)
        speed = torch.linalg.vector_norm(raw[:, -1, 1:, 2:4], dim=-1).clamp_min(1e-3)
        limit = torch.minimum(torch.full_like(speed, 0.6), 4 / speed)
        action[..., 1] = torch.maximum(torch.minimum(action[..., 1], limit), -limit)
        action *= active[:, 1:, None]
    state = raw[:, -1].clone()
    dynamics = KinematicTrafficDynamics()
    control = torch.zeros((len(state), 7, 2), device=state.device)
    control[:, 1:] = action
    generated = []
    for frame in range(5):
        state = dynamics.step(state, control, active, 0.04)
        state[:, 0] = batch["next_ego_states"][:, frame]
        generated.append(state)
    history = torch.cat((raw[:, 5:], torch.stack(generated, 1)), 1)
    features = (
        _torch_features(
            history, valid, lengths, widths, centers, lane_width, lane_valid
        )
        - feature_mean
    ) / feature_std
    shifted_action = torch.cat(
        (batch["action_block"][:, 1:], torch.zeros_like(batch["action_block"][:, :1])),
        1,
    )
    shifted_valid = torch.cat((mask[:, 1:], torch.zeros_like(mask[:, :1])), 1)
    shifted_future = torch.cat(
        (
            batch["future_background_states"][:, 1:],
            torch.zeros_like(batch["future_background_states"][:, :1]),
        ),
        1,
    )
    return model.loss(
        shifted_action,
        features,
        valid,
        shifted_valid,
        state[:, 1:],
        shifted_future,
        trajectory_weight=trajectory_weight,
        lateral_trajectory_multiplier=lateral_trajectory_multiplier,
    )


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
    source = torch.load(args.base_checkpoint, map_location="cpu")
    model = RollingActionDiffusion(
        ActionDiffusionConfig(**source["model_config"]),
        source["action_mean"],
        source["action_std"],
    )
    model.load_state_dict(source["model_state"])
    model.to(device)
    ema = copy.deepcopy(model).eval()
    feature_mean = torch.as_tensor(source["feature_mean"], device=device)
    feature_std = torch.as_tensor(source["feature_std"], device=device)
    d0_train = CausalActionSelfRolloutDataset(args.config, "train", seed=seed)
    d0_val = CausalActionSelfRolloutDataset(args.config, "validation", seed=seed)
    d1_train = (
        EventAnchoredActionSelfRolloutDataset(
            args.config, "train", seed=seed, fraction=args.d1_fraction
        )
        if args.include_d1
        else None
    )
    d1_val = (
        EventAnchoredActionSelfRolloutDataset(args.config, "validation", seed=seed)
        if args.include_d1
        else None
    )
    train_data = (
        ConcatDataset((d0_train, d1_train)) if d1_train is not None else d0_train
    )
    val_data = ConcatDataset((d0_val, d1_val)) if d1_val is not None else d0_val
    train_view = (
        Subset(train_data, range(min(args.max_train, len(train_data))))
        if args.max_train
        else train_data
    )
    val_view = (
        Subset(val_data, range(min(args.max_validation, len(val_data))))
        if args.max_validation
        else val_data
    )
    make = lambda data, shuffle: DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_loader = make(train_view, True)
    val_loader = make(val_view, False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    method_id = data_augmented_method_id(
        "rolling_action_b2", args.include_d1, args.d1_fraction
    )
    output = Path(config["paths"]["output_dir"]) / f"methods/{method_id}" / str(seed)
    output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stale = 0
    history = []
    started = time.time()
    for epoch in range(args.epochs):
        d0_train.set_epoch(epoch)
        model.train()
        total = standard_total = robust_total = 0.0
        count = 0
        tick = time.time()
        for raw in train_loader:
            batch = _move(raw, device, feature_mean, feature_std)
            optimizer.zero_grad(set_to_none=True)
            standard = model.loss(
                batch["action_block"],
                batch["features"],
                batch["history_valid"],
                batch["action_block_valid"],
                batch["current_background_state"],
                batch["future_background_states"],
            )
            robust = _self_unfold(model, source, batch, feature_mean, feature_std)
            loss = standard["loss"] + args.self_rollout_weight * robust["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            _ema_update(ema, model)
            size = len(batch["features"])
            count += size
            total += float(loss.detach()) * size
            standard_total += float(standard["loss"].detach()) * size
            robust_total += float(robust["loss"].detach()) * size
        validation = _validate(ema, val_loader, device, feature_mean, feature_std)
        row = {
            "epoch": epoch + 1,
            "train_loss": total / count,
            "train_logged_loss": standard_total / count,
            "train_self_unfold_loss": robust_total / count,
            "validation_logged_loss": validation["loss"],
            "seconds": time.time() - tick,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["loss"] < best - 1e-5:
            best = validation["loss"]
            stale = 0
            torch.save(
                {
                    **source,
                    "model_state": ema.state_dict(),
                    "fit_seed": seed,
                    "method_id": method_id,
                    "training_data": "D0+D1" if args.include_d1 else "D0",
                    "d1_training_fraction": (
                        args.d1_fraction if args.include_d1 else 0.0
                    ),
                    "self_rollout_weight": args.self_rollout_weight,
                },
                output / "best.pt",
            )
        else:
            stale += 1
            if stale >= 4:
                break
    report = {
        "method_id": method_id,
        "fit_seed": seed,
        "training_data": "D0+D1" if args.include_d1 else "D0",
        "d1_training_fraction": args.d1_fraction if args.include_d1 else 0.0,
        "d0_training_rows": len(d0_train),
        "d1_training_rows": len(d1_train) if d1_train is not None else 0,
        "training_rows": len(train_view),
        "validation_rows": len(val_view),
        "best_validation_logged_loss": best,
        "epochs_completed": len(history),
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
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260919)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--self-rollout-weight", type=float, default=0.25)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-validation", type=int, default=0)
    p.add_argument("--include-d1", action="store_true")
    p.add_argument("--d1-fraction", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    print(json.dumps(train(a), indent=2))


if __name__ == "__main__":
    main()

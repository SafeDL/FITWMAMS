"""Fine-tune B2 with train-only paired response preservation (B2-R)."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

from npc_behavior_benchmark.data.dataset import (
    CausalActionSelfRolloutDataset,
    EventAnchoredActionSelfRolloutDataset,
)
from npc_behavior_benchmark.data.manifest import load_benchmark_config
from npc_behavior_benchmark.data.response_dataset import ResponsePairDataset
from npc_behavior_benchmark.evaluation.rollout import _torch_features
from npc_behavior_benchmark.policies.rolling_action_policy import (
    ActionDiffusionConfig,
    RollingActionDiffusion,
)
from npc_behavior_benchmark.scripts.train_action import (
    _ema_update,
    _move,
    _validate,
    data_augmented_method_id,
)
from npc_behavior_benchmark.scripts.train_action_b2 import _self_unfold


def _paired_response_loss(
    model, batch, feature_mean, feature_std, teacher_scale, *, timestep_value: int = 79
):
    """Match the next response action after 0.2 s of an observed stimulus."""
    history = batch["history"]
    valid = batch["history_valid"]
    active = valid[:, -1]
    base_tail = batch["logged_future"][:, :5].clone()
    changed_tail = base_tail.clone()
    row = torch.arange(len(history), device=history.device)
    stimulus = batch["stimulus_index"]
    changed_tail[
        row[:, None], torch.arange(5, device=history.device)[None], stimulus[:, None]
    ] = batch["intervention_stimulus"][:, :5]
    base_history = torch.cat((history[:, 5:], base_tail), 1)
    changed_history = torch.cat((history[:, 5:], changed_tail), 1)
    shifted_valid = torch.cat((valid[:, 5:], active[:, None].expand(-1, 5, -1)), 1)
    lines = batch["map_polylines"]
    line_valid = batch["map_valid"]
    lane_valid = line_valid.any(-1)
    lane_centers = (lines[..., 1] * line_valid).sum(-1) / line_valid.sum(-1).clamp_min(
        1
    )
    lane_width = (lines[..., 4] * line_valid).sum(-1) / line_valid.sum(-1).clamp_min(1)

    def features(value):
        return (
            _torch_features(
                value,
                shifted_valid,
                batch["lengths"],
                batch["widths"],
                lane_centers,
                lane_width,
                lane_valid,
            )
            - feature_mean
        ) / feature_std

    mask = (
        active[:, None, 1:]
        .expand(-1, model.config.action_horizon, -1)[..., None]
        .to(history.dtype)
    )
    noise = (
        torch.randn(
            (len(history), model.config.action_horizon, model.config.agents, 2),
            device=history.device,
        )
        * mask
    )
    timestep = torch.full(
        (len(history),), int(timestep_value), dtype=torch.long, device=history.device
    )
    base_output = (
        model.denoiser(noise, timestep, features(base_history), shifted_valid) * mask
    )
    changed_output = (
        model.denoiser(noise, timestep, features(changed_history), shifted_valid) * mask
    )
    base_clean = model.predict_clean(noise, timestep, base_output).clamp(-6.0, 6.0)
    changed_clean = model.predict_clean(noise, timestep, changed_output).clamp(
        -6.0, 6.0
    )
    base_action = base_clean * model.action_std + model.action_mean
    changed_action = changed_clean * model.action_std + model.action_mean
    action_delta = changed_action[:, 0] - base_action[:, 0]
    response_slot = batch["response_index"] - 1
    predicted = action_delta[row, response_slot, 0]
    # Index 1 is the teacher action over 0.2--0.4 s, after the stimulus has
    # become observable at the common 5 Hz policy boundary.
    target = batch["teacher_delta"][:, :, 1, 2].mean(1)
    response = F.smooth_l1_loss(predicted / teacher_scale, target / teacher_scale)
    slots = torch.arange(model.config.agents, device=history.device)[None]
    unrelated = active[:, 1:] & (slots != response_slot[:, None])
    locality = (
        action_delta.square().sum(-1) * unrelated
    ).sum() / unrelated.sum().clamp_min(1)
    direction = F.relu(0.05 + predicted)  # MA-IDM D2 contains added leader braking.
    return {"response": response, "locality": locality, "direction": direction.mean()}


def _validate_response(model, loader, device, feature_mean, feature_std, teacher_scale):
    totals = {"response": 0.0, "locality": 0.0, "direction": 0.0}
    count = 0
    model.eval()
    torch.manual_seed(12345)
    with torch.no_grad():
        for raw in loader:
            batch = {
                key: value.to(device, non_blocking=True) for key, value in raw.items()
            }
            values = _paired_response_loss(
                model, batch, feature_mean, feature_std, teacher_scale
            )
            size = len(batch["history"])
            count += size
            for key in totals:
                totals[key] += float(values[key]) * size
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
    source = torch.load(args.base_checkpoint, map_location="cpu")
    model = RollingActionDiffusion(
        ActionDiffusionConfig(**source["model_config"]),
        source["action_mean"],
        source["action_std"],
    )
    if model.config.training_objective != "diffusion":
        raise ValueError(
            "B2-R paired response preservation currently requires a diffusion checkpoint"
        )
    model.load_state_dict(source["model_state"])
    model.to(device)
    ema = copy.deepcopy(model).eval()
    feature_mean = torch.as_tensor(source["feature_mean"], device=device)
    feature_std = torch.as_tensor(source["feature_std"], device=device)
    d0_action_train = CausalActionSelfRolloutDataset(args.config, "train", seed=seed)
    d0_action_val = CausalActionSelfRolloutDataset(args.config, "validation", seed=seed)
    d1_action_train = (
        EventAnchoredActionSelfRolloutDataset(
            args.config, "train", seed=seed, fraction=args.d1_fraction
        )
        if args.include_d1
        else None
    )
    d1_action_val = (
        EventAnchoredActionSelfRolloutDataset(args.config, "validation", seed=seed)
        if args.include_d1
        else None
    )
    action_train = (
        ConcatDataset((d0_action_train, d1_action_train))
        if d1_action_train is not None
        else d0_action_train
    )
    action_val = (
        ConcatDataset((d0_action_val, d1_action_val))
        if d1_action_val is not None
        else d0_action_val
    )
    if args.max_train:
        action_train = Subset(
            action_train, range(min(args.max_train, len(action_train)))
        )
    if args.max_validation:
        action_val = Subset(
            action_val, range(min(args.max_validation, len(action_val)))
        )
    response_train = ResponsePairDataset(args.config, "train")
    response_val = ResponsePairDataset(args.config, "validation")
    with np.load(
        Path(config["paths"]["output_dir"]) / "data/response_pairs_ma_idm_train_v3.npz",
        allow_pickle=False,
    ) as pairs:
        teacher_scale_value = float(
            np.std(pairs["teacher_response_delta"][..., 2]).clip(0.05)
        )
    teacher_scale = torch.tensor(teacher_scale_value, device=device)
    make = lambda data, batch, shuffle: DataLoader(
        data,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    action_loader = make(action_train, args.batch_size, True)
    action_val_loader = make(action_val, args.batch_size, False)
    response_loader = make(response_train, args.response_batch_size, True)
    response_val_loader = make(response_val, args.response_batch_size, False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4
    )
    method_id = data_augmented_method_id(
        args.method_id, args.include_d1, args.d1_fraction
    )
    output = Path(config["paths"]["output_dir"]) / "methods" / method_id / str(seed)
    output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stale = 0
    history = []
    started = time.time()
    for epoch in range(args.epochs):
        d0_action_train.set_epoch(epoch)
        model.train()
        response_iterator = iter(response_loader)
        totals = {
            "loss": 0.0,
            "logged": 0.0,
            "self": 0.0,
            "response": 0.0,
            "locality": 0.0,
            "direction": 0.0,
        }
        count = 0
        tick = time.time()
        for raw in action_loader:
            try:
                response_raw = next(response_iterator)
            except StopIteration:
                response_iterator = iter(response_loader)
                response_raw = next(response_iterator)
            batch = _move(raw, device, feature_mean, feature_std)
            pair = {
                key: value.to(device, non_blocking=True)
                for key, value in response_raw.items()
            }
            optimizer.zero_grad(set_to_none=True)
            logged = model.loss(
                batch["action_block"],
                batch["features"],
                batch["history_valid"],
                batch["action_block_valid"],
                batch["current_background_state"],
                batch["future_background_states"],
                trajectory_weight=args.trajectory_weight,
                lateral_trajectory_multiplier=args.lateral_trajectory_multiplier,
            )
            robust = _self_unfold(
                model,
                source,
                batch,
                feature_mean,
                feature_std,
                trajectory_weight=args.trajectory_weight,
                lateral_trajectory_multiplier=args.lateral_trajectory_multiplier,
            )
            paired = _paired_response_loss(
                model, pair, feature_mean, feature_std, teacher_scale
            )
            loss = (
                logged["loss"]
                + args.self_rollout_weight * robust["loss"]
                + args.response_weight * paired["response"]
                + args.locality_weight * paired["locality"]
                + args.direction_weight * paired["direction"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            _ema_update(ema, model)
            size = len(batch["features"])
            count += size
            values = {
                "loss": loss,
                "logged": logged["loss"],
                "self": robust["loss"],
                **paired,
            }
            for key in totals:
                totals[key] += float(values[key].detach()) * size
        logged_val = _validate(
            ema, action_val_loader, device, feature_mean, feature_std
        )
        paired_val = _validate_response(
            ema, response_val_loader, device, feature_mean, feature_std, teacher_scale
        )
        criterion = (
            logged_val["loss"]
            + args.response_weight * paired_val["response"]
            + args.locality_weight * paired_val["locality"]
            + args.direction_weight * paired_val["direction"]
        )
        row = {
            "epoch": epoch + 1,
            **{f"train_{key}": value / count for key, value in totals.items()},
            "validation_logged_loss": logged_val["loss"],
            **{f"validation_{key}": value for key, value in paired_val.items()},
            "validation_criterion": criterion,
            "seconds": time.time() - tick,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if criterion < best - 1.0e-5:
            best = criterion
            stale = 0
            torch.save(
                {
                    **source,
                    "model_state": ema.state_dict(),
                    "fit_seed": seed,
                    "method_id": method_id,
                    "training_data": "D0+D1" if args.include_d1 else "D0",
                    "response_weight": args.response_weight,
                    "d1_training_fraction": (
                        args.d1_fraction if args.include_d1 else 0.0
                    ),
                    "locality_weight": args.locality_weight,
                    "direction_weight": args.direction_weight,
                    "response_teacher_scale": teacher_scale_value,
                    "trajectory_weight": args.trajectory_weight,
                    "lateral_trajectory_multiplier": args.lateral_trajectory_multiplier,
                },
                output / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    report = {
        "method_id": method_id,
        "fit_seed": seed,
        "training_data": "D0+D1" if args.include_d1 else "D0",
        "d1_training_fraction": args.d1_fraction if args.include_d1 else 0.0,
        "d0_training_rows": len(d0_action_train),
        "d1_training_rows": len(d1_action_train) if d1_action_train is not None else 0,
        "training_rows": len(action_train),
        "validation_rows": len(action_val),
        "response_train_pairs": len(response_train),
        "response_validation_pairs": len(response_val),
        "teacher_scale": teacher_scale_value,
        "trajectory_weight": args.trajectory_weight,
        "lateral_trajectory_multiplier": args.lateral_trajectory_multiplier,
        "best_validation_criterion": best,
        "epochs_completed": len(history),
        "wall_time_seconds": time.time() - started,
        "history": history,
    }
    (output / "training_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/benchmark_v1.yaml",
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--response-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--self-rollout-weight", type=float, default=0.25)
    parser.add_argument("--response-weight", type=float, default=0.10)
    parser.add_argument("--locality-weight", type=float, default=0.02)
    parser.add_argument("--direction-weight", type=float, default=0.05)
    parser.add_argument("--trajectory-weight", type=float, default=0.02)
    parser.add_argument("--lateral-trajectory-multiplier", type=float, default=1.0)
    parser.add_argument("--method-id", default="rolling_action_b2r")
    parser.add_argument("--include-d1", action="store_true")
    parser.add_argument("--d1-fraction", type=float, default=1.0)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-validation", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()

"""Fine-tune B2-RL with train-only lateral response-sensitivity preservation."""

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
from torch.utils.data import DataLoader, Subset

from interactive_behavior_world_model.data.dataset import CausalActionSelfRolloutDataset
from interactive_behavior_world_model.data.lateral_response_dataset import LateralResponsePairDataset
from interactive_behavior_world_model.data.manifest import load_benchmark_config
from interactive_behavior_world_model.evaluation.rollout import _torch_features
from interactive_behavior_world_model.policies.rolling_action_policy import (
    ActionDiffusionConfig,
    RollingActionDiffusion,
)
from interactive_behavior_world_model.scripts.train_action import _ema_update, _move, _validate
from interactive_behavior_world_model.scripts.train_action_b2 import _self_unfold
from interactive_behavior_world_model.scripts.train_action_b2r import (
    ResponsePairDataset,
    _paired_response_loss,
    _validate_response,
)


def _features(history, valid, batch, feature_mean, feature_std):
    lines = batch["map_polylines"]
    line_valid = batch["map_valid"]
    lane_valid = line_valid.any(-1)
    centers = (lines[..., 1] * line_valid).sum(-1) / line_valid.sum(-1).clamp_min(1)
    widths = (lines[..., 4] * line_valid).sum(-1) / line_valid.sum(-1).clamp_min(1)
    return (
        _torch_features(
            history,
            valid,
            batch["lengths"],
            batch["widths"],
            centers,
            widths,
            lane_valid,
        )
        - feature_mean
    ) / feature_std


def _clean_actions(model, noise, timestep, features, valid, mask):
    output = model.denoiser(noise, timestep, features, valid) * mask
    clean = model.predict_clean(noise, timestep, output).clamp(-6.0, 6.0)
    return clean * model.action_std + model.action_mean


def _lateral_direction_losses(student_response, family, avoidance_sign):
    """Return permissive event response and merge-specific yaw margins.

    The event metric accepts braking *or* away-yaw for merges.  Keeping the
    original permissive margin preserves that semantics, while the second
    term exposes whether training should explicitly purchase active lateral
    avoidance instead of satisfying the margin through braking alone.
    """
    merge = family == 0
    aligned_brake = -student_response[..., 0] / 0.2
    aligned_away = student_response[..., 1] * avoidance_sign[:, None] / 0.015
    merge_score = torch.maximum(aligned_brake, aligned_away)
    cutout_score = student_response[..., 0] / 0.2
    direction_score = torch.where(merge[:, None], merge_score, cutout_score)
    direction = F.relu(1.0 - direction_score).mean()
    merge_count = merge.sum() * student_response.shape[1]
    yaw_direction = (
        F.relu(1.0 - aligned_away) * merge[:, None]
    ).sum() / merge_count.clamp_min(1)
    return direction, yaw_direction


def _lateral_preservation_loss(
    model,
    teacher,
    anchor,
    batch,
    feature_mean,
    feature_std,
    *,
    timestep_value: int = 79,
):
    """Preserve a reactive teacher's paired action Jacobian, not its base action."""
    valid = batch["history_valid"]
    active = valid[:, -1]
    baseline_features = _features(
        batch["baseline_history"], valid, batch, feature_mean, feature_std
    )
    changed_features = _features(
        batch["changed_history"], valid, batch, feature_mean, feature_std
    )
    mask = (
        active[:, None, 1:]
        .expand(-1, model.config.action_horizon, -1)[..., None]
        .to(baseline_features.dtype)
    )
    noise = torch.randn(mask.shape[:-1] + (2,), device=mask.device) * mask
    timestep = torch.full(
        (len(valid),), int(timestep_value), dtype=torch.long, device=mask.device
    )
    student_base = _clean_actions(
        model, noise, timestep, baseline_features, valid, mask
    )
    student_changed = _clean_actions(
        model, noise, timestep, changed_features, valid, mask
    )
    with torch.no_grad():
        teacher_base = _clean_actions(
            teacher, noise, timestep, baseline_features, valid, mask
        )
        teacher_changed = _clean_actions(
            teacher, noise, timestep, changed_features, valid, mask
        )
        anchor_base = _clean_actions(
            anchor, noise, timestep, baseline_features, valid, mask
        )
    student_delta = student_changed - student_base
    teacher_delta = teacher_changed - teacher_base
    row = torch.arange(len(valid), device=valid.device)
    slot = batch["response_index"] - 1
    # The first three decisions govern the response-onset metric.  Physical
    # scales prevent tiny yaw-rate units from being ignored by acceleration.
    student_response = student_delta[row, :3, slot]
    teacher_response = teacher_delta[row, :3, slot]
    scale = student_response.new_tensor([0.35, 0.015])
    sensitivity = F.smooth_l1_loss(student_response / scale, teacher_response / scale)
    # A minimal directional margin complements distillation when the teacher
    # itself lies just below the benchmark's sustained-response threshold.
    direction, yaw_direction = _lateral_direction_losses(
        student_response,
        batch["family"],
        batch["avoidance_sign"],
    )
    slots = torch.arange(model.config.agents, device=valid.device)[None]
    unrelated = active[:, 1:] & (slots != slot[:, None])
    locality = (student_delta[:, :3].square().sum(-1) * unrelated[:, None]).sum() / (
        unrelated.sum() * 3
    ).clamp_min(1)
    # Anchor natural behavior to the checkpoint being fine-tuned.  The
    # response teacher can be an earlier model and is intentionally used only
    # for paired sensitivity; anchoring to it would regress the current base.
    base_anchor = F.smooth_l1_loss(
        student_base[:, :3] / scale, anchor_base[:, :3] / scale
    )
    return {
        "sensitivity": sensitivity,
        "lateral_direction": direction,
        "lateral_yaw_direction": yaw_direction,
        "lateral_locality": locality,
        "teacher_base_anchor": base_anchor,
    }


def _validate_lateral(
    model, teacher, anchor, loader, device, feature_mean, feature_std
):
    totals = {
        "sensitivity": 0.0,
        "lateral_direction": 0.0,
        "lateral_yaw_direction": 0.0,
        "lateral_locality": 0.0,
        "teacher_base_anchor": 0.0,
    }
    count = 0
    model.eval()
    teacher.eval()
    torch.manual_seed(24680)
    with torch.no_grad():
        for raw in loader:
            batch = {
                key: value.to(device, non_blocking=True) for key, value in raw.items()
            }
            values = _lateral_preservation_loss(
                model, teacher, anchor, batch, feature_mean, feature_std
            )
            size = len(batch["history_valid"])
            count += size
            for key in totals:
                totals[key] += float(values[key]) * size
    return {key: value / max(count, 1) for key, value in totals.items()}


def _load_model(checkpoint, device):
    model = RollingActionDiffusion(
        ActionDiffusionConfig(**checkpoint["model_config"]),
        checkpoint["action_mean"],
        checkpoint["action_std"],
    )
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device)


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
    teacher_source = torch.load(args.teacher_checkpoint, map_location="cpu")
    model = _load_model(source, device)
    teacher = _load_model(teacher_source, device).eval()
    anchor = _load_model(source, device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    ema = copy.deepcopy(model).eval()
    feature_mean = torch.as_tensor(source["feature_mean"], device=device)
    feature_std = torch.as_tensor(source["feature_std"], device=device)
    action_train = CausalActionSelfRolloutDataset(args.config, "train", seed=seed)
    action_val = CausalActionSelfRolloutDataset(args.config, "validation", seed=seed)
    lateral_train = LateralResponsePairDataset(args.config, "train", seed=seed)
    lateral_val = LateralResponsePairDataset(args.config, "validation", seed=seed)
    response_train = ResponsePairDataset(args.config, "train")
    response_val = ResponsePairDataset(args.config, "validation")
    if args.max_train:
        action_train_view = Subset(
            action_train, range(min(args.max_train, len(action_train)))
        )
    else:
        action_train_view = action_train
    if args.max_validation:
        action_val_view = Subset(
            action_val, range(min(args.max_validation, len(action_val)))
        )
    else:
        action_val_view = action_val
    lateral_val_view = Subset(
        lateral_val, range(min(args.lateral_validation_pairs, len(lateral_val)))
    )
    with np.load(
        Path(config["paths"]["output_dir"]) / "data/response_pairs_ma_idm_train_v3.npz",
        allow_pickle=False,
    ) as pairs:
        teacher_scale = torch.tensor(
            float(np.std(pairs["teacher_response_delta"][..., 2]).clip(0.05)),
            device=device,
        )
    make = lambda data, batch, shuffle: DataLoader(
        data,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    action_loader = make(action_train_view, args.batch_size, True)
    action_val_loader = make(action_val_view, args.batch_size, False)
    lateral_loader = make(lateral_train, args.lateral_batch_size, True)
    lateral_val_loader = make(lateral_val_view, args.lateral_batch_size, False)
    response_loader = make(response_train, args.response_batch_size, True)
    response_val_loader = make(response_val, args.response_batch_size, False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4
    )
    output = (
        Path(config["paths"]["output_dir"]) / "methods" / args.method_id / str(seed)
    )
    output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stale = 0
    history = []
    started = time.time()
    for epoch in range(args.epochs):
        action_train.set_epoch(epoch)
        lateral_train.set_epoch(epoch)
        model.train()
        lateral_iterator = iter(lateral_loader)
        response_iterator = iter(response_loader)
        keys = (
            "loss",
            "logged",
            "self",
            "longitudinal_response",
            "longitudinal_locality",
            "longitudinal_direction",
            "sensitivity",
            "lateral_direction",
            "lateral_yaw_direction",
            "lateral_locality",
            "teacher_base_anchor",
        )
        totals = {key: 0.0 for key in keys}
        count = 0
        tick = time.time()
        for raw in action_loader:
            try:
                lateral_raw = next(lateral_iterator)
            except StopIteration:
                lateral_iterator = iter(lateral_loader)
                lateral_raw = next(lateral_iterator)
            try:
                response_raw = next(response_iterator)
            except StopIteration:
                response_iterator = iter(response_loader)
                response_raw = next(response_iterator)
            batch = _move(raw, device, feature_mean, feature_std)
            lateral = {
                key: value.to(device, non_blocking=True)
                for key, value in lateral_raw.items()
            }
            response = {
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
            longitudinal = _paired_response_loss(
                model, response, feature_mean, feature_std, teacher_scale
            )
            paired = _lateral_preservation_loss(
                model, teacher, anchor, lateral, feature_mean, feature_std
            )
            loss = (
                logged["loss"]
                + args.self_rollout_weight * robust["loss"]
                + args.response_weight * longitudinal["response"]
                + args.locality_weight * longitudinal["locality"]
                + args.direction_weight * longitudinal["direction"]
                + args.lateral_response_weight * paired["sensitivity"]
                + args.lateral_direction_weight * paired["lateral_direction"]
                + args.lateral_yaw_direction_weight * paired["lateral_yaw_direction"]
                + args.lateral_locality_weight * paired["lateral_locality"]
                + args.base_anchor_weight * paired["teacher_base_anchor"]
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
                "longitudinal_response": longitudinal["response"],
                "longitudinal_locality": longitudinal["locality"],
                "longitudinal_direction": longitudinal["direction"],
                **paired,
            }
            for key in totals:
                totals[key] += float(values[key].detach()) * size
        logged_val = _validate(
            ema, action_val_loader, device, feature_mean, feature_std
        )
        longitudinal_val = _validate_response(
            ema, response_val_loader, device, feature_mean, feature_std, teacher_scale
        )
        lateral_val_metrics = _validate_lateral(
            ema, teacher, anchor, lateral_val_loader, device, feature_mean, feature_std
        )
        criterion = (
            logged_val["loss"]
            + args.response_weight * longitudinal_val["response"]
            + args.locality_weight * longitudinal_val["locality"]
            + args.direction_weight * longitudinal_val["direction"]
            + args.lateral_response_weight * lateral_val_metrics["sensitivity"]
            + args.lateral_direction_weight * lateral_val_metrics["lateral_direction"]
            + args.lateral_yaw_direction_weight
            * lateral_val_metrics["lateral_yaw_direction"]
            + args.lateral_locality_weight * lateral_val_metrics["lateral_locality"]
            + args.base_anchor_weight * lateral_val_metrics["teacher_base_anchor"]
        )
        row = {
            "epoch": epoch + 1,
            **{f"train_{key}": value / count for key, value in totals.items()},
            "validation_logged_loss": logged_val["loss"],
            **{
                f"validation_longitudinal_{key}": value
                for key, value in longitudinal_val.items()
            },
            **{
                f"validation_lateral_{key}": value
                for key, value in lateral_val_metrics.items()
            },
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
                    "method_id": args.method_id,
                    "lateral_teacher_method_id": teacher_source.get("method_id"),
                    "lateral_response_weight": args.lateral_response_weight,
                    "lateral_direction_weight": args.lateral_direction_weight,
                    "lateral_yaw_direction_weight": args.lateral_yaw_direction_weight,
                    "lateral_locality_weight": args.lateral_locality_weight,
                    "base_anchor_weight": args.base_anchor_weight,
                },
                output / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    report = {
        "method_id": args.method_id,
        "fit_seed": seed,
        "base_checkpoint": str(args.base_checkpoint),
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "training_rows": len(action_train_view),
        "validation_rows": len(action_val_view),
        "lateral_train_pairs": len(lateral_train),
        "lateral_validation_pairs": len(lateral_val_view),
        "best_validation_criterion": best,
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
    p.add_argument("--teacher-checkpoint", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260919)
    p.add_argument("--method-id", default="rolling_action_b2rlp")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1.0e-5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--response-batch-size", type=int, default=64)
    p.add_argument("--lateral-batch-size", type=int, default=64)
    p.add_argument("--lateral-validation-pairs", type=int, default=8192)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--self-rollout-weight", type=float, default=0.25)
    p.add_argument("--response-weight", type=float, default=0.10)
    p.add_argument("--locality-weight", type=float, default=0.02)
    p.add_argument("--direction-weight", type=float, default=0.05)
    p.add_argument("--trajectory-weight", type=float, default=0.20)
    p.add_argument("--lateral-trajectory-multiplier", type=float, default=10.0)
    p.add_argument("--lateral-response-weight", type=float, default=0.05)
    p.add_argument("--lateral-direction-weight", type=float, default=0.01)
    p.add_argument("--lateral-yaw-direction-weight", type=float, default=0.0)
    p.add_argument("--lateral-locality-weight", type=float, default=0.02)
    p.add_argument("--base-anchor-weight", type=float, default=0.0)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-validation", type=int, default=0)
    p.add_argument("--device", default="auto")
    print(json.dumps(train(p.parse_args()), indent=2))


if __name__ == "__main__":
    main()

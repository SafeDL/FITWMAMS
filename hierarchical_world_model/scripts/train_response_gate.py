#!/usr/bin/env python3
"""Train the two-layer response gate from causal local samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from hierarchical_world_model.src.response_gate import MODE_MANEUVER, ResponseGate
from ads_learning.verify_protocol import verify_protocol_lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--level", choices=("L1", "L2"), default="L2")
    parser.add_argument(
        "--split-group",
        choices=("auto", "recording", "interaction", "event"),
        default="auto",
        help="Leakage-safe validation grouping; auto prefers recording_id.",
    )
    parser.add_argument("--protocol-lock", type=Path)
    args = parser.parse_args()
    protocol = (
        None
        if args.protocol_lock is None
        else verify_protocol_lock(args.protocol_lock, Path(__file__).resolve().parents[2])
    )
    if protocol is not None and not protocol["verified"]:
        raise RuntimeError(f"protocol lock mismatch: {protocol['mismatches'][:3]}")
    torch.manual_seed(args.seed)
    with np.load(args.data) as values:
        features = torch.from_numpy(np.asarray(values["features"], np.float32))
        mode = torch.from_numpy(np.asarray(values["mode"], np.int64))
        gate = torch.from_numpy(np.asarray(values["gate"], np.float32))
        event_id = np.asarray(
            values["event_id"] if "event_id" in values else np.arange(len(features)).astype(str)
        ).astype(str)
        source = np.asarray(
            values["source"] if "source" in values else np.full(len(features), "unknown")
        ).astype(str)
        frame_index = np.asarray(
            values["frame_index"] if "frame_index" in values else np.arange(len(features)),
            np.int64,
        )
        actor_index = np.asarray(
            values["actor_index"] if "actor_index" in values else np.zeros(len(features)),
            np.int64,
        )
        recording_id = np.asarray(
            values["recording_id"] if "recording_id" in values else np.full(len(features), -1)
        ).astype(str)
        interaction_group = np.asarray(
            values["interaction_group"] if "interaction_group" in values else event_id
        ).astype(str)
        data_protocol_id = str(values["protocol_id"].item()) if "protocol_id" in values else None
        data_protocol_sha256 = (
            str(values["protocol_lock_sha256"].item())
            if "protocol_lock_sha256" in values else None
        )
        base_actions = (
            np.asarray(values["base_actions"], np.float32)
            if "base_actions" in values
            else None
        )
        rule_actions = (
            np.asarray(values["rule_actions"], np.float32)
            if "rule_actions" in values
            else None
        )
        target_actions = (
            np.asarray(values["target_actions"], np.float32)
            if "target_actions" in values
            else None
        )
        target_authority = (
            np.asarray(values["target_authority"], np.float32)
            if "target_authority" in values
            else None
        )
    if protocol is not None and data_protocol_sha256 != protocol["protocol_lock_sha256"]:
        raise ValueError("response-gate data was collected under a different protocol lock")
    if args.level == "L1":
        mode = mode.ne(0).long()
    mode_count = 2 if args.level == "L1" else 3
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must lie in (0,1)")
    rng = np.random.default_rng(args.seed)
    available_groups = {
        "recording": recording_id,
        "interaction": interaction_group,
        "event": event_id,
    }
    if args.split_group == "auto":
        if len(np.unique(recording_id)) >= 2 and not np.all(recording_id == "-1"):
            split_group_name = "recording"
        elif len(np.unique(interaction_group)) >= 2:
            split_group_name = "interaction"
        else:
            split_group_name = "event"
    else:
        split_group_name = args.split_group
    split_group = available_groups[split_group_name]
    groups = np.unique(split_group)
    if len(groups) < 2:
        raise ValueError(f"{split_group_name} split requires at least two groups")
    rng.shuffle(groups)
    validation_group_count = max(1, int(round(len(groups) * args.validation_fraction)))
    validation_groups = set(groups[:validation_group_count])
    validation_mask_numpy = np.asarray([value in validation_groups for value in split_group])
    if validation_mask_numpy.all():
        raise ValueError("group split left no training samples")
    train_indices = torch.from_numpy(np.flatnonzero(~validation_mask_numpy))
    validation_indices = torch.from_numpy(np.flatnonzero(validation_mask_numpy))
    # Causal adjacent-frame pairs provide a deliberately small temporal
    # regularizer. Pair construction never crosses an event, actor, or the
    # held-out validation grouping.
    train_numpy = train_indices.numpy()
    pair_order = np.lexsort(
        (
            frame_index[train_numpy],
            actor_index[train_numpy],
            event_id[train_numpy],
        )
    )
    ordered_train = train_numpy[pair_order]
    pair_left = ordered_train[:-1]
    pair_right = ordered_train[1:]
    adjacent = (
        (event_id[pair_left] == event_id[pair_right])
        & (actor_index[pair_left] == actor_index[pair_right])
        & (frame_index[pair_right] == frame_index[pair_left] + 1)
    )
    temporal_left = torch.from_numpy(pair_left[adjacent])
    temporal_right = torch.from_numpy(pair_right[adjacent])
    device = torch.device(args.device)
    train_class_counts = torch.bincount(
        mode[train_indices], minlength=mode_count
    ).float()
    class_weights = (
        train_class_counts.max() / train_class_counts.clamp_min(1.0)
    ).pow(0.75).to(device)
    class_weights /= class_weights.mean()
    model = ResponseGate(
        input_dim=features.shape[-1], mode_count=mode_count
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    for _ in range(args.epochs):
        order = train_indices[torch.randperm(len(train_indices), generator=generator)]
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            x, y_mode, y_gate = features[index].to(device), mode[index].to(device), gate[index].to(device)
            output = model(x)
            active = y_mode.ne(0)
            gate_loss = (
                F.smooth_l1_loss(output.gate[active], y_gate[active])
                if args.level == "L2" and bool(active.any())
                else output.logits.sum() * 0.0
            )
            loss = (
                F.cross_entropy(output.logits, y_mode, weight=class_weights)
                + 0.5 * gate_loss
            )
            if len(temporal_left):
                pair_count = min(len(index), len(temporal_left))
                pair_selection = torch.randint(
                    len(temporal_left),
                    (pair_count,),
                    generator=generator,
                )
                left_index = temporal_left[pair_selection]
                right_index = temporal_right[pair_selection]
                left_output = model(features[left_index].to(device))
                right_output = model(features[right_index].to(device))
                same_target = mode[left_index].eq(mode[right_index]).to(device)
                pair_weight = torch.where(
                    same_target,
                    torch.ones(pair_count, device=device),
                    torch.full((pair_count,), 0.1, device=device),
                )
                probability_change = (
                    torch.softmax(left_output.logits, dim=-1)
                    - torch.softmax(right_output.logits, dim=-1)
                ).abs().mean(dim=-1)
                gate_change = (left_output.gate - right_output.gate).abs()
                temporal_loss = (
                    pair_weight * (probability_change + 0.5 * gate_change)
                ).mean()
                loss = loss + 0.02 * temporal_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.save(args.output)
    model.eval()
    with torch.no_grad():
        validation_output = model(features[validation_indices].to(device))
        prediction = validation_output.mode.cpu()
        target = mode[validation_indices]
        evaluation_prediction = (
            prediction.eq(MODE_MANEUVER).long()
            if args.level == "L1"
            else prediction
        )
        confusion = torch.zeros(mode_count, mode_count, dtype=torch.long)
        for actual, predicted in zip(target, evaluation_prediction):
            confusion[actual, predicted] += 1
        recall = confusion.diag() / confusion.sum(dim=1).clamp_min(1)
        active = target.ne(0)
        gate_mae = (
            float(
                (validation_output.gate.cpu()[active] - gate[validation_indices][active])
                .abs()
                .mean()
            )
            if args.level == "L2" and bool(active.any())
            else None
        )
    validation_numpy = validation_indices.numpy()
    validation_event = event_id[validation_numpy]
    validation_frame = frame_index[validation_numpy]
    validation_actor = actor_index[validation_numpy]
    true_active = target.numpy() != 0
    predicted_active = evaluation_prediction.numpy() != 0
    action_metrics = None
    if (
        base_actions is not None
        and rule_actions is not None
        and target_actions is not None
    ):
        base_validation = base_actions[validation_numpy]
        rule_validation = rule_actions[validation_numpy]
        predicted_actions = base_validation.copy()
        deployment_mode = prediction.numpy()
        predicted_gate = validation_output.gate.cpu().numpy()
        response_selected = deployment_mode != 0
        maneuver_selected = deployment_mode == MODE_MANEUVER
        predicted_actions[response_selected, 0] = (
            base_validation[response_selected, 0]
            + predicted_gate[response_selected]
            * (
                rule_validation[response_selected, 0]
                - base_validation[response_selected, 0]
            )
        )
        predicted_actions[maneuver_selected, 1] = rule_validation[
            maneuver_selected, 1
        ]
        action_error = np.abs(
            predicted_actions - target_actions[validation_numpy]
        )
        action_metrics = {
            "mean_absolute_acceleration_error_mps2": float(
                action_error[:, 0].mean()
            ),
            "mean_absolute_yaw_rate_error_rps": float(
                action_error[:, 1].mean()
            ),
            "active_target_acceleration_error_mps2": (
                float(action_error[true_active, 0].mean())
                if true_active.any()
                else None
            ),
            "active_target_yaw_rate_error_rps": (
                float(action_error[true_active, 1].mean())
                if true_active.any()
                else None
            ),
        }
    event_recalls = []
    delay_frames = []
    for current_event in np.unique(validation_event):
        selected_event = validation_event == current_event
        event_has_response = bool(true_active[selected_event].any())
        if event_has_response:
            event_recalls.append(bool(predicted_active[selected_event].any()))
        for current_actor in np.unique(validation_actor[selected_event]):
            selected = selected_event & (validation_actor == current_actor)
            true_frames = validation_frame[selected & true_active]
            if not len(true_frames):
                continue
            first_true = int(true_frames.min())
            predicted_frames = validation_frame[
                selected & predicted_active & (validation_frame >= first_true)
            ]
            if len(predicted_frames):
                delay_frames.append(int(predicted_frames.min()) - first_true)
    # Measure the deployed six-NPC batch, including CUDA synchronization.
    benchmark = features[:6].to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(500):
            model(benchmark)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    metrics = {
        "seed": args.seed,
        "level": args.level,
        "sample_counts": {
            "all": int(len(features)),
            "train": int(len(train_indices)),
            "validation": int(len(validation_indices)),
        },
        "event_counts": {
            "all": int(len(np.unique(event_id))),
            "train": int(len(np.unique(event_id[~validation_mask_numpy]))),
            "validation": int(len(np.unique(event_id[validation_mask_numpy]))),
        },
        "validation_grouping": {
            "column": split_group_name,
            "all": int(len(groups)),
            "train": int(len(groups) - validation_group_count),
            "validation": int(validation_group_count),
            "overlap": int(len(
                set(split_group[~validation_mask_numpy])
                & set(split_group[validation_mask_numpy])
            )),
        },
        "training_temporal_pairs": int(len(temporal_left)),
        "temporal_regularization_weight": 0.02,
        "source_counts": {
            value: int((source == value).sum()) for value in np.unique(source)
        },
        "target_authority_by_source": (
            None
            if target_authority is None
            else {
                value: {
                    "mean": float(target_authority[source == value].mean()),
                    "minimum": float(target_authority[source == value].min()),
                    "maximum": float(target_authority[source == value].max()),
                }
                for value in np.unique(source)
            }
        ),
        "train_class_counts": [int(value) for value in train_class_counts],
        "class_weights": [float(value) for value in class_weights.cpu()],
        "validation_confusion_rows_true": confusion.tolist(),
        "validation_class_recall": [float(value) for value in recall],
        "validation_balanced_accuracy": float(recall.mean()),
        "validation_event_activation_recall": (
            float(np.mean(event_recalls)) if event_recalls else None
        ),
        "validation_irrelevant_rewrite_rate": float(
            predicted_active[~true_active].mean() if (~true_active).any() else 0.0
        ),
        "validation_active_gate_mae": gate_mae,
        "validation_action_error": action_metrics,
        "validation_mean_response_delay_s": (
            float(np.mean(delay_frames) * 0.04) if delay_frames else None
        ),
        "inference_six_actor_batch_ms": float(elapsed / 500 * 1.0e3),
        "protocol": protocol,
        "data_protocol_id": data_protocol_id,
        "data_protocol_lock_sha256": data_protocol_sha256,
    }
    metrics_path = args.metrics_output or args.output.with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

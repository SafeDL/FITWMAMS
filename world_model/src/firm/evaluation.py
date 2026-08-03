"""Held-out reconstruction and calibration evaluation for FIRM-WM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from world_model.src.core.initial_behavior_anchor import (
    FrozenLegacyFlowSchema,
    summarize_first_second_states,
)
from world_model.src.core.batching import to_device_batch
from world_model.src.core.metrics import interaction_metrics, physical_diagnostics
from world_model.src.ramp.distribution_evaluation import energy_score, univariate_crps
from world_model.src.core.sequential_dataset import (
    ensure_frozen_flow_behavior_anchor_cache,
    load_sequential_dataset,
    sequence_cache_owner_dir,
)
from world_model.src.core.utils import ensure_dir, file_sha256, save_json, select_device

from .train import _loader, load_firm_checkpoint


def _masked_mean(value: np.ndarray, mask: np.ndarray) -> float:
    weight = np.asarray(mask, dtype=np.float64)
    return float((np.asarray(value, dtype=np.float64) * weight).sum() / max(weight.sum(), 1.0))


def _background_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    ego: np.ndarray,
) -> dict[str, float]:
    """Compute reconstruction metrics for generated background vehicles only."""
    distance = np.linalg.norm(predicted[..., :2] - target[..., :2], axis=-1)
    velocity = np.linalg.norm(predicted[..., 2:4] - target[..., 2:4], axis=-1)
    acceleration = np.linalg.norm(predicted[..., 4:6] - target[..., 4:6], axis=-1)
    result: dict[str, float] = {
        "ADE_m": _masked_mean(distance, valid),
        "FDE_m": _masked_mean(distance[:, -1], valid[:, -1]),
        "velocity_mae_mps": _masked_mean(velocity, valid),
        "acceleration_mae_mps2": _masked_mean(acceleration, valid),
    }
    for second in range(1, min(5, int(predicted.shape[1] // 25)) + 1):
        frame = second * 25 - 1
        result[f"FDE_{second}s_m"] = _masked_mean(distance[:, frame], valid[:, frame])
        result[f"ADE_{second}s_m"] = _masked_mean(distance[:, : frame + 1], valid[:, : frame + 1])
    predicted_gap = np.linalg.norm(predicted[..., :2] - ego[:, :, None, :2], axis=-1)
    target_gap = np.linalg.norm(target[..., :2] - ego[:, :, None, :2], axis=-1)
    result["gap_mae_m"] = _masked_mean(np.abs(predicted_gap - target_gap), valid)
    result["relative_vx_mae_mps"] = _masked_mean(
        np.abs((predicted[..., 2] - ego[:, :, None, 2]) - (target[..., 2] - ego[:, :, None, 2])),
        valid,
    )
    return result


def _behavior_anchor_error(
    predicted: torch.Tensor,
    target_valid: torch.Tensor,
    initial: torch.Tensor,
    initial_valid: torch.Tensor,
    anchor: torch.Tensor,
    anchor_valid: torch.Tensor,
) -> tuple[float, list[float]]:
    states = torch.cat((initial[:, None], predicted[:, :25]), dim=1)
    valid = torch.cat((initial_valid[:, None], target_valid[:, :25]), dim=1)
    summary, summary_valid = summarize_first_second_states(states, valid)
    mask = summary_valid[:, 1:] & anchor_valid.bool()
    error = (summary[:, 1:] - anchor).abs()
    per_feature = []
    for feature in range(error.shape[-1]):
        per_feature.append(
            float((error[..., feature] * mask.float()).sum().cpu() / mask.float().sum().clamp_min(1.0).cpu())
        )
    return float(np.mean(per_feature)), per_feature


def _prefix_nll(model, batch: dict[str, torch.Tensor], rollout: dict[str, torch.Tensor]) -> float:
    current = batch["agent_states"][:, 24]
    values = []
    for response in range(rollout["action_contexts"].shape[1]):
        controls, valid = model._target_controls(batch, response)
        jerks = model._target_jerks(current, controls, valid)
        values.append(
            model.action_flow.nll(
                jerks[:, : model.cfg.execute_frames],
                valid[:, : model.cfg.execute_frames],
                rollout["action_contexts"][:, response],
                center=rollout["raw_joint_jerk_centres"][:, response, : model.cfg.execute_frames],
            )
        )
        current = rollout["predicted_states"][:, (response + 1) * model.cfg.execute_frames - 1]
    return float(torch.stack(values).mean().cpu())


def _calibration_batch(
    model,
    batch: dict[str, torch.Tensor],
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    draws = []
    for draw in range(samples):
        rollout = model.rollout_roll_mode(batch, seed=seed + draw * 1009, deterministic=False)
        draws.append(rollout["predicted_states"][:, :, 1:].cpu().numpy())
    generated = np.stack(draws)
    target = batch["agent_states"][:, 25:, 1:].cpu().numpy()
    valid = batch["agent_valid"][:, 25:, 1:].cpu().numpy().astype(bool)
    position = generated[..., :2]
    observed = target[..., :2]
    mask = np.broadcast_to(valid[..., None], observed.shape)
    rank = (position < observed[None]).sum(axis=0)
    coverage: dict[str, tuple[int, int]] = {}
    for level in (0.5, 0.8, 0.9, 0.95):
        lower, upper = np.quantile(
            position, ((1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0), axis=0
        )
        covered = ((observed >= lower) & (observed <= upper) & mask).sum()
        coverage[str(level)] = (int(covered), int(mask.sum()))
    return {
        "energy_score": energy_score(generated, target, valid),
        "crps": univariate_crps(position, observed, valid),
        "rank_counts": np.bincount(rank[mask].reshape(-1), minlength=samples + 1),
        "coverage": coverage,
    }


def evaluate_firm_world_model(
    config: dict[str, Any],
    *,
    config_dir: Path,
    checkpoint: Path,
    max_sequences: int = 0,
) -> dict[str, Any]:
    paths, evaluation = config["paths"], config.get("evaluation", {})
    output = Path(paths["output_dir"])
    output = output if output.is_absolute() else (config_dir / output).resolve()
    evaluation_dir = ensure_dir(output / "evaluation")
    arrays, manifest = load_sequential_dataset(
        sequence_cache_owner_dir(config, config_dir=config_dir)
    )
    schema_raw = paths.get("flow_schema")
    if not schema_raw:
        raise ValueError("FIRM evaluation requires paths.flow_schema")
    schema = FrozenLegacyFlowSchema.load(
        Path(schema_raw) if Path(schema_raw).is_absolute() else (config_dir / schema_raw).resolve()
    )
    arrays.update(
        ensure_frozen_flow_behavior_anchor_cache(
            sequence_cache_owner_dir(config, config_dir=config_dir), arrays, manifest, schema
        )
    )
    device = select_device(evaluation.get("device", "auto"))
    model = load_firm_checkpoint(checkpoint, device=device)
    loader = _loader(
        arrays,
        "test",
        batch_size=int(evaluation.get("batch_size", 64)),
        maximum=int(max_sequences or evaluation.get("max_sequences", 0)),
        shuffle=False,
        seed=int(evaluation.get("seed", 123)),
        num_workers=int(evaluation.get("num_workers", 0)),
    )
    predicted, target, masks, tails = [], [], [], []
    prefix_nll = []
    anchor_errors: list[float] = []
    anchor_by_feature: list[list[float]] = []
    calibration = {
        "energy": [],
        "crps": [],
        "rank_counts": None,
        "coverage": {str(level): [0, 0] for level in (0.5, 0.8, 0.9, 0.95)},
    }
    calibration_samples = int(evaluation.get("calibration_samples", 8))
    replay_max_error: float | None = None
    with torch.no_grad():
        for batch_index, values in enumerate(loader):
            batch = to_device_batch(values, loader.field_names, device)
            rollout = model.rollout_roll_mode(
                batch, seed=int(evaluation.get("seed", 123)) + batch_index * 997, deterministic=True
            )
            if replay_max_error is None:
                replay_seed = int(evaluation.get("seed", 123)) + 800001
                first = model.rollout_roll_mode(batch, seed=replay_seed, deterministic=False)
                second = model.rollout_roll_mode(batch, seed=replay_seed, deterministic=False)
                replay_max_error = float(
                    (first["predicted_states"] - second["predicted_states"]).abs().max().cpu()
                )
            predicted.append(rollout["predicted_states"].cpu().numpy())
            target.append(rollout["target_states"].cpu().numpy())
            masks.append(rollout["target_valid"].cpu().numpy())
            tails.append(batch["is_evt_tail"].cpu().numpy().astype(bool))
            prefix_nll.append(_prefix_nll(model, batch, rollout))
            error, by_feature = _behavior_anchor_error(
                rollout["predicted_states"],
                rollout["target_valid"],
                batch["agent_states"][:, 24],
                batch["agent_valid"][:, 24],
                batch["behavior_anchor_raw"],
                batch["behavior_anchor_valid"],
            )
            anchor_errors.append(error)
            anchor_by_feature.append(by_feature)
            measured = _calibration_batch(
                model,
                batch,
                seed=int(evaluation.get("seed", 123)) + batch_index * 11939,
                samples=calibration_samples,
            )
            calibration["energy"].append(float(measured["energy_score"]))
            calibration["crps"].append(float(measured["crps"]))
            if calibration["rank_counts"] is None:
                calibration["rank_counts"] = measured["rank_counts"].astype(np.int64)
            else:
                calibration["rank_counts"] += measured["rank_counts"].astype(np.int64)
            for level, (covered, count) in measured["coverage"].items():
                calibration["coverage"][level][0] += covered
                calibration["coverage"][level][1] += count
    pred = np.concatenate(predicted)
    tgt = np.concatenate(target)
    mask = np.concatenate(masks)
    tail = np.concatenate(tails)
    # The ego is externally replayed into every rollout and therefore has
    # identically zero reconstruction error.  FIRM's primary reconstruction
    # metric must score only the generated background traffic; otherwise the
    # reported ADE/FDE is diluted by an observed, non-generated vehicle.
    background_pred, background_tgt, background_mask = pred[:, :, 1:], tgt[:, :, 1:], mask[:, :, 1:]
    full = _background_metrics(background_pred, background_tgt, background_mask, tgt[:, :, 0])
    one = _background_metrics(
        background_pred[:, :25],
        background_tgt[:, :25],
        background_mask[:, :25],
        tgt[:, :25, 0],
    )
    physical = physical_diagnostics(
        pred[:, :, 1:], mask[:, :, 1:], ego_future_states=tgt[:, :, 0], actions=pred[:, :, 1:, 4:6], slot_names=None
    )
    interaction = interaction_metrics(
        pred[:, :, 1:], tgt[:, :, 1:], mask[:, :, 1:], ego_future_states=tgt[:, :, 0]
    )
    evt = (
        _background_metrics(
            background_pred[tail], background_tgt[tail], background_mask[tail], tgt[tail, :, 0]
        )
        if tail.any()
        else {"available": False}
    )
    rank_counts = np.asarray(calibration["rank_counts"], np.int64)
    expected = rank_counts.sum() / max(len(rank_counts), 1)
    calibration_report = {
        "num_random_rollouts_per_condition": calibration_samples,
        "prefix_nll": float(np.mean(prefix_nll)),
        "energy_score": float(np.nanmean(calibration["energy"])),
        "crps": float(np.nanmean(calibration["crps"])),
        "rank_histogram": {
            "counts": rank_counts.tolist(),
            "chi_square_uniform": float(((rank_counts - expected) ** 2 / max(expected, 1.0)).sum()),
        },
        "conditional_coverage": {
            level: covered / max(count, 1)
            for level, (covered, count) in calibration["coverage"].items()
        },
    }
    report = {
        "model_type": model.model_type,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "sequence_cache": manifest,
        "test_sequences": int(len(pred)),
        "one_second_conditional_reconstruction": one,
        "five_second_roll_mode": full,
        "evt_tail": evt,
        "physical_diagnostics": physical,
        "interaction_metrics": interaction,
        "behavior_anchor_execution_l1": float(np.mean(anchor_errors)),
        "behavior_anchor_execution_l1_by_feature": np.mean(anchor_by_feature, axis=0).tolist(),
        "calibration_metrics": calibration_report,
        "fixed_seed_replay_max_abs_error": replay_max_error,
        "information_conditions": {
            "start_uses_single_c0_frame": True,
            "start_uses_frozen_flow_behavior_anchor": True,
            "learned_highd_map_encoder": False,
            "roll_reads_future_ego_or_background": False,
            "baseline_checkpoint_loaded": False,
        },
    }
    save_json(report, evaluation_dir / "evaluation_summary.json")
    return report

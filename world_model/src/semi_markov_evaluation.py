"""Causal-prior evaluation for the semi-Markov relational world model."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import interaction_metrics, physical_diagnostics
from .initial_behavior_anchor import FrozenLegacyFlowSchema
from .semi_markov_train import _loader, _to_batch, load_semi_markov_checkpoint
from .sequential_dataset import ensure_frozen_flow_behavior_anchor_cache, load_sequential_dataset, sequence_cache_owner_dir
from .utils import load_json, save_json, select_device


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, bool)
    return float(np.asarray(values, np.float64)[mask].mean()) if mask.any() else float("nan")


def _metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    dist = np.linalg.norm(pred[..., :2] - target[..., :2], axis=-1)
    result: dict[str, float] = {
        "ADE_m": _masked_mean(dist, mask),
        "FDE_m": _masked_mean(dist[:, -1], mask[:, -1]),
        "velocity_mae_mps": _masked_mean(np.linalg.norm(pred[..., 2:4] - target[..., 2:4], axis=-1), mask),
        "acceleration_mae_mps2": _masked_mean(np.linalg.norm(pred[..., 4:6] - target[..., 4:6], axis=-1), mask),
    }
    for second in range(1, min(5, int(pred.shape[1] // 25)) + 1):
        frame = second * 25 - 1
        result[f"FDE_{second}s_m"] = _masked_mean(dist[:, frame], mask[:, frame])
        result[f"ADE_{second}s_m"] = _masked_mean(dist[:, : frame + 1], mask[:, : frame + 1])
    ego_pred, ego_target = pred[:, :, :1], target[:, :, :1]
    p_gap = np.linalg.norm(pred[..., :2] - ego_pred[..., :2], axis=-1)
    t_gap = np.linalg.norm(target[..., :2] - ego_target[..., :2], axis=-1)
    p_rel = pred[..., 2] - ego_pred[..., 2]
    t_rel = target[..., 2] - ego_target[..., 2]
    result["gap_mae_m"] = _masked_mean(np.abs(p_gap - t_gap), mask)
    result["relative_vx_mae_mps"] = _masked_mean(np.abs(p_rel - t_rel), mask)
    return result


def _concat(items: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(items, axis=0) if items else np.zeros((0,), np.float32)


def _optional_report(
    evaluation: dict[str, Any], key: str, *, config_dir: Path,
) -> tuple[Path | None, dict[str, Any]]:
    """Resolve an optional JSON report configured relative to this experiment."""
    value = evaluation.get(key)
    if not value:
        return None, {}
    path = Path(value)
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    return path, load_json(path) if path.exists() else {}


def _relationship_distribution(states: np.ndarray, valid: np.ndarray, lane_width_m: float = 3.6) -> dict[str, float]:
    """Distribution of same/adjacent/unrelated agent relationships."""
    counts = np.zeros(3, dtype=np.float64)
    for sample_states, sample_valid in zip(states, valid):
        for frame_states, frame_valid in zip(sample_states, sample_valid):
            active = np.flatnonzero(frame_valid)
            if len(active) < 2:
                continue
            lane = np.round(frame_states[:, 1] / float(lane_width_m)).astype(np.int64)
            for left in active:
                for right in active:
                    if left >= right:
                        continue
                    difference = abs(int(lane[left]) - int(lane[right]))
                    counts[0 if difference == 0 else 1 if difference == 1 else 2] += 1
    total = counts.sum()
    values = counts / total if total else counts
    return {"same_lane": float(values[0]), "adjacent_lane": float(values[1]), "unrelated": float(values[2])}


def _duration_calibration(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    if probabilities.size == 0:
        return {"available": False}
    p = np.asarray(probabilities, np.float64).reshape(-1)
    y = np.asarray(targets, np.float64).reshape(-1)
    brier = float(np.mean((p - y) ** 2))
    ece = 0.0
    for low, high in zip(np.linspace(0.0, 0.9, 10), np.linspace(0.1, 1.0, 10)):
        pick = (p >= low) & ((p < high) if high < 1.0 else (p <= high))
        if pick.any():
            ece += float(pick.mean()) * abs(float(p[pick].mean()) - float(y[pick].mean()))
    return {
        "available": True, "brier_score": brier, "expected_calibration_error": float(ece),
        "mean_predicted_boundary_probability": float(p.mean()), "empirical_boundary_rate": float(y.mean()),
    }


def _counterfactual_ego_batch(batch: dict[str, Any], *, kind: str):
    """Replace only future observed ego physics for a causal response probe."""
    import torch

    result = dict(batch)
    states = batch["agent_states"].clone()
    dt = 0.04
    time = torch.arange(1, 126, dtype=states.dtype, device=states.device) * dt
    for item, ego_index in enumerate(batch["ego_index"].long().tolist()):
        initial = states[item, 24, ego_index].clone()
        if kind == "accelerate":
            longitudinal, lateral = 1.5, 0.0
        elif kind == "brake":
            longitudinal, lateral = -2.0, 0.0
        elif kind == "lateral":
            longitudinal, lateral = 0.0, 0.20
        else:  # pragma: no cover - caller-owned finite set
            raise ValueError(f"unknown response perturbation: {kind}")
        states[item, 25:, ego_index, 0] = initial[0] + initial[2] * time + 0.5 * longitudinal * time.square()
        states[item, 25:, ego_index, 1] = initial[1] + initial[3] * time + 0.5 * lateral * time.square()
        states[item, 25:, ego_index, 2] = initial[2] + longitudinal * time
        states[item, 25:, ego_index, 3] = initial[3] + lateral * time
        states[item, 25:, ego_index, 4] = longitudinal
        states[item, 25:, ego_index, 5] = lateral
    result["agent_states"] = states
    return result


def _controlled_response_metrics(model, loader, device, *, seed: int) -> dict[str, Any]:
    """Causal sensitivity diagnostics, not a claimed counterfactual truth test."""
    import torch

    identity_max_error = 0.0
    delta_sum = {name: 0.0 for name in ("accelerate", "brake", "lateral")}
    delta_count = {name: 0 for name in delta_sum}
    continuity_sum = {name: 0.0 for name in delta_sum}
    continuity_count = {name: 0 for name in delta_sum}
    latent_changed = {name: 0 for name in delta_sum}
    episodes = 0
    with torch.no_grad():
        offset = 0
        for values in loader:
            batch = _to_batch(values, loader.field_names, device)
            baseline = model.rollout_prior(batch, seed=int(seed) + offset, deterministic=True)
            repeated = model.rollout_prior(batch, seed=int(seed) + offset, deterministic=True)
            identity_max_error = max(
                identity_max_error,
                float((baseline["predicted_states"] - repeated["predicted_states"]).abs().max().cpu()),
            )
            control_valid = batch["agent_valid"][:, 24:149:5, 1:]
            for name in delta_sum:
                changed = model.rollout_prior(
                    _counterfactual_ego_batch(batch, kind=name), seed=int(seed) + offset, deterministic=True,
                )
                delta = torch.linalg.vector_norm(changed["controls"][:, :, 1:] - baseline["controls"][:, :, 1:], dim=-1)
                weight = control_valid.float()
                delta_sum[name] += float((delta * weight).sum().cpu())
                delta_count[name] += int(weight.sum().cpu())
                jump = torch.linalg.vector_norm(torch.diff(changed["controls"][:, :, 1:], dim=1), dim=-1)
                pair_valid = control_valid[:, 1:] & control_valid[:, :-1]
                continuity_sum[name] += float((jump * pair_valid.float()).sum().cpu())
                continuity_count[name] += int(pair_valid.sum().cpu())
                latent_changed[name] += sum(left != right for left, right in zip(baseline["latent_states"], changed["latent_states"]))
            episodes += len(baseline["latent_states"])
            offset += len(baseline["latent_states"])
    return {
        "available": bool(episodes),
        "protocol": {
            "future_ego_replaced_with": {
                "accelerate": "+1.5 m/s^2 longitudinal", "brake": "-2.0 m/s^2 longitudinal", "lateral": "+0.20 m/s^2 lateral",
            },
            "deterministic_prior": True,
            "ads_identity_not_an_input": True,
            "counterfactual_ground_truth_available": False,
        },
        "episodes": int(episodes),
        "same_physical_ego_identity_invariance_max_abs_error": float(identity_max_error),
        "background_control_delta_norm": {name: float(delta_sum[name] / max(delta_count[name], 1)) for name in delta_sum},
        "background_control_temporal_jump_norm": {name: float(continuity_sum[name] / max(continuity_count[name], 1)) for name in continuity_sum},
        "latent_path_changed_episode_rate": {name: float(latent_changed[name] / max(episodes, 1)) for name in latent_changed},
    }


def evaluate_semi_markov_world_model(
    config: dict[str, Any], *, config_dir: Path, checkpoint: Path | None = None, max_sequences: int = 0,
) -> dict[str, Any]:
    paths = config["paths"]
    out = Path(paths["output_dir"])
    if not out.is_absolute(): out = (config_dir / out).resolve()
    cache_owner = sequence_cache_owner_dir(config, config_dir=config_dir)
    arrays, manifest = load_sequential_dataset(cache_owner)
    evaluation = config.get("evaluation", {})
    device = select_device(str(evaluation.get("device", "auto")))
    checkpoint = checkpoint or out / "checkpoints" / "best_semi_markov_relational.pt"
    model = load_semi_markov_checkpoint(checkpoint, device=device)
    if model.uses_behavior_anchor:
        schema_value = config.get("paths", {}).get("flow_schema")
        if not schema_value:
            raise ValueError("M1 evaluation requires paths.flow_schema")
        schema_path = Path(schema_value)
        if not schema_path.is_absolute():
            schema_path = (config_dir / schema_path).resolve()
        schema = FrozenLegacyFlowSchema.load(schema_path)
        model.set_frozen_flow_schema(schema)
        arrays.update(ensure_frozen_flow_behavior_anchor_cache(cache_owner, arrays, manifest, schema))
    cold_start = evaluation.get("cold_start_history")
    if cold_start is not None:
        model.cfg = replace(model.cfg, cold_start_history=bool(cold_start))
    loader = _loader(arrays, "test", batch_size=int(evaluation.get("batch_size", 16)), maximum=int(max_sequences or evaluation.get("max_sequences", 0)), shuffle=False, seed=int(evaluation.get("seed", 123)))
    import torch
    all_pred: list[np.ndarray] = []; all_target: list[np.ndarray] = []; all_mask: list[np.ndarray] = []; all_tail: list[np.ndarray] = []
    posterior_boundaries: list[np.ndarray] = []; boundary_targets: list[np.ndarray] = []
    posterior_probs: list[np.ndarray] = []; prior_probs: list[np.ndarray] = []
    posterior_anchor_losses: list[float] = []
    causal_anchor_losses: list[float] = []
    states: list[list[int]] = []; durations: list[list[int]] = []
    with torch.no_grad():
        for values in loader:
            batch = _to_batch(values, loader.field_names, device)
            rollout = model.rollout_prior(batch, seed=int(evaluation.get("seed", 123)) + len(states), deterministic=True)
            all_pred.append(rollout["predicted_states"].cpu().numpy())
            all_target.append(rollout["target_states"].cpu().numpy())
            all_mask.append(rollout["target_valid"].cpu().numpy())
            all_tail.append(batch["is_evt_tail"].cpu().numpy().astype(bool))
            states.extend(rollout["latent_states"]); durations.extend(rollout["latent_durations"])
            posterior = model.forward_training(batch, teacher_forcing_ratio=1.0)
            posterior_boundaries.append(posterior["posterior_boundary_probs"][:, 1:].cpu().numpy())
            boundary_targets.append(posterior["boundary_target"][:, 1:].cpu().numpy())
            posterior_probs.append(posterior["posterior_raw_state_probs"].cpu().numpy())
            prior_probs.append(torch.softmax(posterior["prior_logits"], dim=-1).cpu().numpy())
            posterior_anchor_losses.append(float(posterior["anchor_loss"].cpu()))
            if model.uses_behavior_anchor:
                _raw, target_std, _agents, target_valid = model._batch_behavior_anchor(batch)
                causal_raw = rollout["causal_prior_anchor_raw"]
                causal_valid = rollout["causal_prior_anchor_valid"] & target_valid
                causal_std = model.frozen_flow_schema.standardize(causal_raw, causal_valid) if model.frozen_flow_schema else model.behavior_anchor.normalize(causal_raw)
                weight = causal_valid.float().unsqueeze(-1)
                causal_anchor_losses.append(float((causal_std.sub(target_std).abs() * weight).sum().div(weight.sum().clamp_min(1.0)).cpu()))
    pred, target, mask, tail = _concat(all_pred), _concat(all_target), _concat(all_mask), _concat(all_tail)
    full = _metrics(pred, target, mask)
    one_second = _metrics(pred[:, :25], target[:, :25], mask[:, :25])
    tail_metrics = _metrics(pred[tail], target[tail], mask[tail]) if len(tail) and tail.any() else {"available": False}
    physical = physical_diagnostics(
        pred[:, :, 1:], mask[:, :, 1:], ego_future_states=target[:, :, 0], slot_names=None,
    )
    interaction = interaction_metrics(pred[:, :, 1:], target[:, :, 1:], mask[:, :, 1:], ego_future_states=target[:, :, 0])
    predicted_relations = _relationship_distribution(pred[:, :, 1:], mask[:, :, 1:])
    target_relations = _relationship_distribution(target[:, :, 1:], mask[:, :, 1:])
    relation_tv = 0.5 * sum(abs(predicted_relations[key] - target_relations[key]) for key in predicted_relations)
    duration = _duration_calibration(_concat(posterior_boundaries), _concat(boundary_targets))
    q, p = _concat(posterior_probs), _concat(prior_probs)
    prior_posterior = {
        "mean_total_variation": float(0.5 * np.abs(q - p).sum(axis=-1).mean()) if q.size else float("nan"),
        "categorical_cross_entropy": float(-(q * np.log(np.maximum(p, 1.0e-8))).sum(axis=-1).mean()) if q.size else float("nan"),
    }
    baseline_path, baseline = _optional_report(
        evaluation, "required_baseline_summary", config_dir=config_dir,
    )
    baseline_one = baseline.get("closed_loop", {}).get("test", {})
    baseline_rollout = baseline.get("model_state_reconstruction", {}).get("test", {})
    horizon_comparison: dict[str, dict[str, float]] = {}
    for seconds in range(2, 6):
        reference = baseline_rollout.get(f"{seconds}_chunks", {})
        if not reference or f"ADE_{seconds}s_m" not in full:
            continue
        horizon_comparison[f"{seconds}s"] = {
            "candidate_ADE_m": float(full[f"ADE_{seconds}s_m"]),
            "baseline_ADE_m": float(reference.get("ADE_m", float("nan"))),
            "candidate_minus_baseline_ADE_m": float(full[f"ADE_{seconds}s_m"] - reference.get("ADE_m", np.nan)),
            "candidate_FDE_m": float(full[f"FDE_{seconds}s_m"]),
            "baseline_FDE_m": float(reference.get("FDE_m", float("nan"))),
            "candidate_minus_baseline_FDE_m": float(full[f"FDE_{seconds}s_m"] - reference.get("FDE_m", np.nan)),
            "candidate_gap_mae_m": float(interaction["gap_mae_m"]),
            "baseline_gap_mae_m": float(reference.get("gap_mae_m", float("nan"))),
        }
    cache_identity = f"{manifest.get('source_dataset', '')} {manifest.get('adapter', '')}".lower()
    complete_highd = not bool(manifest.get("bounded_development_cache", True)) and "highd" in cache_identity
    # A legacy summary alone is not a paired comparison: it can have a
    # different cache, rollout protocol, or (for CAT-K) a Flow with future
    # action summaries.  Formal promotion requires an explicit paired result.
    paired_path, paired_report = _optional_report(
        evaluation, "paired_baseline_summary", config_dir=config_dir,
    )
    current_hash = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    paired_information_symmetric = bool(paired_report.get("protocol", {}).get("promotion_information_symmetric", False))
    paired_baseline = bool(
        paired_report.get("protocol", {}).get("same_sequence", False)
        and int(paired_report.get("num_paired_sequences", 0)) == int(len(pred))
        and paired_report.get("candidate_checkpoint_sha256") == current_hash
        and paired_report.get("all_primary_error_gates_pass", False)
        and paired_information_symmetric
    )
    paired_long_path, paired_long_report = _optional_report(
        evaluation, "paired_long_horizon_baseline_summary", config_dir=config_dir,
    )
    paired_long_information_symmetric = bool(paired_long_report.get("protocol", {}).get("promotion_information_symmetric", False))
    paired_long = bool(
        paired_long_report.get("protocol", {}).get("same_sequence", False)
        and float(paired_long_report.get("protocol", {}).get("horizon_seconds", 0.0)) >= 5.0
        and int(paired_long_report.get("num_paired_sequences", 0)) == int(len(pred))
        and paired_long_report.get("candidate_checkpoint_sha256") == current_hash
        and paired_long_information_symmetric
    )
    long_errors = bool(paired_long_report.get("all_primary_error_gates_pass", False))
    long_relation_delta = paired_long_report.get("relationship_distribution", {}).get("candidate_minus_baseline_total_variation")
    long_relation_improved = long_relation_delta is not None and np.isfinite(float(long_relation_delta)) and float(long_relation_delta) < 0.0
    # Requirement 18(2): the causal five-second rollout must reduce either
    # accumulated paired error or relation-distribution drift.  A legacy
    # summary on its own is never enough—the paired artifact must use the
    # evaluated checkpoint and every held-out sequence.
    long_horizon_gate = paired_long and (long_errors or long_relation_improved)
    # Archived CAT-K reports may use the legacy START action summary computed
    # from a future trajectory.  Keep those reports visible for historical
    # reproducibility, but they cannot be promotion references under the
    # clean-START specification.  The primary paired paths above must instead
    # explicitly attest information symmetry.
    legacy_paired_path, legacy_paired_report = _optional_report(
        evaluation, "legacy_paired_baseline_summary", config_dir=config_dir,
    )
    legacy_paired_long_path, legacy_paired_long_report = _optional_report(
        evaluation, "legacy_paired_long_horizon_baseline_summary", config_dir=config_dir,
    )
    round_summary_path, round_report = _optional_report(
        evaluation, "round_evaluation_summary", config_dir=config_dir,
    )
    round_manifest = round_report.get("sequence_cache", {})
    round_identity = f"{round_manifest.get('source_dataset', '')} {round_manifest.get('adapter', '')}".lower()
    round_complete = (
        bool(round_report.get("test_sequences", 0))
        and not bool(round_manifest.get("bounded_development_cache", True))
        and "round" in round_identity
    )
    duration_gate = bool(
        duration.get("available", False)
        and np.isfinite(duration.get("brier_score", np.nan))
        and np.isfinite(duration.get("expected_calibration_error", np.nan))
        and duration.get("expected_calibration_error", np.inf) <= 0.10
        and np.isfinite(np.mean([value for item in durations for value in item]) if durations else np.nan)
        and (np.mean([value for item in durations for value in item]) if durations else 0.0) > 1.0
    )
    responsiveness_max = int(evaluation.get("responsiveness_max_sequences", 256))
    if max_sequences:
        responsiveness_max = min(responsiveness_max, int(max_sequences))
    response_loader = _loader(
        arrays, "test", batch_size=int(evaluation.get("batch_size", 16)), maximum=responsiveness_max,
        shuffle=False, seed=int(evaluation.get("seed", 123)) + 91,
    )
    controlled_response = _controlled_response_metrics(
        model, response_loader, device, seed=int(evaluation.get("seed", 123)) + 70_000,
    )
    summary_baseline_gate = bool(baseline_one) and all(one_second[key] <= float(baseline_one.get(key, np.inf)) for key in ("ADE_m", "FDE_m", "gap_mae_m"))
    baseline_gate = paired_baseline and summary_baseline_gate
    promotion_ready = complete_highd and paired_baseline and long_horizon_gate and round_complete and duration_gate and baseline_gate
    if not complete_highd:
        promotion_reason = "Requires complete held-out highD cache."
    elif not paired_baseline:
        promotion_reason = "Requires paired full-highD information-symmetric frozen-baseline comparison."
    elif not long_horizon_gate:
        promotion_reason = "Five-second paired error-accumulation / relationship-drift gate failed."
    elif not round_complete:
        promotion_reason = "Requires an independent full rounD evaluation summary."
    elif not duration_gate:
        promotion_reason = "Semi-Markov duration calibration/persistence gate failed."
    elif not baseline_gate:
        promotion_reason = "One-second frozen-baseline gate failed."
    else:
        promotion_reason = "All configured formal promotion gates passed."
    report = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": current_hash,
        "sequence_cache": manifest, "test_sequences": int(len(pred)), "one_second_conditional_reconstruction": one_second,
        "five_second_causal_prior_rollout": full, "evt_tail": tail_metrics,
        "physical_diagnostics": physical, "interaction_metrics": interaction,
        "relationship_distribution": {"predicted": predicted_relations, "target": target_relations, "total_variation": relation_tv},
        "duration_calibration": duration, "prior_posterior_consistency": prior_posterior,
        "behavior_anchor": {
            "variant": model.cfg.variant,
            "cold_start_history": bool(model.cfg.cold_start_history),
            "active_response_steps": int(model.cfg.behavior_anchor_response_steps),
            "posterior_anchor_l1": float(np.mean(posterior_anchor_losses)) if posterior_anchor_losses else 0.0,
            "causal_prior_anchor_l1": float(np.mean(causal_anchor_losses)) if causal_anchor_losses else 0.0,
        },
        "controlled_response": controlled_response,
        "frozen_baseline_horizon_comparison": {
            "paired_and_full_test": complete_highd and paired_baseline,
            "note": "Numbers become a promotion comparison only after the complete held-out highD cache is evaluated with paired bootstrap.",
            "paired_baseline_summary": str(paired_path) if paired_path is not None else None,
            "horizons": horizon_comparison,
        },
        "legacy_frozen_baseline_comparison": {
            "paired_baseline_summary": str(legacy_paired_path) if legacy_paired_path is not None else None,
            "paired_long_horizon_baseline_summary": str(legacy_paired_long_path) if legacy_paired_long_path is not None else None,
            "uses_future_flow_action_summary": bool(legacy_paired_report.get("protocol", {}).get("baseline_uses_future_flow_action_summary", False)),
            "one_second_all_primary_error_gates_pass": bool(legacy_paired_report.get("all_primary_error_gates_pass", False)),
            "five_second_all_primary_error_gates_pass": bool(legacy_paired_long_report.get("all_primary_error_gates_pass", False)),
            "five_second_relationship_total_variation_candidate_minus_baseline": legacy_paired_long_report.get("relationship_distribution", {}).get("candidate_minus_baseline_total_variation"),
            "promotion_eligible": False,
            "reason": "Historical diagnostic only: legacy CAT-K START uses a future-action summary.",
        },
        "paired_frozen_baseline_long_horizon": {
            "paired_and_full_test": complete_highd and paired_long,
            "paired_baseline_summary": str(paired_long_path) if paired_long_path is not None else None,
            "information_symmetric": paired_long_information_symmetric,
            "all_primary_error_gates_pass": long_errors,
            "relationship_total_variation_candidate_minus_baseline": float(long_relation_delta) if long_relation_delta is not None else None,
            "relationship_drift_improved": bool(long_relation_improved),
            "gate_pass": bool(long_horizon_gate),
        },
        "latent_path": {
            "mean_segments_per_episode": float(np.mean([len(item) for item in states])) if states else float("nan"),
            "mean_duration_response_steps": float(np.mean([value for item in durations for value in item])) if durations else float("nan"),
            "switches_per_episode": float(np.mean([max(len(item) - 1, 0) for item in states])) if states else float("nan"),
        },
        "promotion": {
            "eligible": promotion_ready,
            "one_second_not_weaker_than_frozen_baseline": baseline_gate if complete_highd and paired_baseline else False,
            "duration_calibrated_and_persistent": duration_gate,
            "full_highd_cache": complete_highd,
            "paired_frozen_baseline": paired_baseline,
            "five_second_error_or_relation_drift_improved": long_horizon_gate,
            "round_evaluation_complete": round_complete,
            "status": "pass" if promotion_ready else "not_promoted",
            "reason": promotion_reason,
        },
    }
    save_json(report, out / "semi_markov_evaluation_summary.json")
    return report

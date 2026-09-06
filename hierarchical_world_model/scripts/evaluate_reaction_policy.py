#!/usr/bin/env python3
"""Evaluate reaction policies under one autonomous runtime and reference protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.planner import complete_missing_background_plans, frozen_diffusion_plans  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.reaction_controller import (  # noqa: E402
    CalibratedResidualReactionController, IDMOnlyReactionController,
    IDMResidualReactionController, NoReactionController, RLResidualReactionController,
)
from hierarchical_world_model.src.reaction_evidence import (  # noqa: E402
    EVALUATION_FRAMES, ReactionEventReference, energy_score, event_window,
    recording_cluster_bootstrap,
)
from hierarchical_world_model.src.reaction_training import (  # noqa: E402
    PolicyTrainingConfig, ReactionRollout, reaction_controller_rollout,
)
from hierarchical_world_model.src.rule_models import RuleModelBundle  # noqa: E402
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from world_model.src.core.utils import ensure_dir, save_json, select_device  # noqa: E402


DEFAULT = ROOT / "hierarchical_world_model/config/reaction_policy.yaml"


def _arrays(bundle, rows: np.ndarray) -> dict[str, np.ndarray]:
    result = {name: np.asarray(bundle.arrays[name])[rows] for name in (
        "agent_states", "agent_valid", "map_polylines", "map_polyline_valid",
    )}
    result["row_index"] = np.asarray(rows, np.int64)
    return result


def _load_controller(name: str, path: Path | None, rule: RuleModelBundle, device: torch.device):
    if name == "frozen_hiqr":
        return NoReactionController().to(device).eval()
    if name == "idm_only":
        return IDMOnlyReactionController(rule).to(device).eval()
    if path is None:
        raise ValueError(f"{name} requires a checkpoint")
    payload = torch.load(path, map_location=device, weights_only=False)
    expected = {
        "a1_transfer": ("rl_residual", RLResidualReactionController),
        "a2_transfer": ("rl_residual_idm", lambda: IDMResidualReactionController(rule)),
        "calibrated_residual": (
            "calibrated_residual", lambda: CalibratedResidualReactionController(rule),
        ),
    }
    expected_mode, constructor = expected[name]
    if payload.get("controller_mode") != expected_mode:
        raise ValueError(f"{path} is not a {name} checkpoint")
    controller = constructor().to(device)
    controller.load_state_dict(payload["state_dict"], strict=True)
    return controller.eval()


def _concatenate(parts: list[ReactionRollout]) -> ReactionRollout:
    return ReactionRollout(
        states=np.concatenate([part.states for part in parts]),
        background_actions=np.concatenate([part.background_actions for part in parts]),
        base_background_actions=np.concatenate([part.base_background_actions for part in parts]),
        ego_actions=np.concatenate([part.ego_actions for part in parts]),
        controller_diagnostics={
            name: np.concatenate([part.controller_diagnostics[name] for part in parts])
            for name in parts[0].controller_diagnostics
        },
        collision=np.concatenate([part.collision for part in parts]),
        crashed=np.concatenate([part.crashed for part in parts]),
    )


def _rollout(
    model, arrays: dict[str, np.ndarray], plans: np.ndarray, controller,
    device: torch.device, config: PolicyTrainingConfig, *, seed: int,
    batch_size: int, profile: np.ndarray | None = None,
    deterministic: bool = True,
) -> ReactionRollout:
    parts = []
    for start in range(0, len(plans), batch_size):
        stop = min(start + batch_size, len(plans))
        torch.manual_seed(seed + start)
        parts.append(reaction_controller_rollout(
            model,
            states=arrays["agent_states"][start:stop],
            valid=arrays["agent_valid"][start:stop],
            soft_plans=plans[start:stop],
            maps=arrays["map_polylines"][start:stop],
            map_valid=arrays["map_polyline_valid"][start:stop],
            controller=controller,
            device=device,
            motion_seed=seed + start,
            config=config,
            deterministic_response=deterministic,
            ego_acceleration_offset=profile,
        ))
    return _concatenate(parts)


def factual_metrics(
    rollout: ReactionRollout, states: np.ndarray, valid: np.ndarray,
) -> dict[str, float]:
    target = states[:, ANCHOR_INDEX + 1:ANCHOR_INDEX + 150, 1:, :2]
    present = valid[:, ANCHOR_INDEX + 1:ANCHOR_INDEX + 150, 1:]
    error = np.linalg.norm(rollout.states[:, :, 1:, :2] - target, axis=-1)
    values = error[present]
    final_values = error[:, -1][present[:, -1]]
    return {
        "ade_m": float(values.mean()),
        "fde_m": float(final_values.mean()),
        "p95_m": float(np.quantile(values, 0.95)),
    }


def _future_features(
    rollout: ReactionRollout, onset: int, follower: int, frames: int = EVALUATION_FRAMES,
) -> np.ndarray:
    action = rollout.background_actions[:, :, follower - 1, 0]
    current = action[:, onset:onset + frames]
    previous = action[:, onset - 1:onset + frames - 1]
    state = rollout.states[:, onset:onset + frames]
    leader_state, follower_state = state[:, :, 0], state[:, :, follower]
    gap = leader_state[..., 0] - follower_state[..., 0] - 4.8
    closing = follower_state[..., 2] - leader_state[..., 2]
    ttc = np.where(closing > 1.0e-4, gap / np.maximum(closing, 1.0e-4), 10.0)
    return np.stack((
        current,
        np.abs(current - previous) / 0.04,
        follower_state[..., 2],
        gap,
        closing,
        np.clip(ttc, 0.0, 10.0),
    ), axis=-1).astype(np.float32)


def _latency(acceleration: np.ndarray, initial: np.ndarray) -> np.ndarray:
    crossed = acceleration <= initial[..., None] - 0.1
    return np.where(crossed.any(-1), crossed.argmax(-1), EVALUATION_FRAMES).astype(np.float32)


def _event_statistics(features: np.ndarray, initial_acceleration: np.ndarray) -> np.ndarray:
    primary = features[..., :EVALUATION_FRAMES, :]
    return np.stack((
        _latency(primary[..., 0], initial_acceleration),
        primary[..., 0].min(-1),
        np.quantile(primary[..., 1], 0.95, axis=-1),
        primary[..., 3].mean(-1),
        primary[..., 4].mean(-1),
        np.linalg.norm(features[..., -1, [2, 3, 4, 5]], axis=-1),
    ), axis=-1)


def _train_scales(reference: ReactionEventReference) -> dict[str, float]:
    indices = reference.events.indices(reference.supported_cells)
    observed = event_window(reference.events, recovery=True)[indices]
    initial = reference.events.initial_conditions[indices, 4]
    statistics = _event_statistics(observed[:, None], initial[:, None])[:, 0]
    names = ("latency", "peak_acceleration", "jerk", "gap", "closing", "recovery")
    return {
        name: float(max(np.quantile(statistics[:, index], 0.75) - np.quantile(statistics[:, index], 0.25), 1.0e-3))
        for index, name in enumerate(names)
    }


def evaluate_events(
    model, arrays: dict[str, np.ndarray], plans: np.ndarray,
    reference: ReactionEventReference, controllers: dict[str, object],
    device: torch.device, config: PolicyTrainingConfig, *, seed: int,
    train_reference: ReactionEventReference, limit: int | None,
) -> dict:
    lookup = {int(row): index for index, row in enumerate(arrays["row_index"])}
    selected = [
        index for index in reference.events.indices(reference.supported_cells)
        if reference.events.leader_slot[index] == 0
        and int(reference.events.row_index[index]) in lookup
    ]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise RuntimeError("held-out split has no replayable supported events")
    observed = event_window(reference.events, recovery=True)
    arm_scores = {name: [] for name in controllers}
    arm_errors = {name: [] for name in controllers}
    records = reference.events.recording_id[selected]
    cells = reference.events.cell[selected]
    for order, event_index in enumerate(selected):
        row = lookup[int(reference.events.row_index[event_index])]
        repeats = config.validation_futures
        event_arrays = {
            name: np.repeat(value[row:row + 1], repeats, axis=0)
            for name, value in arrays.items() if name != "row_index"
        }
        event_plans = np.repeat(plans[row:row + 1], repeats, axis=0)
        onset = int(reference.events.local_onset_frame[event_index]) - ANCHOR_INDEX
        follower = int(reference.events.follower_slot[event_index])
        target = observed[event_index]
        target_initial = float(reference.events.initial_conditions[event_index, 4])
        target_statistics = _event_statistics(
            target[None, None], np.asarray([[target_initial]], np.float32),
        )[0, 0]
        for name, controller in controllers.items():
            event_seed = seed + order * 1009
            rollout = _rollout(
                model, event_arrays, event_plans, controller, device, config,
                seed=event_seed, batch_size=repeats, deterministic=False,
            )
            futures = _future_features(rollout, onset, follower, frames=target.shape[0])
            arm_scores[name].append(energy_score(
                futures[:, :EVALUATION_FRAMES, :2],
                target[:EVALUATION_FRAMES, :2],
            ))
            future_statistics = _event_statistics(
                futures, np.full(repeats, target_initial, np.float32),
            )
            arm_errors[name].append(np.abs(future_statistics - target_statistics).mean(0))
    scores = {name: np.asarray(values, np.float64) for name, values in arm_scores.items()}
    errors = {name: np.asarray(values, np.float64) for name, values in arm_errors.items()}
    paired = recording_cluster_bootstrap(
        scores["a2_transfer"] - scores["calibrated_residual"], records,
    )
    diagnostic_names = ("latency", "peak_acceleration", "jerk", "gap", "closing", "recovery")
    scales = _train_scales(train_reference)
    diagnostics = {}
    for index, name in enumerate(diagnostic_names):
        comparison = recording_cluster_bootstrap(
            errors["calibrated_residual"][:, index] - errors["a2_transfer"][:, index],
            records,
        )
        comparison["allowed_degradation"] = 0.1 * scales[name]
        diagnostics[name] = comparison
    arms = {}
    for name in controllers:
        arms[name] = {
            "energy_score_mean": float(scores[name].mean()),
            "diagnostic_error_mean": {
                metric: float(errors[name][:, index].mean())
                for index, metric in enumerate(diagnostic_names)
            },
            "cell_energy_score": {
                str(int(cell)): float(scores[name][cells == cell].mean())
                for cell in np.unique(cells)
            },
        }
    arms["calibrated_residual"]["paired_energy_score"] = paired
    return {
        "events": len(selected),
        "futures_per_event": config.validation_futures,
        "arms": arms,
        "paired_diagnostics": diagnostics,
    }


def _profiles() -> dict[str, np.ndarray]:
    profiles = {}
    for magnitude in (2.0, 4.0, 6.0, 8.0):
        values = np.zeros(149, np.float32)
        values[25:50] = -magnitude
        profiles[f"constant_brake_{magnitude:g}"] = values
    ramp = np.zeros(149, np.float32)
    ramp[25:35] = np.linspace(0.0, -6.0, 10)
    ramp[35:40] = -6.0
    ramp[40:50] = np.linspace(-6.0, 0.0, 10)
    profiles["unseen_ramp"] = ramp
    pulse = np.zeros(149, np.float32)
    pulse[25:30] = -6.0
    profiles["unseen_pulse"] = pulse
    return profiles


def _physical_summary(
    rollout: ReactionRollout, logged_states: np.ndarray, profile: np.ndarray,
) -> dict[str, float | bool]:
    actions = rollout.background_actions[..., 0]
    jerk = np.abs(np.diff(actions, axis=1)) / 0.04
    active = rollout.controller_diagnostics["active"].astype(bool)
    ttc = rollout.controller_diagnostics["influence_predicted_ttc_s"]
    correction = actions - rollout.base_background_actions[..., 0]
    end = int(np.flatnonzero(profile).max()) + 1
    recovery = slice(end, min(end + 75, actions.shape[1]))
    schedule_error = (
        logged_states[:, ANCHOR_INDEX + 1:ANCHOR_INDEX + 150, 1:, 0]
        - rollout.states[:, :, 1:, 0]
    )
    chasing = active & (schedule_error > 2.0) & (correction > 0.5)
    recovery_gap = rollout.controller_diagnostics["influence_predicted_min_gap_m"][:, recovery]
    finite_gap = recovery_gap[np.isfinite(recovery_gap)]
    return {
        "valid": bool(np.isfinite(rollout.states).all() and np.isfinite(actions).all()),
        "action_bounds_valid": bool((actions >= -8.0001).all() and (actions <= 4.0001).all()),
        "jerk_limiter_failed": bool((jerk[active[:, 1:]] > 60.01).any()) if active[:, 1:].any() else False,
        "inactive_max_abs_correction_mps2": float(np.abs(correction[~active]).max()) if (~active).any() else 0.0,
        "rear_collision_rate": float(rollout.crashed[:, :, 1:].any(axis=(1, 2)).mean()),
        "emergency_guard_rate": float((active & (ttc < 2.0)).mean()),
        "recovery_final_abs_correction_mps2": float(
            np.abs(correction[:, recovery]).mean()
        ),
        "schedule_chasing_positive_correction_rate": float(chasing.mean()),
        "recovery_minimum_predicted_gap_m": float(finite_gap.min()) if len(finite_gap) else None,
    }


def evaluate_ood(
    model, arrays: dict[str, np.ndarray], plans: np.ndarray,
    controllers: dict[str, object], device: torch.device,
    config: PolicyTrainingConfig, *, seed: int, batch_size: int,
) -> dict:
    conditions = {}
    for profile_name, profile in _profiles().items():
        arms = {}
        for name, controller in controllers.items():
            rollout = _rollout(
                model, arrays, plans, controller, device, config,
                seed=seed, batch_size=batch_size, profile=profile,
            )
            arms[name] = _physical_summary(rollout, arrays["agent_states"], profile)
        conditions[profile_name] = arms
    return {"human_target_used": False, "conditions": conditions}


def _factual_noninferiority(factual: dict[str, dict[str, float]]) -> bool:
    baseline, candidate = factual["a2_transfer"], factual["calibrated_residual"]
    tolerances = {"ade_m": 0.02, "fde_m": 0.06, "p95_m": 0.10}
    return all(
        candidate[name] - baseline[name] <= absolute
        and candidate[name] <= baseline[name] * 1.05
        for name, absolute in tolerances.items()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--a1-checkpoint", type=Path)
    parser.add_argument("--a2-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    config = load_protocol_config(args.config.resolve())
    base = load_protocol_config(ROOT / config["base_config"])
    device = select_device(config["training"].get("device", "auto"))
    model, _ = load_checkpoint(base["paths"]["evaluation_checkpoint"], device=device)
    experiment = prepare_experiment_data(base, ROOT)
    source_rows = getattr(experiment, f"{args.split}_rows")
    events_root = args.events_dir or Path(config["paths"]["event_reference"])
    split_events = ReactionEventReference.load(events_root / args.split)
    if args.limit is not None:
        event_rows = split_events.events.row_index[
            split_events.events.indices(split_events.supported_cells)
        ][:args.limit]
        source_rows = np.unique(np.concatenate((source_rows[:args.limit], event_rows)))
    arrays = _arrays(experiment.bundle, source_rows)
    output_root = ensure_dir(args.output.parent)
    plans = frozen_diffusion_plans(
        experiment.bundle, source_rows,
        checkpoint=base["paths"]["diffusion_checkpoint"],
        output_dir=output_root / "cache" / args.split,
        device=device,
        batch_size=int(base["training"]["validation_batch_size"]),
        ddim_steps=int(config["training"].get("diffusion_ddim_steps", 20)),
        experiment_scope=base["training"].get("experiment_scope", "full"),
    )
    plans = complete_missing_background_plans(
        plans, arrays["agent_states"], arrays["agent_valid"],
    )
    fields = {
        name: value for name, value in config["training"].items()
        if name in PolicyTrainingConfig.__dataclass_fields__
    }
    training = PolicyTrainingConfig(**fields)
    rule = RuleModelBundle.load(ROOT / config["paths"]["rule_model"])
    controllers = {
        name: _load_controller(name, path, rule, device)
        for name, path in (
            ("frozen_hiqr", None),
            ("idm_only", None),
            ("a2_transfer", args.a2_checkpoint),
            ("calibrated_residual", args.candidate_checkpoint),
        )
    }
    if args.a1_checkpoint is not None:
        controllers["a1_transfer"] = _load_controller(
            "a1_transfer", args.a1_checkpoint, rule, device,
        )

    factual = {}
    for name, controller in controllers.items():
        rollout = _rollout(
            model, arrays, plans, controller, device, training,
            seed=training.seed, batch_size=args.batch_size,
        )
        factual[name] = factual_metrics(
            rollout, arrays["agent_states"], arrays["agent_valid"],
        )
    factual["calibrated_residual"]["noninferior"] = _factual_noninferiority(factual)

    event_report = evaluate_events(
        model, arrays, plans, split_events,
        controllers, device, training, seed=training.seed,
        train_reference=ReactionEventReference.load(events_root / "train"), limit=args.limit,
    )
    ood_rows = np.arange(min(len(source_rows), args.limit or 64))
    physical_ood = evaluate_ood(
        model,
        {name: value[ood_rows] for name, value in arrays.items() if name != "row_index"},
        plans[ood_rows], controllers, device, training,
        seed=training.seed + 77, batch_size=args.batch_size,
    )
    candidate_conditions = [
        arms["calibrated_residual"] for arms in physical_ood["conditions"].values()
    ]
    physical_ood["calibrated_residual"] = {
        "valid": all(
            item["valid"] and item["action_bounds_valid"]
            for item in candidate_conditions
        ),
        "jerk_limiter_failed": any(
            item["jerk_limiter_failed"] for item in candidate_conditions
        ),
    }
    report = {
        "schema_name": "reaction_policy_evaluation",
        "schema_version": 1,
        "split": args.split,
        "protocol": "fixed_k_gt_conditional_resimulation_no_longitudinal_rebase",
        "autonomous_response_scope": True,
        "factual": factual,
        "held_out_events": event_report,
        "physical_ood": physical_ood,
    }
    save_json(report, args.output)
    print(json.dumps({"output": str(args.output), "events": event_report["events"]}))


if __name__ == "__main__":
    main()

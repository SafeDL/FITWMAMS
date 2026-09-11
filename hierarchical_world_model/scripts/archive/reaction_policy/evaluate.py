#!/usr/bin/env python3
"""Evaluate reaction policies under one autonomous runtime and reference protocol."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.influence_graph import dynamic_candidate_scene_mask  # noqa: E402
from hierarchical_world_model.src.planner import complete_missing_background_plans, frozen_diffusion_plans  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.reaction_controller import (  # noqa: E402
    CalibratedResidualReactionController, IDMOnlyReactionController,
    IDMResidualReactionController, NoReactionController, ReactionController,
    RLResidualReactionController,
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


DEFAULT = ROOT / "hierarchical_world_model/config/archive/reaction_policy.yaml"
OOD_HORIZON_FRAMES = 149


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
        "calibrated_supervised": (
            "calibrated_residual", lambda: CalibratedResidualReactionController(rule),
        ),
        "calibrated_initial": (
            "calibrated_residual", lambda: CalibratedResidualReactionController(rule),
        ),
    }
    expected_mode, constructor = expected[name]
    if payload.get("controller_mode") != expected_mode:
        raise ValueError(f"{path} is not a {name} checkpoint")
    controller = constructor().to(device)
    controller.load_state_dict(payload["state_dict"], strict=True)
    return controller.eval()


class DesiredActionController(ReactionController):
    """Evaluation-only calibrated-policy ablation that bypasses its execution guard."""

    mode = "calibrated_residual"

    def __init__(self, controller: CalibratedResidualReactionController) -> None:
        super().__init__()
        self.controller = controller

    def forward(self, context, *, deterministic: bool = False):
        output = self.controller(context, deterministic=deterministic)
        actions = context.base_actions.clone()
        executed = torch.where(
            output.active,
            output.desired_action_ax,
            context.base_actions[:, 0, :, 0],
        )
        actions[:, 0, :, 0] = executed.clamp(
            context.cfg.min_acceleration_mps2,
            context.cfg.max_acceleration_mps2,
        )
        return replace(
            output,
            actions=actions,
            delta_ax=actions[:, 0, :, 0] - context.base_actions[:, 0, :, 0],
        )


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
        collision_pairs=np.concatenate([part.collision_pairs for part in parts]),
        offroad=np.concatenate([part.offroad for part in parts]),
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


def event_selection(
    reference: ReactionEventReference,
    available_rows: np.ndarray,
    limit: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Select replayable events and expose every filtering step."""
    events = reference.events
    all_indices = events.indices()
    supported = events.indices(reference.supported_cells)
    ego_leader = all_indices[events.leader_slot == 0]
    supported_ego = supported[events.leader_slot[supported] == 0]
    available = set(np.asarray(available_rows, np.int64).tolist())
    mapped = np.asarray([
        index for index in supported_ego
        if int(events.row_index[index]) in available
    ], np.int64)
    selected = mapped if limit is None else _recording_balanced_events(events, mapped, limit)
    records, counts = np.unique(events.recording_id[selected], return_counts=True)
    audit = {
        "all_events": int(len(all_indices)),
        "supported_events": int(len(supported)),
        "ego_leader_events": int(len(ego_leader)),
        "supported_ego_leader_events": int(len(supported_ego)),
        "mapped_scene_events": int(len(mapped)),
        "evaluated_events": int(len(selected)),
        "unique_scene_rows": int(len(np.unique(events.row_index[selected]))),
        "recordings": int(len(records)),
        "events_per_recording": {
            str(int(record)): int(count) for record, count in zip(records, counts)
        },
    }
    return selected, audit


def _recording_balanced_events(events, indices: np.ndarray, limit: int) -> np.ndarray:
    """Round-robin recordings; within each recording alternate weak/strong braking."""
    pools = {}
    for record in np.unique(events.recording_id[indices]):
        pool = indices[events.recording_id[indices] == record]
        order = np.argsort(events.initial_conditions[pool, 0], kind="stable")
        ordered = pool[order]
        zigzag = np.ravel(np.column_stack((ordered, ordered[::-1])))
        pools[int(record)] = list(dict.fromkeys(int(index) for index in zigzag))
    selected = []
    while len(selected) < min(limit, len(indices)):
        for record in sorted(pools):
            if pools[record]:
                selected.append(pools[record].pop(0))
                if len(selected) == min(limit, len(indices)):
                    break
    return np.asarray(selected, np.int64)


def mechanism_events(reference: ReactionEventReference, mapped: np.ndarray) -> np.ndarray:
    """Choose the weakest and strongest leader-brake event per recording."""
    selected = []
    events = reference.events
    for record in np.unique(events.recording_id[mapped]):
        pool = mapped[events.recording_id[mapped] == record]
        braking = events.initial_conditions[pool, 0]
        selected.extend((int(pool[np.argmin(braking)]), int(pool[np.argmax(braking)])))
    return np.asarray(list(dict.fromkeys(selected)), np.int64)


def evaluate_events(
    model, arrays: dict[str, np.ndarray], plans: np.ndarray,
    reference: ReactionEventReference, controllers: dict[str, object],
    device: torch.device, config: PolicyTrainingConfig, *, seed: int,
    train_reference: ReactionEventReference, selected: np.ndarray,
    energy_pairs: tuple[tuple[str, str], ...],
    diagnostic_baseline: str,
) -> dict:
    lookup = {int(row): index for index, row in enumerate(arrays["row_index"])}
    if not len(selected):
        raise RuntimeError("held-out split has no replayable supported events")
    observed = event_window(reference.events, recovery=True)
    arm_scores = {name: [] for name in controllers}
    arm_errors = {name: [] for name in controllers}
    arm_collision = {name: [] for name in controllers}
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
            # Event labels identify the actual leader/follower relation; do
            # not call unrelated NPC crashes a rear collision.
            arm_collision[name].append(float(rollout.crashed[:, :, follower].any(axis=1).mean()))
    scores = {name: np.asarray(values, np.float64) for name, values in arm_scores.items()}
    errors = {name: np.asarray(values, np.float64) for name, values in arm_errors.items()}
    collisions = {name: np.asarray(values, np.float64) for name, values in arm_collision.items()}
    paired_energy = {
        f"{left}_minus_{right}": {
            **recording_cluster_bootstrap(scores[left] - scores[right], records),
            "comparison": f"{left} minus {right}; positive favors {right}",
        }
        for left, right in energy_pairs
    }
    diagnostic_names = ("latency", "peak_acceleration", "jerk", "gap", "closing", "recovery")
    scales = _train_scales(train_reference)
    diagnostics = {}
    for index, name in enumerate(diagnostic_names):
        comparison = recording_cluster_bootstrap(
            errors["calibrated_residual"][:, index] - errors[diagnostic_baseline][:, index],
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
    collision_comparisons = {}
    for baseline in ("a2_transfer", "idm_only"):
        if baseline not in collisions:
            continue
        comparison = recording_cluster_bootstrap(
            collisions["calibrated_residual"] - collisions[baseline], records,
        )
        comparison.update({
            "candidate_rate": float(collisions["calibrated_residual"].mean()),
            "baseline_rate": float(collisions[baseline].mean()),
            "comparison": f"calibrated_residual minus {baseline}; positive is worse",
        })
        collision_comparisons[baseline] = comparison
    return {
        "events": len(selected),
        "recordings": int(len(np.unique(records))),
        "futures_per_event": config.validation_futures,
        "arms": arms,
        "paired_energy_score": paired_energy,
        "paired_diagnostics": diagnostics,
        "paired_rear_collision": collision_comparisons,
    }


def _profiles() -> dict[str, np.ndarray]:
    profiles = {}
    for magnitude in (2.0, 4.0, 6.0, 8.0):
        values = np.zeros(OOD_HORIZON_FRAMES, np.float32)
        values[25:50] = -magnitude
        profiles[f"constant_brake_{magnitude:g}"] = values
    ramp = np.zeros(OOD_HORIZON_FRAMES, np.float32)
    ramp[25:35] = np.linspace(0.0, -6.0, 10)
    ramp[35:40] = -6.0
    ramp[40:50] = np.linspace(-6.0, 0.0, 10)
    profiles["unseen_ramp"] = ramp
    pulse = np.zeros(OOD_HORIZON_FRAMES, np.float32)
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
    desired = rollout.controller_diagnostics["desired_action_ax"]
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
        "npc_involved_collision_rate": float(rollout.crashed[:, :, 1:].any(axis=(1, 2)).mean()),
        "guard_activation_condition_rate": float((active & (ttc < 2.0)).mean()),
        "guard_action_rewrite_rate": float((active & (np.abs(actions - desired) > 1.0e-5)).mean()),
        "guard_mean_rewrite_mps2": float(np.abs(actions - desired)[active].mean()) if active.any() else 0.0,
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
    telemetry_output: Path,
) -> dict:
    conditions = {}
    failures = []
    telemetry = []
    desired_controller = DesiredActionController(
        controllers["calibrated_residual"]
    ).to(device).eval()
    for profile_name, profile in _profiles().items():
        arms = {}
        rollouts = {}
        for name, controller in controllers.items():
            rollout = _rollout(
                model, arrays, plans, controller, device, config,
                seed=seed, batch_size=batch_size, profile=profile,
            )
            rollouts[name] = rollout
            arms[name] = _physical_summary(rollout, arrays["agent_states"], profile)
        desired_rollout = _rollout(
            model, arrays, plans, desired_controller, device, config,
            seed=seed, batch_size=batch_size, profile=profile,
        )
        failures.extend(_paired_failures(
            profile_name, arrays["row_index"], rollouts, desired_rollout, telemetry,
        ))
        conditions[profile_name] = arms
    _save_failure_telemetry(telemetry_output, telemetry)
    strict = sum(
        failure["causal"] and failure["a2_avoids"] and failure["idm_avoids"]
        for failure in failures
    )
    return {
        "human_target_used": False,
        "scene_rows": int(len(arrays["agent_states"])),
        "conditions": conditions,
        "failure_analysis": {
            "paired_failures": failures,
            "paired_failure_count": len(failures),
            "strict_causal_regressions": int(strict),
            "telemetry": str(telemetry_output),
        },
    }


def _first_crash(rollout: ReactionRollout, scene: int) -> tuple[int, list[int]] | None:
    timeline = rollout.crashed[scene].any(axis=-1)
    if not timeline.any():
        return None
    frame = int(np.flatnonzero(timeline)[0])
    return frame, np.flatnonzero(rollout.crashed[scene, frame]).astype(int).tolist()


def _causal_crash_slots(rollout: ReactionRollout, scene: int, frame: int, slots: list[int]) -> list[int]:
    roles = rollout.controller_diagnostics["influence_role"][scene, :frame + 1]
    return [
        slot for slot in slots
        if slot > 0 and np.isin(roles[:, slot - 1], (1, 4)).any()
    ]


def _paired_failures(
    profile: str,
    row_indices: np.ndarray,
    rollouts: dict[str, ReactionRollout],
    desired_rollout: ReactionRollout,
    telemetry: list[dict],
) -> list[dict]:
    candidate = rollouts["calibrated_residual"]
    failures = []
    for scene in range(len(row_indices)):
        crash = _first_crash(candidate, scene)
        if crash is None:
            continue
        a2_avoids = _first_crash(rollouts["a2_transfer"], scene) is None
        idm_avoids = _first_crash(rollouts["idm_only"], scene) is None
        if not (a2_avoids or idm_avoids):
            continue
        frame, slots = crash
        causal_slots = _causal_crash_slots(candidate, scene, frame, slots)
        direct_desired_avoids = _first_crash(desired_rollout, scene) is None
        candidate_active = any(
            candidate.controller_diagnostics["active"][scene, :frame + 1, slot - 1].any()
            for slot in causal_slots
        )
        a2_active = any(
            rollouts["a2_transfer"].controller_diagnostics["active"][scene, :frame + 1, slot - 1].any()
            for slot in causal_slots
        )
        if not causal_slots:
            cause = "unrelated_npc"
        elif direct_desired_avoids:
            cause = "execution_jerk_limited"
        elif not candidate_active and a2_active:
            cause = "authority_role_limited"
        else:
            cause = "actor_insufficient"
        item = {
            "profile": profile,
            "row_index": int(row_indices[scene]),
            "first_crash_frame": frame,
            "crashed_slots": slots,
            "causal_slots": causal_slots,
            "causal": bool(causal_slots),
            "a2_avoids": a2_avoids,
            "idm_avoids": idm_avoids,
            "direct_desired_avoids": direct_desired_avoids,
            "cause": cause,
        }
        failures.append(item)
        for slot in causal_slots or [slot for slot in slots if slot > 0]:
            for arm, rollout in rollouts.items():
                _append_telemetry(
                    telemetry, profile, int(row_indices[scene]), arm,
                    rollout, scene, slot, frame,
                )
            _append_telemetry(
                telemetry, profile, int(row_indices[scene]), "calibrated_desired",
                desired_rollout, scene, slot, frame,
            )
    return failures


def _append_telemetry(
    output: list[dict], profile: str, row_index: int, arm: str,
    rollout: ReactionRollout, scene: int, slot: int, crash_frame: int,
) -> None:
    diagnostics = rollout.controller_diagnostics
    for frame in range(len(rollout.states[scene])):
        parent = int(diagnostics["influence_parent"][scene, frame, slot - 1])
        parent = parent if parent >= 0 else 0
        state = rollout.states[scene, frame]
        gap = float(state[parent, 0] - state[slot, 0] - 4.8)
        closing = float(state[slot, 2] - state[parent, 2])
        output.append({
            "profile": profile, "row_index": row_index, "arm": arm,
            "frame": frame, "slot": slot, "crash_frame": crash_frame,
            "base": float(rollout.base_background_actions[scene, frame, slot - 1, 0]),
            "rule": float(diagnostics["rule_action_ax"][scene, frame, slot - 1]),
            "desired": float(diagnostics["desired_action_ax"][scene, frame, slot - 1]),
            "executed": float(rollout.background_actions[scene, frame, slot - 1, 0]),
            "alpha": float(diagnostics["alpha"][scene, frame, slot - 1]),
            "authority": float(diagnostics["influence_authority"][scene, frame, slot - 1]),
            "active": bool(diagnostics["active"][scene, frame, slot - 1]),
            "role": int(diagnostics["influence_role"][scene, frame, slot - 1]),
            "gap": gap, "closing": closing,
            "ttc": float(diagnostics["influence_predicted_ttc_s"][scene, frame, slot - 1]),
        })


def _save_failure_telemetry(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        np.savez_compressed(path, row_index=np.empty(0, np.int64))
        return
    columns = {
        name: np.asarray([row[name] for row in rows])
        for name in rows[0]
    }
    np.savez_compressed(path, **columns)


def _factual_noninferiority(factual: dict[str, dict[str, float]]) -> bool:
    baseline, candidate = factual["frozen_hiqr"], factual["calibrated_residual"]
    tolerances = {"ade_m": 0.02, "fde_m": 0.06, "p95_m": 0.10}
    return all(
        candidate[name] - baseline[name] <= absolute
        and candidate[name] <= baseline[name] * 1.05
        for name, absolute in tolerances.items()
    )


def _policy_training_config(training: dict) -> PolicyTrainingConfig:
    fields = {
        name: value
        for name, value in training.items()
        if name in PolicyTrainingConfig.__dataclass_fields__
    }
    return PolicyTrainingConfig(**fields)


def _write_comparison(
    output: Path, factual: dict[str, dict[str, float]], event_report: dict,
) -> None:
    with output.with_name("comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "controller", "factual_ade_m", "factual_fde_m", "factual_p95_m",
            "event_energy_score", "event_count", "futures_per_event",
        ))
        writer.writeheader()
        for name, arm in event_report["arms"].items():
            writer.writerow({
                "controller": name,
                "factual_ade_m": factual[name]["ade_m"],
                "factual_fde_m": factual[name]["fde_m"],
                "factual_p95_m": factual[name]["p95_m"],
                "event_energy_score": arm["energy_score_mean"],
                "event_count": event_report["events"],
                "futures_per_event": event_report["futures_per_event"],
            })


def _ood_event_indices(
    reference: ReactionEventReference, mapped: np.ndarray, limit: int | None,
) -> np.ndarray:
    ordered = _recording_balanced_events(reference.events, mapped, len(mapped))
    selected = []
    rows = set()
    for event_index in ordered:
        row = int(reference.events.row_index[event_index])
        if row in rows:
            continue
        rows.add(row)
        selected.append(int(event_index))
        if limit is not None and len(selected) >= limit:
            break
    return np.asarray(selected, np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--a1-checkpoint", type=Path)
    parser.add_argument("--a2-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--supervised-checkpoint", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--events-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plans-cache-dir", type=Path)
    parser.add_argument("--event-limit", type=int)
    parser.add_argument("--factual-limit", type=int)
    parser.add_argument("--ood-scene-limit", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    config = load_protocol_config(args.config.resolve())
    base = load_protocol_config(ROOT / config["base_config"])
    device = select_device(config["training"].get("device", "auto"))
    model, _ = load_checkpoint(base["paths"]["evaluation_checkpoint"], device=device)
    experiment = prepare_experiment_data(base, ROOT)
    split_rows = getattr(experiment, f"{args.split}_rows")
    eligible = dynamic_candidate_scene_mask(
        np.asarray(experiment.bundle.arrays["agent_states"]),
        np.asarray(experiment.bundle.arrays["agent_valid"]),
        rows=split_rows,
        radius_m=float(config["training"]["influence_radius_m"]),
        prediction_horizon_s=float(
            config["training"]["influence_prediction_horizon_s"]
        ),
    )
    source_rows = np.asarray(split_rows, np.int64)[eligible]
    events_root = args.events_dir or Path(config["paths"]["event_reference"])
    split_events = ReactionEventReference.load(events_root / args.split)
    supported = split_events.events.indices(split_events.supported_cells)
    supported_ego = supported[
        split_events.events.leader_slot[supported] == 0
    ]
    source_rows = np.unique(np.concatenate((
        source_rows,
        split_events.events.row_index[supported_ego],
    ))).astype(np.int64)
    arrays = _arrays(experiment.bundle, source_rows)
    ensure_dir(args.output.parent)
    cache_dir = args.plans_cache_dir or args.output.parent / "cache" / args.split
    plans = frozen_diffusion_plans(
        experiment.bundle, source_rows,
        checkpoint=base["paths"]["diffusion_checkpoint"],
        output_dir=cache_dir,
        device=device,
        batch_size=int(base["training"]["validation_batch_size"]),
        ddim_steps=int(config["training"].get("diffusion_ddim_steps", 20)),
        experiment_scope=base["training"].get("experiment_scope", "full"),
    )
    plans = complete_missing_background_plans(
        plans, arrays["agent_states"], arrays["agent_valid"],
    )
    training = _policy_training_config(config["training"])
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
    if args.supervised_checkpoint is not None:
        controllers["calibrated_supervised"] = _load_controller(
            "calibrated_supervised", args.supervised_checkpoint, rule, device,
        )
    if args.a1_checkpoint is not None:
        controllers["a1_transfer"] = _load_controller(
            "a1_transfer", args.a1_checkpoint, rule, device,
        )

    factual_count = min(len(source_rows), args.factual_limit or len(source_rows))
    factual_arrays = {
        name: value[:factual_count]
        for name, value in arrays.items() if name != "row_index"
    }
    factual = {}
    for name, controller in controllers.items():
        rollout = _rollout(
            model, factual_arrays, plans[:factual_count], controller, device, training,
            seed=training.seed, batch_size=args.batch_size,
        )
        factual[name] = factual_metrics(
            rollout, factual_arrays["agent_states"], factual_arrays["agent_valid"],
        )
    factual["calibrated_residual"]["noninferior"] = _factual_noninferiority(factual)

    selected, selection_audit = event_selection(
        split_events, arrays["row_index"], args.event_limit,
    )
    energy_pairs = [
        ("a2_transfer", "calibrated_residual"),
        ("idm_only", "calibrated_residual"),
    ]
    if "calibrated_supervised" in controllers:
        energy_pairs.append(("calibrated_supervised", "calibrated_residual"))
    event_report = evaluate_events(
        model, arrays, plans, split_events,
        controllers, device, training, seed=training.seed,
        train_reference=ReactionEventReference.load(events_root / "train"),
        selected=selected, energy_pairs=tuple(energy_pairs),
        diagnostic_baseline="a2_transfer",
    )
    mapped, _ = event_selection(split_events, arrays["row_index"])
    ood_events = _ood_event_indices(split_events, mapped, args.ood_scene_limit)
    row_lookup = {int(row): index for index, row in enumerate(arrays["row_index"])}
    ood_rows = np.asarray([
        row_lookup[int(split_events.events.row_index[index])]
        for index in ood_events
    ], np.int64)
    physical_ood = evaluate_ood(
        model,
        {name: value[ood_rows] for name, value in arrays.items()},
        plans[ood_rows], controllers, device, training,
        seed=training.seed + 77, batch_size=args.batch_size,
        telemetry_output=args.output.with_name("failure_telemetry.npz"),
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
        "schema_version": 2,
        "split": args.split,
        "protocol": "fixed_k_gt_conditional_resimulation_no_longitudinal_rebase",
        "autonomous_response_scope": True,
        "evaluation_config": {
            "config": str(args.config.resolve()),
            "events": str(events_root.resolve()),
            "plans_cache": str(Path(cache_dir).resolve()),
            "a2_checkpoint": str(args.a2_checkpoint.resolve()),
            "supervised_checkpoint": (
                str(args.supervised_checkpoint.resolve())
                if args.supervised_checkpoint else None
            ),
            "candidate_checkpoint": str(args.candidate_checkpoint.resolve()),
            "initial_checkpoint": (
                str(args.initial_checkpoint.resolve()) if args.initial_checkpoint else None
            ),
            "event_limit": args.event_limit,
            "factual_rows": factual_count,
            "ood_scene_limit": args.ood_scene_limit,
            "seed": training.seed,
        },
        "selection_audit": selection_audit,
        "factual": factual,
        "held_out_events": event_report,
        "physical_ood": physical_ood,
    }
    save_json(
        physical_ood["failure_analysis"],
        args.output.with_name("failure_cases.json"),
    )
    if args.initial_checkpoint is not None and args.supervised_checkpoint is not None:
        mechanism_controllers = {
            "calibrated_initial": _load_controller(
                "calibrated_initial", args.initial_checkpoint, rule, device,
            ),
            "calibrated_supervised": controllers["calibrated_supervised"],
            "calibrated_residual": controllers["calibrated_residual"],
        }
        mechanism_selected = mechanism_events(split_events, mapped)
        report["mechanism_diagnostic"] = evaluate_events(
            model, arrays, plans, split_events, mechanism_controllers,
            device, training, seed=training.seed,
            train_reference=ReactionEventReference.load(events_root / "train"),
            selected=mechanism_selected,
            energy_pairs=(
                ("calibrated_initial", "calibrated_supervised"),
                ("calibrated_supervised", "calibrated_residual"),
            ),
            diagnostic_baseline="calibrated_initial",
        )
    save_json(report, args.output)
    _write_comparison(args.output, factual, event_report)
    print(json.dumps({"output": str(args.output), "events": event_report["events"]}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate CIH-WM for factual reconstruction and ADS causal response.

This runner deliberately reuses :func:`hierarchical_world_model.src.evaluation.rollout`
and the frozen diffusion-plan construction used by ``scripts/evaluate.py``.
It reports one full-background factual scope, including the directly affected
follower used by ADS causal tests.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import sys
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.cih_model import (  # noqa: E402
    CausalInfluenceHierarchicalWorldModel,
    build_response_policy,
)
from hierarchical_world_model.src.evaluation import _factual_metrics, rollout  # noqa: E402
from hierarchical_world_model.src.human_response_prior import HumanResponseQuerySet  # noqa: E402
from hierarchical_world_model.src.human_response_training import event_energy_score, legacy_event_energy_score  # noqa: E402
from hierarchical_world_model.src.planner import (  # noqa: E402
    complete_endogenous_response_plans,
    frozen_diffusion_plans,
)
from hierarchical_world_model.src.protocol import (  # noqa: E402
    ACCEPTANCE_GATES,
    load_protocol_config,
)
from hierarchical_world_model.src.reaction_controller import (  # noqa: E402
    CausalInfluenceResponsePolicy,
    NoReactionController,
)
from hierarchical_world_model.src.reaction_evidence import (  # noqa: E402
    ReactionEventReference, event_identity, recording_cluster_bootstrap,
)
from hierarchical_world_model.src.randomness import WorldExogenousState  # noqa: E402
from world_model.src.core.evaluation_scope import scoped_canonical_trajectory  # noqa: E402
from world_model.src.core.utils import file_sha256, select_device, set_seed  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _training_provenance(candidate: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Accept selection only from the CIH-WM trainer's immutable manifest."""
    manifest_path = candidate.parent / "manifest.json"
    if not manifest_path.is_file():
        return {"passed": False, "reason": "formal training manifest is absent"}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"passed": False, "reason": "formal training manifest is not valid JSON"}
    protocol = manifest.get("protocol", {})
    schedule = manifest.get("training_schedule", {})
    actual = manifest.get("actual_optimizer_steps", {})
    passed = (
        manifest.get("schema") == "cih_world_model_training"
        and protocol.get("world_executor") == "hierarchical_world_model.src.evaluation.rollout"
        and protocol.get("highwayenv") == "not imported or executed"
        and protocol.get("selection") == "none during training; full formal validation is mandatory"
        and manifest.get("event_limit") is None
        and int(actual.get("supervised", -1)) > 0
        and int(actual.get("policy_updates_attempted", -1)) > 0
        and int(schedule.get("event_groups_per_policy_update", -1))
        >= int(config["policy_optimization"]["event_groups_per_update"])
        and int(schedule.get("futures_per_event", -1))
        >= int(config["policy_optimization"]["futures_per_event"])
        and bool(manifest.get("termination_reason"))
    )
    return {"passed": bool(passed), "manifest": str(manifest_path), "reason": None if passed else "manifest does not attest CIH-WM training"}


def _concat(items: list[Any], name: str) -> np.ndarray:
    return np.concatenate([getattr(item, name) for item in items], axis=0)


def _rollout_chunks(
    model: torch.nn.Module,
    states: np.ndarray,
    valid: np.ndarray,
    plans: np.ndarray,
    maps: np.ndarray,
    map_valid: np.ndarray,
    *,
    controller: Any,
    device: torch.device,
    intervention: str | None = None,
    dose: float = 0.0,
    chunk: int = 256,
    excluded_slots: tuple[str, ...] | None = None,
    nominal_reference: list[Any] | None = None,
    influence_graph_config: dict[str, float | int] | None = None,
) -> list[Any]:
    if nominal_reference is not None and len(nominal_reference) != (len(states) + chunk - 1) // chunk:
        raise ValueError("nominal rollout chunks must align with the current batch")
    return [
        rollout(
            model, states[start:start + chunk], valid[start:start + chunk],
            plans[start:start + chunk], maps[start:start + chunk], map_valid[start:start + chunk],
            device=device, history_frames=25, motion_seed=None,
            intervention=intervention, dose=dose, controller=controller,
            controller_deterministic=True,
            excluded_slots=excluded_slots,
            nominal_reference=(None if nominal_reference is None else nominal_reference[index]),
            influence_graph_config=influence_graph_config,
            record_policy_trace=False,
        )
        for index, start in enumerate(range(0, len(states), chunk))
    ]


def _controller_diagnostics(items: list[Any], key: str) -> np.ndarray:
    values = [item.controller_diagnostics for item in items]
    if any(value is None or key not in value for value in values):
        raise RuntimeError(f"formal controller rollout did not record {key}")
    return np.concatenate([value[key] for value in values if value is not None], axis=0)


def _causal_probe(
    model: torch.nn.Module,
    states: np.ndarray,
    valid: np.ndarray,
    plans: np.ndarray,
    maps: np.ndarray,
    map_valid: np.ndarray,
    *,
    controller: CausalInfluenceResponsePolicy,
    device: torch.device,
    rows: int,
    influence_graph_config: dict[str, float | int],
) -> dict[str, Any]:
    """Paired ADS intervention check without a nominal-action override."""
    count = min(int(rows), len(states))
    nominal = _rollout_chunks(
        model, states[:count], valid[:count], plans[:count], maps[:count], map_valid[:count],
        controller=controller, device=device, excluded_slots=(),
        influence_graph_config=influence_graph_config,
    )
    result: dict[str, Any] = {
        "rows": count,
        "profiles": {},
        "nominal_reference_is_model_input": False,
    }
    total_selected = total_executed = total_executed_agree = 0
    level_totals = {"direct": {"selected": 0, "executed": 0, "agreed": 0}}
    preprobe_max = 0.0
    jerk_limit = float(controller.correction_jerk_limit_mps3) * float(model.cfg.dt_s) + 1.0e-5
    jerk_ok = True
    maximum_controller_step = 0.0
    dose_effects: list[float] = []
    infeasible_count = 0
    window_effects: list[np.ndarray] = []
    window_rules: list[np.ndarray] = []
    preprobe = _concat(nominal, "states")[:, 24]
    ego, npc = preprobe[:, :1], preprobe[:, 1:]
    dx = npc[..., 0] - ego[..., 0]
    dy = np.abs(npc[..., 1] - ego[..., 1])
    distance = np.linalg.norm(npc[..., :2] - ego[..., :2], axis=-1)
    geometric = (
        valid[:count, ANCHOR_INDEX, 1:] & (dx < 0.0) & (dy < 1.8)
        & (distance <= 50.0) & (-dx - 4.8 > 0.0)
    )
    # Freeze one nearest direct follower per scene before the probe.  This
    # denominator is independent of whether the policy later responds.
    nearest = np.where(geometric, dx, -np.inf).argmax(axis=1)
    fixed_eligibility = np.zeros_like(geometric)
    rows_with_follower = geometric.any(axis=1)
    fixed_eligibility[np.flatnonzero(rows_with_follower), nearest[rows_with_follower]] = True
    for dose in (1.5, 2.25, 3.0):
        treated = _rollout_chunks(
            model, states[:count], valid[:count], plans[:count], maps[:count], map_valid[:count],
            controller=controller, device=device, intervention="brake", dose=dose,
            excluded_slots=(), influence_graph_config=influence_graph_config,
        )
        nominal_actions = _concat(nominal, "background_actions")[..., 0]
        treated_actions = _concat(treated, "background_actions")[..., 0]
        nominal_rule = _controller_diagnostics(nominal, "rule_action_ax")
        treated_rule = _controller_diagnostics(treated, "rule_action_ax")
        action_delta = treated_actions - nominal_actions
        rule_delta = treated_rule - nominal_rule
        active = _controller_diagnostics(treated, "active").astype(bool)
        direct = _controller_diagnostics(treated, "influence_direct").astype(bool)
        selected = (np.abs(rule_delta) >= 0.10) & active & direct
        executed = selected & (np.abs(action_delta) >= 1.0e-4)
        executed_agreement = np.sign(action_delta[executed]) == np.sign(rule_delta[executed])
        calibration = _controller_diagnostics(treated, "calibration_correction_ax")
        infeasible_count += int(
            _controller_diagnostics(treated, "correction_constraint_infeasible").sum()
        )
        if calibration.shape[1] > 1:
            maximum_controller_step = max(
                maximum_controller_step, float(np.abs(np.diff(calibration, axis=1)).max(initial=0.0))
            )
            jerk_ok = jerk_ok and bool(np.all(np.abs(np.diff(calibration, axis=1)) <= jerk_limit))
        prefix = np.abs(action_delta[:, :25]).max(initial=0.0)
        preprobe_max = max(preprobe_max, float(prefix))
        chosen = int(selected.sum())
        executed_count = int(executed.sum())
        total_selected += chosen
        total_executed += executed_count
        total_executed_agree += int(executed_agreement.sum())
        aligned = np.sign(rule_delta[executed]) * action_delta[executed]
        dose_effect = float(aligned.mean()) if executed_count else 0.0
        dose_effects.append(dose_effect)
        window_effects.append(action_delta[:, 25:75].sum(axis=1) * .04)
        window_rules.append(rule_delta[:, 25:75].sum(axis=1) * .04)
        level_report: dict[str, Any] = {}
        for name, mask in (("direct", direct),):
            level_selected = selected & mask
            level_executed = executed & mask
            level_agreement = (
                np.sign(action_delta[level_executed])
                == np.sign(rule_delta[level_executed])
            )
            selected_count = int(level_selected.sum())
            level_executed_count = int(level_executed.sum())
            agreed_count = int(level_agreement.sum())
            level_totals[name]["selected"] += selected_count
            level_totals[name]["executed"] += level_executed_count
            level_totals[name]["agreed"] += agreed_count
            level_report[name] = {
                "selected_slot_frames": selected_count,
                "executed_action_slot_frames": level_executed_count,
                "executed_action_direction_agreement": (
                    None
                    if not level_executed_count
                    else float(level_agreement.mean())
                ),
            }
        result["profiles"][f"brake_{dose:g}"] = {
            "selected_slot_frames": chosen,
            "executed_action_slot_frames": executed_count,
            "executed_action_direction_agreement": None if not executed_count else float(executed_agreement.mean()),
            "bound_saturated_or_unchanged_slot_frames": chosen - executed_count,
            "preprobe_max_abs_action_difference_mps2": float(prefix),
            "mean_rule_aligned_action_effect_mps2": dose_effect,
            "influence_levels": level_report,
        }
    influence_levels = {}
    for name, values in level_totals.items():
        influence_levels[name] = {
            "selected_slot_frames": values["selected"],
            "executed_action_slot_frames": values["executed"],
            "executed_action_direction_agreement": (
                None
                if not values["executed"]
                else values["agreed"] / values["executed"]
            ),
        }
    window_effect = np.stack(window_effects)
    window_rule = np.stack(window_rules)
    comparable = fixed_eligibility[None] & (np.abs(window_rule) >= .05)
    response = np.abs(window_effect) >= .02
    window_success = comparable & response & (np.sign(window_effect) == np.sign(window_rule))
    all_doses_comparable = comparable.all(axis=0)
    aligned_window = np.sign(window_rule) * window_effect
    monotonic_events = all_doses_comparable & (
        (aligned_window[1] + 1.e-6 >= aligned_window[0])
        & (aligned_window[2] + 1.e-6 >= aligned_window[1])
    )
    result.update({
        "selected_slot_frames": total_selected,
        "executed_action_slot_frames": total_executed,
        "executed_action_direction_agreement": None if not total_executed else float(total_executed_agree / total_executed),
        "bound_saturated_or_unchanged_slot_frames": total_selected - total_executed,
        "preprobe_max_abs_action_difference_mps2": preprobe_max,
        "maximum_controller_correction_step_mps2": maximum_controller_step,
        "controller_correction_step_limit_mps2": jerk_limit,
        "controller_induced_jerk_within_limit": bool(jerk_ok),
        "correction_constraint_infeasible_slot_frames": infeasible_count,
        "dose_effects_mps2": dose_effects,
        "dose_monotonic": bool(
            dose_effects[0] <= dose_effects[1] + 1.0e-6
            and dose_effects[1] <= dose_effects[2] + 1.0e-6
        ),
        "fixed_window_audit": {
            "eligibility_slot_events": int(fixed_eligibility.sum()),
            "comparable_dose_slot_events": int(comparable.sum()),
            "direction_success_rate": (
                None if not comparable.any() else float(window_success.sum() / comparable.sum())
            ),
            "no_response_rate": (
                None if not comparable.any() else float((comparable & ~response).sum() / comparable.sum())
            ),
            "all_doses_comparable_slot_events": int(all_doses_comparable.sum()),
            "dose_monotonicity_rate": (
                None if not all_doses_comparable.any()
                else float(monotonic_events.sum() / all_doses_comparable.sum())
            ),
            "window_frames": [25, 75],
            "eligibility": "nearest nominal same-lane follower geometry at frame 24",
        },
        "influence_levels": influence_levels,
    })
    return result


def _natural_response_scores(
    model: torch.nn.Module,
    controller: CausalInfluenceResponsePolicy,
    *,
    states: np.ndarray,
    valid: np.ndarray,
    plans: np.ndarray,
    maps: np.ndarray,
    map_valid: np.ndarray,
    split_rows: np.ndarray,
    reference: ReactionEventReference,
    query: HumanResponseQuerySet,
    device: torch.device,
    influence_graph_config: dict[str, float | int],
    futures: int = 32,
    event_batch: int = 8,
    seed: int = 20260913,
) -> dict[str, Any]:
    """Score independent stochastic response futures on every natural event."""
    row_lookup = {int(row): index for index, row in enumerate(np.asarray(split_rows, np.int64))}
    scores_75: list[float] = []
    scores_25: list[float] = []
    legacy_25: list[float] = []
    recordings: list[int] = []
    event_keys: list[str] = []
    generated_diagnostics: list[dict[str, float | bool]] = []
    observed_diagnostics: list[dict[str, float | bool]] = []

    def response_diagnostic(sequence: np.ndarray, previous: float) -> dict[str, float | bool]:
        acceleration = np.asarray(sequence)[:, 0]
        jerk = np.asarray(sequence)[:, 1]
        below = acceleration <= float(previous) - .1
        starts = np.flatnonzero(np.convolve(below.astype(np.int8), np.ones(3, np.int8), mode="valid") == 3)
        latency = float(starts[0] * .04) if len(starts) else float(len(acceleration) * .04)
        peak_index = int(np.argmin(acceleration))
        recovered = acceleration >= float(previous) - .1
        recovery_starts = np.flatnonzero(
            np.convolve(recovered.astype(np.int8), np.ones(5, np.int8), mode="valid") == 5
        )
        recovery_starts = recovery_starts[recovery_starts >= peak_index]
        censored = not len(recovery_starts)
        return {
            "sustained_latency_s": latency,
            "peak_deceleration_mps2": float(acceleration.min()),
            "p95_abs_jerk_mps3": float(np.quantile(jerk, .95)),
            "peak_abs_jerk_mps3": float(jerk.max(initial=0.0)),
            "braking_dose_mps": float(np.maximum(-acceleration, 0.0).sum() * .04),
            "sustained_recovery_time_s": (
                float(recovery_starts[0] * .04) if not censored else float(len(acceleration) * .04)
            ),
            "recovery_censored": censored,
        }
    kept_query_indices = [
        q for q, event_index in enumerate(query.indices)
        if int(reference.events.row_index[int(event_index)]) in row_lookup
    ]
    for begin in range(0, len(kept_query_indices), event_batch):
        selected = kept_query_indices[begin:begin + event_batch]
        event_indices = [int(query.indices[q]) for q in selected]
        positions = np.asarray([
            row_lookup[int(reference.events.row_index[event_index])]
            for event_index in event_indices for _ in range(futures)
        ], np.int64)
        exogenous = WorldExogenousState.sample(
            len(positions), seed=seed + begin, response_steps=149,
            scene_refresh_responses=int(model.cfg.scene_refresh_responses),
            scene_dim=int(model.cfg.scene_latent_dim), agent_dim=int(model.cfg.agent_latent_dim),
        )
        sample = rollout(
            model, states[positions], valid[positions], plans[positions],
            maps[positions], map_valid[positions], device=device, history_frames=25,
            motion_seed=None, controller=controller, controller_deterministic=False,
            excluded_slots=(), influence_graph_config=influence_graph_config,
            exogenous_state=exogenous,
            record_policy_trace=False,
        )
        actions = torch.from_numpy(sample.background_actions[..., 0]).to(device)
        actions = actions.reshape(len(selected), futures, 149, 6)
        scale = torch.as_tensor(query.response_iqr, device=device)
        for local, (q, event_index) in enumerate(zip(selected, event_indices)):
            onset = int(reference.events.local_onset_frame[event_index]) - ANCHOR_INDEX
            follower = int(reference.events.follower_slot[event_index]) - 1
            if onset <= 0 or onset + 75 > 149:
                continue
            acceleration = actions[local, :, onset:onset + 75, follower]
            previous = actions[local, :, onset - 1, follower, None]
            jerk = (acceleration - torch.cat((previous, acceleration[:, :-1]), dim=1)).abs() / .04
            response = torch.stack((acceleration, jerk), dim=-1)
            observed = torch.as_tensor(query.response[q], device=device)
            scores_75.append(float(event_energy_score(response, observed[:75], scale) / np.sqrt(75.0)))
            scores_25.append(float(event_energy_score(response[:, :25], observed[:25], scale) / 5.0))
            legacy_25.append(float(legacy_event_energy_score(response[:, :25], observed[:25], torch.ones_like(scale))))
            recordings.append(int(reference.events.recording_id[event_index]))
            event_keys.append(event_identity(
                reference.events.recording_id[event_index], reference.events.leader_id[event_index],
                reference.events.follower_id[event_index], reference.events.absolute_onset_frame[event_index],
            ))
            response_np = response.detach().cpu().numpy()
            observed_np = observed[:75].detach().cpu().numpy()
            previous_np = previous[:, 0].detach().cpu().numpy()
            for future, previous_value in zip(response_np, previous_np):
                generated_diagnostics.append(response_diagnostic(future, float(previous_value)))
            observed_previous = float(reference.events.initial_conditions[event_index, 4])
            observed_diagnostics.append(response_diagnostic(observed_np, observed_previous))
    if not scores_75:
        raise RuntimeError("natural response evaluation found no replayable events")
    diagnostic_names = tuple(generated_diagnostics[0])
    return {
        "events": len(scores_75), "recordings": len(np.unique(recordings)), "futures_per_event": futures,
        "event_keys": event_keys, "recording_id": recordings,
        "fair_energy_75_normalized": scores_75,
        "fair_energy_25_normalized": scores_25,
        "legacy_raw_v_25": legacy_25,
        "means": {
            "fair_energy_75_normalized": float(np.mean(scores_75)),
            "fair_energy_25_normalized": float(np.mean(scores_25)),
            "legacy_raw_v_25": float(np.mean(legacy_25)),
        },
        "response_diagnostics": {
            "generated": {
                name: (
                    float(np.mean([float(item[name]) for item in generated_diagnostics]))
                    if name != "recovery_censored"
                    else float(np.mean([bool(item[name]) for item in generated_diagnostics]))
                ) for name in diagnostic_names
            },
            "observed": {
                name: (
                    float(np.mean([float(item[name]) for item in observed_diagnostics]))
                    if name != "recovery_censored"
                    else float(np.mean([bool(item[name]) for item in observed_diagnostics]))
                ) for name in diagnostic_names
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-config", type=Path, default=ROOT / "hierarchical_world_model/config/world_model.yaml")
    parser.add_argument("--controller-config", type=Path, default=ROOT / "hierarchical_world_model/config/cih_world_model.yaml")
    parser.add_argument("--candidate", type=Path, default=ROOT / "results/hierarchical_world_model/cih_wm/candidate_unaccepted/response_policy.pt")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--limit", type=int, default=None, help="diagnostic subset only; omitted means the complete split")
    parser.add_argument("--causal-rows", type=int, default=512)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validation-report", type=Path, default=None,
                        help="required for --split test; accepted complete formal validation for this checkpoint")
    args = parser.parse_args()

    world_config = load_protocol_config(args.world_config.resolve())
    controller_config = _load_yaml(args.controller_config.resolve())
    if controller_config.get("schema_name") != "causal_influence_hierarchical_world_model":
        raise ValueError("controller config is not a CIH-WM contract")
    if bool(controller_config["method"].get("ood_use_nominal_shadow", False)):
        raise ValueError("CIH-WM forbids nominal-shadow actions as model inputs")
    device = select_device(world_config["training"].get("device", "auto"))
    set_seed(int(world_config["training"]["seed"]))
    experiment = prepare_experiment_data(world_config, ROOT)
    split_rows = experiment.test_rows if args.split == "test" else experiment.validation_rows
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        split_rows = split_rows[:args.limit]
    if args.split == "test":
        if args.validation_report is None or not args.validation_report.is_file():
            raise RuntimeError("formal test requires --validation-report from an accepted complete formal validation")
        validation = json.loads(args.validation_report.read_text(encoding="utf-8"))
        if not (
            validation.get("split") == "validation"
            and validation.get("complete_split") is True
            and validation.get("accepted") is True
            and validation.get("evaluation_passed") is True
            and Path(str(validation.get("checkpoint", ""))).resolve() == args.candidate.resolve()
        ):
            raise RuntimeError("validation report is not an accepted complete formal evaluation of this candidate")
    method, checkpoint = CausalInfluenceHierarchicalWorldModel.load(
        controller_config,
        root=ROOT,
        device=device,
        response_checkpoint=args.candidate.resolve(),
    )
    model = method.factual_dynamics
    controller = method.response_policy.eval()
    influence_config = method.influence_graph_config
    training_provenance = _training_provenance(
        args.candidate.resolve(), controller_config
    )
    arrays = experiment.bundle.arrays
    full_states = np.asarray(arrays["agent_states"][split_rows], np.float32)
    full_valid = np.asarray(arrays["agent_valid"][split_rows], bool)
    states, valid = scoped_canonical_trajectory(full_states, full_valid)
    maps = np.asarray(arrays["map_polylines"][split_rows], np.float32)
    map_valid = np.asarray(arrays["map_polyline_valid"][split_rows], bool)
    cache = args.candidate.parent / args.split / "frozen_plans"
    cache.mkdir(parents=True, exist_ok=True)
    with nullcontext(cache):
        plans = frozen_diffusion_plans(
            experiment.bundle, split_rows, checkpoint=world_config["paths"]["diffusion_checkpoint"],
            output_dir=cache, device=device, batch_size=32, ddim_steps=20,
            experiment_scope=str(world_config["training"].get("experiment_scope", "full")),
        )
        response_plans = complete_endogenous_response_plans(
            plans, full_states, full_valid
        )
        baseline = _rollout_chunks(
            model, states, valid, plans, maps, map_valid,
            controller=NoReactionController().to(device), device=device,
        )
        candidate = _rollout_chunks(
            model, states, valid, plans, maps, map_valid,
            controller=controller, device=device,
            influence_graph_config=influence_config,
        )
        active = valid[:, ANCHOR_INDEX, 1:]
        target = states[:, ANCHOR_INDEX + 1:174]
        baseline_metrics = _factual_metrics(_concat(baseline, "states"), target, active)
        candidate_metrics = _factual_metrics(_concat(candidate, "states"), target, active)
        baseline_actions = _concat(baseline, "background_actions")
        candidate_actions = _concat(candidate, "background_actions")
        action_delta = np.abs(candidate_actions - baseline_actions)
        candidate_active = _controller_diagnostics(candidate, "active").astype(bool)
        event_mask = np.asarray(arrays["is_evt_tail"][split_rows], bool)
        non_event = ~event_mask
        factual_delta = {
            key: float(candidate_metrics[key] - baseline_metrics[key])
            for key in ("ADE_m", "FDE_m", "P95_displacement_error_m")
        }
        same_all_slot_inputs = (
            np.array_equal(states, full_states) and np.array_equal(valid, full_valid)
            and np.array_equal(plans, response_plans)
        )
        if same_all_slot_inputs:
            all_baseline, all_candidate = baseline, candidate
        else:
            all_baseline = _rollout_chunks(
                model, full_states, full_valid, response_plans, maps, map_valid,
                controller=NoReactionController().to(device), device=device, excluded_slots=(),
            )
            all_candidate = _rollout_chunks(
                model, full_states, full_valid, response_plans, maps, map_valid,
                controller=controller, device=device, excluded_slots=(),
                influence_graph_config=influence_config,
            )
        all_active = full_valid[:, ANCHOR_INDEX, 1:]
        all_target = full_states[:, ANCHOR_INDEX + 1:174]
        all_baseline_metrics = _factual_metrics(
            _concat(all_baseline, "states"), all_target, all_active
        )
        all_candidate_metrics = _factual_metrics(
            _concat(all_candidate, "states"), all_target, all_active
        )
        response_active = np.zeros_like(all_active)
        response_active[:, 1] = all_active[:, 1]
        response_baseline_metrics = _factual_metrics(
            _concat(all_baseline, "states"), all_target, response_active
        )
        response_candidate_metrics = _factual_metrics(
            _concat(all_candidate, "states"), all_target, response_active
        )
        response_factual_delta = {
            key: float(response_candidate_metrics[key] - response_baseline_metrics[key])
            for key in ("ADE_m", "FDE_m", "P95_displacement_error_m")
        }
        tolerance = world_config["training"]
        noninferior = all(
            factual_delta[key] <= float(tolerance[f"factual_{key.split('_m')[0].lower()}_tolerance_m"])
            if key != "P95_displacement_error_m"
            else factual_delta[key] <= float(tolerance["factual_p95_tolerance_m"])
            for key in factual_delta
        ) and all(
            factual_delta[key] <= float(controller_config["evaluation"]["factual_relative_tolerance"]) * baseline_metrics[key]
            for key in factual_delta
        ) and all(
            candidate_metrics[key] <= float(ACCEPTANCE_GATES["factual_limits_m"][key])
            for key in ACCEPTANCE_GATES["factual_limits_m"]
        )
        response_noninferior = all(
            response_factual_delta[key]
            <= float(
                controller_config["evaluation"]["factual_absolute_tolerance_m"][
                    "p95" if key == "P95_displacement_error_m" else key.split("_m")[0].lower()
                ]
            )
            for key in response_factual_delta
        ) and all(
            response_factual_delta[key]
            <= float(controller_config["evaluation"]["factual_relative_tolerance"])
            * response_baseline_metrics[key]
            for key in response_factual_delta
        )
        all_baseline_actions = _concat(all_baseline, "background_actions")
        all_candidate_actions = _concat(all_candidate, "background_actions")
        all_action_delta = np.abs(all_candidate_actions - all_baseline_actions)
        all_candidate_active = _controller_diagnostics(all_candidate, "active").astype(bool)
        causal = _causal_probe(
            model, full_states, full_valid, response_plans, maps, map_valid, controller=controller,
            device=device, rows=args.causal_rows,
            influence_graph_config=influence_config,
        )
        event_reference = ReactionEventReference.load(
            ROOT / controller_config["paths"]["event_reference"] / args.split
        )
        response_query = HumanResponseQuerySet.load(
            ROOT / controller_config["paths"]["human_response_evidence"] / args.split
        )
        candidate_response = _natural_response_scores(
            model, controller, states=full_states, valid=full_valid,
            plans=response_plans, maps=maps, map_valid=map_valid,
            split_rows=split_rows, reference=event_reference, query=response_query,
            device=device, influence_graph_config=influence_config,
        )
        supervised_path = args.candidate.parent / "response_supervised.pt"
        supervised_response = None
        ppo_increment = None
        if supervised_path.is_file() and supervised_path.resolve() != args.candidate.resolve():
            supervised_controller = build_response_policy(
                controller_config, root=ROOT, device=device, checkpoint=supervised_path
            ).eval()
            supervised_response = _natural_response_scores(
                model, supervised_controller, states=full_states, valid=full_valid,
                plans=response_plans, maps=maps, map_valid=map_valid,
                split_rows=split_rows, reference=event_reference, query=response_query,
                device=device, influence_graph_config=influence_config,
            )
            final_values = np.asarray(candidate_response["fair_energy_75_normalized"])
            supervised_values = np.asarray(supervised_response["fair_energy_75_normalized"])
            delta = supervised_values - final_values
            interval = recording_cluster_bootstrap(
                delta, np.asarray(candidate_response["recording_id"]), seed=20260913,
            )
            ppo_increment = {
                "mean_supervised_minus_final": float(delta.mean()),
                "recording_cluster_bootstrap": interval,
                "significant_improvement": bool(interval["ci95_low"] > 0.0),
            }
    direct = causal["influence_levels"]["direct"]
    fixed_window = causal["fixed_window_audit"]
    distribution_passed = bool(
        ppo_increment is None or ppo_increment["significant_improvement"]
    )
    evaluation_passed = bool(
        noninferior
        and response_noninferior
        and causal["selected_slot_frames"] > 0
        and causal["executed_action_slot_frames"] > 0
        and causal["executed_action_direction_agreement"] is not None
        and causal["executed_action_direction_agreement"] >= 0.95
        and direct["executed_action_slot_frames"] > 0
        and causal["preprobe_max_abs_action_difference_mps2"] == 0.0
        and causal["controller_induced_jerk_within_limit"]
        and causal["dose_monotonic"]
        and fixed_window["direction_success_rate"] is not None
        and fixed_window["direction_success_rate"] >= 0.95
        and fixed_window["dose_monotonicity_rate"] is not None
        and fixed_window["dose_monotonicity_rate"] >= 0.95
        and distribution_passed
    )
    selection_eligible = bool(
        args.split == "validation" and args.limit is None
        and training_provenance["passed"] and evaluation_passed
    )
    report = {
        "schema": "cih_world_model_evaluation",
        "split": args.split,
        "complete_split": args.limit is None,
        "sequences": int(len(split_rows)),
        "checkpoint": str(args.candidate.resolve()),
        "checkpoint_sha256": file_sha256(args.candidate.resolve()),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_diff_hash": hashlib.sha256(
            subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
        ).hexdigest(),
        "world_checkpoint": str(ROOT / world_config["paths"]["evaluation_checkpoint"]),
        "world_checkpoint_epoch": int(checkpoint["epoch"]),
        "protocol": {
            "factual": "released scoped comparison plus all-slot response reconstruction under the same frozen plans",
            "controller": "frozen diffusion-plan-conditioned factual transition; realized-state direct follower routing; mechanism-guided, constraint-preserving human-response calibration",
            "causal": "paired ADS braking rollouts use the nominal branch only as an evaluator reference; no nominal action, future highD action, or intervention label enters the policy",
            "lateral": "factual yaw-rate reconstruction only; causal lateral response is disabled pending evidence",
        },
        "training_provenance": training_provenance,
        "factual": {
            "released_comparison_scope": {
                "factual_dynamics": baseline_metrics,
                "cih_world_model": candidate_metrics,
                "delta": factual_delta,
                "passed": bool(noninferior),
                "excluded_slots": [],
            },
            "all_slot_response_scope": {
                "factual_dynamics": all_baseline_metrics,
                "cih_world_model": all_candidate_metrics,
                "excluded_slots": [],
            },
            "same_rear_response_reconstruction": {
                "factual_dynamics": response_baseline_metrics,
                "cih_world_model": response_candidate_metrics,
                "delta": response_factual_delta,
                "passed": bool(response_noninferior),
            },
        },
        "non_event_action_drift": {
            "mean_abs_action_difference_mps2": float(all_action_delta[non_event].mean()) if non_event.any() else None,
            "p95_abs_action_difference_mps2": float(np.quantile(all_action_delta[non_event], .95)) if non_event.any() else None,
            "event_rows": int(event_mask.sum()),
            "non_event_rows": int(non_event.sum()),
        },
        "controller_scope_audit": {
            "released_comparison_excluded_slots": [],
            "response_and_ads_excluded_slots": [],
            "active_slot_frames_in_all_slot_scope": int(all_candidate_active.sum()),
            "changed_action_slot_frames_in_all_slot_scope": int((all_action_delta[..., 0] > 1.0e-6).sum()),
            "material_action_change_slot_frames_in_all_slot_scope": int((all_action_delta[..., 0] > 0.05).sum()),
            "maximum_abs_action_difference_mps2_in_all_slot_scope": float(all_action_delta[..., 0].max(initial=0.0)),
        },
        "causal_probe": causal,
        "human_response_distribution": {
            "candidate": candidate_response,
            "supervised": supervised_response,
            "ppo_increment": ppo_increment,
            "passed": distribution_passed,
        },
        "evaluation_passed": evaluation_passed,
        "accepted": bool(selection_eligible),
        "promotion_status": (
            "validation_selected_for_one_shot_test"
            if selection_eligible else
            "not_promotable_until_complete_validation_and_training_provenance_pass"
        ),
    }
    output = args.output or ROOT / controller_config["paths"]["output_dir"] / f"cih_{args.split}_{args.candidate.stem}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

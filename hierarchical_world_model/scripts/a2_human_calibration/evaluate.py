#!/usr/bin/env python3
"""Formal full-split evaluation and one-shot acceptance gate."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

from common import ROOT, load_config, require_aligned_factual_protocol, result_root
from train import arrays_for, plans_for, policy_config, event_indices, event_key

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.data import prepare_experiment_data
from hierarchical_world_model.src.human_response_prior import HumanResponseQuerySet
from hierarchical_world_model.src.reaction_controller import A2HumanCalibrationController, IDMResidualReactionController, NoReactionController
from hierarchical_world_model.src.reaction_evidence import ReactionEventReference, energy_score, recording_cluster_bootstrap
from hierarchical_world_model.src.reaction_training import reaction_controller_rollout
from hierarchical_world_model.src.rule_models import RuleModelBundle
from hierarchical_world_model.src.train import load_checkpoint
from world_model.src.core.utils import select_device


def load_adapter(path, rule, config, device):
    controller = A2HumanCalibrationController(rule, a2_checkpoint=str(ROOT / config["paths"]["baseline_checkpoint"]), device=device).to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    controller.adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    return controller.eval()


def policy_on_stress(model, arrays, plans, controller, config, device, *, chunk: int) -> dict:
    """Measure stochastic all-background policy drift; this is not a factual gate."""
    all_error, all_final, per_row = [], [], []
    for start in range(0, len(plans), chunk):
        stop = min(start + chunk, len(plans))
        values = {name: value[start:stop] for name, value in arrays.items()}
        rollout = reaction_controller_rollout(
            model, states=values["agent_states"], valid=values["agent_valid"], soft_plans=plans[start:stop],
            maps=values["map_polylines"], map_valid=values["map_polyline_valid"], controller=controller,
            device=device, motion_seed=17000 + start, config=config, deterministic_response=False,
        )
        target = values["agent_states"][:, 25:174, 1:, :2]; valid = values["agent_valid"][:, 25:174, 1:]
        error = np.linalg.norm(rollout.states[:, :, 1:, :2] - target, axis=-1)
        all_error.append(error[valid]); all_final.append(error[:, -1][valid[:, -1]])
        per_row.extend({"row_index": int(row), "ade_m": float(error[i][valid[i]].mean()), "fde_m": float(error[i, -1][valid[i, -1]].mean())} for i, row in enumerate(values["row_index"]))
    error, final = np.concatenate(all_error), np.concatenate(all_final)
    return {"ade_m": float(error.mean()), "fde_m": float(final.mean()), "p95_m": float(np.quantile(error, .95)), "per_row": per_row}


def normalized_energy(futures: np.ndarray, observed: np.ndarray, iqr: np.ndarray) -> float:
    return energy_score(futures / iqr[None, None], observed / iqr[None])


RESPONSE_METRICS = ("latency_s", "peak_acceleration_mps2", "peak_abs_jerk_mps3", "minimum_gap_m", "maximum_closing_mps", "minimum_ttc_s", "recovery_mps2")


def response_metrics(action: np.ndarray, states: np.ndarray, onset: int, follower: int, initial_acceleration: float) -> np.ndarray:
    """Per-future temporal response diagnostics from only executed history."""
    acceleration = action[:, onset:onset + 25, follower]
    previous = action[:, onset - 1:onset + 24, follower]
    jerk = np.abs(acceleration - previous) / .04
    leader, rear = states[:, onset:onset + 25, 0], states[:, onset:onset + 25, follower + 1]
    gap = leader[..., 0] - rear[..., 0] - 4.8
    closing = rear[..., 2] - leader[..., 2]
    ttc = np.where(closing > 1.e-4, gap / np.maximum(closing, 1.e-4), 10.)
    crossed = acceleration <= float(initial_acceleration) - .1
    latency = np.where(crossed.any(1), crossed.argmax(1) * .04, 25 * .04)
    return np.stack((
        latency, acceleration.min(1), np.quantile(jerk, .95, axis=1), gap.min(1),
        closing.max(1), np.clip(ttc, 0., 10.).min(1), np.abs(acceleration[:, -1] - float(initial_acceleration)),
    ), axis=1)


def observed_response_metrics(reference: ReactionEventReference, index: int) -> np.ndarray:
    trajectory = reference.events.trajectory[index, 25:50]
    acceleration, jerk = trajectory[:, 0], trajectory[:, 1]
    initial = float(reference.events.initial_conditions[index, 4])
    crossed = acceleration <= initial - .1
    latency = float(crossed.argmax() * .04) if crossed.any() else 25 * .04
    return np.asarray((latency, acceleration.min(), np.quantile(jerk, .95), trajectory[:, 3].min(), trajectory[:, 4].max(), trajectory[:, 5].min(), abs(acceleration[-1] - initial)), np.float64)


def event_scores(model, arrays, plans, reference, query, controller, config, device, futures: int, *, selected_indices: np.ndarray | None = None) -> dict:
    lookup = {int(row): index for index, row in enumerate(arrays["row_index"])}
    query_lookup = {int(key): index for index, key in enumerate(query.event_key)}
    rows = []
    indices = event_indices(reference, arrays) if selected_indices is None else np.asarray(selected_indices, np.int64)
    for ordinal, index in enumerate(indices):
        query_index = query_lookup[event_key(reference, int(index))]
        row = lookup[int(reference.events.row_index[index])]
        values = {name: np.repeat(value[row:row + 1], futures, axis=0) for name, value in arrays.items() if name != "row_index"}
        rollout = reaction_controller_rollout(
            model, states=values["agent_states"], valid=values["agent_valid"], soft_plans=np.repeat(plans[row:row + 1], futures, axis=0),
            maps=values["map_polylines"], map_valid=values["map_polyline_valid"], controller=controller,
            device=device, motion_seed=26000 + ordinal * 1009, config=config, deterministic_response=False,
        )
        onset = int(reference.events.local_onset_frame[index]) - 24; follower = int(reference.events.follower_slot[index]) - 1
        action = rollout.background_actions[:, :, follower, 0]
        acceleration = action[:, onset:onset + 25]
        jerk = np.abs(acceleration - action[:, onset - 1:onset + 24]) / .04
        response = np.stack((acceleration, jerk), -1)
        observed = query.response[query_index]
        target_metrics = observed_response_metrics(reference, int(index))
        model_metrics = response_metrics(action, rollout.states, onset, follower, float(reference.events.initial_conditions[index, 4]))
        rows.append({
            "event_key": int(query.event_key[query_index]), "recording_id": int(query.recording_id[query_index]),
            "support": str(query.support_label[query_index]), "raw_es": energy_score(response, observed),
            "normalized_es": normalized_energy(response, observed, query.response_iqr),
            "response_metric_error": np.abs(model_metrics - target_metrics[None]).mean(0).tolist(),
            "response_metric_target": target_metrics.tolist(),
            "target_pair_collision": float((rollout.collision_pairs[:, :, 0, follower + 1]).any(axis=1).mean()),
            "unrelated_collision": float((rollout.collision_pairs.any(axis=(1, 2, 3)) & ~rollout.collision_pairs[:, :, 0, follower + 1].any(axis=1)).mean()),
            "offroad": float(rollout.offroad.any(axis=(1, 2)).mean()),
        })
    return {"events": rows, "raw_mean": float(np.mean([row["raw_es"] for row in rows])), "normalized_mean": float(np.mean([row["normalized_es"] for row in rows]))}


def response_dimension_gate(final: dict, reference: dict, train_reference: ReactionEventReference, *, draws: int, seed: int) -> dict:
    """Statistical no-material-regression gate for all temporal diagnostics."""
    paired = {row["event_key"]: row for row in reference["events"]}
    by_metric = {name: [] for name in RESPONSE_METRICS}
    records = []
    for row in final["events"]:
        other = paired.get(row["event_key"])
        if other is None:
            continue
        records.append(row["recording_id"])
        delta = np.asarray(row["response_metric_error"]) - np.asarray(other["response_metric_error"])
        for metric, value in zip(RESPONSE_METRICS, delta):
            by_metric[metric].append(value)
    train_values = np.asarray([observed_response_metrics(train_reference, int(index)) for index in train_reference.events.indices(train_reference.supported_cells)])
    result = {}
    recording_ids = np.asarray(records)
    for column, metric in enumerate(RESPONSE_METRICS):
        values = np.asarray(by_metric[metric], np.float64)
        bootstrap = recording_cluster_bootstrap(values, recording_ids, draws=draws, seed=seed + column)
        groups = [values[recording_ids == record] for record in np.unique(recording_ids)]
        rng = np.random.default_rng(seed + column)
        samples = np.asarray([np.concatenate([groups[index] for index in rng.integers(len(groups), size=len(groups))]).mean() for _ in range(draws)])
        scale = float(max(np.subtract(*np.quantile(train_values[:, column], [.75, .25])), 1.e-6))
        result[metric] = {**bootstrap, "ucb95": float(np.quantile(samples, .95)), "train_iqr": scale, "allowed_degradation": .1 * scale}
    return {"metrics": result, "passed": bool(all(value["ucb95"] <= value["allowed_degradation"] for value in result.values()))}


def compare(left: dict, right: dict, key: str, *, draws: int, seed: int, supported_only: bool = False) -> dict:
    by_key = {row["event_key"]: row for row in right["events"]}
    pairs = [(row, by_key[row["event_key"]]) for row in left["events"] if row["event_key"] in by_key and (not supported_only or row["support"] == "empirically_supported")]
    delta = np.asarray([a[key] - b[key] for a, b in pairs])
    records = np.asarray([a["recording_id"] for a, _ in pairs])
    result = recording_cluster_bootstrap(delta, records, draws=draws, seed=seed)
    groups = [delta[records == record] for record in np.unique(records)]
    rng = np.random.default_rng(seed)
    samples = np.asarray([np.concatenate([groups[index] for index in rng.integers(len(groups), size=len(groups))]).mean() for _ in range(draws)])
    result["ucb95"] = float(np.quantile(samples, .95))
    return result


def non_event_diagnostics(model, arrays, plans, supervised, final, config, device) -> dict:
    """Fixed ordinary-row audit preventing an always-on calibration layer."""
    count, futures = min(265, len(plans)), int(config["evaluation"]["non_event_futures"])
    rows = np.linspace(0, len(plans) - 1, count, dtype=np.int64)
    values = {name: np.repeat(value[rows], futures, axis=0) for name, value in arrays.items() if name != "row_index"}
    repeated_plans = np.repeat(plans[rows], futures, axis=0)
    runtime = policy_config(config["ppo"])
    def audit(controller):
        rollout = reaction_controller_rollout(model, states=values["agent_states"], valid=values["agent_valid"], soft_plans=repeated_plans, maps=values["map_polylines"], map_valid=values["map_polyline_valid"], controller=controller, device=device, motion_seed=41000, config=runtime, deterministic_response=False)
        hiqr = rollout.base_background_actions[..., 0]; final_action = rollout.background_actions[..., 0]
        calibration = rollout.controller_diagnostics["calibration_correction_ax"]
        return {"mean_abs_new_minus_hiqr": float(np.abs(final_action - hiqr).mean()), "p95_abs_new_minus_hiqr": float(np.quantile(np.abs(final_action - hiqr), .95)), "activation_rate": float((np.abs(calibration) > .05).mean()), "large_calibration_rate": float((np.abs(calibration) > .25).mean())}
    supervised_values, final_values = audit(supervised), audit(final)
    passed = final_values["mean_abs_new_minus_hiqr"] <= 1.10 * supervised_values["mean_abs_new_minus_hiqr"] and final_values["p95_abs_new_minus_hiqr"] <= 1.10 * supervised_values["p95_abs_new_minus_hiqr"] and final_values["activation_rate"] <= supervised_values["activation_rate"] + .05 and final_values["large_calibration_rate"] <= supervised_values["large_calibration_rate"] + .01
    return {"supervised": supervised_values, "final": final_values, "passed": bool(passed)}


def ood_profiles() -> dict[str, np.ndarray]:
    profiles = {}
    for magnitude in (2., 4., 6., 8.):
        profile = np.zeros(149, np.float32); profile[25:50] = -magnitude; profiles[f"brake_{magnitude:g}"] = profile
    ramp = np.zeros(149, np.float32); ramp[25:35] = np.linspace(0., -6., 10); ramp[35:40] = -6.; ramp[40:50] = np.linspace(-6., 0., 10); profiles["ramp"] = ramp
    pulse = np.zeros(149, np.float32); pulse[25:30] = -6.; profiles["pulse"] = pulse
    return profiles


def ood_diagnostics(model, arrays, plans, final, config, device) -> dict:
    count = min(int(config["evaluation"]["ood_rows"]), len(plans)); rows = np.linspace(0, len(plans) - 1, count, dtype=np.int64)
    values = {name: value[rows] for name, value in arrays.items() if name != "row_index"}; runtime = policy_config(config["ppo"])
    selected, agreed, finite, bounds, jerk_ok, unrelated, offroad = 0, 0, True, True, True, 0, 0
    conditions = {}
    for order, (name, profile) in enumerate(ood_profiles().items()):
        baseline = reaction_controller_rollout(model, states=values["agent_states"], valid=values["agent_valid"], soft_plans=plans[rows], maps=values["map_polylines"], map_valid=values["map_polyline_valid"], controller=final, device=device, motion_seed=51000 + order, config=runtime, deterministic_response=False)
        intervention = reaction_controller_rollout(model, states=values["agent_states"], valid=values["agent_valid"], soft_plans=plans[rows], maps=values["map_polylines"], map_valid=values["map_polyline_valid"], controller=final, device=device, motion_seed=51000 + order, config=runtime, deterministic_response=False, ego_acceleration_offset=profile)
        base_action, intervention_action = baseline.background_actions[..., 0], intervention.background_actions[..., 0]
        base_idm, intervention_idm = baseline.controller_diagnostics["rule_action_ax"], intervention.controller_diagnostics["rule_action_ax"]
        role = baseline.controller_diagnostics["influence_role"] == 1
        direct = baseline.controller_diagnostics["influence_direct"].astype(bool) & intervention.controller_diagnostics["influence_direct"].astype(bool) & role
        idm_delta = intervention_idm - base_idm; model_delta = intervention_action - base_action; usable = direct & (np.abs(idm_delta) > .25)
        selected += int(usable.sum()); agreed += int((np.sign(idm_delta[usable]) * np.sign(model_delta[usable]) >= 0).sum())
        calibration = intervention.controller_diagnostics["calibration_correction_ax"]
        jerk = np.abs(np.diff(calibration, axis=1)) / .04
        finite &= bool(np.isfinite(intervention.states).all() and np.isfinite(intervention_action).all())
        bounds &= bool((intervention_action >= -8.0001).all() and (intervention_action <= 4.0001).all())
        jerk_ok &= bool((jerk <= 12.0001).all())
        unrelated += int((intervention.collision_pairs.any(axis=(1, 2, 3)) & ~intervention.collision_pairs[:, :, 0].any(axis=(1, 2))).sum())
        offroad += int(intervention.offroad.any(axis=(1, 2)).sum())
        conditions[name] = {"selected": int(usable.sum()), "agreement": float((np.sign(idm_delta[usable]) * np.sign(model_delta[usable]) >= 0).mean()) if usable.any() else None}
    agreement = float(agreed / selected) if selected else 0.0
    passed = finite and bounds and jerk_ok and agreement >= .95 and unrelated == 0 and offroad == 0
    return {"conditions": conditions, "direction_agreement": agreement, "selected": selected, "finite": finite, "action_bounds": bounds, "controller_induced_jerk": jerk_ok, "introduced_unrelated_collision": unrelated, "response_induced_offroad": offroad, "passed": bool(passed)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args(); config, world = load_config(); require_aligned_factual_protocol(config)
    root = result_root(config)
    report_path = root / ("full_validation.json" if args.split == "validation" else "test.json")
    if report_path.exists(): raise RuntimeError(f"{report_path.name} is single-use and will not be overwritten")
    if args.split == "test":
        if not (root / "acceptance.json").is_file() or not json.loads((root / "acceptance.json").read_text()).get("accepted", False):
            raise RuntimeError("test requires an accepted validation artifact")
    device = select_device("auto"); model, _ = load_checkpoint(ROOT / world["paths"]["evaluation_checkpoint"], device=device)
    experiment = prepare_experiment_data(world, ROOT); rows = getattr(experiment, f"{args.split}_rows")
    arrays, plans = plans_for(experiment.bundle, rows, world, root / "cache" / args.split, device)
    reference = ReactionEventReference.load(ROOT / config["paths"]["event_reference"] / args.split)
    train_reference = ReactionEventReference.load(ROOT / config["paths"]["event_reference"] / "train")
    query = HumanResponseQuerySet.load(root / "human_support" / args.split); rule = RuleModelBundle.load(ROOT / config["paths"]["rule_model"])
    arms = {"frozen_hiqr": NoReactionController().to(device), "legacy_a2": IDMResidualReactionController(rule).to(device)}
    arms["legacy_a2"].load_state_dict(torch.load(ROOT / config["paths"]["baseline_checkpoint"], map_location=device, weights_only=False)["state_dict"])
    arms["supervised"] = load_adapter(root / "supervised.pt", rule, config, device)
    arms["final"] = load_adapter(root / "checkpoint.pt", rule, config, device)
    runtime = policy_config(config["ppo"]); chunk = int(config["evaluation"]["policy_on_stress_chunk_rows"])
    factual_metrics = {name: policy_on_stress(model, arrays, plans, arm.eval(), runtime, device, chunk=chunk) for name, arm in arms.items()}
    scores = {name: event_scores(model, arrays, plans, reference, query, arm.eval(), runtime, device, int(config["evaluation"]["validation_futures"])) for name, arm in arms.items()}
    baseline = factual_metrics["frozen_hiqr"]
    tolerance = config["evaluation"]["factual_absolute_tolerance_m"]
    candidate_delta = {key: factual_metrics["final"][f"{key}_m"] - baseline[f"{key}_m"] for key in ("ade", "fde", "p95")}
    factual_pass = all(candidate_delta[key] <= float(tolerance[key]) and candidate_delta[key] / max(baseline[f"{key}_m"], 1.e-6) <= float(config["evaluation"]["factual_relative_tolerance"]) for key in candidate_delta)
    a2_delta = {key: factual_metrics["legacy_a2"][f"{key}_m"] - baseline[f"{key}_m"] for key in ("ade", "fde", "p95")}
    a2_factual_feasible = all(a2_delta[key] <= float(tolerance[key]) and a2_delta[key] / max(baseline[f"{key}_m"], 1.e-6) <= float(config["evaluation"]["factual_relative_tolerance"]) for key in a2_delta)
    draws, seed = int(config["evaluation"]["bootstrap_draws"]), int(config["evaluation"]["bootstrap_seed"])
    ppo_increment = compare(scores["supervised"], scores["final"], "normalized_es", draws=draws, seed=seed, supported_only=True)
    a2_norm = compare(scores["final"], scores["legacy_a2"], "normalized_es", draws=draws, seed=seed, supported_only=True)
    a2_raw = compare(scores["final"], scores["legacy_a2"], "raw_es", draws=draws, seed=seed)
    frozen_norm = compare(scores["final"], scores["frozen_hiqr"], "normalized_es", draws=draws, seed=seed, supported_only=True)
    frozen_raw = compare(scores["final"], scores["frozen_hiqr"], "raw_es", draws=draws, seed=seed)
    response_dimensions = response_dimension_gate(scores["final"], scores["legacy_a2"] if a2_factual_feasible else scores["frozen_hiqr"], train_reference, draws=draws, seed=seed)
    non_event = non_event_diagnostics(model, arrays, plans, arms["supervised"], arms["final"], config, device)
    ood = ood_diagnostics(model, arrays, plans, arms["final"], config, device)
    response_reference_pass = (a2_norm["ucb95"] <= 0.0 and a2_raw["ucb95"] <= 0.0) if a2_factual_feasible else (frozen_norm["ucb95"] <= 0.0 and frozen_raw["ucb95"] <= 0.0)
    accepted = bool(factual_pass and ppo_increment["lcb95"] > 0.0 and response_reference_pass and response_dimensions["passed"] and non_event["passed"] and ood["passed"])
    report = {"schema": "a2_human_calibration_evaluation", "split": args.split, "factual": factual_metrics, "event_scores": scores, "candidate_factual_delta": candidate_delta, "legacy_a2_factual_delta": a2_delta, "legacy_a2_factual_feasible": a2_factual_feasible, "ppo_increment": ppo_increment, "a2_normalized_noninferiority": a2_norm, "a2_raw_noninferiority": a2_raw, "frozen_normalized_noninferiority": frozen_norm, "frozen_raw_noninferiority": frozen_raw, "response_dimensions": response_dimensions, "non_event": non_event, "ood": ood, "accepted": accepted}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.split == "validation":
        (root / "acceptance.json").write_text(json.dumps({"accepted": accepted, "factual_noninferiority": factual_pass, "ppo_increment": ppo_increment["lcb95"] > 0.0, "legacy_a2_factual_feasible": a2_factual_feasible, "response_reference_noninferiority": response_reference_pass, "response_dimensions": response_dimensions["passed"], "non_event": non_event["passed"], "ood": ood["passed"], "test_not_run": True}, indent=2) + "\n")
    print(json.dumps({"accepted": accepted, "output": str(report_path)}, indent=2))


if __name__ == "__main__": main()

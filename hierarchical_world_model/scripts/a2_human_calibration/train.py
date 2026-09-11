#!/usr/bin/env python3
"""Train a stochastic calibration adapter over a frozen A2 response prior."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import torch
from torch.nn import functional

from common import ROOT, load_config, require_aligned_factual_protocol, result_root

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.data import prepare_experiment_data
from hierarchical_world_model.src.human_response_prior import HumanResponseQuerySet
from hierarchical_world_model.src.human_response_training import (
    mechanism_auxiliary_loss, prefix_loo_event_energy_rewards,
)
from hierarchical_world_model.src.influence_graph import ROLE_SAME_LANE_FOLLOWER
from hierarchical_world_model.src.planner import complete_missing_background_plans, frozen_diffusion_plans
from hierarchical_world_model.src.reaction_controller import A2HumanCalibrationController
from hierarchical_world_model.src.reaction_evidence import EVALUATION_FRAMES, ReactionEventReference
from hierarchical_world_model.src.reaction_training import (
    PolicyTrainingConfig, ReactionEpisode, ReactionTrainingEnvironment, _gae,
    reaction_controller_rollout,
)
from hierarchical_world_model.src.rule_models import RuleModelBundle
from hierarchical_world_model.src.train import load_checkpoint
from world_model.src.core.utils import file_sha256, select_device


def arrays_for(bundle, rows: np.ndarray) -> dict[str, np.ndarray]:
    names = ("agent_states", "agent_valid", "map_polylines", "map_polyline_valid")
    result = {name: np.asarray(bundle.arrays[name])[rows] for name in names}
    result["row_index"] = np.asarray(rows, np.int64)
    return result


def plans_for(bundle, rows: np.ndarray, world: dict, output, device: torch.device) -> tuple[dict[str, np.ndarray], np.ndarray]:
    arrays = arrays_for(bundle, rows)
    plans = frozen_diffusion_plans(
        bundle, rows, checkpoint=ROOT / world["paths"]["diffusion_checkpoint"], output_dir=output,
        device=device, batch_size=int(world["training"]["validation_batch_size"]), ddim_steps=20,
        experiment_scope=world["training"].get("experiment_scope", "full"),
    )
    return arrays, complete_missing_background_plans(plans, arrays["agent_states"], arrays["agent_valid"])


def policy_config(values: dict) -> PolicyTrainingConfig:
    return PolicyTrainingConfig(
        rollout_steps=int(values["rollout_steps"]), episodes_per_rollout=64, event_episodes=64,
        non_event_episodes=0, synthetic_episodes=0, epochs_per_update=int(values["epochs_per_update"]),
        minibatch_size=int(values["minibatch_size"]), updates=int(values["maximum_updates"]),
        learning_rate=float(values["learning_rate_start"]), gamma=float(values["gamma"]),
        gae_lambda=float(values["gae_lambda"]), clip_ratio=float(values["clip_ratio"]),
        value_coefficient=float(values["value_coefficient"]), entropy_coefficient=float(values["entropy_coefficient"]),
        max_grad_norm=float(values["max_grad_norm"]), validation_interval_updates=int(values["quick_validation_interval"]),
        validation_events=int(values["quick_validation_events"]), validation_futures=int(values["quick_validation_futures"]),
        validation_patience=int(values["early_stopping_patience"]), validation_minimum_improvement=float(values["quick_improvement_fraction"]),
        seed=int(values["seed"]),
    )


def event_indices(reference: ReactionEventReference, arrays: dict[str, np.ndarray]) -> np.ndarray:
    rows = set(np.asarray(arrays["row_index"], np.int64).tolist())
    events = reference.events
    return np.asarray([index for index in events.indices(reference.supported_cells) if events.leader_slot[index] == 0 and int(events.row_index[index]) in rows], np.int64)


def query_lookup(query: HumanResponseQuerySet) -> dict[int, int]:
    return {int(key): index for index, key in enumerate(query.event_key)}


def event_key(reference: ReactionEventReference, index: int) -> int:
    events = reference.events
    return int(events.absolute_onset_frame[index] + 10_000_000 * events.recording_id[index])


def teacher_cache(environment: ReactionTrainingEnvironment, reference: ReactionEventReference, output, *, batch_size: int = 32) -> dict[str, torch.Tensor]:
    """Build adapter inputs on logged histories through the shared runtime context."""
    if output.exists():
        return torch.load(output, map_location="cpu", weights_only=False)
    records = {name: [] for name in ("features", "nominal", "a2_correction", "authority", "active", "previous", "target", "event")}
    pool = event_indices(reference, environment.arrays)
    for start in range(0, len(pool), batch_size):
        selected = pool[start:start + batch_size]
        episodes = [ReactionEpisode(environment.row_lookup[int(reference.events.row_index[index])], "event", int(index)) for index in selected]
        environment.reset(episodes)
        assert environment.world is not None and environment.logged_ego is not None and environment.logged_background is not None
        local_rows = np.asarray([episode.row_index for episode in episodes], np.int64)
        onsets = np.asarray([int(reference.events.local_onset_frame[index]) - 24 for index in selected], np.int64)
        previous_calibration = torch.zeros((len(selected), 6), device=environment.device)
        for step in range(environment.config.rollout_steps):
            transition = environment.world.advance_response(environment.logged_ego[:, step])
            for batch, index in enumerate(selected):
                offset = step - onsets[batch]
                if -EVALUATION_FRAMES <= offset < EVALUATION_FRAMES:
                    follower = int(reference.events.follower_slot[index]) - 1
                    for name, value in (
                        ("features", transition["controller_features"][batch, follower]),
                        ("nominal", transition["controller_a2_nominal_action_ax"][batch, follower]),
                        ("a2_correction", transition["controller_a2_correction_ax"][batch, follower]),
                        ("authority", transition["influence_authority"][batch, follower]),
                        ("active", transition["controller_active"][batch, follower]),
                        ("previous", previous_calibration[batch, follower]),
                        ("target", environment.logged_background[batch, step, follower]),
                    ):
                        records[name].append(value.detach().cpu())
                    records["event"].append(torch.tensor(offset >= 0))
            previous_calibration = transition["controller_calibration_correction_ax"].detach()
            if step + 1 < environment.config.rollout_steps:
                next_state = torch.from_numpy(environment.states[local_rows, 25 + step + 1]).to(environment.device)
                next_valid = torch.from_numpy(environment.valid[local_rows, 25 + step + 1]).to(environment.device)
                action = torch.zeros((len(selected), 6, 2), device=environment.device)
                action[..., 0] = environment.logged_background[:, step]
                environment.world.teacher_force_next_state(next_state, next_valid, action)
    cache = {name: torch.stack(value) for name, value in records.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    output.with_suffix(".json").write_text(json.dumps({
        "schema": "a2_human_calibration_teacher_forced_cache",
        "frames": {"pre_event": 25, "event": 25},
        "online_safe_fields": ["features", "nominal", "a2_correction", "authority", "active", "previous"],
        "offline_only_fields": ["target", "event"],
        "records": int(len(cache["features"])),
    }, indent=2) + "\n")
    return cache


def supervised_loss(controller: A2HumanCalibrationController, cache: dict[str, torch.Tensor], indices: torch.Tensor, *, prior_weight: float, non_weight: float) -> torch.Tensor:
    features = cache["features"][indices].to(next(controller.adapter.parameters()).device)
    distribution, _ = controller.adapter.distribution_and_value(features)
    mapped = controller.map_calibration_tensors(
        nominal=cache["nominal"][indices].to(features), a2_correction=cache["a2_correction"][indices].to(features),
        active=cache["active"][indices].to(features).bool(), authority=cache["authority"][indices].to(features),
        raw=distribution.mean, previous_calibration=cache["previous"][indices].to(features),
        minimum=-8.0, maximum=4.0, dt_s=.04,
    )
    target = cache["target"][indices].to(features)
    event = cache["event"][indices].to(features).bool()
    event_loss = functional.huber_loss(mapped["final"][event], target[event]) if event.any() else mapped["final"].sum() * 0.0
    non_loss = functional.huber_loss(mapped["final"][~event], target[~event]) if (~event).any() else mapped["final"].sum() * 0.0
    non_calibration = mapped["calibration"][~event].abs().mean() if (~event).any() else mapped["final"].sum() * 0.0
    return event_loss + non_loss + float(prior_weight) * distribution.mean.square().mean() + float(non_weight) * non_calibration


def natural_buffer(environment: ReactionTrainingEnvironment, reference: ReactionEventReference, query: HumanResponseQuerySet, groups: np.ndarray, futures: int, config: PolicyTrainingConfig, weights: dict[str, float]) -> dict[str, torch.Tensor]:
    episodes = []
    for index in groups:
        row = environment.row_lookup[int(reference.events.row_index[index])]
        episodes.extend(ReactionEpisode(row, "event", int(index)) for _ in range(futures))
    environment.reset(episodes)
    names = ("features", "raw_action", "log_prob", "value", "active", "done")
    buffer = {name: [] for name in names}
    actions = []
    for _ in range(config.rollout_steps):
        _, _, done, info = environment.step()
        for name in ("features", "raw_action", "log_prob", "value", "active"):
            buffer[name].append(info[name].detach())
        buffer["done"].append(done.detach()); actions.append(info["final_action"].detach())
    final = torch.stack(actions)
    reward = torch.zeros_like(final)
    mask = torch.zeros_like(buffer["active"][0], dtype=torch.bool)[None].expand_as(final).clone()
    lookup = query_lookup(query)
    scale = torch.as_tensor(query.response_iqr, device=final.device)
    for group, index in enumerate(groups):
        query_index = lookup.get(event_key(reference, int(index)))
        if query_index is None:
            continue
        support = float(weights[str(query.support_label[query_index])])
        if support == 0.0:
            continue
        onset = int(reference.events.local_onset_frame[index]) - 24
        follower = int(reference.events.follower_slot[index]) - 1
        start = group * futures
        acceleration = final[onset:onset + 25, start:start + futures, follower].transpose(0, 1)
        previous = final[onset - 1, start:start + futures, follower][:, None]
        jerk = (acceleration - torch.cat((previous, acceleration[:, :-1]), 1)).abs() / .04
        response = torch.stack((acceleration, jerk), -1)
        observed = torch.as_tensor(query.response[query_index], device=final.device)
        for prefix, contribution in prefix_loo_event_energy_rewards(response, observed, scale).items():
            reward[onset + prefix - 1, start:start + futures, follower] += support * contribution
        mask[onset:onset + 25, start:start + futures, follower] = True
    return {name: torch.stack(value) if isinstance(value, list) else value for name, value in {**buffer, "reward": reward, "mask": mask}.items()}


def ppo_step(controller: A2HumanCalibrationController, reference_adapter, optimizer, buffer: dict[str, torch.Tensor], config: PolicyTrainingConfig, values: dict, coefficient: float) -> tuple[dict, float]:
    advantage, returns = _gae(buffer["reward"], buffer["value"], buffer["done"], config.gamma, config.gae_lambda)
    active = (buffer["mask"] & buffer["active"]).reshape(-1)
    features = buffer["features"].reshape(-1, buffer["features"].shape[-1])[active]
    raw = buffer["raw_action"].reshape(-1, 2)[active]
    old = buffer["log_prob"].reshape(-1)[active]
    advantage, returns = advantage.reshape(-1)[active], returns.reshape(-1)[active]
    advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1.e-6)
    losses, kls, references = [], [], []
    for _ in range(int(values["epochs_per_update"])):
        current_kls = []
        for subset in torch.randperm(len(features), device=features.device).split(config.minibatch_size):
            log_prob, entropy, value = controller.evaluate_raw_action(features[subset], raw[subset])
            ratio = (log_prob - old[subset]).exp()
            clipped = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
            policy = -torch.minimum(ratio * advantage[subset], clipped * advantage[subset]).mean()
            current_dist, _ = controller.adapter.distribution_and_value(features[subset])
            with torch.no_grad(): ref_dist, _ = reference_adapter.distribution_and_value(features[subset])
            reference_kl = torch.distributions.kl_divergence(current_dist, ref_dist).sum(-1).mean()
            loss = policy + config.value_coefficient * functional.mse_loss(value, returns[subset]) - float(values["entropy_coefficient"]) * entropy.mean() + coefficient * reference_kl
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(controller.adapter.parameters(), config.max_grad_norm); optimizer.step()
            approx = float((old[subset] - log_prob).mean().detach()); current_kls.append(approx); kls.append(approx); references.append(float(reference_kl.detach())); losses.append(float(loss.detach()))
        if current_kls and max(current_kls) > float(values["inner_stop_kl"]):
            break
    mean_reference = float(np.mean(references))
    target = float(values["reference_kl_target"])
    if mean_reference > 1.5 * target: coefficient *= 2.0
    elif mean_reference < target / 1.5: coefficient *= .5
    coefficient = float(np.clip(coefficient, values["reference_kl_minimum"], values["reference_kl_maximum"]))
    return {"ppo_loss": float(np.mean(losses)), "approx_kl": float(np.mean(kls)), "reference_kl": mean_reference}, coefficient


def synthetic_mechanism_step(model, controller, arrays, plans, environment, optimizer, runtime, *, seed: int, pairs: int) -> tuple[torch.Tensor, dict]:
    """Paired, detached visited-state mechanism loss; it never becomes PPO reward."""
    rows = environment.rng.choice(np.arange(len(arrays["agent_states"])), size=pairs, replace=len(arrays["agent_states"]) < pairs)
    values = {name: value[rows] for name, value in arrays.items() if name != "row_index"}
    baseline = reaction_controller_rollout(model, states=values["agent_states"], valid=values["agent_valid"], soft_plans=plans[rows], maps=values["map_polylines"], map_valid=values["map_polyline_valid"], controller=controller, device=environment.device, motion_seed=seed, config=runtime, deterministic_response=False)
    intervention = reaction_controller_rollout(model, states=values["agent_states"], valid=values["agent_valid"], soft_plans=plans[rows], maps=values["map_polylines"], map_valid=values["map_polyline_valid"], controller=controller, device=environment.device, motion_seed=seed, intervention="brake", dose=8.0, intervention_start=25, intervention_duration_frames=25, config=runtime, deterministic_response=False)
    def mapped(rollout):
        diag = rollout.controller_diagnostics
        features = torch.as_tensor(diag["features"], device=environment.device)
        distribution, _ = controller.adapter.distribution_and_value(features)
        recorded = torch.as_tensor(diag["raw_action"], device=environment.device)
        innovation = (recorded - distribution.mean.detach()) / distribution.stddev.detach().clamp_min(1.e-6)
        raw = distribution.mean + distribution.stddev * innovation
        calibration = torch.as_tensor(diag["calibration_correction_ax"], device=environment.device)
        previous = torch.cat((torch.zeros_like(calibration[:, :1]), calibration[:, :-1]), dim=1)
        output = controller.map_calibration_tensors(
            nominal=torch.as_tensor(diag["a2_nominal_action_ax"], device=environment.device),
            a2_correction=torch.as_tensor(diag["a2_correction_ax"], device=environment.device),
            active=torch.as_tensor(diag["active"], device=environment.device).bool(),
            authority=torch.as_tensor(diag["influence_authority"], device=environment.device), raw=raw,
            previous_calibration=previous, minimum=-8.0, maximum=4.0, dt_s=.04,
        )
        direct_following = torch.as_tensor(diag["influence_direct"], device=environment.device).bool() & torch.as_tensor(diag["influence_role"], device=environment.device).eq(ROLE_SAME_LANE_FOLLOWER)
        return output["final"], torch.as_tensor(diag["rule_action_ax"], device=environment.device), direct_following
    base_action, base_idm, base_mask = mapped(baseline)
    intervention_action, intervention_idm, intervention_mask = mapped(intervention)
    mask = base_mask & intervention_mask
    loss, summary = mechanism_auxiliary_loss(model_intervention=intervention_action[mask], model_baseline=base_action[mask], idm_intervention=intervention_idm[mask], idm_baseline=base_idm[mask])
    return loss, {name: float(value.detach()) for name, value in summary.items()}


def recording_balanced_events(reference: ReactionEventReference, arrays: dict[str, np.ndarray], count: int) -> np.ndarray:
    """Choose a fixed, recording-balanced validation subset without online labels."""
    candidates = event_indices(reference, arrays)
    recordings = reference.events.recording_id[candidates]
    selected: list[int] = []
    for recording in np.unique(recordings):
        selected.append(int(candidates[np.flatnonzero(recordings == recording)[0]]))
        if len(selected) == count:
            return np.asarray(selected, np.int64)
    for candidate in candidates:
        if int(candidate) not in selected:
            selected.append(int(candidate))
        if len(selected) == count:
            break
    return np.asarray(selected, np.int64)


def quick_validation(model, arrays, plans, reference, query, controller, frozen_stress: dict, config: dict, runtime, device, selected_events: np.ndarray, factual_rows: np.ndarray) -> dict:
    """Fixed-CRN diagnostic: event ES plus a fixed policy-on stress population."""
    # Imported lazily because evaluate imports this module's shared data helpers.
    from evaluate import event_scores, policy_on_stress

    event = event_scores(model, arrays, plans, reference, query, controller.eval(), runtime, device,
                         int(config["ppo"]["quick_validation_futures"]), selected_indices=selected_events)
    subset = {name: value[factual_rows] for name, value in arrays.items()}
    stress_values = policy_on_stress(model, subset, plans[factual_rows], controller.eval(), runtime, device,
                             chunk=min(64, len(factual_rows)))
    tolerance = config["evaluation"]["factual_absolute_tolerance_m"]
    stress_pass = all(
        stress_values[f"{key}_m"] - frozen_stress[f"{key}_m"] <= float(tolerance[key])
        and (stress_values[f"{key}_m"] - frozen_stress[f"{key}_m"])
        / max(frozen_stress[f"{key}_m"], 1.e-6) <= float(config["evaluation"]["factual_relative_tolerance"])
        for key in ("ade", "fde", "p95")
    )
    return {
        "normalized_es": event["normalized_mean"], "raw_es": event["raw_mean"],
        "policy_on_stress": {key: stress_values[key] for key in ("ade_m", "fde_m", "p95_m")},
        "policy_on_stress_pass": bool(stress_pass),
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--updates", type=int); parser.add_argument("--skip-cache", action="store_true")
    args = parser.parse_args()
    config, world = load_config(); require_aligned_factual_protocol(config)
    root = result_root(config); root.mkdir(parents=True, exist_ok=True)
    values = config["ppo"]; device = select_device("auto"); torch.manual_seed(int(values["seed"])); np.random.seed(int(values["seed"]))
    model, _ = load_checkpoint(ROOT / world["paths"]["evaluation_checkpoint"], device=device)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    (root / "manifest.json").write_text(json.dumps({
        "schema": "a2_human_calibration_run", "controller": "a2_human_calibration",
        "frozen": ["Flow", "Diffusion", "HiQR", "HighwayEnv", "legacy_A2"],
        "artifact_sha256": {"world": file_sha256(ROOT / world["paths"]["evaluation_checkpoint"]), "a2": file_sha256(ROOT / config["paths"]["baseline_checkpoint"]), "idm": file_sha256(ROOT / config["paths"]["rule_model"])},
        "rng": {"policy_response_innovations": "world_rng:2", "policy_calibration_innovations": "world_rng:4"},
        "training": config["ppo"],
    }, indent=2) + "\n")
    experiment = prepare_experiment_data(world, ROOT)
    train_arrays, train_plans = plans_for(experiment.bundle, experiment.train_rows, world, root / "cache" / "train", device)
    train_reference = ReactionEventReference.load(ROOT / config["paths"]["event_reference"] / "train")
    train_query = HumanResponseQuerySet.load(root / "human_support" / "train")
    rule = RuleModelBundle.load(ROOT / config["paths"]["rule_model"])
    controller = A2HumanCalibrationController(rule, a2_checkpoint=str(ROOT / config["paths"]["baseline_checkpoint"]), device=device).to(device)
    runtime = policy_config(values)
    environment = ReactionTrainingEnvironment(model, arrays=train_arrays, soft_plans=train_plans, controller=controller, device=device, config=runtime, event_reference=train_reference, deterministic_response=False)
    cache = teacher_cache(environment, train_reference, root / "teacher_forced_response_cache.pt") if not args.skip_cache else torch.load(root / "teacher_forced_response_cache.pt", map_location="cpu", weights_only=False)
    optimizer = torch.optim.Adam(controller.adapter.parameters(), lr=float(values["learning_rate_start"]))
    history = []
    for step in range(int(config["supervised"]["updates"])):
        indices = torch.randint(len(cache["features"]), (int(config["supervised"]["minibatch_size"]),))
        loss = supervised_loss(controller, cache, indices, prior_weight=float(config["supervised"]["prior_regularization"]), non_weight=float(config["supervised"]["non_event_calibration_weight"]))
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step % 20 == 0: history.append({"stage": "supervised", "update": step + 1, "loss": float(loss.detach())})
    torch.save({"schema": "a2_human_calibration", "stage": "supervised", "adapter_state_dict": controller.adapter.state_dict(), "a2_sha256": file_sha256(ROOT / config["paths"]["baseline_checkpoint"])}, root / "supervised.pt")
    subprocess.run([sys.executable, str(ROOT / "hierarchical_world_model/scripts/a2_human_calibration/supervised_gate.py")], cwd=ROOT, check=True)
    reference_adapter = copy.deepcopy(controller.adapter).eval()
    # This population is fixed before PPO and is never reused as a training source.
    validation_arrays, validation_plans = plans_for(experiment.bundle, experiment.validation_rows, world, root / "cache" / "validation", device)
    validation_reference = ReactionEventReference.load(ROOT / config["paths"]["event_reference"] / "validation")
    validation_query = HumanResponseQuerySet.load(root / "human_support" / "validation")
    selected_events = recording_balanced_events(validation_reference, validation_arrays, int(values["quick_validation_events"]))
    factual_rows = np.linspace(0, len(validation_plans) - 1, int(values["quick_policy_on_stress_rows"]), dtype=np.int64)
    from evaluate import policy_on_stress
    from hierarchical_world_model.src.reaction_controller import NoReactionController
    frozen_stress = policy_on_stress(model, {name: value[factual_rows] for name, value in validation_arrays.items()}, validation_plans[factual_rows], NoReactionController().to(device).eval(), runtime, device, chunk=64)
    initial_quick = quick_validation(model, validation_arrays, validation_plans, validation_reference, validation_query, controller, frozen_stress, config, runtime, device, selected_events, factual_rows)
    if not initial_quick["policy_on_stress_pass"]:
        raise RuntimeError("supervised adapter failed its fixed policy-on stress diagnostic")
    coefficient, best, stale = float(values["reference_kl_coefficient"]), float(initial_quick["normalized_es"]), 0
    torch.save({"schema": "a2_human_calibration", "stage": "quick_selected_supervised", "adapter_state_dict": controller.adapter.state_dict(), "a2_sha256": file_sha256(ROOT / config["paths"]["baseline_checkpoint"])}, root / "selected.pt")
    history.append({"stage": "quick_validation", "update": 0, **initial_quick, "selected": True})
    maximum = int(values["maximum_updates"] if args.updates is None else args.updates)
    support_weights = config["human_response"]["support_weight"]
    for update in range(maximum):
        fraction = update / max(1, maximum - 1)
        optimizer.param_groups[0]["lr"] = float(values["learning_rate_start"]) + fraction * (float(values["learning_rate_end"]) - float(values["learning_rate_start"]))
        groups = environment.rng.choice(event_indices(train_reference, train_arrays), size=int(values["event_groups_per_update"]), replace=True)
        buffer = natural_buffer(environment, train_reference, train_query, groups, int(values["futures_per_event"]), runtime, support_weights)
        entry, coefficient = ppo_step(controller, reference_adapter, optimizer, buffer, runtime, values, coefficient)
        anchor_indices = torch.randint(len(cache["features"]), (int(config["supervised"]["minibatch_size"]),))
        anchor = supervised_loss(controller, cache, anchor_indices, prior_weight=0.0, non_weight=float(config["supervised"]["non_event_calibration_weight"]))
        mechanism, mechanism_info = synthetic_mechanism_step(model, controller, train_arrays, train_plans, environment, optimizer, runtime, seed=int(values["seed"]) + update, pairs=int(config["auxiliary"]["synthetic_pairs_per_update"]))
        auxiliary = float(config["auxiliary"]["anchor_weight"]) * anchor + float(config["auxiliary"]["mechanism_weight"]) * mechanism
        optimizer.zero_grad(set_to_none=True); auxiliary.backward(); torch.nn.utils.clip_grad_norm_(controller.adapter.parameters(), runtime.max_grad_norm); optimizer.step()
        entry.update({"anchor_loss": float(anchor.detach()), "mechanism_loss": float(mechanism.detach()), **{f"mechanism_{name}": value for name, value in mechanism_info.items()}})
        entry["update"] = update + 1; history.append(entry)
        if (update + 1) % int(values["quick_validation_interval"]) == 0:
            quick = quick_validation(model, validation_arrays, validation_plans, validation_reference, validation_query, controller, frozen_stress, config, runtime, device, selected_events, factual_rows)
            score = float(quick["normalized_es"])
            entry.update({f"quick_{key}": value for key, value in quick.items()})
            if quick["policy_on_stress_pass"] and score < best * (1.0 - float(values["quick_improvement_fraction"])):
                best, stale = score, 0; torch.save({"schema": "a2_human_calibration", "stage": "quick_selected", "adapter_state_dict": controller.adapter.state_dict(), "a2_sha256": file_sha256(ROOT / config["paths"]["baseline_checkpoint"])}, root / "selected.pt")
            else: stale += 1
            if update + 1 >= int(values["minimum_updates"]) and stale >= int(values["early_stopping_patience"]): break
    (root / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    selected = torch.load(root / "selected.pt", map_location="cpu", weights_only=False)
    torch.save({**selected, "stage": "candidate_selected"}, root / "checkpoint.pt")


if __name__ == "__main__":
    main()

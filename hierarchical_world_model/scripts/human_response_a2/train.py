#!/usr/bin/env python3
"""Train human-response A2 on fresh closed-loop natural-event rollouts.

The only policy updated here is ``HumanResponseA2Controller``.  Flow,
diffusion, HiQR and the legacy A2 checkpoint remain read-only.  Natural
events provide the PPO signal; paired synthetic worlds provide a direct
semi-gradient mechanism regularizer; logged, teacher-forced states provide
the factual action anchor.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from common import ROOT, load_config, result_root

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.human_response_prior import HumanResponsePrior  # noqa: E402
from hierarchical_world_model.src.human_response_training import (  # noqa: E402
    loo_energy_rewards, mechanism_auxiliary_loss,
)
from hierarchical_world_model.src.influence_graph import ROLE_SAME_LANE_FOLLOWER  # noqa: E402
from hierarchical_world_model.src.planner import (  # noqa: E402
    complete_missing_background_plans, frozen_diffusion_plans,
)
from hierarchical_world_model.src.randomness import WorldExogenousState  # noqa: E402
from hierarchical_world_model.src.reaction_controller import HumanResponseA2Controller, NoReactionController  # noqa: E402
from hierarchical_world_model.src.reaction_evidence import EVALUATION_FRAMES, ReactionEventReference  # noqa: E402
from hierarchical_world_model.src.reaction_training import (  # noqa: E402
    PolicyTrainingConfig, ReactionEpisode, ReactionTrainingEnvironment, _gae,
    reaction_controller_rollout, validation_energy_score,
)
from hierarchical_world_model.src.rule_models import RuleModelBundle  # noqa: E402
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from world_model.src.core.utils import file_sha256, save_json, select_device  # noqa: E402


def arrays_for(bundle, rows: np.ndarray) -> dict[str, np.ndarray]:
    fields = ("agent_states", "agent_valid", "map_polylines", "map_polyline_valid")
    result = {field: np.asarray(bundle.arrays[field])[rows] for field in fields}
    result["row_index"] = np.asarray(rows, np.int64)
    return result


def event_rows(reference: ReactionEventReference) -> np.ndarray:
    indices = reference.events.indices(reference.supported_cells)
    indices = indices[reference.events.leader_slot[indices] == 0]
    return np.unique(reference.events.row_index[indices]).astype(np.int64)


def plans_for(experiment, rows: np.ndarray, base: dict, cache: Path, device: torch.device) -> tuple[dict[str, np.ndarray], np.ndarray]:
    arrays = arrays_for(experiment.bundle, rows)
    plans = frozen_diffusion_plans(
        experiment.bundle, rows, checkpoint=ROOT / base["paths"]["diffusion_checkpoint"],
        output_dir=cache, device=device,
        batch_size=int(base["training"]["validation_batch_size"]), ddim_steps=20,
        experiment_scope=base["training"].get("experiment_scope", "full"),
    )
    return arrays, complete_missing_background_plans(plans, arrays["agent_states"], arrays["agent_valid"])


def training_config(values: dict, *, updates: int | None = None) -> PolicyTrainingConfig:
    """Translate just the shared environment fields; PPO has its own loop."""
    result = PolicyTrainingConfig(
        rollout_steps=int(values["rollout_steps"]), episodes_per_rollout=64,
        event_episodes=64, non_event_episodes=0, synthetic_episodes=0,
        epochs_per_update=int(values["ppo_epochs_per_update"]),
        minibatch_size=int(values["minibatch_size"]), updates=int(values["max_ppo_updates"]),
        learning_rate=float(values["learning_rate_start"]), gamma=float(values["gamma"]),
        gae_lambda=float(values["gae_lambda"]), clip_ratio=float(values["clip_ratio"]),
        value_coefficient=float(values["value_coefficient"]),
        entropy_coefficient=float(values["entropy_start"]), max_grad_norm=float(values["max_grad_norm"]),
        validation_interval_updates=int(values["quick_validation_interval"]),
        validation_events=int(values["quick_validation_events"]),
        validation_futures=int(values["quick_validation_futures"]),
        validation_patience=int(values["early_stopping_patience"]),
        validation_minimum_improvement=float(values["quick_improvement_fraction"]),
        seed=int(values["seed"]),
    )
    return replace(result, updates=int(updates)) if updates is not None else result


def payload(controller: HumanResponseA2Controller, config: dict, *, stage: str) -> dict:
    return {
        "schema_name": "human_response_a2_policy", "schema_version": 1,
        "controller_mode": controller.mode, "stage": stage,
        "state_dict": controller.state_dict(), "training": config["training"],
        "frozen_world_model": True,
    }


def copy_a2_hidden(controller: HumanResponseA2Controller, checkpoint: Path, device: torch.device) -> list[str]:
    """Use A2 only as a representation initializer, never as an action head."""
    source = torch.load(checkpoint, map_location=device, weights_only=False)
    state = source.get("state_dict", source)
    target = controller.state_dict()
    copied = []
    for name in ("actor.0.weight", "actor.0.bias", "actor.2.weight", "actor.2.bias"):
        if name in state and name in target and state[name].shape == target[name].shape:
            target[name] = state[name].detach().clone()
            copied.append(name)
    controller.load_state_dict(target)
    return copied


def _follower_slot(reference: ReactionEventReference, event_index: int) -> int:
    return int(reference.events.follower_slot[event_index]) - 1


def _event_key(reference: ReactionEventReference, event_index: int) -> int:
    event = reference.events
    return int(event.absolute_onset_frame[event_index] + 10_000_000 * event.recording_id[event_index])


def natural_rollout(
    environment: ReactionTrainingEnvironment, reference: ReactionEventReference,
    prior: HumanResponsePrior, groups: np.ndarray, futures: int, config: PolicyTrainingConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Collect fresh worlds and attach LOO rewards only to post-onset followers."""
    episodes = []
    for event_index in groups:
        row = environment.row_lookup[int(reference.events.row_index[event_index])]
        episodes.extend(ReactionEpisode(row, "event", int(event_index)) for _ in range(futures))
    environment.reset(episodes)
    names = ("features", "raw_action", "log_prob", "value", "active", "done")
    buffer = {name: [] for name in names}
    finals: list[torch.Tensor] = []
    for _ in range(config.rollout_steps):
        _, _, done, info = environment.step()
        for name in ("features", "raw_action", "log_prob", "value", "active"):
            buffer[name].append(info[name].detach())
        buffer["done"].append(done.detach())
        finals.append(info["final_action"].detach())
    final = torch.stack(finals)
    rewards = torch.zeros_like(final)
    reward_mask = torch.zeros_like(buffer["active"][0], dtype=torch.bool)[None].expand_as(final).clone()
    prior_lookup = {int(key): index for index, key in enumerate(prior.event_key)}
    used, contributions = 0, []
    for group_index, event_index in enumerate(groups):
        start = group_index * futures
        event = reference.events
        onset = int(event.local_onset_frame[event_index]) - 24
        follower = _follower_slot(reference, int(event_index))
        prior_index = prior_lookup.get(_event_key(reference, int(event_index)))
        if prior_index is None or onset < 1 or onset + EVALUATION_FRAMES > config.rollout_steps:
            continue
        neighbors = prior.neighbor_indices(
            prior.descriptor[prior_index], recording_id=int(prior.recording_id[prior_index]),
            leader_id=int(prior.leader_id[prior_index]), follower_id=int(prior.follower_id[prior_index]),
            event_key=int(prior.event_key[prior_index]), neighbors=16,
        )
        if len(neighbors) != 16:
            continue
        acceleration = final[onset:onset + EVALUATION_FRAMES, start:start + futures, follower].transpose(0, 1)
        preceding = final[onset - 1, start:start + futures, follower][:, None]
        jerk = (acceleration - torch.cat((preceding, acceleration[:, :-1]), dim=1)).abs() / .04
        model_response = torch.stack((acceleration, jerk), dim=-1)
        human = torch.as_tensor(prior.response[neighbors], device=final.device)
        score = loo_energy_rewards(model_response, human, torch.as_tensor(prior.response_iqr, device=final.device))
        rewards[onset:onset + EVALUATION_FRAMES, start:start + futures, follower] = score[None] / EVALUATION_FRAMES
        reward_mask[onset:onset + EVALUATION_FRAMES, start:start + futures, follower] = True
        contributions.extend(score.detach().cpu().tolist())
        used += 1
    if not used:
        raise RuntimeError("natural PPO batch had no train-only human references")
    buffer["reward"] = rewards
    buffer["reward_mask"] = reward_mask
    return {name: torch.stack(value) if isinstance(value, list) else value for name, value in buffer.items()}, {
        "event_groups_with_reference": float(used),
        "loo_reward_mean": float(np.mean(contributions)),
    }


def ppo_update(
    controller: HumanResponseA2Controller, optimizer: torch.optim.Optimizer,
    buffer: dict[str, torch.Tensor], config: PolicyTrainingConfig, training: dict,
    update: int,
) -> dict[str, float]:
    rewards, values, dones = buffer["reward"], buffer["value"], buffer["done"]
    advantages, returns = _gae(rewards, values, dones, config.gamma, config.gae_lambda)
    mask = (buffer["reward_mask"] & buffer["active"]).reshape(-1)
    if not mask.any():
        raise RuntimeError("PPO batch contains no active post-onset human-response actions")
    features = buffer["features"].reshape(-1, buffer["features"].shape[-1])[mask]
    raw = buffer["raw_action"].reshape(-1, 3)[mask]
    old_log_prob = buffer["log_prob"].reshape(-1)[mask]
    advantages = advantages.reshape(-1)[mask]
    returns = returns.reshape(-1)[mask]
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1.e-6)
    fraction = min(1.0, (update + 1) / float(training["max_ppo_updates"]))
    learning_rate = float(training["learning_rate_start"]) + fraction * (
        float(training["learning_rate_end"]) - float(training["learning_rate_start"])
    )
    entropy_coefficient = float(training["entropy_start"]) + fraction * (
        float(training["entropy_end"]) - float(training["entropy_start"])
    )
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    losses, kls, clips, norms = [], [], [], []
    executed_epochs = 0
    for _ in range(int(training["ppo_epochs_per_update"])):
        epoch_kls = []
        for indices in torch.randperm(len(features), device=features.device).split(config.minibatch_size):
            log_prob, entropy, estimate = controller.evaluate_raw_action(features[indices], raw[indices])
            log_ratio = log_prob - old_log_prob[indices]
            ratio = log_ratio.exp()
            clipped = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
            policy_loss = -torch.minimum(ratio * advantages[indices], clipped * advantages[indices]).mean()
            value_loss = functional.mse_loss(estimate, returns[indices])
            loss = policy_loss + config.value_coefficient * value_loss - entropy_coefficient * entropy.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), config.max_grad_norm)
            optimizer.step()
            approx_kl = float((-log_ratio).mean().detach())
            epoch_kls.append(approx_kl); kls.append(approx_kl)
            clips.append(float((ratio.sub(1.0).abs() > config.clip_ratio).float().mean().detach()))
            losses.append(float(loss.detach())); norms.append(float(norm))
        executed_epochs += 1
        if epoch_kls and max(epoch_kls) > float(training["inner_epoch_stop_kl"]):
            break
    return {
        "ppo_loss": float(np.mean(losses)), "approx_kl": float(np.mean(kls)),
        "clip_fraction": float(np.mean(clips)), "gradient_norm": float(np.mean(norms)),
        "ppo_epochs_executed": float(executed_epochs), "learning_rate": learning_rate,
        "entropy_coefficient": entropy_coefficient,
    }


def teacher_anchor_loss(
    environment: ReactionTrainingEnvironment, controller: HumanResponseA2Controller,
    reference: ReactionEventReference,
) -> torch.Tensor:
    """One transition from reset is a teacher-forced logged state, never a deviated state."""
    choices = environment.rng.choice(environment.event_pool, size=64, replace=len(environment.event_pool) < 64)
    # The response corpus contains replayable event rows only.  Their anchor
    # history is still an untouched logged state (well before the sampled
    # leader onset), so it is valid teacher-forced factual supervision and
    # does not require inventing a synthetic non-event label.
    rows = np.asarray([
        environment.row_lookup[int(reference.events.row_index[event_index])]
        for event_index in choices
    ], np.int64)
    environment.reset([ReactionEpisode(int(row), "non_event") for row in rows])
    _, _, _, info = environment.step()
    features = info["features"].detach()
    distribution, _ = controller.distribution_and_value(features)
    active = info["active"].detach()
    mapped = controller.map_final_action(
        base=info["base_action"].detach(), rule=info["rule_action"].detach(),
        authority=info["authority"].detach(), active=active, mechanism_allowed=active,
        raw=distribution.mean, previous_correction=torch.zeros_like(info["base_action"]),
        minimum=-8.0, maximum=4.0, dt_s=.04,
    )
    if not active.any():
        return mapped["final"].sum() * 0.0
    return functional.huber_loss(mapped["final"][active], info["target_action"].detach()[active])


def synthetic_mechanism_loss(
    *, model, controller: HumanResponseA2Controller, arrays: dict[str, np.ndarray], plans: np.ndarray,
    device: torch.device, config: PolicyTrainingConfig, seed: int, pairs: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Paired closed-loop states, then direct gradients through *both* action maps."""
    rows = np.arange(len(arrays["agent_states"]))[:pairs]
    # A deterministic rotating subset avoids an uncontrolled data-dependent sampler here.
    rows = (rows + seed % len(rows)) % len(arrays["agent_states"])
    values = {name: value[rows] for name, value in arrays.items()}
    baseline = reaction_controller_rollout(
        model, states=values["agent_states"], valid=values["agent_valid"], soft_plans=plans[rows],
        maps=values["map_polylines"], map_valid=values["map_polyline_valid"], controller=controller,
        device=device, motion_seed=seed, config=config, deterministic_response=False,
    )
    intervention = reaction_controller_rollout(
        model, states=values["agent_states"], valid=values["agent_valid"], soft_plans=plans[rows],
        maps=values["map_polylines"], map_valid=values["map_polyline_valid"], controller=controller,
        device=device, motion_seed=seed, intervention="brake", dose=8.0, intervention_start=25,
        intervention_duration_frames=25, config=config, deterministic_response=False,
    )

    def mapped(rollout):
        diagnostics = rollout.controller_diagnostics
        feature = torch.as_tensor(diagnostics["features"], device=device)
        base = torch.as_tensor(rollout.base_background_actions[..., 0], device=device)
        rule = torch.as_tensor(diagnostics["rule_action_ax"], device=device)
        authority = torch.as_tensor(diagnostics["influence_authority"], device=device)
        role = torch.as_tensor(diagnostics["influence_role"], device=device, dtype=torch.long)
        active = torch.as_tensor(diagnostics["active"], device=device, dtype=torch.bool)
        preceding = torch.cat((torch.zeros_like(base[:, :1]), torch.as_tensor(
            rollout.background_actions[:, :-1, :, 0] - rollout.base_background_actions[:, :-1, :, 0], device=device,
        )), dim=1)
        exogenous = WorldExogenousState.sample(
            len(base), seed=seed, response_steps=149,
            scene_dim=model.cfg.scene_latent_dim, agent_dim=model.cfg.agent_latent_dim,
        )
        noise = np.concatenate((exogenous.policy_response_innovations, exogenous.policy_response_extra_innovations), axis=-1)
        distribution, _ = controller.distribution_and_value(feature)
        raw = distribution.mean + distribution.stddev * torch.as_tensor(noise, device=device)
        output = controller.map_final_action(
            base=base, rule=rule, authority=authority, active=active,
            mechanism_allowed=role.eq(ROLE_SAME_LANE_FOLLOWER), raw=raw,
            previous_correction=preceding, minimum=-8.0, maximum=4.0, dt_s=.04,
        )
        return output["final"], rule, role.eq(ROLE_SAME_LANE_FOLLOWER)

    final_base, idm_base, same_base = mapped(baseline)
    final_intervention, idm_intervention, same_intervention = mapped(intervention)
    loss, values = mechanism_auxiliary_loss(
        model_intervention=final_intervention[same_base & same_intervention],
        model_baseline=final_base[same_base & same_intervention],
        idm_intervention=idm_intervention[same_base & same_intervention],
        idm_baseline=idm_base[same_base & same_intervention],
    )
    return loss, {name: float(value.detach()) for name, value in values.items()}


@torch.no_grad()
def factual_quick(
    model, controller: HumanResponseA2Controller, arrays: dict[str, np.ndarray], plans: np.ndarray,
    device: torch.device, config: PolicyTrainingConfig, seed: int, count: int | None = None,
) -> dict[str, float | bool]:
    """A fixed CRN factual check; formal acceptance still uses full validation."""
    count = min(32 if count is None else int(count), len(arrays["agent_states"]))
    values = {name: value[:count] for name, value in arrays.items()}
    candidate = reaction_controller_rollout(model, states=values["agent_states"], valid=values["agent_valid"], soft_plans=plans[:count], maps=values["map_polylines"], map_valid=values["map_polyline_valid"], controller=controller, device=device, motion_seed=seed, config=config)
    frozen = reaction_controller_rollout(model, states=values["agent_states"], valid=values["agent_valid"], soft_plans=plans[:count], maps=values["map_polylines"], map_valid=values["map_polyline_valid"], controller=NoReactionController(), device=device, motion_seed=seed, config=config)
    target = values["agent_states"][:, 25:174, :, :2]
    valid = values["agent_valid"][:, 25:174]
    def metric(states: np.ndarray) -> tuple[float, float, float]:
        error = np.linalg.norm(states[..., :2] - target, axis=-1)[valid]
        final = np.linalg.norm(states[:, -1, :, :2] - target[:, -1], axis=-1)[valid[:, -1]]
        return float(error.mean()), float(final.mean()), float(np.quantile(error, .95))
    candidate_metrics, frozen_metrics = metric(candidate.states), metric(frozen.states)
    delta = np.asarray(candidate_metrics) - np.asarray(frozen_metrics)
    relative = delta / np.maximum(np.asarray(frozen_metrics), 1.e-6)
    passed = bool(np.all(delta <= np.asarray((.02, .06, .10))) and np.all(relative <= .05))
    return {"ade_delta_m": float(delta[0]), "fde_delta_m": float(delta[1]), "p95_delta_m": float(delta[2]), "passed": passed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, help="bounded execution for smoke validation; default is the configured 500")
    parser.add_argument("--skip-supervised", action="store_true", help="resume only after a stored supervised checkpoint")
    args = parser.parse_args()
    config, base = load_config()
    training = config["training"]
    device = select_device(training["device"])
    torch.manual_seed(int(training["seed"])); np.random.seed(int(training["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(training["seed"]))
    root = result_root(config); root.mkdir(parents=True, exist_ok=True)
    model, _ = load_checkpoint(ROOT / base["paths"]["evaluation_checkpoint"], device=device)
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    experiment = prepare_experiment_data(base, ROOT)
    train_reference = ReactionEventReference.load(ROOT / config["paths"]["event_reference"] / "train")
    validation_reference = ReactionEventReference.load(ROOT / config["paths"]["event_reference"] / "validation")
    train_arrays, train_plans = plans_for(experiment, event_rows(train_reference), base, root / "cache" / "train", device)
    validation_arrays, validation_plans = plans_for(experiment, event_rows(validation_reference), base, root / "cache" / "validation", device)
    train_prior = HumanResponsePrior.load(root / "human_reference" / "train")
    rule = RuleModelBundle.load(ROOT / config["paths"]["rule_model"])
    run_config = training_config(training, updates=args.updates)
    controller = HumanResponseA2Controller(rule).to(device)
    copied = copy_a2_hidden(controller, ROOT / config["paths"]["baseline_checkpoint"], device)
    environment = ReactionTrainingEnvironment(model, arrays=train_arrays, soft_plans=train_plans, controller=controller, device=device, config=run_config, event_reference=train_reference)
    optimizer = torch.optim.Adam(controller.parameters(), lr=float(training["learning_rate_start"]))
    manifest = {
        "schema_name": "human_response_a2_run", "schema_version": 1,
        "controller": controller.mode, "frozen": ["flow", "diffusion", "HiQR", "HighwayEnv"],
        "baseline_a2": str(ROOT / config["paths"]["baseline_checkpoint"]),
        "artifact_sha256": {key: file_sha256(ROOT / path) for key, path in {
            "world": base["paths"]["evaluation_checkpoint"], "baseline_a2": config["paths"]["baseline_checkpoint"], "idm": config["paths"]["rule_model"],
        }.items()},
        "training": training, "a2_hidden_initialization": copied,
        "synthetic": "detached on-policy visited-state auxiliary semi-gradient; not PPO reward",
        "anchor": "teacher-forced logged initial histories only",
    }
    save_json(manifest, root / "manifest.json")
    supervised_path, progress_path = root / "supervised.pt", root / "training_progress.pt"
    history: list[dict] = []
    start, best_score, best_state, stale = 0, float("inf"), None, 0
    if progress_path.exists():
        state = torch.load(progress_path, map_location=device, weights_only=False)
        controller.load_state_dict(state["state_dict"]); optimizer.load_state_dict(state["optimizer_state"])
        start, best_score, best_state, stale = int(state["next_update"]), float(state["best_score"]), state["best_state"], int(state["stale"])
        history = list(state["history"])
    elif supervised_path.exists():
        controller.load_state_dict(torch.load(supervised_path, map_location=device, weights_only=False)["state_dict"])
    elif not args.skip_supervised:
        supervised_history = []
        controller.train()
        for index in range(int(training["pretrain_updates"])):
            loss = teacher_anchor_loss(environment, controller, train_reference)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), run_config.max_grad_norm); optimizer.step()
            if index % 20 == 0 or index + 1 == int(training["pretrain_updates"]):
                supervised_history.append({"update": index + 1, "anchor_loss": float(loss.detach()), "gradient_norm": float(norm)})
        torch.save(payload(controller, config, stage="supervised"), supervised_path)
        quick = factual_quick(model, controller, validation_arrays, validation_plans, device, run_config, int(training["seed"]))
        save_json({"supervised_updates": int(training["pretrain_updates"]), "history": supervised_history, "quick_factual": quick, "passed": bool(quick["passed"])}, root / "supervised_gate.json")
        if not quick["passed"]:
            raise RuntimeError("supervised factual quick gate failed; PPO was not started")
    else:
        raise FileNotFoundError("--skip-supervised requires human_response_a2/supervised.pt")

    for update in range(start, run_config.updates):
        controller.train()
        groups = environment.rng.choice(environment.event_pool, size=int(training["event_groups_per_update"]), replace=len(environment.event_pool) < int(training["event_groups_per_update"]))
        buffer, natural = natural_rollout(environment, train_reference, train_prior, groups, int(training["futures_per_event"]), run_config)
        entry = {"update": update + 1, **natural, **ppo_update(controller, optimizer, buffer, run_config, training, update)}
        mechanism, mechanism_info = synthetic_mechanism_loss(model=model, controller=controller, arrays=train_arrays, plans=train_plans, device=device, config=run_config, seed=int(training["seed"]) + update, pairs=int(training["synthetic_pairs_per_update"]))
        anchor = teacher_anchor_loss(environment, controller, train_reference)
        auxiliary = float(training["mechanism_weight"]) * mechanism + float(training["anchor_weight"]) * anchor
        optimizer.zero_grad(set_to_none=True); auxiliary.backward()
        norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), run_config.max_grad_norm); optimizer.step()
        entry.update({"mechanism_loss": float(mechanism.detach()), "anchor_loss": float(anchor.detach()), "auxiliary_gradient_norm": float(norm), **{f"mechanism_{key}": value for key, value in mechanism_info.items()}})
        if not np.isfinite(list(entry.values())).all():
            raise FloatingPointError("non-finite training statistic")
        should_validate = (update + 1) % int(training["quick_validation_interval"]) == 0 or update + 1 == run_config.updates
        if should_validate:
            controller.eval()
            score = validation_energy_score(model, controller, arrays=validation_arrays, soft_plans=validation_plans, reference=validation_reference, device=device, config=run_config)
            factual = factual_quick(model, controller, validation_arrays, validation_plans, device, run_config, int(training["seed"]) + update)
            entry.update({"quick_energy_score": score, **{f"quick_factual_{key}": value for key, value in factual.items()}})
            if factual["passed"] and score < best_score * (1.0 - float(training["quick_improvement_fraction"])):
                best_score, best_state, stale = score, copy.deepcopy(controller.state_dict()), 0
                torch.save(payload(controller, config, stage="quick_selected"), root / "selected.pt")
            else:
                stale += 1
        history.append(entry)
        torch.save({"schema_name": "human_response_a2_progress", "next_update": update + 1, "state_dict": controller.state_dict(), "optimizer_state": optimizer.state_dict(), "history": history, "best_score": best_score, "best_state": best_state, "stale": stale}, progress_path)
        (root / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(entry), flush=True)
        if should_validate and update + 1 >= int(training["minimum_ppo_updates"]) and stale >= int(training["early_stopping_patience"]):
            break
    if best_state is None:
        torch.save(payload(controller, config, stage="rejected_no_quick_gate"), root / "rejected.pt")
        raise RuntimeError("no PPO checkpoint passed the quick factual gate")
    controller.load_state_dict(best_state)
    torch.save(payload(controller, config, stage="selected_for_full_validation"), root / "checkpoint.pt")
    save_json({"status": "pending_full_validation", "checkpoint": str(root / "checkpoint.pt"), "best_quick_energy_score": best_score, "updates_completed": len(history), "test_not_run": True}, root / "decision.json")


if __name__ == "__main__":
    main()

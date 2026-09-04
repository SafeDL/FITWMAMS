#!/usr/bin/env python3
"""Train the frozen longitudinal highD GAIL human-action prior."""

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as functional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_world_model.src.data import ego_controls, prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.human_prior import (  # noqa: E402
    HumanPriorConfig,
    build_human_expert_samples,
    human_prior_features,
    train_human_prior,
)
from hierarchical_world_model.src.highway import HighwayEnvClosedLoopWorld  # noqa: E402
from hierarchical_world_model.src.influence_graph import dynamic_candidate_scene_mask  # noqa: E402
from hierarchical_world_model.src.planner import complete_missing_background_plans, frozen_diffusion_plans  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.randomness import WorldExogenousState  # noqa: E402
from hierarchical_world_model.src.reaction_controller import HumanPriorNominalController  # noqa: E402
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from world_model.src.core.utils import ensure_dir, file_sha256, save_json, select_device  # noqa: E402


DEFAULT = ROOT / "hierarchical_world_model/config/reaction_naturalistic.yaml"


def _drop_mapped_pages(*arrays: np.ndarray) -> None:
    """Ask the kernel to evict scanned memmap pages between formal phases."""
    for array in arrays:
        mapping = getattr(array, "_mmap", None)
        if mapping is not None and hasattr(mapping, "madvise"):
            try:
                mapping.madvise(4)  # Linux MADV_DONTNEED; best-effort only.
            except (BufferError, OSError, ValueError):
                pass


def _anchor_relations(states: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose one causal parent/child relation per HighwayEnv reset."""
    keep, slots, parents, roles = [], [], [], []
    frame = ANCHOR_INDEX
    for sequence in range(len(states)):
        candidates: list[tuple[float, int, int]] = []
        for child in range(1, 7):
            if not valid[sequence, frame, child]:
                continue
            child_state = states[sequence, frame, child]
            for parent in range(7):
                if parent == child or not valid[sequence, frame, parent]:
                    continue
                parent_state = states[sequence, frame, parent]
                gap = float(parent_state[0] - child_state[0] - 4.8)
                lateral = float(parent_state[1] - child_state[1])
                future_lateral = lateral + float(parent_state[3] - child_state[3]) * 1.5
                following = .1 < gap < 60. and abs(lateral) < 1.8
                cutin = .1 < gap < 35. and abs(lateral) >= 1.8 and min(abs(lateral), abs(future_lateral)) < 1.8
                if following or cutin:
                    closing = float(child_state[2] - parent_state[2])
                    ttc = gap / closing if closing > 1.e-4 else 10.
                    candidates.append((min(ttc, 10.) + (2. if cutin else 0.), child, parent))
        if candidates:
            _, child, parent = min(candidates)
            lateral = abs(float(states[sequence, frame, parent, 1] - states[sequence, frame, child, 1]))
            role = 2 if lateral >= 1.8 else 1 if parent == 0 else 4
            keep.append(sequence); slots.append(child - 1); parents.append(parent); roles.append(role)
    return (np.asarray(keep, np.int64), np.asarray(slots, np.int64),
            np.asarray(parents, np.int64), np.asarray(roles, np.int64))


def _anchor_relations_indexed(
    arrays: dict[str, np.ndarray], rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find nominal HighwayEnv parent/child pairs without copying a split.

    The highD bundle is memory mapped.  Materialising ``agent_states[rows]``
    for the complete training split costs several GB and previously made the
    formal GAIL pass get killed by the OS.  Relation selection only needs one
    recording at a time, so keep the bundle indexed and return row ids.
    """
    states_all, valid_all = arrays["agent_states"], arrays["agent_valid"]
    keep, slots, parents, roles = [], [], [], []
    frame = ANCHOR_INDEX
    for row in np.asarray(rows, np.int64):
        states, valid = states_all[int(row)], valid_all[int(row)]
        candidates: list[tuple[float, int, int]] = []
        for child in range(1, 7):
            # The highD slot assignment may begin inside the one-second
            # history window; those missing history frames are causally zero
            # padded by human_prior_features.  The *future* expert action
            # sequence, however, must be continuously observed for 2 s.
            if not valid[frame:frame + 51, child].all():
                continue
            child_state = states[frame, child]
            for parent in range(7):
                if parent == child or not valid[frame:frame + 50, parent].all():
                    continue
                parent_state = states[frame, parent]
                gap = float(parent_state[0] - child_state[0] - 4.8)
                lateral = float(parent_state[1] - child_state[1])
                future_lateral = lateral + float(parent_state[3] - child_state[3]) * 1.5
                following = .1 < gap < 60. and abs(lateral) < 1.8
                cutin = .1 < gap < 35. and abs(lateral) >= 1.8 and min(abs(lateral), abs(future_lateral)) < 1.8
                if following or cutin:
                    closing = float(child_state[2] - parent_state[2])
                    ttc = gap / closing if closing > 1.e-4 else 10.
                    candidates.append((min(ttc, 10.) + (2. if cutin else 0.), child, parent))
        if candidates:
            _, child, parent = min(candidates)
            lateral = abs(float(states[frame, parent, 1] - states[frame, child, 1]))
            role = 2 if lateral >= 1.8 else 1 if parent == 0 else 4
            keep.append(int(row)); slots.append(child - 1); parents.append(parent); roles.append(role)
    return (np.asarray(keep, np.int64), np.asarray(slots, np.int64),
            np.asarray(parents, np.int64), np.asarray(roles, np.int64))


def _paired_source_expert_sequence_batch(
    arrays: dict[str, np.ndarray], rows: np.ndarray, slots: np.ndarray,
    parents: np.ndarray, roles: np.ndarray, *, frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HighD expert trajectory paired to each HighwayEnv reset relation."""
    features, actions = [], []
    for row, slot, parent, role in zip(rows.tolist(), slots.tolist(), parents.tolist(), roles.tolist()):
        states = arrays["agent_states"][int(row)]
        child = int(slot) + 1
        sequence_x, sequence_a = [], []
        for frame in range(ANCHOR_INDEX, ANCHOR_INDEX + int(frames)):
            history = torch.from_numpy(states[None, frame - 24:frame + 1].astype(np.float32))
            parent_speed = np.linalg.norm(states[frame - 5:frame + 1, int(parent), 2:4], axis=-1)
            controls = torch.zeros(1, 5, 2)
            controls[0, :, 0] = torch.from_numpy(np.diff(parent_speed).astype(np.float32) / .04)
            sequence_x.append(human_prior_features(
                history, controls, target_slot_index=int(slot),
                parent_index=torch.tensor((int(parent),), dtype=torch.long),
                role=torch.tensor((int(role),), dtype=torch.long),
            )[0])
            child_speed = np.linalg.norm(states[frame:frame + 2, child, 2:4], axis=-1)
            sequence_a.append(float(np.clip((child_speed[1] - child_speed[0]) / .04, -8., 4.)))
        features.append(torch.stack(sequence_x))
        actions.append(torch.tensor(sequence_a, dtype=torch.float32))
    return torch.stack(features).to(device), torch.stack(actions).to(device)


def _complete_plans_indexed(
    plans: np.ndarray, arrays: dict[str, np.ndarray], rows: np.ndarray,
    *, batch_size: int = 256,
) -> np.ndarray:
    """Fill masked diffusion slots in bounded chunks against the memmap."""
    result = np.asarray(plans, np.float32).copy()
    rows = np.asarray(rows, np.int64)
    if len(result) != len(rows):
        raise ValueError("plan and row counts differ")
    for start in range(0, len(rows), max(int(batch_size), 1)):
        stop = min(start + max(int(batch_size), 1), len(rows))
        source = rows[start:stop]
        result[start:stop] = complete_missing_background_plans(
            result[start:stop], arrays["agent_states"][source], arrays["agent_valid"][source]
        )
    return result


def _expert_sequence_batch(
    arrays: dict[str, np.ndarray], metadata: dict[str, np.ndarray],
    indices: np.ndarray, *, frames: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialise true contiguous highD state/action sequences on demand."""
    states_all, valid_all = arrays["agent_states"], arrays["agent_valid"]
    sequence_features, sequence_actions = [], []
    for index in indices.tolist():
        row = int(metadata["row"][index])
        end = int(metadata["frame"][index])
        child = int(metadata["child"][index])
        parent = int(metadata["parent"][index])
        start = end - int(frames) + 1
        states, valid = states_all[row], valid_all[row]
        if start < 24 or end + 1 >= len(states) or not (
            valid[start - 24:end + 2, child].all()
            and valid[start - 24:end + 1, parent].all()
        ):
            raise ValueError("expert sequence index is not continuously valid")
        features, actions = [], []
        for frame in range(start, end + 1):
            history = torch.from_numpy(states[None, frame - 24:frame + 1].astype(np.float32))
            controls = torch.zeros(1, 5, 2)
            parent_speed = np.linalg.norm(states[frame - 5:frame + 1, parent, 2:4], axis=-1)
            controls[0, :, 0] = torch.from_numpy(np.diff(parent_speed).astype(np.float32) / .04)
            features.append(human_prior_features(
                history, controls, target_slot_index=child - 1,
                parent_index=torch.tensor((parent,), dtype=torch.long),
                role=torch.tensor((int(metadata["role"][index]),), dtype=torch.long),
            )[0])
            child_speed = np.linalg.norm(states[frame:frame + 2, child, 2:4], axis=-1)
            actions.append(float(np.clip((child_speed[1] - child_speed[0]) / .04, -8., 4.)))
        sequence_features.append(torch.stack(features))
        sequence_actions.append(torch.tensor(actions, dtype=torch.float32))
    return torch.stack(sequence_features).to(device), torch.stack(sequence_actions).to(device)


def _auc(expert_score: np.ndarray, generated_score: np.ndarray) -> float:
    values = np.concatenate((expert_score, generated_score))
    order = np.argsort(np.argsort(values)) + 1
    n0, n1 = len(expert_score), len(generated_score)
    return float((order[n0:].sum() - n1 * (n1 + 1) / 2) / max(n0 * n1, 1))


def _label_shuffle_auc(expert_score: np.ndarray, generated_score: np.ndarray, rng: np.random.Generator) -> float:
    """Sanity AUC after destroying expert/generated labels (should be ~0.5)."""
    scores = np.concatenate((expert_score, generated_score)).reshape(-1)
    labels = np.concatenate((np.zeros(len(expert_score), np.int8), np.ones(len(generated_score), np.int8)))
    rng.shuffle(labels)
    ranks = np.argsort(np.argsort(scores)) + 1
    positive = labels.astype(bool)
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n0 == 0 or n1 == 0:
        return 0.5
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def _w1(left: np.ndarray, right: np.ndarray) -> float:
    count = min(len(left), len(right))
    if count == 0:
        return float("inf")
    q = np.linspace(0., 1., count)
    return float(np.abs(np.quantile(left, q) - np.quantile(right, q)).mean())


def _gae(
    reward: torch.Tensor,
    value: torch.Tensor,
    done: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite-horizon GAE for one batched GAIL rollout."""
    advantage = torch.zeros_like(reward)
    carry = torch.zeros(reward.shape[0], dtype=reward.dtype, device=reward.device)
    next_value = torch.zeros_like(carry)
    for step in range(reward.shape[1] - 1, -1, -1):
        alive = (~done[:, step]).to(reward.dtype)
        delta = reward[:, step] + float(gamma) * next_value * alive - value[:, step]
        carry = delta + float(gamma * gae_lambda) * alive * carry
        advantage[:, step] = carry
        next_value = value[:, step]
    return advantage, advantage + value


def _highway_gail_refine(
    prior, *, model, arrays, source_rows, plans, target_slots, parent_indices, target_roles,
    expert_arrays, expert_features, expert_actions, expert_metadata,
    validation_features, validation_actions, config, device,
    progress_path: Path | None = None,
    evidence_path: Path | None = None,
    artifact_metadata: dict | None = None,
):
    """PPO generator refinement from 2-second nominal HighwayEnv trajectories.

    Each discriminator input already contains the previous 25 realised state
    frames; rollout transitions add a generated terminal action. This keeps
    the discriminator causal and provides the requested two-second action
    sequence, rather than the former 16-frame pair-only refinement.
    """
    actor_parameters = [
        *prior.actor.parameters(), *prior.mean.parameters(), *prior.log_std.parameters(),
    ]
    critic_parameters = list(prior.value.parameters())
    actor_optimizer = torch.optim.Adam(actor_parameters, lr=config.actor_learning_rate)
    critic_optimizer = torch.optim.Adam(critic_parameters, lr=config.critic_learning_rate)
    discriminator_optimizer = torch.optim.Adam(
        [*prior.discriminator.parameters(), *prior.discriminator_head.parameters()],
        lr=config.discriminator_learning_rate,
    )
    rng, history = np.random.default_rng(config.seed), []
    sequence_frames = int(config.trajectory_frames)
    # Generated scenes are interleaved by role/TTC. Their expert sequences
    # are the exact paired highD relation from the same reset below, so the
    # discriminator cannot exploit a mismatched context population.
    anchor = arrays["agent_states"][source_rows, ANCHOR_INDEX]
    batch_index = np.arange(len(source_rows))
    child = target_slots + 1
    parent_state = anchor[batch_index, parent_indices]
    child_state = anchor[batch_index, child]
    anchor_gap = parent_state[:, 0] - child_state[:, 0] - 4.8
    anchor_closing = child_state[:, 2] - parent_state[:, 2]
    anchor_ttc = np.where(
        (anchor_gap > 0.) & (anchor_closing > 1.e-4),
        anchor_gap / np.maximum(anchor_closing, 1.e-4), 10.,
    )
    generated_bucket = target_roles * 3 + np.digitize(anchor_ttc, (2.0, 4.0), right=False)

    def stratified_scene_order() -> np.ndarray:
        groups = [np.flatnonzero(generated_bucket == key) for key in np.unique(generated_bucket)]
        for group in groups:
            rng.shuffle(group)
        ordered, cursor = [], [0] * len(groups)
        while True:
            emitted = False
            for index, group in enumerate(groups):
                if cursor[index] < len(group):
                    ordered.append(int(group[cursor[index]]))
                    cursor[index] += 1
                    emitted = True
            if not emitted:
                return np.asarray(ordered, np.int64)
    best_state, best_score, stale, start_pass = None, float("inf"), 0, 0
    bc_rollout_action: np.ndarray | None = None
    bc_rollout_jerk: np.ndarray | None = None
    if evidence_path is not None and evidence_path.exists():
        with np.load(evidence_path) as retained:
            bc_rollout_action = retained["bc_action_mps2"].copy()
            bc_rollout_jerk = retained["bc_jerk_mps3"].copy()
    batch_size = max(1, int(config.highway_rollout_batch))
    validation_x = torch.from_numpy(validation_features).to(device)
    validation_a = torch.from_numpy(validation_actions).to(device)
    with torch.no_grad():
        bc_validation_nll = float(
            -prior.action_log_prob(validation_x, validation_a).mean().cpu()
        )
    # NLL can legitimately be negative for a bounded density.  A ratio such
    # as ``1.05 * nll`` reverses the inequality in that case, rejecting every
    # update.  Use a symmetric 5% absolute tolerance around the BC anchor.
    maximum_validation_nll = bc_validation_nll + .05 * max(abs(bc_validation_nll), 1.0)
    if progress_path is not None and progress_path.exists():
        progress = torch.load(progress_path, map_location=device, weights_only=False)
        if (
            progress.get("schema") == "longitudinal_gail_highway_progress_v4"
            and progress.get("artifact_metadata") == (artifact_metadata or {})
        ):
            prior.load_state_dict(progress["state_dict"])
            actor_optimizer.load_state_dict(progress["actor_optimizer"])
            critic_optimizer.load_state_dict(progress["critic_optimizer"])
            discriminator_optimizer.load_state_dict(progress["discriminator_optimizer"])
            rng.bit_generator.state = progress["rng_state"]
            history = list(progress["history"])
            best_state = progress.get("best_state")
            best_score = float(progress.get("best_score", float("inf")))
            stale = int(progress.get("stale", 0))
            start_pass = int(progress.get("next_pass", 0))
            # A complete pass with zero admissible actor updates means the
            # fixed 5% held-out NLL trust region has converged. Repeating the
            # unchanged generator only changes Monte-Carlo W1 estimates and
            # wastes full HighwayEnv passes.
            if (
                len(history) >= 2
                and float(history[-1].get("actor_update_accepted", 1.0)) == 0.0
            ):
                start_pass = int(config.gail_epochs)
    for pass_index in range(start_pass, int(config.gail_epochs)):
        order = stratified_scene_order()
        metric_names = (
            "policy_loss", "value_loss", "bc_anchor",
            "discriminator_loss", "generator_reward", "actor_update_accepted",
        )
        pass_metrics: dict[str, list[float]] = {name: [] for name in metric_names}
        generated_sample: list[np.ndarray] = []
        generated_jerk: list[np.ndarray] = []
        expert_sample: list[np.ndarray] = []
        expert_jerk: list[np.ndarray] = []
        expert_score: list[np.ndarray] = []
        generated_score: list[np.ndarray] = []
        collision_sequences = 0
        for start in range(0, len(order), batch_size):
            pick = order[start:start + batch_size]
            count = len(pick)
            rows = source_rows[pick]
            states = arrays["agent_states"][rows]
            valid = arrays["agent_valid"][rows]
            current = states[:, ANCHOR_INDEX]
            past = states[:, ANCHOR_INDEX - 24:ANCHOR_INDEX + 1]
            logged = ego_controls(
                states[:, ANCHOR_INDEX:173, 0],
                states[:, ANCHOR_INDEX + 1:174, 0], .04,
            )
            committed = ego_controls(
                states[:, :ANCHOR_INDEX, 0],
                states[:, 1:ANCHOR_INDEX + 1, 0], .04,
            )
            slots = torch.from_numpy(target_slots[pick]).to(device)
            parents = torch.from_numpy(parent_indices[pick]).to(device)
            roles = torch.from_numpy(target_roles[pick]).to(device)
            controller = HumanPriorNominalController(
                prior, target_slot_index=slots, parent_index=parents, role=roles,
            ).to(device)
            world = HighwayEnvClosedLoopWorld(model, device=device, controller=controller)
            world.reset(
                torch.from_numpy(current),
                torch.from_numpy(valid[:, ANCHOR_INDEX]),
                torch.from_numpy(plans[pick]),
                torch.from_numpy(arrays["map_polylines"][rows]),
                torch.from_numpy(arrays["map_polyline_valid"][rows]),
                initial_history=torch.from_numpy(past),
                initial_history_valid=torch.from_numpy(
                    valid[:, ANCHOR_INDEX - 24:ANCHOR_INDEX + 1]
                ),
                committed_ego_controls=torch.from_numpy(committed),
                deterministic_response=False,
                exogenous_state=WorldExogenousState.sample(
                    count,
                    seed=config.seed + pass_index * max(len(order), 1) + start,
                    response_steps=149,
                    scene_dim=model.cfg.scene_latent_dim,
                    agent_dim=model.cfg.agent_latent_dim,
                ),
            )
            generated_x, generated_a, generated_raw = [], [], []
            old_logp, values, done = [], [], []
            batch = torch.arange(count, device=device)
            for step in range(int(config.trajectory_frames)):
                transition = world.advance_response(
                    torch.from_numpy(logged[:, step]).to(device)
                )
                generated_x.append(transition["controller_features"])
                generated_a.append(
                    transition["background_actions"][batch, 0, slots, 0]
                )
                generated_raw.append(transition["controller_raw_action"])
                old_logp.append(transition["controller_log_prob"])
                values.append(transition["controller_value"])
                done.append(transition["crashed"].any(1))
            sequence_x = torch.stack(generated_x, dim=1)
            sequence_action = torch.stack(generated_a, dim=1)
            x = sequence_x.reshape(-1, sequence_x.shape[-1])
            raw = torch.stack(generated_raw, dim=1).reshape(-1)
            old = torch.stack(old_logp, dim=1).reshape(-1)
            value_sequence = torch.stack(values, dim=1)
            done_sequence = torch.stack(done, dim=1)
            ex, ea = _paired_source_expert_sequence_batch(
                expert_arrays, source_rows[pick], target_slots[pick],
                parent_indices[pick], target_roles[pick],
                frames=sequence_frames, device=device,
            )
            fake_x = sequence_x.detach()
            fake_a = sequence_action.detach()
            expert_step_logits = prior.discriminator_step_logits(ex, ea)
            fake_step_logits = prior.discriminator_step_logits(fake_x, fake_a)
            discriminator_loss = (
                functional.binary_cross_entropy_with_logits(
                    expert_step_logits, torch.zeros_like(expert_step_logits)
                )
                + functional.binary_cross_entropy_with_logits(
                    fake_step_logits, torch.ones_like(fake_step_logits)
                )
            )
            discriminator_optimizer.zero_grad(set_to_none=True)
            discriminator_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*prior.discriminator.parameters(), *prior.discriminator_head.parameters()], 1.0
            )
            discriminator_optimizer.step()
            with torch.no_grad():
                generated_step_logits = prior.discriminator_step_logits(fake_x, fake_a)
                reward_sequence = -functional.logsigmoid(generated_step_logits)
                # Adversarial reward is per tick, but a memoryless action
                # prior can still exploit it by matching acceleration while
                # alternating at 25 Hz.  Use the paired expert sequence as a
                # small temporal naturalness signal.  This is not an IDM
                # target and does not expose any future ego command: both
                # jerk terms are computed from already generated/recorded
                # longitudinal actions within the same two-second window.
                generated_jerk_sequence = torch.zeros_like(sequence_action)
                expert_jerk_sequence = torch.zeros_like(ea)
                generated_jerk_sequence[:, 1:] = (
                    sequence_action[:, 1:] - sequence_action[:, :-1]
                ).abs() / .04
                expert_jerk_sequence[:, 1:] = (
                    ea[:, 1:] - ea[:, :-1]
                ).abs() / .04
                jerk_error = (
                    generated_jerk_sequence - expert_jerk_sequence
                ).abs() / 20.0
                reward_sequence = reward_sequence - float(config.gail_jerk_weight) * jerk_error
                advantage_sequence, return_sequence = _gae(
                    reward_sequence,
                    value_sequence.detach(),
                    done_sequence,
                    gamma=config.gamma,
                    gae_lambda=config.gae_lambda,
                )
            advantage = advantage_sequence.reshape(-1)
            advantage = (
                advantage - advantage.mean()
            ) / advantage.std().clamp_min(1.e-5)
            target_return = return_sequence.reshape(-1).detach()
            bc_weight = float(config.gail_bc_initial_weight) * max(
                0.0, 1.0 - pass_index / float(max(config.gail_bc_decay_passes, 1))
            )
            policy_values, value_losses, bc_values = [], [], []
            actor_state_before = {
                name: value.detach().clone()
                for name, value in prior.state_dict().items()
                if name.startswith(("actor.", "mean.", "log_std."))
            }
            actor_optimizer_before = copy.deepcopy(actor_optimizer.state_dict())
            for _ in range(int(config.ppo_epochs)):
                new_logp = prior.distribution(x).log_prob(raw)
                ratio = (new_logp - old.detach()).exp()
                policy = -torch.minimum(
                    ratio * advantage,
                    ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantage,
                ).mean()
                bc_anchor = -prior.action_log_prob(
                    ex.reshape(-1, ex.shape[-1]), ea.reshape(-1)
                ).mean()
                actor_loss = policy + bc_weight * bc_anchor
                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_parameters, 1.0)
                actor_optimizer.step()

                value_loss = functional.mse_loss(
                    prior.value(x).squeeze(-1), target_return
                )
                critic_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic_parameters, 1.0)
                critic_optimizer.step()
                policy_values.append(float(policy.detach()))
                value_losses.append(float(value_loss.detach()))
                bc_values.append(float(bc_anchor.detach()))
            with torch.no_grad():
                post_update_validation_nll = float(
                    -prior.action_log_prob(validation_x, validation_a).mean().cpu()
                )
            actor_update_accepted = post_update_validation_nll <= maximum_validation_nll
            if not actor_update_accepted:
                current = prior.state_dict()
                current.update(actor_state_before)
                prior.load_state_dict(current)
                actor_optimizer.load_state_dict(actor_optimizer_before)
            batch_metrics = {
                "policy_loss": float(np.mean(policy_values)),
                "value_loss": float(np.mean(value_losses)),
                "bc_anchor": float(np.mean(bc_values)),
                "discriminator_loss": float(discriminator_loss.detach()),
                "generator_reward": float(reward_sequence.mean().detach()),
                "actor_update_accepted": float(actor_update_accepted),
            }
            for name, metric in batch_metrics.items():
                pass_metrics[name].append(metric)
            generated_np = sequence_action.detach().cpu().numpy()
            expert_np = ea.detach().cpu().numpy()
            if bc_rollout_action is None:
                bc_rollout_action = generated_np.reshape(-1).copy()
                bc_rollout_jerk = (np.abs(np.diff(generated_np, axis=1)) / .04).reshape(-1).copy()
            retained = sum(len(value) for value in generated_sample)
            remaining = max(0, 20000 - retained)
            if remaining:
                generated_sample.append(generated_np.reshape(-1)[:remaining])
                expert_sample.append(expert_np.reshape(-1)[:remaining])
                generated_jerk.append(
                    (np.abs(np.diff(generated_np, axis=1)) / .04).reshape(-1)[:remaining]
                )
                expert_jerk.append(
                    (np.abs(np.diff(expert_np, axis=1)) / .04).reshape(-1)[:remaining]
                )
            with torch.no_grad():
                expert_score.append(prior.discriminator_logits(ex, ea).cpu().numpy())
                generated_score.append(prior.discriminator_logits(fake_x, fake_a).cpu().numpy())
            collision_sequences += int(transition["crashed"].any(1).sum().cpu())
        with torch.no_grad():
            validation_nll = float(
                -prior.action_log_prob(validation_x, validation_a).mean().cpu()
            )
        action_w1 = _w1(
            np.concatenate(generated_sample), np.concatenate(expert_sample)
        )
        jerk_w1 = _w1(
            np.concatenate(generated_jerk), np.concatenate(expert_jerk)
        )
        discriminator_auc = _auc(
            np.concatenate(expert_score), np.concatenate(generated_score)
        )
        label_shuffle_auc = _label_shuffle_auc(
            np.concatenate(expert_score), np.concatenate(generated_score), rng
        )
        violation_rate = float(collision_sequences / max(len(plans), 1))
        # Joint selection prevents a deceptively low BC NLL from winning
        # after the discriminator has separated the two occupancies.  AUC is
        # symmetric around chance because either label convention is valid.
        selection_score = (
            validation_nll + action_w1 + .01 * jerk_w1
            + abs(discriminator_auc - .5) + 10. * violation_rate
        )
        entry = {
            "pass": int(pass_index), "scenes": int(len(plans)),
            "trajectory_frames": int(config.trajectory_frames),
            **{
                name: float(np.mean(metric_values))
                for name, metric_values in pass_metrics.items()
            },
            "validation_nll": validation_nll,
            "bc_validation_nll": bc_validation_nll,
            "maximum_validation_nll": maximum_validation_nll,
            "acceleration_w1_mps2": action_w1,
            "jerk_w1_mps3": jerk_w1,
            "discriminator_auc": discriminator_auc,
            "label_shuffle_auc": label_shuffle_auc,
            "closed_loop_violation_rate": violation_rate,
            "selection_score": selection_score,
        }
        history.append(entry)
        if selection_score < best_score - 1.e-5:
            best_score, stale = selection_score, 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in prior.state_dict().items()
            }
            if evidence_path is not None:
                np.savez_compressed(
                    evidence_path,
                    highd_action_mps2=np.concatenate(expert_sample),
                    highd_jerk_mps3=np.concatenate(expert_jerk),
                    bc_action_mps2=bc_rollout_action,
                    bc_jerk_mps3=bc_rollout_jerk,
                    gail_action_mps2=np.concatenate(generated_sample),
                    gail_jerk_mps3=np.concatenate(generated_jerk),
                    selected_pass=np.asarray(pass_index, np.int64),
                )
        else:
            stale += 1
        if progress_path is not None:
            torch.save({
                "schema": "longitudinal_gail_highway_progress_v4",
                "next_pass": int(pass_index + 1),
                "state_dict": prior.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "discriminator_optimizer": discriminator_optimizer.state_dict(),
                "rng_state": rng.bit_generator.state,
                "history": history, "best_state": best_state,
                "best_score": best_score, "stale": stale,
                "artifact_metadata": artifact_metadata or {},
            }, progress_path)
        if stale >= int(config.patience):
            break
    if best_state is not None:
        prior.load_state_dict(best_state)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BC-initialized longitudinal GAIL human prior.")
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bc-epochs", type=int, default=None)
    parser.add_argument("--gail-epochs", type=int, default=None)
    parser.add_argument("--trajectory-frames", type=int, default=None)
    parser.add_argument("--refinement-source-rows", type=int, default=None)
    args = parser.parse_args()
    config = load_protocol_config(args.config.resolve())
    base = load_protocol_config(ROOT / config.get("base_config", "hierarchical_world_model/config/release.yaml"))
    experiment = prepare_experiment_data(base, ROOT)
    # Diffusion planning needs only the normalized C0 and slot mask.  Drop the
    # other Flow columns after the split/sequence contract has been checked;
    # retaining the several-GB natural table while scanning highD trajectories
    # previously made formal GAIL jobs exceed the host memory limit.
    for key in list(experiment.bundle.flow_arrays):
        if key not in {"features_normalized", "slot_mask"}:
            del experiment.bundle.flow_arrays[key]
    rows = experiment.train_rows if args.limit is None else experiment.train_rows[:int(args.limit)]
    fields = {key: value for key, value in config["human_prior"].items() if key in HumanPriorConfig.__dataclass_fields__}
    prior_config = HumanPriorConfig(**fields)
    overrides = {
        key: value for key, value in {
            "bc_epochs": args.bc_epochs,
            "gail_epochs": args.gail_epochs,
            "trajectory_frames": args.trajectory_frames,
            "refinement_source_rows": args.refinement_source_rows,
        }.items() if value is not None
    }
    if overrides:
        prior_config = replace(prior_config, **overrides)
    # Restrict expert extraction to the same dynamic-candidate population used
    # by closed-loop refinement.  The complete split is still scanned by the
    # offline selector; non-causal recordings simply cannot contribute a
    # longitudinal response prior.
    eligible_mask = dynamic_candidate_scene_mask(
        experiment.bundle.arrays["agent_states"],
        experiment.bundle.arrays["agent_valid"],
        rows=rows,
        radius_m=float(config["training"].get("influence_radius_m", 50.0)),
        prediction_horizon_s=float(config["training"].get("influence_prediction_horizon_s", 1.5)),
    )
    eligible_rows = rows[eligible_mask]
    if not len(eligible_rows):
        raise RuntimeError("no dynamic causal-influence candidates in GAIL training rows")
    features, actions, metadata = build_human_expert_samples(
        experiment.bundle.arrays, eligible_rows,
        max_samples=prior_config.max_expert_samples,
        seed=prior_config.seed,
        return_metadata=True,
    )
    validation_features, validation_actions = build_human_expert_samples(
        experiment.bundle.arrays,
        experiment.validation_rows,
        max_samples=20000,
        seed=prior_config.seed + 1,
    )
    _drop_mapped_pages(
        experiment.bundle.arrays["agent_states"],
        experiment.bundle.arrays["agent_valid"],
    )
    device = select_device(config["training"].get("device", "auto"))
    prior, summary = train_human_prior(
        features, actions, config=prior_config, device=device,
        validation_features=validation_features,
        validation_actions=validation_actions,
    )
    output = Path(config["paths"]["human_prior"]) if args.output is None else args.output
    ensure_dir(output.parent)
    bc_checkpoint = output.parent / "human_prior_bc.pt"
    torch.save({
        "schema": "longitudinal_gail_human_prior_v4",
        "stage": "probability_bc",
        "state_dict": prior.state_dict(),
        "config": prior_config.__dict__,
    }, bc_checkpoint)
    # The GAIL generator's final refinement is a real HighwayEnv rollout:
    # its target rear car follows the prior while other NPCs retain HiQR.
    model, _ = load_checkpoint(base["paths"]["evaluation_checkpoint"], device=device)
    # HighwayEnv refinement is restricted to the dynamic-candidate population,
    # but every selected recording is visited in every pass.  Selection uses
    # the union over observed frames only to define the offline population;
    # the online graph still sees realized state one tick at a time.
    source_count = (
        len(eligible_rows) if int(prior_config.refinement_source_rows) <= 0
        else min(len(eligible_rows), int(prior_config.refinement_source_rows))
    )
    source_index = np.linspace(0, len(eligible_rows) - 1, source_count, dtype=np.int64)
    source_rows = eligible_rows[source_index]
    small_rows, target_slots, parent_indices, target_roles = _anchor_relations_indexed(
        experiment.bundle.arrays, source_rows,
    )
    target_slots = target_slots.astype(np.int64)
    parent_indices = parent_indices.astype(np.int64)
    target_roles = target_roles.astype(np.int64)
    if not len(small_rows):
        raise RuntimeError("no causal parent--child HighwayEnv GAIL reset scenes")
    plans = frozen_diffusion_plans(experiment.bundle, small_rows, checkpoint=base["paths"]["diffusion_checkpoint"],
        output_dir=output.parent / "cache", device=device,
        batch_size=int(base["training"]["validation_batch_size"]), ddim_steps=int(config["training"]["diffusion_ddim_steps"]), experiment_scope=base["training"].get("experiment_scope", "full"))
    plans = _complete_plans_indexed(plans, experiment.bundle.arrays, small_rows)
    # Keep the source arrays as the repository's memory-mapped bundle.  Only
    # one HighwayEnv batch is materialised inside each full GAIL pass; making
    # a second full-copy ``compact`` dictionary was the source of the formal
    # run's multi-GB memory spike.
    artifact_metadata = {
        "training_rows_sha256": hashlib.sha256(np.asarray(rows, np.int64).tobytes()).hexdigest(),
        "highway_rows_sha256": hashlib.sha256(np.asarray(small_rows, np.int64).tobytes()).hexdigest(),
        "world_checkpoint_sha256": file_sha256(ROOT / base["paths"]["evaluation_checkpoint"]),
        "diffusion_checkpoint_sha256": file_sha256(ROOT / base["paths"]["diffusion_checkpoint"]),
    }
    summary["highway_gail_ppo"] = _highway_gail_refine(
        prior, model=model, arrays=experiment.bundle.arrays,
        source_rows=small_rows, plans=plans,
        target_slots=target_slots, parent_indices=parent_indices,
        target_roles=target_roles,
        expert_arrays=experiment.bundle.arrays,
        expert_features=features, expert_actions=actions,
        expert_metadata=metadata,
        validation_features=validation_features,
        validation_actions=validation_actions,
        config=prior_config, device=device,
        progress_path=output.parent / "training_progress.pt",
        evidence_path=output.parent / "training_distribution_samples.npz",
        artifact_metadata=artifact_metadata,
    )
    summary["gail_converged_no_admissible_update"] = bool(
        len(summary["highway_gail_ppo"]) >= 2
        and summary["highway_gail_ppo"][-1].get("actor_update_accepted") == 0.0
    )
    torch.save({"schema": "longitudinal_gail_human_prior_v4", "state_dict": prior.state_dict(),
                "config": prior_config.__dict__, "artifact_metadata": artifact_metadata}, output)
    save_json({**summary, "checkpoint": str(output), "bc_checkpoint": str(bc_checkpoint),
               "training_rows": int(len(rows)),
               "full_training_split": args.limit is None,
               "expert_corpus_mode": "all distinct parent-child role/risk relations from every eligible highD training recording",
               "expert_role_counts": {str(role): int((metadata["role"] == role).sum()) for role in (1, 2, 4)},
               "highway_refinement_source_rows": int(len(small_rows)),
               "highway_refinement_pass_contract": "each GAIL pass visits every anchor-eligible train scene once with a continuous 2-second rollout",
               "artifact_metadata": artifact_metadata},
              output.parent / "training_summary.json")
    print(output)


if __name__ == "__main__":
    main()

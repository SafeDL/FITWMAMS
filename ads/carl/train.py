"""CaRL PPO with local heads and fixed-teacher Beta retention."""

import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from ads.carl.policy import CaRLPolicy
from ads.common.instances import file_hash

TRAINABLE_PREFIXES = ("policy_head.", "dist_mu.", "dist_sigma.", "value_head.")


def configure_local_updates(policy):
    names = []
    for name, parameter in policy.network.named_parameters():
        parameter.requires_grad_(name.startswith(TRAINABLE_PREFIXES))
        if parameter.requires_grad:
            names.append(name)
    if not all(any(n.startswith(p) for n in names) for p in TRAINABLE_PREFIXES):
        raise ValueError("CaRL checkpoint does not expose the registered local heads")
    return names


def training_mode(policy, enabled):
    policy.network.eval()
    if enabled:
        for name in ("policy_head", "dist_mu", "dist_sigma", "value_head"):
            getattr(policy.network, name).train()


def retention_loss(policy, teacher, observations):
    """Use the connected Beta distribution, not CaRL's detached mu/sigma outputs."""
    training_mode(policy, False)
    batch = policy.tensors(observations, batched=True)
    with torch.no_grad():
        reference = teacher.network.get_action_dist_from_features(
            teacher.network.get_features(batch)
        )[0].distribution
    distribution = policy.network.get_action_dist_from_features(
        policy.network.get_features(batch)
    )[0].distribution
    return torch.distributions.kl_divergence(reference, distribution).sum(-1).mean()


def retention_groups(samples):
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["scene_id"]].append(sample["observation"])
    if not groups:
        raise ValueError("Prepare both native and highway CaRL retention observations")
    return groups


def choose_observations(groups, rng, count):
    scenes = sorted(groups)
    selected = []
    for _ in range(count):
        frames = groups[scenes[int(rng.integers(len(scenes)))]]
        selected.append(frames[int(rng.integers(len(frames)))])
    return selected


def train(
    policy,
    env,
    *,
    steps,
    seed,
    settings,
    retention,
    retention_coefficient=1.0,
    progress=None,
    checkpoint=None,
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    retention_rng = np.random.default_rng(np.random.SeedSequence([seed, 32]))
    groups = {name: retention_groups(retention[name]) for name in ("native", "keep")}
    network = policy.network
    names = configure_local_updates(policy)
    parameters = [p for p in network.parameters() if p.requires_grad]
    frozen = {
        name: value.detach().cpu().clone()
        for name, value in network.state_dict().items()
        if name not in names
    }
    official = Path(__file__).parent / "artifacts/official.pth"
    teacher = CaRLPolicy.load(official, device=str(policy.device))
    teacher.network.eval().requires_grad_(False)
    teacher_hash = file_hash(official)
    if policy.optimizer_state is not None and policy.repair_state is None:
        raise ValueError(
            "Start CaRL repair from official or capability-preserving weights"
        )
    if policy.repair_state is not None and (
        policy.repair_state["teacher_sha256"] != teacher_hash
        or policy.repair_state["trainable_parameters"] != names
    ):
        raise ValueError("CaRL repair teacher or local parameter set changed")
    optimizer = torch.optim.Adam(parameters, lr=settings["learning_rate"], eps=1e-5)
    if policy.optimizer_state is not None:
        optimizer.load_state_dict(policy.optimizer_state)
    for group in optimizer.param_groups:
        group["lr"] = settings["learning_rate"]
    observation, info = env.reset(seed=seed)
    event_id, event_decisions = info["event_id"], Counter()
    started, collected, updates = time.perf_counter(), 0, 0
    losses, retention_counts = [], {"native": 0, "keep": 0}
    gamma, lam = settings["gamma"], settings["gae_lambda"]
    while collected < steps:
        n = min(settings["rollout_steps"], steps - collected)
        observations = {
            k: np.empty((n, *v.shape), v.dtype) for k, v in observation.items()
        }
        actions = np.empty((n, 2), np.float32)
        rewards, dones, values, log_probs = (np.empty(n, np.float32) for _ in range(4))
        network.eval()
        for i in range(n):
            event_decisions[event_id] += 1
            for key, buffer in observations.items():
                buffer[i] = observation[key]
            with torch.no_grad():
                output = network(policy.tensors(observation))
            action = output[0][0].cpu().numpy()
            actions[i], log_probs[i], values[i] = (
                action,
                float(output[1][0]),
                float(output[3][0]),
            )
            observation, reward, terminated, truncated, _ = env.step(action)
            if truncated and not terminated:
                with torch.no_grad():
                    reward += gamma * float(
                        network.get_value(policy.tensors(observation))[0]
                    )
            rewards[i], dones[i] = reward, terminated or truncated
            if terminated or truncated:
                observation, info = env.reset()
                event_id = info["event_id"]
            if progress and (collected + i + 1) % 1024 == 0:
                progress(collected + i + 1, steps)
        with torch.no_grad():
            final_value = float(network.get_value(policy.tensors(observation))[0])
        advantages = np.empty(n, np.float32)
        running = 0.0
        for i in range(n - 1, -1, -1):
            next_value = final_value if i == n - 1 else values[i + 1]
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
            running = delta + gamma * lam * (1 - dones[i]) * running
            advantages[i] = running
        returns = advantages + values
        for _ in range(settings["epochs"]):
            permutation = rng.permutation(n)
            for start in range(0, n, settings["batch_size"]):
                logical = permutation[start : start + settings["batch_size"]]
                advantage = advantages[logical].copy()
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                optimizer.zero_grad()
                training_mode(policy, True)
                ppo_value = 0.0
                for offset in range(0, len(logical), settings["micro_batch_size"]):
                    indices = logical[offset : offset + settings["micro_batch_size"]]
                    batch = policy.tensors(
                        {k: v[indices] for k, v in observations.items()}, batched=True
                    )
                    output = network(
                        batch,
                        actions=torch.as_tensor(actions[indices], device=policy.device),
                    )
                    new_log, entropy, prediction = (
                        output[1],
                        output[2],
                        output[3].flatten(),
                    )
                    old_log = torch.as_tensor(log_probs[indices], device=policy.device)
                    ratio = (new_log - old_log).exp()
                    adv = torch.as_tensor(
                        advantage[offset : offset + len(indices)], device=policy.device
                    )
                    clip = settings["clip_range"]
                    policy_loss = -torch.minimum(
                        ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv
                    ).mean()
                    old_values = torch.as_tensor(values[indices], device=policy.device)
                    target = torch.as_tensor(returns[indices], device=policy.device)
                    clipped = old_values + (prediction - old_values).clamp(-clip, clip)
                    value_loss = torch.maximum(
                        (prediction - target).square(), (clipped - target).square()
                    ).mean()
                    loss = (
                        policy_loss
                        + settings["value_coefficient"] * value_loss
                        - settings["entropy_coefficient"] * entropy.mean()
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite CaRL PPO loss")
                    (loss * len(indices) / len(logical)).backward()
                    ppo_value += float(loss.detach()) * len(indices) / len(logical)
                selected = []
                for name in ("native", "keep"):
                    selected += choose_observations(groups[name], retention_rng, 32)
                    retention_counts[name] += 32
                retain_value = 0.0
                for offset in range(0, 64, settings["micro_batch_size"]):
                    micro = selected[offset : offset + settings["micro_batch_size"]]
                    observations_batch = {
                        key: np.stack([frame[key] for frame in micro])
                        for key in micro[0]
                    }
                    loss = retention_loss(policy, teacher, observations_batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite CaRL Beta retention loss")
                    (retention_coefficient * loss * len(micro) / 64).backward()
                    retain_value += float(loss.detach()) * len(micro) / 64
                gradient = float(torch.nn.utils.clip_grad_norm_(parameters, 0.5))
                optimizer.step()
                updates += 1
                losses.append(
                    {
                        "optimizer_update": updates,
                        "ppo_loss": ppo_value,
                        "retention_loss": retain_value,
                        "gradient_norm": gradient,
                    }
                )
        collected += n
        for group in optimizer.param_groups:
            group["lr"] = settings["learning_rate"] * max(1 - collected / steps, 0.05)
        policy.optimizer_state = optimizer.state_dict()
        policy.repair_state = {
            "schema": "carl_capability_preserving_repair_v1",
            "teacher_sha256": teacher_hash,
            "trainable_parameters": names,
            "decision_steps": collected,
            "optimizer_updates": updates,
            "seed": seed,
            "retention_coefficient": retention_coefficient,
            "retention_batch_quotas": {"native": 32, "keep": 32},
        }
        if checkpoint and collected in (min(8192, steps), steps):
            training_mode(policy, False)
            checkpoint(policy, collected)
    policy.optimizer_state = optimizer.state_dict()
    network.eval()
    unchanged = all(
        torch.equal(value, network.state_dict()[name].cpu())
        for name, value in frozen.items()
    )
    if not unchanged:
        raise AssertionError("Frozen CaRL features or normalization state changed")
    if progress:
        progress(collected, steps)
    return {
        "actual_steps": collected,
        "optimizer_updates": updates,
        "elapsed_s": time.perf_counter() - started,
        "category_resets": dict(env.category_counts),
        "event_decisions": dict(event_decisions),
        "teacher_sha256": teacher_hash,
        "trainable_parameters": names,
        "trainable_parameter_count": sum(p.numel() for p in parameters),
        "frozen_state_identical": unchanged,
        "retention_observations": retention_counts,
        "losses": losses,
    }

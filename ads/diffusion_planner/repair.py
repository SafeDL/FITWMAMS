"""Native DP repair with scene-balanced replay, a fixed teacher and local updates."""

import time
from collections import defaultdict
from copy import copy, deepcopy

import numpy as np
import torch

from ads.common.instances import file_hash
from ads.diffusion_planner.policy import DiffusionPlannerPolicy
from ads.nuplan.assets import official_path
from ads.nuplan.losses import diffusion_batch, diffusion_loss, mean_valid

TRAINABLE_PREFIXES = (
    "decoder.decoder.dit.blocks.2.",
    "decoder.decoder.dit.final_layer.",
)


def configure_local_updates(policy):
    names = []
    for name, parameter in policy.network.named_parameters():
        parameter.requires_grad_(name.startswith(TRAINABLE_PREFIXES))
        if parameter.requires_grad:
            names.append(name)
    if not names or not all(
        any(name.startswith(prefix) for name in names) for prefix in TRAINABLE_PREFIXES
    ):
        raise ValueError("DP checkpoint does not expose the registered local modules")
    return names


def training_mode(policy, enabled):
    policy.network.eval()
    if enabled:
        policy.network.decoder.decoder.dit.blocks[2].train()
        policy.network.decoder.decoder.dit.final_layer.train()


def retention_loss(student, teacher, samples):
    """Same native noisy input/time; eval removes dropout without detaching student."""
    training_mode(student, False)
    inputs, trajectories, time, _, valid, _, _ = diffusion_batch(student, samples)
    with torch.no_grad():
        expected = teacher.denoise(inputs, trajectories, time)[:, :, 1:]
    prediction = student.denoise(inputs, trajectories, time)[:, :, 1:]
    error = (prediction - expected).square().sum(-1)
    return mean_valid(error[:, 0], valid[:, 0]) + mean_valid(error[:, 1:], valid[:, 1:])


def grouped_samples(samples):
    groups = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        if np.asarray(sample["future_valid"])[0].any():
            groups[sample["domain"]][sample["scene_id"]].append(sample)
    return groups


def choose(groups, rng, count):
    result = []
    domains = sorted(groups)
    for _ in range(count):
        scenes = groups[domains[int(rng.integers(len(domains)))]]
        identities = sorted(scenes)
        frames = scenes[identities[int(rng.integers(len(identities)))]]
        result.append(frames[int(rng.integers(len(frames)))])
    return result


def train(
    policy,
    banks,
    *,
    updates,
    seed,
    round_index=0,
    micro_batch_size=8,
    learning_rate=1e-5,
    retention_coefficient=1.0,
    checkpoint=None,
    progress=None,
):
    """Every update uses 32 native, 16 highway-keep and 16 correction samples."""
    started = time.perf_counter()
    groups = {
        name: grouped_samples(banks.get(name, []))
        for name in ("native", "keep", "normal", "new_failure", "history")
    }
    if not any(
        sample["label_source"] == "expert"
        and np.asarray(sample["future_valid"])[0].any()
        for category in ("normal", "new_failure")
        for sample in banks.get(category, [])
    ):
        return {
            "gradient_updates": 0,
            "skipped": "no_qualified_demonstrations",
            "elapsed_s": time.perf_counter() - started,
        }
    if not groups["native"] or not groups["keep"]:
        raise ValueError(
            "Prepare native and successful highway retention material before repair"
        )
    # Snapshot the accepted deployment before restoring its trainable raw state.
    previous = copy(policy)
    previous.network = deepcopy(policy.network).eval().requires_grad_(False)
    if policy.training_state is not None:
        policy.network.load_state_dict(policy.training_state, strict=True)
    names = configure_local_updates(policy)
    teacher = DiffusionPlannerPolicy.load(
        official_path("diffusion_planner"), device=str(policy.device)
    )
    teacher.network.eval().requires_grad_(False)
    teacher_hash = file_hash(official_path("diffusion_planner"))
    if policy.repair_state and (
        policy.repair_state["teacher_sha256"] != teacher_hash
        or policy.repair_state["trainable_parameters"] != names
    ):
        raise ValueError("Repair teacher or local parameter set changed")
    parameters = [p for p in policy.network.parameters() if p.requires_grad]
    frozen = {
        key: value.detach().cpu().clone()
        for key, value in policy.network.state_dict().items()
        if key not in names
    }
    ema = {key: value.clone() for key, value in policy.ema_state.items()}
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    if policy.optimizer_state is not None:
        optimizer.load_state_dict(policy.optimizer_state)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    categories = ("normal", "new_failure", "history")
    weights = np.asarray((0.5, 0.5, 0.0) if round_index == 0 else (0.5, 0.3, 0.2))
    for index in (1, 2):
        if not groups[categories[index]]:
            weights[0] += weights[index]
            weights[index] = 0
    if not groups["normal"]:
        weights[0] = 0
    weights /= weights.sum()
    counts = dict.fromkeys(groups, 0)
    objective_counts = {
        "native_supervision": 0,
        "original_retention": 0,
        "successful_retention": 0,
    }
    losses = []

    def deploy_snapshot(done):
        policy.training_state = {
            key: value.detach().cpu().clone()
            for key, value in policy.network.state_dict().items()
        }
        policy.optimizer_state = optimizer.state_dict()
        policy.ema_state = ema
        policy.repair_state = {
            "schema": "dp_capability_preserving_repair_v2",
            "teacher_sha256": teacher_hash,
            "trainable_parameters": names,
            "seed": seed,
            "round_index": round_index,
            "gradient_updates": done,
            "retention_coefficient": retention_coefficient,
            "batch_quotas": {"native": 32, "keep": 16, "correction": 16},
            "successful_reference": "round_start_deployment",
        }
        policy.network.load_state_dict(ema, strict=True)
        training_mode(policy, False)

    for update in range(updates):
        native = choose(groups["native"], rng, 32)
        keep = native + choose(groups["keep"], rng, 16)
        counts["native"] += 32
        counts["keep"] += 16
        corrections = []
        for category in rng.choice(categories, size=16, p=weights):
            corrections += choose(groups[category], rng, 1)
            counts[category] += 1
        supervised = native + [
            sample for sample in corrections if sample["label_source"] == "expert"
        ]
        successful = [
            sample
            for sample in corrections
            if sample["label_source"] == "policy_success"
        ]
        objective_counts["native_supervision"] += len(supervised)
        objective_counts["original_retention"] += len(keep)
        objective_counts["successful_retention"] += len(successful)
        optimizer.zero_grad(set_to_none=True)
        native_value, retain_value, success_value = 0.0, 0.0, 0.0
        for start in range(0, len(supervised), micro_batch_size):
            micro = supervised[start : start + micro_batch_size]
            training_mode(policy, True)
            loss = diffusion_loss(policy, micro)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite native DP objective")
            (loss * len(micro) / 64).backward()
            native_value += float(loss.detach()) * len(micro) / 64
        for start in range(0, 48, micro_batch_size):
            micro = keep[start : start + micro_batch_size]
            loss = retention_loss(policy, teacher, micro)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite DP teacher constraint")
            (retention_coefficient * loss * len(micro) / 48).backward()
            retain_value += float(loss.detach()) * len(micro) / 48
        for start in range(0, len(successful), micro_batch_size):
            micro = successful[start : start + micro_batch_size]
            loss = retention_loss(policy, previous, micro)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite successful-policy constraint")
            (retention_coefficient * loss * len(micro) / 64).backward()
            success_value += float(loss.detach()) * len(micro) / 64
        gradient = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
        optimizer.step()
        for name, parameter in policy.network.named_parameters():
            if parameter.requires_grad:
                ema[name].mul_(0.999).add_(parameter.detach().cpu(), alpha=0.001)
        losses.append(
            {
                "update": update + 1,
                "native_loss": native_value,
                "retention_loss": retain_value,
                "successful_retention_loss": success_value,
                "gradient_norm": gradient,
            }
        )
        if progress and ((update + 1) % 10 == 0 or update + 1 == updates):
            progress(update + 1, updates, losses[-1])
        if checkpoint and (update + 1 in (500, 1000) or update + 1 == updates):
            deploy_snapshot(update + 1)
            checkpoint(policy, update + 1)
            policy.network.load_state_dict(policy.training_state, strict=True)
    for key, value in frozen.items():
        if not torch.equal(
            value, policy.network.state_dict()[key].detach().cpu()
        ) or not torch.equal(value, ema[key]):
            raise AssertionError(f"Frozen DP state changed: {key}")
    deploy_snapshot(updates)
    return {
        "gradient_updates": updates,
        "batch_size": 64,
        "category_samples": counts,
        "objective_samples": objective_counts,
        "trainable_parameters": names,
        "trainable_elements": sum(p.numel() for p in parameters),
        "teacher_sha256": teacher_hash,
        "retention_coefficient": retention_coefficient,
        "losses": losses,
        "frozen_state_unchanged": True,
        "deployment": "student_ema",
        "successful_reference": "round_start_deployment",
        "elapsed_s": time.perf_counter() - started,
    }

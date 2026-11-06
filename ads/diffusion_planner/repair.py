"""Native DP repair with scene-balanced replay, a fixed teacher and local updates."""

import time

import numpy as np
import torch

from ads.common.instances import file_hash
from ads.diffusion_planner.constraints import constrain_step, flatten_gradients
from ads.diffusion_planner.policy import DiffusionPlannerPolicy
from ads.diffusion_planner.sampling import BalancedSampler
from ads.nuplan.assets import official_path
from ads.nuplan.losses import diffusion_batch, diffusion_loss, mean_valid

TRAINABLE_PREFIXES = (
    "decoder.decoder.dit.blocks.2.",
    "decoder.decoder.dit.final_layer.",
)


def normalized_schedule(updates, reference_updates, learning_rate, ema_decay):
    """Cover the full pool without scaling the learning-rate integral or EMA age."""

    def mass(count):
        return np.maximum(0.1, 1 - np.arange(count) / count).sum()

    return {
        "reference_updates": reference_updates,
        "physical_updates": updates,
        "learning_rate": float(learning_rate * mass(reference_updates) / mass(updates)),
        "ema_decay": float(ema_decay ** (reference_updates / updates)),
        "reference_learning_rate_sum": float(learning_rate * mass(reference_updates)),
        "reference_ema_retention": float(ema_decay**reference_updates),
    }


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


def retention_loss(student, teacher, samples, *, denominators=None):
    """Same native noisy input/time; eval removes dropout without detaching student."""
    training_mode(student, False)
    inputs, trajectories, time, _, valid, _, _ = diffusion_batch(student, samples)
    with torch.no_grad():
        expected = teacher.denoise(inputs, trajectories, time)[:, :, 1:]
    prediction = student.denoise(inputs, trajectories, time)[:, :, 1:]
    error = (prediction - expected).square().sum(-1)
    if denominators is None:
        return mean_valid(error[:, 0], valid[:, 0]) + mean_valid(
            error[:, 1:], valid[:, 1:]
        )
    return (error[:, 0] * valid[:, 0]).sum() / max(denominators[0], 1) + (
        error[:, 1:] * valid[:, 1:]
    ).sum() / max(denominators[1], 1)


def valid_counts(samples):
    counts = np.zeros(2, dtype=np.int64)
    for sample in samples:
        valid = np.asarray(sample["future_valid"], bool)
        counts += (valid[0].sum(), valid[1:11].sum())
    return counts.tolist()


def train(
    policy,
    banks,
    *,
    updates,
    seed,
    phase_updates=None,
    phase_offset=0,
    round_index=0,
    micro_batch_size=8,
    learning_rate=1e-5,
    ema_decay=0.999,
    retention_coefficient=1.0,
    checkpoint=None,
    progress=None,
    protected_roots=(),
):
    """One phase of 64 native / 32 successful-keep / 32 expert updates."""
    started = time.perf_counter()
    phase_updates = updates if phase_updates is None else phase_updates
    corrections, successful = [], []
    for category in ("normal", "new_failure", "history"):
        for sample in banks.get(category, []):
            if not np.asarray(sample["future_valid"])[0].any():
                continue
            if sample["label_source"] == "expert":
                corrections.append(sample)
            elif sample["label_source"] == "policy_success":
                successful.append(sample)
    if not corrections:
        return {
            "gradient_updates": 0,
            "skipped": "no_qualified_demonstrations",
            "elapsed_s": time.perf_counter() - started,
        }
    if not banks.get("native") or not banks.get("keep"):
        raise ValueError(
            "Prepare native and successful highway retention material before repair"
        )
    previous_state = policy.repair_state or {}
    done_before = previous_state.get("gradient_updates", 0)
    if updates < 1 or not 0 < learning_rate <= 1e-5 or not 0 < ema_decay < 1:
        raise ValueError(
            "Repair needs positive updates, learning rate <= 1e-5 and EMA decay in (0, 1)"
        )
    if policy.training_state is not None:
        policy.network.load_state_dict(policy.training_state, strict=True)
    names = configure_local_updates(policy)
    teacher = DiffusionPlannerPolicy.load(
        official_path("diffusion_planner"), device=str(policy.device)
    )
    teacher.network.eval().requires_grad_(False)
    teacher_hash = file_hash(official_path("diffusion_planner"))
    if previous_state and (
        previous_state["teacher_sha256"] != teacher_hash
        or previous_state["trainable_parameters"] != names
    ):
        raise ValueError("Repair teacher or local parameter set changed")
    parameters = [p for p in policy.network.parameters() if p.requires_grad]
    frozen = {
        k: v.detach().cpu().clone()
        for k, v in policy.network.state_dict().items()
        if k not in names
    }
    ema = {k: v.clone() for k, v in policy.ema_state.items()}
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    if policy.optimizer_state is not None:
        optimizer.load_state_dict(policy.optimizer_state)
    material = {
        "native": banks["native"],
        "keep": list(banks["keep"]) + successful if successful else banks["keep"],
        "correction": corrections,
    }
    states = previous_state.get("samplers", {})
    samplers = {
        name: BalancedSampler(
            values,
            seed + index,
            state=states.get(name),
            protected_roots=protected_roots,
            key_windows=name == "correction",
        )
        for index, (name, values) in enumerate(material.items())
    }
    trainable = [
        (name, p) for name, p in policy.network.named_parameters() if p.requires_grad
    ]
    ema_parameters = [
        torch.nn.Parameter(ema[name].to(policy.device).clone(), requires_grad=False)
        for name, _ in trainable
    ]
    torch.manual_seed(seed + done_before)
    losses = []

    def deploy_snapshot(done):
        for (name, _), parameter in zip(trainable, ema_parameters):
            ema[name] = parameter.detach().cpu().clone()
        policy.training_state = {
            k: v.detach().cpu().clone() for k, v in policy.network.state_dict().items()
        }
        policy.optimizer_state = optimizer.state_dict()
        policy.ema_state = {k: v.clone() for k, v in ema.items()}
        policy.repair_state = {
            "schema": "dp_capability_preserving_repair_full",
            "teacher_sha256": teacher_hash,
            "trainable_parameters": names,
            "seed": seed,
            "round_index": round_index,
            "gradient_updates": done,
            "retention_coefficient": retention_coefficient,
            "ema_decay": ema_decay,
            "training_elapsed_s": previous_state.get("training_elapsed_s", 0.0)
            + time.perf_counter()
            - started,
            "batch_quotas": {"native": 64, "keep": 32, "correction": 32},
            "successful_reference": "official_initial_deployment",
            "samplers": {
                name: sampler.state_dict() for name, sampler in samplers.items()
            },
            "exposure": {name: sampler.report() for name, sampler in samplers.items()},
            "losses": previous_state.get("losses", []) + list(losses),
            "rng_cpu": torch.get_rng_state().tolist(),
            "rng_cuda": (
                torch.cuda.get_rng_state(policy.device).tolist()
                if policy.device.type == "cuda"
                else None
            ),
        }
        policy.network.load_state_dict(ema, strict=True)
        training_mode(policy, False)

    if "rng_cpu" in previous_state:
        torch.set_rng_state(torch.tensor(previous_state["rng_cpu"], dtype=torch.uint8))
        if policy.device.type == "cuda" and previous_state.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(
                torch.tensor(previous_state["rng_cuda"], dtype=torch.uint8),
                policy.device,
            )

    def accumulate(samples, objective, coefficient):
        denominator = valid_counts(samples)
        value = 0.0
        for start in range(0, len(samples), micro_batch_size):
            micro = samples[start : start + micro_batch_size]
            loss = objective(micro, denominator)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite DP repair objective")
            (coefficient * loss).backward()
            value += float(loss.detach())
        return value

    for update in range(updates):
        done = done_before + update
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * max(
                0.1, 1 - (phase_offset + update) / phase_updates
            )
        native = samplers["native"].draw(64)
        keep = samplers["keep"].draw(32)
        corrected = samplers["correction"].draw(32)
        optimizer.zero_grad(set_to_none=True)
        native_parts, correction_parts = {}, {}

        def native_objective(micro, denominator, parts=native_parts):
            training_mode(policy, True)
            supervised = diffusion_loss(
                policy, micro, denominators=denominator, components=parts
            )
            retained = retention_loss(policy, teacher, micro, denominators=denominator)
            return supervised + retention_coefficient * retained

        native_value = accumulate(native, native_objective, 0.5)
        native_gradient = flatten_gradients(parameters).clone()
        optimizer.zero_grad(set_to_none=True)
        keep_value = accumulate(
            keep,
            lambda micro, denom: retention_loss(
                policy, teacher, micro, denominators=denom
            ),
            0.25 * retention_coefficient,
        )
        keep_gradient = flatten_gradients(parameters).clone()
        combined = native_gradient + keep_gradient
        offset = 0
        for parameter in parameters:
            parameter.grad = (
                combined[offset : offset + parameter.numel()]
                .reshape_as(parameter)
                .clone()
            )
            offset += parameter.numel()

        def corrective_objective(micro, denominator, parts=correction_parts):
            training_mode(policy, True)
            return diffusion_loss(
                policy, micro, denominators=denominator, components=parts
            )

        correction_value = accumulate(corrected, corrective_objective, 0.25)
        gradient = float(
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        )
        before = torch.cat([p.detach().reshape(-1) for p in parameters]).clone()
        optimizer.step()
        constraint_started = time.perf_counter()
        constraint = constrain_step(
            parameters, before, [native_gradient, keep_gradient]
        )
        constraint["elapsed_s"] = time.perf_counter() - constraint_started
        ema_before = torch.cat([p.detach().reshape(-1) for p in ema_parameters]).clone()
        with torch.no_grad():
            for (_, parameter), averaged in zip(trainable, ema_parameters):
                averaged.add_(parameter.detach() - averaged, alpha=1 - ema_decay)
        ema_constraint_started = time.perf_counter()
        ema_constraint = constrain_step(
            ema_parameters, ema_before, [native_gradient, keep_gradient]
        )
        ema_constraint["elapsed_s"] = time.perf_counter() - ema_constraint_started
        losses.append(
            {
                "update": done + 1,
                "native_loss": native_value,
                "retention_loss": keep_value,
                "correction_loss": correction_value,
                "gradient_norm": gradient,
                "constraint": constraint,
                "ema_constraint": ema_constraint,
                "native_supervised_loss": sum(native_parts.values()),
                "native_teacher_loss": native_value - sum(native_parts.values()),
                "correction_ego_loss": correction_parts.get("ego", 0.0),
                "correction_neighbor_loss": correction_parts.get("neighbors", 0.0),
                "total_loss": 0.5 * native_value
                + 0.25 * retention_coefficient * keep_value
                + 0.25 * correction_value,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "ema_decay": ema_decay,
            }
        )
        if progress and ((update + 1) % 10 == 0 or update + 1 == updates):
            progress(done + 1, done_before + updates, losses[-1])
        if checkpoint and (update + 1) % 50 == 0 and update + 1 < updates:
            deploy_snapshot(done + 1)
            checkpoint(policy, done + 1)
            policy.network.load_state_dict(policy.training_state, strict=True)
    for key, value in frozen.items():
        if not torch.equal(
            value, policy.network.state_dict()[key].detach().cpu()
        ) or not torch.equal(value, ema[key]):
            raise AssertionError(f"Frozen DP state changed: {key}")
    deploy_snapshot(done_before + updates)
    if checkpoint:
        checkpoint(policy, done_before + updates)
    return {
        "gradient_updates": updates,
        "cumulative_updates": done_before + updates,
        "batch_size": 128,
        "micro_batch_size": micro_batch_size,
        "category_samples": {
            k: updates * n
            for k, n in (("native", 64), ("keep", 32), ("correction", 32))
        },
        "exposure": policy.repair_state["exposure"],
        "trainable_parameters": names,
        "trainable_elements": sum(p.numel() for p in parameters),
        "teacher_sha256": teacher_hash,
        "retention_coefficient": retention_coefficient,
        "losses": losses,
        "frozen_state_unchanged": True,
        "deployment": "student_ema",
        "successful_reference": "official_initial_deployment",
        "elapsed_s": time.perf_counter() - started,
    }

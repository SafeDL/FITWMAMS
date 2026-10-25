"""Bounded supervised repair with native objectives and observed-future masks."""

import time

import numpy as np
import torch

from ads.nuplan.losses import OBJECTIVES, dtpp_candidates


def train(
    policy,
    banks,
    *,
    updates,
    seed,
    batch_size=64,
    micro_batch_size=8,
    learning_rate=1e-5,
    round_index=0,
    progress=None,
):
    names = ("normal", "new_failure", "history")
    usable = {
        name: [s for s in banks.get(name, []) if np.asarray(s["future_valid"])[0].any()]
        for name in names
    }
    if not any(usable.values()):
        return {
            "gradient_updates": 0,
            "skipped": "no_qualified_demonstrations",
            "batch_size": batch_size,
            "category_samples": dict.fromkeys(names, 0),
            "elapsed_s": 0.0,
        }
    if (
        policy.algorithm == "dtpp"
        and getattr(policy.network["decoder"], "batch_mask_layout", None)
        != "batch_then_head"
    ):
        raise RuntimeError(
            "DTPP batch-mask correction requires a fresh training worker; "
            "reuse the completed search with --training-only"
        )
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    weights = np.array((0.5, 0.5, 0.0) if round_index == 0 else (0.5, 0.3, 0.2))
    for index in (1, 2):
        if not usable[names[index]]:
            weights[0] += weights[index]
            weights[index] = 0
    # There can be no successful prior demonstrations even when defects have an
    # expert. Allocate actual labels proportionally; record this coverage gap.
    if not usable["normal"]:
        weights[0] = 0
        if weights.sum() == 0:
            weights = np.array([bool(usable[name]) for name in names], float)
    weights /= weights.sum()
    groups = {}
    for name in names:
        groups[name] = {}
        for sample in usable[name]:
            groups[name].setdefault(sample["domain"], []).append(sample)
    started = time.perf_counter()
    if policy.algorithm == "dtpp":
        policy.network.eval()
        for samples in usable.values():
            for sample in samples:
                sample["dtpp_candidates"] = dtpp_candidates(policy, sample)
    if policy.training_state is not None:
        policy.network.load_state_dict(policy.training_state, strict=True)
    ema = policy.ema_state
    optimizer = torch.optim.AdamW(
        policy.network.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    if policy.optimizer_state is not None:
        optimizer.load_state_dict(policy.optimizer_state)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    counts = dict.fromkeys(names, 0)
    policy.network.train()
    if policy.algorithm == "str2":
        policy.network.transformer.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        policy.network.transformer.config.use_cache = False
    # This is also how small native repair batches retain pretrained BN statistics.
    for module in policy.network.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    last_loss = 0.0
    for update in range(updates):
        samples = []
        for _ in range(batch_size):
            name = str(rng.choice(names, p=weights))
            candidates = (
                usable[name]
                if name == "normal"
                else groups[name][str(rng.choice(sorted(groups[name])))]
            )
            samples.append(candidates[int(rng.integers(len(candidates)))])
            counts[name] += 1
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, batch_size, micro_batch_size):
            micro = samples[start : start + micro_batch_size]
            with torch.autocast(
                device_type=policy.device.type,
                dtype=torch.bfloat16,
                enabled=policy.algorithm == "str2" and policy.device.type == "cuda",
            ):
                loss = OBJECTIVES[policy.algorithm](policy, micro)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite native planner repair loss")
            (loss * len(micro) / batch_size).backward()
            last_loss = float(loss.detach())
        torch.nn.utils.clip_grad_norm_(
            policy.network.parameters(), 5 if policy.algorithm == "dtpp" else 1
        )
        optimizer.step()
        if ema is not None:
            # The published DP and Flow Planner both use decay 0.999.
            for key, value in policy.network.state_dict().items():
                current = value.detach().cpu()
                if current.is_floating_point():
                    ema[key].mul_(0.999).add_(current, alpha=0.001)
                else:
                    ema[key].copy_(current)
        if progress and ((update + 1) % 100 == 0 or update + 1 == updates):
            progress(update + 1, updates)
    policy.optimizer_state = optimizer.state_dict()
    if ema is not None:
        policy.training_state = {
            key: value.detach().cpu().clone()
            for key, value in policy.network.state_dict().items()
        }
        policy.network.load_state_dict(ema, strict=True)
    policy.network.eval()
    if policy.algorithm == "str2":
        policy.network.transformer.config.use_cache = policy.config.get(
            "use_cache", True
        )
    report = {
        "gradient_updates": updates,
        "batch_size": batch_size,
        "effective_category_weights": weights.tolist(),
        "category_samples": counts,
        "last_loss": last_loss,
        "elapsed_s": time.perf_counter() - started,
        "objective": policy.algorithm + " native masked objective",
        "new_defect_supervision_available": bool(usable["new_failure"]),
        "normal_supervision_available": bool(usable["normal"]),
    }
    if policy.algorithm == "dtpp":
        report["dtpp_attention_mask_layout"] = policy.network[
            "decoder"
        ].batch_mask_layout
    return report

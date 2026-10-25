"""Supervised repair keeps the official planning and object-forecast losses."""

import time

import numpy as np
import torch
from torch.nn import functional as F

from ads.plant2.policy import collate


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
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    names = ("normal", "new_failure", "history")
    probabilities = np.asarray((0.5, 0.5, 0.0) if round_index == 0 else (0.5, 0.3, 0.2))
    if not any(banks.get(name) for name in names):
        return {
            "gradient_updates": 0,
            "skipped": "no_qualified_demonstrations",
            "batch_size": batch_size,
            "category_samples": dict.fromkeys(names, 0),
            "elapsed_s": 0.0,
        }
    for i, name in enumerate(names[1:], 1):
        if not banks.get(name):
            probabilities[0] += probabilities[i]
            probabilities[i] = 0
    if not banks.get("normal"):
        probabilities[0] = 0
        if probabilities.sum() == 0:
            probabilities = np.array([bool(banks.get(name)) for name in names], float)
    probabilities /= probabilities.sum()
    groups = {name: {} for name in names}
    for name in names:
        for sample in banks.get(name, []):
            groups[name].setdefault(sample["domain"], []).append(sample)
    optimizer = policy.network.configure_optimizers(policy.cfg.model.training)
    if policy.optimizer_state is not None:
        optimizer.load_state_dict(policy.optimizer_state)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    counts = {name: 0 for name in names}
    policy.network.train()
    started = time.perf_counter()
    last_losses = {}
    for update in range(updates):
        samples = []
        for _ in range(batch_size):
            name = str(rng.choice(names, p=probabilities))
            if name == "normal":
                candidates = banks[name]
            else:
                domains = sorted(groups[name])
                candidates = groups[name][domains[int(rng.integers(len(domains)))]]
            samples.append(candidates[int(rng.integers(len(candidates)))])
            counts[name] += 1
        optimizer.zero_grad()
        for start in range(0, batch_size, micro_batch_size):
            micro = samples[start : start + micro_batch_size]
            logits, targets, (path, waypoints, _), _ = policy.network(
                collate(micro, policy.device, training=True)
            )
            path_target = torch.as_tensor(
                np.stack([s["path"] for s in micro]), device=policy.device
            )
            path_valid = torch.as_tensor(
                np.stack([s["path_valid"] for s in micro]), device=policy.device
            )
            path_loss = F.l1_loss(path[path_valid], path_target[path_valid])
            wp_loss = F.l1_loss(
                waypoints,
                torch.as_tensor(
                    np.stack([s["waypoints"] for s in micro]), device=policy.device
                ),
            )
            forecast_losses = []
            for pred, target in zip(logits, targets):
                valid = target != -999
                if valid.any():
                    forecast_losses.append(F.cross_entropy(pred[valid], target[valid]))
            forecast_loss = (
                torch.stack(forecast_losses).mean()
                if forecast_losses
                else path_loss * 0
            )
            loss = 5 * path_loss + wp_loss + forecast_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite PlanT supervised loss")
            (loss * len(micro) / batch_size).backward()
            last_losses = {
                "path_l1": float(path_loss.detach()),
                "waypoint_l1": float(wp_loss.detach()),
                "forecast_ce": float(forecast_loss.detach()),
            }
        torch.nn.utils.clip_grad_norm_(policy.network.parameters(), 1)
        optimizer.step()
        if progress and ((update + 1) % 100 == 0 or update + 1 == updates):
            progress(update + 1, updates)
    policy.optimizer_state = optimizer.state_dict()
    policy.network.eval()
    return {
        "gradient_updates": updates,
        "batch_size": batch_size,
        "category_samples": counts,
        "last_microbatch_losses": last_losses,
        "elapsed_s": time.perf_counter() - started,
        "new_defect_supervision_available": bool(banks.get("new_failure")),
    }

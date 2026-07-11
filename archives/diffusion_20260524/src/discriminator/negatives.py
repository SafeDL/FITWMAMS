"""Negative sample generation for the naturalness discriminator."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


RANDOM_PERTURB_SOURCE = "random_perturb"
RULE_BRAKE_SOURCE = "rule_brake"


def _as_actions(actions: np.ndarray) -> np.ndarray:
    x = np.asarray(actions, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] < 1:
        raise ValueError(f"Expected actions shape [B,H,1+], got {x.shape}")
    return x


def generate_random_perturb_negatives(
    actions: np.ndarray,
    *,
    rng: np.random.Generator,
    config: dict,
    copies_per_positive: int = 1,
) -> np.ndarray:
    """Generate easy non-natural perturbations from real highD futures."""
    base = _as_actions(actions)
    action_cfg = config.get("action", {})
    jerk_abs_max = float(action_cfg.get("jerk_abs_max", 12.0))
    out: list[np.ndarray] = []
    for copy_idx in range(max(0, int(copies_per_positive))):
        x = base.copy()
        variant = copy_idx % 5
        if variant == 0:
            scale = rng.uniform(1.5, 3.0, size=(len(x), 1, 1)).astype(np.float32)
            x = x + rng.normal(0.0, 0.8, size=x.shape).astype(np.float32) * scale
        elif variant == 1:
            idx = rng.integers(0, x.shape[1], size=len(x))
            spike = rng.choice([-1.0, 1.0], size=len(x)).astype(np.float32) * rng.uniform(8.0, 16.0, size=len(x))
            x[np.arange(len(x)), idx, 0] += spike.astype(np.float32)
        elif variant == 2:
            cut = x.shape[1] // 2
            x[:, :cut] = x[:, :cut][:, ::-1]
        elif variant == 3:
            t = np.arange(x.shape[1], dtype=np.float32)[None, :, None]
            amp = rng.uniform(4.0, 10.0, size=(len(x), 1, 1)).astype(np.float32)
            x = x + amp * np.sin(2.0 * np.pi * t / 6.0)
        else:
            x = x[:, ::-1].copy()
        out.append(np.clip(x, -2.0 * jerk_abs_max, 2.0 * jerk_abs_max).astype(np.float32))
    if not out:
        return np.zeros((0, *base.shape[1:]), dtype=np.float32)
    return np.concatenate(out, axis=0)


def _ax_profile_to_actions(ax: np.ndarray, prev_ax: np.ndarray, schema: dict, config: dict) -> np.ndarray:
    rep = str(schema.get("action_representation", config.get("action", {}).get("representation", "jerk"))).lower()
    dt = float(schema.get("dt", 0.04))
    action_cfg = config.get("action", {})
    if rep == "acceleration":
        return ax[:, :, None].astype(np.float32)
    if rep != "jerk":
        raise ValueError(f"Unsupported action representation: {rep}")
    prev = np.concatenate([prev_ax[:, None], ax[:, :-1]], axis=1)
    jx = (ax - prev) / max(dt, 1e-6)
    jerk_abs_max = float(action_cfg.get("jerk_abs_max", 12.0))
    return np.clip(jx, -2.0 * jerk_abs_max, 2.0 * jerk_abs_max).astype(np.float32)[:, :, None]


def generate_rule_brake_negatives(
    actions: np.ndarray,
    context_states: np.ndarray,
    *,
    rng: np.random.Generator,
    schema: dict,
    config: dict,
    copies_per_positive: int = 1,
) -> np.ndarray:
    """Generate rule-like hard braking futures."""
    base = _as_actions(actions)
    context_states = np.asarray(context_states, dtype=np.float32)
    h = base.shape[1]
    prev_ax = context_states[:, -1, 1, 4].astype(np.float32)
    out: list[np.ndarray] = []
    ax_min = float(config.get("action", {}).get("ax_min", -8.0))
    for copy_idx in range(max(0, int(copies_per_positive))):
        mode = copy_idx % 4
        target = np.zeros((len(base), h), dtype=np.float32)
        if mode == 0:
            target[:] = rng.uniform(ax_min, min(-4.0, ax_min + 2.0), size=(len(base), 1)).astype(np.float32)
        elif mode == 1:
            pulse_start = rng.integers(2, max(3, h // 2), size=len(base))
            pulse_len = rng.integers(5, max(6, h // 2), size=len(base))
            for i in range(len(base)):
                end = min(h, int(pulse_start[i] + pulse_len[i]))
                target[i, pulse_start[i]:end] = rng.uniform(ax_min, -5.0)
        elif mode == 2:
            start = prev_ax[:, None]
            end = rng.uniform(ax_min, -4.5, size=(len(base), 1)).astype(np.float32)
            target = np.linspace(0.0, 1.0, h, dtype=np.float32)[None, :] * (end - start) + start
        else:
            target[:] = rng.uniform(ax_min * 1.15, ax_min * 0.9, size=(len(base), 1)).astype(np.float32)
        out.append(_ax_profile_to_actions(target.astype(np.float32), prev_ax, schema, config))
    if not out:
        return np.zeros((0, *base.shape[1:]), dtype=np.float32)
    return np.concatenate(out, axis=0)


def load_external_negatives(path: str | Path | None) -> dict[str, Any] | None:
    """Load externally mined hard negatives.

    Expected fields are ``actions`` and optionally metadata fields such as
    ``sample_index`` or ``source_type``. Returning ``None`` keeps the first
    implementation usable without Stage 3 or highway-env generators.
    """
    if path is None or str(path) == "":
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"External negative file does not exist: {p}")
    data = np.load(p, allow_pickle=True)
    if "actions" not in data.files:
        raise ValueError(f"External negative file must contain an actions array: {p}")
    return {key: data[key] for key in data.files}

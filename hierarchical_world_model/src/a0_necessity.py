"""Role-stratified A0 intervention diagnostic.

This module is deliberately diagnostic-only.  It evaluates the released HiQR
response controller before adding a human-prior or reinforcement-learning
component.  Role labels may use logged future motion for *offline strata
selection*, but no such label or future value is passed to the model.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

from diffusion.src.data import ANCHOR_INDEX
from tools.plot_style import GENERATED_COLOR, REAL_COLOR, get_pyplot, style_axes
from world_model.src.core.evaluation_scope import (
    evaluation_scope_contract,
    scoped_canonical_trajectory,
)
from world_model.src.core.utils import ensure_dir, file_sha256, save_json

from .calibration import evaluation_response_calibration
from .data import ego_controls, prepare_experiment_data
from .evaluation import Rollout, rollout
from .planner import frozen_diffusion_plans
from .protocol import canonical_hash
from .train import load_checkpoint


DT_S = 0.04
ONSET = 25
HORIZON = 20
ROLE_NAMES = (
    "same_lane_front",
    "cut_in_or_committed",
    "adjacent_rear_left",
    "adjacent_rear_right",
    "target_lane_follower",
)
DOSES = {
    "brake": (1.5, 2.25, 3.0),
    "accelerate": (1.0, 1.5, 2.0),
    "left": (0.08, 0.12, 0.16),
}


@dataclass(frozen=True)
class A0DiagnosticConfig:
    """Fixed, pre-registered diagnostic settings."""

    seed: int = 20260830
    batch_size: int = 32
    calibration_stride: int = 5
    per_cell_cap: int = 128
    bootstrap_replicates: int = 1_000
    confidence: float = 0.95


def role_masks(
    states: np.ndarray,
    valid: np.ndarray,
    *,
    time_index: int,
    future_frames: int = 50,
) -> dict[str, np.ndarray]:
    """Return offline diagnostic role masks with shape ``[batch, 6]``.

    ``cut_in_or_committed`` and ``target_lane_follower`` are labels derived
    from the logged continuation.  They are only used after a rollout to
    stratify results and cannot affect a generated action.
    """
    values, present = scoped_canonical_trajectory(states, valid)
    values = np.asarray(values, np.float32)
    present = np.asarray(present, bool)
    if values.ndim != 4 or values.shape[2:] != (7, 6):
        raise ValueError("states must be [batch,time,7,6]")
    if present.shape != values.shape[:3]:
        raise ValueError("valid must align with states")
    if not 0 <= int(time_index) < values.shape[1]:
        raise ValueError("time_index lies outside the sequence")

    current = values[:, int(time_index)]
    active = present[:, int(time_index), 1:] & present[:, int(time_index), :1]
    relative_x = current[:, 1:, 0] - current[:, :1, 0]
    relative_y = current[:, 1:, 1] - current[:, :1, 1]
    same_lane = np.abs(relative_y) <= 1.8
    adjacent_left = (relative_y > 1.8) & (relative_y <= 5.4)
    adjacent_right = (relative_y < -1.8) & (relative_y >= -5.4)
    rear = relative_x < -4.8

    stop = min(values.shape[1], int(time_index) + int(future_frames) + 1)
    future_relative_y = (
        values[:, int(time_index) : stop, 1:, 1]
        - values[:, int(time_index) : stop, :1, 1]
    )
    future_relative_x = (
        values[:, int(time_index) : stop, 1:, 0]
        - values[:, int(time_index) : stop, :1, 0]
    )
    future_active = present[:, int(time_index) : stop, 1:] & present[
        :, int(time_index) : stop, :1
    ]
    enters_ego_lane = (
        (np.abs(future_relative_y) <= 1.0)
        & (future_relative_x > -4.8)
        & future_active
    ).any(axis=1)
    committed = same_lane & (np.abs(current[:, 1:, 3]) >= 0.15)

    ego_lateral_change = (
        values[:, stop - 1, 0, 1] - values[:, int(time_index), 0, 1]
    )
    target_ego_y = current[:, :1, 1] + np.sign(ego_lateral_change)[:, None] * 3.6
    target_lane = np.abs(current[:, 1:, 1] - target_ego_y) <= 1.8
    target_lane_follower = (
        (np.abs(ego_lateral_change) >= 1.0)[:, None] & target_lane & rear
    )
    return {
        "same_lane_front": active & same_lane & (relative_x > 4.8),
        "cut_in_or_committed": active
        & ((~same_lane & enters_ego_lane) | committed),
        "adjacent_rear_left": active & adjacent_left & rear,
        "adjacent_rear_right": active & adjacent_right & rear,
        "target_lane_follower": active & target_lane_follower,
    }


def _feature_key(current: np.ndarray) -> np.ndarray:
    """Coarse train-only matching key for response calibration."""
    ego, background = current[:, :1], current[:, 1:]
    dx = background[..., 0] - ego[..., 0]
    dy = background[..., 1] - ego[..., 1]
    relative_speed = background[..., 2] - ego[..., 2]
    gap = np.abs(dx)
    return (
        np.clip(np.floor(gap / 10.0), 0, 20).astype(np.int16) * 10_000
        + np.clip(np.floor((relative_speed + 20.0) / 2.0), 0, 20).astype(np.int16)
        * 100
        + np.clip(np.floor(np.abs(dy) / 1.8), 0, 4).astype(np.int16) * 10
        + np.clip(np.floor((background[..., 4] + 8.0) / 2.0), 0, 9).astype(np.int16)
    )


def _append_capped(
    table: dict[int, list[float]], keys: np.ndarray, values: np.ndarray, cap: int
) -> None:
    for key, value in zip(np.asarray(keys).reshape(-1), np.asarray(values).reshape(-1)):
        bucket = table[int(key)]
        if len(bucket) < int(cap):
            bucket.append(float(value))


def training_role_envelopes(
    arrays: dict[str, np.ndarray],
    rows: np.ndarray,
    settings: A0DiagnosticConfig,
) -> dict[str, Any]:
    """Build role-conditioned train-only P10--P90 response envelopes.

    The estimator follows the existing calibration's neutral-context matching,
    but separates its samples by diagnostic role.  It is intentionally a
    coarse necessity test, not a learned human-response model.
    """
    states, valid = scoped_canonical_trajectory(
        np.asarray(arrays["agent_states"])[np.asarray(rows, np.int64)],
        np.asarray(arrays["agent_valid"])[np.asarray(rows, np.int64)],
    )
    states = np.asarray(states, np.float32)
    valid = np.asarray(valid, bool)
    neutral: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    treated: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    last = states.shape[1] - HORIZON - 1
    for time_index in range(ANCHOR_INDEX, last, settings.calibration_stride):
        current = states[:, time_index]
        future = states[:, time_index + HORIZON]
        controls = ego_controls(
            states[:, time_index, 0], states[:, time_index + 1, 0], DT_S
        )
        longitudinal = (future[:, 1:, 2] - current[:, 1:, 2]) / (HORIZON * DT_S)
        separation = (
            np.abs(future[:, 1:, 1] - future[:, :1, 1])
            - np.abs(current[:, 1:, 1] - current[:, :1, 1])
        ) / (HORIZON * DT_S)
        keys = _feature_key(current)
        labels = role_masks(states, valid, time_index=time_index, future_frames=HORIZON)
        neutral_long = np.abs(controls[:, 0]) < 0.25
        neutral_left = np.abs(controls[:, 1]) < 0.02
        selections = {
            "brake": (controls[:, 0] <= -0.75, neutral_long, -1.0, longitudinal),
            "accelerate": (controls[:, 0] >= 0.75, neutral_long, 1.0, longitudinal),
            "left": (np.abs(controls[:, 1]) >= 0.05, neutral_left, 1.0, separation),
        }
        for role, role_mask in labels.items():
            for kind, (event, control, sign, outcome) in selections.items():
                group = (role, kind)
                event_mask = role_mask & event[:, None]
                neutral_mask = role_mask & control[:, None]
                _append_capped(
                    treated[group], keys[event_mask], sign * outcome[event_mask], settings.per_cell_cap
                )
                _append_capped(
                    neutral[group], keys[neutral_mask], outcome[neutral_mask], settings.per_cell_cap
                )
    result: dict[str, Any] = {}
    for role in ROLE_NAMES:
        result[role] = {}
        for kind in DOSES:
            group = (role, kind)
            sign = -1.0 if kind == "brake" else 1.0
            effects: list[float] = []
            for key, values in treated[group].items():
                controls = neutral[group].get(key, [])
                if len(controls) >= 5:
                    effects.extend(sign * (np.asarray(values) - np.median(controls)))
            usable = np.asarray(effects, np.float32)
            usable = usable[np.isfinite(usable)]
            if kind != "left":
                usable = usable[usable >= 0.0]
            result[role][kind] = {
                "reference_split": "train",
                "matched_effects": int(len(usable)),
                "p10_p50_p90": (
                    np.quantile(usable, (0.1, 0.5, 0.9)).tolist()
                    if len(usable)
                    else None
                ),
                "method": "role_conditioned_neutral_context_matching",
            }
    return result


def _bootstrap(values: np.ndarray, *, seed: int, settings: A0DiagnosticConfig) -> dict[str, float] | None:
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    generator = np.random.default_rng(int(seed))
    draws = generator.integers(0, len(values), size=(settings.bootstrap_replicates, len(values)))
    means = values[draws].mean(axis=1)
    tail = (1.0 - settings.confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci_low": float(np.quantile(means, tail)),
        "ci_high": float(np.quantile(means, 1.0 - tail)),
        "samples": int(len(values)),
    }


def _adjust_bh(p_values: dict[str, float]) -> dict[str, float]:
    """Benjamini--Hochberg correction without an optional dependency."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = max(len(ordered), 1)
    adjusted: dict[str, float] = {}
    running = 1.0
    for rank, (name, value) in reversed(list(enumerate(ordered, start=1))):
        running = min(running, float(value) * count / rank)
        adjusted[name] = float(min(running, 1.0))
    return adjusted


def _min_gap_and_collision(states: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ego, background = states[:, :, :1], states[:, :, 1:]
    dx = background[..., 0] - ego[..., 0]
    dy = np.abs(background[..., 1] - ego[..., 1])
    selected = np.broadcast_to(mask[:, None], dx.shape)
    gaps = np.where(selected & (dx > 0.0) & (dy <= 1.8), np.maximum(dx - 4.8, 0.0), np.inf)
    collision = selected & (np.abs(dx) < 4.8) & (dy < 1.8)
    return gaps.min(axis=(1, 2)), collision.any(axis=(1, 2))


def _role_values(
    baseline: Rollout,
    mild: Rollout,
    strong: Rollout,
    logged: np.ndarray,
    role_mask: np.ndarray,
    kind: str,
    reference: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Compute one cluster-level value per context for a role and intervention."""
    active = np.asarray(role_mask, bool)
    expected = -1.0 if kind == "brake" else 1.0
    after = slice(ONSET + 1, ONSET + HORIZON + 1)
    mild_delta = mild.background_actions[:, after] - baseline.background_actions[:, after]
    strong_delta = strong.background_actions[:, after] - baseline.background_actions[:, after]
    action_change = np.linalg.norm(mild_delta, axis=-1)
    count = active.sum(axis=1)
    valid_context = count > 0
    mean_action = np.divide(
        (action_change * active[:, None]).sum(axis=(1, 2)),
        (count * action_change.shape[1]).clip(1),
    )
    other = ~active
    other_count = other.sum(axis=1)
    other_action = np.divide(
        (action_change * other[:, None]).sum(axis=(1, 2)),
        (other_count * action_change.shape[1]).clip(1),
    )
    result: dict[str, np.ndarray] = {
        "valid": valid_context,
        "locality_difference": mean_action - other_action,
        "min_gap_m": _min_gap_and_collision(mild.states, active)[0],
        "collision": _min_gap_and_collision(mild.states, active)[1].astype(float),
        "invalid": (~np.isfinite(mild.states).all(axis=(1, 2, 3))).astype(float),
    }
    target = logged[:, ANCHOR_INDEX + 1 : ANCHOR_INDEX + 150]
    error = np.linalg.norm(baseline.states[..., 1:, :2] - target[..., 1:, :2], axis=-1)
    result["factual_ADE_m"] = np.divide(
        (error * active[:, None]).sum(axis=(1, 2)),
        (count * error.shape[1]).clip(1),
    )
    peak = action_change.max(axis=(1, 2))
    threshold = np.maximum(peak * 0.05, 1.0e-6)
    detected = action_change.mean(axis=2) > threshold[:, None]
    first = np.argmax(detected, axis=1)
    result["latency_s"] = np.where(
        detected.any(axis=1), (first + 1) * DT_S, np.nan
    )
    if kind in {"brake", "accelerate"}:
        effect = expected * (
            mild.states[:, ONSET + HORIZON, 1:, 2]
            - baseline.states[:, ONSET + HORIZON, 1:, 2]
        ) / (HORIZON * DT_S)
        strong_effect = expected * (
            strong.states[:, ONSET + HORIZON, 1:, 2]
            - baseline.states[:, ONSET + HORIZON, 1:, 2]
        ) / (HORIZON * DT_S)
        role_effect = np.divide((effect * active).sum(1), count.clip(1))
        role_strong = np.divide((strong_effect * active).sum(1), count.clip(1))
        p10_p50_p90 = reference.get("p10_p50_p90")
        result["effect_mps2"] = role_effect
        result["direction"] = (role_effect > 0.0).astype(float)
        result["dose_monotone"] = (role_strong > role_effect).astype(float)
        if p10_p50_p90 is not None:
            low, middle, high = (float(value) for value in p10_p50_p90)
            width = max(high - low, 1.0e-6)
            result["within_envelope"] = (
                (role_effect >= low) & (role_effect <= high)
            ).astype(float)
            result["outside_envelope_widths"] = np.maximum(
                np.maximum(low - role_effect, role_effect - high), 0.0
            ) / width
            result["effect_minus_human_median_mps2"] = role_effect - middle
    else:
        base_sep = np.abs(
            baseline.states[:, ONSET + HORIZON, 1:, 1]
            - baseline.states[:, ONSET + HORIZON, :1, 1]
        )
        mild_sep = np.abs(
            mild.states[:, ONSET + HORIZON, 1:, 1]
            - mild.states[:, ONSET + HORIZON, :1, 1]
        )
        separation = np.divide(((mild_sep - base_sep) * active).sum(1), count.clip(1))
        result["lateral_separation_change_m"] = separation
        result["direction"] = (separation >= 0.0).astype(float)
    for name, value in tuple(result.items()):
        result[name] = np.asarray(value)[valid_context]
    return result


def _summarize_role(
    values: dict[str, list[np.ndarray]], *, seed: int, settings: A0DiagnosticConfig
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, fragments in values.items():
        joined = np.concatenate(fragments) if fragments else np.empty(0, np.float32)
        summary[name] = _bootstrap(joined, seed=seed + len(name), settings=settings)
    return summary


def _write_figures(report: dict[str, Any], output: Path) -> list[str]:
    figures = ensure_dir(output / "figures")
    plt = get_pyplot()
    roles = list(ROLE_NAMES)
    role_counts = [report["roles"][role]["contexts"] for role in roles]
    figure, axis = plt.subplots(figsize=(10.5, 4.0), constrained_layout=True)
    axis.bar(np.arange(len(roles)), role_counts, color=REAL_COLOR)
    axis.set(
        title="A0 held-out intervention contexts by diagnostic role",
        ylabel="Contexts with at least one valid background",
        xticks=np.arange(len(roles)),
        xticklabels=[name.replace("_", "\n") for name in roles],
    )
    style_axes(axis)
    path = figures / "role_context_counts.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)

    figure, axes = plt.subplots(2, len(roles), figsize=(18.0, 6.2), constrained_layout=True)
    for column, role in enumerate(roles):
        for row, kind in enumerate(("brake", "accelerate")):
            axis = axes[row, column]
            item = report["roles"][role][kind]
            reference = item["human_envelope"]
            if reference["p10_p50_p90"] is not None:
                low, middle, high = reference["p10_p50_p90"]
                axis.axhspan(low, high, color=REAL_COLOR, alpha=0.18, label="train P10-P90")
                axis.axhline(middle, color=REAL_COLOR, linestyle=":", linewidth=1.0)
            effect = item["statistics"].get("effect_mps2")
            if effect is not None:
                axis.errorbar(
                    [0], [effect["mean"]],
                    yerr=[[effect["mean"] - effect["ci_low"]], [effect["ci_high"] - effect["mean"]]],
                    fmt="o", color=GENERATED_COLOR, label="A0 bootstrap CI",
                )
            axis.set(title=f"{role}: {kind}", xlim=(-0.7, 0.7), xticks=[])
            style_axes(axis)
    axes[0, 0].legend(frameon=False, fontsize=7)
    path2 = figures / "human_envelope_vs_a0.png"
    figure.savefig(path2, dpi=300)
    plt.close(figure)

    metrics = ("direction", "dose_monotone", "latency_s", "locality_difference")
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), constrained_layout=True)
    for axis, metric in zip(axes.flat, metrics):
        labels, means, lows, highs = [], [], [], []
        for role in roles:
            for kind in DOSES:
                value = report["roles"][role][kind]["statistics"].get(metric)
                if value is None:
                    continue
                labels.append(f"{role}\n{kind}")
                means.append(value["mean"])
                lows.append(value["mean"] - value["ci_low"])
                highs.append(value["ci_high"] - value["mean"])
        if means:
            position = np.arange(len(means))
            axis.errorbar(position, means, yerr=np.asarray((lows, highs)), fmt="o", color=GENERATED_COLOR)
            axis.set(xticks=position, xticklabels=labels, title=metric.replace("_", " "))
            axis.tick_params(axis="x", rotation=65, labelsize=7)
        style_axes(axis)
    path3 = figures / "a0_response_bootstrap.png"
    figure.savefig(path3, dpi=300)
    plt.close(figure)
    return [str(path), str(path2), str(path3)]


def _draw_playback(
    path: Path,
    baseline: Rollout,
    treatment: Rollout,
    role: str,
    kind: str,
    active: np.ndarray,
) -> None:
    """Render a compact factual-versus-intervention causal replay GIF."""
    from hierarchical_world_model.src.visualization import _draw_lane_markings, _draw_vehicle

    plt = get_pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), dpi=100)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, mode="I", duration=80, loop=0) as writer:
        for frame in range(0, baseline.states.shape[1], 2):
            for axis, values, title in zip(axes, (baseline.states[0], treatment.states[0]), ("factual", kind)):
                axis.clear()
                ego_origin = values[frame, 0, :2]
                local = values[frame].copy()
                local[:, :2] -= ego_origin
                axis.set_facecolor("#6f7378")
                _draw_lane_markings(axis)
                for slot in range(1, 7):
                    if active[slot - 1]:
                        _draw_vehicle(axis, local[slot], color=GENERATED_COLOR, label=f"b{slot}", filled=True, alpha=0.65)
                _draw_vehicle(axis, local[0], color="#D62728", label="ego", filled=True, alpha=0.9)
                axis.set(
                    xlim=(-55, 55), ylim=(-8.2, 8.2), aspect="equal",
                    title=f"{role} | {title} | t={frame * DT_S:.2f}s",
                    xlabel="ego-relative x [m]",
                )
                axis.tick_params(labelsize=7)
            figure.canvas.draw()
            writer.append_data(np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(figure)


def _write_playbacks(
    *,
    model: Any,
    states: np.ndarray,
    valid: np.ndarray,
    maps: np.ndarray,
    map_valid: np.ndarray,
    diffusion: np.ndarray,
    rows: np.ndarray,
    role_masks_all: dict[str, np.ndarray],
    output: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    playbacks = ensure_dir(output / "playbacks")
    records: list[dict[str, Any]] = []
    for role in ROLE_NAMES:
        candidates = np.flatnonzero(role_masks_all[role].any(axis=1))
        if not len(candidates):
            continue
        index = int(candidates[0])
        for kind, doses in DOSES.items():
            baseline = rollout(
                model, states[index:index + 1], valid[index:index + 1], diffusion[index:index + 1],
                maps[index:index + 1], map_valid[index:index + 1], device=device,
                history_frames=25, motion_seed=None,
            )
            treatment = rollout(
                model, states[index:index + 1], valid[index:index + 1], diffusion[index:index + 1],
                maps[index:index + 1], map_valid[index:index + 1], device=device,
                history_frames=25, motion_seed=None, intervention=kind, dose=doses[-1],
            )
            archive = playbacks / f"{role}_{kind}.npz"
            np.savez_compressed(
                archive,
                row_index=np.asarray(int(rows[index]), np.int64),
                role_mask=role_masks_all[role][index],
                baseline_states=baseline.states,
                treatment_states=treatment.states,
                baseline_actions=baseline.background_actions,
                treatment_actions=treatment.background_actions,
            )
            gif = playbacks / f"{role}_{kind}.gif"
            _draw_playback(gif, baseline, treatment, role, kind, role_masks_all[role][index])
            records.append({
                "role": role,
                "intervention": kind,
                "dose": float(doses[-1]),
                "context_row": int(rows[index]),
                "archive": str(archive),
                "gif": str(gif),
                "randomness": "deterministic A0 mean action; paired branches share all inputs",
            })
    return records


def run_a0_necessity_diagnostic(
    config: dict[str, Any],
    *,
    config_dir: Path,
    output: Path,
    maximum_contexts: int = 0,
) -> dict[str, Any]:
    """Run the full A0 gate without changing the released world model."""
    settings = A0DiagnosticConfig()
    output = ensure_dir(output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment = prepare_experiment_data(config, config_dir)
    rows = np.asarray(experiment.test_rows, np.int64)
    if int(maximum_contexts) > 0:
        rows = rows[: int(maximum_contexts)]
    model, checkpoint = load_checkpoint(config["paths"]["evaluation_checkpoint"], device=device)
    model.eval()
    states = np.asarray(experiment.bundle.arrays["agent_states"][rows], np.float32)
    valid = np.asarray(experiment.bundle.arrays["agent_valid"][rows], bool)
    states, valid = scoped_canonical_trajectory(states, valid)
    states, valid = np.asarray(states, np.float32), np.asarray(valid, bool)
    maps = np.asarray(experiment.bundle.arrays["map_polylines"][rows], np.float32)
    map_valid = np.asarray(experiment.bundle.arrays["map_polyline_valid"][rows], bool)
    envelope = training_role_envelopes(experiment.bundle.arrays, experiment.train_rows, settings)
    reference_time = ANCHOR_INDEX + ONSET
    labels = role_masks(states, valid, time_index=reference_time, future_frames=50)
    diffusion = frozen_diffusion_plans(
        experiment.bundle, rows, checkpoint=config["paths"]["diffusion_checkpoint"],
        output_dir=output / "diffusion_cache", device=device, batch_size=settings.batch_size,
        ddim_steps=20, experiment_scope="full",
    )
    raw: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
        role: {kind: defaultdict(list) for kind in DOSES} for role in ROLE_NAMES
    }
    for start in range(0, len(rows), settings.batch_size):
        stop = min(start + settings.batch_size, len(rows))
        batch_seed = settings.seed + start
        baseline = rollout(
            model, states[start:stop], valid[start:stop], diffusion[start:stop], maps[start:stop], map_valid[start:stop],
            device=device, history_frames=25, motion_seed=batch_seed,
        )
        for kind, doses in DOSES.items():
            mild = rollout(
                model, states[start:stop], valid[start:stop], diffusion[start:stop], maps[start:stop], map_valid[start:stop],
                device=device, history_frames=25, motion_seed=batch_seed, intervention=kind, dose=doses[0],
            )
            strong = rollout(
                model, states[start:stop], valid[start:stop], diffusion[start:stop], maps[start:stop], map_valid[start:stop],
                device=device, history_frames=25, motion_seed=batch_seed, intervention=kind, dose=doses[-1],
            )
            for role in ROLE_NAMES:
                values = _role_values(
                    baseline, mild, strong, states[start:stop], labels[role][start:stop], kind,
                    envelope[role][kind],
                )
                for name, value in values.items():
                    raw[role][kind][name].append(value)
    p_values: dict[str, float] = {}
    roles: dict[str, Any] = {}
    for role in ROLE_NAMES:
        roles[role] = {"contexts": int(labels[role].any(axis=1).sum())}
        for kind in DOSES:
            statistics = _summarize_role(raw[role][kind], seed=settings.seed, settings=settings)
            item = {
                "human_envelope": envelope[role][kind],
                "statistics": statistics,
            }
            coverage = statistics.get("within_envelope")
            outside = statistics.get("outside_envelope_widths")
            if coverage is not None and outside is not None:
                # Bootstrap probability for the data-derived nominal P10--P90
                # coverage.  The final decision also requires a full-band
                # magnitude deficit, avoiding a significance-only conclusion.
                p_values[f"{role}/{kind}"] = float(
                    np.mean(np.random.default_rng(settings.seed + len(role)).normal(
                        coverage["mean"], max((coverage["ci_high"] - coverage["ci_low"]) / 3.92, 1.0e-6),
                        settings.bootstrap_replicates,
                    ) >= 0.8)
                )
            roles[role][kind] = item
    adjusted = _adjust_bh(p_values)
    evidence: list[dict[str, Any]] = []
    for name, adjusted_p in adjusted.items():
        role, kind = name.split("/")
        stats = roles[role][kind]["statistics"]
        coverage, outside = stats.get("within_envelope"), stats.get("outside_envelope_widths")
        if coverage is None or outside is None:
            continue
        gap = bool(adjusted_p < 0.05 and coverage["ci_high"] < 0.8 and outside["ci_low"] > 1.0)
        roles[role][kind]["envelope_gap_test"] = {
            "nominal_train_p10_p90_coverage": 0.8,
            "bh_adjusted_p_value": adjusted_p,
            "coverage_deficit": bool(coverage["ci_high"] < 0.8),
            "full_band_error": bool(outside["ci_low"] > 1.0),
            "evidence_of_material_gap": gap,
        }
        if gap:
            evidence.append({"role": role, "intervention": kind})
    report: dict[str, Any] = {
        "schema": "a0_role_stratified_necessity_v1",
        "decision": {
            "gail_ppo_necessary_in_current_scope": bool(evidence),
            "next_step": "implement A2 only" if evidence else "stop: retain A0 and do not add GAIL/PPO",
            "evidence": evidence,
            "rule": "requires FDR-corrected envelope deficit and bootstrap lower CI of outside-envelope error greater than one train P10-P90 band width",
        },
        "evaluation_scope": evaluation_scope_contract(),
        "test_space": "empirical_test_fixed_k_gt",
        "contexts": int(len(rows)),
        "calibration": {"source_split": "train", "roles": envelope},
        "roles": roles,
        "settings": settings.__dict__,
        "provenance": {
            "checkpoint": str(config["paths"]["evaluation_checkpoint"]),
            "checkpoint_sha256": file_sha256(config["paths"]["evaluation_checkpoint"]),
            "config_hash": canonical_hash(config),
            "model_epoch": int(checkpoint["epoch"]),
            "role_labels": "offline held-out strata only; never model inputs",
        },
    }
    report["figures"] = _write_figures(report, output)
    report["playbacks"] = _write_playbacks(
        model=model, states=states, valid=valid, maps=maps, map_valid=map_valid,
        diffusion=diffusion, rows=rows, role_masks_all=labels, output=output, device=device,
    )
    save_json(report, output / "a0_necessity_summary.json")
    save_json(
        {
            "schema": "a0_necessity_playbacks_v1",
            "evaluation_scope": evaluation_scope_contract(),
            "episodes": report["playbacks"],
            "provenance": report["provenance"],
        },
        output / "playbacks" / "playback_manifest.json",
    )
    return report

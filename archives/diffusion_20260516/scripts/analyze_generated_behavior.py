#!/usr/bin/env python3
"""Analyze generated action naturalness, risk control, and feasibility."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diffusion.scripts.evaluate_action_diffusion import (
    _actions_to_ax,
    _decode_actions,
    _encode_risk_condition,
    _jerk_from_ax,
    _resolve_checkpoint_path,
    _resolve_output_dir,
)
from diffusion.src.data import SPLIT_TO_INDEX, load_normalized_dataset
from diffusion.src.kinematics import integrate_following_actions
from diffusion.src.model import build_model_from_schema
from diffusion.src.risk import constant_velocity_rollout, score_future_risk
from diffusion.src.types import VehicleBox, VehicleState
from diffusion.src.utils import load_json, load_yaml, save_json, select_device, set_seed, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "diffusion_following.yaml"
DEFAULT_CHECKPOINTS = "checkpoints/best.pt"
DEFAULT_SPLIT = "val"
DEFAULT_NUM_CONTEXTS = 256
DEFAULT_SAMPLES_PER_CONTEXT = 8
DEFAULT_RISK_LEVELS = "0.50,0.80,0.90,0.95"
DEFAULT_GUIDANCE_SCALES = "1.0,1.5,2.0"
DEFAULT_SEED = 42
DEFAULT_SAMPLE_BATCH_SIZE = 256
DEFAULT_DEVICE = None
DEFAULT_LOG_LEVEL = "INFO"
logger = logging.getLogger(__name__)


def _parse_float_list(value: str | None, default: list[float]) -> list[float]:
    if value is None or not str(value).strip():
        return default
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _parse_checkpoints(args: argparse.Namespace) -> list[str]:
    values: list[str] = []
    checkpoints_arg = str(args.checkpoints or "").strip()
    if checkpoints_arg and (checkpoints_arg != DEFAULT_CHECKPOINTS or not args.checkpoint):
        values.extend(part.strip() for part in checkpoints_arg.split(",") if part.strip())
    if args.checkpoint:
        values.extend(args.checkpoint)
    return values or [DEFAULT_CHECKPOINTS]


def _checkpoint_label(path: Path) -> str:
    name = path.name
    return name[:-3] if name.endswith(".pt") else path.stem


def _summarize(values: np.ndarray) -> dict[str, float | int | None]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"count": 0, "mean": None, "std": None, "p05": None, "p50": None, "p95": None}
    q = np.quantile(x, [0.05, 0.50, 0.95])
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p05": float(q[0]),
        "p50": float(q[1]),
        "p95": float(q[2]),
    }


def _hist_l1(a: np.ndarray, b: np.ndarray, bins: int = 80) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    if hi <= lo:
        return 0.0
    hx, edges = np.histogram(x, bins=bins, range=(lo, hi), density=True)
    hy, _ = np.histogram(y, bins=edges, density=True)
    widths = np.diff(edges)
    return float(np.sum(np.abs(hx - hy) * widths))


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    values = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, values, side="right") / len(x)
    cdf_y = np.searchsorted(y, values, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _wasserstein_1d(a: np.ndarray, b: np.ndarray, quantiles: int = 1024) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    q = np.linspace(0.0, 1.0, int(quantiles))
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def _load_model(checkpoint_path: Path, schema: dict, config: dict, device: torch.device):
    model = build_model_from_schema(schema, config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


@torch.no_grad()
def _sample_actions(
    model,
    arrays: dict[str, np.ndarray],
    context_idx: np.ndarray,
    risk_condition: np.ndarray,
    samples_per_context: int,
    device: torch.device,
    guidance_scale: float,
    batch_size: int,
) -> np.ndarray:
    flat_idx = np.repeat(context_idx, int(samples_per_context))
    flat_risk = np.repeat(np.asarray(risk_condition, dtype=np.float32), int(samples_per_context), axis=0)
    out: list[np.ndarray] = []
    for start in range(0, len(flat_idx), int(batch_size)):
        end = min(start + int(batch_size), len(flat_idx))
        idx = flat_idx[start:end]
        history = torch.from_numpy(arrays["context_states"][idx]).float().to(device)
        context = torch.from_numpy(arrays["context_features"][idx]).float().to(device)
        relative = torch.from_numpy(arrays["relative_history"][idx]).float().to(device)
        risk = torch.from_numpy(flat_risk[start:end]).float().to(device)
        sample = model.sample(len(idx), history, context, relative, risk, guidance_scale=guidance_scale)
        out.append(sample.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def _trajectory_details(
    ax: np.ndarray,
    context_states: np.ndarray,
    meta: dict[str, np.ndarray],
    schema: dict,
    config: dict,
) -> dict[str, np.ndarray]:
    dt = float(schema.get("dt", 0.04))
    risks: list[float] = []
    min_gaps: list[float] = []
    final_gaps: list[float] = []
    final_speeds: list[float] = []
    neg_gap: list[float] = []
    collision_like: list[float] = []
    speed_neg: list[float] = []
    stop: list[float] = []
    gap_profiles: list[np.ndarray] = []
    speed_profiles: list[np.ndarray] = []
    for i in range(ax.shape[0]):
        lead0 = context_states[i, -1, 1]
        ego0 = context_states[i, -1, 0]
        adv_len = float(meta["adv_length"][i])
        ego_len = float(meta["ego_length"][i])
        lane_w = float(meta["lane_width"][i])
        lead_state = VehicleState(
            x=float(lead0[0]),
            y=float(lead0[1]),
            vx=float(lead0[2]),
            vy=float(lead0[3]),
            ax=float(lead0[4]),
            ay=float(lead0[5]),
            box=VehicleBox(length=adv_len),
        )
        adv = integrate_following_actions(lead_state, ax[i, :, None], dt)[1:]
        ego = constant_velocity_rollout(ego0, ax.shape[1], dt)[1:]
        gaps = adv[:, 0] - ego[:, 0] - 0.5 * (ego_len + adv_len)
        vx = adv[:, 2]
        risks.append(score_future_risk("following", ego, adv, ego_len, adv_len, lane_w, config.get("risk", {})))
        min_gaps.append(float(np.min(gaps)))
        final_gaps.append(float(gaps[-1]))
        final_speeds.append(float(vx[-1]))
        neg_gap.append(float(np.any(gaps < 0.0)))
        collision_like.append(float(np.min(gaps) < 0.2))
        speed_neg.append(float(np.any(vx < -1e-6)))
        stop.append(float(vx[-1] < 0.1))
        gap_profiles.append(gaps.astype(np.float32))
        speed_profiles.append(vx.astype(np.float32))
    return {
        "risk": np.asarray(risks, dtype=np.float32),
        "min_gap": np.asarray(min_gaps, dtype=np.float32),
        "final_gap": np.asarray(final_gaps, dtype=np.float32),
        "lead_final_speed": np.asarray(final_speeds, dtype=np.float32),
        "negative_gap": np.asarray(neg_gap, dtype=np.float32),
        "collision_like": np.asarray(collision_like, dtype=np.float32),
        "lead_speed_negative": np.asarray(speed_neg, dtype=np.float32),
        "lead_stop": np.asarray(stop, dtype=np.float32),
        "gap_profiles": np.stack(gap_profiles, axis=0),
        "speed_profiles": np.stack(speed_profiles, axis=0),
    }


def _feasibility_summary(details: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "min_gap_mean": float(np.mean(details["min_gap"])),
        "min_gap_p05": float(np.quantile(details["min_gap"], 0.05)),
        "negative_gap_rate": float(np.mean(details["negative_gap"])),
        "collision_like_rate": float(np.mean(details["collision_like"])),
        "final_gap_mean": float(np.mean(details["final_gap"])),
        "lead_final_speed_mean": float(np.mean(details["lead_final_speed"])),
        "lead_speed_negative_rate": float(np.mean(details["lead_speed_negative"])),
        "lead_stop_ratio": float(np.mean(details["lead_stop"])),
    }


def _naturality_metrics(
    real_ax: np.ndarray,
    gen_ax: np.ndarray,
    gen_ax_unclipped: np.ndarray,
    real_context: np.ndarray,
    gen_context: np.ndarray,
    schema: dict,
    config: dict,
) -> dict[str, Any]:
    dt = float(schema.get("dt", 0.04))
    ax_min = float(config.get("action", {}).get("ax_min", -8.0))
    ax_max = float(config.get("action", {}).get("ax_max", 4.0))
    jerk_limit = float(config.get("action", {}).get("jerk_abs_max", 12.0))
    real_jerk = _jerk_from_ax(real_ax, real_context, dt)
    gen_jerk = _jerk_from_ax(gen_ax, gen_context, dt)
    return {
        "real_ax": _summarize(real_ax),
        "generated_ax": _summarize(gen_ax),
        "real_jerk": _summarize(real_jerk),
        "generated_jerk": _summarize(gen_jerk),
        "hard_brake_ratio": float(np.mean(gen_ax < -3.0)),
        "strong_brake_ratio": float(np.mean(gen_ax < -5.0)),
        "action_clip_rate": float(np.mean((gen_ax_unclipped < ax_min) | (gen_ax_unclipped > ax_max))),
        "speed_negative_rate": None,
        "jerk_violation_rate": float(np.mean(np.abs(gen_jerk) > jerk_limit)),
        "distance": {
            "wasserstein_ax": _wasserstein_1d(real_ax, gen_ax),
            "wasserstein_jerk": _wasserstein_1d(real_jerk, gen_jerk),
            "ks_ax": _ks_statistic(real_ax, gen_ax),
            "ks_jerk": _ks_statistic(real_jerk, gen_jerk),
            "hist_l1_ax": _hist_l1(real_ax, gen_ax),
            "hist_l1_jerk": _hist_l1(real_jerk, gen_jerk),
        },
    }


def _risk_control_summary(
    risk_by_level: dict[str, np.ndarray],
    levels: list[float],
    samples_per_context: int,
) -> dict[str, Any]:
    level_keys = [f"p{int(round(level * 100))}" for level in levels]
    means = {key: float(np.mean(risk_by_level[key])) for key in level_keys}
    medians = {key: float(np.median(risk_by_level[key])) for key in level_keys}
    p90 = {key: float(np.quantile(risk_by_level[key], 0.90)) for key in level_keys}
    per_context = []
    for key in level_keys:
        values = risk_by_level[key].reshape(-1, int(samples_per_context)).mean(axis=1)
        per_context.append(values)
    mat = np.stack(per_context, axis=1)
    monotonic_context_ratio = float(np.mean(np.all(np.diff(mat, axis=1) >= -1e-6, axis=1)))
    first = level_keys[0]
    last = level_keys[-1]
    return {
        "mean_generated_risk_by_level": means,
        "median_generated_risk_by_level": medians,
        "p90_generated_risk_by_level": p90,
        "monotonic_context_ratio": monotonic_context_ratio,
        "global_monotonic": bool(np.all(np.diff([means[key] for key in level_keys]) >= -1e-6)),
        f"risk_lift_{last}_vs_{first}": float(means[last] - means[first]),
    }


def _condition_sensitivity(normal: np.ndarray, zero: np.ndarray, shuffle: np.ndarray) -> dict[str, float]:
    normal_mean = float(np.mean(normal))
    zero_mean = float(np.mean(zero))
    shuffle_mean = float(np.mean(shuffle))
    return {
        "risk_mean_normal": normal_mean,
        "risk_mean_zero": zero_mean,
        "risk_mean_shuffle": shuffle_mean,
        "risk_lift_normal_vs_zero": normal_mean - zero_mean,
        "risk_lift_normal_vs_shuffle": normal_mean - shuffle_mean,
    }


def _make_plots(
    plot_dir: Path,
    plot_cache_dir: Path,
    real_ax: np.ndarray,
    gen_ax: np.ndarray,
    real_jerk: np.ndarray,
    gen_jerk: np.ndarray,
    summary_by_checkpoint: dict[str, Any],
    primary_label: str,
    levels: list[float],
) -> list[str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache_dir / "xdg"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("Matplotlib unavailable; skipping behavior plots: %s", exc)
        return []

    written: list[Path] = []

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.hist(real_ax.reshape(-1), bins=80, density=True, alpha=0.55, label="real")
    ax.hist(gen_ax.reshape(-1), bins=80, density=True, alpha=0.55, label="generated")
    ax.set_title("Acceleration Distribution")
    ax.set_xlabel("ax (m/s^2)")
    ax.set_ylabel("density")
    ax.legend()
    path = plot_dir / "ax_distribution_real_vs_generated.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.hist(real_jerk.reshape(-1), bins=80, density=True, alpha=0.55, label="real")
    ax.hist(gen_jerk.reshape(-1), bins=80, density=True, alpha=0.55, label="generated")
    ax.set_title("Jerk Distribution")
    ax.set_xlabel("jerk (m/s^3)")
    ax.set_ylabel("density")
    ax.legend()
    path = plot_dir / "jerk_distribution_real_vs_generated.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    level_labels = [f"p{int(round(level * 100))}" for level in levels]
    primary = summary_by_checkpoint[primary_label]
    primary_guidance = str(primary["primary_guidance_scale"])
    risk_curve = primary["by_guidance"][primary_guidance]["risk_control"]["mean_generated_risk_by_level"]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(levels, [risk_curve[label] for label in level_labels], marker="o")
    ax.set_title("Risk Control Curve")
    ax.set_xlabel("risk condition percentile")
    ax.set_ylabel("mean generated risk")
    ax.grid(True, alpha=0.3)
    path = plot_dir / "risk_control_curve.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for label, ckpt_summary in summary_by_checkpoint.items():
        xs = []
        ys = []
        for gs, gs_summary in ckpt_summary["by_guidance"].items():
            xs.append(float(gs))
            ys.append(gs_summary["risk_control"]["mean_generated_risk_by_level"][level_labels[-1]])
        order = np.argsort(xs)
        ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], marker="o", label=label)
    ax.set_title(f"Risk By Guidance Scale ({level_labels[-1]})")
    ax.set_xlabel("guidance scale")
    ax.set_ylabel("mean generated risk")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = plot_dir / "risk_by_guidance_scale.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    min_gap = primary["by_guidance"][primary_guidance]["min_gap_mean_by_level"]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(levels, [min_gap[label] for label in level_labels], marker="o")
    ax.set_title("Minimum Gap By Risk Level")
    ax.set_xlabel("risk condition percentile")
    ax.set_ylabel("mean min gap (m)")
    ax.grid(True, alpha=0.3)
    path = plot_dir / "min_gap_by_risk_level.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    sens = primary["by_guidance"][primary_guidance]["condition_sensitivity"]
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    names = ["normal", "zero", "shuffle"]
    values = [sens["risk_mean_normal"], sens["risk_mean_zero"], sens["risk_mean_shuffle"]]
    ax.bar(names, values)
    ax.set_title("Condition Ablation")
    ax.set_ylabel("mean generated risk")
    path = plot_dir / "condition_ablation_bar.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    examples = primary["example_top_risk_rollouts"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ex in examples:
        t = np.arange(len(ex["gap"]), dtype=np.float32)
        axes[0].plot(t, ex["gap"], alpha=0.75)
        axes[1].plot(t, ex["lead_speed"], alpha=0.75)
    axes[0].set_title("Top Risk Example Gaps")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("gap (m)")
    axes[1].set_title("Top Risk Example Lead Speeds")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("vx (m/s)")
    path = plot_dir / "example_rollouts_top_risk.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    return [str(p) for p in written]


def _analyze_checkpoint(
    checkpoint_path: Path,
    label: str,
    arrays: dict[str, np.ndarray],
    raw: dict[str, np.ndarray],
    schema: dict,
    stats: dict,
    config: dict,
    context_idx: np.ndarray,
    levels: list[float],
    guidance_scales: list[float],
    samples_per_context: int,
    sample_batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    model = _load_model(checkpoint_path, schema, config, device)
    train_raw = raw["risk_raw"][raw["split_index"] == SPLIT_TO_INDEX["train"]]
    raw_levels = np.quantile(train_raw, np.asarray(levels, dtype=np.float32))
    level_labels = [f"p{int(round(level * 100))}" for level in levels]
    real_context = raw["context_states"][context_idx]
    real_actions = raw["actions"][context_idx]
    real_ax, _ = _actions_to_ax(real_actions, real_context, schema, config)
    meta_context = {k: raw[k][context_idx] for k in ("ego_length", "adv_length", "lane_width")}

    primary_guidance = float(guidance_scales[0])
    by_guidance: dict[str, Any] = {}
    primary_details_for_examples = None

    for guidance_i, guidance in enumerate(guidance_scales):
        risk_by_level: dict[str, np.ndarray] = {}
        min_gap_by_level: dict[str, float] = {}
        feasibility_by_level: dict[str, Any] = {}
        for level_i, (level_label, raw_level) in enumerate(zip(level_labels, raw_levels)):
            cond = _encode_risk_condition(
                np.full((len(context_idx),), float(raw_level), dtype=np.float32),
                train_raw,
                config,
                stats,
            )
            set_seed(seed + guidance_i * 10_000 + level_i)
            gen_norm = _sample_actions(
                model,
                arrays,
                context_idx,
                cond,
                samples_per_context,
                device,
                guidance,
                sample_batch_size,
            )
            gen_actions = _decode_actions(gen_norm, stats)
            repeated_context = np.repeat(real_context, samples_per_context, axis=0)
            repeated_meta = {k: np.repeat(v, samples_per_context, axis=0) for k, v in meta_context.items()}
            gen_ax, gen_ax_unclipped = _actions_to_ax(gen_actions, repeated_context, schema, config)
            details = _trajectory_details(gen_ax, repeated_context, repeated_meta, schema, config)
            risk_by_level[level_label] = details["risk"]
            min_gap_by_level[level_label] = float(np.mean(details["min_gap"]))
            feasibility_by_level[level_label] = _feasibility_summary(details)
            if guidance == primary_guidance and level_label == level_labels[-1]:
                primary_details_for_examples = details

        risk_control = _risk_control_summary(risk_by_level, levels, samples_per_context)

        normal_cond = arrays["risk_condition"][context_idx]
        zero_cond = np.zeros_like(normal_cond)
        shuffle_cond = normal_cond.copy()
        np.random.default_rng(seed + guidance_i + 99_000).shuffle(shuffle_cond)
        ablation_risks: dict[str, np.ndarray] = {}
        for ablation_i, (name, cond) in enumerate({"normal": normal_cond, "zero": zero_cond, "shuffle": shuffle_cond}.items()):
            set_seed(seed + guidance_i * 10_000 + 5_000 + ablation_i)
            gen_norm = _sample_actions(
                model,
                arrays,
                context_idx,
                cond,
                samples_per_context,
                device,
                guidance,
                sample_batch_size,
            )
            gen_actions = _decode_actions(gen_norm, stats)
            repeated_context = np.repeat(real_context, samples_per_context, axis=0)
            repeated_meta = {k: np.repeat(v, samples_per_context, axis=0) for k, v in meta_context.items()}
            gen_ax, _ = _actions_to_ax(gen_actions, repeated_context, schema, config)
            ablation_risks[name] = _trajectory_details(gen_ax, repeated_context, repeated_meta, schema, config)["risk"]

        by_guidance[str(float(guidance))] = {
            "risk_control": risk_control,
            "raw_risk_levels": {key: float(value) for key, value in zip(level_labels, raw_levels)},
            "min_gap_mean_by_level": min_gap_by_level,
            "feasibility_by_level": feasibility_by_level,
            "condition_sensitivity": _condition_sensitivity(
                ablation_risks["normal"],
                ablation_risks["zero"],
                ablation_risks["shuffle"],
            ),
        }

    normal_cond = arrays["risk_condition"][context_idx]
    set_seed(seed + 777)
    normal_gen_norm = _sample_actions(
        model,
        arrays,
        context_idx,
        normal_cond,
        samples_per_context,
        device,
        primary_guidance,
        sample_batch_size,
    )
    normal_gen_actions = _decode_actions(normal_gen_norm, stats)
    repeated_context = np.repeat(real_context, samples_per_context, axis=0)
    normal_gen_ax, normal_gen_ax_unclipped = _actions_to_ax(normal_gen_actions, repeated_context, schema, config)
    normal_details = _trajectory_details(
        normal_gen_ax,
        repeated_context,
        {k: np.repeat(v, samples_per_context, axis=0) for k, v in meta_context.items()},
        schema,
        config,
    )
    naturality = _naturality_metrics(
        real_ax,
        normal_gen_ax,
        normal_gen_ax_unclipped,
        real_context,
        repeated_context,
        schema,
        config,
    )
    naturality["speed_negative_rate"] = float(np.mean(normal_details["lead_speed_negative"]))

    example_details = primary_details_for_examples or normal_details
    top_n = min(8, len(example_details["risk"]))
    top_idx = np.argsort(example_details["risk"])[-top_n:][::-1]
    examples = [
        {
            "risk": float(example_details["risk"][i]),
            "min_gap": float(example_details["min_gap"][i]),
            "gap": [float(x) for x in example_details["gap_profiles"][i]],
            "lead_speed": [float(x) for x in example_details["speed_profiles"][i]],
        }
        for i in top_idx
    ]

    return {
        "checkpoint": str(checkpoint_path),
        "label": label,
        "primary_guidance_scale": primary_guidance,
        "naturality": naturality,
        "normal_condition_feasibility": _feasibility_summary(normal_details),
        "by_guidance": by_guidance,
        "example_top_risk_rollouts": examples,
        "_plot_arrays": {
            "real_ax": real_ax,
            "real_context": real_context,
            "generated_ax": normal_gen_ax,
            "generated_context": repeated_context,
        },
    }


def analyze(config: dict, config_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _resolve_output_dir(config, config_dir)
    analysis_dir = Path(args.output_dir).resolve() if args.output_dir else output_dir / "behavior_analysis"
    schema = load_json(output_dir / "feature_schema.json")
    stats = load_json(output_dir / "normalization_stats.json")
    arrays = load_normalized_dataset(output_dir)
    raw_npz = np.load(output_dir / "dataset.npz", allow_pickle=True)
    raw = {k: raw_npz[k] for k in raw_npz.files}

    split = str(args.split)
    mask_idx = np.where(arrays["split_index"] == SPLIT_TO_INDEX[split])[0]
    if len(mask_idx) == 0:
        raise RuntimeError(f"No samples for split={split}")
    rng = np.random.default_rng(int(args.seed))
    context_idx = mask_idx.copy()
    rng.shuffle(context_idx)
    context_idx = context_idx[: min(int(args.num_contexts), len(context_idx))]

    levels = _parse_float_list(args.risk_levels, [0.50, 0.80, 0.90, 0.95])
    guidance_scales = _parse_float_list(args.guidance_scales, [1.0, 1.5, 2.0])
    checkpoints = [_resolve_checkpoint_path(value, output_dir) for value in _parse_checkpoints(args)]
    device = select_device(args.device or config.get("training", {}).get("device", "auto"))
    set_seed(int(args.seed))

    summary_by_checkpoint: dict[str, Any] = {}
    for ckpt_i, checkpoint_path in enumerate(checkpoints):
        label = _checkpoint_label(checkpoint_path)
        if label in summary_by_checkpoint:
            label = f"{label}_{ckpt_i + 1}"
        logger.info("Analyzing checkpoint=%s", checkpoint_path)
        summary_by_checkpoint[label] = _analyze_checkpoint(
            checkpoint_path,
            label,
            arrays,
            raw,
            schema,
            stats,
            config,
            context_idx,
            levels,
            guidance_scales,
            int(args.samples_per_context),
            int(args.sample_batch_size),
            device,
            int(args.seed) + ckpt_i * 1_000_000,
        )

    primary_label = next(iter(summary_by_checkpoint))
    plot_arrays = summary_by_checkpoint[primary_label].pop("_plot_arrays")
    real_ax = plot_arrays["real_ax"]
    gen_ax = plot_arrays["generated_ax"]
    dt = float(schema.get("dt", 0.04))
    real_jerk = _jerk_from_ax(real_ax, plot_arrays["real_context"], dt)
    gen_jerk = _jerk_from_ax(gen_ax, plot_arrays["generated_context"], dt)
    for value in summary_by_checkpoint.values():
        value.pop("_plot_arrays", None)

    plots = _make_plots(
        analysis_dir,
        output_dir / ".plot_cache",
        real_ax,
        gen_ax,
        real_jerk,
        gen_jerk,
        summary_by_checkpoint,
        primary_label,
        levels,
    )

    summary = {
        "config_path": str(Path(args.config).resolve() if args.config else DEFAULT_CONFIG_PATH),
        "split": split,
        "num_contexts": int(len(context_idx)),
        "samples_per_context": int(args.samples_per_context),
        "risk_levels": levels,
        "guidance_scales": guidance_scales,
        "seed": int(args.seed),
        "analysis_dir": str(analysis_dir),
        "checkpoints": summary_by_checkpoint,
        "plots": plots,
    }
    summary_path = analysis_dir / "generated_behavior_summary.json"
    save_json(summary, summary_path)
    logger.info("Wrote %s", summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Checkpoint path. May be passed multiple times. Overrides the default --checkpoints value when used alone.",
    )
    parser.add_argument("--checkpoints", default=DEFAULT_CHECKPOINTS, help="Comma-separated checkpoint paths.")
    parser.add_argument("--split", choices=("val", "test"), default=DEFAULT_SPLIT, help="Dataset split to analyze.")
    parser.add_argument("--num-contexts", type=int, default=DEFAULT_NUM_CONTEXTS, help="Number of contexts sampled from the split.")
    parser.add_argument("--samples-per-context", type=int, default=DEFAULT_SAMPLES_PER_CONTEXT, help="Generated samples per context.")
    parser.add_argument("--risk-levels", default=DEFAULT_RISK_LEVELS, help="Comma-separated risk condition percentiles.")
    parser.add_argument("--guidance-scales", default=DEFAULT_GUIDANCE_SCALES, help="Comma-separated classifier-free guidance scales.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--sample-batch-size", type=int, default=DEFAULT_SAMPLE_BATCH_SIZE, help="Model sampling batch size.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="cpu/cuda/auto; defaults to training.device.")
    parser.add_argument("--output-dir", default=None, help="Output directory; defaults to config output_dir/behavior_analysis.")
    parser.add_argument("--log-level", default=DEFAULT_LOG_LEVEL, help="Logging level.")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg_path = Path(args.config).resolve()
    analyze(load_yaml(cfg_path), cfg_path.parent, args)


if __name__ == "__main__":
    main()

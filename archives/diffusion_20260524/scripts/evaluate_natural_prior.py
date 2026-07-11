#!/usr/bin/env python3
"""Evaluate the highD car-following natural action diffusion prior."""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diffusion.src.data import SPLIT_TO_INDEX, load_normalized_dataset
from diffusion.src.kinematics import integrate_following_actions
from diffusion.src.model import build_model_from_schema
from diffusion.src.train import _epoch, _make_loader
from diffusion.src.types import VehicleBox, VehicleState
from diffusion.src.utils import load_json, load_yaml, save_json, select_device, set_seed, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "natural_following.yaml"
DEFAULT_CHECKPOINT_PATH = "checkpoints/best.pt"
DEFAULT_SPLIT = "test"
DEFAULT_LOG_LEVEL = "INFO"
logger = logging.getLogger(__name__)


def _resolve_output_dir(config: dict, config_dir: Path) -> Path:
    return (config_dir / config.get("paths", {}).get("output_dir", "../../../data/diffusion_natural/following")).resolve()


def _resolve_checkpoint_path(checkpoint: str | None, output_dir: Path) -> Path:
    path = Path(checkpoint or DEFAULT_CHECKPOINT_PATH)
    if path.is_absolute():
        return path
    cwd_path = path.resolve()
    if cwd_path.exists():
        return cwd_path
    return (output_dir / path).resolve()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _decode_actions(x: np.ndarray, stats: dict) -> np.ndarray:
    norm = stats["actions"]
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    return (x * std + mean).astype(np.float32)


def _actions_to_ax(actions: np.ndarray, context_states: np.ndarray, schema: dict, config: dict) -> tuple[np.ndarray, np.ndarray]:
    action_cfg = config.get("action", {})
    rep = str(schema.get("action_representation", action_cfg.get("representation", "acceleration"))).lower()
    ax_min = float(action_cfg.get("ax_min", -8.0))
    ax_max = float(action_cfg.get("ax_max", 4.0))
    dt = float(schema.get("dt", 0.04))
    if rep == "jerk":
        prev_ax = context_states[:, -1, 1, 4].astype(np.float32)
        ax = prev_ax[:, None] + np.cumsum(actions[:, :, 0], axis=1) * dt
    else:
        ax = actions[:, :, 0]
    ax = ax.astype(np.float32)
    return np.clip(ax, ax_min, ax_max).astype(np.float32), ax


def _actions_to_jerk(actions: np.ndarray, ax: np.ndarray, context_states: np.ndarray, schema: dict, config: dict) -> np.ndarray:
    rep = str(schema.get("action_representation", config.get("action", {}).get("representation", "acceleration"))).lower()
    if rep == "jerk":
        return actions[:, :, 0].astype(np.float32)
    dt = float(schema.get("dt", 0.04))
    prev_ax = context_states[:, -1, 1, 4].astype(np.float32)
    return (np.diff(np.concatenate([prev_ax[:, None], ax], axis=1), axis=1) / max(dt, 1e-6)).astype(np.float32)


def _summary(x: np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {f"{prefix}_{k}": float("nan") for k in ("mean", "std", "p05", "p50", "p95")}
    q05, q50, q95 = np.quantile(arr, [0.05, 0.50, 0.95])
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_p05": float(q05),
        f"{prefix}_p50": float(q50),
        f"{prefix}_p95": float(q95),
    }


def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    n = max(len(x), len(y))
    q = (np.arange(n, dtype=np.float64) + 0.5) / n
    xp = np.interp(q, (np.arange(len(x), dtype=np.float64) + 0.5) / len(x), x)
    yp = np.interp(q, (np.arange(len(y), dtype=np.float64) + 0.5) / len(y), y)
    return float(np.mean(np.abs(xp - yp)))


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    values = np.sort(np.unique(np.concatenate([x, y])))
    cdf_x = np.searchsorted(x, values, side="right") / len(x)
    cdf_y = np.searchsorted(y, values, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _histogram_l1(a: np.ndarray, b: np.ndarray, bins: int = 60) -> float:
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
    width = float(edges[1] - edges[0])
    return float(np.sum(np.abs(hx - hy)) * width)


def _integrate_lead_batch(ax: np.ndarray, context_states: np.ndarray, meta: dict[str, np.ndarray], schema: dict) -> np.ndarray:
    dt = float(schema.get("dt", 0.04))
    trajectories: list[np.ndarray] = []
    for i in range(ax.shape[0]):
        lead0 = context_states[i, -1, 1]
        adv_len = float(meta["adv_length"][i])
        lead_state = VehicleState(
            x=float(lead0[0]),
            y=float(lead0[1]),
            vx=float(lead0[2]),
            vy=float(lead0[3]),
            ax=float(lead0[4]),
            ay=float(lead0[5]),
            box=VehicleBox(length=adv_len),
        )
        lead = integrate_following_actions(lead_state, ax[i, :, None], dt)[1:]
        trajectories.append(lead)
    return np.stack(trajectories, axis=0)


def _sample_actions(model, arrays: dict, idx: np.ndarray, device: torch.device) -> np.ndarray:
    history = torch.from_numpy(arrays["context_states"][idx]).float().to(device)
    context = torch.from_numpy(arrays["context_features"][idx]).float().to(device)
    relative = torch.from_numpy(arrays["relative_history"][idx]).float().to(device)
    sample = model.sample(len(idx), history, context, relative)
    return sample.detach().cpu().numpy()


def _distribution_metrics(real_ax: np.ndarray, gen_ax: np.ndarray, real_j: np.ndarray, gen_j: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(_summary(real_ax, "real_ax"))
    out.update(_summary(gen_ax, "gen_ax"))
    out.update(_summary(real_j, "real_jerk"))
    out.update(_summary(gen_j, "gen_jerk"))
    out["ax_wasserstein"] = _wasserstein_1d(real_ax, gen_ax)
    out["jerk_wasserstein"] = _wasserstein_1d(real_j, gen_j)
    out["ax_ks"] = _ks_statistic(real_ax, gen_ax)
    out["jerk_ks"] = _ks_statistic(real_j, gen_j)
    out["ax_histogram_l1"] = _histogram_l1(real_ax, gen_ax)
    out["jerk_histogram_l1"] = _histogram_l1(real_j, gen_j)
    return out


def _feasibility_metrics(
    gen_unclipped_ax: np.ndarray,
    gen_jerk: np.ndarray,
    trajectories: np.ndarray,
    config: dict,
) -> dict[str, float]:
    action_cfg = config.get("action", {})
    ax_min = float(action_cfg.get("ax_min", -8.0))
    ax_max = float(action_cfg.get("ax_max", 4.0))
    jerk_abs_max = float(action_cfg.get("jerk_abs_max", 12.0))
    jumps = np.abs(np.diff(trajectories[:, :, 0], axis=1))
    return {
        "action_clip_rate": float(np.mean((gen_unclipped_ax < ax_min) | (gen_unclipped_ax > ax_max))),
        "speed_negative_rate": float(np.mean(trajectories[:, :, 2] < 0.0)),
        "jerk_violation_rate": float(np.mean(np.abs(gen_jerk) > jerk_abs_max)),
        "ax_violation_rate": float(np.mean((gen_unclipped_ax < ax_min) | (gen_unclipped_ax > ax_max))),
        "trajectory_discontinuity_rate": float(np.mean(jumps > float(config.get("filters", {}).get("max_position_jump", 5.0)))),
    }


def _distribution_distance_metrics(real: np.ndarray, gen: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_wasserstein": _wasserstein_1d(real, gen),
        f"{prefix}_ks": _ks_statistic(real, gen),
        f"{prefix}_histogram_l1": _histogram_l1(real, gen),
    }


def _trajectory_metrics(real_traj: np.ndarray, gen_traj: np.ndarray, lead_anchor_x: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(_summary(real_traj[:, :, 2], "real_lead_speed"))
    out.update(_summary(gen_traj[:, :, 2], "gen_lead_speed"))
    anchor_x = np.asarray(lead_anchor_x, dtype=np.float32)
    real_disp = real_traj[:, -1, 0] - anchor_x
    gen_disp = gen_traj[:, -1, 0] - anchor_x
    out.update(_summary(real_traj[:, -1, 2], "real_lead_final_speed"))
    out.update(_summary(gen_traj[:, -1, 2], "gen_lead_final_speed"))
    out.update(_summary(real_disp, "real_lead_displacement"))
    out.update(_summary(gen_disp, "gen_lead_displacement"))
    out.update(_distribution_distance_metrics(real_traj[:, :, 2], gen_traj[:, :, 2], "lead_speed"))
    out.update(_distribution_distance_metrics(real_traj[:, -1, 2], gen_traj[:, -1, 2], "lead_final_speed"))
    out.update(_distribution_distance_metrics(real_disp, gen_disp, "lead_displacement"))
    return out


def _interaction_series(
    ego_traj: np.ndarray,
    lead_traj: np.ndarray,
    meta: dict[str, np.ndarray],
    config: dict,
) -> dict[str, np.ndarray]:
    half_lengths = 0.5 * (
        np.asarray(meta["ego_length"], dtype=np.float32)[:, None]
        + np.asarray(meta["adv_length"], dtype=np.float32)[:, None]
    )
    gap = lead_traj[:, :, 0] - ego_traj[:, :, 0] - half_lengths
    relative_speed = ego_traj[:, :, 2] - lead_traj[:, :, 2]
    closing_speed = np.maximum(relative_speed, 0.0)
    eps = 1e-6
    ttc_cap = float(config.get("evaluation", {}).get("ttc_cap", 1000.0))
    thw_cap = float(config.get("evaluation", {}).get("thw_cap", 200.0))
    ttc = np.where(closing_speed > eps, gap / np.maximum(closing_speed, eps), ttc_cap)
    thw = gap / np.maximum(ego_traj[:, :, 2], eps)
    return {
        "gap": gap.astype(np.float32),
        "ttc": np.clip(ttc, 0.0, ttc_cap).astype(np.float32),
        "thw": np.clip(thw, 0.0, thw_cap).astype(np.float32),
        "relative_speed": relative_speed.astype(np.float32),
        "closing_speed": closing_speed.astype(np.float32),
    }


def _interaction_metrics(
    real_interaction: dict[str, np.ndarray],
    gen_interaction: dict[str, np.ndarray],
    config: dict,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("gap", "ttc", "thw", "relative_speed", "closing_speed"):
        out.update(_summary(real_interaction[key], f"real_{key}"))
        out.update(_summary(gen_interaction[key], f"gen_{key}"))
        out.update(_distribution_distance_metrics(real_interaction[key], gen_interaction[key], key))
    real_min_gap = np.min(real_interaction["gap"], axis=1)
    gen_min_gap = np.min(gen_interaction["gap"], axis=1)
    real_final_gap = real_interaction["gap"][:, -1]
    gen_final_gap = gen_interaction["gap"][:, -1]
    real_min_ttc = np.min(real_interaction["ttc"], axis=1)
    gen_min_ttc = np.min(gen_interaction["ttc"], axis=1)
    for real, gen, key in (
        (real_min_gap, gen_min_gap, "min_gap"),
        (real_final_gap, gen_final_gap, "final_gap"),
        (real_min_ttc, gen_min_ttc, "min_ttc"),
    ):
        out.update(_summary(real, f"real_{key}"))
        out.update(_summary(gen, f"gen_{key}"))
        out.update(_distribution_distance_metrics(real, gen, key))
    near_gap = float(config.get("evaluation", {}).get("near_collision_gap", 2.0))
    out["real_collision_rate"] = float(np.mean(real_interaction["gap"] <= 0.0))
    out["gen_collision_rate"] = float(np.mean(gen_interaction["gap"] <= 0.0))
    out["real_near_collision_rate"] = float(np.mean(real_interaction["gap"] < near_gap))
    out["gen_near_collision_rate"] = float(np.mean(gen_interaction["gap"] < near_gap))
    return out


def _diversity_summary(
    model,
    arrays: dict,
    raw: dict,
    stats: dict,
    schema: dict,
    config: dict,
    idx: np.ndarray,
    device: torch.device,
) -> dict[str, float | int]:
    eval_cfg = config.get("evaluation", {})
    n_contexts = min(int(eval_cfg.get("diversity_contexts", 32)), len(idx))
    samples_per_context = int(eval_cfg.get("samples_per_context", 8))
    if n_contexts == 0 or samples_per_context <= 0:
        return {"num_contexts": 0, "samples_per_context": int(samples_per_context)}
    context_idx = idx[:n_contexts]
    repeated = np.repeat(context_idx, samples_per_context)
    gen = _decode_actions(_sample_actions(model, arrays, repeated, device), stats)
    context = np.repeat(raw["context_states"][context_idx], samples_per_context, axis=0)
    ax, _ = _actions_to_ax(gen, context, schema, config)
    meta = {
        "ego_length": np.repeat(raw["ego_length"][context_idx], samples_per_context),
        "adv_length": np.repeat(raw["adv_length"][context_idx], samples_per_context),
    }
    traj = _integrate_lead_batch(ax, context, meta, schema)
    action_group = gen.reshape(n_contexts, samples_per_context, *gen.shape[1:])
    traj_group = traj.reshape(n_contexts, samples_per_context, *traj.shape[1:])
    final_x_std = np.std(traj_group[:, :, -1, 0], axis=1)
    final_v_std = np.std(traj_group[:, :, -1, 2], axis=1)
    action_std = np.mean(np.std(action_group, axis=1), axis=(1, 2))
    collapse_threshold = float(eval_cfg.get("mode_collapse_std_threshold", 1e-3))
    return {
        "num_contexts": int(n_contexts),
        "samples_per_context": int(samples_per_context),
        "sample_std_action": float(np.mean(action_std)),
        "sample_std_final_position": float(np.mean(final_x_std)),
        "sample_std_final_speed": float(np.mean(final_v_std)),
        "mode_collapse_indicator": float(np.mean(action_std < collapse_threshold)),
    }


def _write_metrics_csv(path: Path, sections: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "metric", "value"])
        writer.writeheader()
        for section, metrics in sections.items():
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.floating)):
                    writer.writerow({"section": section, "metric": key, "value": float(value)})


def _write_plots(
    output_dir: Path,
    eval_cfg: dict,
    real_ax: np.ndarray,
    gen_ax: np.ndarray,
    real_j: np.ndarray,
    gen_j: np.ndarray,
    real_traj: np.ndarray,
    gen_traj: np.ndarray,
    real_gaps: np.ndarray,
    gen_gaps: np.ndarray,
    schema: dict,
) -> list[str]:
    plot_dir = output_dir / str(eval_cfg.get("plot_dir", "natural_prior_plots"))
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("Matplotlib unavailable; skipping plots: %s", exc)
        return []

    written: list[Path] = []
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.hist(real_ax.reshape(-1), bins=60, alpha=0.55, density=True, label="highD")
    ax.hist(gen_ax.reshape(-1), bins=60, alpha=0.55, density=True, label="generated")
    ax.set_title("Acceleration Distribution")
    ax.set_xlabel("ax (m/s^2)")
    ax.set_ylabel("density")
    ax.legend()
    path = plot_dir / "ax_distribution_real_vs_generated.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.hist(real_j.reshape(-1), bins=60, alpha=0.55, density=True, label="highD")
    ax.hist(gen_j.reshape(-1), bins=60, alpha=0.55, density=True, label="generated")
    ax.set_title("Jerk Distribution")
    ax.set_xlabel("jx (m/s^3)")
    ax.set_ylabel("density")
    ax.legend()
    path = plot_dir / "jerk_distribution_real_vs_generated.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.hist(real_traj[:, :, 2].reshape(-1), bins=60, alpha=0.55, density=True, label="highD")
    ax.hist(gen_traj[:, :, 2].reshape(-1), bins=60, alpha=0.55, density=True, label="generated")
    ax.set_title("Lead Speed Distribution")
    ax.set_xlabel("vx (m/s)")
    ax.set_ylabel("density")
    ax.legend()
    path = plot_dir / "speed_distribution_real_vs_generated.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    n = min(6, gen_traj.shape[0])
    dt = float(schema.get("dt", 0.04))
    t = np.arange(gen_traj.shape[1], dtype=np.float32) * dt
    fig, axes = plt.subplots(n, 2, figsize=(9, max(2.2 * n, 3)), constrained_layout=True, squeeze=False)
    for i in range(n):
        axes[i, 0].plot(t, real_traj[i, :, 2], label="highD")
        axes[i, 0].plot(t, gen_traj[i, :, 2], label="generated")
        axes[i, 0].set_ylabel("vx")
        axes[i, 1].plot(t, real_gaps[i], label="highD")
        axes[i, 1].plot(t, gen_gaps[i], label="generated")
        axes[i, 1].set_ylabel("gap")
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    path = plot_dir / "example_rollouts.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)
    return [str(p) for p in written]


def evaluate(
    config: dict,
    config_dir: Path,
    *,
    checkpoint: str | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    output_dir = _resolve_output_dir(config, config_dir)
    schema = load_json(output_dir / "feature_schema.json")
    stats = load_json(output_dir / "normalization_stats.json")
    arrays = load_normalized_dataset(output_dir)
    raw = _load_npz(output_dir / "dataset.npz")
    if "future_states" not in raw:
        raise RuntimeError("dataset.npz is missing future_states; rebuild it with build_natural_dataset.py.")

    eval_cfg = config.get("evaluation", {})
    seed = int(eval_cfg.get("seed", config.get("training", {}).get("seed", 42)))
    set_seed(seed)
    checkpoint_path = _resolve_checkpoint_path(checkpoint, output_dir)
    device = select_device(config.get("training", {}).get("device", "auto"))
    model = build_model_from_schema(schema, config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    split_name = str(split or eval_cfg.get("split", "val"))
    mask_idx = np.where(arrays["split_index"] == SPLIT_TO_INDEX[split_name])[0]
    if len(mask_idx) == 0:
        raise RuntimeError(f"No samples for split={split_name}")
    max_samples = int(eval_cfg.get("max_samples", 512))
    idx = mask_idx[:max_samples] if max_samples > 0 else mask_idx

    loader = _make_loader(arrays, split_name, int(config.get("training", {}).get("batch_size", 256)), False, int(config.get("training", {}).get("num_workers", 0)))
    with torch.no_grad():
        validation = {f"val_{k}": float(v) for k, v in _epoch(model, loader, device, None).items()}

    gen_norm = _sample_actions(model, arrays, idx, device)
    gen_actions = _decode_actions(gen_norm, stats)
    real_actions = raw["actions"][idx]
    real_context = raw["context_states"][idx]
    future_states = raw["future_states"][idx].astype(np.float32)
    real_ego_traj = future_states[:, :, 0]
    real_traj = future_states[:, :, 1]
    real_ax, _ = _actions_to_ax(real_actions, real_context, schema, config)
    gen_ax, gen_unclipped_ax = _actions_to_ax(gen_actions, real_context, schema, config)
    real_j = _actions_to_jerk(real_actions, real_ax, real_context, schema, config)
    gen_j = _actions_to_jerk(gen_actions, gen_ax, real_context, schema, config)
    meta = {k: raw[k][idx] for k in ("ego_length", "adv_length")}
    gen_traj = _integrate_lead_batch(gen_ax, real_context, meta, schema)
    real_interaction = _interaction_series(real_ego_traj, real_traj, meta, config)
    gen_interaction = _interaction_series(real_ego_traj, gen_traj, meta, config)

    distribution = _distribution_metrics(real_ax, gen_ax, real_j, gen_j)
    feasibility = _feasibility_metrics(gen_unclipped_ax, gen_j, gen_traj, config)
    trajectory = _trajectory_metrics(real_traj, gen_traj, real_context[:, -1, 1, 0])
    interaction = _interaction_metrics(real_interaction, gen_interaction, config)
    diversity = _diversity_summary(model, arrays, raw, stats, schema, config, idx, device)
    sections = {
        "validation": validation,
        "action_distribution": distribution,
        "physical_feasibility": feasibility,
        "trajectory_naturalness": trajectory,
        "interaction_naturalness": interaction,
        "diversity": diversity,
    }
    plots = _write_plots(
        output_dir,
        eval_cfg,
        real_ax,
        gen_ax,
        real_j,
        gen_j,
        real_traj,
        gen_traj,
        real_interaction["gap"],
        gen_interaction["gap"],
        schema,
    )
    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "split": split_name,
        "num_samples": int(len(idx)),
        "action_representation": schema.get("action_representation"),
        "sections": sections,
        "plots": plots,
    }
    save_json(summary, output_dir / "naturalness_summary.json")
    save_json(diversity, output_dir / "diversity_summary.json")
    _write_metrics_csv(output_dir / "naturalness_metrics.csv", sections)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT_PATH,
        help="Checkpoint path. Relative paths are resolved from cwd if present, otherwise from config output_dir.",
    )
    parser.add_argument("--split", choices=("val", "test"), default=DEFAULT_SPLIT, help="Evaluation split.")
    parser.add_argument("--log-level", default=DEFAULT_LOG_LEVEL, help="Logging level.")
    args = parser.parse_args()
    setup_logging(args.log_level)
    cfg_path = Path(args.config).resolve()
    evaluate(load_yaml(cfg_path), cfg_path.parent, checkpoint=args.checkpoint, split=args.split)


if __name__ == "__main__":
    main()

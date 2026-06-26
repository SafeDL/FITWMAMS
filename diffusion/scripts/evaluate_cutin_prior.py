#!/usr/bin/env python3
"""Evaluate a trained cut-in action diffusion prior."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diffusion.src.data import load_normalized_dataset, split_indices
from diffusion.src.kinematics import integrate_cutin_acceleration_actions
from diffusion.src.model import build_model_from_schema
from diffusion.src.train import _epoch, _make_loader
from diffusion.src.utils import load_json, load_yaml, save_json, select_device, set_seed, setup_logging
from tools.io import load_npz

from diffusion.src.evaluation import (
    _actions_to_jerk,
    _conditional_sample_metrics,
    _decode_actions,
    _distribution_metrics,
    _feasibility_metrics,
    _interaction_metrics,
    _interaction_series,
    _resolve_checkpoint_path,
    _resolve_output_dir,
    _rollout_risk_series,
    _rollout_shift_metrics,
    _sample_actions,
    _trajectory_metrics,
    _trajectory_reconstruction_metrics,
    _write_plots,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "natural_cutin.yaml"
DEFAULT_CHECKPOINT_PATH = "checkpoints/best_noise_mse_train_val_test.pt"
DEFAULT_LOG_LEVEL = "INFO"
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cut-in specific action integration helpers
# ---------------------------------------------------------------------------

def _integrate_cutin_accel_plan(
    actions: np.ndarray,
    initial_states: np.ndarray,
    schema: dict,
    config: dict,
) -> np.ndarray:
    action_cfg = config.get("action", {})
    projection_cfg = config.get("trajectory_projection", {})
    return integrate_cutin_acceleration_actions(
        initial_states,
        actions,
        float(schema["dt"]),
        ax_min=float(action_cfg.get("ax_min", -8.0)),
        ax_max=float(action_cfg.get("ax_max", 4.0)),
        ay_abs_max=float(action_cfg.get("ay_abs_max", 4.0)),
        speed_min=float(projection_cfg.get("speed_min", 0.0)),
        speed_max=float(projection_cfg.get("speed_max", 50.0)),
    )


def _is_cutin_ax_ay_action(schema: dict) -> bool:
    return (
        str(schema.get("event_type", "")).lower() == "cut_in"
        and str(schema.get("generation_target", "")).lower() == "action"
        and str(schema.get("action_representation", "")).lower()
        in {"ax_ay", "acceleration"}
    )


# ---------------------------------------------------------------------------
# Cut-in specific metrics
# ---------------------------------------------------------------------------


def _masked_event_lateral_metrics(
    raw: dict,
    idx: np.ndarray,
    real_traj: np.ndarray,
    gen_traj: np.ndarray,
) -> dict[str, float]:
    out: dict[str, float] = {}

    def _one(index_key: str, mask_key: str, prefix: str) -> None:
        if index_key not in raw or mask_key not in raw:
            return
        indices = raw[index_key][idx].astype(np.int64)
        mask = raw[mask_key][idx].astype(np.float32) > 0.5
        valid = mask & (indices >= 0) & (indices < real_traj.shape[1])
        out[f"{prefix}_valid_count"] = float(np.sum(valid))
        out[f"{prefix}_valid_rate"] = float(np.mean(valid)) if len(valid) else 0.0
        if not np.any(valid):
            out[f"{prefix}_y_l1"] = float("nan")
            return
        rows = np.arange(len(indices), dtype=np.int64)[valid]
        steps = indices[valid]
        out[f"{prefix}_y_l1"] = float(
            np.mean(np.abs(gen_traj[rows, steps, 1] - real_traj[rows, steps, 1]))
        )

    _one("future_cross_index", "cross_mask", "cross")
    _one("future_cutin_end_index", "cutin_end_mask", "cutin_end")
    return out


def _lateral_motion_metrics(
    real_traj: np.ndarray,
    gen_traj: np.ndarray,
    schema: dict,
    config: dict,
) -> dict[str, float]:
    """Independent lateral motion evaluation: speed, accel, yaw rate distributions."""
    from diffusion.src.evaluation import _distribution_distance_metrics, _spectral_l1, _summary

    out: dict[str, float] = {}
    # target lateral speed
    real_vy = real_traj[:, :, 3]
    gen_vy = gen_traj[:, :, 3]
    out.update(_summary(real_vy, "real_target_lateral_speed"))
    out.update(_summary(gen_vy, "gen_target_lateral_speed"))
    out.update(_distribution_distance_metrics(real_vy, gen_vy, "target_lateral_speed"))
    # target lateral acceleration
    real_ay = real_traj[:, :, 5]
    gen_ay = gen_traj[:, :, 5]
    out.update(_summary(real_ay, "real_target_lateral_accel"))
    out.update(_summary(gen_ay, "gen_target_lateral_accel"))
    out.update(_distribution_distance_metrics(real_ay, gen_ay, "target_lateral_accel"))
    # yaw rate (derived from heading)
    dt = float(schema.get("dt", 0.04))
    real_heading = np.unwrap(
        np.arctan2(real_traj[:, :, 3].astype(np.float64),
                   np.maximum(real_traj[:, :, 2].astype(np.float64), 1.0e-6)),
        axis=1,
    )
    gen_heading = np.unwrap(
        np.arctan2(gen_traj[:, :, 3].astype(np.float64),
                   np.maximum(gen_traj[:, :, 2].astype(np.float64), 1.0e-6)),
        axis=1,
    )
    real_yaw = np.diff(real_heading, axis=1) / max(dt, 1.0e-6)
    gen_yaw = np.diff(gen_heading, axis=1) / max(dt, 1.0e-6)
    out.update(_summary(real_yaw, "real_target_yaw_rate"))
    out.update(_summary(gen_yaw, "gen_target_yaw_rate"))
    out.update(_distribution_distance_metrics(real_yaw, gen_yaw, "target_yaw_rate"))
    # lateral displacement
    real_lat_disp = real_traj[:, -1, 1] - real_traj[:, 0, 1]
    gen_lat_disp = gen_traj[:, -1, 1] - gen_traj[:, 0, 1]
    out.update(_summary(real_lat_disp, "real_lateral_displacement"))
    out.update(_summary(gen_lat_disp, "gen_lateral_displacement"))
    out.update(_distribution_distance_metrics(real_lat_disp, gen_lat_disp, "lateral_displacement"))
    # spectral L1 for lateral motion
    out["target_lateral_speed_spectral_l1"] = _spectral_l1(real_vy, gen_vy)
    out["target_lateral_accel_spectral_l1"] = _spectral_l1(real_ay, gen_ay)
    out["target_lateral_position_spectral_l1"] = _spectral_l1(
        real_traj[:, :, 1], gen_traj[:, :, 1],
    )
    return out


# ---------------------------------------------------------------------------
# Diversity summary (cut-in)
# ---------------------------------------------------------------------------


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
    eval_cfg = config["evaluation"]
    samples_per_context = int(eval_cfg.get("samples_per_context", 8))
    if len(idx) == 0 or samples_per_context <= 0:
        return {"num_contexts": 0, "samples_per_context": int(samples_per_context)}
    context_idx = idx
    n_contexts = len(context_idx)
    repeated = np.repeat(context_idx, samples_per_context)
    gen = _decode_actions(
        _sample_actions(model, arrays, repeated, device, int(eval_cfg.get("sample_batch_size", 512))),
        stats,
    )
    initial_states = np.repeat(raw["initial_states"][context_idx], samples_per_context, axis=0)
    if not _is_cutin_ax_ay_action(schema):
        raise RuntimeError("Cut-in evaluation requires action_representation='ax_ay'")
    traj = _integrate_cutin_accel_plan(gen, initial_states, schema, config)
    action_group = gen.reshape(n_contexts, samples_per_context, *gen.shape[1:])
    traj_group = traj.reshape(n_contexts, samples_per_context, *traj.shape[1:])
    final_x_std = np.std(traj_group[:, :, -1, 0], axis=1)
    final_y_std = np.std(traj_group[:, :, -1, 1], axis=1)
    final_v_std = np.std(traj_group[:, :, -1, 2], axis=1)
    action_std = np.mean(np.std(action_group, axis=1), axis=(1, 2))
    collapse_threshold = float(eval_cfg.get("mode_collapse_std_threshold", 1e-3))
    return {
        "num_contexts": int(n_contexts),
        "samples_per_context": int(samples_per_context),
        "sample_std_action": float(np.mean(action_std)),
        "sample_std_final_position": float(np.mean(final_x_std)),
        "sample_std_final_lateral_position": float(np.mean(final_y_std)),
        "sample_std_final_speed": float(np.mean(final_v_std)),
        "mode_collapse_indicator": float(np.mean(action_std < collapse_threshold)),
    }


# ---------------------------------------------------------------------------
# Main evaluate entry point (cut-in)
# ---------------------------------------------------------------------------


def evaluate(
    config: dict,
    config_dir: Path,
    *,
    checkpoint: str | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    output_dir = _resolve_output_dir(config, config_dir)
    schema = load_json(output_dir / "feature_schema.json")
    if not _is_cutin_ax_ay_action(schema):
        raise RuntimeError(
            "Cut-in evaluation requires an anchor-scenario action dataset with "
            "action_representation='ax_ay'. Rebuild the dataset and checkpoint."
        )
    stats = load_json(output_dir / "normalization_stats.json")
    arrays = load_normalized_dataset(output_dir)
    raw = load_npz(output_dir / "dataset.npz")
    if "future_states" not in raw:
        raise RuntimeError(
            "dataset.npz is missing future_states; rebuild it with "
            "process_highD/scripts/build_natural_dataset.py."
        )

    eval_cfg = config["evaluation"]
    seed = int(eval_cfg["seed"])
    set_seed(seed)
    checkpoint_path = _resolve_checkpoint_path(checkpoint or DEFAULT_CHECKPOINT_PATH, output_dir)
    device = select_device(config["training"]["device"])
    model = build_model_from_schema(schema, config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    split_name = str(split or eval_cfg.get("split", "test")).lower()
    mask_idx = split_indices(arrays, split_name)
    if len(mask_idx) == 0:
        raise RuntimeError(f"No samples for split={split_name}")
    eval_max_samples = int(eval_cfg.get("max_samples", 500))
    if eval_max_samples > 0 and len(mask_idx) > eval_max_samples:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(mask_idx, size=eval_max_samples, replace=False))
        sampling_method = "random_without_replacement"
    else:
        idx = mask_idx
        sampling_method = "full_split"

    loader = _make_loader(
        arrays,
        split_name,
        int(config.get("training", {}).get("batch_size", 256)),
        False,
        int(config.get("training", {}).get("num_workers", 0)),
    )
    with torch.no_grad():
        denoising_loss = {
            f"{split_name}_{k}": float(v)
            for k, v in _epoch(
                model,
                loader,
                device,
            ).items()
        }

    sample_batch_size = int(eval_cfg.get("sample_batch_size", config.get("training", {}).get("batch_size", 256)))
    gen_norm = _sample_actions(model, arrays, idx, device, batch_size=sample_batch_size)
    gen_actions = _decode_actions(gen_norm, stats)
    real_actions = raw["actions"][idx]
    initial_states = raw["initial_states"][idx]
    future_states = raw["future_states"][idx].astype(np.float32)
    real_ego_traj = future_states[:, :, 0]
    real_traj = future_states[:, :, 1]
    meta = {k: raw[k][idx] for k in ("ego_length", "adv_length")}

    # --- cut-in specific action → trajectory decoding ---
    gen_traj = _integrate_cutin_accel_plan(
        gen_actions,
        initial_states,
        schema,
        config,
    )
    real_ax = real_actions[:, :, 0].astype(np.float32)
    gen_unclipped_ax = gen_actions[:, :, 0].astype(np.float32)
    gen_ax = gen_traj[:, :, 4].astype(np.float32)
    real_j = _actions_to_jerk(real_actions, real_ax, initial_states, schema, config)
    gen_j = _actions_to_jerk(gen_actions, gen_ax, initial_states, schema, config)

    real_interaction = _interaction_series(real_ego_traj, real_traj, meta, config)
    gen_interaction = _interaction_series(real_ego_traj, gen_traj, meta, config)

    # --- compute all metric sections ---
    distribution = _distribution_metrics(
        real_ax,
        gen_ax,
        real_j,
        gen_j,
    )
    feasibility = _feasibility_metrics(
        gen_unclipped_ax, gen_j, gen_traj, config,
        dt=float(schema["dt"]), is_cutin=True,
    )
    trajectory = _trajectory_metrics(
        real_traj, gen_traj,
        initial_states[:, 1, 0], initial_states[:, 1, 1],
        is_cutin=True,
    )
    interaction = _interaction_metrics(real_interaction, gen_interaction, config, is_cutin=True)
    real_rollout = _rollout_risk_series(
        real_ego_traj,
        real_traj,
        meta,
        config,
        is_cutin=True,
    )
    gen_rollout = _rollout_risk_series(
        real_ego_traj,
        gen_traj,
        meta,
        config,
        is_cutin=True,
    )
    rollout_shift = _rollout_shift_metrics(real_rollout, gen_rollout)
    conditional = _conditional_sample_metrics(model, arrays, idx, device, eval_cfg)
    diversity = _diversity_summary(model, arrays, raw, stats, schema, config, idx, device)
    trajectory_reconstruction = _trajectory_reconstruction_metrics(
        real_traj, gen_traj, dt=float(schema["dt"]),
    )
    masked_events = _masked_event_lateral_metrics(raw, idx, real_traj, gen_traj)
    lateral_motion = _lateral_motion_metrics(real_traj, gen_traj, schema, config)

    sections = {
        "denoising_loss": denoising_loss,
        "action_distribution": distribution,
        "physical_feasibility": feasibility,
        "trajectory_naturalness": trajectory,
        "interaction_naturalness": interaction,
        "record_conditioned_rollout_shift": rollout_shift,
        "conditional_sample_quality": conditional,
        "diversity": diversity,
        "trajectory_reconstruction": trajectory_reconstruction,
        "masked_event_lateral_errors": masked_events,
        "lateral_motion_naturalness": lateral_motion,
    }
    plots = _write_plots(
        output_dir, eval_cfg,
        real_ax, gen_ax, real_j, gen_j,
        real_traj, gen_traj,
        real_interaction["gap"], gen_interaction["gap"],
        real_interaction["lateral_offset"], gen_interaction["lateral_offset"],
        real_interaction["relative_speed"], gen_interaction["relative_speed"],
        schema,
    )
    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "split": split_name,
        "num_samples": int(len(idx)),
        "num_available_split_samples": int(len(mask_idx)),
        "sample_selection": sampling_method,
        "sampler": "ddim",
        "action_representation": schema["action_representation"],
        "sections": sections,
        "plots": plots,
    }
    save_json(summary, output_dir / "naturalness_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to natural cut-in diffusion config.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. Relative paths are resolved under paths.output_dir.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Evaluation split: train, val, test, or all. Defaults to evaluation.split.",
    )
    args = parser.parse_args()
    setup_logging(DEFAULT_LOG_LEVEL)
    cfg_path = Path(args.config).resolve()
    evaluate(
        load_yaml(cfg_path),
        cfg_path.parent,
        checkpoint=args.checkpoint,
        split=args.split,
    )


if __name__ == "__main__":
    main()

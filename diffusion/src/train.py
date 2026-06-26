"""Training loop for naturalistic action diffusion priors."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from .data import (
    build_action_dataset,
    load_normalized_dataset,
    sequence_config,
    split_indices,
    split_mode,
)
from .model import GaussianActionDiffusion, build_model_from_schema
from .utils import load_json, save_json, select_device, set_seed

logger = logging.getLogger(__name__)


def _make_loader(
    arrays: dict,
    split: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    max_samples: int = 0,
    include_cutin_context: bool = False,
) -> DataLoader:
    idx = split_indices(arrays, split)
    if len(idx) == 0:
        raise RuntimeError(f"No samples for split={split}")
    if max_samples and max_samples > 0:
        idx = idx[: int(max_samples)]
    tensor_items = [
        torch.from_numpy(arrays["scenario_conditions"][idx]).float(),
        torch.from_numpy(arrays["actions"][idx]).float(),
    ]
    if include_cutin_context:
        required = (
            "raw_scenario_conditions",
            "initial_states",
        )
        missing = [key for key in required if key not in arrays]
        if missing:
            raise KeyError(
                "Cut-in trajectory loss requires raw dataset arrays missing from "
                f"training data: {missing}. Rebuild dataset.npz if needed."
            )
        tensor_items.extend(
            [
                torch.from_numpy(arrays["raw_scenario_conditions"][idx]).float(),
                torch.from_numpy(arrays["initial_states"][idx]).float(),
            ]
        )
    return DataLoader(
        TensorDataset(*tensor_items),
        batch_size=int(batch_size),
        shuffle=shuffle,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
    )


def _checkpoint_filename(stem: str, config: dict) -> str:
    del config
    return f"{stem}_train_val_test.pt"


def _epoch(
    model: GaussianActionDiffusion,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: float = 0.0,
    trajectory_loss_cfg: dict | None = None,
    action_mean: torch.Tensor | None = None,
    action_std: torch.Tensor | None = None,
) -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals: Dict[str, float] = {}
    total_n = 0
    for batch in loader:
        scenario_conditions, actions = batch[:2]
        scenario_conditions = scenario_conditions.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        trajectory_context = None
        if trajectory_loss_cfg is not None:
            if len(batch) < 4:
                raise RuntimeError(
                    "Cut-in trajectory loss is enabled but the loader did not "
                    "provide raw cut-in context tensors."
                )
            trajectory_context = {
                "scenario_conditions": batch[2].to(device, non_blocking=True),
                "initial_states": batch[3].to(device, non_blocking=True),
            }
            if action_mean is not None and action_std is not None:
                trajectory_context["action_mean"] = action_mean.to(device)
                trajectory_context["action_std"] = action_std.to(device)
        with torch.set_grad_enabled(train):
            losses = model.p_losses(
                actions,
                scenario_conditions,
                trajectory_context=trajectory_context,
                trajectory_loss_cfg=trajectory_loss_cfg,
            )
            loss = losses["loss"]
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()
        n = int(actions.shape[0])
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * n
        total_n += n
    return {key: value / max(total_n, 1) for key, value in totals.items()}


def _torch_generator_for(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _fixed_noise_losses(
    model: GaussianActionDiffusion,
    actions: torch.Tensor,
    scenario_conditions: torch.Tensor,
    timestep: int,
    noise_seed: int,
) -> Dict[str, torch.Tensor]:
    t = torch.full((actions.shape[0],), int(timestep), device=actions.device, dtype=torch.long)
    generator = _torch_generator_for(actions.device, int(noise_seed))
    noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype, generator=generator)
    noisy = model.q_sample(actions, t, noise)
    pred = model.denoiser(noisy, t, scenario_conditions)
    noise_mse = F.mse_loss(pred, noise)
    x0 = model.predict_start_from_noise(noisy, t, pred)
    x0_l1 = F.l1_loss(x0, actions)
    if x0.shape[1] > 1:
        smooth = torch.mean(torch.abs(x0[:, 1:] - x0[:, :-1]))
    else:
        smooth = torch.zeros((), device=actions.device, dtype=actions.dtype)
    loss = noise_mse + model.denoiser.cfg.x0_weight * x0_l1 + model.denoiser.cfg.smooth_weight * smooth
    return {
        "loss": loss,
        "noise_mse": noise_mse.detach(),
        "x0_l1": x0_l1.detach(),
        "smooth": smooth.detach(),
    }


@torch.no_grad()
def _deterministic_epoch(
    model: GaussianActionDiffusion,
    loader: DataLoader,
    device: torch.device,
    timesteps: list[int],
    noise_seed: int,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    total_n = 0
    for batch_idx, batch in enumerate(loader):
        scenario_conditions, actions = batch[:2]
        scenario_conditions = scenario_conditions.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        n = int(actions.shape[0])
        for offset, timestep in enumerate(timesteps):
            losses = _fixed_noise_losses(
                model,
                actions,
                scenario_conditions,
                timestep,
                int(noise_seed) + batch_idx * 1009 + offset * 9173,
            )
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * n
            total_n += n
    return {key: value / max(total_n, 1) for key, value in totals.items()}


def _fixed_timesteps_from_config(training: dict, model: GaussianActionDiffusion) -> list[int]:
    raw = training.get("fixed_eval_timesteps", [0, 25, 50, 75, 99])
    out = sorted({max(0, min(int(t), model.num_steps - 1)) for t in raw})
    if not out:
        raise ValueError("training.fixed_eval_timesteps must contain at least one timestep")
    return out


def _validate_schema_matches_config(schema: dict, config: dict, output_dir: Path) -> None:
    expected = {
        "event_type": str(config.get("event", {}).get("event_type", "")),
        "horizon_steps": int(sequence_config(config).get("horizon_steps", -1)),
        "action_representation": str(config.get("action", {}).get("representation", "")),
        "conditioning_mode": "anchor_scenario",
    }
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        if expected_value in {"", -1}:
            continue
        actual = schema.get(key)
        if isinstance(expected_value, int):
            actual = int(actual)
        else:
            actual = str(actual)
        if actual != expected_value:
            mismatches.append(f"{key}: schema={actual!r}, config={expected_value!r}")
    if mismatches:
        joined = "; ".join(mismatches)
        raise RuntimeError(
            "Existing diffusion dataset schema does not match the training "
            f"config in {output_dir}: {joined}. Rebuild the dataset first with "
            "process_highD/scripts/build_natural_dataset.py or set "
            "dataset.rebuild=true for one run."
        )
    schema_mode = str(schema.get("split_mode", "")).lower()
    if schema_mode != "train_val_test":
        raise RuntimeError(
            "Existing diffusion dataset must be built with split.mode="
            f"'train_val_test', got {schema_mode!r} in {output_dir}. "
            "Rebuild the dataset with process_highD/scripts/build_natural_dataset.py "
            "or set dataset.rebuild=true for one run."
        )


def _load_training_arrays(output_dir: Path, *, include_cutin_context: bool) -> dict:
    arrays = load_normalized_dataset(output_dir)
    if not include_cutin_context:
        return arrays
    raw_path = output_dir / "dataset.npz"
    if not raw_path.exists():
        raise FileNotFoundError(
            "Cut-in trajectory loss requires raw trajectory arrays, but "
            f"{raw_path} does not exist."
        )
    raw = np.load(raw_path, allow_pickle=True)
    for key in ("initial_states",):
        if key not in raw.files:
            raise KeyError(f"{raw_path} is missing required array {key!r}")
        arrays[key] = raw[key]
    if "scenario_conditions" not in raw.files:
        raise KeyError(f"{raw_path} is missing required array 'scenario_conditions'")
    arrays["raw_scenario_conditions"] = raw["scenario_conditions"]
    return arrays


def _cutin_trajectory_loss_config(config: dict, schema: dict) -> dict | None:
    if str(schema.get("event_type", "")).lower() != "cut_in":
        return None
    cfg = dict(config.get("cutin_trajectory_loss", {}))
    if not bool(cfg.get("enabled", False)):
        return None
    weighted_keys = [key for key in cfg if key.endswith("_weight")]
    if not weighted_keys or all(float(cfg.get(key, 0.0)) <= 0.0 for key in weighted_keys):
        return None
    action_cfg = config.get("action", {})
    projection_cfg = config.get("trajectory_projection", {})
    cutin_cfg = config.get("cutin_risk", {})
    merged = {
        **cfg,
        "ax_min": float(action_cfg.get("ax_min", -8.0)),
        "ax_max": float(action_cfg.get("ax_max", 4.0)),
        "ay_abs_max": float(action_cfg.get("ay_abs_max", 4.0)),
        "lateral_jerk_abs_max": float(action_cfg.get("lateral_jerk_abs_max", 8.0)),
        "speed_min": float(projection_cfg.get("speed_min", 0.0)),
        "speed_max": float(projection_cfg.get("speed_max", 50.0)),
        "lateral_overlap_threshold": float(
            cutin_cfg.get("lateral_overlap_threshold", cfg.get("lateral_overlap_threshold", 1.0))
        ),
        "cutin_lateral_offset": float(
            cutin_cfg.get("cutin_lateral_offset", cfg.get("cutin_lateral_offset", 1.0))
        ),
        "post_cutin_window_seconds": float(
            cutin_cfg.get("post_cutin_window_seconds", cfg.get("post_cutin_window_seconds", 3.0))
        ),
    }
    return merged


def train_action_diffusion(config: dict, *, config_dir: str | Path | None = None) -> dict:
    paths = config.get("paths", {})
    if "output_dir" not in paths:
        raise KeyError("Config paths.output_dir is required")
    base = Path(config_dir).resolve() if config_dir is not None else Path.cwd()
    output_dir = (base / paths["output_dir"]).resolve()
    dataset_path = output_dir / "dataset_normalized.npz"
    if bool(config.get("dataset", {}).get("rebuild", False)):
        build_action_dataset(config, config_dir=base)
    elif not dataset_path.exists():
        raise FileNotFoundError(f"Diffusion dataset not found: {dataset_path}")

    schema = load_json(output_dir / "feature_schema.json")
    _validate_schema_matches_config(schema, config, output_dir)
    stats = load_json(output_dir / "normalization_stats.json")
    trajectory_loss_cfg = _cutin_trajectory_loss_config(config, schema)
    include_cutin_context = trajectory_loss_cfg is not None
    arrays = _load_training_arrays(output_dir, include_cutin_context=include_cutin_context)
    training = config.get("training", {})
    set_seed(int(training.get("seed", 42)))
    device = select_device(training.get("device", "auto"))
    model = build_model_from_schema(schema, config).to(device)
    action_mean = None
    action_std = None
    if include_cutin_context:
        action_stats = stats["actions"]
        action_mean = torch.tensor(action_stats["mean"], dtype=torch.float32, device=device).view(1, 1, -1)
        action_std = torch.tensor(action_stats["std"], dtype=torch.float32, device=device).view(1, 1, -1)

    batch_size = int(training.get("batch_size", 256))
    num_workers = int(training.get("num_workers", 0))
    train_loader = _make_loader(
        arrays,
        "train",
        batch_size,
        True,
        num_workers,
        include_cutin_context=include_cutin_context,
    )
    val_loader = _make_loader(
        arrays,
        "val",
        batch_size,
        False,
        num_workers,
        include_cutin_context=include_cutin_context,
    )
    fixed_eval_max_samples = int(
        training.get("fixed_eval_max_samples", 512)
    )
    fixed_eval_split = "val"
    fixed_eval_loader = _make_loader(
        arrays,
        fixed_eval_split,
        batch_size,
        False,
        num_workers,
        fixed_eval_max_samples,
        include_cutin_context=include_cutin_context,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("lr", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    epochs = int(training.get("epochs", 160))
    grad_clip = float(training.get("grad_clip", 1.0))
    min_lr = float(training.get("min_lr", 5e-5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - 1), eta_min=min_lr)
    fixed_eval_timesteps = _fixed_timesteps_from_config(training, model)
    fixed_eval_seed = int(training.get("fixed_eval_seed", 12345))
    best_noise_mse = float("inf")
    best_epoch = 0
    best_monitor_loss = float("inf")
    best_val_loss = float("inf")
    final_metrics: dict[str, float] = {}
    history: list[dict[str, float]] = []
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / _checkpoint_filename("best_noise_mse", config)
    final_checkpoint_path = checkpoint_dir / _checkpoint_filename("final", config)
    tensorboard_dir = output_dir / "tensorboard"
    if bool(training.get("clear_tensorboard_dir", False)) and tensorboard_dir.exists():
        for path in tensorboard_dir.glob("events.out.tfevents.*"):
            path.unlink()

    logger.info(
        "Training on %s for %d epochs; split_mode=%s; samples=%d",
        device,
        epochs,
        split_mode(config),
        int(arrays["actions"].shape[0]),
    )
    with SummaryWriter(log_dir=str(tensorboard_dir)) as writer:
        for epoch in range(1, epochs + 1):
            train_metrics = _epoch(
                model,
                train_loader,
                device,
                optimizer,
                grad_clip,
                trajectory_loss_cfg=trajectory_loss_cfg,
                action_mean=action_mean,
                action_std=action_std,
            )
            with torch.no_grad():
                val_metrics = _epoch(
                    model,
                    val_loader,
                    device,
                    None,
                    trajectory_loss_cfg=trajectory_loss_cfg,
                    action_mean=action_mean,
                    action_std=action_std,
                )
            fixed_eval_metrics = _deterministic_epoch(
                model,
                fixed_eval_loader,
                device,
                fixed_eval_timesteps,
                fixed_eval_seed,
            )
            final_metrics = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "fixed_eval_loss": fixed_eval_metrics["loss"],
                "train_noise_mse": train_metrics["noise_mse"],
                "fixed_eval_noise_mse": fixed_eval_metrics["noise_mse"],
                "train_x0_l1": train_metrics["x0_l1"],
                "fixed_eval_x0_l1": fixed_eval_metrics["x0_l1"],
                "train_smooth": train_metrics["smooth"],
                "fixed_eval_smooth": fixed_eval_metrics["smooth"],
            }
            final_metrics.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_noise_mse": val_metrics["noise_mse"],
                    "val_x0_l1": val_metrics["x0_l1"],
                    "val_smooth": val_metrics["smooth"],
                }
            )
            for key in (
                "cutin_constraint_loss",
                "end_y_loss",
                "post_lane_loss",
                "lateral_jerk_loss",
                "trajectory_x_l1",
                "trajectory_y_l1",
                "trajectory_vx_l1",
                "trajectory_vy_l1",
                "endpoint_x_l1",
                "endpoint_y_l1",
                "cross_y_l1",
                "end_y_l1",
                "kinematic_consistency_l1",
            ):
                if key in train_metrics:
                    final_metrics[f"train_{key}"] = train_metrics[key]
                if key in val_metrics:
                    final_metrics[f"val_{key}"] = val_metrics[key]
            history.append({key: float(value) for key, value in final_metrics.items()})
            best_val_loss = min(best_val_loss, float(val_metrics["loss"]))
            monitor_noise_mse = float(val_metrics["noise_mse"])
            monitor_loss = float(val_metrics["loss"])
            monitor_split = "val"
            if monitor_noise_mse < best_noise_mse:
                best_noise_mse = monitor_noise_mse
                best_monitor_loss = monitor_loss
                best_epoch = epoch
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "schema": schema,
                        "config": config,
                        "epoch": epoch,
                        "monitor_split": monitor_split,
                        "monitor_noise_mse": best_noise_mse,
                        "monitor_loss": monitor_loss,
                        "val_noise_mse": best_noise_mse,
                        "val_loss": monitor_loss,
                    },
                    best_checkpoint_path,
                )
            writer.add_scalar("loss/train", float(train_metrics["loss"]), epoch)
            writer.add_scalar(f"loss/fixed_{fixed_eval_split}", float(fixed_eval_metrics["loss"]), epoch)
            writer.add_scalar("loss/val", float(val_metrics["loss"]), epoch)
            writer.add_scalar(
                "noise_mse/train",
                float(train_metrics["noise_mse"]),
                epoch,
            )
            writer.add_scalar(
                f"noise_mse/fixed_{fixed_eval_split}",
                float(fixed_eval_metrics["noise_mse"]),
                epoch,
            )
            writer.add_scalar("noise_mse/val", float(val_metrics["noise_mse"]), epoch)
            writer.add_scalar("learning_rate", float(scheduler.get_last_lr()[0]), epoch)
            writer.add_scalar(f"best/{monitor_split}_noise_mse", float(best_noise_mse), epoch)
            for key in (
                "cutin_constraint_loss",
                "end_y_loss",
                "post_lane_loss",
                "lateral_jerk_loss",
                "trajectory_x_l1",
                "trajectory_y_l1",
                "trajectory_vx_l1",
                "trajectory_vy_l1",
                "endpoint_x_l1",
                "endpoint_y_l1",
                "cross_y_l1",
                "end_y_l1",
                "kinematic_consistency_l1",
            ):
                if key in train_metrics:
                    writer.add_scalar(f"{key}/train", float(train_metrics[key]), epoch)
                if key in val_metrics:
                    writer.add_scalar(f"{key}/val", float(val_metrics[key]), epoch)
            if epoch == 1 or epoch % int(training.get("log_every_epochs", 10)) == 0 or epoch == epochs:
                logger.info(
                    "epoch=%03d train_noise_mse=%.6f val_noise_mse=%.6f",
                    epoch,
                    train_metrics["noise_mse"],
                    val_metrics["noise_mse"],
                )
            scheduler.step()

    torch.save(
        {
            "model_state": model.state_dict(),
            "schema": schema,
            "config": config,
            "epoch": epochs,
            "monitor_split": "final",
            "monitor_noise_mse": final_metrics.get("fixed_eval_noise_mse"),
            "monitor_loss": final_metrics.get("fixed_eval_loss"),
        },
        final_checkpoint_path,
    )

    save_json(
        history,
        output_dir / "training_history.json",
    )
    save_json(
        {
            "checkpoint": str(best_checkpoint_path),
            "final_checkpoint": str(final_checkpoint_path),
            "split_mode": split_mode(config),
            "validation_enabled": True,
            "monitor_split": "val",
            "best_epoch": int(best_epoch),
            "best_val_loss": best_val_loss,
            "best_val_noise_mse": best_noise_mse,
            "best_monitor_loss": best_monitor_loss,
            "best_monitor_noise_mse": best_noise_mse,
            "final_metrics": final_metrics,
            "epochs": epochs,
            "lr_schedule": "cosine",
            "min_lr": min_lr,
            "fixed_eval_timesteps": fixed_eval_timesteps,
            "fixed_eval_seed": fixed_eval_seed,
            "fixed_eval_split": fixed_eval_split,
            "fixed_eval_max_samples": fixed_eval_max_samples,
            "tensorboard_dir": str(tensorboard_dir),
            "training_history": str(output_dir / "training_history.json"),
        },
        output_dir / "training_summary.json",
    )
    return {
        "output_dir": output_dir,
        "best_val_loss": best_val_loss,
        "epochs": epochs,
        "training_history": output_dir / "training_history.json",
    }

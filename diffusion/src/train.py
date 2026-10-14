"""Training and checkpoint resume for highD background diffusion."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from normalizing_flow.src.constraints import derived_modes
from tools.plot_style import (
    GENERATED_COLOR,
    PAPER_FIGURE_DPI,
    PAPER_PANEL_LABELSIZE,
    REAL_COLOR,
    get_pyplot,
    style_axes,
)

from hierarchical_world_model.src.core.utils import (
    ensure_dir,
    load_json,
    save_json,
    select_device,
    set_seed,
)

from .data import (
    BackgroundTrajectoryDataset,
    trajectory_data_contract,
    fit_constraint_statistics,
    load_data_bundle,
    make_loader,
    split_rows,
)
from .model import BackgroundTrajectoryDiffusion, DiffusionModelConfig

LOGGER = logging.getLogger(__name__)
CHECKPOINT_SCHEMA = "highd_background_trajectory_diffusion"


def save_training_curves(history: list[dict[str, float]], path: Path) -> None:
    """Save the compact diagnostics needed to judge convergence."""
    plt = get_pyplot()
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 4, figsize=(13.2, 3.5))
    panels = (
        ("loss", "Total loss"),
        ("denoising_mse", "Denoising objective"),
        ("long_horizon", "Knot residual error"),
        ("trajectory_position", "Position residual loss"),
    )
    for label, axis, (metric, title) in zip("abcd", axes, panels):
        axis.plot(
            epochs,
            [row.get(f"train_{metric}", 0.0) for row in history],
            color=REAL_COLOR,
            label="Train",
        )
        axis.plot(
            epochs,
            [row.get(f"val_{metric}", 0.0) for row in history],
            color=GENERATED_COLOR,
            label="Validation",
        )
        axis.set(title=title, xlabel="Epoch")
        axis.set_yscale("log")
        style_axes(axis)
        axis.text(
            -0.12,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=PAPER_PANEL_LABELSIZE,
            fontweight="bold",
        )
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=PAPER_FIGURE_DPI)
    plt.close(figure)


def _model_config(
    config: dict[str, Any], contract: dict[str, Any]
) -> DiffusionModelConfig:
    model = config["model"]
    diffusion = config["diffusion"]
    loss = config["loss"]
    return DiffusionModelConfig(
        condition_dim=int(contract["condition_dim"]),
        c0_dim=int(contract["c0_dim"]),
        constraint_agents=int(contract["constraint_agents"]),
        constraint_dim=len(contract["trajectory_constraint"]["feature_names"]),
        hidden_dim=int(model["hidden_dim"]),
        num_layers=int(model["num_layers"]),
        num_heads=int(model["num_heads"]),
        dropout=float(model["dropout"]),
        diffusion_steps=int(diffusion["steps"]),
        min_snr_gamma=float(diffusion["min_snr_gamma"]),
        x0_clip_abs=float(diffusion["x0_clip_abs"]),
        x0_weight=float(loss["x0_weight"]),
        smooth_weight=float(loss["smooth_weight"]),
        long_horizon_weight=float(loss["long_horizon_weight"]),
        trajectory_position_weight=float(loss["trajectory_position_weight"]),
        trajectory_velocity_weight=float(loss["trajectory_velocity_weight"]),
        target_scale_x=float(contract["position_residual"]["std"][0]),
        target_scale_y=float(contract["position_residual"]["std"][1]),
        dt_s=float(contract["dt_s"]),
    )


def prepare_training_data(
    config: dict[str, Any], config_dir: Path
) -> tuple[Any, dict[str, Any]]:
    output = ensure_dir(config["paths"]["output_dir"])
    contract_path = output / "dataset_contract.json"
    bundle = load_data_bundle(config, config_dir)
    train_rows = split_rows(bundle.arrays, "train")
    condition_mode = str(config.get("dataset", {}).get("condition_mode", ""))
    if condition_mode != "c0_position_knots":
        raise ValueError("diffusion condition_mode must be c0_position_knots")
    if bool(config.get("dataset", {}).get("include_ego_future", False)):
        raise ValueError("dataset.include_ego_future must remain false")
    target_representation = "single_smoothed_pchip_position_residual"
    if config["dataset"]["target_representation"] != target_representation:
        raise ValueError(
            f"diffusion target_representation must be {target_representation}"
        )
    contract = None
    if contract_path.exists():
        saved = load_json(contract_path)
        current = trajectory_data_contract(
            bundle,
            {
                "c0": saved["c0_normalization"],
                "constraint": saved["trajectory_constraint"],
                "position_residual": saved["position_residual"],
            },
        )
        if saved == current:
            contract = saved
    if contract is None:
        sorted_rows = train_rows.copy()
        sorted_rows.sort()
        LOGGER.info(
            "Fitting knot and position-residual statistics on %d train sequences",
            len(train_rows),
        )
        statistics = fit_constraint_statistics(bundle, sorted_rows)
        contract = trajectory_data_contract(
            bundle,
            statistics,
        )
        save_json(contract, contract_path)
    return bundle, contract


def _move(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key
        in {
            "condition",
            "target",
            "target_mask",
        }
    }


def _train_epoch(
    model: BackgroundTrajectoryDiffusion,
    ema_model: BackgroundTrajectoryDiffusion,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    ema_decay: float,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "denoising_mse": 0.0,
        "x0_l1": 0.0,
        "smooth": 0.0,
        "long_horizon": 0.0,
        "trajectory_position": 0.0,
        "trajectory_velocity": 0.0,
    }
    samples = 0
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        losses = model.loss(batch["target"], batch["condition"], batch["target_mask"])
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        _update_ema(ema_model, model, ema_decay)
        size = int(batch["target_mask"].flatten(1).any(dim=1).sum())
        for name in totals:
            totals[name] += float(losses[name].detach()) * size
        samples += size
    return {name: value / max(samples, 1) for name, value in totals.items()}


@torch.no_grad()
def _fixed_validation(
    model: BackgroundTrajectoryDiffusion,
    loader,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "denoising_mse": 0.0,
        "x0_l1": 0.0,
        "smooth": 0.0,
        "long_horizon": 0.0,
        "trajectory_position": 0.0,
        "trajectory_velocity": 0.0,
    }
    samples = 0
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        size = int(batch["target_mask"].flatten(1).any(dim=1).sum())
        timesteps = torch.randint(
            model.config.diffusion_steps,
            (batch["target"].shape[0],),
            device=device,
            generator=generator,
        )
        noise = torch.randn(
            batch["target"].shape,
            device=device,
            dtype=batch["target"].dtype,
            generator=generator,
        )
        losses = model.loss(
            batch["target"],
            batch["condition"],
            batch["target_mask"],
            timesteps=timesteps,
            noise=noise,
        )
        for name in totals:
            totals[name] += float(losses[name]) * size
        samples += size
    return {name: value / max(samples, 1) for name, value in totals.items()}


def _selection_state(
    history: list[dict[str, float]], minimum_delta: float
) -> tuple[float, int]:
    """Reconstruct patience state from durable deterministic-validation history."""
    best = float("inf")
    last_improvement = 0
    for row in history:
        value = float(row["val_loss"])
        if value < best - float(minimum_delta):
            best = value
            last_improvement = int(row["epoch"])
    return best, last_improvement


def _checkpoint(
    model: BackgroundTrajectoryDiffusion,
    optimizer: torch.optim.Optimizer,
    scheduler,
    contract: dict[str, Any],
    epoch: int,
    best: float,
    ema_model: BackgroundTrajectoryDiffusion,
) -> dict[str, Any]:
    return {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "epoch": int(epoch),
        "best_validation_loss": float(best),
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "ema_model_state": ema_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "dataset_contract": contract,
    }


def _inference_checkpoint(state: dict[str, Any]) -> dict[str, Any]:
    """Keep only the EMA sampling weights and their exact data contract."""
    return {
        "checkpoint_schema": state["checkpoint_schema"],
        "epoch": state["epoch"],
        "best_validation_loss": state["best_validation_loss"],
        "model_config": state["model_config"],
        "model_state": state["ema_model_state"],
        "dataset_contract": state["dataset_contract"],
    }


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[BackgroundTrajectoryDiffusion, dict[str, Any]]:
    state = torch.load(Path(path), map_location=device, weights_only=False)
    if state.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"not a {CHECKPOINT_SCHEMA} checkpoint: {path}")
    model_config = dict(state["model_config"])
    model = BackgroundTrajectoryDiffusion(DiffusionModelConfig(**model_config)).to(
        device
    )
    model.load_state_dict(state["model_state"])
    return model, state


@torch.no_grad()
def _update_ema(
    ema_model: BackgroundTrajectoryDiffusion,
    model: BackgroundTrajectoryDiffusion,
    decay: float,
) -> None:
    """Update sampling weights without tracking optimizer-scale noise."""
    factor = float(decay)
    if not 0.0 <= factor < 1.0:
        raise ValueError("training.ema_decay must be in [0, 1)")
    online = dict(model.named_parameters())
    for name, target in ema_model.named_parameters():
        target.lerp_(online[name].detach(), 1.0 - factor)
    online_buffers = dict(model.named_buffers())
    for name, target in ema_model.named_buffers():
        target.copy_(online_buffers[name])


def train_background_diffusion(
    config: dict[str, Any],
    *,
    config_dir: Path,
    resume: Path | None = None,
) -> dict[str, Any]:
    training = config["training"]
    output = ensure_dir(config["paths"]["output_dir"])
    checkpoint_dir = output / "checkpoints"
    last_path = checkpoint_dir / "last_training_state.pt"
    best_path = checkpoint_dir / "best_background_diffusion.pt"
    set_seed(int(training.get("seed", 42)))
    torch.set_float32_matmul_precision("high")
    device = select_device(training.get("device", "auto"))
    bundle, contract = prepare_training_data(config, config_dir)
    train_rows = split_rows(bundle.arrays, "train")
    val_rows = split_rows(bundle.arrays, "val")
    flow_rows = bundle.flow_row_for_sequence[train_rows]
    train_has_background = np.asarray(bundle.flow_arrays["slot_mask"])[flow_rows].any(
        axis=1
    )
    val_flow_rows = bundle.flow_row_for_sequence[val_rows]
    val_has_background = np.asarray(bundle.flow_arrays["slot_mask"])[val_flow_rows].any(
        axis=1
    )
    modes = derived_modes(
        np.asarray(bundle.flow_arrays["trajectory_constraint"])[flow_rows],
        np.asarray(bundle.flow_arrays["slot_mask"])[flow_rows],
    )
    sampling_manifest = {
        "sampling": "full_recording_level_train_split",
        "unique_natural_sequences": int(len(train_rows)),
        "evt_tail_unique_sequences": int(
            np.asarray(bundle.arrays["is_evt_tail"])[train_rows].sum()
        ),
        "lane_change_unique_sequences": int((modes[..., 1] != 0).any(1).sum()),
        "epoch_training_samples": int(len(train_rows)),
        "sequences_with_background_targets": int(train_has_background.sum()),
    }
    train_dataset = BackgroundTrajectoryDataset(bundle, train_rows, contract)
    val_dataset = BackgroundTrajectoryDataset(bundle, val_rows, contract)
    train_loader = make_loader(
        train_dataset,
        batch_size=int(training.get("batch_size", 128)),
        shuffle=True,
        workers=int(training.get("num_workers", 4)),
        seed=int(training.get("seed", 42)),
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=int(training.get("val_batch_size", 128)),
        shuffle=False,
        workers=int(training.get("num_workers", 4)),
        seed=0,
    )
    model = BackgroundTrajectoryDiffusion(_model_config(config, contract)).to(device)
    ema_model = copy.deepcopy(model).requires_grad_(False).eval()
    ema_decay = float(training.get("ema_decay", 0.999))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 3.0e-4)),
        weight_decay=float(training.get("weight_decay", 1.0e-4)),
    )
    epochs = int(training.get("epochs", 100))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(training.get("minimum_learning_rate", 3.0e-5)),
    )
    checkpoint_dir = ensure_dir(checkpoint_dir)
    history_path = output / "training.json"
    start_epoch = 0
    best = float("inf")
    history: list[dict[str, float]] = []
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=False)
        required_training_state = {"optimizer_state", "scheduler_state"}
        if not required_training_state.issubset(state):
            raise ValueError(
                "--resume requires an unfinished training-state checkpoint; "
                "a compact final checkpoint cannot resume optimization"
            )
        if state["dataset_contract"] != contract:
            raise ValueError("cannot resume with a different dataset contract")
        if state["model_config"] != model.config.to_dict():
            raise ValueError(
                "resume checkpoint model config does not match the run config"
            )
        model.load_state_dict(state["model_state"])
        ema_model.load_state_dict(state["ema_model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state["epoch"])
        best = float(state["best_validation_loss"])
        history = load_json(history_path)["history"]
    early_stopping_patience = max(int(training.get("early_stopping_patience", 0)), 0)
    early_stopping_minimum_delta = max(
        float(training.get("early_stopping_minimum_delta", 0.0)), 0.0
    )
    last_improvement_epoch = start_epoch
    if history:
        best, last_improvement_epoch = _selection_state(
            history, early_stopping_minimum_delta
        )
    LOGGER.info(
        "Training background diffusion on %s: train=%d val=%d epochs=%d resume_epoch=%d",
        device,
        len(train_dataset),
        len(val_dataset),
        epochs,
        start_epoch,
    )
    completed_epoch = start_epoch
    early_stopped = False
    for epoch in range(start_epoch + 1, epochs + 1):
        train_metrics = _train_epoch(
            model,
            ema_model,
            train_loader,
            optimizer,
            device,
            float(training.get("grad_clip", 1.0)),
            ema_decay,
        )
        validation = _fixed_validation(
            ema_model,
            val_loader,
            device,
            int(training.get("fixed_validation_seed", 12345)),
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        save_json({"history": history}, history_path)
        improved = validation["loss"] < best - early_stopping_minimum_delta
        if improved:
            best = validation["loss"]
            last_improvement_epoch = epoch
        state = _checkpoint(
            model,
            optimizer,
            scheduler,
            contract,
            epoch,
            best,
            ema_model=ema_model,
        )
        torch.save(state, last_path)
        if improved:
            torch.save(_inference_checkpoint(state), best_path)
        LOGGER.info(
            "epoch=%d train=%.5f val=%.5f best=%.5f",
            epoch,
            train_metrics["loss"],
            validation["loss"],
            best,
        )
        completed_epoch = epoch
        if (
            early_stopping_patience > 0
            and epoch - last_improvement_epoch >= early_stopping_patience
        ):
            early_stopped = True
            LOGGER.info(
                "early stopping at epoch=%d after %d epochs without "
                "validation improvement >= %.3g",
                epoch,
                early_stopping_patience,
                early_stopping_minimum_delta,
            )
            break
    summary = {
        "status": "complete",
        "requested_epochs": epochs,
        "epochs": completed_epoch,
        "early_stopped": early_stopped,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_minimum_delta": early_stopping_minimum_delta,
        "best_epoch": last_improvement_epoch,
        "best_validation_loss": best,
        "best_checkpoint": "checkpoints/best_background_diffusion.pt",
        "train_sequences": len(train_rows),
        "train_epoch_samples": len(train_dataset),
        "validation_sequences": len(val_dataset),
        "train_sequences_with_background_targets": int(train_has_background.sum()),
        "validation_sequences_with_background_targets": int(val_has_background.sum()),
        "lane_change_train_sequences": int(
            sampling_manifest["lane_change_unique_sequences"]
        ),
        "experiment_scope": "full",
        "full_training": True,
        "ema_decay": ema_decay,
        "validation_and_sampling_weights": "EMA weights",
        "sampling": sampling_manifest,
    }
    save_json({"summary": summary, "history": history}, history_path)
    save_training_curves(
        history,
        ensure_dir(output / "figures") / "training_curves.png",
    )
    last_path.unlink(missing_ok=True)
    return summary

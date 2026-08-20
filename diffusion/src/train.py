"""Training and checkpoint resume for highD background diffusion."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from normalizing_flow.src.constraints import derived_modes

from world_model.src.core.utils import (
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
    pilot_rows,
    split_rows,
)
from .model import BackgroundTrajectoryDiffusion, DiffusionModelConfig

LOGGER = logging.getLogger(__name__)
CHECKPOINT_SCHEMA = "highd_background_trajectory_diffusion"


def _tensorboard_writer(path: Path):
    """Defer the optional TensorBoard import until logging is enabled."""
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(str(path))


def save_training_curves(history: list[dict[str, float]], path: Path) -> None:
    """Save the compact diagnostics needed to judge convergence."""
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    panels = (
        ("loss", "total loss"),
        ("denoising_mse", "denoising objective"),
        ("long_horizon", "state-knot residual"),
        ("trajectory_position", "position residual"),
    )
    for axis, (metric, title) in zip(axes, panels):
        axis.plot(
            epochs, [row.get(f"train_{metric}", 0.0) for row in history], label="train"
        )
        axis.plot(
            epochs,
            [row.get(f"val_{metric}", 0.0) for row in history],
            label="validation",
        )
        axis.set(title=title, xlabel="epoch")
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _model_config(
    config: dict[str, Any], contract: dict[str, Any] | None = None
) -> DiffusionModelConfig:
    model = config.get("model", {})
    diffusion = config.get("diffusion", {})
    loss = config.get("loss", {})
    return DiffusionModelConfig(
        condition_dim=int((contract or {}).get("condition_dim", 118)),
        constraint_agents=int((contract or {}).get("constraint_agents", 6)),
        constraint_dim=len(
            (contract or {})
            .get("trajectory_constraint", {})
            .get("feature_names", range(12))
        ),
        denoiser=str(model.get("denoiser", "factorized_spatiotemporal")),
        hidden_dim=int(model.get("hidden_dim", 128)),
        num_layers=int(model.get("num_layers", 4)),
        num_heads=int(model.get("num_heads", 4)),
        dropout=float(model.get("dropout", 0.1)),
        diffusion_steps=int(diffusion.get("steps", 100)),
        prediction_type=str(diffusion.get("prediction_type", "epsilon")),
        condition_dropout_prob=float(diffusion.get("condition_dropout_prob", 0.0)),
        min_snr_gamma=float(diffusion.get("min_snr_gamma", 0.0)),
        x0_clip_abs=float(diffusion.get("x0_clip_abs", 10.0)),
        x0_weight=float(loss.get("x0_weight", 0.1)),
        smooth_weight=float(loss.get("smooth_weight", 0.0)),
        long_horizon_weight=float(loss.get("long_horizon_weight", 0.0)),
        trajectory_position_weight=float(loss.get("trajectory_position_weight", 0.0)),
        trajectory_velocity_weight=float(loss.get("trajectory_velocity_weight", 0.0)),
        target_scale_x=float(
            (contract or {}).get("position_residual", {}).get("std", [1.0, 1.0])[0]
        ),
        target_scale_y=float(
            (contract or {}).get("position_residual", {}).get("std", [1.0, 1.0])[1]
        ),
        dt_s=float((contract or {}).get("dt_s", 0.04)),
    )


def prepare_training_data(
    config: dict[str, Any], config_dir: Path
) -> tuple[Any, dict[str, Any]]:
    output = ensure_dir(config["paths"]["output_dir"])
    contract_path = output / "dataset_contract.json"
    bundle = load_data_bundle(config, config_dir)
    train_rows = split_rows(bundle.arrays, "train")
    condition_mode = str(config.get("dataset", {}).get("condition_mode", ""))
    if condition_mode != "c0_long_horizon_state_knots":
        raise ValueError("diffusion condition_mode must be c0_long_horizon_state_knots")
    if bool(config.get("dataset", {}).get("include_ego_future", False)):
        raise ValueError("dataset.include_ego_future must remain false")
    target_representation = "smooth_reference_relative_dx_dy_residual"
    if contract_path.exists():
        contract = load_json(contract_path)
        compatible = (
            contract.get("target_representation") == target_representation
            and contract.get("condition_mode") == "c0_long_horizon_state_knots"
            and not bool(contract.get("ego_future_in_condition", True))
        )
        if compatible:
            current = trajectory_data_contract(
                bundle,
                {
                    "constraint": contract["trajectory_constraint"],
                    "position_residual": contract["position_residual"],
                },
            )
            for key in ("sequence_manifest_sha256",):
                if contract.get(key) != current[key]:
                    LOGGER.warning(
                        "Dataset contract no longer matches %s; refitting "
                        "statistics for the current canonical data",
                        key,
                    )
                    contract = None
                    break
        else:
            contract = None
    else:
        contract = None
    if contract is None:
        sorted_rows = train_rows.copy()
        sorted_rows.sort()
        LOGGER.info(
            "Fitting trajectory-knot and smooth-position statistics on %d train sequences",
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


def _expanded_training_rows(
    rows: np.ndarray,
    arrays: dict[str, np.ndarray],
    semantic_cutin: np.ndarray,
    training: dict[str, Any],
) -> tuple[np.ndarray, dict[str, int]]:
    """Retain every natural row and add a small number of rare repeats."""
    selected = np.asarray(rows, dtype=np.int64)
    cutin = np.asarray(semantic_cutin, dtype=bool)
    if cutin.shape != selected.shape:
        raise ValueError("semantic cut-in mask must align with training rows")
    rare = training.get("rare_cohort_sampling", {})
    evt_repeats = max(int(rare.get("evt_tail_extra_repeats", 0)), 0)
    cutin_repeats = max(int(rare.get("semantic_cutin_extra_repeats", 0)), 0)
    tail = np.asarray(arrays["is_evt_tail"])[selected].astype(bool)
    parts = [selected]
    if evt_repeats:
        parts.extend([selected[tail]] * evt_repeats)
    if cutin_repeats:
        parts.extend([selected[cutin]] * cutin_repeats)
    expanded = np.concatenate(parts)
    np.random.default_rng(int(training.get("seed", 42))).shuffle(expanded)
    return expanded, {
        "unique_natural_sequences": int(len(selected)),
        "evt_tail_unique_sequences": int(tail.sum()),
        "semantic_cutin_unique_sequences": int(cutin.sum()),
        "evt_tail_extra_repeats": evt_repeats,
        "semantic_cutin_extra_repeats": cutin_repeats,
        "epoch_training_samples": int(len(expanded)),
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
        size = int(batch["target"].shape[0])
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
    generator = (
        torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
    )
    generator.manual_seed(int(seed))
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        size = int(batch["target"].shape[0])
        timesteps = torch.randint(
            model.config.diffusion_steps,
            (size,),
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
    ema_model: BackgroundTrajectoryDiffusion | None = None,
) -> dict[str, Any]:
    return {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "epoch": int(epoch),
        "best_validation_loss": float(best),
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "ema_model_state": (
            model.state_dict() if ema_model is None else ema_model.state_dict()
        ),
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
    use_ema: bool = True,
) -> tuple[BackgroundTrajectoryDiffusion, dict[str, Any]]:
    state = torch.load(Path(path), map_location=device, weights_only=False)
    if state.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"not a {CHECKPOINT_SCHEMA} checkpoint: {path}")
    model_config = dict(state["model_config"])
    model = BackgroundTrajectoryDiffusion(DiffusionModelConfig(**model_config)).to(
        device
    )
    weights = (
        state.get("ema_model_state", state["model_state"])
        if use_ema
        else state["model_state"]
    )
    model.load_state_dict(weights)
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
    warm_start: Path | None = None,
) -> dict[str, Any]:
    training = config["training"]
    output = ensure_dir(config["paths"]["output_dir"])
    checkpoint_dir = output / "checkpoints"
    last_path = checkpoint_dir / "last_training_state.pt"
    best_path = checkpoint_dir / "best_background_diffusion.pt"
    existing = last_path if last_path.exists() else best_path
    if resume is None and warm_start is None and existing.exists():
        raise FileExistsError(
            f"refusing to overwrite existing checkpoint: {existing}; "
            "use --resume, --warm-start or a new output directory"
        )
    set_seed(int(training.get("seed", 42)))
    device = select_device(training.get("device", "auto"))
    bundle, contract = prepare_training_data(config, config_dir)
    maximum_train = int(config.get("dataset", {}).get("max_train_sequences", 0))
    maximum_val = int(config.get("dataset", {}).get("max_val_sequences", 2048))
    train_rows = pilot_rows(
        bundle,
        "train",
        maximum=maximum_train,
        seed=int(training.get("seed", 42)),
    )
    val_rows = pilot_rows(
        bundle,
        "val",
        maximum=maximum_val,
        seed=int(training.get("fixed_validation_seed", 12345)),
    )
    epoch_train_rows = train_rows
    flow_rows = bundle.flow_row_for_sequence[train_rows]
    modes = derived_modes(
        np.asarray(bundle.flow_arrays["trajectory_constraint"])[flow_rows],
        np.asarray(bundle.flow_arrays["slot_mask"])[flow_rows],
    )
    sampling_manifest = {
        "sampling": (
            "full_recording_level_train_split"
            if maximum_train <= 0
            else "fixed_size_stratified_pilot"
        ),
        "unique_natural_sequences": int(len(train_rows)),
        "evt_tail_unique_sequences": int(
            np.asarray(bundle.arrays["is_evt_tail"])[train_rows].sum()
        ),
        "lane_change_unique_sequences": int(
            (modes[..., 1] != 0).any(1).sum()
        ),
        "epoch_training_samples": int(len(train_rows)),
    }
    train_dataset = BackgroundTrajectoryDataset(bundle, epoch_train_rows, contract)
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
    history_path = output / "training_history.json"
    start_epoch = 0
    best = float("inf")
    history: list[dict[str, float]] = []
    warm_start_epoch: int | None = None
    if warm_start is not None:
        restored, state = load_checkpoint(warm_start, device=device)
        if state["dataset_contract"] != contract:
            raise ValueError("cannot warm-start with a different dataset contract")
        structural_fields = (
            "condition_dim",
            "horizon_steps",
            "target_dim",
            "hidden_dim",
            "num_layers",
            "num_heads",
            "diffusion_steps",
            "prediction_type",
            "constraint_agents",
            "constraint_dim",
            "denoiser",
        )
        if any(
            getattr(restored.config, name) != getattr(model.config, name)
            for name in structural_fields
        ):
            raise ValueError(
                "cannot warm-start across incompatible diffusion structures"
            )
        model.load_state_dict(restored.state_dict())
        ema_model.load_state_dict(restored.state_dict())
        warm_start_epoch = int(state["epoch"])
    if resume is not None:
        restored, state = load_checkpoint(resume, device=device, use_ema=False)
        required_training_state = {"optimizer_state", "scheduler_state"}
        if not required_training_state.issubset(state):
            raise ValueError(
                "--resume requires an unfinished training-state checkpoint; "
                "use --warm-start with a final checkpoint"
            )
        if state["dataset_contract"] != contract:
            raise ValueError("cannot resume with a different dataset contract")
        if restored.config != model.config:
            raise ValueError(
                "resume checkpoint model config does not match the run config"
            )
        model.load_state_dict(restored.state_dict())
        ema_model.load_state_dict(state.get("ema_model_state", state["model_state"]))
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state["epoch"])
        best = float(state["best_validation_loss"])
        history = list(load_json(history_path)) if history_path.exists() else []
    early_stopping_patience = max(int(training.get("early_stopping_patience", 0)), 0)
    early_stopping_minimum_delta = max(
        float(training.get("early_stopping_minimum_delta", 0.0)), 0.0
    )
    last_improvement_epoch = start_epoch
    if history:
        best, last_improvement_epoch = _selection_state(
            history, early_stopping_minimum_delta
        )
    writer = (
        _tensorboard_writer(output / "tensorboard")
        if training.get("tensorboard", True)
        else None
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
    try:
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
            save_json(history, history_path)
            if writer is not None:
                for name, value in row.items():
                    if name != "epoch":
                        writer.add_scalar(f"background_diffusion/{name}", value, epoch)
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
    finally:
        if writer is not None:
            writer.close()
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
        "lane_change_train_sequences": int(
            sampling_manifest["lane_change_unique_sequences"]
        ),
        "experiment_scope": (
            "full" if maximum_train <= 0 and maximum_val <= 0 else "pilot"
        ),
        "full_training": bool(maximum_train <= 0 and maximum_val <= 0),
        "warm_start_checkpoint": str(warm_start) if warm_start is not None else None,
        "warm_start_epoch": warm_start_epoch,
        "ema_decay": ema_decay,
        "validation_and_sampling_weights": "EMA weights",
        "sampling": sampling_manifest,
    }
    save_json(summary, output / "training_summary.json")
    save_training_curves(history, output / "training_curves.png")
    last_path.unlink(missing_ok=True)
    return summary

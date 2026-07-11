"""Training loop for naturalistic action diffusion priors."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import SPLIT_TO_INDEX, build_action_dataset, load_normalized_dataset
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
) -> DataLoader:
    mask = arrays["split_index"] == SPLIT_TO_INDEX[split]
    if not np.any(mask):
        raise RuntimeError(f"No samples for split={split}")
    idx = np.where(mask)[0]
    if max_samples and max_samples > 0:
        idx = idx[: int(max_samples)]
    relative = arrays.get("relative_history")
    if relative is None:
        relative = np.zeros((arrays["context_states"].shape[0], arrays["context_states"].shape[1], 6), dtype=np.float32)
    tensors = (
        torch.from_numpy(arrays["context_states"][idx]).float(),
        torch.from_numpy(arrays["context_features"][idx]).float(),
        torch.from_numpy(relative[idx]).float(),
        torch.from_numpy(arrays["actions"][idx]).float(),
    )
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=int(batch_size),
        shuffle=shuffle,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
    )


def _epoch(
    model: GaussianActionDiffusion,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: float = 0.0,
) -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals: Dict[str, float] = {}
    total_n = 0
    for history, context, relative, actions in loader:
        history = history.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)
        relative = relative.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            losses = model.p_losses(actions, history, context, relative)
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
    history: torch.Tensor,
    context: torch.Tensor,
    relative: torch.Tensor,
    timestep: int,
    noise_seed: int,
) -> Dict[str, torch.Tensor]:
    t = torch.full((actions.shape[0],), int(timestep), device=actions.device, dtype=torch.long)
    generator = _torch_generator_for(actions.device, int(noise_seed))
    noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype, generator=generator)
    noisy = model.q_sample(actions, t, noise)
    pred = model.denoiser(noisy, t, history, context, relative)
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
    for batch_idx, (history, context, relative, actions) in enumerate(loader):
        history = history.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)
        relative = relative.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        n = int(actions.shape[0])
        for offset, timestep in enumerate(timesteps):
            losses = _fixed_noise_losses(
                model,
                actions,
                history,
                context,
                relative,
                timestep,
                int(noise_seed) + batch_idx * 1009 + offset * 9173,
            )
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * n
            total_n += n
    return {key: value / max(total_n, 1) for key, value in totals.items()}


def _make_writer(output_dir: Path, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # noqa: BLE001
        logger.warning("TensorBoard is unavailable: %s", exc)
        return None
    return SummaryWriter(log_dir=str(output_dir / "runs"))


def _write_minimal_tensorboard(
    writer,
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    fixed_val_metrics: Dict[str, float],
) -> None:
    """Only the essential training signals, by design."""
    if writer is None:
        return
    for key, value in train_metrics.items():
        writer.add_scalar(f"loss/train_{key}", value, epoch)
    for key, value in val_metrics.items():
        writer.add_scalar(f"loss/val_{key}", value, epoch)
    for key, value in fixed_val_metrics.items():
        writer.add_scalar(f"loss/fixed_val_{key}", value, epoch)


def _write_history_csv(path: Path, history: list[dict]) -> None:
    if not history:
        return
    keys: list[str] = []
    for row in history:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def _fixed_timesteps_from_config(training: dict, model: GaussianActionDiffusion) -> list[int]:
    raw = training.get("fixed_val_timesteps", [0, 25, 50, 75, 99])
    if isinstance(raw, str):
        raw = [x.strip() for x in raw.split(",") if x.strip()]
    out = sorted({max(0, min(int(t), model.num_steps - 1)) for t in raw})
    return out or [0, model.num_steps - 1]


def train_action_diffusion(config: dict, *, config_dir: str | Path | None = None) -> dict:
    paths = config.get("paths", {})
    base = Path(config_dir).resolve() if config_dir is not None else Path.cwd()
    output_dir = (base / paths.get("output_dir", "../../../data/diffusion_natural/following")).resolve()
    dataset_path = output_dir / "dataset_normalized.npz"
    if bool(config.get("dataset", {}).get("rebuild", False)) or not dataset_path.exists():
        build_action_dataset(config, config_dir=base)

    schema = load_json(output_dir / "feature_schema.json")
    arrays = load_normalized_dataset(output_dir)
    training = config.get("training", {})
    set_seed(int(training.get("seed", 42)))
    device = select_device(training.get("device", "auto"))
    model = build_model_from_schema(schema, config).to(device)

    batch_size = int(training.get("batch_size", 256))
    num_workers = int(training.get("num_workers", 0))
    train_loader = _make_loader(arrays, "train", batch_size, True, num_workers)
    val_loader = _make_loader(arrays, "val", batch_size, False, num_workers)
    fixed_val_max_samples = int(training.get("fixed_val_max_samples", 512))
    fixed_val_loader = _make_loader(arrays, "val", batch_size, False, num_workers, fixed_val_max_samples)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("lr", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    epochs = int(training.get("epochs", 160))
    grad_clip = float(training.get("grad_clip", 1.0))
    min_lr = float(training.get("min_lr", 5e-5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - 1), eta_min=min_lr)
    fixed_val_timesteps = _fixed_timesteps_from_config(training, model)
    fixed_val_seed = int(training.get("fixed_val_seed", 12345))
    writer = _make_writer(output_dir, bool(training.get("tensorboard", True)))
    best_val = float("inf")
    best_noise_mse = float("inf")
    history: list[dict] = []
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training on %s for %d epochs; samples=%d", device, epochs, int(arrays["actions"].shape[0]))
    for epoch in range(1, epochs + 1):
        train_metrics = _epoch(model, train_loader, device, optimizer, grad_clip)
        with torch.no_grad():
            val_metrics = _epoch(model, val_loader, device, None)
        fixed_val_metrics = _deterministic_epoch(model, fixed_val_loader, device, fixed_val_timesteps, fixed_val_seed)
        _write_minimal_tensorboard(writer, epoch, train_metrics, val_metrics, fixed_val_metrics)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "fixed_val_loss": fixed_val_metrics["loss"],
            "train_noise_mse": train_metrics.get("noise_mse", train_metrics["loss"]),
            "val_noise_mse": val_metrics.get("noise_mse", val_metrics["loss"]),
            "fixed_val_noise_mse": fixed_val_metrics.get("noise_mse", fixed_val_metrics["loss"]),
            "train_x0_l1": train_metrics.get("x0_l1", 0.0),
            "val_x0_l1": val_metrics.get("x0_l1", 0.0),
            "fixed_val_x0_l1": fixed_val_metrics.get("x0_l1", 0.0),
            "train_smooth": train_metrics.get("smooth", 0.0),
            "val_smooth": val_metrics.get("smooth", 0.0),
            "fixed_val_smooth": fixed_val_metrics.get("smooth", 0.0),
        }
        history.append(row)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "schema": schema,
                    "config": config,
                    "epoch": epoch,
                    "val_loss": best_val,
                },
                checkpoint_dir / "best.pt",
            )
        val_noise_mse = float(val_metrics.get("noise_mse", val_metrics["loss"]))
        if val_noise_mse < best_noise_mse:
            best_noise_mse = val_noise_mse
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "schema": schema,
                    "config": config,
                    "epoch": epoch,
                    "val_noise_mse": best_noise_mse,
                    "val_loss": val_metrics["loss"],
                },
                checkpoint_dir / "best_noise_mse.pt",
            )
        if epoch == 1 or epoch % int(training.get("log_every_epochs", 10)) == 0 or epoch == epochs:
            logger.info(
                "epoch=%03d train_noise_mse=%.6f val_noise_mse=%.6f",
                epoch,
                train_metrics.get("noise_mse", train_metrics["loss"]),
                val_metrics.get("noise_mse", val_metrics["loss"]),
            )
        scheduler.step()

    torch.save(
        {
            "model_state": model.state_dict(),
            "schema": schema,
            "config": config,
            "epoch": epochs,
            "val_loss": history[-1]["val_loss"],
        },
        checkpoint_dir / "last.pt",
    )
    save_json(history, output_dir / "training_history.json")
    _write_history_csv(output_dir / "training_history.csv", history)
    save_json(
        {
            "best_val_loss": best_val,
            "best_val_noise_mse": best_noise_mse,
            "history": history,
            "lr_schedule": "cosine",
            "min_lr": min_lr,
            "fixed_val_timesteps": fixed_val_timesteps,
            "fixed_val_seed": fixed_val_seed,
            "fixed_val_max_samples": fixed_val_max_samples,
        },
        output_dir / "training_summary.json",
    )
    if writer is not None:
        writer.close()
    return {"output_dir": output_dir, "best_val_loss": best_val, "epochs": epochs}

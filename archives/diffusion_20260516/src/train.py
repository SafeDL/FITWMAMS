"""Training loop for event-specific action diffusion."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

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
    training_cfg: dict | None = None,
) -> DataLoader:
    mask = arrays["split_index"] == SPLIT_TO_INDEX[split]
    if not np.any(mask):
        raise RuntimeError(f"No samples for split={split}")
    n = int(np.sum(mask))
    relative = arrays.get("relative_history")
    if relative is None:
        relative = np.zeros((arrays["context_states"].shape[0], arrays["context_states"].shape[1], 6), dtype=np.float32)
    risk_condition = arrays.get("risk_condition")
    if risk_condition is None:
        risk_condition = arrays["risk"].reshape(-1, 1)
    tensors = (
        torch.from_numpy(arrays["context_states"][mask]).float(),
        torch.from_numpy(arrays["context_features"][mask]).float(),
        torch.from_numpy(relative[mask]).float(),
        torch.from_numpy(risk_condition[mask]).float(),
        torch.from_numpy(arrays["actions"][mask]).float(),
    )
    sampler = None
    if shuffle and split == "train" and training_cfg and bool(training_cfg.get("risk_stratified_sampling", False)):
        pct = arrays.get("risk_percentile")
        if pct is None:
            risk = arrays.get("risk_raw", arrays["risk"])
            pct = np.argsort(np.argsort(risk)).astype(np.float32) / max(len(risk) - 1, 1)
        p = np.asarray(pct[mask], dtype=np.float32)
        edges = np.asarray(training_cfg.get("risk_sampling_bins", [0.5, 0.8, 0.9, 0.95, 0.99]), dtype=np.float32)
        buckets = np.digitize(p, edges, right=True)
        counts = np.bincount(buckets, minlength=len(edges) + 1).astype(np.float64)
        counts[counts == 0.0] = 1.0
        weights = 1.0 / counts[buckets]
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=n, replacement=True)
        shuffle = False
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=int(batch_size),
        shuffle=shuffle,
        sampler=sampler,
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
    for history, context, relative, risk_condition, actions in loader:
        history = history.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)
        relative = relative.to(device, non_blocking=True)
        risk_condition = risk_condition.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            losses = model.p_losses(actions, history, context, relative, risk_condition)
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


def _make_writer(output_dir: Path, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # noqa: BLE001
        logger.warning("TensorBoard is unavailable: %s", exc)
        return None
    return SummaryWriter(log_dir=str(output_dir / "runs"))


def _write_minimal_tensorboard(writer, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
    """Only the essential training signals, by design."""
    if writer is None:
        return
    for key, value in train_metrics.items():
        writer.add_scalar(f"loss/train_{key}", value, epoch)
    for key, value in val_metrics.items():
        writer.add_scalar(f"loss/val_{key}", value, epoch)


def train_action_diffusion(config: dict, *, config_dir: str | Path | None = None) -> dict:
    paths = config.get("paths", {})
    base = Path(config_dir).resolve() if config_dir is not None else Path.cwd()
    output_dir = (base / paths.get("output_dir", "../../../data/diffusion/following")).resolve()
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
    train_loader = _make_loader(arrays, "train", batch_size, True, num_workers, training)
    val_loader = _make_loader(arrays, "val", batch_size, False, num_workers, training)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("lr", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    epochs = int(training.get("epochs", 100))
    grad_clip = float(training.get("grad_clip", 1.0))
    writer = _make_writer(output_dir, bool(training.get("tensorboard", True)))
    best_val = float("inf")
    history: list[dict] = []
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training on %s for %d epochs; samples=%d", device, epochs, int(arrays["actions"].shape[0]))
    for epoch in range(1, epochs + 1):
        train_metrics = _epoch(model, train_loader, device, optimizer, grad_clip)
        with torch.no_grad():
            val_metrics = _epoch(model, val_loader, device, None)
        lr = float(optimizer.param_groups[0]["lr"])
        _write_minimal_tensorboard(writer, epoch, train_metrics, val_metrics)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_noise_mse": train_metrics.get("noise_mse", train_metrics["loss"]),
            "val_noise_mse": val_metrics.get("noise_mse", val_metrics["loss"]),
            "train_x0_l1": train_metrics.get("x0_l1", 0.0),
            "val_x0_l1": val_metrics.get("x0_l1", 0.0),
            "train_smooth": train_metrics.get("smooth", 0.0),
            "val_smooth": val_metrics.get("smooth", 0.0),
            "lr": lr,
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
        if epoch == 1 or epoch % int(training.get("log_every_epochs", 10)) == 0 or epoch == epochs:
            logger.info(
                "epoch=%03d train_noise_mse=%.6f val_noise_mse=%.6f",
                epoch,
                train_metrics.get("noise_mse", train_metrics["loss"]),
                val_metrics.get("noise_mse", val_metrics["loss"]),
            )

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
    save_json({"best_val_loss": best_val, "history": history[-20:]}, output_dir / "training_summary.json")
    if writer is not None:
        writer.close()
    return {"output_dir": output_dir, "best_val_loss": best_val, "epochs": epochs}

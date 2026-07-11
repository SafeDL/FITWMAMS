"""Training loop for the Stage 2 naturalness discriminator."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from diffusion.src.data import SPLIT_TO_INDEX
from diffusion.src.utils import load_json, save_json, select_device, set_seed

from .data import build_discriminator_dataset, load_discriminator_dataset
from .model import NaturalnessDiscriminator, build_discriminator_from_schema

logger = logging.getLogger(__name__)


def _resolve_output_dir(config: dict, config_dir: str | Path | None) -> Path:
    base = Path(config_dir).resolve() if config_dir is not None else Path.cwd()
    return (base / config.get("paths", {}).get("output_dir", "../../../data/diffusion_natural/following/discriminator")).resolve()


def _make_loader(
    arrays: dict[str, np.ndarray],
    split: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    *,
    source_balanced: bool = False,
) -> DataLoader:
    idx = np.where(arrays["split_index"] == SPLIT_TO_INDEX[split])[0]
    if len(idx) == 0:
        raise RuntimeError(f"No discriminator samples for split={split}")
    tensors = (
        torch.from_numpy(arrays["context_states"][idx]).float(),
        torch.from_numpy(arrays["context_features"][idx]).float(),
        torch.from_numpy(arrays["relative_history"][idx]).float(),
        torch.from_numpy(arrays["future_action_features"][idx]).float(),
        torch.from_numpy(arrays["summary_features"][idx]).float(),
        torch.from_numpy(arrays["labels"][idx]).float(),
        torch.from_numpy(arrays["sample_weights"][idx]).float(),
    )
    sampler = None
    if source_balanced:
        source = arrays["source_type"][idx].astype(str)
        _, inverse, counts = np.unique(source, return_inverse=True, return_counts=True)
        weights = 1.0 / np.maximum(counts[inverse], 1)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(idx), replacement=True)
        shuffle = False
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=int(batch_size),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
    )


def _binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    try:
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            brier_score_loss,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("sklearn unavailable; returning minimal metrics: %s", exc)
        pred = scores >= 0.5
        return {"accuracy": float(np.mean(pred == (labels >= 0.5)))}
    pred = scores >= 0.5
    out = {
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "brier": float(brier_score_loss(labels, scores)),
    }
    if len(np.unique(labels)) > 1:
        out["auc"] = float(roc_auc_score(labels, scores))
        out["pr_auc"] = float(average_precision_score(labels, scores))
    else:
        out["auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def _source_metrics(labels: np.ndarray, scores: np.ndarray, source_type: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for source in sorted(np.unique(source_type.astype(str)).tolist()):
        mask = source_type.astype(str) == source
        if not np.any(mask):
            continue
        if np.mean(labels[mask]) >= 0.5:
            out[f"{source}_accept_rate"] = float(np.mean(scores[mask] >= 0.5))
            out[f"{source}_mean_score"] = float(np.mean(scores[mask]))
        else:
            out[f"{source}_reject_rate"] = float(np.mean(scores[mask] < 0.5))
            out[f"{source}_mean_score"] = float(np.mean(scores[mask]))
    return out


def _epoch(
    model: NaturalnessDiscriminator,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 0.0,
    positive_smoothing: float = 0.9,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_n = 0
    for history, context, relative, future, summary, labels, weights in loader:
        history = history.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)
        relative = relative.to(device, non_blocking=True)
        future = future.to(device, non_blocking=True)
        summary = summary.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            logits = model(history, context, relative, future, summary)
            targets = torch.where(labels > 0.5, torch.full_like(labels, float(positive_smoothing)), torch.zeros_like(labels))
            losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            loss = torch.sum(losses * weights) / torch.clamp(torch.sum(weights), min=1.0)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()
        n = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * n
        total_n += n
    return {"loss": total_loss / max(total_n, 1)}


@torch.no_grad()
def _predict_split(
    model: NaturalnessDiscriminator,
    arrays: dict[str, np.ndarray],
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = _make_loader(arrays, split, batch_size, False, num_workers)
    model.eval()
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    for history, context, relative, future, summary, labels, _weights in loader:
        logits = model(
            history.to(device, non_blocking=True),
            context.to(device, non_blocking=True),
            relative.to(device, non_blocking=True),
            future.to(device, non_blocking=True),
            summary.to(device, non_blocking=True),
        )
        logits_list.append(logits.detach().cpu().numpy())
        labels_list.append(labels.numpy())
    idx = np.where(arrays["split_index"] == SPLIT_TO_INDEX[split])[0]
    logits_np = np.concatenate(logits_list, axis=0)
    labels_np = np.concatenate(labels_list, axis=0)
    return labels_np, 1.0 / (1.0 + np.exp(-logits_np)), arrays["source_type"][idx]


def _write_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _make_writer(output_dir: Path, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # noqa: BLE001
        logger.warning("TensorBoard unavailable: %s", exc)
        return None
    return SummaryWriter(str(output_dir / "runs"))


def _hard_negative_score(source_metrics: dict[str, float]) -> float:
    preferred = [
        "rss_over_guided_reject_rate",
        "highway_env_hard_negative_reject_rate",
        "rule_brake_reject_rate",
    ]
    values = [source_metrics[key] for key in preferred if key in source_metrics]
    if values:
        return float(np.mean(values))
    values = [value for key, value in source_metrics.items() if key.endswith("_reject_rate")]
    return float(np.mean(values)) if values else float("-inf")


def train_discriminator(config: dict, *, config_dir: str | Path | None = None) -> dict[str, Any]:
    output_dir = _resolve_output_dir(config, config_dir)
    dataset_path = output_dir / "discriminator_dataset.npz"
    if bool(config.get("data", {}).get("rebuild", False)) or not dataset_path.exists():
        build_discriminator_dataset(config, config_dir=config_dir)
    schema = load_json(output_dir / "discriminator_schema.json")
    arrays = load_discriminator_dataset(output_dir)
    training = config.get("training", {})
    set_seed(int(training.get("seed", 42)))
    device = select_device(training.get("device", "auto"))
    model = build_discriminator_from_schema(schema, config).to(device)
    batch_size = int(training.get("batch_size", 256))
    num_workers = int(training.get("num_workers", 0))
    train_loader = _make_loader(
        arrays,
        "train",
        batch_size,
        True,
        num_workers,
        source_balanced=bool(training.get("source_balanced_sampling", True)),
    )
    val_loader = _make_loader(arrays, "val", batch_size, False, num_workers)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("lr", 1e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    epochs = int(training.get("epochs", 80))
    patience = int(training.get("early_stopping_patience", 10))
    grad_clip = float(training.get("grad_clip", 1.0))
    positive_smoothing = float(training.get("label_smoothing_positive", 0.9))
    writer = _make_writer(output_dir, bool(training.get("tensorboard", True)))
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_auc = float("-inf")
    best_hard = float("-inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []

    logger.info("Training naturalness discriminator on %s for %d epochs", device, epochs)
    for epoch in range(1, epochs + 1):
        train_loss = _epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            grad_clip=grad_clip,
            positive_smoothing=positive_smoothing,
        )["loss"]
        val_loss = _epoch(model, val_loader, device, positive_smoothing=positive_smoothing)["loss"]
        labels, scores, sources = _predict_split(model, arrays, "val", batch_size, num_workers, device)
        metrics = _binary_metrics(labels, scores)
        source_metrics = _source_metrics(labels, scores, sources)
        hard_score = _hard_negative_score(source_metrics)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in metrics.items()}, **source_metrics}
        history.append(row)
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            for key, value in metrics.items():
                writer.add_scalar(f"metrics/val_{key}", value, epoch)
        auc = float(metrics.get("auc", float("-inf")))
        if np.isfinite(auc) and auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            torch.save({"model_state": model.state_dict(), "schema": schema, "config": config, "epoch": epoch, "val_auc": auc}, checkpoint_dir / "best_auc.pt")
        if hard_score > best_hard:
            best_hard = hard_score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "schema": schema,
                    "config": config,
                    "epoch": epoch,
                    "hard_negative_reject_score": hard_score,
                },
                checkpoint_dir / "best_hard_negative.pt",
            )
        if epoch == 1 or epoch % int(training.get("log_every_epochs", 5)) == 0 or epoch == epochs:
            logger.info("epoch=%03d train_loss=%.5f val_loss=%.5f val_auc=%.5f hard_reject=%.5f", epoch, train_loss, val_loss, auc, hard_score)
        if epoch - best_epoch >= patience:
            logger.info("Early stopping at epoch=%d; best_auc_epoch=%d", epoch, best_epoch)
            break

    torch.save({"model_state": model.state_dict(), "schema": schema, "config": config, "epoch": history[-1]["epoch"]}, checkpoint_dir / "last.pt")
    _write_history_csv(output_dir / "training_history.csv", history)
    save_json(
        {
            "best_val_auc": best_auc,
            "best_hard_negative_reject_score": best_hard,
            "best_epoch": best_epoch,
            "epochs_completed": int(history[-1]["epoch"]),
            "history": history,
        },
        output_dir / "training_summary.json",
    )
    if writer is not None:
        writer.close()
    return {"output_dir": output_dir, "best_val_auc": best_auc, "epochs_completed": int(history[-1]["epoch"])}

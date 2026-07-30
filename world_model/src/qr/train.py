"""From-scratch full-cache training for the Query-Refine World Model."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from world_model.src.core.sequential_dataset import load_sequential_dataset, sequence_cache_owner_dir
from world_model.src.core.utils import ensure_dir, save_json, select_device, set_seed
from world_model.src.core.batching import make_sequence_loader, to_device_batch

from .config import QRWorldModelConfig
from .model import QueryRefineWorldModel

logger = logging.getLogger(__name__)


def _config(source: dict[str, Any]) -> QRWorldModelConfig:
    allowed = {key: value for key, value in source.items() if key in QRWorldModelConfig.__dataclass_fields__}
    return QRWorldModelConfig(**allowed)


def _tensorboard_writer(output: Path, training: dict[str, Any]):
    """Create the optional QR-WM TensorBoard writer under the run output."""
    if not bool(training.get("tensorboard", True)):
        return None, None
    from torch.utils.tensorboard import SummaryWriter

    configured = Path(str(training.get("tensorboard_dir", "tensorboard")))
    log_dir = configured if configured.is_absolute() else output / configured
    return SummaryWriter(log_dir=str(ensure_dir(log_dir)), flush_secs=30), log_dir


def _write_tensorboard_epoch(tensorboard_writer: Any, row: dict[str, Any]) -> None:
    """Record scalar loss terms and the selection metric for one epoch."""
    epoch = int(row["epoch"])
    tensorboard_writer.add_scalar("training/rollout_seconds", float(row["rollout_seconds"]), epoch)
    tensorboard_writer.add_scalar("selection/validation_fde_m", float(row["selection_metric"]), epoch)
    for key, value in row.items():
        if key.startswith("train_"):
            tag = f"epoch/train/{key.removeprefix('train_')}"
        elif key.startswith("val_"):
            tag = f"epoch/validation/{key.removeprefix('val_')}"
        else:
            continue
        scalar = float(value)
        if math.isfinite(scalar):
            tensorboard_writer.add_scalar(tag, scalar, epoch)
    tensorboard_writer.flush()


def _roll_fde(model: QueryRefineWorldModel, loader, device: torch.device, *, response_steps: int) -> float:
    model.eval()
    total = count = 0.0
    with torch.no_grad():
        for values in loader:
            rollout = model.rollout(
                to_device_batch(values, loader.field_names, device), response_steps=response_steps, deterministic=True
            )
            predicted, target, valid = rollout["predicted_states"], rollout["target_states"], rollout["target_valid"]
            distance = torch.linalg.vector_norm(predicted[:, -1, 1:, :2] - target[:, -1, 1:, :2], dim=-1)
            final_valid = valid[:, -1, 1:]
            total += float((distance * final_valid.float()).sum().cpu())
            count += float(final_valid.sum().cpu())
    return total / max(count, 1.0)


def _mean_training_terms(
    model: QueryRefineWorldModel, loader, device: torch.device, *, response_steps: int
) -> dict[str, float]:
    """Deterministic validation objective; the posterior remains training-only."""
    totals: dict[str, float] = {}
    batches = 0
    model.eval()
    with torch.no_grad():
        for values in loader:
            result = model.rollout(
                to_device_batch(values, loader.field_names, device), response_steps=response_steps, deterministic=True
            )
            terms = {key: torch.stack([item[key] for item in result["loss_terms"]]).mean() for key in result["loss_terms"][0]}
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + float(value.cpu())
            batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def train_qr_world_model(config: dict[str, Any], *, config_dir: Path) -> dict[str, Any]:
    """Train QR-WM from random initialization on every cached train sequence."""
    training, paths = config.get("training", {}), config["paths"]
    output = Path(paths["output_dir"])
    if not output.is_absolute():
        output = (config_dir / output).resolve()
    checkpoint_dir = ensure_dir(output / "checkpoints")
    device = select_device(str(training.get("device", "auto")))
    set_seed(int(training.get("seed", 42)))
    cache_owner = sequence_cache_owner_dir(config, config_dir=config_dir)
    arrays, manifest = load_sequential_dataset(cache_owner)
    if manifest.get("bounded_development_cache", True):
        raise RuntimeError("formal QR-WM training requires the complete immutable sequence cache")
    if int(config.get("dataset", {}).get("max_sequences", 0)) != 0:
        raise RuntimeError("formal QR-WM training must keep dataset.max_sequences=0")
    workers = int(training.get("num_workers", 0))
    batch_size = int(training.get("batch_size", 64))
    train_loader = make_sequence_loader(
        arrays, "train", batch_size=batch_size, maximum=0, shuffle=True,
        seed=int(training.get("seed", 42)), num_workers=workers,
    )
    val_loader = make_sequence_loader(
        arrays, "val", batch_size=int(training.get("val_batch_size", batch_size)), maximum=0,
        shuffle=False, seed=int(training.get("seed", 42)) + 1, num_workers=workers,
    )
    model = QueryRefineWorldModel(_config(config.get("model", {}))).to(device)
    stages = training.get("stages", [
        {"name": "buffer_warmup", "epochs": 1, "rollout_seconds": 1.0},
        {"name": "full_refinement", "epochs": 3, "rollout_seconds": 5.0},
    ])
    if not stages:
        raise ValueError("training.stages must not be empty")
    configured_epochs = int(training.get("epochs", sum(int(stage["epochs"]) for stage in stages)))
    staged_epochs = sum(int(stage["epochs"]) for stage in stages)
    if configured_epochs != staged_epochs:
        raise ValueError(
            "training.epochs must equal the sum of training.stages[*].epochs "
            f"({configured_epochs} != {staged_epochs})"
        )
    tensorboard_writer, tensorboard_dir = _tensorboard_writer(output, training)
    if tensorboard_writer is not None:
        tensorboard_writer.add_text("run/model_type", QueryRefineWorldModel.model_type, 0)
        tensorboard_writer.add_text("run/training_config", json.dumps(training, sort_keys=True), 0)
        tensorboard_writer.add_scalar("training/configured_epochs", configured_epochs, 0)
    history: list[dict[str, Any]] = []
    best = float("inf")
    best_path = checkpoint_dir / "best_qr_world_model.pt"
    epoch = 0
    for stage in stages:
        name, stage_epochs = str(stage["name"]), int(stage["epochs"])
        if stage_epochs <= 0:
            raise ValueError(f"stage {name!r} must contain at least one epoch")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(stage.get("learning_rate", training.get("learning_rate", 8.0e-5))),
            weight_decay=float(training.get("weight_decay", 1.0e-4)),
        )
        response_steps = min(
            model.cfg.response_steps,
            max(1, round(float(stage.get("rollout_seconds", 5.0)) / model.cfg.response_interval_s)),
        )
        for stage_epoch in range(stage_epochs):
            epoch += 1
            model.train()
            totals: dict[str, float] = {}
            batches = 0
            for values in train_loader:
                batch = to_device_batch(values, train_loader.field_names, device)
                optimizer.zero_grad(set_to_none=True)
                result = model.forward_training(
                    batch, response_steps=response_steps, tbptt_steps=int(training.get("tbptt_response_steps", 5))
                )
                result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("grad_clip", 1.0)))
                optimizer.step()
                if tensorboard_writer is not None:
                    global_step = (epoch - 1) * len(train_loader) + batches
                    tensorboard_writer.add_scalar("batch/train/loss", float(result["loss"].detach().cpu()), global_step)
                for key, value in result.items():
                    if isinstance(value, torch.Tensor) and value.ndim == 0:
                        totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
                batches += 1
            validation = _mean_training_terms(model, val_loader, device, response_steps=response_steps)
            fde = _roll_fde(model, val_loader, device, response_steps=response_steps)
            row = {
                "epoch": epoch, "stage": name, "stage_epoch": stage_epoch + 1,
                "rollout_seconds": response_steps * model.cfg.response_interval_s,
                **{f"train_{key}": value / max(batches, 1) for key, value in totals.items()},
                **{f"val_{key}": value for key, value in validation.items()},
                "selection_metric": fde,
            }
            history.append(row)
            if tensorboard_writer is not None:
                _write_tensorboard_epoch(tensorboard_writer, row)
            logger.info(
                "QR-WM epoch=%d stage=%s train=%.5f val_position=%.5f fde=%.5f",
                epoch, name, row["train_loss"], validation["position"], fde,
            )
            # A one-second warmup checkpoint is never selected as the formal
            # artifact; this preserves the target's long-horizon requirement.
            if response_steps == model.cfg.response_steps and fde < best:
                best = fde
                payload = {
                    **model.checkpoint_payload(), "epoch": epoch, "stage": name,
                    "selection_metric": fde, "validation": validation,
                    "training_protocol": {
                        "from_scratch": True, "uses_baseline_checkpoint": False,
                        "complete_train_cache": True, "traffic_light_inputs": False,
                        "future_encoder_training_only": True,
                    },
                }
                torch.save(payload, best_path)
            save_json(
                {
                    "status": "running", "epoch": epoch, "stage": name,
                    "stage_epoch": stage_epoch + 1, "rollout_seconds": row["rollout_seconds"],
                    "last_epoch": row, "best_validation_fde": best,
                    "best_checkpoint": str(best_path) if best_path.exists() else None,
                    "completed_train_sequences_per_epoch": int(len(train_loader.dataset)),
                }, output / "training_progress.json",
            )
    if not best_path.exists():
        if tensorboard_writer is not None:
            tensorboard_writer.close()
        raise RuntimeError("QR-WM training did not reach a full-horizon stage; no formal checkpoint was produced")
    fields = sorted({key for row in history for key in row})
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=fields)
        csv_writer.writeheader(); csv_writer.writerows(history)
    checkpoint_hash = hashlib.sha256(best_path.read_bytes()).hexdigest()
    report = {
        "model_type": model.model_type, "best_checkpoint": str(best_path),
        "best_validation_fde": best, "epochs_completed": epoch, "configured_epochs": configured_epochs,
        "sequence_cache": manifest, "from_scratch": True,
        "uses_baseline_checkpoint": False, "complete_train_cache": True,
        "model_config": asdict(model.cfg), "checkpoint_sha256": checkpoint_hash,
        "train_sequences_per_epoch": int(len(train_loader.dataset)),
        "validation_sequences_per_epoch": int(len(val_loader.dataset)),
        "tensorboard_log_dir": str(tensorboard_dir) if tensorboard_dir is not None else None,
    }
    save_json(report, output / "training_summary.json")
    save_json(
        {"status": "completed", "epoch": epoch, "best_validation_fde": best,
         "best_checkpoint": str(best_path), "completed_train_sequences_per_epoch": int(len(train_loader.dataset))},
        output / "training_progress.json",
    )
    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar("training/completed", 1, epoch)
        tensorboard_writer.close()
    return report


def load_qr_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> QueryRefineWorldModel:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("model_type") != QueryRefineWorldModel.model_type:
        raise ValueError(f"Not a QR-WM checkpoint: {path}")
    model = QueryRefineWorldModel(_config(dict(payload["model_config"])))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.checkpoint_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return model.to(device).eval()

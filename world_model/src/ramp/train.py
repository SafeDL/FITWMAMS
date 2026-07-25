"""From-scratch staged training for RAMP-WM."""

from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from world_model.src.semi_markov_train import _loader, _to_batch
from world_model.src.initial_behavior_anchor import FrozenLegacyFlowSchema
from world_model.src.sequential_dataset import (
    ensure_frozen_flow_behavior_anchor_cache,
    load_sequential_dataset,
    sequence_cache_owner_dir,
)
from world_model.src.utils import ensure_dir, save_json, select_device, set_seed
from .config import RAMPConfig
from .model import RAMPWorldModel

logger = logging.getLogger(__name__)


def _config(source: dict[str, Any]) -> RAMPConfig:
    return RAMPConfig(
        **{
            key: value
            for key, value in source.items()
            if key in RAMPConfig.__dataclass_fields__
        }
    )


def _mean(
    model: RAMPWorldModel, loader, device, steps: int, active_candidates: int
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    with torch.no_grad():
        for values in loader:
            result = model.forward_training(
                _to_batch(values, loader.field_names, device),
                response_steps=steps,
                active_candidates=active_candidates,
            )
            for key in (
                "loss",
                "prefix_position",
                "prefix_velocity",
                "prefix_control",
                "mixture",
                "overlap",
                "diversity",
                "joint",
            ):
                totals[key] = totals.get(key, 0.0) + float(result[key].cpu())
            batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def _roll_fde(
    model: RAMPWorldModel,
    loader,
    device,
    *,
    response_steps: int,
    active_candidates: int,
) -> float:
    """Deterministic endpoint FDE on the same closed-loop horizon as a stage."""
    model.eval()
    total = count = 0.0
    with torch.no_grad():
        for values in loader:
            result = model._closed_loop(
                _to_batch(values, loader.field_names, device),
                response_steps,
                deterministic=True,
                active_candidates=active_candidates,
            )
            distance = torch.linalg.vector_norm(
                result["predicted_states"][:, -1, 1:, :2]
                - result["target_states"][:, -1, 1:, :2],
                dim=-1,
            )
            valid = result["target_valid"][:, -1, 1:]
            total += float((distance * valid).sum().cpu())
            count += float(valid.sum().cpu())
    return total / max(count, 1.0)


def train_ramp_world_model(
    config: dict[str, Any], *, config_dir: Path
) -> dict[str, Any]:
    training, paths = config.get("training", {}), config["paths"]
    output = Path(paths["output_dir"])
    if not output.is_absolute():
        output = (config_dir / output).resolve()
    checkpoint_dir = ensure_dir(output / "checkpoints")
    device = select_device(training.get("device", "auto"))
    set_seed(int(training.get("seed", 42)))
    arrays, manifest = load_sequential_dataset(
        sequence_cache_owner_dir(config, config_dir=config_dir)
    )
    if manifest.get("bounded_development_cache", True):
        raise RuntimeError(
            "formal RAMP training requires the complete immutable sequence cache"
        )
    schema_path = paths.get("flow_schema")
    if bool(config.get("model", {}).get("use_start_anchor", True)):
        if not schema_path:
            raise ValueError(
                "RAMP START requires paths.flow_schema for the frozen B0 cache contract"
            )
        schema = FrozenLegacyFlowSchema.load(
            Path(schema_path)
            if Path(schema_path).is_absolute()
            else (config_dir / schema_path).resolve()
        )
        arrays.update(
            ensure_frozen_flow_behavior_anchor_cache(
                sequence_cache_owner_dir(config, config_dir=config_dir),
                arrays,
                manifest,
                schema,
            )
        )
    batch_size = int(training.get("batch_size", 64))
    workers = int(training.get("num_workers", 4))
    seed = int(training.get("seed", 42))
    train_loader = _loader(
        arrays,
        "train",
        batch_size=batch_size,
        maximum=int(training.get("max_train_sequences", 0)),
        shuffle=True,
        seed=seed,
        num_workers=workers,
    )
    val_loader = _loader(
        arrays,
        "val",
        batch_size=int(training.get("val_batch_size", batch_size)),
        maximum=int(training.get("max_val_sequences", 0)),
        shuffle=False,
        seed=seed + 1,
        num_workers=workers,
    )
    model = RAMPWorldModel(_config(config.get("model", {}))).to(device)
    stages = training.get(
        "stages",
        [
            {"name": "nominal", "epochs": 8, "rollout_seconds": 1.0},
            {"name": "joint", "epochs": 12, "rollout_seconds": 3.0},
            {"name": "five_second", "epochs": 20, "rollout_seconds": 5.0},
        ],
    )
    history: list[dict[str, Any]] = []
    best = float("inf")
    best_path = checkpoint_dir / "best_ramp_world_model.pt"
    patience = int(training.get("early_stopping_patience", 4))
    epoch = 0
    for stage in stages:
        stage_name, stage_epochs = str(stage["name"]), int(stage["epochs"])
        active_candidates = 1 if stage_name == "nominal" else model.cfg.num_candidates
        # Metrics from one-second pretraining and five-second fine-tuning are
        # not comparable.  Each stage owns its checkpoint competition; the
        # final Stage C therefore necessarily selects by deterministic 5 s FDE.
        stage_best, stale, previous_response_steps = float("inf"), 0, None
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(stage.get("learning_rate", training.get("learning_rate", 5e-5))),
            weight_decay=float(training.get("weight_decay", 1e-4)),
        )
        curriculum_seconds = stage.get(
            "rollout_curriculum_seconds", [stage.get("rollout_seconds", 5.0)]
        )
        if not isinstance(curriculum_seconds, list) or not curriculum_seconds:
            raise ValueError(
                f"stage {stage_name!r} needs a non-empty rollout curriculum"
            )
        for stage_epoch in range(stage_epochs):
            # Stage B deliberately traverses 1 -> 2 -> 3 seconds.  Every
            # segment receives an equal contiguous share of its epochs.
            curriculum_index = min(
                len(curriculum_seconds) - 1,
                stage_epoch * len(curriculum_seconds) // stage_epochs,
            )
            response_steps = min(
                model.cfg.response_steps,
                max(
                    1,
                    round(
                        float(curriculum_seconds[curriculum_index])
                        / model.cfg.response_interval_s
                    ),
                ),
            )
            # FDEs from different curriculum horizons are not comparable.
            # Reset the local checkpoint race at each boundary; Stage C still
            # owns the final 5 s checkpoint race.
            if (
                previous_response_steps is not None
                and response_steps != previous_response_steps
            ):
                stage_best, stale = float("inf"), 0
            previous_response_steps = response_steps
            epoch += 1
            model.train()
            totals: dict[str, float] = {}
            count = 0
            for values in train_loader:
                batch = _to_batch(values, train_loader.field_names, device)
                optimizer.zero_grad(set_to_none=True)
                result = model.forward_training(
                    batch,
                    response_steps=response_steps,
                    tbptt_steps=int(training.get("tbptt_response_steps", 5)),
                    active_candidates=active_candidates,
                )
                result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training.get("grad_clip", 1.0))
                )
                optimizer.step()
                for key in (
                    "loss",
                    "prefix_position",
                    "prefix_velocity",
                    "prefix_control",
                    "mixture",
                    "overlap",
                    "diversity",
                    "joint",
                ):
                    totals[key] = totals.get(key, 0.0) + float(
                        result[key].detach().cpu()
                    )
                count += 1
            val = _mean(model, val_loader, device, response_steps, active_candidates)
            selection = _roll_fde(
                model,
                val_loader,
                device,
                response_steps=response_steps,
                active_candidates=active_candidates,
            )
            row = {
                "epoch": epoch,
                "stage": stage_name,
                "stage_epoch": stage_epoch + 1,
                "rollout_seconds": response_steps * model.cfg.response_interval_s,
                **{
                    f"train_{key}": value / max(count, 1)
                    for key, value in totals.items()
                },
                **{f"val_{key}": value for key, value in val.items()},
                "selection_metric": selection,
            }
            history.append(row)
            logger.info(
                "RAMP epoch=%d stage=%s train=%.5f val=%.5f selection=%.5f",
                epoch,
                stage_name,
                row["train_loss"],
                val["loss"],
                selection,
            )
            payload = {
                **model.checkpoint_payload(),
                "epoch": epoch,
                "stage": stage_name,
                "validation": val,
                "selection_metric": selection,
                "training_protocol": {
                    "from_scratch": True,
                    "background_teacher_forcing": 0,
                    "uses_baseline_checkpoint": False,
                },
            }
            if selection < stage_best - float(training.get("min_delta", 1e-4)):
                stage_best, stale = selection, 0
                best = selection
                torch.save(payload, best_path)
            else:
                stale += 1
            if (
                response_steps == model.cfg.response_steps
                and epoch >= int(training.get("min_epochs_before_stopping", 6))
                and stale >= patience
            ):
                logger.info(
                    "RAMP early stopping after %d non-improving full-horizon epochs",
                    stale,
                )
                break
        if response_steps == model.cfg.response_steps and stale >= patience:
            break
    fields = sorted({key for row in history for key in row})
    with (output / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    report = {
        "model_type": model.model_type,
        "best_checkpoint": str(best_path),
        "best_validation_metric": best,
        "epochs_completed": epoch,
        "sequence_cache": manifest,
        "from_scratch": True,
        "uses_baseline_checkpoint": False,
        "model_config": asdict(model.cfg),
        "checkpoint_sha256": (
            hashlib.sha256(best_path.read_bytes()).hexdigest()
            if best_path.exists()
            else None
        ),
    }
    save_json(report, output / "training_summary.json")
    return report


def load_ramp_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> RAMPWorldModel:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("model_type") != RAMPWorldModel.model_type:
        raise ValueError(f"Not a RAMP-WM checkpoint: {path}")
    model = RAMPWorldModel(_config(payload["model_config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.checkpoint_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return model.to(device).eval()

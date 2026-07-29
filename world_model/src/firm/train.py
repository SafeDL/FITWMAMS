"""From-scratch staged training for FIRM-WM on the immutable highD cache."""

from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from world_model.src.core.data import SPLIT_TO_INDEX
from world_model.src.core.initial_behavior_anchor import FrozenLegacyFlowSchema
from world_model.src.semi_markov.train import FIELDS, OPTIONAL_FIELDS, _to_batch
from world_model.src.core.sequential_dataset import (
    FLOW_ANCHOR_ARRAYS,
    ensure_frozen_flow_behavior_anchor_cache,
    load_sequential_dataset,
    sequence_cache_owner_dir,
)
from world_model.src.core.utils import ensure_dir, save_json, select_device, set_seed

from .config import FIRMConfig
from .model import FIRMWorldModel

logger = logging.getLogger(__name__)


def _config(source: dict[str, Any]) -> FIRMConfig:
    return FIRMConfig(
        **{key: value for key, value in source.items() if key in FIRMConfig.__dataclass_fields__}
    )


def _loader(
    arrays: dict[str, np.ndarray],
    split: str,
    *,
    batch_size: int,
    maximum: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    tail_fraction: float | None = None,
):
    indices = np.flatnonzero(np.asarray(arrays["split_index"]) == SPLIT_TO_INDEX[split])
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    if maximum > 0:
        indices = indices[: int(maximum)]
    if not len(indices):
        raise RuntimeError(f"No FIRM sequences in split={split}")
    fields = tuple(
        [
            *FIELDS,
            *[key for key in OPTIONAL_FIELDS if key in arrays],
            *[key for key in FLOW_ANCHOR_ARRAYS if key in arrays],
        ]
    )

    class SequenceDataset(Dataset):
        def __len__(self) -> int:
            return len(indices)

        def __getitem__(self, item: int):
            row = int(indices[int(item)])
            return tuple(torch.from_numpy(np.asarray(arrays[key][row]).copy()) for key in fields)

    sampler = None
    if tail_fraction is not None and 0.0 < float(tail_fraction) < 1.0:
        tail = np.asarray(arrays["is_evt_tail"])[indices].astype(bool)
        tail_count, natural_count = int(tail.sum()), int((~tail).sum())
        if tail_count and natural_count:
            weights = np.where(
                tail,
                float(tail_fraction) / tail_count,
                (1.0 - float(tail_fraction)) / natural_count,
            )
            sampler = WeightedRandomSampler(
                torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(indices),
                replacement=True,
                generator=torch.Generator().manual_seed(int(seed)),
            )
    loader = DataLoader(
        SequenceDataset(),
        batch_size=int(batch_size),
        shuffle=bool(shuffle and sampler is None),
        sampler=sampler,
        num_workers=max(0, int(num_workers)),
        persistent_workers=bool(num_workers),
        drop_last=False,
    )
    loader.field_names = fields
    loader.tail_sampling = {
        "enabled": sampler is not None,
        "requested_tail_fraction": None if tail_fraction is None else float(tail_fraction),
        "available_tail_sequences": int(np.asarray(arrays["is_evt_tail"])[indices].sum()),
        "available_sequences": int(len(indices)),
    }
    return loader


def _mean(model: FIRMWorldModel, loader, device, response_steps: int) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    with torch.no_grad():
        for values in loader:
            result = model.forward_training(
                _to_batch(values, loader.field_names, device), response_steps=response_steps
            )
            for key in (
                "loss",
                "prefix_nll",
                "prefix_position",
                "prefix_velocity",
                "prefix_control",
                "plan_horizon_control",
                "plan_horizon_position",
                "behavior_anchor",
                "overlap",
                "interaction",
                "physical",
                "sampled_physical",
            ):
                totals[key] = totals.get(key, 0.0) + float(result[key].cpu())
            batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def _roll_fde(
    model: FIRMWorldModel, loader, device, *, response_steps: int
) -> float:
    model.eval()
    total = count = 0.0
    with torch.no_grad():
        for values in loader:
            rollout = model._closed_loop(
                _to_batch(values, loader.field_names, device),
                response_steps,
                deterministic=True,
            )
            distance = torch.linalg.vector_norm(
                rollout["predicted_states"][:, -1, 1:, :2]
                - rollout["target_states"][:, -1, 1:, :2],
                dim=-1,
            )
            valid = rollout["target_valid"][:, -1, 1:]
            total += float((distance * valid).sum().cpu())
            count += float(valid.sum().cpu())
    return total / max(count, 1.0)


def train_firm_world_model(
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
        raise RuntimeError("formal FIRM training requires the complete immutable sequence cache")
    schema_path = paths.get("flow_schema")
    if not schema_path:
        raise ValueError("FIRM START requires the frozen Flow schema for the B0 contract")
    schema_file = Path(schema_path)
    if not schema_file.is_absolute():
        schema_file = (config_dir / schema_file).resolve()
    schema = FrozenLegacyFlowSchema.load(schema_file)
    arrays.update(
        ensure_frozen_flow_behavior_anchor_cache(
            sequence_cache_owner_dir(config, config_dir=config_dir), arrays, manifest, schema
        )
    )
    dataset = config.get("dataset", {})
    workers = int(training.get("num_workers", 0))
    train_loader = _loader(
        arrays,
        "train",
        batch_size=int(training.get("batch_size", 64)),
        maximum=int(dataset.get("max_sequences", 0)),
        shuffle=True,
        seed=int(training.get("seed", 42)),
        num_workers=workers,
        tail_fraction=float(training.get("tail_fraction", 0.25)),
    )
    val_loader = _loader(
        arrays,
        "val",
        batch_size=int(training.get("val_batch_size", training.get("batch_size", 64))),
        maximum=int(training.get("validation_max_sequences", 0)),
        shuffle=False,
        seed=int(training.get("seed", 42)) + 1,
        num_workers=workers,
    )
    model = FIRMWorldModel(_config(config.get("model", {}))).to(device)
    stages = training.get(
        "stages",
        [
            {"name": "start_prefix", "epochs": 8, "rollout_seconds": 1.0},
            {"name": "closed_loop", "epochs": 12, "rollout_curriculum_seconds": [1.0, 3.0]},
            {"name": "five_second", "epochs": 20, "rollout_seconds": 5.0},
        ],
    )
    history: list[dict[str, Any]] = []
    best = float("inf")
    best_path = checkpoint_dir / "best_firm_world_model.pt"
    patience = int(training.get("early_stopping_patience", 5))
    minimum_full_horizon_epochs = int(training.get("minimum_full_horizon_epochs", 0))
    epoch = 0
    for stage in stages:
        name, stage_epochs = str(stage["name"]), int(stage["epochs"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(stage.get("learning_rate", training.get("learning_rate", 5e-5))),
            weight_decay=float(training.get("weight_decay", 1e-4)),
        )
        curriculum = stage.get("rollout_curriculum_seconds", [stage.get("rollout_seconds", 5.0)])
        if not isinstance(curriculum, list) or not curriculum:
            raise ValueError(f"stage {name!r} needs a non-empty rollout curriculum")
        stage_best, stale, prior_steps = float("inf"), 0, None
        for stage_epoch in range(stage_epochs):
            index = min(len(curriculum) - 1, stage_epoch * len(curriculum) // stage_epochs)
            response_steps = min(
                model.cfg.response_steps,
                max(1, round(float(curriculum[index]) / model.cfg.response_interval_s)),
            )
            if prior_steps is not None and response_steps != prior_steps:
                stage_best, stale = float("inf"), 0
            prior_steps = response_steps
            epoch += 1
            model.train()
            totals: dict[str, float] = {}
            batches = tail_samples = total_samples = 0
            for values in train_loader:
                batch = _to_batch(values, train_loader.field_names, device)
                optimizer.zero_grad(set_to_none=True)
                result = model.forward_training(
                    batch,
                    response_steps=response_steps,
                    tbptt_steps=int(training.get("tbptt_response_steps", 5)),
                )
                result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training.get("grad_clip", 1.0))
                )
                optimizer.step()
                for key in (
                    "loss",
                    "prefix_nll",
                    "prefix_position",
                    "prefix_velocity",
                    "prefix_control",
                    "plan_horizon_control",
                    "plan_horizon_position",
                    "behavior_anchor",
                    "overlap",
                    "interaction",
                    "physical",
                    "sampled_physical",
                ):
                    totals[key] = totals.get(key, 0.0) + float(result[key].detach().cpu())
                tail_samples += int(batch["is_evt_tail"].sum().item())
                total_samples += int(batch["is_evt_tail"].numel())
                batches += 1
            validation = _mean(model, val_loader, device, response_steps)
            selection = _roll_fde(
                model, val_loader, device, response_steps=response_steps
            )
            row = {
                "epoch": epoch,
                "stage": name,
                "stage_epoch": stage_epoch + 1,
                "rollout_seconds": response_steps * model.cfg.response_interval_s,
                "observed_tail_fraction": tail_samples / max(total_samples, 1),
                **{f"train_{key}": value / max(batches, 1) for key, value in totals.items()},
                **{f"val_{key}": value for key, value in validation.items()},
                "selection_metric": selection,
            }
            history.append(row)
            logger.info(
                "FIRM epoch=%d stage=%s train=%.5f val=%.5f fde=%.5f tail=%.3f",
                epoch,
                name,
                row["train_loss"],
                validation["loss"],
                selection,
                row["observed_tail_fraction"],
            )
            if (
                response_steps == model.cfg.response_steps
                and selection < stage_best - float(training.get("min_delta", 1e-4))
            ):
                stage_best, stale, best = selection, 0, selection
                payload = {
                    **model.checkpoint_payload(),
                    "epoch": epoch,
                    "stage": name,
                    "validation": validation,
                    "selection_metric": selection,
                    "training_protocol": {
                        "from_scratch": True,
                        "uses_baseline_checkpoint": False,
                        "start_uses_single_c0_frame": True,
                        "learned_highd_map_encoder": False,
                        "tail_sampling": train_loader.tail_sampling,
                    },
                }
                torch.save(payload, best_path)
            else:
                stale += 1
            save_json(
                {
                    "status": "running",
                    "epoch": epoch,
                    "stage": name,
                    "stage_epoch": stage_epoch + 1,
                    "rollout_seconds": row["rollout_seconds"],
                    "last_epoch": row,
                    "best_validation_fde": best,
                    "best_checkpoint": str(best_path) if best_path.exists() else None,
                    "tail_sampling": train_loader.tail_sampling,
                },
                output / "training_progress.json",
            )
            if (
                response_steps == model.cfg.response_steps
                and stale >= patience
                and stage_epoch + 1 >= minimum_full_horizon_epochs
            ):
                logger.info("FIRM early stopping after %d non-improving full-horizon epochs", stale)
                break
        if (
            prior_steps == model.cfg.response_steps
            and stale >= patience
            and stage_epoch + 1 >= minimum_full_horizon_epochs
        ):
            break
    fields = sorted({key for row in history for key in row})
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    checkpoint_hash = (
        hashlib.sha256(best_path.read_bytes()).hexdigest()
        if best_path.exists()
        else None
    )
    report = {
        "model_type": model.model_type,
        "best_checkpoint": str(best_path),
        "best_validation_fde": best,
        "epochs_completed": epoch,
        "sequence_cache": manifest,
        "from_scratch": True,
        "uses_baseline_checkpoint": False,
        "tail_sampling": train_loader.tail_sampling,
        "model_config": asdict(model.cfg),
        "checkpoint_sha256": checkpoint_hash,
    }
    save_json(report, output / "training_summary.json")
    save_json(
        {
            "status": "completed",
            "epoch": epoch,
            "best_validation_fde": best,
            "best_checkpoint": str(best_path) if best_path.exists() else None,
        },
        output / "training_progress.json",
    )
    return report


def load_firm_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> FIRMWorldModel:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("model_type") not in {FIRMWorldModel.model_type}:
        raise ValueError(f"Not a FIRM-WM checkpoint: {path}")
    model = FIRMWorldModel(_config(payload["model_config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.checkpoint_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return model.to(device).eval()

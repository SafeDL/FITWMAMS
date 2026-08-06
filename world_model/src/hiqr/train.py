"""From-scratch closed-loop training for the independent HiQR-WM."""

from __future__ import annotations

import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from world_model.src.core.initial_behavior_anchor import FrozenLegacyFlowSchema
from world_model.src.core.sequential_dataset import (
    QR_SEQUENCE_CACHE_FORMAT,
    is_canonical_qr_manifest,
    sequence_cache_owner_dir,
)
from world_model.src.core.utils import (
    ensure_dir,
    file_sha256,
    save_json,
    select_device,
    set_seed,
)

from .config import HiQRWorldModelConfig
from .data import (
    hiqr_sidecar_root,
    load_hiqr_training_arrays,
    make_hiqr_loader,
    to_hiqr_batch,
)
from .model import HierarchicalInteractionQueryRefineWorldModel

LAST_TRAINING_STATE_NAME = "last_training_state.pt"


def _training_protocol(
    manifest: dict[str, Any], schema: FrozenLegacyFlowSchema
) -> dict[str, Any]:
    return {
        "from_scratch": True,
        "sequence_cache_format": manifest["cache_format"],
        "total_transition_frames": 149,
        "start_reconstruction_frames": 25,
        "roll_transition_frames": 124,
        "unified_start_roll_encoder": True,
        "flow_b0_usage": "interaction_state_initialization_only",
        "hierarchical_response_innovations": True,
        "h0_event_structure": "slot_mask_only_causal",
        "flow_schema_sha256": schema.schema_sha256,
    }


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch"}
    if not required <= state.keys():
        raise ValueError("training state is missing Python, NumPy, or Torch RNG")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # The training-state payload is loaded with ``map_location=device``.
    # When resuming on CUDA this also maps the CPU generator state to CUDA,
    # but ``set_rng_state`` specifically requires a CPU ByteTensor.
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state:
        if not torch.cuda.is_available():
            raise ValueError("training state requires CUDA RNG restoration")
        # CUDA's RNG setter likewise expects CPU ByteTensors even when the
        # checkpoint was loaded with a CUDA map location.
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _save_training_state(
    path: Path,
    *,
    model: HierarchicalInteractionQueryRefineWorldModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    stage_index: int,
    stage_epoch: int,
    global_step: int,
    best_validation_fde: float,
    training_protocol: dict[str, Any],
) -> None:
    payload = {
        **model.checkpoint_payload(),
        "training_state_version": 1,
        "training_protocol": training_protocol,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": int(epoch),
        "stage_index": int(stage_index),
        "stage_epoch": int(stage_epoch),
        "global_step": int(global_step),
        "best_validation_fde": float(best_validation_fde),
        "rng_state": _capture_rng_state(),
    }
    torch.save(payload, path)


def _load_training_state(
    path: Path,
    *,
    model: HierarchicalInteractionQueryRefineWorldModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    flow_schema_sha256: str,
) -> dict[str, int | float]:
    device = next(model.parameters()).device
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("training_state_version") != 1:
        raise ValueError(f"unsupported HiQR training state: {path}")
    if payload.get("model_type") != model.model_type:
        raise ValueError(f"training state has an incompatible model type: {path}")
    if payload.get("model_config") != asdict(model.cfg):
        raise ValueError("training state model configuration does not match this run")
    if (
        payload.get("flow_interface", {}).get("flow_schema_sha256")
        != flow_schema_sha256
    ):
        raise ValueError("training state Flow schema hash does not match this run")
    if payload.get("training_protocol", {}).get("h0_event_structure") != (
        "slot_mask_only_causal"
    ):
        raise ValueError("training state predates HiQR's causal h0 protocol")
    required = {
        "state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
        "stage_index",
        "stage_epoch",
        "global_step",
        "best_validation_fde",
        "rng_state",
    }
    if not required <= payload.keys():
        raise ValueError("training state is incomplete")
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    _restore_rng_state(payload["rng_state"])
    return {
        "epoch": int(payload["epoch"]),
        "stage_index": int(payload["stage_index"]),
        "stage_epoch": int(payload["stage_epoch"]),
        "global_step": int(payload["global_step"]),
        "best_validation_fde": float(payload["best_validation_fde"]),
    }


def _config(source: dict[str, Any]) -> HiQRWorldModelConfig:
    return HiQRWorldModelConfig(
        **{
            key: value
            for key, value in source.items()
            if key in HiQRWorldModelConfig.__dataclass_fields__
        }
    )


def _mean_terms(
    model: HierarchicalInteractionQueryRefineWorldModel,
    loader,
    device: torch.device,
    response_steps: int,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    batches = 0
    model.eval()
    with torch.no_grad():
        for values in loader:
            terms = model.supervised_terms(
                to_hiqr_batch(values, loader.field_names, device),
                response_steps=response_steps,
            )
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + float(value.cpu())
            batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


def _roll_fde(
    model: HierarchicalInteractionQueryRefineWorldModel,
    loader,
    device: torch.device,
    response_steps: int,
) -> float:
    total = count = 0.0
    model.eval()
    with torch.no_grad():
        for values in loader:
            rollout = model.rollout_reconstruction(
                to_hiqr_batch(values, loader.field_names, device),
                response_steps=response_steps,
                deterministic=True,
            )
            distance = torch.linalg.vector_norm(
                rollout["predicted_states"][:, -1, 1:, :2]
                - rollout["target_states"][:, -1, 1:, :2],
                dim=-1,
            )
            valid = rollout["target_valid"][:, -1, 1:]
            total += float((distance * valid.float()).sum().cpu())
            count += float(valid.sum().cpu())
    return total / max(count, 1.0)


def _tensorboard_writer(output: Path, training: dict[str, Any]):
    """Create a lightweight per-run TensorBoard writer."""
    if not bool(training.get("tensorboard", True)):
        return None, None
    from torch.utils.tensorboard import SummaryWriter

    configured = Path(str(training.get("tensorboard_dir", "tensorboard")))
    log_dir = configured if configured.is_absolute() else output / configured
    return SummaryWriter(log_dir=str(ensure_dir(log_dir)), flush_secs=30), log_dir


def _write_tensorboard_epoch(writer: Any, row: dict[str, Any]) -> None:
    epoch = int(row["epoch"])
    writer.add_scalar("training/rollout_seconds", float(row["rollout_seconds"]), epoch)
    writer.add_scalar(
        "selection/validation_fde_m", float(row["selection_metric"]), epoch
    )
    for key, value in row.items():
        if key.startswith("train_"):
            tag = f"epoch/train/{key.removeprefix('train_')}"
        elif key.startswith("val_"):
            tag = f"epoch/validation/{key.removeprefix('val_')}"
        else:
            continue
        if math.isfinite(float(value)):
            writer.add_scalar(tag, float(value), epoch)
    writer.flush()


def train_hiqr_world_model(
    config: dict[str, Any], *, config_dir: Path, resume: Path | None = None
) -> dict[str, Any]:
    """Train HiQR without mutating QR data, code, checkpoints or results."""
    paths, training = config["paths"], config.get("training", {})
    output = Path(paths["output_dir"])
    output = output if output.is_absolute() else (config_dir / output).resolve()
    device = select_device(str(training.get("device", "auto")))
    set_seed(int(training.get("seed", 42)))
    cache_owner = sequence_cache_owner_dir(config, config_dir=config_dir)
    schema_path = Path(paths["flow_schema"])
    schema_path = (
        schema_path
        if schema_path.is_absolute()
        else (config_dir / schema_path).resolve()
    )
    schema = FrozenLegacyFlowSchema.load(schema_path)
    arrays, manifest = load_hiqr_training_arrays(
        cache_owner=cache_owner,
        output_dir=output,
        flow_schema=schema,
        source_dataset_dir=paths["source_dataset_dir"],
    )
    if (
        not is_canonical_qr_manifest(manifest)
        or manifest.get("cache_format") != QR_SEQUENCE_CACHE_FORMAT
    ):
        raise RuntimeError("HiQR requires the immutable raw-150 QR START+ROLL cache")
    if (
        manifest.get("bounded_development_cache", True)
        or int(config.get("dataset", {}).get("max_sequences", 0)) != 0
    ):
        raise RuntimeError(
            "formal HiQR training requires the complete immutable sequence cache"
        )
    model = HierarchicalInteractionQueryRefineWorldModel(
        _config(config.get("model", {}))
    ).to(device)
    model.flow_schema_sha256 = schema.schema_sha256
    batch_size, workers = int(training.get("batch_size", 64)), int(
        training.get("num_workers", 0)
    )
    stages = training.get(
        "stages",
        [
            {"name": "warmup", "epochs": 1, "rollout_seconds": 1.0},
            {"name": "full", "epochs": 1, "rollout_seconds": 5.96},
        ],
    )
    configured_epochs = int(
        training.get("epochs", sum(int(stage["epochs"]) for stage in stages))
    )
    if configured_epochs != sum(int(stage["epochs"]) for stage in stages):
        raise ValueError(
            "training.epochs must equal the sum of training.stages[*].epochs"
        )
    checkpoint_dir = ensure_dir(output / "checkpoints")
    best_path = checkpoint_dir / "best_hiqr_world_model.pt"
    last_state_path = checkpoint_dir / LAST_TRAINING_STATE_NAME
    tensorboard_writer, tensorboard_dir = _tensorboard_writer(output, training)
    if tensorboard_writer is not None:
        tensorboard_writer.add_text("run/model_type", model.model_type, 0)
        tensorboard_writer.add_scalar(
            "training/configured_epochs", configured_epochs, 0
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(stages[0].get("learning_rate", training.get("learning_rate", 8e-5))),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training.get("scheduler_factor", 0.5)),
        patience=int(training.get("scheduler_patience", 4)),
        min_lr=float(training.get("scheduler_min_lr", 1e-6)),
    )
    protocol = _training_protocol(manifest, schema)
    best, epoch, global_step = float("inf"), 0, 0
    start_stage, start_stage_epoch = 0, 0
    resumed_from = None
    if resume is not None:
        resume = Path(resume).resolve()
        state = _load_training_state(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            flow_schema_sha256=schema.schema_sha256,
        )
        best = float(state["best_validation_fde"])
        epoch = int(state["epoch"])
        global_step = int(state["global_step"])
        start_stage = int(state["stage_index"])
        start_stage_epoch = int(state["stage_epoch"])
        resumed_from = str(resume)
        if tensorboard_writer is not None:
            tensorboard_writer.add_text("run/resumed_from", resumed_from, epoch)
    if not 0 <= start_stage < len(stages):
        raise ValueError("training state has an invalid curriculum stage")

    for stage_index, stage in enumerate(stages):
        if stage_index < start_stage:
            continue
        stage_epochs = int(stage["epochs"])
        completed_epochs = start_stage_epoch if stage_index == start_stage else 0
        if not 0 <= completed_epochs <= stage_epochs:
            raise ValueError("training state has an invalid stage epoch")
        if completed_epochs == stage_epochs:
            continue
        if stage_index != start_stage or completed_epochs == 0:
            learning_rate = float(
                stage.get("learning_rate", training.get("learning_rate", 8e-5))
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        response_steps = min(
            model.cfg.response_steps,
            max(
                1,
                round(
                    float(stage.get("rollout_seconds", 5.96))
                    / model.cfg.response_interval_s
                ),
            ),
        )
        train_loader = make_hiqr_loader(
            arrays,
            "train",
            batch_size=int(stage.get("batch_size", batch_size)),
            maximum=0,
            shuffle=True,
            seed=int(training.get("seed", 42)),
            num_workers=workers,
        )
        val_loader = make_hiqr_loader(
            arrays,
            "val",
            batch_size=int(stage.get("val_batch_size", batch_size)),
            maximum=0,
            shuffle=False,
            seed=int(training.get("seed", 42)) + 1,
            num_workers=workers,
        )
        for stage_epoch in range(completed_epochs, stage_epochs):
            epoch += 1
            model.train()
            totals: dict[str, float] = {}
            batches = 0
            for values in train_loader:
                optimizer.zero_grad(set_to_none=True)
                result = model.forward_training(
                    to_hiqr_batch(values, train_loader.field_names, device),
                    response_steps=response_steps,
                    tbptt_steps=int(training.get("tbptt_response_steps", 5)),
                )
                result["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training.get("grad_clip", 1.0))
                )
                optimizer.step()
                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar(
                        "batch/train/loss",
                        float(result["loss"].detach().cpu()),
                        global_step,
                    )
                global_step += 1
                for key, value in result.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
                batches += 1
            validation = _mean_terms(model, val_loader, device, response_steps)
            fde = _roll_fde(model, val_loader, device, response_steps)
            row = {
                "epoch": epoch,
                "stage": str(stage["name"]),
                "stage_epoch": stage_epoch + 1,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "rollout_seconds": model.cfg.rollout_seconds_for_responses(
                    response_steps
                ),
                "selection_metric": fde,
                **{
                    f"train_{key}": value / max(1, batches)
                    for key, value in totals.items()
                },
                **{f"val_{key}": value for key, value in validation.items()},
            }
            if tensorboard_writer is not None:
                _write_tensorboard_epoch(tensorboard_writer, row)
                tensorboard_writer.add_scalar(
                    "training/learning_rate", row["learning_rate"], epoch
                )
            if response_steps == model.cfg.response_steps and fde < best:
                best = fde
                payload = {
                    **model.checkpoint_payload(),
                    "epoch": epoch,
                    "stage": stage["name"],
                    "selection_metric": fde,
                    "training_protocol": protocol,
                }
                torch.save(payload, best_path)
            scheduler.step(fde)
            _save_training_state(
                last_state_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                stage_index=stage_index,
                stage_epoch=stage_epoch + 1,
                global_step=global_step,
                best_validation_fde=best,
                training_protocol=protocol,
            )
    if not best_path.exists():
        if tensorboard_writer is not None:
            tensorboard_writer.close()
        raise RuntimeError("HiQR training did not include a full 5.96 s stage")
    report = {
        "model_type": model.model_type,
        "best_checkpoint": str(best_path),
        "last_training_state": str(last_state_path),
        "best_validation_fde": best,
        "epochs_completed": epoch,
        "model_config": asdict(model.cfg),
        "checkpoint_sha256": file_sha256(best_path),
        "flow_schema_sha256": schema.schema_sha256,
        "sequence_cache": manifest,
        "hiqr_sidecar": str(hiqr_sidecar_root(output, cache_owner)),
        "tensorboard_log_dir": (
            None if tensorboard_dir is None else str(tensorboard_dir)
        ),
        "from_scratch": True,
        "resumed_from": resumed_from,
    }
    save_json(report, output / "training_summary.json")
    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar("training/completed", 1, epoch)
        tensorboard_writer.close()
    return report


def load_hiqr_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> HierarchicalInteractionQueryRefineWorldModel:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if (
        payload.get("model_type")
        != HierarchicalInteractionQueryRefineWorldModel.model_type
    ):
        raise ValueError(f"Checkpoint is incompatible with HiQR-WM: {path}")
    if not isinstance(payload.get("model_config"), dict) or not isinstance(
        payload.get("state_dict"), dict
    ):
        raise ValueError("HiQR checkpoint is missing model_config or state_dict")
    model = HierarchicalInteractionQueryRefineWorldModel(
        _config(dict(payload["model_config"]))
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.flow_schema_sha256 = payload.get("flow_interface", {}).get(
        "flow_schema_sha256"
    )
    model.training_protocol = dict(payload.get("training_protocol", {}))
    return model.to(device).eval()


def require_canonical_hiqr_checkpoint(
    model: HierarchicalInteractionQueryRefineWorldModel,
) -> None:
    protocol = getattr(model, "training_protocol", {})
    required = {
        "sequence_cache_format": QR_SEQUENCE_CACHE_FORMAT,
        "total_transition_frames": 149,
        "start_reconstruction_frames": 25,
        "roll_transition_frames": 124,
        "unified_start_roll_encoder": True,
        "flow_b0_usage": "interaction_state_initialization_only",
        "hierarchical_response_innovations": True,
        "h0_event_structure": "slot_mask_only_causal",
    }
    if any(protocol.get(key) != value for key, value in required.items()):
        raise RuntimeError(
            "checkpoint was not trained on HiQR's canonical "
            "Flow-initialized START+ROLL protocol"
        )
    if (
        not getattr(model, "flow_schema_sha256", None)
        or protocol.get("flow_schema_sha256") != model.flow_schema_sha256
    ):
        raise RuntimeError(
            "HiQR checkpoint is missing the matching frozen Flow schema hash"
        )

"""Training and checkpointing for the independent prior-driven HiQR-v2."""

from __future__ import annotations

import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from world_model.src.core.initial_behavior_anchor import FrozenLegacyFlowSchema
from world_model.src.core.sequential_dataset import sequence_cache_owner_dir
from world_model.src.core.utils import (
    ensure_dir,
    file_sha256,
    load_json,
    save_json,
    select_device,
    set_seed,
)

from .config import HiQRV2Config
from .data import load_hiqr_v2_arrays, make_hiqr_v2_loader, to_hiqr_v2_batch
from .model import HiQRV2WorldModel

LAST_TRAINING_STATE_NAME = "last_hiqr_v2_training_state.pt"


def _config(values: dict[str, Any]) -> HiQRV2Config:
    return HiQRV2Config(
        **{
            key: value
            for key, value in values.items()
            if key in HiQRV2Config.__dataclass_fields__
        }
    )


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
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _save_state(
    path: Path,
    *,
    model: HiQRV2WorldModel,
    optimizer,
    scheduler,
    epoch: int,
    stage_index: int,
    stage_epoch: int,
    global_step: int,
    best_fde: float,
    cohort: dict[str, Any],
) -> None:
    payload = {
        **model.checkpoint_payload(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": int(epoch),
        "stage_index": int(stage_index),
        "stage_epoch": int(stage_epoch),
        "global_step": int(global_step),
        "best_validation_fde": float(best_fde),
        "rng_state": _capture_rng_state(),
        "cohort": cohort,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_state(
    path: Path,
    *,
    model: HiQRV2WorldModel,
    optimizer,
    scheduler,
    flow_schema_sha256: str,
) -> dict[str, Any]:
    payload = torch.load(
        path, map_location=next(model.parameters()).device, weights_only=False
    )
    required = {
        "state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "rng_state",
        "epoch",
        "stage_index",
        "stage_epoch",
        "global_step",
        "best_validation_fde",
        "cohort",
    }
    if not required <= payload.keys():
        raise ValueError("incompatible HiQR-v2 training state")
    if payload.get("model_type") != model.model_type or payload.get(
        "model_config"
    ) != asdict(model.cfg):
        raise ValueError("training state does not match this HiQR-v2 model")
    if (
        payload.get("flow_interface", {}).get("flow_schema_sha256")
        != flow_schema_sha256
    ):
        raise ValueError("training state has a different frozen Flow schema")
    model.load_state_dict(payload["state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    _restore_rng_state(payload["rng_state"])
    return payload


def _mean_terms(
    model: HiQRV2WorldModel, loader, device: torch.device, response_steps: int
) -> dict[str, float]:
    totals: dict[str, float] = {}
    batches = 0
    model.eval()
    with torch.no_grad():
        for values in loader:
            terms = model.supervised_terms(
                to_hiqr_v2_batch(values, loader.field_names, device),
                response_steps=response_steps,
            )
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
            batches += 1
    return {name: value / max(1, batches) for name, value in totals.items()}


def _roll_fde(
    model: HiQRV2WorldModel, loader, device: torch.device, response_steps: int
) -> float:
    total = count = 0.0
    model.eval()
    with torch.no_grad():
        for values in loader:
            rollout = model.rollout_reconstruction(
                to_hiqr_v2_batch(values, loader.field_names, device),
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


def _writer(output: Path, training: dict[str, Any]):
    if not bool(training.get("tensorboard", True)):
        return None, None
    from torch.utils.tensorboard import SummaryWriter

    configured = Path(str(training.get("tensorboard_dir", "tensorboard")))
    directory = configured if configured.is_absolute() else output / configured
    return SummaryWriter(log_dir=str(ensure_dir(directory)), flush_secs=30), directory


def _write_epoch(writer, row: dict[str, Any]) -> None:
    epoch = int(row["epoch"])
    writer.add_scalar(
        "hiqr_v2/training/rollout_seconds", float(row["rollout_seconds"]), epoch
    )
    writer.add_scalar(
        "hiqr_v2/selection/validation_fde_m", float(row["selection_metric"]), epoch
    )
    for name, value in row.items():
        if name.startswith("train_"):
            tag = f"hiqr_v2/epoch/train/{name.removeprefix('train_')}"
        elif name.startswith("val_"):
            tag = f"hiqr_v2/epoch/validation/{name.removeprefix('val_')}"
        else:
            continue
        if math.isfinite(float(value)):
            writer.add_scalar(tag, float(value), epoch)
    writer.flush()


def _response_steps(seconds: float, cfg: HiQRV2Config) -> int:
    return min(
        cfg.response_steps,
        max(1, int(math.ceil(float(seconds) / cfg.response_interval_s))),
    )


def _set_stage_learning_rate(optimizer, stage: dict[str, Any], completed: int) -> None:
    """Install a stage LR only on first entry, never on a mid-stage resume."""
    if "learning_rate" not in stage or completed != 0:
        return
    for group in optimizer.param_groups:
        group["lr"] = float(stage["learning_rate"])


def _epoch_shuffle_seed(base_seed: int, epoch: int) -> int:
    """Address the shuffle order by epoch so restart requires no iterator state."""
    return int(base_seed) + int(epoch)


def _stale_full_rollout_epochs(
    history: list[dict[str, Any]], full_rollout_seconds: float
) -> int:
    """Count full-horizon epochs since the most recent strict improvement."""
    best = float("inf")
    stale = 0
    for row in history:
        if not math.isclose(
            float(row["rollout_seconds"]), full_rollout_seconds, abs_tol=1.0e-9
        ):
            continue
        value = float(row["selection_metric"])
        if value < best:
            best, stale = value, 0
        else:
            stale += 1
    return stale


def train_hiqr_v2_world_model(
    config: dict[str, Any],
    *,
    config_dir: Path,
    resume: Path | None = None,
) -> dict[str, Any]:
    paths, training = config["paths"], config["training"]
    output = Path(paths["output_dir"])
    output = output if output.is_absolute() else (config_dir / output).resolve()
    ensure_dir(output)
    model_cfg = _config(config.get("model", {}))
    device = select_device(str(training.get("device", "auto")))
    set_seed(int(training.get("seed", 42)))
    schema_path = Path(paths["flow_schema"])
    schema_path = (
        schema_path
        if schema_path.is_absolute()
        else (config_dir / schema_path).resolve()
    )
    schema = FrozenLegacyFlowSchema.load(schema_path)
    arrays, manifest = load_hiqr_v2_arrays(
        cache_owner=sequence_cache_owner_dir(config, config_dir=config_dir),
        hiqr_sidecar_output_dir=paths["hiqr_sidecar_output_dir"],
        flow_schema=schema,
        source_dataset_dir=paths["source_dataset_dir"],
    )
    cohort = manifest["hiqr_v2_cohort"]
    save_json(
        {"hiqr_v2_cohort": cohort, "sequence_cache": manifest},
        output / "cohort_manifest.json",
    )
    model = HiQRV2WorldModel(model_cfg).to(device)
    model.flow_schema_sha256 = schema.schema_sha256
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 8e-5)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training.get("scheduler_factor", 0.5)),
        patience=int(training.get("scheduler_patience", 6)),
        min_lr=float(training.get("scheduler_min_lr", 1e-6)),
    )
    checkpoint_dir = ensure_dir(output / "checkpoints")
    state_path, best_path = (
        checkpoint_dir / LAST_TRAINING_STATE_NAME,
        checkpoint_dir / "best_hiqr_v2_world_model.pt",
    )
    last_model_path = checkpoint_dir / "last_hiqr_v2_world_model.pt"
    writer, tensorboard_dir = _writer(output, training)
    stages = list(training["stages"])
    configured_epochs = int(training["epochs"])
    if configured_epochs != sum(int(stage["epochs"]) for stage in stages):
        raise ValueError("training.epochs must match curriculum stage epochs")
    if writer is not None:
        writer.add_scalar("hiqr_v2/training/configured_epochs", configured_epochs, 0)
        writer.add_text("hiqr_v2/run/cohort", str(cohort), 0)
    epoch = global_step = stage_start = stage_epoch_start = 0
    best = float("inf")
    history_path = output / "training_history.json"
    history: list[dict[str, Any]] = (
        list(load_json(history_path))
        if resume is not None and history_path.exists()
        else []
    )
    if resume is not None:
        state = _load_state(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            flow_schema_sha256=schema.schema_sha256,
        )
        epoch, global_step, stage_start, stage_epoch_start, best = (
            int(state["epoch"]),
            int(state["global_step"]),
            int(state["stage_index"]),
            int(state["stage_epoch"]),
            float(state["best_validation_fde"]),
        )
        if state["cohort"] != cohort:
            raise ValueError("cannot resume HiQR-v2 across cohort contracts")
    accumulation = max(1, int(training.get("gradient_accumulation_steps", 1)))
    max_sequences = int(config.get("dataset", {}).get("max_sequences", 0))
    full_rollout_seconds = model_cfg.rollout_seconds_for_responses(
        model_cfg.response_steps
    )
    early_stop_patience = max(
        0, int(training.get("full_rollout_early_stopping_patience", 0))
    )
    stale_full_epochs = _stale_full_rollout_epochs(history, full_rollout_seconds)
    early_stopped = bool(
        early_stop_patience and stale_full_epochs >= early_stop_patience
    )
    for stage_index, stage in enumerate(stages):
        if early_stopped:
            break
        if stage_index < stage_start:
            continue
        completed = stage_epoch_start if stage_index == stage_start else 0
        # Preserve the scheduler-adjusted learning rate when resuming midway
        # through a stage.  A new stage still installs its configured rate.
        _set_stage_learning_rate(optimizer, stage, completed)
        response_steps = _response_steps(float(stage["rollout_seconds"]), model_cfg)
        val_loader = make_hiqr_v2_loader(
            arrays,
            "val",
            batch_size=int(
                stage.get(
                    "val_batch_size",
                    training.get("val_batch_size", training.get("batch_size", 48)),
                )
            ),
            maximum=max_sequences,
            shuffle=False,
            seed=int(training.get("seed", 42)) + 1,
            num_workers=int(training.get("num_workers", 0)),
        )
        for local_epoch in range(completed, int(stage["epochs"])):
            epoch += 1
            # Epoch-addressed shuffle seeds make the next batch order
            # identical after restart without serializing DataLoader internals.
            train_loader = make_hiqr_v2_loader(
                arrays,
                "train",
                batch_size=int(stage.get("batch_size", training.get("batch_size", 48))),
                maximum=max_sequences,
                shuffle=True,
                seed=_epoch_shuffle_seed(int(training.get("seed", 42)), epoch),
                num_workers=int(training.get("num_workers", 0)),
            )
            model.train()
            totals: dict[str, float] = {}
            batches = 0
            optimizer.zero_grad(set_to_none=True)
            for batch_index, values in enumerate(train_loader, start=1):
                result = model.forward_training(
                    to_hiqr_v2_batch(values, train_loader.field_names, device),
                    response_steps=response_steps,
                    tbptt_steps=int(training.get("tbptt_response_steps", 10)),
                )
                (result["loss"] / accumulation).backward()
                if batch_index % accumulation == 0 or batch_index == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(training.get("grad_clip", 1.0))
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if writer is not None:
                    writer.add_scalar(
                        "hiqr_v2/batch/train/loss",
                        float(result["loss"].detach().cpu()),
                        global_step,
                    )
                global_step += 1
                for name, value in result.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
                batches += 1
            validation = _mean_terms(model, val_loader, device, response_steps)
            fde = _roll_fde(model, val_loader, device, response_steps)
            row = {
                "epoch": epoch,
                "stage": str(stage["name"]),
                "stage_epoch": local_epoch + 1,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "rollout_seconds": model_cfg.rollout_seconds_for_responses(
                    response_steps
                ),
                "selection_metric": fde,
                **{
                    f"train_{name}": value / max(1, batches)
                    for name, value in totals.items()
                },
                **{f"val_{name}": value for name, value in validation.items()},
            }
            if writer is not None:
                _write_epoch(writer, row)
                writer.add_scalar(
                    "hiqr_v2/training/learning_rate", row["learning_rate"], epoch
                )
            if response_steps == model_cfg.response_steps and fde < best:
                best = fde
                torch.save(
                    {
                        **model.checkpoint_payload(),
                        "epoch": epoch,
                        "stage": stage["name"],
                        "selection_metric": fde,
                        "cohort": cohort,
                    },
                    best_path,
                )
            torch.save(
                {
                    **model.checkpoint_payload(),
                    "epoch": epoch,
                    "stage": stage["name"],
                    "selection_metric": fde,
                    "cohort": cohort,
                },
                last_model_path,
            )
            if local_epoch + 1 == int(stage["epochs"]):
                torch.save(
                    {
                        **model.checkpoint_payload(),
                        "epoch": epoch,
                        "stage": stage["name"],
                        "selection_metric": fde,
                        "cohort": cohort,
                    },
                    checkpoint_dir / f"{stage['name']}_hiqr_v2_world_model.pt",
                )
            scheduler.step(fde)
            _save_state(
                state_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                stage_index=stage_index,
                stage_epoch=local_epoch + 1,
                global_step=global_step,
                best_fde=best,
                cohort=cohort,
            )
            history.append(row)
            save_json(history, history_path)
            if response_steps == model_cfg.response_steps:
                stale_full_epochs = 0 if fde <= best else stale_full_epochs + 1
                if early_stop_patience and stale_full_epochs >= early_stop_patience:
                    early_stopped = True
                    break
        if early_stopped:
            break
    report = {
        "model_type": model.model_type,
        "best_checkpoint": str(best_path),
        "last_training_state": str(state_path),
        "last_model_checkpoint": str(last_model_path),
        "best_validation_fde": best,
        "epochs_completed": epoch,
        "early_stopped": early_stopped,
        "full_rollout_early_stopping_patience": early_stop_patience,
        "model_config": asdict(model_cfg),
        "cohort": cohort,
        "sequence_cache": manifest,
        "tensorboard_log_dir": (
            None if tensorboard_dir is None else str(tensorboard_dir)
        ),
        "checkpoint_sha256": None if not best_path.exists() else file_sha256(best_path),
        "flow_schema_sha256": schema.schema_sha256,
    }
    save_json(report, output / "training_summary.json")
    if writer is not None:
        writer.add_scalar("hiqr_v2/training/completed", 1, epoch)
        writer.close()
    return report


def load_hiqr_v2_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> HiQRV2WorldModel:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("model_type") != HiQRV2WorldModel.model_type:
        raise ValueError("checkpoint is not HiQR-v2")
    model = HiQRV2WorldModel(HiQRV2Config(**payload["model_config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.flow_schema_sha256 = payload.get("flow_interface", {}).get(
        "flow_schema_sha256"
    )
    return model.eval()

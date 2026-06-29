"""Training loop for the highD EVT-tail conditional normalizing flow."""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    build_tail_flow_dataset,
    dataset_schema_is_current,
    load_tail_dataset,
    output_dir_from_config,
    split_indices,
)
from .features import SLOT_NAMES
from .model import build_maf_flow, save_checkpoint
from .utils import (
    ensure_dir,
    repo_root_from_file,
    resolve_path,
    save_json,
    select_device,
    set_seed,
)


logger = logging.getLogger(__name__)


def _sample_weight_cfg(train_cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(train_cfg.get("sample_weighting", {}))


def compute_sample_weights(
    arrays: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> np.ndarray:
    weights = np.ones(len(arrays["features"]), dtype=np.float32)
    if not bool(cfg.get("enabled", False)):
        return weights

    train = arrays["split_index"] == 0
    slot_mask = arrays["slot_mask"].astype(bool)

    slot_power = float(cfg.get("slot_inverse_frequency_power", 0.0))
    if slot_power > 0:
        slot_counts = np.maximum(np.sum(slot_mask[train], axis=0).astype(np.float64), 1.0)
        slot_ref = float(np.max(slot_counts))
        slot_factor = np.power(slot_ref / slot_counts, slot_power).astype(np.float32)
        active_counts = np.maximum(np.sum(slot_mask, axis=1), 1)
        row_factor = (slot_mask.astype(np.float32) @ slot_factor) / active_counts
        weights *= np.maximum(row_factor, 1.0e-6).astype(np.float32)

    targeted = dict(cfg.get("targeted_slot_multipliers", {}))
    for slot_name, multiplier in targeted.items():
        if slot_name not in SLOT_NAMES:
            raise ValueError(f"Unknown targeted_slot_multipliers key: {slot_name!r}")
        slot_idx = SLOT_NAMES.index(slot_name)
        active = slot_mask[:, slot_idx]
        weights[active] *= float(multiplier)

    mask_power = float(cfg.get("mask_pattern_inverse_frequency_power", 0.0))
    if mask_power > 0:
        train_patterns = arrays["mask_pattern"][train].astype(np.int64)
        unique, counts = np.unique(train_patterns, return_counts=True)
        count_map = {int(k): float(v) for k, v in zip(unique, counts)}
        ref = float(np.median(counts.astype(np.float64))) if len(counts) else 1.0
        ref = max(ref, 1.0)
        factors = np.asarray(
            [
                (ref / max(count_map.get(int(pattern), 1.0), 1.0)) ** mask_power
                for pattern in arrays["mask_pattern"].astype(np.int64)
            ],
            dtype=np.float32,
        )
        weights *= factors

    primary_power = float(cfg.get("primary_inverse_frequency_power", 0.0))
    if primary_power > 0:
        train_primary = arrays["primary_slot_index"][train].astype(np.int64)
        counts = np.bincount(train_primary, minlength=len(SLOT_NAMES)).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        ref = float(np.max(counts))
        primary_factor = np.power(ref / counts, primary_power).astype(np.float32)
        weights *= primary_factor[arrays["primary_slot_index"].astype(np.int64)]

    min_weight = float(cfg.get("min_weight", 0.25))
    max_weight = float(cfg.get("max_weight", 6.0))
    weights = np.clip(weights, min_weight, max_weight).astype(np.float32)
    if bool(cfg.get("normalize_train_mean", True)) and np.any(train):
        train_mean = float(np.mean(weights[train]))
        if np.isfinite(train_mean) and train_mean > 1.0e-8:
            weights /= train_mean
    weights = np.clip(weights, min_weight, max_weight).astype(np.float32)
    return weights


def _sample_weight_summary(
    arrays: dict[str, np.ndarray],
    weights: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    train = arrays["split_index"] == 0
    train_weights = weights[train]
    if len(train_weights) == 0:
        return {"enabled": bool(cfg.get("enabled", False))}
    summary: dict[str, Any] = {
        "enabled": bool(cfg.get("enabled", False)),
        "min": float(np.min(train_weights)),
        "max": float(np.max(train_weights)),
        "mean": float(np.mean(train_weights)),
        "p50": float(np.quantile(train_weights, 0.50)),
        "p90": float(np.quantile(train_weights, 0.90)),
        "p99": float(np.quantile(train_weights, 0.99)),
        "effective_train_sample_size": float(
            np.square(np.sum(train_weights)) / max(np.sum(np.square(train_weights)), 1.0e-8)
        ),
        "config": cfg,
    }
    slot_summary: dict[str, Any] = {}
    slot_mask = arrays["slot_mask"].astype(bool)
    for slot_idx, slot_name in enumerate(SLOT_NAMES):
        active = train & slot_mask[:, slot_idx]
        slot_summary[slot_name] = {
            "train_active_count": int(np.sum(active)),
            "mean_weight": float(np.mean(weights[active])) if np.any(active) else float("nan"),
        }
    summary["slot_active"] = slot_summary
    return summary


def _make_loaders(
    arrays: dict[str, np.ndarray],
    *,
    batch_size: int,
    sample_weights: np.ndarray | None = None,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    loaders = {}
    if sample_weights is None:
        sample_weights = np.ones(len(arrays["features"]), dtype=np.float32)
    for split in ("train", "val"):
        idx = split_indices(arrays, split)
        x = torch.from_numpy(arrays["features_normalized"][idx]).float()
        c = torch.from_numpy(arrays["contexts"][idx]).float()
        w = torch.from_numpy(np.asarray(sample_weights[idx], dtype=np.float32)).float()
        loaders[split] = DataLoader(
            TensorDataset(x, c, w),
            batch_size=int(batch_size),
            shuffle=(split == "train"),
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def _run_epoch(
    flow,
    loader,
    device,
    *,
    optimizer=None,
    grad_clip: float = 0.0,
    use_sample_weights: bool = False,
) -> float:
    import torch

    is_train = optimizer is not None
    flow.train(is_train)
    total_loss = 0.0
    total_weight = 0.0
    for batch_x, batch_c, batch_w in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_c = batch_c.to(device, non_blocking=True)
        batch_w = batch_w.to(device, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            nll = -flow.log_prob(batch_x, context=batch_c)
            if is_train and use_sample_weights:
                denom = torch.clamp(batch_w.sum(), min=1.0e-8)
                loss = (nll * batch_w).sum() / denom
                batch_total = float((nll.detach() * batch_w).sum().cpu())
                batch_weight = float(batch_w.detach().sum().cpu())
            else:
                loss = nll.mean()
                batch_total = float(nll.detach().sum().cpu())
                batch_weight = float(batch_x.shape[0])
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(flow.parameters(), float(grad_clip))
                optimizer.step()
        total_loss += batch_total
        total_weight += batch_weight
    return total_loss / max(total_weight, 1.0e-8)


def _tensorboard_writer(config: dict[str, Any], output_dir: Path):
    tb_cfg = dict(config.get("tensorboard", {}))
    if not bool(tb_cfg.get("enabled", True)):
        return None, None
    log_dir = Path(str(tb_cfg.get("log_dir", "tensorboard")))
    if not log_dir.is_absolute():
        log_dir = output_dir / log_dir
    ensure_dir(log_dir)
    if bool(tb_cfg.get("clear_existing", False)):
        for path in log_dir.glob("events.out.tfevents.*"):
            path.unlink()
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover - optional runtime dependency.
        logger.warning("TensorBoard logging disabled: %s", exc)
        return None, log_dir
    return SummaryWriter(log_dir=str(log_dir), flush_secs=int(tb_cfg.get("flush_secs", 30))), log_dir


def _tb_scalar(writer, tag: str, value: Any, step: int) -> None:
    if writer is None:
        return
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return
    if np.isfinite(value_f):
        writer.add_scalar(tag, value_f, int(step))


def _checkpoint_path(raw_path: Any, *, config_dir: Path) -> Path | None:
    if raw_path in (None, ""):
        return None
    return resolve_path(str(raw_path), base=config_dir)


def _load_resume_checkpoint(
    flow,
    *,
    checkpoint_path: Path,
    schema: dict[str, Any],
    model_cfg: dict[str, Any],
    device,
) -> dict[str, Any]:
    import torch

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"resume_from_checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    payload_schema = dict(payload.get("schema", {}))
    if list(payload_schema.get("feature_names", [])) != list(schema["feature_names"]):
        raise ValueError("resume_from_checkpoint feature schema does not match current dataset")
    if list(payload_schema.get("context_names", [])) != list(schema["context_names"]):
        raise ValueError("resume_from_checkpoint context schema does not match current dataset")
    if list(payload_schema.get("model_feature_transforms", [])) != list(
        schema.get("model_feature_transforms", [])
    ):
        raise ValueError("resume_from_checkpoint feature transforms do not match current dataset")
    payload_model_cfg = dict(payload.get("model_cfg", {}))
    if payload_model_cfg != dict(model_cfg):
        raise ValueError(
            "resume_from_checkpoint model config does not match current model config; "
            "start from scratch for architecture changes"
        )
    flow.load_state_dict(payload["state_dict"])
    return payload


def _stage_config(
    config: dict[str, Any],
    stage: dict[str, Any],
    *,
    previous_checkpoints: dict[str, str],
) -> dict[str, Any]:
    stage_cfg = copy.deepcopy(config)
    stage_paths = dict(stage_cfg.get("paths", {}))
    stage_paths.update(dict(stage.get("paths", {})))
    if "output_dir" in stage:
        stage_paths["output_dir"] = stage["output_dir"]
    stage_cfg["paths"] = stage_paths

    training_root = dict(config.get("training", {}))
    stage_training = copy.deepcopy(dict(training_root.get("defaults", {})))
    stage_training.update(copy.deepcopy(dict(stage.get("training", {}))))
    resume_from_stage = stage.get("resume_from_stage")
    if resume_from_stage:
        resume_key = str(resume_from_stage)
        if resume_key not in previous_checkpoints:
            raise ValueError(
                f"Stage {stage.get('name', '<unnamed>')!r} depends on unknown "
                f"resume_from_stage={resume_key!r}"
            )
        stage_training["resume_from_checkpoint"] = previous_checkpoints[resume_key]
    stage_cfg["training"] = stage_training
    return stage_cfg


def _train_tail_flow_stages(
    config: dict[str, Any],
    *,
    config_dir: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    config_dir = Path(config_dir).resolve()
    repo_root = Path(repo_root).resolve() if repo_root else repo_root_from_file(config_dir)
    stages = list(dict(config.get("training", {})).get("stages", []))
    if not stages:
        raise ValueError("training.stages must contain at least one stage")

    final_output_dir = output_dir_from_config(config, config_dir)
    previous_checkpoints: dict[str, str] = {}
    stage_summaries: list[dict[str, Any]] = []
    for stage_idx, stage in enumerate(stages, start=1):
        if not isinstance(stage, dict):
            raise TypeError(f"training.stages[{stage_idx - 1}] must be a mapping")
        stage_name = str(stage.get("name", f"stage_{stage_idx}"))
        current_config = _stage_config(
            config,
            stage,
            previous_checkpoints=previous_checkpoints,
        )
        stage_output_dir = output_dir_from_config(current_config, config_dir)
        logger.info(
            "Training staged tail flow %d/%d: %s -> %s",
            stage_idx,
            len(stages),
            stage_name,
            stage_output_dir,
        )
        summary = _train_tail_flow_single(
            current_config,
            config_dir=config_dir,
            repo_root=repo_root,
        )
        checkpoint = str(summary["checkpoint"])
        previous_checkpoints[stage_name] = checkpoint
        stage_summaries.append(
            {
                "name": stage_name,
                "output_dir": str(stage_output_dir),
                **summary,
            }
        )

    final_summary = {
        "final_stage": stage_summaries[-1]["name"],
        "checkpoint": stage_summaries[-1]["checkpoint"],
        "dataset": stage_summaries[-1]["dataset"],
        "stages": stage_summaries,
    }
    ensure_dir(final_output_dir)
    save_json(final_summary, final_output_dir / "staged_training_summary.json")
    return final_summary


def train_tail_flow(
    config: dict[str, Any],
    *,
    config_dir: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    if dict(config.get("training", {})).get("stages"):
        return _train_tail_flow_stages(
            config,
            config_dir=config_dir,
            repo_root=repo_root,
        )
    return _train_tail_flow_single(
        config,
        config_dir=config_dir,
        repo_root=repo_root,
    )


def _train_tail_flow_single(
    config: dict[str, Any],
    *,
    config_dir: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    import torch

    config_dir = Path(config_dir).resolve()
    repo_root = Path(repo_root).resolve() if repo_root else repo_root_from_file(config_dir)
    output_dir = output_dir_from_config(config, config_dir)
    set_seed(int(config.get("seed", 42)))

    dataset_cfg = dict(config.get("dataset", {}))
    dataset_current = (output_dir / "dataset.npz").exists() and dataset_schema_is_current(output_dir)
    if bool(dataset_cfg.get("rebuild", False)) or not dataset_current:
        build_tail_flow_dataset(
            config,
            config_dir=config_dir,
            rebuild_tail_contexts=bool(dataset_cfg.get("rebuild_tail_contexts", False)),
        )
    arrays, schema = load_tail_dataset(output_dir)

    model_type = str(config.get("model", {}).get("type", "conditional_maf")).lower()
    if model_type != "conditional_maf":
        raise ValueError(f"Unsupported model.type={model_type!r}; expected 'conditional_maf'")

    device = select_device(str(config.get("device", "auto")))
    logger.info("Training tail-event conditional MAF on device=%s", device)
    writer, tensorboard_dir = _tensorboard_writer(config, output_dir)
    train_cfg = dict(config.get("training", {}))
    batch_size = int(train_cfg.get("batch_size", 256))
    sample_weight_cfg = _sample_weight_cfg(train_cfg)
    sample_weights = compute_sample_weights(arrays, sample_weight_cfg)
    sample_weight_summary = _sample_weight_summary(arrays, sample_weights, sample_weight_cfg)
    loaders = _make_loaders(arrays, batch_size=batch_size, sample_weights=sample_weights)
    if sample_weight_summary.get("enabled"):
        logger.info("Using tail sample weighting: %s", sample_weight_summary)

    flow = build_maf_flow(
        num_features=len(schema["feature_names"]),
        context_features=len(schema["context_names"]),
        model_cfg=dict(config.get("model", {})),
        repo_root=repo_root,
    ).to(device)
    resume_checkpoint = _checkpoint_path(
        train_cfg.get("resume_from_checkpoint"),
        config_dir=config_dir,
    )
    resume_summary: dict[str, Any] = {}
    if resume_checkpoint is not None:
        payload = _load_resume_checkpoint(
            flow,
            checkpoint_path=resume_checkpoint,
            schema=schema,
            model_cfg=dict(config.get("model", {})),
            device=device,
        )
        initial_val_nll = _run_epoch(flow, loaders["val"], device)
        resume_summary = {
            "resume_from_checkpoint": str(resume_checkpoint),
            "resume_checkpoint_metrics": dict(payload.get("metrics", {})),
            "initial_val_nll": float(initial_val_nll),
        }
        logger.info(
            "Loaded resume checkpoint %s with initial val_nll=%.4f",
            resume_checkpoint,
            initial_val_nll,
        )
    optimizer = torch.optim.Adam(
        flow.parameters(),
        lr=float(train_cfg.get("learning_rate", 5.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(train_cfg.get("lr_decay_factor", 0.5)),
        patience=int(train_cfg.get("lr_patience", 5)),
        min_lr=float(train_cfg.get("min_lr", 1.0e-5)),
    )

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(flow.state_dict())
    best_val = (
        float(resume_summary["initial_val_nll"])
        if "initial_val_nll" in resume_summary
        else float("inf")
    )
    bad_epochs = 0
    max_epochs = int(train_cfg.get("max_epochs", 100))
    patience = int(train_cfg.get("patience", 15))
    grad_clip = float(train_cfg.get("grad_clip", 5.0))

    for epoch in range(1, max_epochs + 1):
        train_nll = _run_epoch(
            flow,
            loaders["train"],
            device,
            optimizer=optimizer,
            grad_clip=grad_clip,
            use_sample_weights=bool(sample_weight_summary.get("enabled", False)),
        )
        val_nll = _run_epoch(flow, loaders["val"], device)
        scheduler.step(val_nll)
        lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": int(epoch),
                "train_nll": float(train_nll),
                "val_nll": float(val_nll),
                "learning_rate": lr,
            }
        )
        _tb_scalar(writer, "nll/train", train_nll, epoch)
        _tb_scalar(writer, "nll/val", val_nll, epoch)
        _tb_scalar(writer, "nll/best_val", min(best_val, val_nll), epoch)
        _tb_scalar(writer, "learning_rate", lr, epoch)
        logger.info(
            "Tail flow epoch %03d train_nll=%.4f val_nll=%.4f lr=%.2g",
            epoch,
            train_nll,
            val_nll,
            lr,
        )
        if val_nll < best_val:
            best_val = float(val_nll)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in flow.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info("Early stopping after %d stale epochs", bad_epochs)
                break

    flow.load_state_dict(best_state)
    flow.to(device)
    checkpoint_path = output_dir / "checkpoints" / "best_tail_conditional_maf.pt"
    metrics = {
        "best_val_nll": float(best_val),
        "epochs": int(history[-1]["epoch"]) if history else 0,
    }
    save_checkpoint(
        checkpoint_path,
        flow=flow,
        model_cfg=dict(config.get("model", {})),
        schema=schema,
        metrics=metrics,
    )
    history_path = output_dir / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    summary = {
        "checkpoint": str(checkpoint_path),
        "dataset": schema["dataset_npz"],
        "training_history_csv": str(history_path),
        "sample_weighting": sample_weight_summary,
        **resume_summary,
        **metrics,
    }
    if tensorboard_dir is not None:
        summary["tensorboard_dir"] = str(tensorboard_dir)
    save_json(summary, output_dir / "training_summary.json")
    if writer is not None:
        writer.flush()
        writer.close()
    logger.info("Wrote training checkpoint: %s", checkpoint_path)
    return summary

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
from .model import build_maf_flow, save_checkpoint
from .utils import ensure_dir, repo_root_from_file, save_json, select_device, set_seed


logger = logging.getLogger(__name__)


def _make_loaders(arrays: dict[str, np.ndarray], *, batch_size: int):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    loaders = {}
    for split in ("train", "val"):
        idx = split_indices(arrays, split)
        x = torch.from_numpy(arrays["features_normalized"][idx]).float()
        c = torch.from_numpy(arrays["contexts"][idx]).float()
        valid = torch.from_numpy(arrays["feature_valid"][idx]).bool()
        loaders[split] = DataLoader(
            TensorDataset(x, c, valid),
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
    inactive_noise_std: float = 0.0,
) -> float:
    import torch

    is_train = optimizer is not None
    flow.train(is_train)
    total = 0.0
    total_n = 0
    for batch_x, batch_c, batch_valid in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_c = batch_c.to(device, non_blocking=True)
        if is_train and inactive_noise_std > 0:
            batch_valid = batch_valid.to(device, non_blocking=True)
            noise = torch.randn_like(batch_x) * float(inactive_noise_std)
            batch_x = torch.where(batch_valid, batch_x, noise)
        with torch.set_grad_enabled(is_train):
            loss = -flow.log_prob(batch_x, context=batch_c).mean()
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(flow.parameters(), float(grad_clip))
                optimizer.step()
        n = int(batch_x.shape[0])
        total += float(loss.detach().cpu()) * n
        total_n += n
    return total / max(total_n, 1)


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


def train_tail_flow(
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
    loaders = _make_loaders(arrays, batch_size=batch_size)

    flow = build_maf_flow(
        num_features=len(schema["feature_names"]),
        context_features=len(schema["context_names"]),
        model_cfg=dict(config.get("model", {})),
        repo_root=repo_root,
    ).to(device)
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
    best_val = float("inf")
    bad_epochs = 0
    max_epochs = int(train_cfg.get("max_epochs", 100))
    patience = int(train_cfg.get("patience", 15))
    grad_clip = float(train_cfg.get("grad_clip", 5.0))
    inactive_noise_std = float(train_cfg.get("inactive_noise_std", 0.0))

    for epoch in range(1, max_epochs + 1):
        train_nll = _run_epoch(
            flow,
            loaders["train"],
            device,
            optimizer=optimizer,
            grad_clip=grad_clip,
            inactive_noise_std=inactive_noise_std,
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
        "tensorboard_dir": str(tensorboard_dir) if tensorboard_dir is not None else "",
        **metrics,
    }
    save_json(summary, output_dir / "training_summary.json")
    if writer is not None:
        writer.flush()
        writer.close()
    logger.info("Wrote training checkpoint: %s", checkpoint_path)
    return summary

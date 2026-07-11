"""Training loop for the FiT-AMS START/ROLL background world model."""
from __future__ import annotations

import contextlib
import csv
import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .data import (
    build_world_model_dataset,
    checkpoint_path,
    dataset_dir_from_config,
    load_world_model_training_dataset,
    output_dir_from_config,
    prepared_dataset_available,
    split_indices,
    training_uses_auxiliary_states,
    training_uses_relation_features,
)
from .model import (
    build_model_from_schema,
    gaussian_nll,
    masked_action_mse,
    model_config_payload,
)
from .schema import ROLL_MODE_INDEX, START_MODE_INDEX
from .utils import ensure_dir, load_json, save_json, select_device, set_seed

logger = logging.getLogger(__name__)

HISTORY_FIELDS = [
    "epoch",
    "lr",
    "train_loss",
    "train_action_mse_norm",
    "val_loss",
    "val_action_mse_norm",
    "val_start_loss",
    "val_start_action_mse_norm",
    "val_roll_loss",
    "val_roll_action_mse_norm",
    "val_evt_tail_loss",
    "val_evt_tail_action_mse_norm",
]


def _torch():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    return torch, DataLoader, TensorDataset


CORE_BATCH_FIELDS = (
    ("history_states", "history_states_normalized", "float"),
    ("history_valid", "history_valid", "bool"),
    ("current_states", "current_states_normalized", "float"),
    ("current_valid", "current_valid", "bool"),
    ("mode_index", "mode_index", "long"),
    ("primary_slot_index", "primary_slot_index", "long"),
    ("flow_action_summary", "flow_action_summary_normalized", "float"),
    ("target_actions", "target_actions_normalized", "float"),
    ("target_valid", "target_valid", "bool"),
    ("sample_weight", "sample_weight", "float"),
)
RELATION_BATCH_FIELDS = (
    ("relation_features", "relation_features_normalized", "float"),
)
AUXILIARY_BATCH_FIELDS = (
    ("current_states_raw", "current_states", "float"),
    ("target_states_raw", "target_states", "float"),
    ("ego_future_states_raw", "ego_future_states", "float"),
)
DEFAULT_BATCH_KEYS = (
    "history_states",
    "history_valid",
    "current_states",
    "current_valid",
    "mode_index",
    "primary_slot_index",
    "flow_action_summary",
    "relation_features",
    "target_actions",
    "target_valid",
    "sample_weight",
    "current_states_raw",
    "target_states_raw",
    "ego_future_states_raw",
)


def _tensor_from_array(torch, array: np.ndarray, dtype_name: str):
    tensor = torch.from_numpy(array)
    if dtype_name == "float":
        return tensor.float()
    if dtype_name == "bool":
        return tensor.bool()
    if dtype_name == "long":
        return tensor.long()
    raise ValueError(f"Unsupported tensor dtype={dtype_name!r}")


def _make_loader(
    arrays: dict[str, np.ndarray],
    split: str,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    max_samples: int = 0,
    mode_index: int | None = None,
    evt_tail: bool | None = None,
    include_relation_features: bool = False,
    include_auxiliary_states: bool = False,
):
    torch, DataLoader, TensorDataset = _torch()
    idx = split_indices(arrays, split)
    if mode_index is not None:
        idx = idx[arrays["mode_index"][idx] == int(mode_index)]
    if evt_tail is not None:
        idx = idx[arrays["is_evt_tail"][idx].astype(bool) == bool(evt_tail)]
    if max_samples and int(max_samples) > 0:
        idx = idx[: int(max_samples)]
    if len(idx) == 0:
        raise RuntimeError(f"No samples for split={split}")
    fields = list(CORE_BATCH_FIELDS)
    if include_relation_features:
        fields.extend(RELATION_BATCH_FIELDS)
    if include_auxiliary_states:
        fields.extend(AUXILIARY_BATCH_FIELDS)
    tensors = [
        _tensor_from_array(torch, arrays[array_key][idx], dtype_name)
        for _name, array_key, dtype_name in fields
    ]
    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
    )
    loader.world_model_batch_keys = tuple(name for name, _array_key, _dtype_name in fields)
    loader.world_model_indices = idx
    return loader


def _batch_to_device(batch, device, keys: tuple[str, ...] | None = None) -> dict[str, Any]:
    keys = keys or DEFAULT_BATCH_KEYS
    return {key: value.to(device, non_blocking=True) for key, value in zip(keys, batch)}


def _performance_config(training_cfg: dict[str, Any]) -> dict[str, Any]:
    performance = dict(training_cfg.get("performance", {}))
    for key in (
        "allow_tf32",
        "float32_matmul_precision",
        "mixed_precision",
        "fused_adamw",
        "compile_model",
        "compile_mode",
    ):
        if key in training_cfg and key not in performance:
            performance[key] = training_cfg[key]
    return performance


def _configure_torch_performance(torch, training_cfg: dict[str, Any], device) -> dict[str, Any]:
    performance = _performance_config(training_cfg)
    allow_tf32 = bool(performance.get("allow_tf32", False)) and str(device).startswith("cuda")
    if str(device).startswith("cuda"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
            torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        except Exception as exc:  # noqa: BLE001 - backend knobs vary by torch build.
            logger.warning("Unable to configure TF32 backend flags: %s", exc)
    requested_precision = str(performance.get("float32_matmul_precision", "high" if allow_tf32 else "highest"))
    matmul_precision = requested_precision if str(device).startswith("cuda") else "highest"
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision(matmul_precision)
        except Exception as exc:  # noqa: BLE001 - older torch may reject newer values.
            logger.warning("Unable to set float32 matmul precision=%s: %s", matmul_precision, exc)
    return {
        "allow_tf32": bool(allow_tf32),
        "float32_matmul_precision": matmul_precision,
    }


def _resolve_mixed_precision(torch, training_cfg: dict[str, Any], device) -> tuple[bool, Any, str]:
    performance = _performance_config(training_cfg)
    requested = performance.get("mixed_precision", "off")
    if isinstance(requested, bool):
        requested = "auto" if requested else "off"
    mode = str(requested).strip().lower()
    if mode in {"", "0", "false", "none", "no", "off", "fp32", "float32"}:
        return False, None, "off"
    if not str(device).startswith("cuda"):
        logger.warning("Mixed precision requested as %s but device=%s is not CUDA; disabling.", mode, device)
        return False, None, "off"
    if mode in {"auto", "bf16", "bfloat16"}:
        bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        if bf16_supported:
            return True, torch.bfloat16, "bf16"
        if mode in {"bf16", "bfloat16"}:
            logger.warning("BF16 mixed precision requested but this CUDA device/build does not support it; disabling.")
            return False, None, "off"
    if mode in {"auto", "fp16", "float16", "half"}:
        return True, torch.float16, "fp16"
    logger.warning("Unknown mixed_precision=%r; disabling.", requested)
    return False, None, "off"


def _autocast_context(torch, device, enabled: bool, dtype):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=str(device).split(":", 1)[0], dtype=dtype, enabled=True)


def _make_grad_scaler(torch, mixed_precision_mode: str):
    if mixed_precision_mode != "fp16":
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:  # noqa: BLE001 - torch < 2.0 fallback.
        return torch.cuda.amp.GradScaler(enabled=True)


def _make_adamw_optimizer(torch, model, training_cfg: dict[str, Any], device):
    kwargs = {
        "lr": float(training_cfg.get("learning_rate", 2.0e-4)),
        "weight_decay": float(training_cfg.get("weight_decay", 1.0e-4)),
    }
    performance = _performance_config(training_cfg)
    fused_requested = bool(performance.get("fused_adamw", False)) and str(device).startswith("cuda")
    if fused_requested:
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs), True
        except (TypeError, RuntimeError) as exc:
            logger.warning("Fused AdamW unavailable, falling back to standard AdamW: %s", exc)
    return torch.optim.AdamW(model.parameters(), **kwargs), False


def _maybe_compile_model(torch, model, training_cfg: dict[str, Any]):
    performance = _performance_config(training_cfg)
    if not bool(performance.get("compile_model", False)):
        return model, False
    if not hasattr(torch, "compile"):
        logger.warning("compile_model requested but torch.compile is unavailable.")
        return model, False
    mode = str(performance.get("compile_mode", "default"))
    try:
        return torch.compile(model, mode=mode), True
    except Exception as exc:  # noqa: BLE001 - keep training runnable on unsupported builds.
        logger.warning("torch.compile failed during setup; continuing without compilation: %s", exc)
        return model, False


def _make_loader_from_existing_dataset(
    base_loader,
    positions: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
):
    if len(positions) == 0:
        raise RuntimeError("No samples for diagnostic loader")
    torch, DataLoader, _TensorDataset = _torch()
    from torch.utils.data import Subset

    loader = DataLoader(
        Subset(base_loader.dataset, np.asarray(positions, dtype=np.int64)),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
    )
    loader.world_model_batch_keys = getattr(base_loader, "world_model_batch_keys", DEFAULT_BATCH_KEYS)
    base_indices = getattr(base_loader, "world_model_indices", None)
    if base_indices is not None:
        loader.world_model_indices = np.asarray(base_indices)[positions]
    return loader


def _maybe_make_diagnostic_loader_from_val(
    val_loader,
    arrays: dict[str, np.ndarray],
    *,
    batch_size: int,
    num_workers: int,
    max_samples: int,
    mode_index: int | None = None,
    evt_tail: bool | None = None,
):
    if max_samples and int(max_samples) > 0:
        return None
    val_idx = getattr(val_loader, "world_model_indices", None)
    if val_idx is None:
        return None
    val_idx = np.asarray(val_idx, dtype=np.int64)
    mask = np.ones(len(val_idx), dtype=bool)
    if mode_index is not None:
        mask &= arrays["mode_index"][val_idx] == int(mode_index)
    if evt_tail is not None:
        mask &= arrays["is_evt_tail"][val_idx].astype(bool) == bool(evt_tail)
    positions = np.flatnonzero(mask)
    try:
        return _make_loader_from_existing_dataset(
            val_loader,
            positions,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    except RuntimeError:
        return None


def _history_corruption_noise(training_cfg: dict[str, Any], schema: dict[str, Any]) -> np.ndarray:
    adaptation = dict(training_cfg.get("closed_loop_adaptation", {}))
    raw = adaptation.get(
        "state_noise_std",
        {
            "x_m": 0.20,
            "y_left_m": 0.05,
            "vx_mps": 0.20,
            "vy_left_mps": 0.05,
            "ax_mps2": 0.15,
            "ay_left_mps2": 0.05,
        },
    )
    names = list(schema["state_features"])
    state_std = np.asarray(schema["normalization"]["state"]["std"], dtype=np.float32)
    noise = np.zeros(len(names), dtype=np.float32)
    for idx, name in enumerate(names):
        noise[idx] = float(raw.get(name, 0.0)) / max(float(state_std[idx]), 1.0e-6)
    return noise.astype(np.float32)


def _apply_history_corruption(
    item: dict[str, Any],
    training_cfg: dict[str, Any],
    schema: dict[str, Any],
    *,
    noise_scale=None,
    noise_enabled: bool | None = None,
) -> None:
    adaptation = dict(training_cfg.get("closed_loop_adaptation", {}))
    if not bool(adaptation.get("enabled", True)):
        return
    torch, _, _ = _torch()
    probability = float(adaptation.get("probability", 0.35))
    if probability <= 0.0:
        return
    roll_mask = item["mode_index"].long() == 1
    if not bool(torch.any(roll_mask)):
        return
    selected = roll_mask & (torch.rand_like(item["sample_weight"]) < probability)
    if not bool(torch.any(selected)):
        return

    if noise_enabled is None:
        noise_values = _history_corruption_noise(training_cfg, schema)
        noise_enabled = bool(np.any(noise_values > 0))
        noise_scale = torch.from_numpy(noise_values).to(
            item["history_states"].device,
            dtype=item["history_states"].dtype,
        )
    if noise_enabled:
        if noise_scale is None:
            noise_scale = torch.from_numpy(_history_corruption_noise(training_cfg, schema)).to(
                item["history_states"].device,
                dtype=item["history_states"].dtype,
            )
        else:
            noise_scale = noise_scale.to(device=item["history_states"].device, dtype=item["history_states"].dtype)
        hist_noise = torch.randn_like(item["history_states"]) * noise_scale.view(1, 1, 1, -1)
        curr_noise = torch.randn_like(item["current_states"]) * noise_scale.view(1, 1, -1)
        hist_mask = selected.view(-1, 1, 1, 1) & item["history_valid"].unsqueeze(-1)
        curr_mask = selected.view(-1, 1, 1) & item["current_valid"].unsqueeze(-1)
        item["history_states"] = torch.where(hist_mask, item["history_states"] + hist_noise, item["history_states"])
        item["current_states"] = torch.where(curr_mask, item["current_states"] + curr_noise, item["current_states"])

    dropout_prob = float(adaptation.get("history_dropout_probability", 0.05))
    if dropout_prob > 0.0:
        drop = (
            torch.rand(item["history_valid"].shape, device=item["history_valid"].device)
            < dropout_prob
        ) & selected.view(-1, 1, 1)
        # Keep the current frame and ego history so ROLL remains well-defined.
        drop[:, -1, :] = False
        drop[:, :, 0] = False
        item["history_valid"] = item["history_valid"] & ~drop
        item["history_states"] = item["history_states"].masked_fill(drop.unsqueeze(-1), 0.0)


def _loss_weights(training_cfg: dict[str, Any]) -> dict[str, float]:
    weights = dict(training_cfg.get("loss_weights", {}))
    return {key: float(value) for key, value in weights.items()}


def _unnormalize_actions_torch(actions_normalized, schema: dict[str, Any]):
    torch, _, _ = _torch()
    norm = schema["normalization"]["action"]
    mean = torch.as_tensor(norm["mean"], device=actions_normalized.device, dtype=actions_normalized.dtype).view(1, 1, 1, -1)
    std = torch.as_tensor(norm["std"], device=actions_normalized.device, dtype=actions_normalized.dtype).view(1, 1, 1, -1)
    return actions_normalized * std + mean


def _integrate_actions_torch(current_states, actions, *, dt: float):
    torch, _, _ = _torch()
    bg = current_states[:, 1:, :].clone()
    states = []
    dt_value = float(dt)
    for step in range(actions.shape[1]):
        ax = torch.clamp(actions[:, step, :, 0], -8.0, 4.0)
        ay = torch.clamp(actions[:, step, :, 1], -4.0, 4.0)
        bg = bg.clone()
        bg[:, :, 0] = bg[:, :, 0] + bg[:, :, 2] * dt_value + 0.5 * ax * dt_value * dt_value
        bg[:, :, 1] = bg[:, :, 1] + bg[:, :, 3] * dt_value + 0.5 * ay * dt_value * dt_value
        bg[:, :, 2] = torch.clamp(bg[:, :, 2] + ax * dt_value, 0.0, 50.0)
        bg[:, :, 3] = bg[:, :, 3] + ay * dt_value
        bg[:, :, 4] = ax
        bg[:, :, 5] = ay
        states.append(bg)
    return torch.stack(states, dim=1)


def _auxiliary_rollout_losses(
    item: dict[str, Any],
    actions_normalized,
    schema: dict[str, Any],
    training_cfg: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    torch, _, _ = _torch()
    weights = weights if weights is not None else _loss_weights(training_cfg)
    if not any(float(weights.get(key, 0.0)) > 0.0 for key in ("trajectory", "interaction", "physics")):
        return {}
    missing = [
        key
        for key in ("current_states_raw", "target_states_raw", "ego_future_states_raw")
        if key not in item
    ]
    if missing:
        raise RuntimeError(f"Auxiliary rollout losses require raw state tensors; missing={missing}")
    actions_raw = _unnormalize_actions_torch(actions_normalized, schema)
    pred = _integrate_actions_torch(
        item["current_states_raw"],
        actions_raw,
        dt=1.0 / float(schema["fps"]),
    )
    target = item["target_states_raw"]
    valid = item["target_valid"].float().unsqueeze(-1)
    out: dict[str, Any] = {}
    if float(weights.get("trajectory", 0.0)) > 0.0:
        pos_mse = ((pred[..., :2] - target[..., :2]).pow(2) * valid).sum() / valid.sum().clamp_min(1.0)
        vel_mse = ((pred[..., 2:4] - target[..., 2:4]).pow(2) * valid).sum() / valid.sum().clamp_min(1.0)
        out["trajectory_mse_m2"] = pos_mse + 0.10 * vel_mse
    if float(weights.get("interaction", 0.0)) > 0.0:
        ego = item["ego_future_states_raw"]
        pred_gap = torch.abs(pred[..., 0] - ego[:, :, None, 0])
        target_gap = torch.abs(target[..., 0] - ego[:, :, None, 0])
        pred_rel_v = pred[..., 2] - ego[:, :, None, 2]
        target_rel_v = target[..., 2] - ego[:, :, None, 2]
        mask = item["target_valid"].float()
        gap_loss = (torch.abs(pred_gap - target_gap) * mask).sum() / mask.sum().clamp_min(1.0)
        rel_v_loss = (torch.abs(pred_rel_v - target_rel_v) * mask).sum() / mask.sum().clamp_min(1.0)
        out["interaction_mae"] = gap_loss + 0.25 * rel_v_loss
    if float(weights.get("physics", 0.0)) > 0.0:
        ego = item["ego_future_states_raw"]
        rel_x = pred[..., 0] - ego[:, :, None, 0]
        rel_y = pred[..., 1] - ego[:, :, None, 1]
        longitudinal_gap = torch.abs(rel_x) - 4.75
        lateral_clearance = torch.abs(rel_y) - 1.9
        mask = item["target_valid"].float()
        negative_gap = torch.nn.functional.softplus(-longitudinal_gap) * mask
        overlap = (
            torch.nn.functional.softplus(-longitudinal_gap)
            * torch.nn.functional.softplus(-lateral_clearance)
            * mask
        )
        out["physics_soft_penalty"] = (negative_gap + overlap).sum() / mask.sum().clamp_min(1.0)
    return out


def _epoch(
    model,
    loader,
    device,
    *,
    schema: dict[str, Any],
    training_cfg: dict[str, Any],
    optimizer=None,
    grad_clip: float = 0.0,
    amp_enabled: bool = False,
    amp_dtype=None,
    grad_scaler=None,
) -> dict[str, float]:
    torch, _, _ = _torch()
    train = optimizer is not None
    model.train(train)
    totals: dict[str, list[Any]] = {}
    total_n = 0
    batch_keys = getattr(loader, "world_model_batch_keys", DEFAULT_BATCH_KEYS)
    loss_weights = _loss_weights(training_cfg)
    history_noise_scale = None
    history_noise_enabled: bool | None = None
    if train:
        history_noise_values = _history_corruption_noise(training_cfg, schema)
        history_noise_enabled = bool(np.any(history_noise_values > 0))
        if history_noise_enabled:
            history_noise_scale = torch.from_numpy(history_noise_values).to(device=device)

    def add_metric(name: str, value, n: int) -> None:
        if not hasattr(value, "detach") or value.ndim != 0:
            return
        scaled = value.detach().to(dtype=torch.float64) * float(n)
        totals.setdefault(name, []).append(scaled)

    epoch_context = torch.enable_grad if train else getattr(torch, "inference_mode", torch.no_grad)
    with epoch_context():
        for batch in loader:
            item = _batch_to_device(batch, device, batch_keys)
            if train:
                _apply_history_corruption(
                    item,
                    training_cfg,
                    schema,
                    noise_scale=history_noise_scale,
                    noise_enabled=history_noise_enabled,
                )
            with _autocast_context(torch, device, amp_enabled, amp_dtype):
                if hasattr(model, "p_losses"):
                    losses = model.p_losses(item)
                    loss = losses["loss"]
                    if "pred_actions_normalized" in losses:
                        aux_losses = _auxiliary_rollout_losses(
                            item,
                            losses["pred_actions_normalized"],
                            schema,
                            training_cfg,
                            weights=loss_weights,
                        )
                        for aux_name, aux_value in aux_losses.items():
                            loss = loss + float(loss_weights.get(
                                aux_name.replace("_mse_m2", "").replace("_mae", "").replace("_soft_penalty", ""),
                                0.0,
                            )) * aux_value
                            losses[aux_name] = aux_value
                        losses["loss"] = loss
                    mse = losses["action_mse_norm"]
                else:
                    mean, log_std = model(
                        item["history_states"],
                        item["history_valid"],
                        item["current_states"],
                        item["current_valid"],
                        item["mode_index"],
                        item["primary_slot_index"],
                        item["flow_action_summary"],
                        item.get("relation_features"),
                    )
                    loss = gaussian_nll(
                        item["target_actions"],
                        mean,
                        log_std,
                        item["target_valid"],
                        item["sample_weight"],
                    )
                    losses = {
                        "loss": loss,
                        "action_mse_norm": masked_action_mse(item["target_actions"], mean, item["target_valid"]),
                    }
                    aux_losses = _auxiliary_rollout_losses(item, mean, schema, training_cfg, weights=loss_weights)
                    for aux_name, aux_value in aux_losses.items():
                        loss = loss + float(loss_weights.get(
                            aux_name.replace("_mse_m2", "").replace("_mae", "").replace("_soft_penalty", ""),
                            0.0,
                        )) * aux_value
                        losses[aux_name] = aux_value
                    losses["loss"] = loss
                    mse = masked_action_mse(item["target_actions"], mean, item["target_valid"])
            if train:
                optimizer.zero_grad(set_to_none=True)
                if grad_scaler is not None:
                    grad_scaler.scale(loss).backward()
                    if grad_clip and float(grad_clip) > 0:
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    if grad_clip and float(grad_clip) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                    optimizer.step()
            n = int(item["target_actions"].shape[0])
            add_metric("loss", loss, n)
            add_metric("action_mse_norm", mse, n)
            for loss_name, loss_value in losses.items():
                if loss_name in {"loss", "action_mse_norm"}:
                    continue
                add_metric(loss_name, loss_value, n)
            total_n += n
        denom = float(max(total_n, 1))
        metrics: dict[str, float] = {}
        for key, values in totals.items():
            stacked = torch.stack([value.reshape(()) for value in values])
            metrics[key] = float(sum(stacked.detach().cpu().tolist()) / denom)
    return metrics


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames = [field for field in HISTORY_FIELDS if any(field in row for row in rows)]
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _numeric_metric(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _metric_or_nan(metrics: dict[str, float] | None, key: str) -> float:
    if not metrics:
        return float("nan")
    return float(metrics.get(key, float("nan")))


def _maybe_make_loader(
    arrays: dict[str, np.ndarray],
    split: str,
    *,
    batch_size: int,
    num_workers: int,
    max_samples: int,
    mode_index: int | None = None,
    evt_tail: bool | None = None,
    include_relation_features: bool = False,
    include_auxiliary_states: bool = False,
):
    try:
        return _make_loader(
            arrays,
            split,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            max_samples=max_samples,
            mode_index=mode_index,
            evt_tail=evt_tail,
            include_relation_features=include_relation_features,
            include_auxiliary_states=include_auxiliary_states,
        )
    except RuntimeError:
        return None


def _make_summary_writer(out_dir: Path):
    log_dir = ensure_dir(out_dir / "tensorboard")
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # noqa: BLE001 - tensorboard is optional outside tread.
        logger.warning("TensorBoard writer unavailable: %s", exc)
        return None, log_dir
    return SummaryWriter(log_dir=str(log_dir)), log_dir


def _write_tensorboard_scalars(writer, row: dict[str, Any]) -> None:
    if writer is None:
        return
    epoch = int(row["epoch"])
    mapping = {
        "loss/train": "train_loss",
        "loss/val": "val_loss",
        "loss/val_start": "val_start_loss",
        "loss/val_roll": "val_roll_loss",
        "loss/val_evt_tail": "val_evt_tail_loss",
        "mse/train_action_norm": "train_action_mse_norm",
        "mse/val_action_norm": "val_action_mse_norm",
        "mse/val_start_action_norm": "val_start_action_mse_norm",
        "mse/val_roll_action_norm": "val_roll_action_mse_norm",
        "mse/val_evt_tail_action_norm": "val_evt_tail_action_mse_norm",
        "lr": "lr",
    }
    for tag, key in mapping.items():
        value = _numeric_metric(row, key)
        if np.isfinite(value):
            writer.add_scalar(tag, value, epoch)


def train_world_model(
    config: dict[str, Any],
    *,
    config_dir: str | Path,
    epochs: int | None = None,
    max_train_samples: int = 0,
    max_val_samples: int = 0,
    rebuild_dataset: bool = False,
    dataset_max_segments: int | None = None,
) -> dict[str, Any]:
    config_dir = Path(config_dir).resolve()
    out_dir = output_dir_from_config(config, config_dir)
    data_dir = dataset_dir_from_config(config, config_dir)
    training = dict(config.get("training", {}))
    total_epochs = int(epochs or training.get("epochs", 30))
    summary_path = out_dir / "training_summary.json"
    fast_resume_enabled = (
        bool(training.get("resume_from_checkpoint", False))
        and not bool(rebuild_dataset)
        and int(max_train_samples) <= 0
        and int(max_val_samples) <= 0
        and summary_path.exists()
    )
    if fast_resume_enabled:
        existing_summary = load_json(summary_path)
        if int(existing_summary.get("epochs_completed", 0)) >= total_epochs:
            logger.info(
                "Existing training summary already reached epoch=%d; skipping data cache/load.",
                int(existing_summary.get("epochs_completed", 0)),
            )
            return existing_summary

    data_available = prepared_dataset_available(data_dir)
    if rebuild_dataset or not data_available:
        build_world_model_dataset(
            config,
            config_dir=config_dir,
            max_segments=dataset_max_segments,
            rebuild=rebuild_dataset,
        )

    set_seed(int(training.get("seed", 42)))
    device = select_device(training.get("device", "auto"))
    torch, _, _ = _torch()
    torch_performance = _configure_torch_performance(torch, training, device)
    amp_enabled, amp_dtype, mixed_precision_mode = _resolve_mixed_precision(torch, training, device)
    grad_scaler = _make_grad_scaler(torch, mixed_precision_mode)
    arrays, schema, dataset_cache_path = load_world_model_training_dataset(
        data_dir,
        config=config,
    )
    model = build_model_from_schema(schema, config).to(device)
    batch_size = int(training.get("batch_size", 256))
    num_workers = int(training.get("num_workers", 0))
    include_relation_features = training_uses_relation_features(config)
    include_auxiliary_states = training_uses_auxiliary_states(config)
    train_loader = _make_loader(
        arrays,
        "train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        max_samples=max_train_samples,
        include_relation_features=include_relation_features,
        include_auxiliary_states=include_auxiliary_states,
    )
    val_loader = _make_loader(
        arrays,
        "val",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        max_samples=max_val_samples,
        include_relation_features=include_relation_features,
        include_auxiliary_states=include_auxiliary_states,
    )
    val_start_loader = _maybe_make_diagnostic_loader_from_val(
        val_loader,
        arrays,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_val_samples,
        mode_index=START_MODE_INDEX,
    ) or _maybe_make_loader(
        arrays,
        "val",
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_val_samples,
        mode_index=START_MODE_INDEX,
        include_relation_features=include_relation_features,
        include_auxiliary_states=include_auxiliary_states,
    )
    val_roll_loader = _maybe_make_diagnostic_loader_from_val(
        val_loader,
        arrays,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_val_samples,
        mode_index=ROLL_MODE_INDEX,
    ) or _maybe_make_loader(
        arrays,
        "val",
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_val_samples,
        mode_index=ROLL_MODE_INDEX,
        include_relation_features=include_relation_features,
        include_auxiliary_states=include_auxiliary_states,
    )
    val_evt_tail_loader = _maybe_make_diagnostic_loader_from_val(
        val_loader,
        arrays,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_val_samples,
        evt_tail=True,
    ) or _maybe_make_loader(
        arrays,
        "val",
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_val_samples,
        evt_tail=True,
        include_relation_features=include_relation_features,
        include_auxiliary_states=include_auxiliary_states,
    )
    num_train_samples = int(len(train_loader.dataset))
    num_val_samples = int(len(val_loader.dataset))
    del arrays
    gc.collect()
    optimizer, fused_adamw_enabled = _make_adamw_optimizer(torch, model, training, device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs, 1),
        eta_min=float(training.get("min_learning_rate", 1.0e-5)),
    )
    grad_clip = float(training.get("grad_clip", 1.0))
    best_val = float("inf")
    best_epoch = -1
    ckpt = checkpoint_path(out_dir)
    ensure_dir(ckpt.parent)
    history_path = out_dir / "training_history.csv"
    resume_enabled = bool(training.get("resume_from_checkpoint", False)) and not bool(rebuild_dataset)
    history_rows = _read_history(history_path) if resume_enabled else []
    start_epoch = 0
    resumed_from_checkpoint = False
    if history_rows:
        start_epoch = max(int(_numeric_metric(row, "epoch", 0)) for row in history_rows)
        finite_val_rows = [
            row for row in history_rows if np.isfinite(_numeric_metric(row, "val_loss"))
        ]
        if finite_val_rows:
            best_row = min(finite_val_rows, key=lambda row: _numeric_metric(row, "val_loss"))
            best_val = _numeric_metric(best_row, "val_loss")
            best_epoch = int(_numeric_metric(best_row, "epoch", -1))

    if resume_enabled and ckpt.exists():
        payload = torch.load(str(ckpt), map_location=device)
        missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
        disallowed_missing = [
            key for key in missing
            if not key.startswith("relation_proj.")
        ]
        if disallowed_missing or unexpected:
            raise RuntimeError(
                "Checkpoint state_dict mismatch: "
                f"missing={disallowed_missing}, unexpected={list(unexpected)}"
            )
        best_val = float(payload.get("best_val_loss", best_val))
        best_epoch = int(payload.get("best_epoch", best_epoch))
        if history_rows and best_epoch >= 0 and best_epoch < start_epoch:
            history_rows = [
                row for row in history_rows
                if int(_numeric_metric(row, "epoch", 0)) <= best_epoch
            ]
            start_epoch = best_epoch
            logger.info(
                "Truncated history to checkpoint epoch=%d before continuing.",
                best_epoch,
            )
        elif not history_rows and best_epoch > start_epoch:
            start_epoch = best_epoch
            logger.info("Continuing from checkpoint epoch=%d without history rows.", best_epoch)
        if "optimizer_state" in payload:
            try:
                optimizer.load_state_dict(payload["optimizer_state"])
            except (TypeError, ValueError, RuntimeError) as exc:
                logger.warning("Skipping incompatible optimizer state from %s: %s", ckpt, exc)
        if "scheduler_state" in payload:
            try:
                scheduler.load_state_dict(payload["scheduler_state"])
            except (TypeError, ValueError, RuntimeError) as exc:
                logger.warning("Skipping incompatible scheduler state from %s: %s", ckpt, exc)
        resumed_from_checkpoint = True
        logger.info(
            "Resumed model from %s at best_epoch=%s best_val=%.6f",
            ckpt,
            best_epoch,
            best_val,
        )

    if start_epoch >= total_epochs:
        logger.info(
            "Existing history already reached epoch=%d; set a larger epoch count to continue.",
            start_epoch,
        )
    runtime_model, compile_enabled = (
        _maybe_compile_model(torch, model, training)
        if start_epoch < total_epochs
        else (model, False)
    )
    patience = int(training.get("no_improvement_patience", 0) or 0)
    min_delta = float(training.get("no_improvement_min_delta", 0.0) or 0.0)
    patience_best_val = float(best_val)
    epochs_without_improvement = 0
    writer, tensorboard_dir = _make_summary_writer(out_dir)
    logger.info(
        "Training world model on %s for epochs %d..%d, device=%s, train=%d val=%d",
        out_dir,
        start_epoch + 1,
        total_epochs,
        device,
        num_train_samples,
        num_val_samples,
    )
    stopped_early = False
    for epoch in range(start_epoch + 1, total_epochs + 1):
        detail_interval = int(training.get("diagnostic_validation_interval", 5) or 0)
        run_diagnostic_val = (
            detail_interval <= 1
            or epoch == start_epoch + 1
            or epoch == total_epochs
            or (detail_interval > 1 and epoch % detail_interval == 0)
        )
        train_metrics = _epoch(
            runtime_model,
            train_loader,
            device,
            schema=schema,
            training_cfg=training,
            optimizer=optimizer,
            grad_clip=grad_clip,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            grad_scaler=grad_scaler,
        )
        val_metrics = _epoch(
            runtime_model,
            val_loader,
            device,
            schema=schema,
            training_cfg=training,
            optimizer=None,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        val_start_metrics = (
            _epoch(
                runtime_model,
                val_start_loader,
                device,
                schema=schema,
                training_cfg=training,
                optimizer=None,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            if run_diagnostic_val and val_start_loader is not None
            else None
        )
        val_roll_metrics = (
            _epoch(
                runtime_model,
                val_roll_loader,
                device,
                schema=schema,
                training_cfg=training,
                optimizer=None,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            if run_diagnostic_val and val_roll_loader is not None
            else None
        )
        val_evt_tail_metrics = (
            _epoch(
                runtime_model,
                val_evt_tail_loader,
                device,
                schema=schema,
                training_cfg=training,
                optimizer=None,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            if run_diagnostic_val and val_evt_tail_loader is not None
            else None
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_metrics["loss"]),
            "train_action_mse_norm": float(train_metrics.get("action_mse_norm", float("nan"))),
            "val_loss": float(val_metrics["loss"]),
            "val_action_mse_norm": float(val_metrics.get("action_mse_norm", float("nan"))),
            "val_start_loss": _metric_or_nan(val_start_metrics, "loss"),
            "val_start_action_mse_norm": _metric_or_nan(val_start_metrics, "action_mse_norm"),
            "val_roll_loss": _metric_or_nan(val_roll_metrics, "loss"),
            "val_roll_action_mse_norm": _metric_or_nan(val_roll_metrics, "action_mse_norm"),
            "val_evt_tail_loss": _metric_or_nan(val_evt_tail_metrics, "loss"),
            "val_evt_tail_action_mse_norm": _metric_or_nan(val_evt_tail_metrics, "action_mse_norm"),
        }
        for prefix, metrics in (
            ("train", train_metrics),
            ("val", val_metrics),
            ("val_start", val_start_metrics),
            ("val_roll", val_roll_metrics),
            ("val_evt_tail", val_evt_tail_metrics),
        ):
            if not metrics:
                continue
            for key, value in metrics.items():
                out_key = f"{prefix}_{key}"
                if out_key not in row and np.isfinite(float(value)):
                    row[out_key] = float(value)
        history_rows.append(row)
        _write_history(history_path, history_rows)
        _write_tensorboard_scalars(writer, row)
        if writer is not None:
            writer.flush()
        logger.info(
            "epoch=%03d train_loss=%.5f val_loss=%.5f val_start=%.5f val_roll=%.5f val_evt_tail=%.5f",
            epoch,
            row["train_loss"],
            row["val_loss"],
            row["val_start_loss"],
            row["val_roll_loss"],
            row["val_evt_tail_loss"],
        )
        strict_improved = row["val_loss"] < best_val
        material_improved = row["val_loss"] < patience_best_val - min_delta
        if strict_improved:
            best_val = row["val_loss"]
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "model_config": model_config_payload(model),
                    "schema": schema,
                    "config": config,
                    "best_epoch": int(best_epoch),
                    "best_val_loss": float(best_val),
                },
                ckpt,
            )
        if material_improved:
            patience_best_val = row["val_loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                logger.info(
                    "Stopping after %d epochs without val improvement >= %.6g",
                    epochs_without_improvement,
                    min_delta,
                )
                stopped_early = True
                break

    _write_history(history_path, history_rows)
    if writer is not None:
        writer.close()
    epochs_completed = max(
        [int(_numeric_metric(row, "epoch", 0)) for row in history_rows] or [0]
    )
    summary = {
        "checkpoint": str(ckpt),
        "training_history": str(history_path),
        "dataset_dir": str(data_dir),
        "dataset_cache": str(dataset_cache_path),
        "dataset_format": str(schema.get("dataset_format", "")),
        "dataset_cache_uses_relation_features": bool(include_relation_features),
        "dataset_cache_uses_auxiliary_states": bool(include_auxiliary_states),
        "diagnostic_validation_interval": int(training.get("diagnostic_validation_interval", 5) or 0),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "epochs": int(total_epochs),
        "epochs_completed": int(epochs_completed),
        "stopped_early": bool(stopped_early),
        "no_improvement_patience": int(patience),
        "no_improvement_min_delta": float(min_delta),
        "patience_best_val_loss": float(patience_best_val),
        "start_epoch": int(start_epoch),
        "resumed_from_checkpoint": bool(resumed_from_checkpoint),
        "tensorboard_log_dir": str(tensorboard_dir),
        "tensorboard_available": writer is not None,
        "device": str(device),
        "performance": {
            **torch_performance,
            "mixed_precision": mixed_precision_mode,
            "mixed_precision_enabled": bool(amp_enabled),
            "fused_adamw": bool(fused_adamw_enabled),
            "torch_compile": bool(compile_enabled),
            "torch_compile_requested": bool(_performance_config(training).get("compile_model", False)),
        },
        "num_train_samples": int(num_train_samples),
        "num_val_samples": int(num_val_samples),
        "use_start_flow_summary": bool(config.get("model", {}).get("use_start_flow_summary", False)),
        "closed_loop_adaptation": dict(training.get("closed_loop_adaptation", {})),
    }
    save_json(summary, out_dir / "training_summary.json")
    return summary

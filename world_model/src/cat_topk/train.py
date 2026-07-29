"""Training loop for the FiT-AMS START/ROLL background world model."""
from __future__ import annotations

import contextlib
import csv
import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np

from world_model.src.core.data import (
    aligned_multichunk_indices,
    build_world_model_dataset,
    checkpoint_path,
    dataset_dir_from_config,
    load_world_model_training_dataset,
    output_dir_from_config,
    prepared_dataset_available,
    split_indices,
)
from .model import (
    build_model_from_schema,
    model_config_payload,
)
from world_model.src.core.schema import ROLL_MODE_INDEX, START_MODE_INDEX
from world_model.src.core.utils import ensure_dir, load_json, save_json, select_device, set_seed

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
    ("relation_features", "relation_features_normalized", "float"),
    ("target_actions", "target_actions_normalized", "float"),
    ("target_valid", "target_valid", "bool"),
    ("sample_weight", "sample_weight", "float"),
)
SEQUENCE_BATCH_FIELDS = (
    ("history_states", "history_states_normalized", "float"),
    ("history_valid", "history_valid", "bool"),
    ("current_states", "current_states_normalized", "float"),
    ("current_states_raw", "current_states", "float"),
    ("current_valid", "current_valid", "bool"),
    ("mode_index", "mode_index", "long"),
    ("primary_slot_index", "primary_slot_index", "long"),
    ("flow_action_summary", "flow_action_summary_normalized", "float"),
    ("target_actions", "target_actions_normalized", "float"),
    ("target_valid", "target_valid", "bool"),
    ("sample_weight", "sample_weight", "float"),
    ("ego_future_states_raw", "ego_future_states", "float"),
    ("ego_future_valid", "ego_future_valid", "bool"),
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
    fields = CORE_BATCH_FIELDS
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


def _make_multichunk_loader(
    arrays: dict[str, np.ndarray],
    split: str,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    max_sequences: int,
    horizon_steps: int,
    max_chunks: int,
):
    """Create a lazy loader over aligned cached START/ROLL rows."""
    torch, DataLoader, _TensorDataset = _torch()
    from torch.utils.data import Dataset

    sequence_indices = aligned_multichunk_indices(
        arrays,
        split,
        horizon_steps=int(horizon_steps),
        max_chunks=int(max_chunks),
    )
    if max_sequences and int(max_sequences) > 0:
        sequence_indices = sequence_indices[: int(max_sequences)]
    if len(sequence_indices) == 0:
        raise RuntimeError(f"No aligned multi-chunk sequences for split={split}")

    fields = tuple(SEQUENCE_BATCH_FIELDS)

    class _SequenceDataset(Dataset):
        def __len__(self) -> int:
            return len(sequence_indices)

        def __getitem__(self, item_index: int):
            sample_indices = sequence_indices[int(item_index)]
            return tuple(
                _tensor_from_array(torch, np.asarray(arrays[array_key][sample_indices]), dtype_name)
                for _name, array_key, dtype_name in fields
            )

    loader = DataLoader(
        _SequenceDataset(),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
    )
    loader.world_model_batch_keys = tuple(name for name, _array_key, _dtype_name in fields)
    loader.world_model_sequence = True
    loader.world_model_sequence_count = int(len(sequence_indices))
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


def _normalize_torch(values, valid, normalization: dict[str, Any]):
    """按 schema 归一化并将无效项置零。"""
    torch, _, _ = _torch()
    shape = (*([1] * (values.ndim - 1)), -1)
    mean = torch.as_tensor(normalization["mean"], device=values.device, dtype=values.dtype).view(shape)
    std = torch.as_tensor(normalization["std"], device=values.device, dtype=values.dtype).view(shape)
    normalized = (values - mean) / std
    return torch.where(valid.unsqueeze(-1), normalized, torch.zeros_like(normalized))


def _normalize_states_torch(raw_states, valid, schema: dict[str, Any]):
    return _normalize_torch(raw_states, valid, schema["normalization"]["state"])


def _unnormalize_states_torch(normalized_states, valid, schema: dict[str, Any]):
    torch, _, _ = _torch()
    norm = schema["normalization"]["state"]
    mean = torch.as_tensor(norm["mean"], device=normalized_states.device, dtype=normalized_states.dtype)
    std = torch.as_tensor(norm["std"], device=normalized_states.device, dtype=normalized_states.dtype)
    raw = normalized_states * std.view(*([1] * (normalized_states.ndim - 1)), -1) + mean.view(*([1] * (normalized_states.ndim - 1)), -1)
    return torch.where(valid.unsqueeze(-1), raw, torch.zeros_like(raw))


def _normalize_relation_torch(raw_features, slot_valid, schema: dict[str, Any]):
    return _normalize_torch(raw_features, slot_valid, schema["normalization"]["relation_features"])


def _relation_features_torch(current_states, current_valid, primary_slot_index):
    """Torch equivalent of the fixed current-state relation transform."""
    torch, _, _ = _torch()
    ego = current_states[:, :1, :]
    slots = current_states[:, 1:, :]
    valid = current_valid[:, 1:]
    rel_x = slots[..., 0] - ego[..., 0]
    rel_y = slots[..., 1] - ego[..., 1]
    rel_vx = slots[..., 2] - ego[..., 2]
    rel_vy = slots[..., 3] - ego[..., 3]
    gap = torch.clamp(torch.abs(rel_x) - 4.75, min=0.0)
    closing = torch.clamp(torch.where(rel_x >= 0.0, -rel_vx, rel_vx), min=0.0)
    ttc = torch.clamp(gap / closing.clamp_min(1.0e-3), min=0.0, max=10.0)
    drac = torch.clamp(closing.square() / (2.0 * gap.clamp_min(1.0e-3)), min=0.0, max=12.0)
    primary = torch.nn.functional.one_hot(
        primary_slot_index.long().clamp(0, slots.shape[1] - 1),
        num_classes=slots.shape[1],
    ).to(dtype=current_states.dtype)
    features = torch.stack(
        (rel_x, gap, rel_y, rel_vx, rel_vy, closing, ttc, drac, primary, valid.float()),
        dim=-1,
    )
    return torch.where(valid.unsqueeze(-1), features, torch.zeros_like(features))


def _sequence_chunk_item(sequence: dict[str, Any], chunk_index: int, schema: dict[str, Any], previous: dict[str, Any] | None):
    """Build one START/ROLL condition, replacing only background history after START."""
    torch, _, _ = _torch()
    if previous is None:
        current_raw = sequence["current_states_raw"][:, chunk_index]
        current_valid = sequence["current_valid"][:, chunk_index]
        history_valid = sequence["history_valid"][:, chunk_index]
        flow_summary = sequence["flow_action_summary"][:, chunk_index]
    else:
        history_raw_logged = _unnormalize_states_torch(
            sequence["history_states"][:, chunk_index],
            sequence["history_valid"][:, chunk_index],
            schema,
        )
        ego_history = history_raw_logged[:, :, 0, :]
        ego_valid = sequence["history_valid"][:, chunk_index, :, 0]
        background_history = previous["background_states"].clone()
        ego_shift = previous["ego_future_states"][:, -1, :2]
        background_history[..., :2] = background_history[..., :2] - ego_shift[:, None, None, :]
        history_raw = torch.zeros(
            (
                background_history.shape[0],
                background_history.shape[1],
                1 + background_history.shape[2],
                background_history.shape[3],
            ),
            device=background_history.device,
            dtype=background_history.dtype,
        )
        history_valid = torch.zeros(
            history_raw.shape[:-1],
            device=background_history.device,
            dtype=torch.bool,
        )
        history_raw[:, :, 0, :] = ego_history
        history_valid[:, :, 0] = ego_valid
        history_raw[:, :, 1:, :] = background_history
        history_valid[:, :, 1:] = previous["background_valid"].unsqueeze(1)
        history_raw = torch.where(history_valid.unsqueeze(-1), history_raw, torch.zeros_like(history_raw))
        current_raw = history_raw[:, -1]
        current_valid = history_valid[:, -1]
        flow_summary = torch.zeros_like(sequence["flow_action_summary"][:, chunk_index])
    primary = sequence["primary_slot_index"][:, chunk_index]
    relation_raw = _relation_features_torch(current_raw, current_valid, primary)
    return {
        "history_states": (
            sequence["history_states"][:, chunk_index]
            if previous is None
            else _normalize_states_torch(history_raw, history_valid, schema)
        ),
        "history_valid": history_valid,
        "current_states": _normalize_states_torch(current_raw, current_valid, schema),
        "current_valid": current_valid,
        "current_states_raw": current_raw,
        "mode_index": sequence["mode_index"][:, chunk_index],
        "primary_slot_index": primary,
        "flow_action_summary": flow_summary,
        "relation_features": _normalize_relation_torch(relation_raw, current_valid[:, 1:], schema),
        "target_actions": sequence["target_actions"][:, chunk_index],
        "target_valid": sequence["target_valid"][:, chunk_index],
        "sample_weight": sequence["sample_weight"][:, chunk_index],
    }


def _multichunk_epoch(
    model,
    loader,
    device,
    *,
    schema: dict[str, Any],
    optimizer,
    grad_clip: float,
    amp_enabled: bool,
    amp_dtype,
    grad_scaler,
    active_chunks: int,
    state_consistency_weight: float,
) -> dict[str, float]:
    """Train on model-generated background history across consecutive chunks."""
    torch, _, _ = _torch()
    model.train(True)
    totals: dict[str, float] = {}
    total_n = 0
    batch_keys = getattr(loader, "world_model_batch_keys", DEFAULT_BATCH_KEYS)
    for batch in loader:
        sequence = _batch_to_device(batch, device, batch_keys)
        max_chunks = min(int(active_chunks), int(sequence["mode_index"].shape[1]))
        previous: dict[str, Any] | None = None
        aggregate_loss = None
        scalar_metrics: dict[str, Any] = {}
        for chunk_index in range(max_chunks):
            item = _sequence_chunk_item(sequence, chunk_index, schema, previous)
            with _autocast_context(torch, device, amp_enabled, amp_dtype):
                losses = model.p_losses(item)
                chunk_loss = losses["loss"]
                selected_actions, selected_xi = model.select_map_actions_st(
                    losses["candidate_actions_normalized"],
                    losses["candidate_probabilities"],
                )
                actions_raw = _unnormalize_actions_torch(selected_actions, schema)
                predicted_background = _integrate_actions_torch(
                    item["current_states_raw"],
                    actions_raw,
                    dt=1.0 / float(schema["fps"]),
                )
                background_valid = item["current_valid"][:, 1:]
                state_loss = torch.zeros((), device=device, dtype=chunk_loss.dtype)
                if chunk_index + 1 < max_chunks:
                    ego_shift = sequence["ego_future_states_raw"][:, chunk_index, -1, :2]
                    predicted_next = predicted_background[:, -1].clone()
                    predicted_next[..., :2] = predicted_next[..., :2] - ego_shift[:, None, :]
                    target_next = sequence["current_states_raw"][:, chunk_index + 1, 1:, :]
                    valid_next = background_valid & sequence["current_valid"][:, chunk_index + 1, 1:]
                    valid_f = valid_next.float().unsqueeze(-1)
                    position_mse = ((predicted_next[..., :2] - target_next[..., :2]).square() * valid_f).sum()
                    velocity_mse = ((predicted_next[..., 2:4] - target_next[..., 2:4]).square() * valid_f).sum()
                    state_loss = (position_mse + 0.10 * velocity_mse) / (2.0 * valid_f.sum().clamp_min(1.0))
                    chunk_loss = chunk_loss + float(state_consistency_weight) * state_loss
                aggregate_loss = chunk_loss if aggregate_loss is None else aggregate_loss + chunk_loss
                scalar_metrics["state_consistency_m2"] = scalar_metrics.get("state_consistency_m2", 0.0) + state_loss.detach()
                scalar_metrics["selected_xi_mean"] = scalar_metrics.get("selected_xi_mean", 0.0) + selected_xi.float().mean().detach()
                for key, value in losses.items():
                    if hasattr(value, "ndim") and value.ndim == 0:
                        scalar_metrics[key] = scalar_metrics.get(key, 0.0) + value.detach()
            previous = {
                "background_states": predicted_background,
                "background_valid": background_valid,
                "ego_future_states": sequence["ego_future_states_raw"][:, chunk_index],
            }
        if aggregate_loss is None:
            continue
        aggregate_loss = aggregate_loss / float(max_chunks)
        optimizer.zero_grad(set_to_none=True)
        if grad_scaler is not None:
            grad_scaler.scale(aggregate_loss).backward()
            if grad_clip > 0.0:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            aggregate_loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
        n = int(sequence["target_actions"].shape[0])
        totals["loss"] = totals.get("loss", 0.0) + float(aggregate_loss.detach().cpu()) * n
        for key, value in scalar_metrics.items():
            totals[key] = totals.get(key, 0.0) + float((value / float(max_chunks)).detach().cpu()) * n
        total_n += n
    return {key: value / float(max(total_n, 1)) for key, value in totals.items()}


def _epoch(
    model,
    loader,
    device,
    *,
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

    def add_metric(name: str, value, n: int) -> None:
        if not hasattr(value, "detach") or value.ndim != 0:
            return
        scaled = value.detach().to(dtype=torch.float64) * float(n)
        totals.setdefault(name, []).append(scaled)

    epoch_context = torch.enable_grad if train else getattr(torch, "inference_mode", torch.no_grad)
    with epoch_context():
        for batch in loader:
            item = _batch_to_device(batch, device, batch_keys)
            with _autocast_context(torch, device, amp_enabled, amp_dtype):
                losses = model.p_losses(item)
                loss = losses["loss"]
                mse = losses["action_mse_norm"]
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


def _checkpoint_payload(
    model,
    optimizer,
    scheduler,
    *,
    schema: dict[str, Any],
    config: dict[str, Any],
    epoch: int,
    best_epoch: int,
    best_val_loss: float,
) -> dict[str, Any]:
    return {
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "model_config": model_config_payload(model),
        "schema": schema,
        "config": config,
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
    }


def _maybe_make_loader(
    arrays: dict[str, np.ndarray],
    split: str,
    *,
    batch_size: int,
    num_workers: int,
    max_samples: int,
    mode_index: int | None = None,
    evt_tail: bool | None = None,
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
    arrays, schema, dataset_cache_path = load_world_model_training_dataset(data_dir)
    model = build_model_from_schema(schema, config).to(device)
    batch_size = int(training.get("batch_size", 256))
    num_workers = int(training.get("num_workers", 0))
    closed_loop_training = dict(training.get("model_state_closed_loop", {}))
    max_closed_loop_chunks = max(1, int(closed_loop_training.get("max_chunks", 5)))
    train_loader = _make_multichunk_loader(
        arrays,
        "train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        max_sequences=max_train_samples,
        horizon_steps=int(schema["horizon_steps"]),
        max_chunks=max_closed_loop_chunks,
    )
    val_loader = _make_loader(
        arrays,
        "val",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        max_samples=max_val_samples,
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
    latest_ckpt = ckpt.with_name("latest_world_model.pt")
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

    resume_ckpt = latest_ckpt if latest_ckpt.exists() else ckpt
    if resume_enabled and resume_ckpt.exists():
        payload = torch.load(str(resume_ckpt), map_location=device)
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
        resume_epoch = int(payload.get("epoch", best_epoch))
        if history_rows and resume_epoch >= 0 and resume_epoch < start_epoch:
            history_rows = [
                row for row in history_rows
                if int(_numeric_metric(row, "epoch", 0)) <= resume_epoch
            ]
            start_epoch = resume_epoch
            logger.info(
                "Truncated history to checkpoint epoch=%d before continuing.",
                resume_epoch,
            )
        elif not history_rows and resume_epoch > start_epoch:
            start_epoch = resume_epoch
            logger.info("Continuing from checkpoint epoch=%d without history rows.", resume_epoch)
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
            resume_ckpt,
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
        curriculum_epochs = max(1, int(closed_loop_training.get("curriculum_epochs_per_chunk", 8)))
        active_closed_loop_chunks = min(
            max_closed_loop_chunks,
            1 + (epoch - 1) // curriculum_epochs,
        )
        train_metrics = _multichunk_epoch(
            runtime_model,
            train_loader,
            device,
            schema=schema,
            optimizer=optimizer,
            grad_clip=grad_clip,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            grad_scaler=grad_scaler,
            active_chunks=active_closed_loop_chunks,
            state_consistency_weight=float(closed_loop_training.get("state_consistency_weight", 0.10)),
        )
        val_metrics = _epoch(
            runtime_model,
            val_loader,
            device,
            optimizer=None,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        val_start_metrics = (
            _epoch(
                runtime_model,
                val_start_loader,
                device,
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
            "active_closed_loop_chunks": int(active_closed_loop_chunks),
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
                _checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    schema=schema,
                    config=config,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_val_loss=best_val,
                ),
                ckpt,
            )
        torch.save(
            _checkpoint_payload(
                model,
                optimizer,
                scheduler,
                schema=schema,
                config=config,
                epoch=epoch,
                best_epoch=best_epoch,
                best_val_loss=best_val,
            ),
            latest_ckpt,
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
    # A resumable checkpoint is useful only while training is in progress.
    # The completed artifact exposes the validation-best checkpoint alone.
    latest_ckpt.unlink(missing_ok=True)
    epochs_completed = max(
        [int(_numeric_metric(row, "epoch", 0)) for row in history_rows] or [0]
    )
    summary = {
        "checkpoint": str(ckpt),
        "training_history": str(history_path),
        "dataset_dir": str(data_dir),
        "dataset_cache": str(dataset_cache_path),
        "dataset_format": str(schema.get("dataset_format", "")),
        "dataset_cache_uses_relation_features": True,
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
        "model_state_closed_loop": {
            **closed_loop_training,
            "enabled": True,
            "max_chunks": int(max_closed_loop_chunks),
        },
    }
    save_json(summary, out_dir / "training_summary.json")
    return summary

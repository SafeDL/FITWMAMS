"""Evaluation for open-loop, START/ROLL and tail-event world-model replay."""
from __future__ import annotations

import logging
import csv
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import (
    constant_acceleration_actions,
    constant_velocity_actions,
    idm_like_same_lane_actions,
)
from .data import (
    checkpoint_path,
    dataset_dir_from_config,
    load_world_model_dataset,
    output_dir_from_config,
    split_indices,
)
from .metrics import (
    action_error_metrics,
    branch_diversity_metrics,
    interaction_metrics,
    min_sample_trajectory_metrics,
    physical_diagnostics,
    trajectory_error_metrics,
)
from .model import gaussian_nll, load_checkpoint, numpy_batch_to_torch
from .rollout import (
    build_relation_features_from_current,
    integrate_background_actions,
    integrate_background_actions_batch,
    normalize_relation_features,
    normalize_states,
    unnormalize_actions,
)
from .schema import MODE_NAMES, ROLL_MODE_INDEX, START_MODE_INDEX, SLOT_NAMES
from .utils import ensure_dir, save_json, select_device, set_seed


logger = logging.getLogger(__name__)


def _torch():
    import torch

    return torch


def _evaluation_performance_config(config: dict[str, Any]) -> dict[str, Any]:
    evaluation = dict(config.get("evaluation", {}))
    performance = dict(evaluation.get("performance", {}))
    return {
        "allow_tf32": bool(performance.get("allow_tf32", False)),
        "float32_matmul_precision": str(performance.get("float32_matmul_precision", "highest")),
    }


def _configure_evaluation_performance(config: dict[str, Any], device) -> dict[str, Any]:
    torch = _torch()
    performance = _evaluation_performance_config(config)
    allow_tf32 = bool(performance["allow_tf32"]) and str(device).startswith("cuda")
    if str(device).startswith("cuda"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
            torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        except Exception as exc:  # noqa: BLE001 - backend knobs vary by torch build.
            logger.warning("Unable to configure evaluation TF32 backend flags: %s", exc)
    matmul_precision = str(performance["float32_matmul_precision"])
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision(matmul_precision)
        except Exception as exc:  # noqa: BLE001 - older torch may reject newer values.
            logger.warning("Unable to set evaluation float32 matmul precision=%s: %s", matmul_precision, exc)
    return {
        "allow_tf32": bool(allow_tf32),
        "float32_matmul_precision": matmul_precision,
    }


def _batched_indices(idx: np.ndarray, batch_size: int):
    for start in range(0, len(idx), int(batch_size)):
        yield idx[start : start + int(batch_size)]


def _sample_id_for(arrays: dict[str, np.ndarray], sample_idx: int) -> str:
    if "sample_id" in arrays:
        return str(arrays["sample_id"][sample_idx])
    return f"wm_{int(sample_idx):08d}"


def _target_frame_for(arrays: dict[str, np.ndarray], sample_idx: int) -> int:
    if "target_frame" in arrays:
        return int(arrays["target_frame"][sample_idx])
    return int(arrays["anchor_frame"][sample_idx]) + int(arrays["offset"][sample_idx])


def _torch_generator(device, seed: int):
    torch = _torch()
    if str(device).startswith("cuda"):
        generator = torch.Generator(device=device)
    else:
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _model_actions_normalized(
    model,
    batch: dict[str, Any],
    *,
    device,
    deterministic: bool,
    temperature: float,
    seed: int,
    std_floor_normalized: np.ndarray | None = None,
) -> np.ndarray:
    torch = _torch()
    with torch.no_grad():
        if hasattr(model, "sample_actions"):
            actions = model.sample_actions(
                batch,
                deterministic=deterministic,
                temperature=float(temperature),
                generator=_torch_generator(device, seed),
            )
        elif deterministic:
            actions, _log_std = model(
                batch["history_states"],
                batch["history_valid"],
                batch["current_states"],
                batch["current_valid"],
                batch["mode_index"],
                batch["primary_slot_index"],
                batch["flow_action_summary"],
                batch["relation_features"],
            )
        else:
            mean, log_std = model(
                batch["history_states"],
                batch["history_valid"],
                batch["current_states"],
                batch["current_valid"],
                batch["mode_index"],
                batch["primary_slot_index"],
                batch["flow_action_summary"],
                batch["relation_features"],
            )
            std = log_std.exp()
            if std_floor_normalized is not None:
                floor = torch.as_tensor(std_floor_normalized, device=std.device, dtype=std.dtype).view(1, 1, 1, -1)
                std = torch.maximum(std, floor)
            noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=_torch_generator(device, seed))
            actions = mean + noise * std * max(float(temperature), 0.0)
    return actions.detach().cpu().numpy()


def _sampling_std_floor_normalized(config: dict[str, Any], schema: dict[str, Any]) -> np.ndarray | None:
    raw = config.get("evaluation", {}).get("sampling_action_std_floor_mps2")
    if raw is None:
        return None
    if isinstance(raw, dict):
        values = [
            float(raw.get("ax_mps2", 0.0)),
            float(raw.get("ay_left_mps2", 0.0)),
        ]
    else:
        values = [float(x) for x in raw]
    if len(values) != 2 or max(values) <= 0.0:
        return None
    action_std = np.asarray(schema["normalization"]["action"]["std"], dtype=np.float32)
    return (np.asarray(values, dtype=np.float32) / np.maximum(action_std, 1.0e-6)).astype(np.float32)


def _model_weighted_nll_or_nan(model, batch: dict[str, Any]) -> float:
    torch = _torch()
    if hasattr(model, "sample_actions"):
        return float("nan")
    with torch.no_grad():
        mean, log_std = model(
            batch["history_states"],
            batch["history_valid"],
            batch["current_states"],
            batch["current_valid"],
            batch["mode_index"],
            batch["primary_slot_index"],
            batch["flow_action_summary"],
            batch["relation_features"],
        )
        nll = gaussian_nll(
            batch["target_actions"],
            mean,
            log_std,
            batch["target_valid"],
            batch["sample_weight"],
        )
    return float(nll.detach().cpu())


def _predict_indices(
    model,
    arrays: dict[str, np.ndarray],
    schema: dict[str, Any],
    idx: np.ndarray,
    *,
    device,
    batch_size: int,
    num_branches: int,
    sampling_temperature: float,
    std_floor_normalized: np.ndarray | None,
    label: str = "",
) -> dict[str, Any]:
    torch = _torch()
    dt = 1.0 / float(schema["fps"])
    pred_actions: list[np.ndarray] = []
    pred_states: list[np.ndarray] = []
    nll_values: list[float] = []
    mse_values: list[float] = []
    branch_metric_sums: dict[str, float] = {}
    branch_metric_count = 0
    model.eval()
    with torch.no_grad():
        for batch_number, batch_idx in enumerate(_batched_indices(idx, batch_size), start=1):
            if label and (batch_number == 1 or batch_number % 50 == 0):
                logger.info(
                    "Evaluating %s batch %d/%d",
                    label,
                    batch_number,
                    int(np.ceil(len(idx) / max(int(batch_size), 1))),
                )
            batch = numpy_batch_to_torch(arrays, batch_idx, device)
            pred_norm = _model_actions_normalized(
                model,
                batch,
                device=device,
                deterministic=True,
                temperature=1.0,
                seed=2003 + int(batch_idx[0]),
                std_floor_normalized=None,
            )
            nll_value = _model_weighted_nll_or_nan(model, batch)
            valid_f = batch["target_valid"].float().unsqueeze(-1)
            pred_t = torch.from_numpy(pred_norm).float().to(device)
            mse = ((batch["target_actions"] - pred_t).pow(2) * valid_f).sum() / valid_f.sum().clamp_min(1.0)
            if np.isfinite(nll_value):
                nll_values.append(nll_value)
            mse_values.append(float(mse.detach().cpu()))
            mean_raw = unnormalize_actions(pred_norm, schema)
            pred_actions.append(mean_raw)
            chunk_states, _chunk_valid = integrate_background_actions_batch(
                arrays["current_states"][batch_idx],
                arrays["current_valid"][batch_idx],
                mean_raw,
                dt=dt,
            )
            pred_states.append(chunk_states.astype(np.float32))
            if int(num_branches) > 0:
                branch_chunks: list[np.ndarray] = []
                for branch in range(int(num_branches)):
                    sample_norm = _model_actions_normalized(
                        model,
                        batch,
                        device=device,
                        deterministic=False,
                        temperature=sampling_temperature,
                        seed=1009 + int(branch),
                        std_floor_normalized=std_floor_normalized,
                    )
                    sample_raw = unnormalize_actions(sample_norm, schema)
                    sample_states, _sample_valid = integrate_background_actions_batch(
                        arrays["current_states"][batch_idx],
                        arrays["current_valid"][batch_idx],
                        sample_raw,
                        dt=dt,
                    )
                    branch_chunks.append(sample_states.astype(np.float32))
                sampled_arr = np.stack(branch_chunks, axis=0)
                branch_metrics = {}
                branch_metrics.update(
                    min_sample_trajectory_metrics(
                        sampled_arr,
                        arrays["target_states"][batch_idx],
                        arrays["target_valid"][batch_idx],
                    )
                )
                branch_metrics.update(
                    branch_diversity_metrics(sampled_arr, arrays["target_valid"][batch_idx])
                )
                batch_n = int(len(batch_idx))
                for key, value in branch_metrics.items():
                    if np.isfinite(float(value)):
                        branch_metric_sums[key] = branch_metric_sums.get(key, 0.0) + float(value) * batch_n
                branch_metric_count += batch_n

    pred_actions_arr = np.concatenate(pred_actions, axis=0)
    pred_states_arr = np.concatenate(pred_states, axis=0)
    result: dict[str, Any] = {
        "num_samples": int(len(idx)),
        "weighted_nll_normalized": float(np.mean(nll_values)) if nll_values else float("nan"),
        "action_mse_normalized": float(np.mean(mse_values)) if mse_values else float("nan"),
    }
    target_actions = arrays["target_actions"][idx]
    target_states = arrays["target_states"][idx]
    target_valid = arrays["target_valid"][idx]
    result.update(action_error_metrics(pred_actions_arr, target_actions, target_valid))
    result.update(trajectory_error_metrics(pred_states_arr, target_states, target_valid))
    result.update(
        physical_diagnostics(
            pred_states_arr,
            target_valid,
            ego_future_states=arrays["ego_future_states"][idx],
            actions=pred_actions_arr,
            dt=dt,
        )
    )
    result["target_physical"] = physical_diagnostics(
        target_states,
        target_valid,
        ego_future_states=arrays["ego_future_states"][idx],
        actions=target_actions,
        dt=dt,
    )
    result.update(
        interaction_metrics(
            pred_states_arr,
            target_states,
            target_valid,
            ego_future_states=arrays["ego_future_states"][idx],
        )
    )
    if branch_metric_count > 0:
        result.update(
            {
                key: float(value / max(branch_metric_count, 1))
                for key, value in branch_metric_sums.items()
            }
        )
        if np.isfinite(result.get("ADE_m", float("nan"))) and np.isfinite(result.get("minADE_m", float("nan"))):
            ade = max(float(result["ADE_m"]), 1.0e-12)
            result["minADE_improvement_ratio"] = float((float(result["ADE_m"]) - float(result["minADE_m"])) / ade)
    baseline_metrics: dict[str, Any] = {}
    for baseline_name, baseline_actions in {
        "constant_velocity": constant_velocity_actions(
            arrays["current_states"][idx],
            int(schema["horizon_steps"]),
        ),
        "constant_acceleration": constant_acceleration_actions(
            arrays["current_states"][idx],
            int(schema["horizon_steps"]),
        ),
        "idm_like_same_lane": idm_like_same_lane_actions(
            arrays["current_states"][idx],
            arrays["current_valid"][idx],
            int(schema["horizon_steps"]),
        ),
    }.items():
        baseline_states_arr, _baseline_valid = integrate_background_actions_batch(
            arrays["current_states"][idx],
            arrays["current_valid"][idx],
            baseline_actions,
            dt=dt,
        )
        metrics = {}
        metrics.update(action_error_metrics(baseline_actions, target_actions, target_valid))
        metrics.update(trajectory_error_metrics(baseline_states_arr, target_states, target_valid))
        metrics.update(
            physical_diagnostics(
                baseline_states_arr,
                target_valid,
                ego_future_states=arrays["ego_future_states"][idx],
                actions=baseline_actions,
                dt=dt,
            )
        )
        baseline_metrics[baseline_name] = metrics
    result["baselines"] = baseline_metrics
    return result


def _filter_indices(
    arrays: dict[str, np.ndarray],
    split: str,
    *,
    mode_index: int | None = None,
    evt_tail: bool | None = None,
    max_samples: int = 0,
) -> np.ndarray:
    idx = split_indices(arrays, split)
    if mode_index is not None:
        idx = idx[arrays["mode_index"][idx] == int(mode_index)]
    if evt_tail is not None:
        idx = idx[arrays["is_evt_tail"][idx].astype(bool) == bool(evt_tail)]
    if max_samples and int(max_samples) > 0:
        idx = idx[: int(max_samples)]
    return idx


def _closed_loop_start_roll(
    model,
    arrays: dict[str, np.ndarray],
    schema: dict[str, Any],
    split: str,
    *,
    device,
    batch_size: int,
    max_samples: int = 0,
    evt_tail: bool | None = None,
) -> dict[str, Any]:
    torch = _torch()
    horizon = int(schema["horizon_steps"])
    dt = 1.0 / float(schema["fps"])
    split_idx = split_indices(arrays, split)
    if evt_tail is not None:
        split_idx = split_idx[arrays["is_evt_tail"][split_idx].astype(bool) == bool(evt_tail)]
    start_idx = split_idx[arrays["mode_index"][split_idx] == START_MODE_INDEX]
    roll_idx = split_idx[
        (arrays["mode_index"][split_idx] == ROLL_MODE_INDEX)
        & (arrays["offset"][split_idx] == horizon)
    ]
    roll_by_segment = {
        str(arrays["segment_id"][idx]): int(idx)
        for idx in roll_idx
    }
    pairs = [
        (int(idx), roll_by_segment[str(arrays["segment_id"][idx])])
        for idx in start_idx
        if str(arrays["segment_id"][idx]) in roll_by_segment
    ]
    if max_samples and int(max_samples) > 0:
        pairs = pairs[: int(max_samples)]
    if not pairs:
        suffix = " EVT-tail" if evt_tail else ""
        return {"available": False, "reason": f"no{suffix} START/ROLL pairs for split={split}"}

    start_indices = np.asarray([p[0] for p in pairs], dtype=np.int64)
    roll_indices = np.asarray([p[1] for p in pairs], dtype=np.int64)
    pred_actions: list[np.ndarray] = []
    pred_states: list[np.ndarray] = []
    target_actions: list[np.ndarray] = []
    target_states: list[np.ndarray] = []
    target_valid: list[np.ndarray] = []
    target_ego: list[np.ndarray] = []
    nll_values: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch_start in _batched_indices(np.arange(len(pairs), dtype=np.int64), batch_size):
            sidx = start_indices[batch_start]
            ridx = roll_indices[batch_start]
            start_batch = numpy_batch_to_torch(arrays, sidx, device)
            start_pred_norm = _model_actions_normalized(
                model,
                start_batch,
                device=device,
                deterministic=True,
                temperature=1.0,
                seed=4001 + int(sidx[0]),
                std_floor_normalized=None,
            )
            start_actions_raw = unnormalize_actions(start_pred_norm, schema)
            roll_history_raw = []
            roll_history_valid = []
            roll_current_raw = []
            roll_current_valid = []
            roll_relation_features = []
            for local, start_sample_idx in enumerate(sidx):
                first_bg, first_bg_valid = integrate_background_actions(
                    arrays["current_states"][start_sample_idx],
                    arrays["current_valid"][start_sample_idx],
                    start_actions_raw[local],
                    dt=dt,
                )
                ego_hist = arrays["ego_future_states"][start_sample_idx].copy()
                ego_valid = arrays["ego_future_valid"][start_sample_idx].copy()
                hist = np.zeros((horizon, 1 + len(SLOT_NAMES), len(schema["state_features"])), dtype=np.float32)
                valid = np.zeros((horizon, 1 + len(SLOT_NAMES)), dtype=bool)
                hist[:, 0, :] = ego_hist
                valid[:, 0] = ego_valid
                hist[:, 1:, :] = first_bg
                valid[:, 1:] = first_bg_valid
                origin_xy = hist[-1, 0, :2].copy()
                hist[:, :, 0] -= origin_xy[0]
                hist[:, :, 1] -= origin_xy[1]
                hist[~valid] = 0.0
                roll_history_raw.append(hist)
                roll_history_valid.append(valid)
                roll_current_raw.append(hist[-1])
                roll_current_valid.append(valid[-1])
                relation = build_relation_features_from_current(
                    hist[-1],
                    valid[-1],
                    primary_slot_index=int(arrays["primary_slot_index"][ridx[local]]),
                )
                roll_relation_features.append(relation)

            roll_history_raw_arr = np.stack(roll_history_raw).astype(np.float32)
            roll_history_valid_arr = np.stack(roll_history_valid).astype(bool)
            roll_current_raw_arr = np.stack(roll_current_raw).astype(np.float32)
            roll_current_valid_arr = np.stack(roll_current_valid).astype(bool)
            roll_relation_arr = np.stack(roll_relation_features).astype(np.float32)
            roll_relation_valid = roll_current_valid_arr[:, 1:]
            roll_batch = {
                "history_states": torch.from_numpy(
                    normalize_states(roll_history_raw_arr, roll_history_valid_arr, schema)
                ).float().to(device),
                "history_valid": torch.from_numpy(roll_history_valid_arr).bool().to(device),
                "current_states": torch.from_numpy(
                    normalize_states(roll_current_raw_arr, roll_current_valid_arr, schema)
                ).float().to(device),
                "current_valid": torch.from_numpy(roll_current_valid_arr).bool().to(device),
                "mode_index": torch.full((len(ridx),), ROLL_MODE_INDEX, dtype=torch.long, device=device),
                "primary_slot_index": torch.from_numpy(arrays["primary_slot_index"][ridx]).long().to(device),
                "flow_action_summary": torch.from_numpy(arrays["flow_action_summary_normalized"][ridx]).float().to(device),
                "relation_features": torch.from_numpy(
                    normalize_relation_features(roll_relation_arr, roll_relation_valid, schema)
                ).float().to(device),
                "target_actions": torch.from_numpy(arrays["target_actions_normalized"][ridx]).float().to(device),
                "target_valid": torch.from_numpy(arrays["target_valid"][ridx]).bool().to(device),
                "sample_weight": torch.from_numpy(arrays["sample_weight"][ridx]).float().to(device),
            }
            roll_pred_norm = _model_actions_normalized(
                model,
                roll_batch,
                device=device,
                deterministic=True,
                temperature=1.0,
                seed=5003 + int(ridx[0]),
                std_floor_normalized=None,
            )
            nll_value = _model_weighted_nll_or_nan(model, roll_batch)
            if np.isfinite(nll_value):
                nll_values.append(nll_value)
            roll_actions_raw = unnormalize_actions(roll_pred_norm, schema)
            chunk_states = []
            for local in range(len(ridx)):
                second_bg, _second_valid = integrate_background_actions(
                    roll_current_raw_arr[local],
                    roll_current_valid_arr[local],
                    roll_actions_raw[local],
                    dt=dt,
                )
                chunk_states.append(second_bg)
            pred_actions.append(roll_actions_raw)
            pred_states.append(np.stack(chunk_states).astype(np.float32))
            target_actions.append(arrays["target_actions"][ridx])
            target_states.append(arrays["target_states"][ridx])
            target_valid.append(arrays["target_valid"][ridx])
            target_ego.append(arrays["ego_future_states"][ridx])

    pred_actions_arr = np.concatenate(pred_actions, axis=0)
    pred_states_arr = np.concatenate(pred_states, axis=0)
    target_actions_arr = np.concatenate(target_actions, axis=0)
    target_states_arr = np.concatenate(target_states, axis=0)
    target_valid_arr = np.concatenate(target_valid, axis=0)
    target_ego_arr = np.concatenate(target_ego, axis=0)
    out: dict[str, Any] = {
        "available": True,
        "num_pairs": int(len(pairs)),
        "start_to_roll_offset_steps": int(horizon),
        "weighted_nll_normalized": float(np.mean(nll_values)) if nll_values else float("nan"),
    }
    out.update(action_error_metrics(pred_actions_arr, target_actions_arr, target_valid_arr))
    out.update(trajectory_error_metrics(pred_states_arr, target_states_arr, target_valid_arr))
    out.update(
        physical_diagnostics(
            pred_states_arr,
            target_valid_arr,
            ego_future_states=target_ego_arr,
            actions=pred_actions_arr,
            dt=dt,
        )
    )
    out["target_physical"] = physical_diagnostics(
        target_states_arr,
        target_valid_arr,
        ego_future_states=target_ego_arr,
        actions=target_actions_arr,
        dt=dt,
    )
    out.update(
        interaction_metrics(
            pred_states_arr,
            target_states_arr,
            target_valid_arr,
            ego_future_states=target_ego_arr,
        )
    )
    return out


def _tail_event_reconstruction(
    model,
    arrays: dict[str, np.ndarray],
    schema: dict[str, Any],
    splits: list[str],
    *,
    device,
    batch_size: int,
    max_samples: int,
    num_branches: int,
    sampling_temperature: float,
    std_floor_normalized: np.ndarray | None,
) -> dict[str, Any]:
    """Replay logged EVT-tail events with true highD ego response as reference."""
    out: dict[str, Any] = {
        "source": "logged_highd_evt_tail",
        "uses_normalizing_generated_samples": False,
        "ego_response": "logged_highd_ego_trajectory",
        "splits": {},
    }
    for split in splits:
        split_out: dict[str, Any] = {}
        start_idx = _filter_indices(
            arrays,
            split,
            mode_index=START_MODE_INDEX,
            evt_tail=True,
            max_samples=max_samples,
        )
        if len(start_idx) > 0:
                split_out["START_first_second"] = _predict_indices(
                model,
                arrays,
                schema,
                start_idx,
                device=device,
                batch_size=batch_size,
                num_branches=num_branches,
                sampling_temperature=sampling_temperature,
                std_floor_normalized=std_floor_normalized,
                label=f"tail/{split}/START_first_second",
            )
        roll_idx = _filter_indices(
            arrays,
            split,
            mode_index=ROLL_MODE_INDEX,
            evt_tail=True,
            max_samples=max_samples,
        )
        if len(roll_idx) > 0:
                split_out["ROLL_logged_history"] = _predict_indices(
                model,
                arrays,
                schema,
                roll_idx,
                device=device,
                batch_size=batch_size,
                num_branches=num_branches,
                sampling_temperature=sampling_temperature,
                std_floor_normalized=std_floor_normalized,
                label=f"tail/{split}/ROLL_logged_history",
            )
        split_out["START_to_ROLL_logged_ego"] = _closed_loop_start_roll(
            model,
            arrays,
            schema,
            split,
            device=device,
            batch_size=batch_size,
            max_samples=max_samples,
            evt_tail=True,
        )
        if split_out:
            out["splits"][split] = split_out
    return out


def _sample_event_rows(
    model,
    arrays: dict[str, np.ndarray],
    schema: dict[str, Any],
    idx: np.ndarray,
    *,
    split: str,
    mode_name: str,
    device,
    batch_size: int,
) -> list[dict[str, Any]]:
    dt = 1.0 / float(schema["fps"])
    rows: list[dict[str, Any]] = []
    model.eval()
    for batch_idx in _batched_indices(idx, batch_size):
        batch = numpy_batch_to_torch(arrays, batch_idx, device)
        pred_norm = _model_actions_normalized(
            model,
            batch,
            device=device,
            deterministic=True,
            temperature=1.0,
            seed=7001 + int(batch_idx[0]),
            std_floor_normalized=None,
        )
        pred_actions = unnormalize_actions(pred_norm, schema)
        for local, sample_idx in enumerate(batch_idx):
            pred_states, _valid = integrate_background_actions(
                arrays["current_states"][sample_idx],
                arrays["current_valid"][sample_idx],
                pred_actions[local],
                dt=dt,
            )
            target_states = arrays["target_states"][sample_idx]
            target_actions = arrays["target_actions"][sample_idx]
            target_valid = arrays["target_valid"][sample_idx].astype(bool)
            ego = arrays["ego_future_states"][sample_idx]
            valid_count = max(int(np.sum(target_valid)), 1)
            dist = np.linalg.norm(pred_states[..., :2] - target_states[..., :2], axis=-1)
            ade = float(np.sum(dist[target_valid]) / valid_count)
            final_mask = target_valid[-1]
            fde = float(np.mean(dist[-1, final_mask])) if np.any(final_mask) else float("nan")
            action_err = np.abs(pred_actions[local] - target_actions)
            gap_pred = np.abs(pred_states[..., 0] - ego[:, None, 0])
            gap_target = np.abs(target_states[..., 0] - ego[:, None, 0])
            rel_v_pred = pred_states[..., 2] - ego[:, None, 2]
            rel_v_target = target_states[..., 2] - ego[:, None, 2]
            rel_x = pred_states[..., 0] - ego[:, None, 0]
            rel_y = pred_states[..., 1] - ego[:, None, 1]
            negative_gap = target_valid & ((np.abs(rel_x) - 4.75) <= 0.0)
            overlap = target_valid & (np.abs(rel_x) < 4.75) & (np.abs(rel_y) < 1.9)
            rows.append(
                {
                    "split": split,
                    "mode": mode_name,
                    "sample_id": _sample_id_for(arrays, int(sample_idx)),
                    "segment_id": str(arrays["segment_id"][sample_idx]),
                    "recording_id": int(arrays["recording_id"][sample_idx]),
                    "ego_id": int(arrays["ego_id"][sample_idx]),
                    "anchor_frame": int(arrays["anchor_frame"][sample_idx]),
                    "target_frame": _target_frame_for(arrays, int(sample_idx)),
                    "offset": int(arrays["offset"][sample_idx]),
                    "primary_slot_index": int(arrays["primary_slot_index"][sample_idx]),
                    "event_risk": float(arrays["event_risk"][sample_idx]),
                    "ADE_m": ade,
                    "FDE_m": fde,
                    "action_mae_ax_mps2": float(np.sum(action_err[..., 0][target_valid]) / valid_count),
                    "action_mae_ay_left_mps2": float(np.sum(action_err[..., 1][target_valid]) / valid_count),
                    "gap_mae_m": float(np.sum(np.abs(gap_pred - gap_target)[target_valid]) / valid_count),
                    "relative_vx_mae_mps": float(np.sum(np.abs(rel_v_pred - rel_v_target)[target_valid]) / valid_count),
                    "negative_gap_rate": float(np.sum(negative_gap) / valid_count),
                    "overlap_rate": float(np.sum(overlap) / valid_count),
                }
            )
    return rows


def _write_event_metrics_csv(
    model,
    arrays: dict[str, np.ndarray],
    schema: dict[str, Any],
    splits: list[str],
    *,
    out_dir: Path,
    device,
    batch_size: int,
    max_samples: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for split in splits:
        for mode_name, mode_idx in {"START": START_MODE_INDEX, "ROLL": ROLL_MODE_INDEX}.items():
            idx = _filter_indices(
                arrays,
                split,
                mode_index=mode_idx,
                evt_tail=True,
                max_samples=max_samples,
            )
            if len(idx) == 0:
                continue
            rows.extend(
                _sample_event_rows(
                    model,
                    arrays,
                    schema,
                    idx,
                    split=split,
                    mode_name=mode_name,
                    device=device,
                    batch_size=batch_size,
                )
            )
    path = ensure_dir(out_dir) / "event_metrics.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("split,mode,sample_id,segment_id\n")
    return {
        "event_metrics_csv": str(path),
        "num_event_metric_rows": int(len(rows)),
    }


def _deterministic_predictions_for_indices(
    model,
    arrays: dict[str, np.ndarray],
    schema: dict[str, Any],
    idx: np.ndarray,
    *,
    device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt = 1.0 / float(schema["fps"])
    pred_actions: list[np.ndarray] = []
    pred_states: list[np.ndarray] = []
    target_actions: list[np.ndarray] = []
    target_states: list[np.ndarray] = []
    target_valid: list[np.ndarray] = []
    for batch_idx in _batched_indices(idx, batch_size):
        batch = numpy_batch_to_torch(arrays, batch_idx, device)
        pred_norm = _model_actions_normalized(
            model,
            batch,
            device=device,
            deterministic=True,
            temperature=1.0,
            seed=9001 + int(batch_idx[0]),
            std_floor_normalized=None,
        )
        actions = unnormalize_actions(pred_norm, schema)
        states = []
        for local, sample_idx in enumerate(batch_idx):
            one_states, _valid = integrate_background_actions(
                arrays["current_states"][sample_idx],
                arrays["current_valid"][sample_idx],
                actions[local],
                dt=dt,
            )
            states.append(one_states)
        pred_actions.append(actions)
        pred_states.append(np.stack(states).astype(np.float32))
        target_actions.append(arrays["target_actions"][batch_idx])
        target_states.append(arrays["target_states"][batch_idx])
        target_valid.append(arrays["target_valid"][batch_idx])
    return (
        np.concatenate(pred_actions, axis=0),
        np.concatenate(pred_states, axis=0),
        np.concatenate(target_actions, axis=0),
        np.concatenate(target_states, axis=0),
        np.concatenate(target_valid, axis=0),
    )


def _write_basic_visualizations(
    model,
    arrays: dict[str, np.ndarray],
    schema: dict[str, Any],
    result: dict[str, Any],
    *,
    out_dir: Path,
    device,
    batch_size: int,
    num_branches: int,
    sampling_temperature: float,
    std_floor_normalized: np.ndarray | None,
    max_samples: int,
) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 - visualization is diagnostic-only.
        return {"visualizations_available": False, "reason": str(exc)}

    vis_dir = ensure_dir(out_dir / "visualizations")
    split = "test" if "test" in result.get("open_loop", {}) else next(iter(result.get("open_loop", {"val": {}})))
    idx_parts = [
        _filter_indices(arrays, split, mode_index=START_MODE_INDEX, max_samples=max_samples // 3),
        _filter_indices(arrays, split, mode_index=ROLL_MODE_INDEX, max_samples=max_samples // 3),
        _filter_indices(arrays, split, evt_tail=True, max_samples=max_samples // 3),
    ]
    idx = np.unique(np.concatenate([part for part in idx_parts if len(part) > 0]))
    if len(idx) == 0:
        return {"visualizations_available": False, "reason": "no test samples"}
    if len(idx) > max_samples:
        idx = idx[:max_samples]
    pred_actions, pred_states, target_actions, target_states, target_valid = _deterministic_predictions_for_indices(
        model,
        arrays,
        schema,
        idx,
        device=device,
        batch_size=batch_size,
    )

    dist = np.linalg.norm(pred_states[..., :2] - target_states[..., :2], axis=-1)
    valid = target_valid.astype(bool)
    ade_by_sample = (dist * valid).sum(axis=(1, 2)) / valid.sum(axis=(1, 2)).clip(min=1)
    fde_by_sample = np.nanmean(np.where(valid[:, -1, :], dist[:, -1, :], np.nan), axis=1)

    paths: dict[str, str] = {}
    plt.figure(figsize=(8, 4))
    plt.hist(ade_by_sample[np.isfinite(ade_by_sample)], bins=50, alpha=0.75, label="ADE")
    plt.hist(fde_by_sample[np.isfinite(fde_by_sample)], bins=50, alpha=0.55, label="FDE")
    plt.xlabel("trajectory error (m)")
    plt.ylabel("samples")
    plt.legend()
    plt.tight_layout()
    path = vis_dir / "trajectory_reconstruction_errors.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths["trajectory_reconstruction_errors"] = str(path)

    speed_pred = np.linalg.norm(pred_states[..., 2:4], axis=-1)[valid]
    speed_target = np.linalg.norm(target_states[..., 2:4], axis=-1)[valid]
    plt.figure(figsize=(8, 4))
    plt.hist(speed_target, bins=60, alpha=0.55, density=True, label="highD")
    plt.hist(speed_pred, bins=60, alpha=0.55, density=True, label="generated")
    plt.xlabel("speed (m/s)")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    path = vis_dir / "speed_distribution_real_vs_generated.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths["speed_distribution_real_vs_generated"] = str(path)

    for action_idx, name in enumerate(("ax_mps2", "ay_left_mps2")):
        plt.figure(figsize=(8, 4))
        plt.hist(target_actions[..., action_idx][valid], bins=80, alpha=0.55, density=True, label="highD")
        plt.hist(pred_actions[..., action_idx][valid], bins=80, alpha=0.55, density=True, label="generated")
        plt.xlabel(name)
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        path = vis_dir / f"{name}_distribution_real_vs_generated.png"
        plt.savefig(path, dpi=160)
        plt.close()
        paths[f"{name}_distribution_real_vs_generated"] = str(path)

    ego = arrays["ego_future_states"][idx]
    gap_target = np.abs(target_states[..., 0] - ego[:, :, None, 0])
    gap_pred = np.abs(pred_states[..., 0] - ego[:, :, None, 0])
    dv_target = target_states[..., 2] - ego[:, :, None, 2]
    dv_pred = pred_states[..., 2] - ego[:, :, None, 2]
    plt.figure(figsize=(6, 5))
    plt.scatter(gap_target[valid], dv_target[valid], s=3, alpha=0.18, label="highD")
    plt.scatter(gap_pred[valid], dv_pred[valid], s=3, alpha=0.18, label="generated")
    plt.xlabel("gap (m)")
    plt.ylabel("delta vx (m/s)")
    plt.legend(markerscale=4)
    plt.tight_layout()
    path = vis_dir / "phase_space_gap_delta_v.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths["phase_space_gap_delta_v"] = str(path)

    example_idx = idx[0]
    tail_start = _filter_indices(arrays, split, mode_index=START_MODE_INDEX, evt_tail=True, max_samples=1)
    if len(tail_start) > 0:
        example_idx = int(tail_start[0])
    if example_idx in set(idx.tolist()):
        local = int(np.where(idx == example_idx)[0][0])
    else:
        example_idx = int(idx[0])
        local = 0
    plt.figure(figsize=(7, 6))
    for slot_idx, slot_name in enumerate(SLOT_NAMES):
        mask = target_valid[local, :, slot_idx].astype(bool)
        if not np.any(mask):
            continue
        plt.plot(
            target_states[local, mask, slot_idx, 0],
            target_states[local, mask, slot_idx, 1],
            linewidth=1.5,
            label=f"{slot_name} highD",
        )
        plt.plot(
            pred_states[local, mask, slot_idx, 0],
            pred_states[local, mask, slot_idx, 1],
            linestyle="--",
            linewidth=1.2,
            label=f"{slot_name} gen",
        )
    plt.xlabel("x relative (m)")
    plt.ylabel("y left relative (m)")
    plt.title(
        f"segment={arrays['segment_id'][example_idx]} frame={int(arrays['anchor_frame'][example_idx])}"
    )
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    path = vis_dir / "example_rollouts.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths["example_rollouts"] = str(path)

    if num_branches > 1:
        branch_batch = numpy_batch_to_torch(arrays, np.asarray([example_idx], dtype=np.int64), device)
        dt = 1.0 / float(schema["fps"])
        plt.figure(figsize=(7, 6))
        for branch in range(int(num_branches)):
            sample_norm = _model_actions_normalized(
                model,
                branch_batch,
                device=device,
                deterministic=False,
                temperature=sampling_temperature,
                seed=10009 + branch,
                std_floor_normalized=std_floor_normalized,
            )
            sample_raw = unnormalize_actions(sample_norm, schema)
            sample_states, _valid = integrate_background_actions(
                arrays["current_states"][example_idx],
                arrays["current_valid"][example_idx],
                sample_raw[0],
                dt=dt,
            )
            for slot_idx in range(len(SLOT_NAMES)):
                mask = arrays["target_valid"][example_idx, :, slot_idx].astype(bool)
                if np.any(mask):
                    plt.plot(sample_states[mask, slot_idx, 0], sample_states[mask, slot_idx, 1], color="tab:blue", alpha=0.20)
        for slot_idx in range(len(SLOT_NAMES)):
            mask = arrays["target_valid"][example_idx, :, slot_idx].astype(bool)
            if np.any(mask):
                plt.plot(
                    arrays["target_states"][example_idx, mask, slot_idx, 0],
                    arrays["target_states"][example_idx, mask, slot_idx, 1],
                    color="black",
                    linewidth=1.4,
                )
        plt.xlabel("x relative (m)")
        plt.ylabel("y left relative (m)")
        plt.tight_layout()
        path = vis_dir / "multi_branch_envelope.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths["multi_branch_envelope"] = str(path)

    return {"visualizations_available": True, "visualizations": paths, "visualization_sample_count": int(len(idx))}


def evaluate_world_model(
    config: dict[str, Any],
    *,
    config_dir: str | Path,
    checkpoint: str | Path | None = None,
    max_samples: int = 0,
    num_branches: int = 4,
) -> dict[str, Any]:
    config_dir = Path(config_dir).resolve()
    out_dir = output_dir_from_config(config, config_dir)
    data_dir = dataset_dir_from_config(config, config_dir)
    arrays, schema = load_world_model_dataset(data_dir)
    ckpt = Path(checkpoint) if checkpoint is not None else checkpoint_path(out_dir)
    device = select_device(config.get("evaluation", {}).get("device", config.get("training", {}).get("device", "auto")))
    performance = _configure_evaluation_performance(config, device)
    set_seed(int(config.get("evaluation", {}).get("seed", 123)))
    model, payload = load_checkpoint(str(ckpt), device)
    batch_size = int(config.get("evaluation", {}).get("batch_size", config.get("training", {}).get("batch_size", 256)))
    sampling_temperature = float(config.get("evaluation", {}).get("sampling_temperature", 1.0))
    std_floor_normalized = _sampling_std_floor_normalized(config, schema)
    splits = config.get("evaluation", {}).get("splits", ["val", "test"])
    result: dict[str, Any] = {
        "checkpoint": str(ckpt),
        "dataset_dir": str(data_dir),
        "best_epoch": int(payload.get("best_epoch", -1)),
        "best_val_loss": float(payload.get("best_val_loss", float("nan"))),
        "dataset": {
            "num_samples": int(len(arrays["split_index"])),
            "schema_horizon_steps": int(schema["horizon_steps"]),
            "schema_history_steps": int(schema["history_steps"]),
            "fps": float(schema["fps"]),
        },
        "sampling_temperature": float(sampling_temperature),
        "sampling_action_std_floor_normalized": (
            std_floor_normalized.astype(float).tolist() if std_floor_normalized is not None else None
        ),
        "performance": performance,
        "open_loop": {},
    }
    for split in splits:
        split_result: dict[str, Any] = {}
        for name, mode_idx in {"START": START_MODE_INDEX, "ROLL": ROLL_MODE_INDEX}.items():
            idx = _filter_indices(arrays, split, mode_index=mode_idx, max_samples=max_samples)
            if len(idx) > 0:
                split_result[name] = _predict_indices(
                    model,
                    arrays,
                    schema,
                    idx,
                    device=device,
                    batch_size=batch_size,
                    num_branches=num_branches,
                    sampling_temperature=sampling_temperature,
                    std_floor_normalized=std_floor_normalized,
                    label=f"{split}/{name}",
                )
        tail_idx = _filter_indices(arrays, split, evt_tail=True, max_samples=max_samples)
        if len(tail_idx) > 0:
            split_result["EVT_tail"] = _predict_indices(
                model,
                arrays,
                schema,
                tail_idx,
                device=device,
                batch_size=batch_size,
                num_branches=num_branches,
                sampling_temperature=sampling_temperature,
                std_floor_normalized=std_floor_normalized,
                label=f"{split}/EVT_tail",
            )
        result["open_loop"][split] = split_result

    closed_loop_cfg = dict(config.get("evaluation", {}).get("closed_loop", {}))
    if bool(closed_loop_cfg.get("enabled", True)):
        result["closed_loop"] = {}
        for split in splits:
            result["closed_loop"][split] = _closed_loop_start_roll(
                model,
                arrays,
                schema,
                split,
                device=device,
                batch_size=batch_size,
                max_samples=int(closed_loop_cfg.get("max_samples", 0) or 0),
            )

    tail_recon_cfg = dict(config.get("evaluation", {}).get("tail_event_reconstruction", {}))
    if bool(tail_recon_cfg.get("enabled", True)):
        result["tail_event_reconstruction"] = _tail_event_reconstruction(
            model,
            arrays,
            schema,
            list(splits),
            device=device,
            batch_size=batch_size,
            max_samples=int(tail_recon_cfg.get("max_samples", max_samples) or max_samples),
            num_branches=num_branches,
            sampling_temperature=sampling_temperature,
            std_floor_normalized=std_floor_normalized,
        )
        if bool(tail_recon_cfg.get("write_event_metrics", False)):
            event_metric_summary = _write_event_metrics_csv(
                model,
                arrays,
                schema,
                list(splits),
                out_dir=out_dir,
                device=device,
                batch_size=batch_size,
                max_samples=int(tail_recon_cfg.get("max_samples", max_samples) or max_samples),
            )
            result["tail_event_reconstruction"].update(event_metric_summary)

    vis_cfg = dict(config.get("evaluation", {}).get("visualizations", {}))
    if bool(vis_cfg.get("enabled", False)):
        result["visualization_summary"] = _write_basic_visualizations(
            model,
            arrays,
            schema,
            result,
            out_dir=out_dir,
            device=device,
            batch_size=batch_size,
            num_branches=int(vis_cfg.get("num_branches", min(max(int(num_branches), 2), 8))),
            sampling_temperature=sampling_temperature,
            std_floor_normalized=std_floor_normalized,
            max_samples=int(vis_cfg.get("max_samples", 512)),
        )

    out_path = out_dir / "evaluation_summary.json"
    save_json(result, out_path)
    logger.info("Wrote world-model evaluation: %s", out_path)
    return result

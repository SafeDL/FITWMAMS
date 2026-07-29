"""Dataset construction and loading for the START/ROLL world model."""
from __future__ import annotations

import logging
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from normalizing_flow.src.features import (
    SLOT_NAMES,
    mask_pattern_from_slot_mask,
)
from process_highD.src.io_utils import load_config as load_highd_config
from process_highD.src.io_utils import resolve_data_path
from process_highD.src.natural_segments import (
    _build_vehicle_cache,
    _position_at,
    _slice_range,
    _values,
)
from process_highD.src.preprocess import prepare_recording

from .schema import (
    AGENT_NAMES,
    AGENT_STATE_FEATURES,
    FLOW_ACTION_SUMMARY_FEATURES,
    RELATION_FEATURES,
    ROLL_MODE_INDEX,
    START_MODE_INDEX,
    WorldModelSchema,
    primary_slot_index,
)
from .utils import (
    ensure_dir,
    finite_mean_std,
    load_json,
    normalize_with_mask,
    resolve_path,
    save_json,
)
from world_model.src.cat_topk.rollout import build_relation_features_from_current


logger = logging.getLogger(__name__)

SPLIT_TO_INDEX = {"train": 0, "val": 1, "test": 2}
INDEX_TO_SPLIT = {value: key for key, value in SPLIT_TO_INDEX.items()}


def output_dir_from_config(config: dict[str, Any], config_dir: str | Path) -> Path:
    return ensure_dir(resolve_path(config["paths"]["output_dir"], base=config_dir))


def dataset_dir_from_config(config: dict[str, Any], config_dir: str | Path) -> Path:
    raw = config.get("paths", {}).get("dataset_dir")
    if raw:
        return ensure_dir(resolve_path(raw, base=config_dir))
    return output_dir_from_config(config, config_dir)


def prepared_dataset_array_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "dataset_cache_arrays"


def schema_json_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "dataset_schema.json"


def checkpoint_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "checkpoints" / "best_world_model.pt"


TRAINING_CORE_ARRAY_KEYS = (
    "history_states_normalized",
    "history_valid",
    "current_states_normalized",
    "current_valid",
    "mode_index",
    "primary_slot_index",
    "flow_action_summary_normalized",
    "target_actions_normalized",
    "target_valid",
    "sample_weight",
    "split_index",
    "is_evt_tail",
)
TRAINING_RELATION_ARRAY_KEYS = ("relation_features_normalized",)
MODEL_STATE_CLOSED_LOOP_ARRAY_KEYS = (
    "current_states",
    "ego_future_states",
    "ego_future_valid",
    "segment_id",
    "offset",
)
PREPARED_DATASET_ARRAY_KEYS = (
    "history_states_normalized",
    "history_valid",
    "current_states",
    "current_valid",
    "mode_index",
    "primary_slot_index",
    "flow_action_summary_normalized",
    "relation_features_normalized",
    "target_actions",
    "target_valid",
    "target_states",
    "ego_future_states",
    "ego_future_valid",
    "sample_weight",
    "split_index",
    "is_evt_tail",
    "segment_id",
    "recording_id",
    "ego_id",
    "anchor_frame",
    "offset",
    "event_risk",
)

TRAINING_ARRAY_KEYS = (
    *TRAINING_CORE_ARRAY_KEYS,
    *TRAINING_RELATION_ARRAY_KEYS,
    *MODEL_STATE_CLOSED_LOOP_ARRAY_KEYS,
)


def _resolve_highd_evt_config(config: dict[str, Any], config_dir: Path) -> Path:
    return resolve_path(config["paths"]["highd_evt_config"], base=config_dir)


def _resolve_raw_dir(config: dict[str, Any], config_dir: Path) -> Path:
    raw = config["paths"].get("raw_dir")
    if raw:
        return resolve_path(raw, base=config_dir)
    highd_cfg_path = _resolve_highd_evt_config(config, config_dir)
    highd_cfg = load_highd_config(highd_cfg_path)
    return resolve_data_path(highd_cfg["paths"]["raw_dir"], highd_cfg_path)


def _resolve_natural_segments_csv(config: dict[str, Any], config_dir: Path) -> Path:
    raw = config["paths"].get("natural_segments_csv")
    if raw:
        return resolve_path(raw, base=config_dir)
    highd_cfg_path = _resolve_highd_evt_config(config, config_dir)
    highd_cfg = load_highd_config(highd_cfg_path)
    out_dir = resolve_data_path(highd_cfg["paths"]["output_dir"], highd_cfg_path)
    return out_dir / "natural_segments.csv"


def _resolve_tail_context_csv(config: dict[str, Any], config_dir: Path) -> Path | None:
    raw = config["paths"].get("tail_context_csv")
    if raw:
        return resolve_path(raw, base=config_dir)
    highd_cfg_path = _resolve_highd_evt_config(config, config_dir)
    highd_cfg = load_highd_config(highd_cfg_path)
    out_dir = resolve_data_path(highd_cfg["paths"]["output_dir"], highd_cfg_path)
    path = out_dir / "natural_tail_contexts.csv"
    return path if path.exists() else None


def _group_values(rows: pd.DataFrame, mode: str) -> np.ndarray:
    mode = str(mode).lower()
    if mode == "recording":
        return rows["recording_id"].astype(str).to_numpy()
    if mode in {"recording_ego", "vehicle", "ego"}:
        return (
            rows["recording_id"].astype(str) + ":" + rows["ego_id"].astype(str)
        ).to_numpy()
    if mode in {"segment", "none"}:
        return rows["segment_id"].astype(str).to_numpy()
    raise ValueError(f"Unsupported split group mode: {mode}")


def split_indices_by_group(rows: pd.DataFrame, split_cfg: dict[str, Any]) -> np.ndarray:
    groups = _group_values(rows, str(split_cfg.get("group_by", "recording_ego")))
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise RuntimeError("Need at least three split groups for train/val/test")
    rng = np.random.default_rng(int(split_cfg.get("seed", 42)))
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    ratios = np.asarray(split_cfg.get("ratios", [0.70, 0.15, 0.15]), dtype=np.float64)
    ratios = ratios / np.maximum(np.sum(ratios), 1.0e-12)
    n = len(shuffled)
    n_train = max(1, int(round(n * ratios[0])))
    n_val = max(1, int(round(n * ratios[1])))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train_groups = set(shuffled[:n_train])
    val_groups = set(shuffled[n_train : n_train + n_val])
    split_index = np.full(len(rows), SPLIT_TO_INDEX["test"], dtype=np.int64)
    for idx, group in enumerate(groups):
        if group in train_groups:
            split_index[idx] = SPLIT_TO_INDEX["train"]
        elif group in val_groups:
            split_index[idx] = SPLIT_TO_INDEX["val"]
    return split_index


def split_indices(arrays: dict[str, np.ndarray], split: str) -> np.ndarray:
    split_name = str(split).lower()
    if split_name in {"all", "full", "dataset"}:
        return np.arange(len(arrays["split_index"]), dtype=np.int64)
    if split_name not in SPLIT_TO_INDEX:
        raise KeyError(f"Unknown split={split!r}; expected {sorted(SPLIT_TO_INDEX)} or 'all'")
    return np.where(arrays["split_index"] == SPLIT_TO_INDEX[split_name])[0]


def aligned_multichunk_indices(
    arrays: dict[str, np.ndarray],
    split: str,
    *,
    horizon_steps: int,
    max_chunks: int,
) -> np.ndarray:
    """Return START plus aligned ROLL sample indices for model-state training.

    Rows remain references into the existing dense cache.  A sequence contains
    START at offset 0 followed by ROLL offsets 25, 50, 75 and 100 for the
    default highD schema; no new cache field or future conditioning is added.
    """
    chunks = max(1, int(max_chunks))
    split_idx = split_indices(arrays, split)
    starts = split_idx[
        (arrays["mode_index"][split_idx] == START_MODE_INDEX)
        & (arrays["offset"][split_idx] == 0)
    ]
    roll_lookup: dict[tuple[str, int], int] = {}
    for index in split_idx[arrays["mode_index"][split_idx] == ROLL_MODE_INDEX]:
        roll_lookup[(str(arrays["segment_id"][index]), int(arrays["offset"][index]))] = int(index)
    sequences: list[list[int]] = []
    for start_index in starts:
        segment_id = str(arrays["segment_id"][start_index])
        row = [int(start_index)]
        for chunk in range(1, chunks):
            roll_index = roll_lookup.get((segment_id, int(chunk * horizon_steps)))
            if roll_index is None:
                break
            row.append(int(roll_index))
        if len(row) == chunks:
            sequences.append(row)
    if not sequences:
        return np.empty((0, chunks), dtype=np.int64)
    return np.asarray(sequences, dtype=np.int64)


def _segment_slot_ids(row: pd.Series) -> np.ndarray:
    return np.asarray([int(row.get(f"{slot}_id", -1)) for slot in SLOT_NAMES], dtype=np.int64)


def _vehicle_sequence(
    vehicle: dict[str, Any],
    *,
    start_frame: int,
    steps: int,
    origin_x: float,
    origin_y_left: float,
) -> tuple[np.ndarray, np.ndarray]:
    selector, present = _slice_range(vehicle, int(start_frame), int(steps))
    if isinstance(selector, slice) and bool(np.all(present)):
        abnormal = np.asarray(vehicle["abnormal"][selector], dtype=bool)
    else:
        abnormal = np.zeros(int(steps), dtype=bool)
        if np.any(present):
            abnormal[present] = np.asarray(vehicle["abnormal"])[np.asarray(selector)[present]]
    valid = np.asarray(present, dtype=bool) & ~abnormal
    state = np.zeros((int(steps), len(AGENT_STATE_FEATURES)), dtype=np.float32)
    if not np.any(valid):
        return state, valid
    state[:, 0] = _values(vehicle, "x", selector, valid) - float(origin_x)
    state[:, 1] = _values(vehicle, "y_left", selector, valid) - float(origin_y_left)
    state[:, 2] = _values(vehicle, "vx", selector, valid)
    state[:, 3] = _values(vehicle, "vy_left", selector, valid)
    state[:, 4] = _values(vehicle, "ax", selector, valid)
    state[:, 5] = _values(vehicle, "ay_left", selector, valid)
    state[~valid] = 0.0
    return state.astype(np.float32), valid


def _origin_for(vehicles: dict[int, dict[str, Any]], ego_id: int, frame: int) -> tuple[float, float] | None:
    ego = vehicles.get(int(ego_id))
    if ego is None:
        return None
    pos = _position_at(ego, int(frame))
    if pos is None or bool(ego["abnormal"][pos]):
        return None
    return float(ego["x"][pos]), float(ego["y_left"][pos])


def _state_window(
    vehicles: dict[int, dict[str, Any]],
    *,
    ego_id: int,
    slot_ids: np.ndarray,
    start_frame: int,
    steps: int,
    origin_frame: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    origin = _origin_for(vehicles, int(ego_id), int(origin_frame))
    if origin is None:
        return None
    origin_x, origin_y_left = origin
    states = np.zeros(
        (int(steps), len(AGENT_NAMES), len(AGENT_STATE_FEATURES)),
        dtype=np.float32,
    )
    valid = np.zeros((int(steps), len(AGENT_NAMES)), dtype=bool)
    ids = [int(ego_id), *[int(value) for value in np.asarray(slot_ids, dtype=np.int64)]]
    for agent_idx, vehicle_id in enumerate(ids):
        if vehicle_id < 0 or vehicle_id not in vehicles:
            continue
        seq, seq_valid = _vehicle_sequence(
            vehicles[vehicle_id],
            start_frame=int(start_frame),
            steps=int(steps),
            origin_x=origin_x,
            origin_y_left=origin_y_left,
        )
        states[:, agent_idx, :] = seq
        valid[:, agent_idx] = seq_valid
    return states, valid


def _action_window(
    vehicles: dict[int, dict[str, Any]],
    *,
    slot_ids: np.ndarray,
    start_frame: int,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    actions = np.zeros((int(steps), len(SLOT_NAMES), 2), dtype=np.float32)
    valid = np.zeros((int(steps), len(SLOT_NAMES)), dtype=bool)
    for slot_idx, vehicle_id in enumerate(np.asarray(slot_ids, dtype=np.int64)):
        if int(vehicle_id) < 0 or int(vehicle_id) not in vehicles:
            continue
        vehicle = vehicles[int(vehicle_id)]
        selector, present = _slice_range(vehicle, int(start_frame), int(steps))
        if isinstance(selector, slice) and bool(np.all(present)):
            abnormal = np.asarray(vehicle["abnormal"][selector], dtype=bool)
        else:
            abnormal = np.zeros(int(steps), dtype=bool)
            if np.any(present):
                abnormal[present] = np.asarray(vehicle["abnormal"])[np.asarray(selector)[present]]
        action_valid = np.asarray(present, dtype=bool) & ~abnormal
        valid[:, slot_idx] = action_valid
        actions[:, slot_idx, 0] = _values(vehicle, "ax", selector, action_valid)
        actions[:, slot_idx, 1] = _values(vehicle, "ay_left", selector, action_valid)
        actions[~action_valid, slot_idx, :] = 0.0
    return actions, valid


def _flow_action_summary_from_actions(actions: np.ndarray, valid: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    del fps
    summary = np.zeros((len(SLOT_NAMES), len(FLOW_ACTION_SUMMARY_FEATURES)), dtype=np.float32)
    summary_valid = np.zeros((len(SLOT_NAMES), len(FLOW_ACTION_SUMMARY_FEATURES)), dtype=bool)
    for slot_idx in range(len(SLOT_NAMES)):
        mask = np.asarray(valid[:, slot_idx], dtype=bool)
        if not np.any(mask):
            continue
        ax = actions[mask, slot_idx, 0]
        ay = actions[mask, slot_idx, 1]
        summary[slot_idx, 2] = float(np.mean(ax))
        summary[slot_idx, 3] = float(np.min(ax))
        summary[slot_idx, 4] = float(ax[-1])
        summary[slot_idx, 5] = float(np.mean(ay))
        # Integrate acceleration over the valid one-second window as a leakage-controlled
        # training-time proxy for Flow's optional START intent prior.
        dt = 1.0 / max(float(len(mask)), 1.0)
        summary[slot_idx, 0] = float(np.sum(actions[:, slot_idx, 0]) * dt)
        summary[slot_idx, 1] = float(np.sum(actions[:, slot_idx, 1]) * dt)
        summary_valid[slot_idx, :] = True
    return summary, summary_valid


def _build_one_sample(
    vehicles: dict[int, dict[str, Any]],
    row: pd.Series,
    *,
    mode_index: int,
    offset: int,
    history_steps: int,
    horizon_steps: int,
    fps: float,
    is_evt_tail: bool,
    segment_split_index: int,
) -> dict[str, Any] | None:
    anchor_frame = int(row["anchor_frame"])
    target_frame = anchor_frame + int(offset)
    ego_id = int(row["ego_id"])
    slot_ids = _segment_slot_ids(row)
    slot_mask = slot_ids >= 0

    current = _state_window(
        vehicles,
        ego_id=ego_id,
        slot_ids=slot_ids,
        start_frame=target_frame,
        steps=1,
        origin_frame=target_frame,
    )
    if current is None or not bool(current[1][0, 0]):
        return None

    if int(mode_index) == START_MODE_INDEX:
        history_states = np.zeros(
            (history_steps, len(AGENT_NAMES), len(AGENT_STATE_FEATURES)),
            dtype=np.float32,
        )
        history_valid = np.zeros((history_steps, len(AGENT_NAMES)), dtype=bool)
        history_states[-1] = current[0][0]
        history_valid[-1] = current[1][0]
    else:
        history_start = target_frame - int(history_steps) + 1
        history = _state_window(
            vehicles,
            ego_id=ego_id,
            slot_ids=slot_ids,
            start_frame=history_start,
            steps=history_steps,
            origin_frame=target_frame,
        )
        if history is None:
            return None
        history_states, history_valid = history
        if not bool(history_valid[-1, 0]):
            return None

    actions, action_valid = _action_window(
        vehicles,
        slot_ids=slot_ids,
        start_frame=target_frame,
        steps=horizon_steps,
    )
    future = _state_window(
        vehicles,
        ego_id=ego_id,
        slot_ids=slot_ids,
        start_frame=target_frame + 1,
        steps=horizon_steps,
        origin_frame=target_frame,
    )
    if future is None:
        return None
    target_states_all, future_valid_all = future
    target_valid = action_valid & future_valid_all[:, 1:]
    actions[~target_valid] = 0.0
    target_states = target_states_all[:, 1:, :]
    target_states[~future_valid_all[:, 1:]] = 0.0
    ego_future_states = target_states_all[:, 0, :]

    flow_summary, flow_summary_valid = _flow_action_summary_from_actions(
        actions,
        target_valid,
        fps=float(fps),
    )
    primary_idx = primary_slot_index(row.get("peak_slot_name", "none"), slot_mask=slot_mask)
    return {
        "history_states": history_states,
        "history_valid": history_valid,
        "current_states": current[0][0],
        "current_valid": current[1][0],
        "target_actions": actions,
        "target_valid": target_valid,
        "target_states": target_states,
        "target_state_valid": future_valid_all[:, 1:],
        "ego_future_states": ego_future_states,
        "ego_future_valid": future_valid_all[:, 0],
        "slot_mask": slot_mask.astype(bool),
        "flow_action_summary": flow_summary,
        "flow_action_summary_valid": flow_summary_valid,
        "relation_features": build_relation_features_from_current(
            current[0][0],
            current[1][0],
            primary_slot_index=int(primary_idx),
        ),
        "mode_index": int(mode_index),
        "primary_slot_index": int(primary_idx),
        "split_index": int(segment_split_index),
        "is_evt_tail": int(bool(is_evt_tail)),
        "metadata": {
            "segment_id": str(row["segment_id"]),
            "recording_id": int(row["recording_id"]),
            "ego_id": ego_id,
            "anchor_frame": anchor_frame,
            "target_frame": target_frame,
            "offset": int(offset),
            "event_risk": float(row.get("event_risk", 0.0)),
            "mask_pattern": int(mask_pattern_from_slot_mask(slot_mask)[0]),
        },
    }


def _valid_roll_offset(offset: int, history_steps: int, horizon_steps: int, window_steps: int) -> bool:
    if int(offset) < int(history_steps):
        return False
    return int(offset) + int(horizon_steps) < int(window_steps)


def _seconds_list_to_offsets(
    values: Any,
    *,
    fps: float,
    history_steps: int,
    horizon_steps: int,
    window_steps: int,
) -> list[int]:
    out: list[int] = []
    for seconds in values:
        offset = int(round(float(seconds) * float(fps)))
        if _valid_roll_offset(offset, history_steps, horizon_steps, window_steps):
            out.append(offset)
    return sorted(set(out))


def _dense_roll_offsets(
    dataset_cfg: dict[str, Any],
    *,
    split_name: str | None,
    fps: float,
    history_steps: int,
    horizon_steps: int,
    window_steps: int,
) -> list[int]:
    prefix = f"{split_name}_" if split_name else ""
    stride_seconds = dataset_cfg.get(f"{prefix}roll_stride_seconds", dataset_cfg.get("roll_stride_seconds"))
    if stride_seconds is None:
        return []
    stride = max(int(round(float(stride_seconds) * float(fps))), 1)
    default_start = float(dataset_cfg.get("history_seconds", 1.0))
    start_seconds = float(dataset_cfg.get(f"{prefix}roll_start_seconds", dataset_cfg.get("roll_start_seconds", default_start)))
    max_end_offset = int(window_steps) - int(horizon_steps) - 1
    default_end_seconds = max_end_offset / max(float(fps), 1.0)
    end_seconds = float(dataset_cfg.get(f"{prefix}roll_end_seconds", dataset_cfg.get("roll_end_seconds", default_end_seconds)))
    start_offset = max(int(round(start_seconds * float(fps))), int(history_steps))
    end_offset = min(int(round(end_seconds * float(fps))), max_end_offset)
    if end_offset < start_offset:
        return []
    out = [
        int(offset)
        for offset in range(start_offset, end_offset + 1, stride)
        if _valid_roll_offset(offset, history_steps, horizon_steps, window_steps)
    ]
    return sorted(set(out))


def _roll_offsets(
    config: dict[str, Any],
    fps: float,
    history_steps: int,
    horizon_steps: int,
    window_steps: int,
    *,
    split_name: str | None = None,
) -> list[int]:
    dataset_cfg = dict(config.get("dataset", {}))
    dense = _dense_roll_offsets(
        dataset_cfg,
        split_name=split_name,
        fps=fps,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        window_steps=window_steps,
    )
    if dense:
        return dense

    prefix = f"{split_name}_" if split_name else ""
    raw_offsets = dataset_cfg.get(f"{prefix}roll_offsets_seconds", dataset_cfg.get("roll_offsets_seconds"))
    if raw_offsets is None:
        raw_offsets = [1.0, 3.0]
    return _seconds_list_to_offsets(
        raw_offsets,
        fps=fps,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        window_steps=window_steps,
    )


def _select_segment_rows(
    rows: pd.DataFrame,
    *,
    max_segments: int,
    tail_ids: set[str],
    seed: int,
) -> pd.DataFrame:
    if int(max_segments) <= 0 or len(rows) <= int(max_segments):
        return rows.copy()
    rows = rows.copy()
    rows["_wm_original_index"] = np.arange(len(rows), dtype=np.int64)
    tail_mask = rows["segment_id"].astype(str).isin(tail_ids)
    tail_rows = rows[tail_mask].sort_values("event_risk", ascending=False)
    natural_rows = rows[~tail_mask]
    tail_quota = min(len(tail_rows), max(1, int(round(int(max_segments) * 0.25))))
    natural_quota = int(max_segments) - int(tail_quota)
    selected_parts = []
    if tail_quota > 0 and len(tail_rows) > 0:
        selected_parts.append(tail_rows.iloc[:tail_quota])
    if natural_quota > 0 and len(natural_rows) > 0:
        if len(natural_rows) <= natural_quota:
            selected_parts.append(natural_rows)
        else:
            rng = np.random.default_rng(int(seed))
            # Draw across the full table so smoke datasets cover many recordings.
            lin = np.linspace(0, len(natural_rows) - 1, natural_quota, dtype=np.int64)
            jitter = rng.integers(-2, 3, size=natural_quota)
            take = np.clip(lin + jitter, 0, len(natural_rows) - 1)
            take = np.unique(take)
            while len(take) < natural_quota:
                extra = rng.choice(len(natural_rows), size=natural_quota - len(take), replace=False)
                take = np.unique(np.concatenate([take, extra]))
            selected_parts.append(natural_rows.iloc[take[:natural_quota]])
    selected = pd.concat(selected_parts, axis=0).sort_values("_wm_original_index")
    return selected.drop(columns=["_wm_original_index"]).reset_index(drop=True)


def _stack_samples(samples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    array_keys = (
        "history_states",
        "history_valid",
        "current_states",
        "current_valid",
        "target_actions",
        "target_valid",
        "target_states",
        "target_state_valid",
        "ego_future_states",
        "ego_future_valid",
        "slot_mask",
        "flow_action_summary",
        "flow_action_summary_valid",
        "relation_features",
    )
    arrays = {key: np.stack([sample[key] for sample in samples]) for key in array_keys}
    for key in ("mode_index", "primary_slot_index", "split_index", "is_evt_tail"):
        arrays[key] = np.asarray([sample[key] for sample in samples], dtype=np.int64)
    meta = [sample["metadata"] for sample in samples]
    arrays.update(
        {
            "segment_id": np.asarray([item["segment_id"] for item in meta], dtype="U64"),
            "recording_id": np.asarray([item["recording_id"] for item in meta], dtype=np.int64),
            "ego_id": np.asarray([item["ego_id"] for item in meta], dtype=np.int64),
            "anchor_frame": np.asarray([item["anchor_frame"] for item in meta], dtype=np.int64),
            "offset": np.asarray([item["offset"] for item in meta], dtype=np.int64),
            "event_risk": np.asarray([item["event_risk"] for item in meta], dtype=np.float32),
            "mask_pattern": np.asarray([item["mask_pattern"] for item in meta], dtype=np.int64),
        }
    )
    return arrays


def _fit_and_apply_normalization(arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    train = arrays["split_index"] == SPLIT_TO_INDEX["train"]
    history_feature_valid = np.repeat(arrays["history_valid"][..., None], len(AGENT_STATE_FEATURES), axis=-1)
    current_feature_valid = np.repeat(arrays["current_valid"][..., None], len(AGENT_STATE_FEATURES), axis=-1)
    state_values = np.concatenate(
        [
            arrays["history_states"][train].reshape(-1, len(AGENT_STATE_FEATURES)),
            arrays["current_states"][train].reshape(-1, len(AGENT_STATE_FEATURES)),
        ],
        axis=0,
    )
    state_valid = np.concatenate(
        [
            history_feature_valid[train].reshape(-1, len(AGENT_STATE_FEATURES)),
            current_feature_valid[train].reshape(-1, len(AGENT_STATE_FEATURES)),
        ],
        axis=0,
    )
    state_mean, state_std = finite_mean_std(state_values, state_valid)

    action_valid_feat = np.repeat(arrays["target_valid"][..., None], 2, axis=-1)
    action_mean, action_std = finite_mean_std(
        arrays["target_actions"][train],
        action_valid_feat[train],
    )
    flow_valid = arrays["flow_action_summary_valid"]
    flow_mean, flow_std = finite_mean_std(
        arrays["flow_action_summary"][train],
        flow_valid[train],
    )

    normalized = dict(arrays)
    normalized["history_states_normalized"] = normalize_with_mask(
        arrays["history_states"],
        history_feature_valid,
        state_mean,
        state_std,
    )
    normalized["current_states_normalized"] = normalize_with_mask(
        arrays["current_states"],
        current_feature_valid,
        state_mean,
        state_std,
    )
    normalized["target_actions_normalized"] = normalize_with_mask(
        arrays["target_actions"],
        action_valid_feat,
        action_mean,
        action_std,
    )
    normalized["flow_action_summary_normalized"] = normalize_with_mask(
        arrays["flow_action_summary"],
        arrays["flow_action_summary_valid"],
        flow_mean,
        flow_std,
    )
    relation_valid = np.repeat(arrays["current_valid"][:, 1:, None], len(RELATION_FEATURES), axis=-1)
    relation_mean, relation_std = finite_mean_std(
        arrays["relation_features"][train],
        relation_valid[train],
    )
    normalized["relation_features_normalized"] = normalize_with_mask(
        arrays["relation_features"],
        relation_valid,
        relation_mean,
        relation_std,
    )
    weights = np.ones(len(arrays["split_index"]), dtype=np.float32)
    risk_scale = float(np.percentile(arrays["event_risk"][train], 95)) if np.any(train) else 1.0
    risk_scale = max(risk_scale, 1.0e-6)
    risk_weight = 1.0 + 0.75 * np.clip(arrays["event_risk"] / risk_scale, 0.0, 2.0)
    weights *= risk_weight.astype(np.float32)
    weights[arrays["is_evt_tail"].astype(bool)] *= 2.0
    mask_counts = Counter(arrays["mask_pattern"][train].astype(np.int64).tolist())
    if mask_counts:
        mean_count = float(np.mean(list(mask_counts.values())))
        for pattern, count in mask_counts.items():
            mask = arrays["mask_pattern"] == int(pattern)
            weights[mask] *= float(np.sqrt(mean_count / max(float(count), 1.0)))
    if np.any(train):
        weights /= max(float(np.mean(weights[train])), 1.0e-6)
    normalized["sample_weight"] = np.clip(weights, 0.25, 8.0).astype(np.float32)

    norm_schema = {
        "state": {
            "mean": state_mean.astype(float).tolist(),
            "std": state_std.astype(float).tolist(),
        },
        "action": {
            "mean": action_mean.astype(float).tolist(),
            "std": action_std.astype(float).tolist(),
        },
        "flow_action_summary": {
            "mean": flow_mean.astype(float).tolist(),
            "std": flow_std.astype(float).tolist(),
        },
        "relation_features": {
            "mean": relation_mean.astype(float).tolist(),
            "std": relation_std.astype(float).tolist(),
            "names": list(RELATION_FEATURES),
        },
        "sample_weight_policy": "event_risk_x_evt_tail_x_rare_mask_normalized_on_train",
    }
    return normalized, norm_schema


def build_world_model_dataset(
    config: dict[str, Any],
    *,
    config_dir: str | Path,
    max_segments: int | None = None,
    rebuild: bool = False,
) -> dict[str, Any]:
    config_dir = Path(config_dir).resolve()
    output_dir = dataset_dir_from_config(config, config_dir)
    prepared_array_dir = prepared_dataset_array_dir(output_dir)
    schema_path = schema_json_path(output_dir)
    highd_cfg_path = _resolve_highd_evt_config(config, config_dir)
    highd_cfg = load_highd_config(highd_cfg_path)
    dataset_cfg = dict(config.get("dataset", {}))
    fps = float(config.get("sampling", {}).get("target_fps", highd_cfg.get("sampling", {}).get("target_fps", 25)))
    history_steps = int(round(float(dataset_cfg.get("history_seconds", 1.0)) * fps))
    horizon_steps = int(round(float(dataset_cfg.get("horizon_seconds", 1.0)) * fps))
    window_steps = int(round(float(dataset_cfg.get("segment_window_seconds", 6.0)) * fps))
    roll_offsets_by_split = {
        name: _roll_offsets(
            config,
            fps,
            history_steps,
            horizon_steps,
            window_steps,
            split_name=name,
        )
        for name in SPLIT_TO_INDEX
    }
    all_roll_offsets = sorted({offset for offsets in roll_offsets_by_split.values() for offset in offsets})
    if not all_roll_offsets:
        raise RuntimeError("No valid ROLL offsets remain after history/horizon/window constraints")
    if not rebuild and prepared_dataset_available(output_dir) and schema_path.exists():
        cached = load_json(schema_path)
        cached_offsets = [int(value) for value in cached.get("roll_offsets", [])]
        cached_by_split = {
            name: [int(value) for value in cached.get("roll_offsets_by_split", {}).get(name, cached_offsets)]
            for name in SPLIT_TO_INDEX
        }
        cache_matches = (
            int(cached.get("history_steps", -1)) == int(history_steps)
            and int(cached.get("horizon_steps", -1)) == int(horizon_steps)
            and abs(float(cached.get("fps", -1.0)) - float(fps)) < 1.0e-6
            and cached_offsets == [int(value) for value in all_roll_offsets]
            and cached_by_split == {
                name: [int(value) for value in offsets]
                for name, offsets in roll_offsets_by_split.items()
            }
        )
        if cache_matches:
            return cached
        logger.warning(
            "Existing world-model dataset schema does not match current config; rebuilding %s",
            output_dir,
        )

    raw_dir = _resolve_raw_dir(config, config_dir)
    natural_path = _resolve_natural_segments_csv(config, config_dir)
    if not natural_path.exists():
        raise FileNotFoundError(f"Missing natural segments CSV: {natural_path}")
    rows = pd.read_csv(natural_path)
    if rows.empty:
        raise RuntimeError(f"Natural segments CSV is empty: {natural_path}")

    tail_path = _resolve_tail_context_csv(config, config_dir)
    tail_ids: set[str] = set()
    if tail_path is not None and tail_path.exists():
        tail = pd.read_csv(tail_path, usecols=["segment_id"])
        tail_ids = set(tail["segment_id"].astype(str).tolist())

    cfg_max_segments = int(dataset_cfg.get("max_segments", 0) or 0)
    if max_segments is not None and int(max_segments) > 0:
        cfg_max_segments = int(max_segments)
    if cfg_max_segments > 0:
        rows = _select_segment_rows(
            rows,
            max_segments=cfg_max_segments,
            tail_ids=tail_ids,
            seed=int(config.get("split", {}).get("seed", 42)),
        )

    segment_split = split_indices_by_group(rows, dict(config.get("split", {})))
    split_by_segment = {
        str(segment_id): int(split)
        for segment_id, split in zip(rows["segment_id"].astype(str), segment_split)
    }

    samples: list[dict[str, Any]] = []
    reject: Counter[str] = Counter()
    logger.info(
        "Building world-model dataset from %d segments, roll_offsets_by_split=%s",
        len(rows),
        roll_offsets_by_split,
    )
    for recording_id, frame in rows.groupby("recording_id", sort=True):
        rec = prepare_recording(raw_dir, int(recording_id), highd_cfg)
        vehicles = _build_vehicle_cache(rec)
        logger.info("Recording %02d: %d segments", int(recording_id), len(frame))
        for _, row in frame.iterrows():
            segment_id = str(row["segment_id"])
            split_idx = split_by_segment[segment_id]
            split_name = INDEX_TO_SPLIT[int(split_idx)]
            is_tail = segment_id in tail_ids
            roll_offsets = roll_offsets_by_split[split_name]
            for mode_idx, offset in [(START_MODE_INDEX, 0), *[(ROLL_MODE_INDEX, off) for off in roll_offsets]]:
                try:
                    sample = _build_one_sample(
                        vehicles,
                        row,
                        mode_index=mode_idx,
                        offset=int(offset),
                        history_steps=history_steps,
                        horizon_steps=horizon_steps,
                        fps=fps,
                        is_evt_tail=is_tail,
                        segment_split_index=split_idx,
                    )
                except Exception as exc:  # noqa: BLE001 - keep reject audit.
                    reject[type(exc).__name__] += 1
                    logger.debug("Reject %s mode=%s offset=%s: %s", segment_id, mode_idx, offset, exc)
                    continue
                if sample is None:
                    reject["empty_sample"] += 1
                    continue
                samples.append(sample)

    if not samples:
        raise RuntimeError("No world-model samples were extracted")

    arrays = _stack_samples(samples)
    normalized, norm_schema = _fit_and_apply_normalization(arrays)

    schema = WorldModelSchema(
        fps=fps,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
    )
    schema_payload = {
        "dataset_cache": str(prepared_array_dir),
        "dataset_array_dir": str(prepared_array_dir),
        "dataset_format": "prepared_npy_dir_v2",
        "natural_segments_csv": str(natural_path),
        "tail_context_csv": str(tail_path) if tail_path is not None else "",
        "raw_dir": str(raw_dir),
        "num_samples": int(len(samples)),
        "num_segments_requested": int(len(rows)),
        "fps": float(fps),
        "history_steps": int(history_steps),
        "horizon_steps": int(horizon_steps),
        "roll_offsets": [int(value) for value in all_roll_offsets],
        "roll_offsets_by_split": {
            name: [int(value) for value in offsets]
            for name, offsets in roll_offsets_by_split.items()
        },
        "mode_names": ["START", "ROLL"],
        "agent_names": list(schema.agent_names),
        "slot_names": list(schema.slot_names),
        "state_features": list(schema.state_features),
        "action_features": list(schema.action_features),
        "flow_action_summary_features": list(schema.flow_action_summary_features),
        "relation_features": list(schema.relation_features),
        "split_index": SPLIT_TO_INDEX,
        "split_summary": {
            name: int(np.sum(arrays["split_index"] == idx))
            for name, idx in SPLIT_TO_INDEX.items()
        },
        "mode_summary": {
            "START": int(np.sum(arrays["mode_index"] == START_MODE_INDEX)),
            "ROLL": int(np.sum(arrays["mode_index"] == ROLL_MODE_INDEX)),
        },
        "evt_tail_samples": int(np.sum(arrays["is_evt_tail"])),
        "normalization": norm_schema,
        "reject_counts": {key: int(value) for key, value in sorted(reject.items())},
    }
    save_json(schema_payload, schema_path)
    logger.info("Wrote world-model schema: %s", schema_path)
    write_world_model_prepared_dataset(
        output_dir,
        source_arrays=normalized,
        rebuild=True,
    )
    return schema_payload


def _training_row_mask(split_index: np.ndarray) -> np.ndarray:
    split = np.asarray(split_index, dtype=np.int64)
    return split != SPLIT_TO_INDEX["test"]


def _array_dir_has_keys(path: Path, required_keys: tuple[str, ...]) -> bool:
    if not path.is_dir():
        return False
    return all((path / f"{key}.npy").exists() for key in required_keys)


def prepared_dataset_cache_path(output_dir: str | Path) -> Path | None:
    out_dir = Path(output_dir)
    array_dir = prepared_dataset_array_dir(out_dir)
    if _array_dir_has_keys(array_dir, PREPARED_DATASET_ARRAY_KEYS):
        return array_dir
    return None


def prepared_dataset_available(output_dir: str | Path) -> bool:
    return prepared_dataset_cache_path(output_dir) is not None


def _prepared_dataset_required_keys(source_keys: set[str]) -> tuple[str, ...]:
    missing = sorted(set(PREPARED_DATASET_ARRAY_KEYS) - source_keys)
    if missing:
        raise KeyError(f"Missing required prepared dataset arrays={missing}; rebuild the full world-model dataset")
    return PREPARED_DATASET_ARRAY_KEYS


def write_world_model_prepared_dataset(
    output_dir: str | Path,
    *,
    source_arrays: dict[str, np.ndarray],
    rebuild: bool = False,
) -> Path:
    """Write the unified prepared cache used by both training and evaluation.

    This cache is a deterministic projection of the full builder output.  It
    stores only fields needed by current training/evaluation code and avoids
    keeping a separate train-only file.  The physical format is one `.npy` per
    array so training can load selected arrays without opening a large NPZ
    container.
    """

    out_dir = Path(output_dir)
    ensure_dir(out_dir)
    cache_path = prepared_dataset_array_dir(out_dir)
    required_keys = PREPARED_DATASET_ARRAY_KEYS
    if not rebuild and _array_dir_has_keys(cache_path, required_keys):
        return cache_path

    source_keys = set(source_arrays.keys())
    required_keys = _prepared_dataset_required_keys(source_keys)
    tmp_path = out_dir / f".{cache_path.name}.tmp"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    ensure_dir(tmp_path)
    for key in required_keys:
        np.save(tmp_path / f"{key}.npy", source_arrays[key], allow_pickle=False)

    if cache_path.exists():
        shutil.rmtree(cache_path)
    tmp_path.rename(cache_path)
    logger.info(
        "Wrote prepared world-model dataset cache: %s keys=%s",
        cache_path,
        ",".join(required_keys),
    )
    return cache_path


def _normalize_current_states_from_raw(arrays: dict[str, np.ndarray], schema: dict[str, Any]) -> np.ndarray:
    valid = np.repeat(arrays["current_valid"][..., None], len(AGENT_STATE_FEATURES), axis=-1)
    norm = schema["normalization"]["state"]
    return normalize_with_mask(
        arrays["current_states"],
        valid,
        np.asarray(norm["mean"], dtype=np.float32),
        np.asarray(norm["std"], dtype=np.float32),
    )


def _normalize_target_actions_from_raw(arrays: dict[str, np.ndarray], schema: dict[str, Any]) -> np.ndarray:
    valid = np.repeat(arrays["target_valid"][..., None], 2, axis=-1)
    norm = schema["normalization"]["action"]
    return normalize_with_mask(
        arrays["target_actions"],
        valid,
        np.asarray(norm["mean"], dtype=np.float32),
        np.asarray(norm["std"], dtype=np.float32),
    )


def _ensure_model_arrays(
    arrays: dict[str, np.ndarray],
    schema: dict[str, Any],
    *,
    require_relation_features: bool = True,
) -> dict[str, np.ndarray]:
    if "current_states_normalized" not in arrays:
        arrays["current_states_normalized"] = _normalize_current_states_from_raw(arrays, schema)
    if "target_actions_normalized" not in arrays:
        arrays["target_actions_normalized"] = _normalize_target_actions_from_raw(arrays, schema)
    if require_relation_features and "relation_features_normalized" not in arrays:
        raise KeyError("Missing relation_features_normalized; rebuild the world-model dataset")
    return arrays


def _missing_prepared_cache_error(out_dir: Path) -> FileNotFoundError:
    array_dir = prepared_dataset_array_dir(out_dir)
    return FileNotFoundError(
        f"Missing prepared world-model dataset cache: {array_dir}. "
        "Restore the archived frozen CAT-K dataset before running a comparison."
    )


def _record_dataset_cache(schema: dict[str, Any], cache_path: Path) -> None:
    schema.setdefault("dataset_cache", str(cache_path))
    schema.setdefault("dataset_array_dir", str(cache_path))


def _load_prepared_array(cache_path: Path, key: str, *, mmap: bool = False) -> np.ndarray:
    if cache_path.is_dir():
        mmap_mode = "r" if mmap else None
        return np.load(cache_path / f"{key}.npy", allow_pickle=False, mmap_mode=mmap_mode)
    raise TypeError(f"Prepared array path is not an array directory: {cache_path}")


def load_world_model_prepared_dataset(
    output_dir: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any], Path]:
    out_dir = Path(output_dir)
    schema_path = schema_json_path(out_dir)
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing world-model schema: {schema_path}")
    schema = load_json(schema_path)
    cache_path = prepared_dataset_cache_path(out_dir)
    if cache_path is None:
        raise _missing_prepared_cache_error(out_dir)
    arrays = {
        key: _load_prepared_array(cache_path, key, mmap=True)
        for key in PREPARED_DATASET_ARRAY_KEYS
    }
    arrays = _ensure_model_arrays(arrays, schema, require_relation_features=True)
    _record_dataset_cache(schema, cache_path)
    return arrays, schema, cache_path


def _training_source_array_keys() -> tuple[str, ...]:
    keys = {"split_index", "mode_index", "is_evt_tail"}
    for key in TRAINING_ARRAY_KEYS:
        if key == "current_states_normalized":
            keys.update(("current_states", "current_valid"))
        elif key == "target_actions_normalized":
            keys.update(("target_actions", "target_valid"))
        else:
            keys.add(key)
    return tuple(sorted(keys))


def load_world_model_training_dataset(
    output_dir: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any], Path]:
    required_keys = TRAINING_ARRAY_KEYS
    out_dir = Path(output_dir)
    schema_path = schema_json_path(out_dir)
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing world-model schema: {schema_path}")
    schema = load_json(schema_path)
    cache_path = prepared_dataset_cache_path(out_dir)
    if cache_path is None:
        raise _missing_prepared_cache_error(out_dir)
    source_keys = _training_source_array_keys()
    split_index = _load_prepared_array(cache_path, "split_index", mmap=True)
    row_mask = _training_row_mask(split_index)
    n_rows = int(len(split_index))
    arrays = {"split_index": np.asarray(split_index[row_mask])}
    for key in source_keys:
        if key == "split_index":
            continue
        value = _load_prepared_array(cache_path, key, mmap=True)
        if hasattr(value, "shape") and len(value) == n_rows:
            arrays[key] = np.asarray(value[row_mask])
        else:
            arrays[key] = np.asarray(value)
    arrays = _ensure_model_arrays(
        arrays,
        schema,
        require_relation_features=True,
    )
    if "target_actions" not in required_keys:
        arrays.pop("target_actions", None)
    if "current_states" not in required_keys:
        arrays.pop("current_states", None)
    missing = [key for key in required_keys if key not in arrays]
    if missing:
        raise RuntimeError(f"Training dataset cache {cache_path} is missing keys={missing}")
    _record_dataset_cache(schema, cache_path)
    return arrays, schema, cache_path


def load_world_model_dataset(output_dir: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays, schema, _cache_path = load_world_model_prepared_dataset(output_dir)
    return arrays, schema

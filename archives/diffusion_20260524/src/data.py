"""Build sliding-window action diffusion datasets from highD events."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from process_highd.src.loader import HighDRecording, load_recording
    from process_highd.src.preprocess import filter_abnormal_tracks, normalize_driving_direction, resample_recording
except ModuleNotFoundError:
    from process_highD.src.loader import HighDRecording, load_recording
    from process_highD.src.preprocess import filter_abnormal_tracks, normalize_driving_direction, resample_recording

from .features import extract_context
from .normalization import apply_normalizers, fit_dataset_normalizers
from .scenario_frame import compute_ego_frame, world_to_ego_states
from .types import (
    FOLLOWING_ACCEL_ACTION_KEYS,
    FOLLOWING_ACTION_KEYS,
    FOLLOWING_JERK_ACTION_KEYS,
    FOLLOWING_RELATIVE_HISTORY_KEYS,
    EventType,
    NUM_ACTORS,
    NUM_STATE_FEATURES,
    STATE_FEATURES,
)
from .utils import save_json

logger = logging.getLogger(__name__)


SPLIT_TO_INDEX = {"train": 0, "val": 1, "test": 2}
INDEX_TO_SPLIT = {v: k for k, v in SPLIT_TO_INDEX.items()}


@dataclass(frozen=True)
class DatasetPaths:
    raw_dir: Path
    events_csv: Path
    output_dir: Path


def _event_value(event_type: EventType | str) -> str:
    return event_type.value if isinstance(event_type, EventType) else str(event_type)


def action_keys_for(event_type: EventType | str, action_representation: str = "acceleration") -> Tuple[str, ...]:
    if _event_value(event_type) == EventType.FOLLOWING.value:
        if str(action_representation).lower() == "jerk":
            return FOLLOWING_JERK_ACTION_KEYS
        if str(action_representation).lower() == "acceleration":
            return FOLLOWING_ACCEL_ACTION_KEYS
        return FOLLOWING_ACTION_KEYS
    raise ValueError(f"Unsupported event_type: {event_type}")


def prepare_recording(raw_dir: str | Path, recording_id: int, config: dict) -> HighDRecording:
    rec = load_recording(str(raw_dir), int(recording_id))
    rec = normalize_driving_direction(rec)
    rec = filter_abnormal_tracks(rec, config)
    target_fps = int(config.get("sampling", {}).get("target_fps", 25))
    rec = resample_recording(rec, target_fps)
    return rec


def _extract_vehicle_states(recording: HighDRecording, vehicle_id: int, frames: np.ndarray) -> Optional[np.ndarray]:
    try:
        track = recording.get_vehicle_track(int(vehicle_id))
    except KeyError:
        return None
    present = track.index.intersection(frames)
    if len(present) != len(frames):
        return None
    sub = track.loc[frames]
    if "_abnormal" in sub.columns and bool(sub["_abnormal"].any()):
        return None
    out = np.zeros((len(frames), NUM_STATE_FEATURES), dtype=np.float32)
    out[:, 0] = sub["x"].values
    out[:, 1] = sub["y"].values
    out[:, 2] = sub["xVelocity"].values
    out[:, 3] = sub["yVelocity"].values if "yVelocity" in sub.columns else 0.0
    out[:, 4] = sub["xAcceleration"].values
    out[:, 5] = sub["yAcceleration"].values if "yAcceleration" in sub.columns else 0.0
    return out


def _build_world_states(recording: HighDRecording, event_row: pd.Series, frames: np.ndarray) -> Optional[np.ndarray]:
    ego = _extract_vehicle_states(recording, int(event_row["ego_id"]), frames)
    adv = _extract_vehicle_states(recording, int(event_row["target_id"]), frames)
    if ego is None or adv is None:
        return None
    return np.stack([ego, adv], axis=1).astype(np.float32)


def _vehicle_length_from_meta(meta: pd.DataFrame, vehicle_id: int) -> float:
    """highD `width` is the longitudinal bounding-box size; `height` is lateral width."""
    return float(meta.loc[int(vehicle_id)]["width"])


def _savgol_smooth_1d(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(y) < 3:
        return y.astype(np.float32)
    w = int(window)
    if w % 2 == 0:
        w += 1
    w = min(w, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if w < 3:
        return y.astype(np.float32)
    p = min(max(int(polyorder), 0), w - 1)
    half = w // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    design = np.vander(x, p + 1, increasing=True)
    coeff = np.linalg.pinv(design)[0]
    padded = np.pad(y, (half, half), mode="edge")
    out = np.convolve(padded, coeff[::-1], mode="valid")
    return out.astype(np.float32)


def _smooth_velocity(values: np.ndarray, action_cfg: dict) -> np.ndarray:
    smoothing = action_cfg.get("smoothing", {})
    method = str(smoothing.get("method", "savgol")).lower()
    if method in {"none", "raw"}:
        return np.asarray(values, dtype=np.float32)
    if method != "savgol":
        raise ValueError(f"Unsupported action smoothing method: {method}")
    return _savgol_smooth_1d(
        np.asarray(values, dtype=np.float32),
        int(smoothing.get("window", 9)),
        int(smoothing.get("polyorder", 2)),
    )


def _following_actions(history_world_states: np.ndarray, future_world_states: np.ndarray, config: dict, dt: float) -> np.ndarray:
    action_cfg = config.get("action", {})
    source = str(action_cfg.get("source", "smoothed_velocity_diff")).lower()
    representation = str(action_cfg.get("representation", "acceleration")).lower()
    ax_min = float(action_cfg.get("ax_min", -8.0))
    ax_max = float(action_cfg.get("ax_max", 4.0))
    jerk_abs_max = float(action_cfg.get("jerk_abs_max", 12.0))
    if source == "raw_acceleration":
        ax = future_world_states[:, 1, 4].astype(np.float32)
    elif source == "smoothed_velocity_diff":
        lead_vx = np.concatenate([history_world_states[:, 1, 2], future_world_states[:, 1, 2]]).astype(np.float32)
        smooth_vx = _smooth_velocity(lead_vx, action_cfg)
        ax_all = np.diff(smooth_vx) / max(float(dt), 1e-6)
        ax = ax_all[len(history_world_states) - 1:len(history_world_states) - 1 + len(future_world_states)]
    else:
        raise ValueError(f"Unsupported action.source: {source}")
    ax = np.clip(ax, ax_min, ax_max).astype(np.float32)
    if representation == "acceleration":
        return ax.reshape(-1, 1)
    if representation == "jerk":
        if source == "smoothed_velocity_diff" and len(history_world_states) >= 2:
            lead_vx = np.concatenate([history_world_states[:, 1, 2], future_world_states[:, 1, 2]]).astype(np.float32)
            smooth_vx = _smooth_velocity(lead_vx, action_cfg)
            ax_all = np.diff(smooth_vx) / max(float(dt), 1e-6)
            prev_ax = float(ax_all[max(len(history_world_states) - 2, 0)])
        else:
            prev_ax = float(history_world_states[-1, 1, 4])
        jx = np.diff(np.concatenate([[prev_ax], ax])) / max(float(dt), 1e-6)
        return np.clip(jx, -jerk_abs_max, jerk_abs_max).astype(np.float32).reshape(-1, 1)
    raise ValueError(f"Unsupported action.representation: {representation}")


def _relative_history(
    history_local: np.ndarray,
    ego_length: float,
    adv_length: float,
) -> np.ndarray:
    ego = np.asarray(history_local[:, 0], dtype=np.float32)
    adv = np.asarray(history_local[:, 1], dtype=np.float32)
    gap = adv[:, 0] - ego[:, 0] - 0.5 * (ego_length + adv_length)
    lateral = adv[:, 1] - ego[:, 1]
    delta_v = ego[:, 2] - adv[:, 2]
    delta_a = ego[:, 4] - adv[:, 4]
    eps = 1e-6
    ttc_cap = 1000.0
    thw_cap = 200.0
    ttc = np.where(delta_v > eps, gap / np.maximum(delta_v, eps), ttc_cap)
    thw = gap / np.maximum(ego[:, 2], eps)
    rel = np.stack(
        [
            gap,
            lateral,
            delta_v,
            delta_a,
            np.clip(ttc, 0.0, ttc_cap),
            np.clip(thw, 0.0, thw_cap),
        ],
        axis=-1,
    )
    return rel.astype(np.float32)


def _stride_for_split(dataset_cfg: dict, split_idx: int) -> int:
    split = INDEX_TO_SPLIT.get(int(split_idx), "train")
    key = f"{split}_stride"
    return int(dataset_cfg.get(key, dataset_cfg.get("stride", 5)))


def _select_event_samples(samples: list[dict], limit: int) -> list[dict]:
    if limit <= 0 or len(samples) <= limit:
        return samples
    selected = np.linspace(0, len(samples) - 1, int(limit), dtype=np.int64)
    return [samples[int(i)] for i in selected]


def _resolve_paths(config: dict, config_dir: str | Path | None) -> DatasetPaths:
    base = Path(config_dir).resolve() if config_dir is not None else Path.cwd()
    paths = config.get("paths", {})
    output_dir = (base / paths.get("output_dir", "../../../data/diffusion/following")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return DatasetPaths(
        raw_dir=(base / paths.get("raw_dir", "")).resolve(),
        events_csv=(base / paths.get("events_csv", "")).resolve(),
        output_dir=output_dir,
    )


def _load_valid_events(paths: DatasetPaths, event_type: str, config: dict) -> pd.DataFrame:
    events = pd.read_csv(paths.events_csv)
    events = events[events["event_type"] == event_type].copy()
    if "is_valid" in events.columns:
        valid = events["is_valid"]
        if valid.dtype != bool:
            valid = valid.astype(str).str.lower().isin({"true", "1", "yes"})
        events = events[valid].copy()
    events = events.reset_index(drop=True)
    if events.empty:
        raise RuntimeError(f"No valid events found for event_type={event_type} in {paths.events_csv}")

    max_recordings = int(config.get("dataset", {}).get("max_recordings", 0))
    if max_recordings > 0:
        keep_rids = sorted(events["recording_id"].unique().tolist())[:max_recordings]
        events = events[events["recording_id"].isin(keep_rids)].reset_index(drop=True)
        logger.warning("dataset.max_recordings=%d: using recordings=%s", max_recordings, keep_rids)
    return events


def _split_by_recording(recording_ids: Iterable[int], cfg: dict) -> Tuple[Dict[int, int], Dict[str, object]]:
    split_cfg = cfg.get("splits", {})
    seed = int(split_cfg.get("random_seed", 42))
    train_r = float(split_cfg.get("train_ratio", 0.70))
    val_r = float(split_cfg.get("val_ratio", 0.15))
    test_r = float(split_cfg.get("test_ratio", 0.15))
    total = max(train_r + val_r + test_r, 1e-6)
    train_r, val_r = train_r / total, val_r / total
    ids = sorted({int(r) for r in recording_ids})
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    if n >= 3:
        n_train = max(1, int(round(train_r * n)))
        n_val = max(1, int(round(val_r * n)))
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - n_train - 1)
    else:
        n_train = max(1, n - 1)
        n_val = max(0, n - n_train)
    mapping: Dict[int, int] = {}
    for rid in ids[:n_train]:
        mapping[int(rid)] = SPLIT_TO_INDEX["train"]
    for rid in ids[n_train:n_train + n_val]:
        mapping[int(rid)] = SPLIT_TO_INDEX["val"]
    for rid in ids[n_train + n_val:]:
        mapping[int(rid)] = SPLIT_TO_INDEX["test"]
    split_meta = {
        "strategy": "recording",
        "random_seed": seed,
        "train_recording_ids": [int(r) for r in ids[:n_train]],
        "val_recording_ids": [int(r) for r in ids[n_train:n_train + n_val]],
        "test_recording_ids": [int(r) for r in ids[n_train + n_val:]],
    }
    return mapping, split_meta


def build_action_dataset(config: dict, *, config_dir: str | Path | None = None) -> dict:
    """Build ``dataset.npz`` for one event type.

    For car-following, each sample is:
    ``(history o_t, context features, relative history, lead action sequence)``.
    """
    event_type = str(config.get("event", {}).get("event_type", "following"))
    if event_type != EventType.FOLLOWING.value:
        raise NotImplementedError("This first training pass supports car-following only.")

    paths = _resolve_paths(config, config_dir)
    events = _load_valid_events(paths, event_type, config)

    sample_cfg = config.get("sampling", {})
    fps = float(sample_cfg.get("target_fps", 25))
    dt = 1.0 / max(fps, 1.0)
    history_steps = int(config.get("context", {}).get("history_steps", 12))
    horizon_steps = int(config.get("generation", {}).get("horizon_steps", 50))
    dataset_cfg = config.get("dataset", {})
    max_windows_per_event = int(dataset_cfg.get("max_windows_per_event", 0))
    min_gap = float(dataset_cfg.get("min_current_gap", 0.5))
    action_representation = str(config.get("action", {}).get("representation", "acceleration")).lower()

    rid_split, split_meta = _split_by_recording(events["recording_id"].tolist(), config)
    grouped = events.groupby("recording_id")
    arrays: Dict[str, list] = {
        "context_states": [],
        "future_states": [],
        "context_features": [],
        "relative_history": [],
        "actions": [],
        "split_index": [],
        "recording_id": [],
        "event_id": [],
        "anchor_frame": [],
        "ego_length": [],
        "adv_length": [],
    }
    context_keys: List[str] | None = None
    skipped = 0
    for rid, rows in grouped:
        recording = prepare_recording(paths.raw_dir, int(rid), config)
        meta = recording.tracks_meta
        for _, row in rows.iterrows():
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            split_idx = rid_split[int(rid)]
            stride = _stride_for_split(dataset_cfg, split_idx)
            candidate_t = list(range(start + history_steps - 1, end - horizon_steps + 1, max(stride, 1)))
            ego_len = _vehicle_length_from_meta(meta, int(row["ego_id"]))
            adv_len = _vehicle_length_from_meta(meta, int(row["target_id"]))
            event_samples: list[dict] = []
            for t in candidate_t:
                frames = np.arange(t - history_steps + 1, t + horizon_steps + 1, dtype=np.int64)
                states = _build_world_states(recording, row, frames)
                if states is None:
                    skipped += 1
                    continue
                history_world = states[:history_steps]
                future_world = states[history_steps:]
                ego_frame = compute_ego_frame(history_world[-1, 0])
                history_local = world_to_ego_states(history_world, ego_frame).astype(np.float32)
                future_local = world_to_ego_states(future_world, ego_frame).astype(np.float32)
                gap_now = history_local[-1, 1, 0] - history_local[-1, 0, 0] - 0.5 * (ego_len + adv_len)
                if gap_now < min_gap:
                    skipped += 1
                    continue
                actions = _following_actions(history_world, future_world, config, dt)
                if not np.all(np.isfinite(actions)):
                    skipped += 1
                    continue
                context_vec, keys = extract_context(history_local, ego_len, adv_len, dt)
                if context_keys is None:
                    context_keys = keys
                event_samples.append(
                    {
                        "context_states": history_local,
                        "future_states": future_local,
                        "context_features": context_vec,
                        "relative_history": _relative_history(history_local, ego_len, adv_len),
                        "actions": actions,
                        "split_index": split_idx,
                        "recording_id": int(rid),
                        "event_id": str(row["event_id"]),
                        "anchor_frame": int(t),
                        "ego_length": float(ego_len),
                        "adv_length": float(adv_len),
                    }
                )
            for sample in _select_event_samples(event_samples, max_windows_per_event):
                for key in arrays:
                    arrays[key].append(sample[key])

    if not arrays["actions"]:
        raise RuntimeError("No diffusion training samples were built. Check window sizes and raw data paths.")

    out_arrays = {
        "context_states": np.asarray(arrays["context_states"], dtype=np.float32),
        "future_states": np.asarray(arrays["future_states"], dtype=np.float32),
        "context_features": np.asarray(arrays["context_features"], dtype=np.float32),
        "relative_history": np.asarray(arrays["relative_history"], dtype=np.float32),
        "actions": np.asarray(arrays["actions"], dtype=np.float32),
        "split_index": np.asarray(arrays["split_index"], dtype=np.int8),
        "recording_id": np.asarray(arrays["recording_id"], dtype=np.int16),
        "event_id": np.asarray(arrays["event_id"], dtype=object),
        "anchor_frame": np.asarray(arrays["anchor_frame"], dtype=np.int64),
        "ego_length": np.asarray(arrays["ego_length"], dtype=np.float32),
        "adv_length": np.asarray(arrays["adv_length"], dtype=np.float32),
    }
    train_mask = out_arrays["split_index"] == SPLIT_TO_INDEX["train"]
    stats = fit_dataset_normalizers(
        out_arrays["context_states"],
        out_arrays["context_features"],
        out_arrays["actions"],
        train_mask,
        out_arrays["relative_history"],
    )
    norm_arrays = apply_normalizers(out_arrays, stats)

    np.savez_compressed(paths.output_dir / "dataset.npz", **out_arrays)
    np.savez_compressed(paths.output_dir / "dataset_normalized.npz", **norm_arrays)
    schema = {
        "event_type": event_type,
        "state_features": list(STATE_FEATURES),
        "future_state_features": list(STATE_FEATURES),
        "future_state_frame": "anchor_ego_local",
        "num_actors": NUM_ACTORS,
        "context_keys": context_keys or [],
        "relative_history_keys": list(FOLLOWING_RELATIVE_HISTORY_KEYS),
        "action_representation": action_representation,
        "action_keys": list(action_keys_for(event_type, action_representation)),
        "history_steps": history_steps,
        "horizon_steps": horizon_steps,
        "dt": dt,
        "skipped_windows": skipped,
        "num_samples": int(out_arrays["actions"].shape[0]),
        "split_counts": {
            name: int(np.sum(out_arrays["split_index"] == idx))
            for name, idx in SPLIT_TO_INDEX.items()
        },
    }
    save_json(schema, paths.output_dir / "feature_schema.json")
    save_json(stats, paths.output_dir / "normalization_stats.json")
    save_json(split_meta, paths.output_dir / "train_val_test_split.json")
    logger.info("Built %d samples at %s; skipped=%d", out_arrays["actions"].shape[0], paths.output_dir, skipped)
    return {"arrays": out_arrays, "schema": schema, "stats": stats, "output_dir": paths.output_dir}


def load_normalized_dataset(dataset_dir: str | Path) -> dict:
    path = Path(dataset_dir) / "dataset_normalized.npz"
    if not path.exists():
        path = Path(dataset_dir) / "dataset.npz"
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}

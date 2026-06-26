"""Build sliding-window action diffusion datasets from highD events."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from process_highD.src.loader import HighDRecording, load_recording
from process_highD.src.preprocess import (
    filter_abnormal_tracks,
    normalize_driving_direction,
    resample_recording,
)

from .features import extract_scenario_condition
from .normalization import apply_normalizers, fit_dataset_normalizers
from .scenario_frame import compute_ego_frame, world_to_ego_states
from .types import (
    CUTIN_ACCEL_ACTION_KEYS,
    FOLLOWING_ACCEL_ACTION_KEYS,
    FOLLOWING_JERK_ACTION_KEYS,
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


def sequence_config(config: dict) -> dict:
    """Return fixed action-sequence settings."""
    return config["sequence"]


def _optional_path(
    value: str | None,
    *,
    base: Path,
) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def _event_value(event_type: EventType | str) -> str:
    return event_type.value if isinstance(event_type, EventType) else str(event_type)


def split_indices(arrays: dict, split: str) -> np.ndarray:
    split_name = str(split).lower()
    if split_name in {"all", "full"}:
        return np.arange(int(arrays["split_index"].shape[0]), dtype=np.int64)
    if split_name not in SPLIT_TO_INDEX:
        valid = sorted([*SPLIT_TO_INDEX, "all"])
        raise KeyError(f"Unknown split={split!r}; expected one of {valid}")
    return np.where(arrays["split_index"] == SPLIT_TO_INDEX[split_name])[0]


def split_config(config: dict) -> dict:
    return config["split"]


def split_mode(config: dict) -> str:
    return str(split_config(config).get("mode", "train_val_test")).lower()


def _optional_int_field(row: pd.Series, key: str) -> int | None:
    value = row.get(key, None)
    if value is None or pd.isna(value):
        return None
    return int(value)


def action_keys_for(
    event_type: EventType | str,
    action_representation: str = "acceleration",
) -> Tuple[str, ...]:
    if _event_value(event_type) == EventType.FOLLOWING.value:
        if str(action_representation).lower() == "jerk":
            return FOLLOWING_JERK_ACTION_KEYS
        if str(action_representation).lower() == "acceleration":
            return FOLLOWING_ACCEL_ACTION_KEYS
        raise ValueError(
            "Unsupported following action_representation: "
            f"{action_representation}"
        )
    if _event_value(event_type) == EventType.CUT_IN.value:
        if str(action_representation).lower() in {
            "acceleration",
            "ax_ay",
        }:
            return CUTIN_ACCEL_ACTION_KEYS
        raise ValueError(
            "Unsupported cut-in action_representation: "
            f"{action_representation}"
        )
    raise ValueError(f"Unsupported event_type: {event_type}")


def prepare_recording(raw_dir: str | Path, recording_id: int, config: dict) -> HighDRecording:
    rec = load_recording(str(raw_dir), int(recording_id))
    rec = normalize_driving_direction(rec)
    rec = filter_abnormal_tracks(rec, config)
    target_fps = int(config["sampling"]["target_fps"])
    rec = resample_recording(rec, target_fps)
    return rec


def _extract_vehicle_states(
    recording: HighDRecording,
    vehicle_id: int,
    frames: np.ndarray,
) -> Optional[np.ndarray]:
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


def _build_world_states(
    recording: HighDRecording,
    event_row: pd.Series,
    frames: np.ndarray,
) -> Optional[np.ndarray]:
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
    smoothing = action_cfg["smoothing"]
    method = str(smoothing["method"]).lower()
    if method in {"none", "raw"}:
        return np.asarray(values, dtype=np.float32)
    if method != "savgol":
        raise ValueError(f"Unsupported action smoothing method: {method}")
    return _savgol_smooth_1d(
        np.asarray(values, dtype=np.float32),
        int(smoothing["window"]),
        int(smoothing["polyorder"]),
    )


def _following_actions(
    history_world_states: np.ndarray,
    future_world_states: np.ndarray,
    config: dict,
    dt: float,
) -> np.ndarray:
    action_cfg = config["action"]
    source = str(action_cfg["source"]).lower()
    representation = str(action_cfg["representation"]).lower()
    ax_min = float(action_cfg["ax_min"])
    ax_max = float(action_cfg["ax_max"])
    jerk_abs_max = float(action_cfg["jerk_abs_max"])
    if source == "raw_acceleration":
        ax = future_world_states[:, 1, 4].astype(np.float32)
    elif source == "smoothed_velocity_diff":
        lead_vx = np.concatenate(
            [history_world_states[:, 1, 2], future_world_states[:, 1, 2]]
        ).astype(np.float32)
        smooth_vx = _smooth_velocity(lead_vx, action_cfg)
        ax_all = np.diff(smooth_vx) / max(float(dt), 1e-6)
        start = len(history_world_states) - 1
        stop = start + len(future_world_states)
        ax = ax_all[start:stop]
    else:
        raise ValueError(f"Unsupported action.source: {source}")
    ax = np.clip(ax, ax_min, ax_max).astype(np.float32)
    if representation == "acceleration":
        return ax.reshape(-1, 1)
    if representation == "jerk":
        if source == "smoothed_velocity_diff" and len(history_world_states) >= 2:
            lead_vx = np.concatenate(
                [history_world_states[:, 1, 2], future_world_states[:, 1, 2]]
            ).astype(np.float32)
            smooth_vx = _smooth_velocity(lead_vx, action_cfg)
            ax_all = np.diff(smooth_vx) / max(float(dt), 1e-6)
            prev_ax = float(ax_all[max(len(history_world_states) - 2, 0)])
        else:
            prev_ax = float(history_world_states[-1, 1, 4])
        jx = np.diff(np.concatenate([[prev_ax], ax])) / max(float(dt), 1e-6)
        return np.clip(jx, -jerk_abs_max, jerk_abs_max).astype(np.float32).reshape(-1, 1)
    raise ValueError(f"Unsupported action.representation: {representation}")



def _stride_for_split(dataset_cfg: dict, split_idx: int) -> int:
    split = INDEX_TO_SPLIT[int(split_idx)]
    key = f"{split}_stride"
    return int(dataset_cfg.get(key, dataset_cfg.get("stride", 5)))


def _select_event_samples(samples: list[dict], limit: int) -> list[dict]:
    if limit <= 0 or len(samples) <= limit:
        return samples
    selected = np.linspace(0, len(samples) - 1, int(limit), dtype=np.int64)
    return [samples[int(i)] for i in selected]



def _cutin_acceleration_targets(
    future_local: np.ndarray,
    config: dict,
) -> np.ndarray:
    action_cfg = config["action"]
    target = future_local[:, 1].astype(np.float32)
    ax = np.clip(
        target[:, 4],
        float(action_cfg.get("ax_min", -8.0)),
        float(action_cfg.get("ax_max", 4.0)),
    )
    ay = np.clip(
        target[:, 5],
        -float(action_cfg.get("ay_abs_max", 4.0)),
        float(action_cfg.get("ay_abs_max", 4.0)),
    )
    return np.stack([ax, ay], axis=-1).astype(np.float32)



def _resolve_paths(config: dict, config_dir: str | Path | None) -> DatasetPaths:
    base = Path(config_dir).resolve() if config_dir is not None else Path.cwd()
    paths = config.get("paths", {})
    missing = [key for key in ("raw_dir", "events_csv", "output_dir") if key not in paths]
    if missing:
        raise KeyError(f"Config paths is missing required keys: {missing}")
    output_dir = (base / paths["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return DatasetPaths(
        raw_dir=(base / paths["raw_dir"]).resolve(),
        events_csv=(base / paths["events_csv"]).resolve(),
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
    dataset_cfg = config.get("dataset", {})
    if event_type == EventType.CUT_IN.value and bool(
        dataset_cfg.get("require_scored_semantic_events", True)
    ):
        score_path = Path(
            str(
                dataset_cfg.get(
                    "scored_events_csv",
                    paths.events_csv.parent / "cutin_event_scores.csv",
                )
            )
        )
        if not score_path.is_absolute():
            score_path = (paths.events_csv.parent / score_path).resolve()
        if not score_path.exists():
            raise FileNotFoundError(
                "Cut-in dataset requires scored semantic events, but "
                f"{score_path} does not exist. Run "
                "process_highD/scripts/extract_highd_events.py first, or set "
                "dataset.require_scored_semantic_events=false."
            )
        scores = pd.read_csv(score_path)
        required = {"event_id", "is_cutin"}
        missing = sorted(required - set(scores.columns))
        if missing:
            raise KeyError(f"{score_path} is missing required columns: {missing}")
        semantic_ids = set(
            scores.loc[scores["is_cutin"].astype(float) >= 0.5, "event_id"].astype(str)
        )
        before = len(events)
        events = events[events["event_id"].astype(str).isin(semantic_ids)].copy()
        logger.info(
            "Restricted cut-in dataset to scored semantic events: %d -> %d",
            before,
            len(events),
        )
    events = events.reset_index(drop=True)
    if events.empty:
        raise RuntimeError(
            f"No valid events found for event_type={event_type} "
            f"in {paths.events_csv}"
        )

    max_recordings = int(config.get("dataset", {}).get("max_recordings", 0))
    if max_recordings > 0:
        keep_rids = sorted(events["recording_id"].unique().tolist())[:max_recordings]
        events = events[events["recording_id"].isin(keep_rids)].reset_index(drop=True)
        logger.warning("dataset.max_recordings=%d: using recordings=%s", max_recordings, keep_rids)
    return events



def _following_segment_cache_path(
    config: dict,
    *,
    config_dir: str | Path | None,
    events_csv: Path,
) -> Path:
    base = Path(config_dir).resolve() if config_dir is not None else Path.cwd()
    paths_cfg = config.get("paths", {})
    configured = _optional_path(paths_cfg.get("following_segment_cache"), base=base)
    if configured is not None:
        return configured
    return events_csv.parent / "following_event_segments.npz"


def _load_following_segment_cache(
    path: Path,
    *,
    target_fps: float,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            "Following segment cache not found: "
            f"{path}. Run process_highD/scripts/extract_highd_events.py first."
        )
    with np.load(path, allow_pickle=True) as archive:
        files = set(archive.files)
        required = {
            "event_id",
            "offset",
            "length",
            "frames",
            "target_fps",
            "world_states",
            "ego_length",
            "adv_length",
        }
        missing = sorted(required - files)
        if missing:
            raise KeyError(
                f"Following segment cache {path} is missing required arrays: {missing}"
            )
        data = {key: archive[key] for key in required}
    states = data["world_states"]
    if states.ndim != 3 or states.shape[1:] != (NUM_ACTORS, NUM_STATE_FEATURES):
        raise ValueError(
            "Following segment cache world_states shape mismatch "
            f"{tuple(states.shape)} in {path}; expected [N, {NUM_ACTORS}, "
            f"{NUM_STATE_FEATURES}]"
        )
    cached_fps = float(data["target_fps"].item())
    if abs(cached_fps - float(target_fps)) > 1.0e-6:
        raise ValueError(
            "Following segment cache target_fps="
            f"{cached_fps:.6g} does not match dataset target_fps={target_fps:.6g}: "
            f"{path}"
        )
    event_ids = [str(value) for value in data["event_id"]]
    index = {event_id: idx for idx, event_id in enumerate(event_ids)}
    logger.info(
        "Using following segment cache: %s (%d events)",
        path,
        len(index),
    )
    return {"path": path, "data": data, "index": index}



def _split_by_recording(
    recording_ids: Iterable[int],
    cfg: dict,
) -> Tuple[Dict[int, int], Dict[str, object]]:
    split_cfg = split_config(cfg)
    mode = split_mode(cfg)
    seed = int(split_cfg.get("random_seed", 42))
    ids = sorted({int(r) for r in recording_ids})
    if mode != "train_val_test":
        raise ValueError(f"split.mode must be 'train_val_test'; got {mode!r}")
    train_r = float(split_cfg["train_ratio"])
    val_r = float(split_cfg["val_ratio"])
    test_r = float(split_cfg["test_ratio"])
    total = max(train_r + val_r + test_r, 1e-6)
    train_r, val_r = train_r / total, val_r / total
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
        "mode": mode,
        "random_seed": seed,
        "train_recording_ids": [int(r) for r in ids[:n_train]],
        "val_recording_ids": [int(r) for r in ids[n_train:n_train + n_val]],
        "test_recording_ids": [int(r) for r in ids[n_train + n_val:]],
    }
    return mapping, split_meta


def _anchor_candidate_items(
    row: pd.Series,
    *,
    event_type: str,
    horizon_steps: int,
    stride: int,
    cutin_pre_cross_steps: int | list[int] | tuple[int, ...] = 25,
    cutin_min_post_cross_steps: int = 0,
    cutin_require_completion: bool = True,
) -> list[tuple[str, int]]:
    start = int(row["start_frame"])
    end = int(row["end_frame"])
    if event_type == EventType.CUT_IN.value:
        anchors = _cutin_completed_anchors(
            row,
            horizon_steps=horizon_steps,
            pre_cross_steps=cutin_pre_cross_steps,
            min_post_cross_steps=cutin_min_post_cross_steps,
            require_completion=cutin_require_completion,
        )
        return [
            ("completed_cross_phase_jittered", int(anchor))
            for anchor in anchors
        ]
    hi = end - int(horizon_steps)
    if hi < start:
        return []
    out = [
        ("sliding", int(t))
        for t in range(start, hi + 1, max(int(stride), 1))
    ]
    if out and out[-1][1] != hi:
        out.append(("sliding", int(hi)))
    return out


def _normalize_cutin_pre_cross_steps(
    value: int | list[int] | tuple[int, ...],
) -> tuple[int, ...]:
    raw = value if isinstance(value, (list, tuple)) else [value]
    offsets = tuple(sorted({int(item) for item in raw}))
    if not offsets or offsets[0] < 0:
        raise ValueError("cut-in pre-cross steps must contain non-negative integers")
    return offsets


def _cutin_completed_anchors(
    row: pd.Series,
    *,
    horizon_steps: int,
    pre_cross_steps: int | list[int] | tuple[int, ...],
    min_post_cross_steps: int = 0,
    require_completion: bool = True,
) -> list[int]:
    """Choose fixed-length anchors that contain crossing, completion and post-cross time."""
    event_start = int(row["start_frame"])
    event_end = int(row["end_frame"])
    cross = _optional_int_field(row, "cross_frame")
    cutin_start = _optional_int_field(row, "cutin_start_frame")
    cutin_end = _optional_int_field(row, "cutin_end_frame")
    if cross is None or cutin_start is None or cutin_end is None:
        return []
    horizon = int(horizon_steps)
    offsets = _normalize_cutin_pre_cross_steps(pre_cross_steps)
    if horizon <= 0:
        return []
    if int(cutin_end) < int(cutin_start):
        return []
    required_end = int(cross) + int(min_post_cross_steps)
    if bool(require_completion):
        required_end = max(required_end, int(cutin_end))
    anchors: list[int] = []
    for offset in offsets:
        anchor = int(cross) - int(offset)
        if anchor < event_start or anchor + horizon > event_end:
            continue
        if anchor > int(cross) or anchor + horizon < required_end:
            continue
        anchors.append(int(anchor))
    return sorted(set(anchors))


def _states_for_anchor_from_following_cache(
    cache: dict[str, Any],
    cached_event_idx: int,
    frames: np.ndarray,
) -> np.ndarray | None:
    cache_data = cache["data"]
    offset = int(cache_data["offset"][cached_event_idx])
    length = int(cache_data["length"][cached_event_idx])
    segment_frames = cache_data["frames"][offset:offset + length]
    pos0 = int(np.searchsorted(segment_frames, int(frames[0])))
    pos1 = pos0 + int(len(frames))
    if (
        0 <= pos0
        and pos1 <= length
        and np.array_equal(segment_frames[pos0:pos1], frames)
    ):
        return cache_data["world_states"][offset + pos0:offset + pos1].astype(np.float32)
    return None


def build_action_dataset(config: dict, *, config_dir: str | Path | None = None) -> dict:
    """Build an anchor-frame scenario-condition action diffusion dataset."""
    event_type = str(config.get("event", {}).get("event_type", "following"))
    if event_type not in {EventType.FOLLOWING.value, EventType.CUT_IN.value}:
        raise NotImplementedError(f"Unsupported event_type={event_type}")

    paths = _resolve_paths(config, config_dir)
    events = _load_valid_events(paths, event_type, config)

    sample_cfg = config["sampling"]
    fps = float(sample_cfg["target_fps"])
    dt = 1.0 / max(fps, 1.0)
    horizon_steps = int(sequence_config(config).get("horizon_steps", 125))
    dataset_cfg = config.get("dataset", {})
    max_windows_per_event = int(dataset_cfg.get("max_windows_per_event", 0))
    min_gap = float(dataset_cfg.get("min_current_gap", 0.5))
    action_representation = str(config["action"]["representation"]).lower()
    cutin_pre_cross_steps = _normalize_cutin_pre_cross_steps(
        dataset_cfg.get(
            "cutin_pre_cross_steps",
            config.get("cutin", {}).get("context_pre_cross_steps", 25),
        )
    )
    cutin_cfg = config.get("cutin", {})
    if "cutin_min_post_cross_seconds" in dataset_cfg:
        cutin_min_post_cross_steps = int(
            np.ceil(float(dataset_cfg["cutin_min_post_cross_seconds"]) * fps)
        )
    elif "min_post_cutin_duration_seconds" in cutin_cfg:
        cutin_min_post_cross_steps = int(
            np.ceil(float(cutin_cfg["min_post_cutin_duration_seconds"]) * fps)
        )
    else:
        cutin_min_post_cross_steps = int(
            dataset_cfg.get(
                "cutin_min_post_cross_steps",
                cutin_cfg.get("min_post_cutin_duration_steps", 0),
            )
        )
    cutin_require_completion = bool(
        dataset_cfg.get("cutin_require_completion_in_window", True)
    )

    following_segment_cache = None
    following_segment_cache_path: Path | None = None
    if event_type == EventType.FOLLOWING.value:
        following_segment_cache_path = _following_segment_cache_path(
            config,
            config_dir=config_dir,
            events_csv=paths.events_csv,
        )
        following_segment_cache = _load_following_segment_cache(
            following_segment_cache_path,
            target_fps=fps,
        )

    rid_split, split_meta = _split_by_recording(events["recording_id"].tolist(), config)
    grouped = events.groupby("recording_id")
    arrays: Dict[str, list] = {
        "scenario_conditions": [],
        "initial_states": [],
        "future_states": [],
        "actions": [],
        "split_index": [],
        "recording_id": [],
        "event_id": [],
        "anchor_frame": [],
        "ego_length": [],
        "adv_length": [],
    }
    condition_keys: List[str] | None = None
    skipped = 0
    skipped_insufficient_future = 0
    skipped_invalid_gap = 0

    for rid, rows in grouped:
        recording: HighDRecording | None = None
        meta: pd.DataFrame | None = None
        for _, row in rows.iterrows():
            split_idx = rid_split[int(rid)]
            stride = _stride_for_split(dataset_cfg, split_idx)
            candidates = _anchor_candidate_items(
                row,
                event_type=event_type,
                horizon_steps=horizon_steps,
                stride=stride,
                cutin_pre_cross_steps=cutin_pre_cross_steps,
                cutin_min_post_cross_steps=cutin_min_post_cross_steps,
                cutin_require_completion=cutin_require_completion,
            )
            if not candidates:
                skipped_insufficient_future += 1
                continue

            cached_following_event_idx: int | None = None
            if event_type == EventType.FOLLOWING.value:
                cached_following_event_idx = following_segment_cache["index"].get(
                    str(row["event_id"])
                )
                if cached_following_event_idx is None:
                    raise KeyError(
                        "Following event is missing from segment cache: "
                        f"{row['event_id']} in {following_segment_cache_path}"
                    )

            event_samples: list[dict] = []
            for _phase_label, t in candidates:
                frames = np.arange(t, t + horizon_steps + 1, dtype=np.int64)
                ego_len: float
                adv_len: float
                if event_type == EventType.FOLLOWING.value:
                    states = _states_for_anchor_from_following_cache(
                        following_segment_cache,
                        cached_following_event_idx,
                        frames,
                    )
                    if states is None:
                        raise ValueError(
                            "Following segment cache does not contain the requested "
                            f"frame window for event {row['event_id']} at anchor {t}"
                        )
                    cache_data = following_segment_cache["data"]
                    ego_len = float(cache_data["ego_length"][cached_following_event_idx])
                    adv_len = float(cache_data["adv_length"][cached_following_event_idx])
                else:
                    if recording is None or meta is None:
                        recording = prepare_recording(paths.raw_dir, int(rid), config)
                        meta = recording.tracks_meta
                    ego_len = _vehicle_length_from_meta(meta, int(row["ego_id"]))
                    adv_len = _vehicle_length_from_meta(meta, int(row["target_id"]))
                    states = _build_world_states(recording, row, frames)
                    if states is None:
                        skipped += 1
                        continue

                initial_world = states[:1]
                future_world = states[1:]
                ego_frame = compute_ego_frame(initial_world[0, 0])
                local = world_to_ego_states(states, ego_frame).astype(np.float32)
                initial_states = local[0]
                future_local = local[1:]
                gap_now = (
                    initial_states[1, 0]
                    - initial_states[0, 0]
                    - 0.5 * (ego_len + adv_len)
                )
                enforce_initial_gap = event_type != EventType.CUT_IN.value or bool(
                    dataset_cfg.get("enforce_cutin_initial_gap", False)
                )
                if enforce_initial_gap and gap_now < min_gap:
                    skipped_invalid_gap += 1
                    continue

                if event_type == EventType.CUT_IN.value:
                    actions = _cutin_acceleration_targets(
                        future_local,
                        config,
                    )
                else:
                    actions = _following_actions(
                        initial_world,
                        future_world,
                        config,
                        dt,
                    )
                if not np.all(np.isfinite(actions)):
                    skipped += 1
                    continue
                if event_type == EventType.CUT_IN.value:
                    cross_frame = _optional_int_field(row, "cross_frame")
                    cutin_start_frame = _optional_int_field(row, "cutin_start_frame")
                    cutin_end_frame = _optional_int_field(row, "cutin_end_frame")
                    if (
                        cross_frame is None
                        or cutin_start_frame is None
                        or cutin_end_frame is None
                    ):
                        skipped += 1
                        continue
                    metadata = {
                        "anchor_frame": int(t),
                        "cross_frame": cross_frame,
                        "cutin_start_frame": cutin_start_frame,
                        "cutin_end_frame": cutin_end_frame,
                    }
                else:
                    metadata = {"anchor_frame": int(t)}
                scenario_condition, keys = extract_scenario_condition(
                    initial_states,
                    future_local,
                    ego_len,
                    adv_len,
                    event_type=event_type,
                    dt=dt,
                    metadata=metadata,
                )
                if not np.all(np.isfinite(scenario_condition)):
                    skipped += 1
                    continue
                if condition_keys is None:
                    condition_keys = keys
                event_samples.append(
                    {
                        "scenario_conditions": scenario_condition,
                        "initial_states": initial_states,
                        "future_states": future_local,
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
        raise RuntimeError(
            "No anchor-frame scenario-condition diffusion training samples were built. "
            "Check horizon_steps, event lengths and raw data paths."
        )

    out_arrays = {
        "scenario_conditions": np.asarray(arrays["scenario_conditions"], dtype=np.float32),
        "initial_states": np.asarray(arrays["initial_states"], dtype=np.float32),
        "future_states": np.asarray(arrays["future_states"], dtype=np.float32),
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
        out_arrays["scenario_conditions"],
        out_arrays["actions"],
        train_mask,
    )
    norm_arrays = apply_normalizers(out_arrays, stats)
    train_arrays = {
        key: norm_arrays[key]
        for key in ("scenario_conditions", "actions", "split_index")
    }

    np.savez_compressed(paths.output_dir / "dataset.npz", **out_arrays)
    np.savez_compressed(paths.output_dir / "dataset_normalized.npz", **train_arrays)
    schema = {
        "event_type": event_type,
        "conditioning_mode": "anchor_scenario",
        "model_input_keys": ["scenario_conditions"],
        "condition_keys": condition_keys or [],
        "trajectory_initial_state_features": list(STATE_FEATURES),
        "initial_state_num_actors": NUM_ACTORS,
        "future_state_features": list(STATE_FEATURES),
        "future_state_frame": "anchor_ego_local",
        "action_representation": action_representation,
        "action_keys": list(action_keys_for(event_type, action_representation)),
        "generation_target": "action",
        "following_segment_cache_path": str(following_segment_cache_path or "")
        if event_type == EventType.FOLLOWING.value
        else "",
        "following_segment_cache_used": bool(following_segment_cache is not None)
        if event_type == EventType.FOLLOWING.value
        else False,
        "cutin_anchor_sampling": (
            "completed_cross_phase_jittered_pre_"
            + "_".join(str(step) for step in cutin_pre_cross_steps)
            + "_steps"
        )
        if event_type == EventType.CUT_IN.value
        else "",
        "cutin_min_post_cross_steps": int(cutin_min_post_cross_steps)
        if event_type == EventType.CUT_IN.value
        else 0,
        "cutin_min_post_cross_seconds": float(
            cutin_min_post_cross_steps * dt
        )
        if event_type == EventType.CUT_IN.value
        else 0.0,
        "cutin_require_completion_in_window": bool(cutin_require_completion)
        if event_type == EventType.CUT_IN.value
        else False,
        "split_mode": split_mode(config),
        "split_strategy": split_meta["strategy"],
        "horizon_steps": horizon_steps,
        "dt": dt,
        "skipped_windows": int(skipped + skipped_insufficient_future + skipped_invalid_gap),
        "skipped_invalid_windows": int(skipped),
        "skipped_insufficient_future": int(skipped_insufficient_future),
        "skipped_invalid_gap": int(skipped_invalid_gap),
        "num_samples": int(out_arrays["actions"].shape[0]),
        "split_counts": {
            name: int(np.sum(out_arrays["split_index"] == idx))
            for name, idx in SPLIT_TO_INDEX.items()
        },
    }
    save_json(schema, paths.output_dir / "feature_schema.json")
    save_json(stats, paths.output_dir / "normalization_stats.json")
    save_json(split_meta, paths.output_dir / "train_val_test_split.json")
    logger.info(
        "Built %d anchor-frame samples at %s; skipped=%d",
        out_arrays["actions"].shape[0],
        paths.output_dir,
        schema["skipped_windows"],
    )
    return {
        "arrays": out_arrays,
        "schema": schema,
        "stats": stats,
        "output_dir": paths.output_dir,
    }



def load_normalized_dataset(dataset_dir: str | Path) -> dict:
    path = Path(dataset_dir) / "dataset_normalized.npz"
    if not path.exists():
        raise FileNotFoundError(f"Normalized diffusion dataset not found: {path}")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}

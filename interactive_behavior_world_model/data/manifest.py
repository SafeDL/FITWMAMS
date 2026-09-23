"""Build the immutable benchmark manifest without modifying legacy caches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

SLOT_NAMES = (
    "same_front",
    "same_rear",
    "left_front",
    "left_rear",
    "right_front",
    "right_rear",
)
SPLIT_NAMES = np.asarray(("train", "validation", "test"))


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path, base: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (base / value).resolve()


def load_benchmark_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text())
    for key, value in tuple(config["paths"].items()):
        config["paths"][key] = str(_resolve(value, config_path.parent))
    return config, config_path


@dataclass(frozen=True)
class CausalWindow:
    history_start_frame: int
    decision_frame: int
    future_end_frame: int
    shift_from_original_anchor: int


def choose_causal_window(
    anchor_frame: int,
    initial_frames: np.ndarray,
    final_frames: np.ndarray,
    *,
    history_frames: int = 25,
    horizon_frames: int = 149,
) -> CausalWindow | None:
    """Choose the nearest complete real prefix/future around a legacy window.

    The start may move at most ``history_frames - 1`` frames earlier than the
    legacy anchor.  This preserves the legacy six-second interval while using
    its first 25 states as actual history whenever possible.
    """
    anchor = int(anchor_frame)
    total_points = int(history_frames) + int(horizon_frames)
    lower = max(int(np.max(initial_frames)), anchor - history_frames + 1)
    upper = min(int(np.min(final_frames)) - total_points + 1, anchor)
    if lower > upper:
        return None
    start = upper
    decision = start + history_frames - 1
    return CausalWindow(start, decision, start + total_points - 1, start - anchor)


def _source_rows_in_cache_order(
    source: pd.DataFrame, sequence_ids: np.ndarray
) -> pd.DataFrame:
    if source["segment_id"].astype(str).duplicated().any():
        raise ValueError("natural segment IDs are not unique")
    indexed = source.assign(segment_id=source["segment_id"].astype(str)).set_index(
        "segment_id"
    )
    missing = set(sequence_ids.astype(str)) - set(indexed.index)
    if missing:
        raise ValueError(
            f"{len(missing)} canonical sequence IDs are absent from source CSV"
        )
    return indexed.loc[sequence_ids.astype(str)].reset_index()


def _recording_metadata(raw_dir: Path, recording_id: int) -> pd.DataFrame:
    path = raw_dir / f"{int(recording_id):02d}_tracksMeta.csv"
    table = pd.read_csv(path).set_index("id")
    required = {"initialFrame", "finalFrame", "width", "height", "class"}
    if not required.issubset(table.columns):
        raise ValueError(f"{path} lacks {sorted(required - set(table.columns))}")
    return table


def build_dataset_manifest(config_path: str | Path) -> dict[str, Any]:
    config, loaded_from = load_benchmark_config(config_path)
    paths = config["paths"]
    cache = Path(paths["canonical_cache"])
    output = Path(paths["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    legacy_manifest_path = cache / "manifest.json"
    legacy_manifest = json.loads(legacy_manifest_path.read_text())
    sequence_ids = np.load(cache / "sequence_id.npy", mmap_mode="r", allow_pickle=False)
    split_index = np.load(cache / "split_index.npy", mmap_mode="r", allow_pickle=False)
    legacy_valid = np.load(cache / "agent_valid.npy", mmap_mode="r", allow_pickle=False)
    source_path = Path(paths["natural_segments_csv"])
    source = _source_rows_in_cache_order(
        pd.read_csv(source_path), np.asarray(sequence_ids)
    )
    raw_dir = Path(paths["raw_highd_dir"])

    count = len(sequence_ids)
    agent_ids = np.full((count, 7), -1, np.int64)
    lengths = np.zeros((count, 7), np.float32)
    widths = np.zeros((count, 7), np.float32)
    vehicle_class = np.full((count, 7), "missing", dtype="U8")
    history_start = np.full(count, -1, np.int64)
    decision_frame = np.full(count, -1, np.int64)
    future_end = np.full(count, -1, np.int64)
    shift = np.full(count, -999, np.int16)
    eligible = np.zeros(count, bool)
    exclusion = {
        "no_complete_174_frame_fixed_population_window": 0,
        "no_background_vehicle": 0,
    }
    metadata_cache: dict[int, pd.DataFrame] = {}

    for row_index, row in source.iterrows():
        recording = int(row["recording_id"])
        if recording not in metadata_cache:
            metadata_cache[recording] = _recording_metadata(raw_dir, recording)
        metadata = metadata_cache[recording]
        ids = np.asarray(
            [
                int(row["ego_id"]),
                *(int(row[f"{slot}_id"]) for slot in SLOT_NAMES),
            ],
            np.int64,
        )
        active = ids >= 0
        agent_ids[row_index] = ids
        if active.sum() <= 1:
            exclusion["no_background_vehicle"] += 1
            continue
        selected = metadata.loc[ids[active]]
        lengths[row_index, active] = selected["width"].to_numpy(np.float32)
        widths[row_index, active] = selected["height"].to_numpy(np.float32)
        vehicle_class[row_index, active] = (
            selected["class"].astype(str).str.lower().to_numpy()
        )
        window = choose_causal_window(
            int(row["anchor_frame"]),
            selected["initialFrame"].to_numpy(np.int64),
            selected["finalFrame"].to_numpy(np.int64),
            history_frames=int(config["benchmark"]["history_frames"]),
            horizon_frames=int(config["benchmark"]["horizon_frames"]),
        )
        if window is None:
            exclusion["no_complete_174_frame_fixed_population_window"] += 1
            continue
        eligible[row_index] = True
        history_start[row_index] = window.history_start_frame
        decision_frame[row_index] = window.decision_frame
        future_end[row_index] = window.future_end_frame
        shift[row_index] = window.shift_from_original_anchor

    # A true 25-frame prefix would have at least the ego valid in every frame.
    # The old cache intentionally has 24 invalid compatibility frames.
    legacy_real_prefix = np.asarray(legacy_valid[:, :25, 0]).all(axis=1)
    recording = source["recording_id"].to_numpy(np.int64)
    leakage: dict[int, set[int]] = {}
    for rec, split in zip(recording, np.asarray(split_index)):
        leakage.setdefault(int(rec), set()).add(int(split))
    cross_split_recordings = sorted(
        rec for rec, splits in leakage.items() if len(splits) > 1
    )

    np.savez_compressed(
        output / "scenario_manifest.npz",
        source_row=np.flatnonzero(eligible),
        sequence_id=np.asarray(sequence_ids)[eligible],
        split_index=np.asarray(split_index)[eligible],
        recording_id=recording[eligible],
        ego_id=source["ego_id"].to_numpy(np.int64)[eligible],
        agent_ids=agent_ids[eligible],
        lengths_m=lengths[eligible],
        widths_m=widths[eligible],
        vehicle_class=vehicle_class[eligible],
        history_start_frame=history_start[eligible],
        decision_frame=decision_frame[eligible],
        future_end_frame=future_end[eligible],
        shift_from_original_anchor=shift[eligible],
    )
    split_summary: dict[str, dict[str, int]] = {}
    for index, name in enumerate(SPLIT_NAMES):
        requested = np.asarray(split_index) == index
        split_summary[str(name)] = {
            "legacy_rows": int(requested.sum()),
            "strict_causal_rows": int((requested & eligible).sum()),
            "excluded_rows": int((requested & ~eligible).sum()),
        }
    report: dict[str, Any] = {
        "benchmark_id": config["benchmark"]["id"],
        "config_path": str(loaded_from),
        "source_commit": config["benchmark"]["source_commit"],
        "legacy_cache": {
            "path": str(cache),
            "manifest_sha256": sha256_file(legacy_manifest_path),
            "cache_format": legacy_manifest.get("cache_format"),
            "rows": int(count),
            "rows_with_25_real_ego_history_frames": int(legacy_real_prefix.sum()),
            "prefix_padding_detected": bool(not legacy_real_prefix.all()),
        },
        "causal_rebuild": {
            "policy": config["execution"]["causal_window_policy"],
            "history_frames": int(config["benchmark"]["history_frames"]),
            "horizon_frames": int(config["benchmark"]["horizon_frames"]),
            "eligible_rows": int(eligible.sum()),
            "excluded_rows": int((~eligible).sum()),
            "exclusion_reasons": exclusion,
            "shift_frames": {
                "minimum": int(shift[eligible].min(initial=0)),
                "maximum": int(shift[eligible].max(initial=0)),
                "median": float(np.median(shift[eligible])),
            },
        },
        "split_summary": split_summary,
        "recording_split_audit": {
            "recordings": int(len(leakage)),
            "cross_split_recordings": cross_split_recordings,
            "passed": not cross_split_recordings,
        },
        "source_files": {
            "natural_segments_csv": str(source_path),
            "natural_segments_sha256": sha256_file(source_path),
            "raw_highd_dir": str(raw_dir),
        },
    }
    (output / "dataset_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

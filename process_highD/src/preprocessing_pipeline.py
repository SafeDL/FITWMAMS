"""Build one canonical table of screened highD natural-driving segments."""

from __future__ import annotations

from collections import Counter
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from process_highD.src.io_utils import (
    json_ready,
    load_config,
    resolve_data_path,
    resolve_recording_ids,
    write_json,
)
from process_highD.src.natural_segments import (
    RISK_COMPONENT_NAMES,
    SLOT_NAMES,
    build_natural_segments_for_recording,
    options_from_config,
)
from process_highD.src.preprocess import prepare_recording

logger = logging.getLogger(__name__)


def parse_recording_override(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.lower() == "all":
        return {"include": "all", "exclude": []}
    return {
        "include": [int(item.strip()) for item in stripped.split(",") if item.strip()],
        "exclude": [],
    }


def validate_raw_dir(raw_dir: Path) -> None:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"highD raw directory does not exist: {raw_dir}")


def risk_quantiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {}
    probs = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0]
    return {f"q{prob:g}": float(np.quantile(finite, prob)) for prob in probs}


def natural_output_paths(cfg: dict[str, Any], config_path: Path) -> dict[str, Path]:
    out_dir = resolve_data_path(cfg["paths"]["output_dir"], config_path)
    return {
        "out_dir": out_dir,
        "segment_csv": out_dir / "natural_segments.csv",
        "risk_trace_npz": out_dir / "natural_risk_traces.npz",
        "summary": out_dir / "natural_segments_summary.json",
    }


def build_natural_segments_dataset(
    *,
    config_path: Path,
    recording_override: str | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    raw_dir = resolve_data_path(cfg["paths"]["raw_dir"], config_path)
    validate_raw_dir(raw_dir)
    paths = natural_output_paths(cfg, config_path)
    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    recordings_cfg = parse_recording_override(recording_override) or cfg["recordings"]
    recording_ids = resolve_recording_ids(raw_dir, recordings_cfg)
    options = options_from_config(cfg)

    logger.info("Processing highD recordings: %s", recording_ids)
    logger.info(
        "Fixed natural window: window=%d frames, stride=%d frames",
        options.total_steps,
        options.anchor_stride_steps,
    )

    all_frames: list[pd.DataFrame] = []
    risk_blocks: list[np.ndarray] = []
    per_recording: list[dict[str, Any]] = []
    reject_totals: Counter[str] = Counter()

    for recording_id in tqdm(recording_ids, desc="Natural highD segments"):
        rec = prepare_recording(raw_dir, int(recording_id), cfg)
        frame, risk_trace, recording_summary = (
            build_natural_segments_for_recording(rec, options)
        )
        if not frame.empty:
            all_frames.append(frame)
            risk_blocks.append(risk_trace)
        per_recording.append(recording_summary)
        reject_totals.update(recording_summary["reject_counts"])
        logger.info(
            "Recording %02d: segments=%d rejects=%s",
            int(recording_id),
            int(recording_summary["num_segments"]),
            recording_summary["reject_counts"],
        )

    if all_frames:
        segments = pd.concat(all_frames, ignore_index=True)
        segments["risk_trace_row"] = np.arange(len(segments), dtype=np.int64)
        risk_trace_all = np.concatenate(risk_blocks, axis=0).astype(np.float32)
    else:
        raise RuntimeError("No valid natural highD segments were built")

    segments.to_csv(paths["segment_csv"], index=False)
    np.savez_compressed(
        paths["risk_trace_npz"],
        risk_trace=risk_trace_all,
    )

    values = pd.to_numeric(segments["event_risk"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    values = values[np.isfinite(values)]
    summary: dict[str, Any] = {
        "config_path": str(config_path),
        "raw_dir": str(raw_dir),
        "output_dir": str(paths["out_dir"]),
        "segment_csv": str(paths["segment_csv"]),
        "risk_trace_npz": str(paths["risk_trace_npz"]),
        "recording_ids": [int(value) for value in recording_ids],
        "num_recordings": int(len(recording_ids)),
        "num_segments": int(len(segments)),
        "fps": float(options.fps),
        "window_steps": int(options.total_steps),
        "window_seconds": float(options.total_steps / options.fps),
        "risk_window_steps": int(options.total_steps),
        "risk_window_seconds": float(options.total_steps / options.fps),
        "total_steps": int(options.total_steps),
        "anchor_stride_steps": int(options.anchor_stride_steps),
        "anchor_stride_seconds": float(options.anchor_stride_steps / options.fps),
        "lateral_event_policy": {
            "require_complete": True,
            "pre_cross_steps": int(options.lateral_pre_cross_steps),
            "post_cross_steps": int(options.lateral_post_cross_steps),
            "stable_lane_steps": int(options.lateral_stable_steps),
            "semantics": "adjacent crossing with stable pre/post lane context",
        },
        "background_slot_stability_policy": {
            "require_c0_active_slots_through_full_window": True,
            "window_steps": int(options.total_steps),
            "semantics": "C0-active background slots remain present and non-abnormal",
        },
        "evaluation_scope": {
            "excluded_risk_slots": [],
            "training_population_modified": False,
        },
        "slot_names": list(SLOT_NAMES),
        "risk_component_names": list(RISK_COMPONENT_NAMES),
        "segment_csv_columns": list(segments.columns),
        "risk_trace_npz_keys": ["risk_trace"],
        "risk_model": "safety_envelope_intrusion",
        "safety_envelope_risk": json_ready(options.sei.__dict__),
        "risk_quantiles": risk_quantiles(values),
        "per_recording": per_recording,
        "reject_totals": {
            key: int(value) for key, value in sorted(reject_totals.items())
        },
    }
    write_json(paths["summary"], summary)
    logger.info("Wrote segment CSV: %s", paths["segment_csv"])
    logger.info("Wrote risk trace cache: %s", paths["risk_trace_npz"])
    logger.info("Wrote summary: %s", paths["summary"])
    return summary

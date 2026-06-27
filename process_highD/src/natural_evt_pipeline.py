"""Pipeline helpers for highD natural equal-length segments and EVT."""
from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import genpareto
from tqdm import tqdm

from process_highD.src.evt_diagnostics import (
    write_evt_diagnostic_plots,
    write_gpd_diagnostic_panel,
)
from process_highD.src.io_utils import (
    ensure_dir,
    load_config,
    resolve_data_path,
    resolve_recording_ids,
)
from process_highD.src.loader import load_recording
from process_highD.src.natural_segments import (
    RISK_COMPONENT_NAMES,
    SLOT_NAMES,
    build_natural_segments_for_recording,
    options_from_config,
)
from process_highD.src.preprocess import (
    filter_abnormal_tracks,
    normalize_driving_direction,
    resample_recording,
)
from tools.evt import fit_evt_model


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
    if not raw_dir.exists():
        raise FileNotFoundError(f"highD raw directory does not exist: {raw_dir}")
    if not list(raw_dir.glob("*_tracks.csv")):
        raise FileNotFoundError(
            f"No *_tracks.csv files found in highD raw directory: {raw_dir}"
        )


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2, ensure_ascii=False)


def risk_quantiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {}
    probs = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0]
    return {f"q{prob:g}": float(np.quantile(finite, prob)) for prob in probs}


def gpd_gof_statistics(values: np.ndarray, *, u: float, xi: float, beta: float) -> dict[str, float]:
    excess = np.asarray(values, dtype=np.float64)
    excess = excess[np.isfinite(excess) & (excess > float(u))] - float(u)
    excess = np.sort(excess[excess > 0.0])
    n = int(excess.size)
    if n == 0:
        return {
            "num_exceedances": 0,
            "ks": float("nan"),
            "cramer_von_mises": float("nan"),
            "anderson_darling": float("nan"),
        }
    cdf = genpareto.cdf(
        excess,
        c=float(xi),
        loc=0.0,
        scale=max(float(beta), 1.0e-12),
    )
    cdf = np.clip(cdf, 1.0e-12, 1.0 - 1.0e-12)
    i = np.arange(1, n + 1, dtype=np.float64)
    ks = float(
        max(
            np.max(np.abs(cdf - (i - 1.0) / n)),
            np.max(np.abs(i / n - cdf)),
        )
    )
    cvm = float(
        (1.0 / (12.0 * n))
        + np.sum((cdf - (2.0 * i - 1.0) / (2.0 * n)) ** 2)
    )
    ad = float(
        -n
        - np.mean(
            (2.0 * i - 1.0)
            * (np.log(cdf) + np.log(1.0 - cdf[::-1]))
        )
    )
    return {
        "num_exceedances": n,
        "ks": ks,
        "cramer_von_mises": cvm,
        "anderson_darling": ad,
    }


def threshold_sensitivity_rows(model: Any) -> list[dict[str, float]]:
    fields = (
        "u",
        "k",
        "exceedance_rate",
        "xi",
        "beta",
        "modified_scale",
        "endpoint",
        "z1000",
    )
    rows: list[dict[str, float]] = []
    for candidate in list(getattr(model, "threshold_candidates", []) or []):
        row: dict[str, float] = {}
        for field in fields:
            value = candidate.get(field, float("nan"))
            row[field] = float(value)
        rows.append(row)
    rows.sort(key=lambda item: float(item["u"]))
    return rows


def write_threshold_sensitivity_csv(path: Path, model: Any) -> list[dict[str, float]]:
    rows = threshold_sensitivity_rows(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return rows


def natural_output_paths(cfg: dict[str, Any], config_path: Path) -> dict[str, Path]:
    out_dir = ensure_dir(resolve_data_path(cfg["paths"]["output_dir"], config_path))
    evt_cfg = dict(cfg.get("evt", {}))
    evt_summary = resolve_data_path(evt_cfg["summary_path"], config_path)
    evt_dir = evt_summary.parent
    return {
        "out_dir": out_dir,
        "segment_csv": out_dir / "natural_segments.csv",
        "risk_trace_npz": out_dir / "natural_risk_traces.npz",
        "summary": out_dir / "natural_segments_summary.json",
        "evt_model": resolve_data_path(evt_cfg["model_path"], config_path),
        "evt_summary": evt_summary,
        "evt_figure_dir": resolve_data_path(evt_cfg["figure_dir"], config_path),
        "evt_threshold_sensitivity": evt_dir / "natural_evt_threshold_sensitivity.csv",
        "tail_contexts": out_dir / "natural_tail_contexts.csv",
        "tail_context_summary": out_dir / "natural_tail_contexts_summary.json",
    }


def _fit_natural_evt_values(
    *,
    values: np.ndarray,
    config: dict[str, Any],
    config_path: Path,
    segment_csv_path: Path,
    evt_kind: str,
    model_path: Path,
    summary_path: Path,
    figure_dir: Path,
    threshold_sensitivity_path: Path,
) -> dict[str, Any]:
    evt_cfg = dict(config.get("evt", {}))
    return_periods = tuple(
        sorted({int(value) for value in evt_cfg.get("return_periods", [20, 50, 100])})
    )

    min_threshold_rate = evt_cfg.get("min_threshold_exceedance_rate", None)
    model = fit_evt_model(
        values,
        return_periods=return_periods,
        min_exceedances=int(evt_cfg.get("min_exceedances", 200)),
        max_tail_fraction=float(evt_cfg.get("max_tail_fraction", 0.20)),
        max_threshold_candidates=int(evt_cfg.get("max_threshold_candidates", 400)),
        min_threshold_exceedance_rate=(
            None if min_threshold_rate is None else float(min_threshold_rate)
        ),
        bootstrap_samples=int(evt_cfg.get("bootstrap_samples", 200)),
        random_seed=int(evt_cfg.get("random_seed", 42)),
    )
    risk_model = "safety_envelope_intrusion"
    model.to_json(
        model_path,
        model_type=f"gpd_pot_highd_natural_{risk_model}_event_risk",
    )
    threshold_sensitivity = write_threshold_sensitivity_csv(
        threshold_sensitivity_path,
        model,
    )

    figures = write_evt_diagnostic_plots(
        figure_dir,
        model=model,
        values=values,
        risk_variable="R_SEI",
        histogram_filename=f"natural_evt_{evt_kind}_event_risk_histogram.png",
        histogram_key=f"natural_{evt_kind}_event_risk_histogram",
    )
    figures.update(
        write_gpd_diagnostic_panel(
            figure_dir,
            model=model,
            values=values,
            risk_variable="R_SEI",
            output_filename=f"natural_evt_{evt_kind}_gpd_diagnostic_panel.png",
            output_key=f"natural_{evt_kind}_gpd_diagnostic_panel",
        )
    )

    tail_count = int(np.sum(values > float(model.u)))
    segment_cfg = dict(config.get("segments", {}))
    sampling_cfg = dict(config.get("sampling", {}))
    fps = float(sampling_cfg.get("target_fps", 25.0))
    risk_window_seconds = float(segment_cfg.get("window_seconds", 6.0))
    risk_window_frames = int(round(risk_window_seconds * fps))
    summary = {
        "evt_kind": evt_kind,
        "model_path": str(model_path),
        "segment_csv": str(segment_csv_path),
        "model_type": f"gpd_pot_highd_natural_{risk_model}_event_risk",
        "risk_model": risk_model,
        "risk_variable": (
            "R_SEI(tau) = Safety-Envelope Intrusion Risk: prefix maximum "
            "of raw positive safety-ellipse intrusion plus linear exposure"
        ),
        "num_calibration_segments": int(values.size),
        "num_tail_segments": tail_count,
        "u": float(model.u),
        "xi": float(model.xi),
        "beta": float(model.beta),
        "exceedance_rate": float(model.exceedance_rate),
        "return_levels": model.return_levels,
        "return_level_ci": model.return_level_ci,
        "gpd_gof": gpd_gof_statistics(
            values,
            u=float(model.u),
            xi=float(model.xi),
            beta=float(model.beta),
        ),
        "threshold_selection": model.threshold_selection,
        "threshold_sensitivity_csv": str(threshold_sensitivity_path),
        "threshold_sensitivity": threshold_sensitivity,
        "figures": figures,
        "audit_protocol": {
            "source": "highD natural equal-length segments only",
            "frequency_hz": fps,
            "risk_window_frames": risk_window_frames,
            "risk_window_seconds": risk_window_seconds,
            "risk_trace": (
                "nondecreasing prefix trajectory score computed on the complete "
                "fixed-length window from raw positive safety-envelope intrusion risk"
            ),
        },
    }
    write_json(summary_path, summary)
    return summary


def fit_natural_evt(
    *,
    values: np.ndarray,
    config: dict[str, Any],
    config_path: Path,
    segment_csv_path: Path,
) -> dict[str, Any]:
    paths = natural_output_paths(config, config_path)
    return _fit_natural_evt_values(
        values=values,
        config=config,
        config_path=config_path,
        segment_csv_path=segment_csv_path,
        evt_kind="raw",
        model_path=paths["evt_model"],
        summary_path=paths["evt_summary"],
        figure_dir=paths["evt_figure_dir"],
        threshold_sensitivity_path=paths["evt_threshold_sensitivity"],
    )


def build_natural_segments_dataset(
    *,
    config_path: Path,
    recording_override: str | None = None,
    fit_evt: bool = False,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    raw_dir = resolve_data_path(cfg["paths"]["raw_dir"], config_path)
    validate_raw_dir(raw_dir)
    paths = natural_output_paths(cfg, config_path)
    recordings_cfg = parse_recording_override(recording_override) or cfg.get("recordings", {})
    recording_ids = resolve_recording_ids(raw_dir, recordings_cfg)
    options = options_from_config(cfg)

    logger.info("Processing highD recordings: %s", recording_ids)
    logger.info(
        "Fixed natural window: window=%d frames, stride=%d frames",
        options.total_steps,
        options.anchor_stride_steps,
    )

    target_fps = int(cfg.get("sampling", {}).get("target_fps", 25))
    all_frames: list[pd.DataFrame] = []
    risk_blocks: list[np.ndarray] = []
    packed_slot_mask_blocks: list[np.ndarray] = []
    per_recording: list[dict[str, Any]] = []
    reject_totals: Counter[str] = Counter()

    for recording_id in tqdm(recording_ids, desc="Natural highD segments"):
        rec = load_recording(str(raw_dir), int(recording_id))
        rec = normalize_driving_direction(rec)
        rec = filter_abnormal_tracks(rec, cfg)
        rec = resample_recording(rec, target_fps)
        frame, risk_trace, slot_time_mask, recording_summary = (
            build_natural_segments_for_recording(rec, options)
        )
        if not frame.empty:
            all_frames.append(frame)
            risk_blocks.append(risk_trace)
            flat_mask = slot_time_mask.reshape(slot_time_mask.shape[0], -1)
            packed_slot_mask_blocks.append(np.packbits(flat_mask, axis=1))
        per_recording.append(recording_summary)
        reject_totals.update(recording_summary.get("reject_counts", {}))
        logger.info(
            "Recording %02d: segments=%d rejects=%s",
            int(recording_id),
            int(recording_summary["num_segments"]),
            recording_summary.get("reject_counts", {}),
        )

    if all_frames:
        segments = pd.concat(all_frames, ignore_index=True)
        segments["risk_trace_row"] = np.arange(len(segments), dtype=np.int64)
        risk_trace_all = np.concatenate(risk_blocks, axis=0).astype(np.float32)
        packed_slot_mask = np.concatenate(packed_slot_mask_blocks, axis=0)
    else:
        raise RuntimeError("No valid natural highD segments were built")

    observed_trace_max = np.max(risk_trace_all, axis=1)
    csv_event_risk = pd.to_numeric(segments["event_risk"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    max_abs_diff = float(np.max(np.abs(observed_trace_max - csv_event_risk)))
    if max_abs_diff > 1.0e-5:
        raise RuntimeError(
            "risk trace/event risk mismatch: "
            f"max_abs_diff={max_abs_diff:.6g}"
        )

    segments.to_csv(paths["segment_csv"], index=False)
    np.savez_compressed(
        paths["risk_trace_npz"],
        risk_trace=risk_trace_all,
        slot_time_mask_packed=packed_slot_mask,
        slot_time_mask_shape=np.asarray(
            [len(segments), options.total_steps, len(SLOT_NAMES)],
            dtype=np.int64,
        ),
        slot_names=np.asarray(SLOT_NAMES, dtype="U32"),
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
        "slot_names": list(SLOT_NAMES),
        "risk_component_names": list(RISK_COMPONENT_NAMES),
        "segment_csv_columns": list(segments.columns),
        "risk_trace_npz_keys": [
            "risk_trace",
            "slot_time_mask_packed",
            "slot_time_mask_shape",
            "slot_names",
        ],
        "risk_model": "safety_envelope_intrusion",
        "safety_envelope_risk": json_ready(options.sei.__dict__),
        "risk_quantiles": risk_quantiles(values),
        "max_trace_event_risk_abs_diff": max_abs_diff,
        "per_recording": per_recording,
        "reject_totals": {
            key: int(value) for key, value in sorted(reject_totals.items())
        },
    }
    summary["evt_summary"] = (
        fit_natural_evt(
            values=values,
            config=cfg,
            config_path=config_path,
            segment_csv_path=paths["segment_csv"],
        )
        if fit_evt
        else None
    )
    write_json(paths["summary"], summary)
    logger.info("Wrote segment CSV: %s", paths["segment_csv"])
    logger.info("Wrote risk trace cache: %s", paths["risk_trace_npz"])
    logger.info("Wrote summary: %s", paths["summary"])
    return summary


def refit_natural_evt(*, config_path: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    raw_dir = resolve_data_path(cfg["paths"]["raw_dir"], config_path)
    validate_raw_dir(raw_dir)
    paths = natural_output_paths(cfg, config_path)
    if not paths["segment_csv"].exists():
        raise FileNotFoundError(
            f"Cannot refit EVT without segment CSV: {paths['segment_csv']}"
        )
    segments = pd.read_csv(paths["segment_csv"])
    values = pd.to_numeric(segments["event_risk"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    values = values[np.isfinite(values)]
    evt_summary = fit_natural_evt(
        values=values,
        config=cfg,
        config_path=config_path,
        segment_csv_path=paths["segment_csv"],
    )
    if paths["summary"].exists():
        with open(paths["summary"], "r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {
            "config_path": str(config_path),
            "raw_dir": str(raw_dir),
            "output_dir": str(paths["out_dir"]),
            "segment_csv": str(paths["segment_csv"]),
            "num_segments": int(len(segments)),
        }
    summary["risk_quantiles"] = risk_quantiles(values)
    summary["evt_summary"] = evt_summary
    write_json(paths["summary"], summary)
    logger.info("Refit EVT from existing segment CSV: %s", paths["segment_csv"])
    logger.info("Wrote summary: %s", paths["summary"])
    return summary


def select_natural_tail_contexts(
    *,
    config_path: Path,
    min_event_risk: float | None = None,
    top_k: int = 0,
    output_csv: Path | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    paths = natural_output_paths(cfg, config_path)
    if not paths["segment_csv"].exists():
        raise FileNotFoundError(
            f"Cannot select tail contexts without segment CSV: {paths['segment_csv']}"
        )
    segments = pd.read_csv(paths["segment_csv"])
    threshold = min_event_risk
    threshold_source = "explicit_min_event_risk"
    threshold_comparator = ">="
    evt_summary_path = paths["evt_summary"]
    evt_summary: dict[str, Any] = {}
    if threshold is None and evt_summary_path.exists():
        with open(evt_summary_path, "r", encoding="utf-8") as f:
            evt_summary = json.load(f)
        threshold = float(evt_summary["u"])
        threshold_source = "evt_pot_threshold_u"
        threshold_comparator = ">"
    if threshold is None:
        threshold = float(segments["event_risk"].quantile(0.999))
        threshold_source = "empirical_q999"
        threshold_comparator = ">="

    risks = segments["event_risk"].astype(float)
    if threshold_comparator == ">":
        selected = segments[risks > float(threshold)].copy()
    else:
        selected = segments[risks >= float(threshold)].copy()
    selected = selected.sort_values(
        ["event_risk", "recording_id", "anchor_frame"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    if int(top_k) > 0:
        selected = selected.head(int(top_k))
    output_path = output_csv or paths["tail_contexts"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)
    summary = {
        "segment_csv": str(paths["segment_csv"]),
        "output_csv": str(output_path),
        "selection_threshold": float(threshold),
        "threshold_source": threshold_source,
        "threshold_comparator": threshold_comparator,
        "top_k": int(top_k),
        "num_selected": int(len(selected)),
    }
    if evt_summary:
        summary["evt_pot_threshold_u"] = float(evt_summary["u"])
    write_json(paths["tail_context_summary"], summary)
    logger.info("Wrote natural tail contexts: %s", output_path)
    return summary

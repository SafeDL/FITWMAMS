#!/usr/bin/env python3
"""Inspect action-diffusion dataset quality and risk/action coupling."""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

from diffusion.src.data import INDEX_TO_SPLIT, SPLIT_TO_INDEX
from diffusion.src.utils import load_json, load_yaml, save_json, setup_logging


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "diffusion_following.yaml"
DEFAULT_DATASET_PATH = "dataset.npz"
DEFAULT_SCHEMA_PATH = "feature_schema.json"
DEFAULT_SPLIT = "all"
DEFAULT_RISK_BINS = "0,0.5,0.8,0.9,0.95,0.99,1"
DEFAULT_LOG_LEVEL = "INFO"
logger = logging.getLogger(__name__)


def _resolve_output_dir(config: dict, config_dir: Path) -> Path:
    return (config_dir / config.get("paths", {}).get("output_dir", "../../../data/diffusion/following")).resolve()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _resolve_input_path(value: str | None, output_dir: Path, default: str) -> Path:
    raw = value or default
    path = Path(raw)
    if path.is_absolute():
        return path
    cwd_path = path.resolve()
    if cwd_path.exists():
        return cwd_path
    return (output_dir / path).resolve()


def _parse_edges(value: str | None) -> list[float]:
    if value is None or not str(value).strip():
        edges = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]
    else:
        edges = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    edges = sorted(set(float(np.clip(x, 0.0, 1.0)) for x in edges))
    if not edges or edges[0] > 0.0:
        edges.insert(0, 0.0)
    if edges[-1] < 1.0:
        edges.append(1.0)
    if len(edges) < 2:
        raise ValueError("risk bin edges must define at least one interval")
    return edges


def _select_indices(arrays: dict[str, np.ndarray], split: str) -> np.ndarray:
    if split == "all":
        return np.arange(arrays["actions"].shape[0])
    return np.where(arrays["split_index"] == SPLIT_TO_INDEX[split])[0]


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p01": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    qs = np.quantile(x, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p01": float(qs[0]),
        "p05": float(qs[1]),
        "p25": float(qs[2]),
        "p50": float(qs[3]),
        "p75": float(qs[4]),
        "p95": float(qs[5]),
        "p99": float(qs[6]),
        "max": float(np.max(x)),
    }


def _split_counts(arrays: dict[str, np.ndarray]) -> dict[str, int]:
    split_index = arrays["split_index"]
    return {name: int(np.sum(split_index == idx)) for idx, name in INDEX_TO_SPLIT.items()}


def _risk_percentile(arrays: dict[str, np.ndarray]) -> np.ndarray:
    if "risk_percentile" in arrays:
        return np.asarray(arrays["risk_percentile"], dtype=np.float32)
    risk = np.asarray(arrays["risk_raw"] if "risk_raw" in arrays else arrays["risk"], dtype=np.float32)
    order = np.argsort(np.argsort(risk))
    return (order.astype(np.float32) / max(len(order) - 1, 1)).astype(np.float32)


def _actions_to_ax(actions: np.ndarray, context_states: np.ndarray, schema: dict, config: dict) -> tuple[np.ndarray, np.ndarray]:
    action_cfg = config.get("action", {})
    rep = str(schema.get("action_representation", action_cfg.get("representation", "acceleration"))).lower()
    ax_min = float(action_cfg.get("ax_min", -8.0))
    ax_max = float(action_cfg.get("ax_max", 4.0))
    dt = float(schema.get("dt", 0.04))
    if rep == "jerk":
        prev_ax = context_states[:, -1, 1, 4].astype(np.float32)
        ax = prev_ax[:, None] + np.cumsum(actions[:, :, 0], axis=1) * dt
    else:
        ax = actions[:, :, 0]
    unclipped = ax.astype(np.float32)
    clipped = np.clip(unclipped, ax_min, ax_max).astype(np.float32)
    return clipped, unclipped


def _history_gap_delta_v(arrays: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, Any]:
    if "relative_history" in arrays:
        rel = np.asarray(arrays["relative_history"][idx], dtype=np.float32)
        return {
            "gap_all_steps": _summary(rel[:, :, 0]),
            "gap_last_step": _summary(rel[:, -1, 0]),
            "relative_speed_all_steps_delta_v": _summary(rel[:, :, 2]),
            "relative_speed_last_step_delta_v": _summary(rel[:, -1, 2]),
            "delta_v_definition": "ego_vx_minus_lead_vx",
        }

    states = np.asarray(arrays["context_states"][idx], dtype=np.float32)
    ego = states[:, :, 0]
    lead = states[:, :, 1]
    total_n = arrays["actions"].shape[0]
    ego_len = np.asarray(arrays.get("ego_length", np.zeros((total_n,))), dtype=np.float32)[idx]
    adv_len = np.asarray(arrays.get("adv_length", np.zeros((total_n,))), dtype=np.float32)[idx]
    gap = lead[:, :, 0] - ego[:, :, 0] - 0.5 * (ego_len[:, None] + adv_len[:, None])
    delta_v = ego[:, :, 2] - lead[:, :, 2]
    return {
        "gap_all_steps": _summary(gap),
        "gap_last_step": _summary(gap[:, -1]),
        "relative_speed_all_steps_delta_v": _summary(delta_v),
        "relative_speed_last_step_delta_v": _summary(delta_v[:, -1]),
        "delta_v_definition": "ego_vx_minus_lead_vx",
    }


def _bin_label(lo: float, hi: float, last: bool) -> str:
    right = "]" if last else ")"
    return f"[{lo:.2f},{hi:.2f}{right}"


def _risk_bin_rows(
    arrays: dict[str, np.ndarray],
    idx: np.ndarray,
    risk_pct: np.ndarray,
    actions: np.ndarray,
    ax_clipped: np.ndarray,
    ax_unclipped: np.ndarray,
    edges: list[float],
    config: dict,
) -> list[dict[str, Any]]:
    risk_raw = np.asarray(arrays["risk_raw"] if "risk_raw" in arrays else arrays["risk"], dtype=np.float32)
    action_flat = actions[:, :, 0]
    ax_min = float(config.get("action", {}).get("ax_min", -8.0))
    ax_max = float(config.get("action", {}).get("ax_max", 4.0))
    rows: list[dict[str, Any]] = []
    pct = risk_pct[idx]
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        mask = (pct >= lo) & (pct <= hi if last else pct < hi)
        local_idx = idx[mask]
        if len(local_idx) == 0:
            rows.append({"risk_bin": _bin_label(lo, hi, last), "count": 0})
            continue
        local_actions = action_flat[mask]
        local_ax = ax_clipped[mask]
        local_unclipped = ax_unclipped[mask]
        clip_mask = (local_unclipped < ax_min) | (local_unclipped > ax_max)
        rows.append(
            {
                "risk_bin": _bin_label(lo, hi, last),
                "risk_percentile_min": float(np.min(pct[mask])),
                "risk_percentile_max": float(np.max(pct[mask])),
                "count": int(len(local_idx)),
                "risk_raw_mean": float(np.mean(risk_raw[local_idx])),
                "risk_raw_p50": float(np.quantile(risk_raw[local_idx], 0.50)),
                "risk_raw_p95": float(np.quantile(risk_raw[local_idx], 0.95)),
                "action_mean": float(np.mean(local_actions)),
                "action_std": float(np.std(local_actions)),
                "action_abs_mean": float(np.mean(np.abs(local_actions))),
                "action_p05": float(np.quantile(local_actions, 0.05)),
                "action_p50": float(np.quantile(local_actions, 0.50)),
                "action_p95": float(np.quantile(local_actions, 0.95)),
                "ax_mean": float(np.mean(local_ax)),
                "ax_std": float(np.std(local_ax)),
                "ax_p05": float(np.quantile(local_ax, 0.05)),
                "ax_p50": float(np.quantile(local_ax, 0.50)),
                "ax_p95": float(np.quantile(local_ax, 0.95)),
                "hard_brake_ratio": float(np.mean(local_ax < -3.0)),
                "strong_brake_ratio": float(np.mean(local_ax < -1.5)),
                "ax_clip_rate": float(np.mean(clip_mask)),
            }
        )
    return rows


def _risk_action_signal(rows: list[dict[str, Any]]) -> dict[str, Any]:
    non_empty = [row for row in rows if int(row.get("count", 0)) > 0]
    if len(non_empty) < 2:
        return {"status": "insufficient_bins"}
    low = non_empty[0]
    high = non_empty[-1]
    pooled_action_std = max(
        float(np.mean([float(row.get("action_std", 0.0)) for row in non_empty])),
        1e-6,
    )
    action_mean_delta = float(high["action_mean"]) - float(low["action_mean"])
    action_effect_size = abs(action_mean_delta) / pooled_action_std
    hard_brake_delta = float(high["hard_brake_ratio"]) - float(low["hard_brake_ratio"])
    ax_mean_delta = float(high["ax_mean"]) - float(low["ax_mean"])
    monotonic_hard_brake = bool(
        np.all(np.diff([float(row["hard_brake_ratio"]) for row in non_empty]) >= -1e-6)
    )
    weak_signal = action_effect_size < 0.10 and abs(hard_brake_delta) < 0.02 and abs(ax_mean_delta) < 0.05
    return {
        "low_bin": low["risk_bin"],
        "high_bin": high["risk_bin"],
        "action_mean_delta_high_minus_low": action_mean_delta,
        "action_effect_size_abs_delta_over_pooled_std": float(action_effect_size),
        "ax_mean_delta_high_minus_low": ax_mean_delta,
        "hard_brake_ratio_delta_high_minus_low": hard_brake_delta,
        "hard_brake_ratio_monotonic_increasing": monotonic_hard_brake,
        "weak_risk_action_signal": bool(weak_signal),
        "interpretation": (
            "High-risk and low-risk real action distributions look very similar; revisit risk labels/window construction."
            if weak_signal
            else "Risk bins show measurable real-action differences; if generation ignores risk, inspect model/training/conditioning."
        ),
    }


def _write_bin_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "risk_bin",
        "risk_percentile_min",
        "risk_percentile_max",
        "count",
        "risk_raw_mean",
        "risk_raw_p50",
        "risk_raw_p95",
        "action_mean",
        "action_std",
        "action_abs_mean",
        "action_p05",
        "action_p50",
        "action_p95",
        "ax_mean",
        "ax_std",
        "ax_p05",
        "ax_p50",
        "ax_p95",
        "hard_brake_ratio",
        "strong_brake_ratio",
        "ax_clip_rate",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def inspect_dataset(config: dict, config_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _resolve_output_dir(config, config_dir)
    dataset_path = _resolve_input_path(args.dataset_path, output_dir, DEFAULT_DATASET_PATH)
    schema_path = _resolve_input_path(args.schema_path, output_dir, DEFAULT_SCHEMA_PATH)
    result_dir = Path(args.output_dir).resolve() if args.output_dir else output_dir

    arrays = _load_npz(dataset_path)
    schema = load_json(schema_path)
    idx = _select_indices(arrays, str(args.split))
    if len(idx) == 0:
        raise RuntimeError(f"No samples for split={args.split}")

    actions = np.asarray(arrays["actions"][idx], dtype=np.float32)
    context_states = np.asarray(arrays["context_states"][idx], dtype=np.float32)
    ax_clipped, ax_unclipped = _actions_to_ax(actions, context_states, schema, config)
    risk_raw = np.asarray(arrays["risk_raw"] if "risk_raw" in arrays else arrays["risk"], dtype=np.float32)
    risk_pct = _risk_percentile(arrays)
    edges = _parse_edges(args.risk_bins)
    ax_min = float(config.get("action", {}).get("ax_min", -8.0))
    ax_max = float(config.get("action", {}).get("ax_max", 4.0))
    clip_mask = (ax_unclipped < ax_min) | (ax_unclipped > ax_max)
    bin_rows = _risk_bin_rows(arrays, idx, risk_pct, actions, ax_clipped, ax_unclipped, edges, config)

    summary: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "schema_path": str(schema_path),
        "split": str(args.split),
        "num_selected_samples": int(len(idx)),
        "split_counts": _split_counts(arrays),
        "action_representation": schema.get("action_representation", config.get("action", {}).get("representation")),
        "action_keys": schema.get("action_keys", []),
        "risk": {
            "risk_raw": _summary(risk_raw[idx]),
            "risk_percentile": _summary(risk_pct[idx]),
        },
        "action_distribution": {
            "action": _summary(actions[:, :, 0]),
            "action_abs": _summary(np.abs(actions[:, :, 0])),
        },
        "ax_from_actions": {
            "unclipped": _summary(ax_unclipped),
            "clipped": _summary(ax_clipped),
            "ax_min": ax_min,
            "ax_max": ax_max,
            "clip_rate": float(np.mean(clip_mask)),
            "clip_low_rate": float(np.mean(ax_unclipped < ax_min)),
            "clip_high_rate": float(np.mean(ax_unclipped > ax_max)),
            "mean_abs_clip_delta": float(np.mean(np.abs(ax_unclipped - ax_clipped))),
        },
        "braking": {
            "hard_brake_threshold": -3.0,
            "hard_brake_ratio": float(np.mean(ax_clipped < -3.0)),
            "strong_brake_threshold": -1.5,
            "strong_brake_ratio": float(np.mean(ax_clipped < -1.5)),
        },
        "history": _history_gap_delta_v(arrays, idx),
        "risk_bins": {
            "edges": edges,
            "rows": bin_rows,
            "risk_action_signal": _risk_action_signal(bin_rows),
        },
    }

    summary_path = result_dir / "action_dataset_inspection_summary.json"
    csv_path = result_dir / "action_dataset_risk_bins.csv"
    save_json(summary, summary_path)
    _write_bin_csv(csv_path, bin_rows)
    logger.info("Wrote %s", summary_path)
    logger.info("Wrote %s", csv_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    parser.add_argument(
        "--dataset-path",
        default=DEFAULT_DATASET_PATH,
        help="Raw dataset.npz path. Relative paths are resolved from cwd if present, otherwise from config output_dir.",
    )
    parser.add_argument(
        "--schema-path",
        default=DEFAULT_SCHEMA_PATH,
        help="feature_schema.json path. Relative paths are resolved from cwd if present, otherwise from config output_dir.",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory for inspection JSON/CSV; defaults to config output_dir.")
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default=DEFAULT_SPLIT, help="Dataset split to inspect.")
    parser.add_argument(
        "--risk-bins",
        default=DEFAULT_RISK_BINS,
        help="Comma-separated risk percentile edges.",
    )
    parser.add_argument("--log-level", default=DEFAULT_LOG_LEVEL, help="Logging level.")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg_path = Path(args.config).resolve()
    inspect_dataset(load_yaml(cfg_path), cfg_path.parent, args)


if __name__ == "__main__":
    main()

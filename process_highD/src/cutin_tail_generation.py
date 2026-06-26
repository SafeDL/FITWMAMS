"""Cut-in EVT tail context generation and visualization."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from diffusion.src.features import CUTIN_SCENARIO_CONDITION_KEYS
from process_highD.src.io_utils import load_config
from tools.evt import load_evt_model
from tools.highd_cutin import load_highd_cutin_event_context_cache
from tools.io import write_json
from tools.plot_style import (
    GENERATED_COLOR,
    REAL_COLOR,
    SAMPLED_COLOR,
    get_pyplot,
    label_for,
    style_axes,
)


logger = logging.getLogger(__name__)

SOURCE_EMPIRICAL_TAIL = "highd_evt_independent_tail_peak"
SOURCE_COPULA_CONTEXT = "highd_evt_gaussian_copula_context"
CONDITION_KEYS: tuple[str, ...] = CUTIN_SCENARIO_CONDITION_KEYS
TAIL_FEATURE_NAMES: tuple[str, ...] = (
    "ego_vx_0",
    "log_initial_gap",
    "initial_lateral_offset",
    "initial_delta_vx",
    "target_ax_0",
    "target_vy_0",
    "target_ay_0",
    "final_lateral_offset",
    "time_to_cross",
    "target_speed_change",
)
INTRINSIC_TRAJECTORY_METRIC_NAMES: tuple[str, ...] = (
    "lane_entry_time",
    "longitudinal_displacement",
    "total_lateral_displacement",
    "lateral_progress_toward_ego_lane",
    "final_abs_lateral_offset",
    "target_speed_change",
    "max_abs_longitudinal_accel",
    "max_abs_lateral_velocity",
    "mean_abs_lateral_accel",
)
LANE_CHANGE_RATE_NAMES: tuple[str, ...] = (
    "lane_entry_rate",
    "post_entry_retention_rate",
    "completed_lane_change_rate",
)
VARIABLE_EPS = 1.0e-8


def _paper_metric_label(name: str) -> str:
    labels = {
        "lane_entry_time": "Lane-entry time\n" + label_for(name),
        "longitudinal_displacement": "Longitudinal displacement\n" + label_for(name),
        "total_lateral_displacement": "Lateral lane-change displacement\n" + label_for(name),
        "lateral_progress_toward_ego_lane": "Lateral progress toward ego lane\n" + label_for(name),
        "final_abs_lateral_offset": "Final absolute lateral offset\n" + label_for(name),
        "target_speed_change": "Target speed change\n" + label_for(name),
        "max_abs_longitudinal_accel": "Maximum absolute longitudinal acceleration\n" + label_for(name),
        "max_abs_lateral_velocity": "Maximum absolute lateral velocity\n" + label_for(name),
        "mean_abs_lateral_accel": "Mean absolute lateral acceleration\n" + label_for(name),
        "ego_vx_0": "Initial ego speed\n" + label_for(name),
        "log_initial_gap": "Initial gap on log scale\n" + label_for(name),
        "initial_lateral_offset": "Initial lateral offset\n" + label_for(name),
        "initial_delta_vx": "Initial relative longitudinal speed\n" + label_for(name),
        "target_ax_0": "Initial target longitudinal acceleration\n" + label_for(name),
        "target_vy_0": "Initial target lateral velocity\n" + label_for(name),
        "target_ay_0": "Initial target lateral acceleration\n" + label_for(name),
        "final_lateral_offset": "Final lateral offset\n" + label_for(name),
        "time_to_cross": "Time to lane crossing\n" + label_for(name),
    }
    return labels.get(str(name), label_for(name))


def _path(config: dict[str, Any], key: str) -> Path:
    return Path(config[key]).resolve()


def _read_evt_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"EVT summary not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_semantic_contexts(path: Path) -> list[dict[str, Any]]:
    rows = load_highd_cutin_event_context_cache(path)
    if not rows:
        raise RuntimeError(f"Cut-in context cache is empty: {path}")
    invalid = [
        str(row.get("event_id", idx))
        for idx, row in enumerate(rows)
        if float(row.get("is_cutin", 0.0)) < 0.5
    ]
    if invalid:
        raise RuntimeError(
            "Cut-in context cache contains non semantic cut-in rows: "
            f"{invalid[:10]} (total={len(invalid)})"
        )
    return rows


def _score_rows_with_evt(
    rows: list[dict[str, Any]],
    *,
    model_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    model = load_evt_model(model_path)
    summary = _read_evt_summary(summary_path)
    values = np.asarray([row["y_cutin"] for row in rows], dtype=np.float64)
    scores = np.asarray(model.score(values), dtype=np.float64)
    survival = np.asarray(model.survival(values), dtype=np.float64)
    for idx, row in enumerate(rows):
        row["risk_score"] = float(scores[idx])
        row["evt_tail_probability"] = float(survival[idx])
    return {
        "evt_model_path": str(model_path),
        "evt_summary_path": str(summary_path),
        "evt_tail_threshold_u": float(model.u),
        "evt_tail_threshold_score": float(model.score(float(model.u))),
        "evt_exceedance_rate": float(model.exceedance_rate),
        "collision_critical_level": float(summary["collision_critical_level"]),
        "collision_critical_level_mode": summary.get("collision_critical_level_mode"),
        "human_calibrated_safety_threshold": summary.get(
            "human_calibrated_safety_threshold"
        ),
        "risk_value_key": "y_cutin",
    }


def _select_independent_tail_peaks(
    rows: list[dict[str, Any]],
    peaks_path: Path,
) -> list[dict[str, Any]]:
    if not peaks_path.exists():
        raise FileNotFoundError(
            "Independent cut-in tail peaks not found: "
            f"{peaks_path}. Run process_highD/scripts/estimate_cutin_exposure.py first."
        )
    peaks = pd.read_csv(peaks_path)
    required = {"representative_event_id", "peak_id", "y_cutin_max"}
    missing = sorted(required - set(peaks.columns))
    if missing:
        raise KeyError(f"{peaks_path} is missing required columns: {missing}")

    rows_by_event = {str(row["event_id"]): row for row in rows}
    selected: list[dict[str, Any]] = []
    missing_events: list[str] = []
    for _, peak in peaks.iterrows():
        event_id = str(peak["representative_event_id"])
        base = rows_by_event.get(event_id)
        if base is None:
            missing_events.append(event_id)
            continue
        item = dict(base)
        for key, value in peak.to_dict().items():
            if hasattr(value, "item"):
                value = value.item()
            item[key] = value
        item["source_type"] = SOURCE_EMPIRICAL_TAIL
        item["base_event_id"] = str(item["event_id"])
        item["base_context_index"] = -1
        item["synthetic_context"] = 0
        selected.append(item)
    if missing_events:
        raise KeyError(
            "Independent cut-in peaks reference events missing from context cache: "
            f"{missing_events[:10]} (total={len(missing_events)})"
        )
    if not selected:
        raise RuntimeError(f"No independent cut-in tail peaks matched {peaks_path}")
    return selected


def _tail_feature(row: dict[str, Any]) -> np.ndarray:
    cond = np.asarray(row["scenario_conditions"], dtype=np.float64)
    gap = max(float(cond[1]), 0.2)
    return np.asarray(
        [
            float(cond[0]),
            np.log(gap),
            float(cond[2]),
            float(cond[3]),
            float(cond[4]),
            float(cond[5]),
            float(cond[6]),
            float(cond[7]),
            float(cond[8]),
            float(cond[9]),
        ],
        dtype=np.float64,
    )


def _feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.stack([_tail_feature(row) for row in rows], axis=0)


def _nearest_base(features: np.ndarray, target: np.ndarray) -> int:
    center = np.median(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    z = (features - center) / scale
    target_z = (target - center) / scale
    return int(np.argmin(np.sum((z - target_z[None, :]) ** 2, axis=1)))


def _reconstruct_cutin_state(
    base_row: dict[str, Any],
    feature: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(base_row["initial_states"], dtype=np.float32).copy()
    ego_length = float(base_row["ego_length"])
    target_length = float(base_row["adv_length"])
    ego_vx = max(float(feature[0]), 0.0)
    gap = float(np.exp(float(feature[1])))
    lateral_offset = float(feature[2])
    delta_vx = float(feature[3])
    target_vx = max(ego_vx - delta_vx, 0.0)

    states[0, 2] = np.float32(ego_vx)
    states[1, 0] = np.float32(states[0, 0] + 0.5 * (ego_length + target_length) + gap)
    states[1, 1] = np.float32(states[0, 1] + lateral_offset)
    states[1, 2] = np.float32(target_vx)
    states[1, 4] = np.float32(np.clip(float(feature[4]), -8.0, 4.0))
    states[1, 3] = np.float32(float(feature[5]))
    states[1, 5] = np.float32(np.clip(float(feature[6]), -4.0, 4.0))

    conditions = np.asarray(
        [
            ego_vx,
            gap,
            lateral_offset,
            delta_vx,
            float(feature[4]),
            float(feature[5]),
            float(feature[6]),
            float(feature[7]),
            float(feature[8]),
            float(feature[9]),
        ],
        dtype=np.float32,
    )
    return conditions, states.astype(np.float32)


def _compute_intrinsic_trajectory_metrics(
    target_trajectory: np.ndarray,
    initial_states: np.ndarray,
    *,
    dt: float,
    lane_threshold: float = 1.0,
    post_cutin_window_seconds: float = 3.0,
) -> dict[str, np.ndarray]:
    """Compute lane-change trajectory metrics intrinsic to the target vehicle.

    These metrics characterise the lane-change manoeuvre itself (lateral motion,
    longitudinal speed profile, trajectory smoothness) and do NOT depend on ego
    interaction behaviour.  They are suitable for evaluating open-loop generated
    trajectories against real-world tail events.

    Parameters
    ----------
    target_trajectory: [B, H, 6]  future states of the target vehicle (t=1..H).
    initial_states:    [B, 2, 6]  initial ego + target states at t=0.
    dt:                seconds per step.
    """
    target_vy = target_trajectory[:, :, 3]
    target_vx = target_trajectory[:, :, 2]
    target_ay = target_trajectory[:, :, 5]
    batch = int(target_trajectory.shape[0])
    horizon = int(target_trajectory.shape[1])

    ego_y = initial_states[:, 0, 1]
    initial_lateral = initial_states[:, 1, 1] - ego_y
    relative_lateral = target_trajectory[:, :, 1] - ego_y[:, None]
    abs_lateral = np.abs(relative_lateral)
    initial_abs_lateral = np.abs(initial_lateral)
    lane_entry = abs_lateral <= float(lane_threshold)
    has_lane_entry = np.any(lane_entry, axis=1)
    first_lane_entry = np.argmax(lane_entry, axis=1)
    lane_entry_time = np.full(batch, np.nan, dtype=np.float64)
    lane_entry_time[has_lane_entry] = (
        first_lane_entry[has_lane_entry].astype(np.float64) + 1.0
    ) * float(dt)
    post_steps = max(
        1,
        int(np.ceil(float(post_cutin_window_seconds) / max(float(dt), 1.0e-6))),
    )
    post_entry_retention = np.zeros(batch, dtype=bool)
    for idx in np.flatnonzero(has_lane_entry):
        start = int(first_lane_entry[idx])
        stop = min(horizon, start + post_steps)
        post_entry_retention[idx] = bool(
            np.max(abs_lateral[idx, start:stop]) <= float(lane_threshold)
        )
    abs_with_initial = np.concatenate(
        [initial_abs_lateral[:, None], abs_lateral],
        axis=1,
    )
    approach_speed = np.maximum(
        -np.diff(abs_with_initial, axis=1) / max(float(dt), 1.0e-6),
        0.0,
    )
    has_approach = np.ones(batch, dtype=bool)
    for idx in range(batch):
        if initial_abs_lateral[idx] > float(lane_threshold):
            stop = int(first_lane_entry[idx]) + 1 if has_lane_entry[idx] else horizon
            has_approach[idx] = bool(
                stop > 0 and float(np.max(approach_speed[idx, :stop])) >= 0.05
            )
    completed_lane_change = has_lane_entry & post_entry_retention & has_approach

    return {
        "lane_entry_time": lane_entry_time,
        "lateral_progress_toward_ego_lane": (
            initial_abs_lateral - abs_lateral[:, -1]
        ),
        "longitudinal_displacement": (
            target_trajectory[:, -1, 0] - initial_states[:, 1, 0]
        ),
        "total_lateral_displacement": np.abs(
            target_trajectory[:, -1, 1] - initial_states[:, 1, 1]
        ),
        "minimum_abs_lateral_offset": np.min(abs_lateral, axis=1),
        "final_abs_lateral_offset": abs_lateral[:, -1],
        "max_abs_lateral_velocity": np.max(np.abs(target_vy), axis=1),
        "max_abs_longitudinal_accel": np.max(
            np.abs(target_trajectory[:, :, 4]), axis=1
        ),
        "mean_abs_lateral_accel": np.mean(np.abs(target_ay), axis=1),
        "target_speed_change": target_vx[:, -1] - initial_states[:, 1, 2],
        "lane_entry_rate": has_lane_entry.astype(np.float64),
        "post_entry_retention_rate": post_entry_retention.astype(np.float64),
        "completed_lane_change_rate": completed_lane_change.astype(np.float64),
    }


def _normal_score_pseudo_observations(features: np.ndarray) -> np.ndarray:
    from scipy.special import ndtri

    ranks = np.empty_like(features, dtype=np.float64)
    n = int(features.shape[0])
    for col in range(int(features.shape[1])):
        order = np.argsort(features[:, col], kind="mergesort")
        ranks[order, col] = np.arange(1, n + 1, dtype=np.float64)
    u = ranks / float(n + 1)
    return ndtri(np.clip(u, 1.0e-6, 1.0 - 1.0e-6))


def _conditions_to_tail_features(conditions: np.ndarray) -> np.ndarray:
    features = np.asarray(conditions, dtype=np.float64).copy()
    if features.ndim != 2 or features.shape[1] != len(TAIL_FEATURE_NAMES):
        raise ValueError(
            "scenario conditions must have shape [N, "
            f"{len(TAIL_FEATURE_NAMES)}], got {tuple(features.shape)}"
        )
    features[:, 1] = np.log(np.maximum(features[:, 1], 0.2))
    return features


def _plot_condition_distribution_comparison(
    empirical_conditions: np.ndarray,
    generated_input_conditions: np.ndarray,
    generated_realized_conditions: np.ndarray,
    *,
    histogram_path: Path,
) -> dict[str, Any]:
    empirical_features = _conditions_to_tail_features(empirical_conditions)
    input_features = _conditions_to_tail_features(generated_input_conditions)
    realized_features = _conditions_to_tail_features(generated_realized_conditions)
    variable = _variable_mask(empirical_features)
    names = tuple(
        name for name, keep in zip(TAIL_FEATURE_NAMES, variable) if bool(keep)
    )
    empirical_variable = empirical_features[:, variable]
    input_variable = input_features[:, variable]
    realized_variable = realized_features[:, variable]
    plt = _matplotlib()
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.4, rows * 3.15))
    flat_axes = np.asarray(axes).reshape(-1)
    metrics: dict[str, Any] = {}
    for idx, name in enumerate(names):
        ax = flat_axes[idx]
        real = empirical_variable[:, idx]
        sampled = input_variable[:, idx]
        realized = realized_variable[:, idx]
        joined = np.concatenate(
            [
                real[np.isfinite(real)],
                sampled[np.isfinite(sampled)],
                realized[np.isfinite(realized)],
            ]
        )
        metrics[name] = {
            "sampled_input_vs_evt_tail": _distribution_metrics(real, sampled),
            "diffusion_realized_vs_evt_tail": _distribution_metrics(real, realized),
        }
        if joined.size:
            lo, hi = np.percentile(joined, [0.5, 99.5])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1.0e-9:
                lo, hi = float(np.min(joined)), float(np.max(joined))
            bins = np.linspace(lo, hi, 34) if hi > lo + 1.0e-9 else 30
            ax.hist(
                real,
                bins=bins,
                density=True,
                alpha=0.58,
                color=REAL_COLOR,
                label="EVT tail",
            )
            ax.hist(
                sampled,
                bins=bins,
                density=True,
                alpha=0.46,
                color=SAMPLED_COLOR,
                label="Copula input",
            )
            ax.hist(
                realized,
                bins=bins,
                density=True,
                alpha=0.42,
                color=GENERATED_COLOR,
                label="Diffusion",
            )
            if hi > lo + 1.0e-9:
                ax.set_xlim(lo, hi)
        ax.set_title(_paper_metric_label(name))
        ax.set_ylabel("Density")
        style_axes(ax)
    for ax in flat_axes[len(names) :]:
        ax.axis("off")
    flat_axes[0].legend(frameon=False)
    fig.tight_layout()
    histogram_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(histogram_path, dpi=180)
    plt.close(fig)
    return {
        "variable_feature_names": list(names),
        "feature_histograms": metrics,
        "figure_interpretation": (
            "Each panel compares one tail-feature marginal: empirical EVT-tail "
            "events, Gaussian-copula sampled diffusion inputs, and conditions "
            "realized after integrating diffusion-generated trajectories."
        ),
    }


def _variable_mask(features: np.ndarray) -> np.ndarray:
    return np.std(np.asarray(features, dtype=np.float64), axis=0) > VARIABLE_EPS


def _fit_gaussian_copula(
    features: np.ndarray,
    *,
    regularization: float,
) -> np.ndarray:
    if features.shape[1] == 0:
        raise RuntimeError("Gaussian copula has no variable feature dimensions")
    z = _normal_score_pseudo_observations(features)
    corr = np.corrcoef(z, rowvar=False)
    corr = np.nan_to_num(np.atleast_2d(corr), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    corr += np.eye(corr.shape[0], dtype=np.float64) * max(float(regularization), 0.0)
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1.0e-8, None)
    corr = (eigvecs * eigvals[None, :]) @ eigvecs.T
    diag = np.sqrt(np.clip(np.diag(corr), 1.0e-12, None))
    corr = corr / diag[:, None] / diag[None, :]
    np.fill_diagonal(corr, 1.0)
    return corr


def _sample_condition_distribution(
    tail_rows: list[dict[str, Any]],
    *,
    count: int,
    rng: np.random.Generator,
    clip_quantile: float,
    regularization: float,
    start_index: int = 0,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    if len(tail_rows) < 2:
        raise RuntimeError("Gaussian copula requires at least two tail contexts")
    features = _feature_matrix(tail_rows)
    variable = _variable_mask(features)
    variable_features = features[:, variable]
    logger.info(
        "Fitting Gaussian copula on %d tail events × %d variable features …",
        len(tail_rows),
        variable_features.shape[1],
    )
    corr_variable = _fit_gaussian_copula(
        variable_features,
        regularization=regularization,
    )
    logger.info("Copula fitted — sampling %d synthetic conditions …", int(count))
    corr = np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
    corr[np.ix_(variable, variable)] = corr_variable
    np.fill_diagonal(corr, 1.0)

    from scipy.special import ndtr

    q = min(max(float(clip_quantile), 0.0), 0.49)
    lower = np.quantile(variable_features, q, axis=0)
    upper = np.quantile(variable_features, 1.0 - q, axis=0)
    time_to_cross_idx = TAIL_FEATURE_NAMES.index("time_to_cross")
    time_to_cross_support = np.unique(features[:, time_to_cross_idx])
    sampled_z = rng.multivariate_normal(
        np.zeros(variable_features.shape[1], dtype=np.float64),
        corr_variable,
        size=int(count),
        check_valid="ignore",
    )
    sampled_u = np.clip(ndtr(sampled_z), 1.0e-6, 1.0 - 1.0e-6)

    out: list[dict[str, Any]] = []
    center = np.median(features, axis=0)
    for idx in range(int(count)):
        absolute_idx = int(start_index) + int(idx)
        variable_target = np.asarray(
            [
                np.quantile(variable_features[:, col], sampled_u[idx, col])
                for col in range(variable_features.shape[1])
            ],
            dtype=np.float64,
        )
        variable_target = np.clip(variable_target, lower, upper)
        target_feature = center.copy()
        target_feature[variable] = variable_target
        # Enforce cut-in semantic: target must end in ego's lane.
        target_feature[TAIL_FEATURE_NAMES.index("final_lateral_offset")] = np.clip(
            target_feature[TAIL_FEATURE_NAMES.index("final_lateral_offset")],
            -1.0,
            1.0,
        )
        target_feature[time_to_cross_idx] = float(
            time_to_cross_support[
                int(np.argmin(np.abs(time_to_cross_support - target_feature[time_to_cross_idx])))
            ]
        )
        base_idx = _nearest_base(features, target_feature)
        base = tail_rows[base_idx]
        conditions, states = _reconstruct_cutin_state(base, target_feature)
        item = dict(base)
        item["scenario_conditions"] = conditions
        item["initial_states"] = states
        item["source_type"] = SOURCE_COPULA_CONTEXT
        item["event_id"] = f"cutin_copula_{absolute_idx:05d}_base_{base['event_id']}"
        item["base_event_id"] = str(base["event_id"])
        item["base_context_index"] = base_idx
        item["synthetic_context"] = 1
        item["initial_gap"] = float(conditions[1])
        item["initial_closing_speed"] = float(conditions[3])
        item["y_cutin"] = np.nan
        item["risk_score"] = np.nan
        item["evt_tail_probability"] = np.nan
        out.append(item)
    return out, corr, variable


def _save_condition_distribution(
    path: Path,
    *,
    empirical_rows: list[dict[str, Any]],
    corr: np.ndarray,
    variable_mask: np.ndarray,
    evt_meta: dict[str, Any],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    features = _feature_matrix(empirical_rows)
    payload: dict[str, np.ndarray] = {
        "scenario_conditions": np.asarray(
            [row["scenario_conditions"] for row in empirical_rows],
            dtype=np.float32,
        ),
        "tail_features": features.astype(np.float32),
        "condition_keys": np.asarray(CONDITION_KEYS, dtype=object),
        "tail_feature_names": np.asarray(TAIL_FEATURE_NAMES, dtype=object),
        "copula_correlation": corr.astype(np.float32),
        "copula_variable_mask": np.asarray(variable_mask, dtype=bool),
        "copula_marginal_values": features.astype(np.float32),
        "source_event_id": np.asarray(
            [row["event_id"] for row in empirical_rows],
            dtype=object,
        ),
        "event_id": np.asarray(
            [row["event_id"] for row in empirical_rows],
            dtype=object,
        ),
        "recording_id": np.asarray(
            [row["recording_id"] for row in empirical_rows],
            dtype=np.int32,
        ),
        "synthetic_context": np.zeros(len(empirical_rows), dtype=np.int8),
        "source_type": np.asarray(
            [SOURCE_EMPIRICAL_TAIL for _ in empirical_rows],
            dtype=object,
        ),
        "source_peak_id": np.asarray(
            [row["peak_id"] for row in empirical_rows],
            dtype=object,
        ),
        "copula_marginal_clip_quantile": np.asarray(
            float(config["copula_marginal_clip_quantile"]),
            dtype=np.float32,
        ),
        "copula_correlation_regularization": np.asarray(
            float(config["copula_correlation_regularization"]),
            dtype=np.float32,
        ),
        "tail_threshold": np.asarray(evt_meta["evt_tail_threshold_u"], dtype=np.float32),
        "collision_critical_level": np.asarray(
            evt_meta["collision_critical_level"],
            dtype=np.float32,
        ),
    }
    np.savez_compressed(path, **payload)


def _context_risk_start_index(row: dict[str, Any], *, dt: float) -> int:
    conditions = np.asarray(row["scenario_conditions"], dtype=np.float32).reshape(-1)
    if conditions.size >= len(CONDITION_KEYS):
        time_to_cross = float(conditions[CONDITION_KEYS.index("time_to_cross")])
        if np.isfinite(time_to_cross):
            return max(0, int(round(time_to_cross / max(float(dt), 1.0e-6))) - 1)
    value = row.get("risk_start_index", None)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0
    if np.isfinite(numeric):
        return max(0, int(round(numeric)))
    return 0


def _save_tail_contexts(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    evt_meta: dict[str, Any],
    dt: float,
) -> None:
    if not rows:
        raise RuntimeError("Cannot save empty cut-in tail context set")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "scenario_conditions": np.asarray(
            [row["scenario_conditions"] for row in rows],
            dtype=np.float32,
        ),
        "initial_states": np.asarray(
            [row["initial_states"] for row in rows],
            dtype=np.float32,
        ),
        "ego_length": np.asarray(
            [float(row["ego_length"]) for row in rows],
            dtype=np.float32,
        ),
        "adv_length": np.asarray(
            [float(row["adv_length"]) for row in rows],
            dtype=np.float32,
        ),
        "source_type": np.asarray(
            [row.get("source_type", "") for row in rows],
            dtype=object,
        ),
        "event_id": np.asarray([row.get("event_id", "") for row in rows], dtype=object),
        "base_event_id": np.asarray(
            [row.get("base_event_id", row.get("event_id", "")) for row in rows],
            dtype=object,
        ),
        "recording_id": np.asarray(
            [int(row.get("recording_id", -1)) for row in rows],
            dtype=np.int32,
        ),
        "ego_id": np.asarray([int(row.get("ego_id", -1)) for row in rows], dtype=np.int32),
        "target_id": np.asarray(
            [int(row.get("target_id", -1)) for row in rows],
            dtype=np.int32,
        ),
        "anchor_frame": np.asarray(
            [int(row.get("anchor_frame", 0)) for row in rows],
            dtype=np.int64,
        ),
        "cross_frame": np.asarray(
            [int(row.get("cross_frame", row.get("anchor_frame", 0))) for row in rows],
            dtype=np.int64,
        ),
        "cutin_start_frame": np.asarray(
            [int(row.get("cutin_start_frame", row.get("anchor_frame", 0))) for row in rows],
            dtype=np.int64,
        ),
        "cutin_end_frame": np.asarray(
            [int(row.get("cutin_end_frame", row.get("cross_frame", 0))) for row in rows],
            dtype=np.int64,
        ),
        "risk_start_index": np.asarray(
            [_context_risk_start_index(row, dt=dt) for row in rows],
            dtype=np.int64,
        ),
        "synthetic_context": np.asarray(
            [int(row.get("synthetic_context", 0)) for row in rows],
            dtype=np.int8,
        ),
        "base_context_index": np.asarray(
            [int(row.get("base_context_index", -1)) for row in rows],
            dtype=np.int32,
        ),
        "context_feature_distance": np.asarray(
            [float(row.get("context_feature_distance", np.nan)) for row in rows],
            dtype=np.float32,
        ),
        "tail_threshold": np.asarray(evt_meta["evt_tail_threshold_u"], dtype=np.float32),
        "collision_critical_level": np.asarray(
            evt_meta["collision_critical_level"],
            dtype=np.float32,
        ),
        "condition_keys": np.asarray(CONDITION_KEYS, dtype=object),
    }
    optional_float_keys = (
        "initial_gap",
        "initial_closing_speed",
        "y_cutin",
        "y_long",
        "risk_score",
        "evt_tail_probability",
        "post_cutin_min_gap",
        "post_cutin_min_ttc",
        "min_abs_lateral_offset",
        "max_abs_lateral_velocity",
        "is_front_cutin",
    )
    for key in optional_float_keys:
        if any(key in row for row in rows):
            payload[key] = np.asarray(
                [float(row.get(key, np.nan)) for row in rows],
                dtype=np.float32,
            )
    np.savez_compressed(path, **payload)


def _matplotlib() -> Any:
    return get_pyplot()


def _clean_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    return left, right


def _distribution_metrics(real: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    from scipy.stats import ks_2samp, wasserstein_distance

    a, b = _clean_pair(real, generated)
    if a.size == 0 or b.size == 0:
        return {
            "real_mean": float("nan"),
            "generated_mean": float("nan"),
            "mean_delta": float("nan"),
            "ks_statistic": float("nan"),
            "wasserstein": float("nan"),
        }
    ks = ks_2samp(a, b)
    return {
        "real_mean": float(np.mean(a)),
        "generated_mean": float(np.mean(b)),
        "mean_delta": float(np.mean(b) - np.mean(a)),
        "real_std": float(np.std(a)),
        "generated_std": float(np.std(b)),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "wasserstein": float(wasserstein_distance(a, b)),
    }


def _plot_hist_grid(
    real: np.ndarray,
    generated: np.ndarray,
    names: tuple[str, ...],
    path: Path,
    *,
    real_label: str,
    generated_label: str,
) -> dict[str, dict[str, float]]:
    plt = _matplotlib()
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.1))
    flat_axes = np.asarray(axes).reshape(-1)
    metrics: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(names):
        ax = flat_axes[idx]
        a, b = _clean_pair(real[:, idx], generated[:, idx])
        metrics[name] = _distribution_metrics(a, b)
        if a.size and b.size:
            joined = np.concatenate([a, b])
            lo, hi = np.percentile(joined, [0.5, 99.5])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1.0e-9:
                lo, hi = float(np.min(joined)), float(np.max(joined))
            bins = np.linspace(lo, hi, 36) if hi > lo + 1.0e-9 else 30
            ax.hist(
                a,
                bins=bins,
                density=True,
                alpha=0.58,
                color=REAL_COLOR,
                label=real_label,
            )
            ax.hist(
                b,
                bins=bins,
                density=True,
                alpha=0.48,
                color=GENERATED_COLOR,
                label=generated_label,
            )
            if hi > lo + 1.0e-9:
                ax.set_xlim(lo, hi)
        ax.set_title(_paper_metric_label(name))
        ax.set_ylabel("Density")
        style_axes(ax)
    for ax in flat_axes[len(names) :]:
        ax.axis("off")
    flat_axes[0].legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return metrics


def _generate_diffusion_scenarios(
    sampled_rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    refill_sampler: Callable[[int], list[dict[str, Any]]] | None = None,
) -> tuple[Path, dict[str, Any]]:
    import torch

    from diffusion.src.kinematics import integrate_cutin_acceleration_actions
    from diffusion.src.utils import set_seed
    from tools.diffusion_adapter import DiffusionPriorAdapter
    from tools.normalization import denormalize_torch, normalize_numpy

    requested = int(config["num_diffusion_scenarios"])
    if requested <= 0:
        raise ValueError("num_diffusion_scenarios must be positive")
    if not sampled_rows:
        raise ValueError("At least one sampled cut-in condition is required")

    output_path = _path(config, "generated_scenarios_path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    natural_dir = _path(config, "diffusion_dataset_dir")
    checkpoint = Path(config["diffusion_checkpoint_path"])
    set_seed(int(config["diffusion_seed"]))
    adapter = DiffusionPriorAdapter.load(
        natural_dir,
        checkpoint,
        device=str(config["diffusion_device"]),
    )
    diffusion_cfg = load_config(_path(config, "diffusion_config_path"))
    _validate_diffusion_alignment(
        adapter=adapter,
        diffusion_config=diffusion_cfg,
        inference_steps=int(config["diffusion_inference_steps"]),
    )
    x0_clip_abs = float(diffusion_cfg.get("diffusion", {}).get("x0_clip_abs", 0.0))
    adapter.model.denoiser.cfg.x0_clip_abs = x0_clip_abs
    schema = adapter.schema

    batch_size = max(int(config["diffusion_batch_size"]), 1)
    inference_steps = int(config["diffusion_inference_steps"])
    guidance_scale = float(config.get("diffusion_guidance_scale", 0.0))
    rejection_cfg = dict(config.get("diffusion_rejection", {}))
    rejection_enabled = bool(rejection_cfg.get("enabled", False))
    enforce_acceptance = bool(rejection_cfg.get("enforce_acceptance", False))
    refill_batch_size = max(
        int(rejection_cfg.get("refill_condition_batch_size", requested)),
        1,
    )
    max_refill_rounds = max(
        int(rejection_cfg.get("max_refill_rounds", 20)),
        0,
    )
    selected = list(sampled_rows[: max(requested, len(sampled_rows))])
    if requested > len(selected):
        raise ValueError(
            "num_diffusion_scenarios cannot exceed num_condition_samples in the "
            "cut-in tail pipeline unless a refill_sampler is provided"
        )

    action_stats = adapter.stats["actions"]
    action_mean = torch.tensor(
        action_stats["mean"],
        dtype=torch.float32,
        device=adapter.device,
    ).view(1, 1, -1)
    action_std = torch.tensor(
        action_stats["std"],
        dtype=torch.float32,
        device=adapter.device,
    ).view(1, 1, -1)
    guidance_config = _cutin_guidance_config(adapter.config, config)

    accepted_conditions: list[np.ndarray] = []
    accepted_initial_states: list[np.ndarray] = []
    accepted_actions: list[np.ndarray] = []
    accepted_trajectories: list[np.ndarray] = []
    accepted_ego_lengths: list[np.ndarray] = []
    accepted_adv_lengths: list[np.ndarray] = []
    accepted_base_event_ids: list[np.ndarray] = []
    accepted_source_types: list[np.ndarray] = []
    generated_total = 0
    semantic_total = 0
    overlap_total = 0
    starts_outside_total = 0
    lateral_progress_total = 0
    post_remain_total = 0
    condition_error_totals = {
        "final_lateral_offset": 0.0,
        "target_speed_change": 0.0,
    }
    log_interval = max(1, int(np.ceil(len(selected) / batch_size)) // 20)
    logger.info(
        "Generating %d scenarios (batch_size=%d, initial_conditions=%d, "
        "guidance=%.3f, semantic_postprocess=%s, enforce_acceptance=%s) …",
        requested,
        batch_size,
        len(selected),
        guidance_scale,
        "on" if rejection_enabled else "off",
        "on" if enforce_acceptance else "off",
    )
    rows_cursor = 0
    batch_idx = 0
    refill_rounds = 0
    adapter.model.eval()
    with torch.no_grad():
        while sum(chunk.shape[0] for chunk in accepted_actions) < requested:
            if rows_cursor >= len(selected):
                can_refill = (
                    rejection_enabled
                    and enforce_acceptance
                    and refill_sampler is not None
                    and refill_rounds < max_refill_rounds
                )
                if not can_refill:
                    break
                accepted_so_far = sum(chunk.shape[0] for chunk in accepted_actions)
                refill_count = max(refill_batch_size, requested - accepted_so_far)
                new_rows = refill_sampler(refill_count)
                refill_rounds += 1
                if not new_rows:
                    break
                selected.extend(new_rows)
                logger.info(
                    "Refilled %d cut-in conditions after semantic filtering "
                    "(round=%d/%d, accepted=%d/%d)",
                    len(new_rows),
                    refill_rounds,
                    max_refill_rounds,
                    accepted_so_far,
                    requested,
                )
                continue
            start = rows_cursor
            end = min(start + batch_size, len(selected))
            rows_cursor = end
            if start >= end:
                break
            batch_rows = selected[start:end]
            conditions = np.asarray(
                [row["scenario_conditions"] for row in batch_rows],
                dtype=np.float32,
            )
            initial_states = np.asarray(
                [row["initial_states"] for row in batch_rows],
                dtype=np.float32,
            )
            normalized_conditions = normalize_numpy(
                conditions,
                adapter.stats,
                "scenario_conditions",
            )
            cond = torch.from_numpy(normalized_conditions).float().to(
                adapter.device
            )
            raw_cond = torch.from_numpy(conditions).float().to(adapter.device)
            init_tensor = torch.from_numpy(initial_states).float().to(adapter.device)
            ego_lengths_batch = np.asarray(
                [row["ego_length"] for row in batch_rows],
                dtype=np.float64,
            )
            adv_lengths_batch = np.asarray(
                [row["adv_length"] for row in batch_rows],
                dtype=np.float64,
            )
            if guidance_scale > 0.0:
                sample = adapter.model.sample_ddim_with_guidance(
                    int(end - start),
                    cond,
                    inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    guidance_context={
                        "scenario_conditions": raw_cond,
                        "initial_states": init_tensor,
                        "action_mean": action_mean,
                        "action_std": action_std,
                        "ego_length": torch.from_numpy(ego_lengths_batch.astype(np.float32)).to(adapter.device),
                        "adv_length": torch.from_numpy(adv_lengths_batch.astype(np.float32)).to(adapter.device),
                    },
                    guidance_config=guidance_config,
                )
            else:
                sample = adapter.model.sample_ddim(
                    int(end - start),
                    cond,
                    inference_steps=inference_steps,
                )
            decoded = denormalize_torch(sample, adapter.stats, "actions")
            action_batch = decoded.detach().cpu().numpy().astype(np.float32)
            action_cfg = adapter.config["action"]
            projection_cfg = adapter.config.get("trajectory_projection", {})
            trajectories = integrate_cutin_acceleration_actions(
                initial_states,
                action_batch,
                float(schema["dt"]),
                ax_min=float(action_cfg["ax_min"]),
                ax_max=float(action_cfg["ax_max"]),
                ay_abs_max=float(action_cfg["ay_abs_max"]),
                speed_min=float(projection_cfg.get("speed_min", 0.0)),
                speed_max=float(projection_cfg.get("speed_max", 50.0)),
            )
            masks = _semantic_cutin_mask(
                target_trajectory=trajectories,
                initial_states=initial_states,
                dt=float(schema["dt"]),
                config=config,
            )
            generated_total += int(len(batch_rows))
            semantic_total += int(np.sum(masks["semantic_cutin"]))
            overlap_total += int(np.sum(masks["has_overlap"]))
            starts_outside_total += int(np.sum(masks["starts_outside_ego_lane"]))
            lateral_progress_total += int(np.sum(masks["has_lateral_progress"]))
            post_remain_total += int(np.sum(masks["post_remain"]))
            realized_batch = _realized_conditions_from_arrays(
                conditions,
                initial_states,
                trajectories,
                dt=float(schema["dt"]),
            )
            for key in condition_error_totals:
                col = CONDITION_KEYS.index(key)
                condition_error_totals[key] += float(
                    np.sum(np.abs(realized_batch[:, col] - conditions[:, col]))
                )
            if batch_idx % log_interval == 0:
                accepted_so_far = sum(chunk.shape[0] for chunk in accepted_actions)
                logger.info(
                    "  batch %4d | generated=%d accepted=%d/%d | "
                    "overlap=%.1f%% post-remain=%.1f%%",
                    batch_idx + 1,
                    generated_total,
                    accepted_so_far,
                    requested,
                    100.0 * overlap_total / max(generated_total, 1),
                    100.0 * post_remain_total / max(generated_total, 1),
                )
            keep = (
                masks["accepted"]
                if rejection_enabled and enforce_acceptance
                else np.ones(len(batch_rows), dtype=bool)
            )
            if enforce_acceptance and not np.any(keep):
                batch_idx += 1
                continue
            remaining = requested - sum(chunk.shape[0] for chunk in accepted_actions)
            keep_indices = np.flatnonzero(keep)[:remaining]
            accepted_conditions.append(conditions[keep_indices])
            accepted_initial_states.append(initial_states[keep_indices])
            accepted_actions.append(action_batch[keep_indices])
            accepted_trajectories.append(trajectories[keep_indices])
            accepted_ego_lengths.append(ego_lengths_batch[keep_indices].astype(np.float32))
            accepted_adv_lengths.append(adv_lengths_batch[keep_indices].astype(np.float32))
            accepted_base_event_ids.append(
                np.asarray([row["base_event_id"] for row in batch_rows], dtype=object)[keep_indices]
            )
            accepted_source_types.append(
                np.asarray([row["source_type"] for row in batch_rows], dtype=object)[keep_indices]
            )
            batch_idx += 1
    if not accepted_actions:
        raise RuntimeError(
            "No diffusion scenarios were accepted. Relax diffusion_rejection "
            "thresholds or disable diffusion_rejection.enforce_acceptance."
        )
    logger.info(
        "Generation complete — %d total candidates, %d accepted",
        generated_total,
        sum(chunk.shape[0] for chunk in accepted_actions),
    )
    conditions = np.concatenate(accepted_conditions, axis=0)
    initial_states = np.concatenate(accepted_initial_states, axis=0)
    action_array = np.concatenate(accepted_actions, axis=0)
    trajectories = np.concatenate(accepted_trajectories, axis=0)
    ego_lengths = np.concatenate(accepted_ego_lengths, axis=0).astype(np.float64)
    adv_lengths = np.concatenate(accepted_adv_lengths, axis=0).astype(np.float64)
    base_event_id = np.concatenate(accepted_base_event_ids, axis=0)
    source_type = np.concatenate(accepted_source_types, axis=0)
    accepted_count = int(action_array.shape[0])
    if rejection_enabled and enforce_acceptance and accepted_count < requested:
        logger.warning(
            "Diffusion semantic rejection accepted %d/%d requested scenarios from %d candidates",
            accepted_count,
            requested,
            generated_total,
        )
    # Post-generation validity statistics.
    ego_y = initial_states[:, 0, 1].astype(np.float64)
    target_final_y = trajectories[:, -1, 1].astype(np.float64)
    final_lateral_offset = target_final_y - ego_y
    valid_lateral = np.abs(final_lateral_offset) < 1.5
    logger.info(
        "Target-lane validity: %.1f%% lateral-complete",
        100.0 * float(np.mean(valid_lateral)),
    )
    final_masks = _semantic_cutin_mask(
        target_trajectory=trajectories,
        initial_states=initial_states,
        dt=float(schema["dt"]),
        config=config,
    )
    realized_conditions = _realized_conditions_from_arrays(
        conditions,
        initial_states,
        trajectories,
        dt=float(schema["dt"]),
    )
    logger.info(
        "Semantic cut-in target manoeuvre: %.1f%% accepted output | %.1f%% overlap | %.1f%% post-remain",
        100.0 * float(np.mean(final_masks["semantic_cutin"])),
        100.0 * float(np.mean(final_masks["has_overlap"])),
        100.0 * float(np.mean(final_masks["post_remain"])),
    )
    np.savez_compressed(
        output_path,
        scenario_conditions=conditions.astype(np.float32),
        initial_states=initial_states.astype(np.float32),
        actions=action_array.astype(np.float32),
        target_trajectory=trajectories.astype(np.float32),
        ego_length=ego_lengths.astype(np.float32),
        adv_length=adv_lengths.astype(np.float32),
        base_event_id=base_event_id,
        source_type=source_type,
        condition_keys=np.asarray(CONDITION_KEYS, dtype=object),
        realized_scenario_conditions=realized_conditions.astype(np.float32),
        semantic_cutin=final_masks["semantic_cutin"].astype(np.int8),
        rejection_accepted=final_masks["accepted"].astype(np.int8),
        has_lane_entry=final_masks["has_overlap"].astype(np.int8),
        has_lateral_approach=final_masks["has_approach"].astype(np.int8),
        starts_outside_ego_lane=final_masks["starts_outside_ego_lane"].astype(np.int8),
        has_lateral_progress=final_masks["has_lateral_progress"].astype(np.int8),
        post_entry_retention=final_masks["post_remain"].astype(np.int8),
    )
    output_condition_adherence_mae = {
        key: float(
            np.mean(
                np.abs(
                    realized_conditions[:, CONDITION_KEYS.index(key)]
                    - conditions[:, CONDITION_KEYS.index(key)]
                )
            )
        )
        for key in condition_error_totals
    }
    summary = {
        "generated_scenarios": str(output_path),
        "num_generated_scenarios": accepted_count,
        "num_requested_scenarios": requested,
        "diffusion_dataset_dir": str(natural_dir),
        "diffusion_checkpoint_path": str(checkpoint),
        "diffusion_inference_steps": inference_steps,
        "trained_diffusion_steps": int(adapter.model.num_steps),
        "diffusion_batch_size": batch_size,
        "diffusion_seed": int(config["diffusion_seed"]),
        "sampler": "ddim",
        "ego_policy": "playback_highway_env_idm",
        "ego_trajectory_output": False,
        "x0_clip_abs": x0_clip_abs,
        "guidance_scale": guidance_scale,
        "rejection_enabled": rejection_enabled,
        "rejection_enforce_acceptance": enforce_acceptance,
        "initial_condition_samples": int(len(sampled_rows)),
        "condition_refill_enabled": bool(
            rejection_enabled and enforce_acceptance and refill_sampler is not None
        ),
        "condition_refill_rounds": int(refill_rounds),
        "condition_refill_batch_size": int(refill_batch_size),
        "condition_max_refill_rounds": int(max_refill_rounds),
        "rejection_candidates_evaluated": generated_total,
        "generation_export_rate": (
            float(accepted_count / generated_total) if generated_total else 0.0
        ),
        "rejection_acceptance_rate": (
            float(accepted_count / generated_total)
            if enforce_acceptance and generated_total
            else float(np.mean(final_masks["accepted"]))
        ),
        "candidate_semantic_cutin_rate": (
            float(semantic_total / generated_total) if generated_total else 0.0
        ),
        "candidate_overlap_rate": (
            float(overlap_total / generated_total) if generated_total else 0.0
        ),
        "candidate_starts_outside_ego_lane_rate": (
            float(starts_outside_total / generated_total) if generated_total else 0.0
        ),
        "candidate_lateral_progress_rate": (
            float(lateral_progress_total / generated_total) if generated_total else 0.0
        ),
        "candidate_post_remain_rate": (
            float(post_remain_total / generated_total) if generated_total else 0.0
        ),
        "candidate_marginal_failure_rates": {
            "no_lane_entry": (
                float(1.0 - overlap_total / generated_total) if generated_total else 0.0
            ),
            "starts_inside_ego_lane": (
                float(1.0 - starts_outside_total / generated_total)
                if generated_total else 0.0
            ),
            "insufficient_lateral_progress": (
                float(1.0 - lateral_progress_total / generated_total)
                if generated_total else 0.0
            ),
            "no_post_entry_retention": (
                float(1.0 - post_remain_total / generated_total) if generated_total else 0.0
            ),
        },
        "candidate_condition_adherence_mae": {
            key: float(value / generated_total) if generated_total else float("nan")
            for key, value in condition_error_totals.items()
        },
        "output_semantic_cutin_rate": float(np.mean(final_masks["semantic_cutin"])),
        "output_rejection_accepted_rate": float(np.mean(final_masks["accepted"])),
        "output_condition_adherence_mae": output_condition_adherence_mae,
        "model_condition_inputs": ["scenario_conditions"],
        "initial_states_role": "trajectory integration initial state, not denoiser input",
    }
    write_json(output_path.with_name("diffusion_generated_scenarios_summary.json"), summary)
    return output_path, summary


def _validate_diffusion_alignment(
    *,
    adapter: Any,
    diffusion_config: dict[str, Any],
    inference_steps: int,
) -> None:
    schema = adapter.schema
    checkpoint_config = adapter.config
    errors: list[str] = []
    if str(schema.get("event_type", "")).lower() != "cut_in":
        errors.append(f"schema.event_type={schema.get('event_type')!r}")
    if list(schema.get("condition_keys", [])) != list(CONDITION_KEYS):
        errors.append(f"schema.condition_keys={schema.get('condition_keys')!r}")
    if list(schema.get("action_keys", [])) != ["ax", "ay"]:
        errors.append(f"schema.action_keys={schema.get('action_keys')!r}")
    if str(schema.get("action_representation", "")).lower() != "ax_ay":
        errors.append(
            f"schema.action_representation={schema.get('action_representation')!r}"
        )
    expected_horizon = int(diffusion_config["sequence"]["horizon_steps"])
    if int(schema.get("horizon_steps", -1)) != expected_horizon:
        errors.append(
            f"schema.horizon_steps={schema.get('horizon_steps')!r}, "
            f"config.sequence.horizon_steps={expected_horizon}"
        )
    expected_dt = 1.0 / float(diffusion_config["sampling"]["target_fps"])
    if abs(float(schema.get("dt", -1.0)) - expected_dt) > 1.0e-9:
        errors.append(f"schema.dt={schema.get('dt')!r}, expected_dt={expected_dt}")
    expected_steps = int(diffusion_config["diffusion"]["steps"])
    if int(adapter.model.num_steps) != expected_steps:
        errors.append(
            f"checkpoint diffusion_steps={adapter.model.num_steps}, "
            f"config.diffusion.steps={expected_steps}"
        )
    if int(inference_steps) != int(adapter.model.num_steps):
        errors.append(
            f"diffusion_inference_steps={inference_steps}, "
            f"checkpoint diffusion_steps={adapter.model.num_steps}"
        )
    for section in ("event", "sequence", "action", "model"):
        if checkpoint_config.get(section) != diffusion_config.get(section):
            errors.append(f"checkpoint config section {section!r} differs from natural_cutin.yaml")
    checkpoint_diffusion = dict(checkpoint_config.get("diffusion", {}))
    expected_diffusion = dict(diffusion_config.get("diffusion", {}))
    checkpoint_diffusion.pop("x0_clip_abs", None)
    expected_diffusion.pop("x0_clip_abs", None)
    if checkpoint_diffusion != expected_diffusion:
        errors.append("checkpoint config section 'diffusion' differs from natural_cutin.yaml")
    if errors:
        raise RuntimeError("Cut-in diffusion configuration is not aligned: " + "; ".join(errors))


def _semantic_cutin_mask(
    *,
    target_trajectory: np.ndarray,
    initial_states: np.ndarray,
    dt: float,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    cfg = dict(config.get("diffusion_rejection", {}))
    overlap_threshold = float(cfg.get("lateral_overlap_threshold", 1.0))
    lane_threshold = float(cfg.get("cutin_lateral_offset", overlap_threshold))
    min_initial_lateral_offset = float(
        cfg.get("min_initial_lateral_offset", lane_threshold)
    )
    min_lateral_progress = float(cfg.get("min_lateral_progress", 0.0))
    min_approach_speed = float(cfg.get("min_lateral_approach_speed", 0.05))
    post_seconds = float(cfg.get("post_cutin_window_seconds", 3.0))

    traj = np.asarray(target_trajectory, dtype=np.float64)
    init = np.asarray(initial_states, dtype=np.float64)
    batch, horizon = int(traj.shape[0]), int(traj.shape[1])
    ego_y = init[:, 0, 1]
    lateral = traj[:, :, 1] - ego_y[:, None]
    abs_lateral = np.abs(lateral)
    initial_abs = np.abs(init[:, 1, 1] - ego_y)
    abs_with_initial = np.concatenate([initial_abs[:, None], abs_lateral], axis=1)
    approach_speed = np.maximum(-np.diff(abs_with_initial, axis=1) / max(float(dt), 1.0e-6), 0.0)
    min_abs_lateral = np.min(abs_lateral, axis=1)
    starts_outside_ego_lane = initial_abs >= min_initial_lateral_offset
    has_lateral_progress = (initial_abs - min_abs_lateral) >= min_lateral_progress

    has_overlap = np.any(abs_lateral <= overlap_threshold, axis=1)
    first_overlap = np.argmax(abs_lateral <= overlap_threshold, axis=1)
    post_steps = max(1, int(np.ceil(post_seconds / max(float(dt), 1.0e-6))))
    has_approach = np.ones(batch, dtype=bool)
    post_remain = np.zeros(batch, dtype=bool)
    for idx in range(batch):
        if initial_abs[idx] > lane_threshold:
            end = int(first_overlap[idx]) + 1 if has_overlap[idx] else horizon
            has_approach[idx] = bool(
                end > 0 and float(np.max(approach_speed[idx, :end])) >= min_approach_speed
            )
        if has_overlap[idx]:
            start = int(first_overlap[idx])
            stop = min(horizon, start + post_steps)
            post_remain[idx] = bool(np.max(abs_lateral[idx, start:stop]) <= lane_threshold)

    semantic = (
        starts_outside_ego_lane
        & has_lateral_progress
        & has_overlap
        & has_approach
        & post_remain
    )
    return {
        "accepted": semantic,
        "semantic_cutin": semantic,
        "has_overlap": has_overlap,
        "has_approach": has_approach,
        "starts_outside_ego_lane": starts_outside_ego_lane,
        "has_lateral_progress": has_lateral_progress,
        "post_remain": post_remain,
        "final_lateral": lateral[:, -1],
    }


def _cutin_guidance_config(adapter_config: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    action_cfg = adapter_config.get("action", {})
    projection_cfg = adapter_config.get("trajectory_projection", {})
    reject_cfg = config.get("diffusion_rejection", {})
    guidance_cfg = dict(config.get("diffusion_guidance", {}))
    return {
        **reject_cfg,
        **guidance_cfg,
        "ax_min": float(action_cfg.get("ax_min", -8.0)),
        "ax_max": float(action_cfg.get("ax_max", 4.0)),
        "ay_abs_max": float(action_cfg.get("ay_abs_max", 4.0)),
        "lateral_jerk_abs_max": float(action_cfg.get("lateral_jerk_abs_max", 8.0)),
        "speed_min": float(projection_cfg.get("speed_min", 0.0)),
        "speed_max": float(projection_cfg.get("speed_max", 50.0)),
    }


def _generated_semantic_metrics(generated_path: Path, dt: float) -> dict[str, np.ndarray]:
    """Compute intrinsic lane-change trajectory metrics for generated scenarios.

    Only metrics that describe the target vehicle's own manoeuvre are included.
    Interaction-dependent quantities (gap, TTC) are intentionally excluded from
    this distribution plot so the figure stays focused on the generated target
    manoeuvre. Closed-loop ego response is generated separately by the playback
    scripts with highway-env IDM.
    """
    data = np.load(generated_path, allow_pickle=True)
    return _compute_intrinsic_trajectory_metrics(
        target_trajectory=data["target_trajectory"].astype(np.float64),
        initial_states=data["initial_states"].astype(np.float64),
        dt=float(dt),
    )


def _realized_conditions_from_arrays(
    input_conditions: np.ndarray,
    initial_states: np.ndarray,
    target_trajectory: np.ndarray,
    *,
    dt: float,
) -> np.ndarray:
    """Replace outcome-dependent input conditions with realized trajectory values."""
    conditions = np.asarray(input_conditions, dtype=np.float64).copy()
    initial = np.asarray(initial_states, dtype=np.float64)
    trajectory = np.asarray(target_trajectory, dtype=np.float64)
    conditions[:, CONDITION_KEYS.index("final_lateral_offset")] = (
        trajectory[:, -1, 1] - initial[:, 0, 1]
    )
    conditions[:, CONDITION_KEYS.index("target_speed_change")] = (
        trajectory[:, -1, 2] - initial[:, 1, 2]
    )
    return conditions


def _generated_realized_conditions(
    generated: Any,
    *,
    dt: float,
) -> np.ndarray:
    if "realized_scenario_conditions" in generated.files:
        return generated["realized_scenario_conditions"].astype(np.float64)
    return _realized_conditions_from_arrays(
        generated["scenario_conditions"],
        generated["initial_states"],
        generated["target_trajectory"],
        dt=dt,
    )


def _real_semantic_matrix(
    rows: list[dict[str, Any]],
    *,
    dataset_dir: Path,
    dt: float,
    horizon_steps: int | None = None,
) -> tuple[tuple[str, ...], np.ndarray, dict[str, np.ndarray]]:
    """Intrinsic lane-change trajectory metrics from real tail event states."""
    if rows and all("future_states" in row for row in rows):
        collected: dict[str, list[float]] = {
            name: []
            for name in (
                *INTRINSIC_TRAJECTORY_METRIC_NAMES,
                *LANE_CHANGE_RATE_NAMES,
            )
        }
        skipped_short = 0
        for row in rows:
            initial = np.asarray(row["initial_states"], dtype=np.float64)[None, :, :]
            future = np.asarray(row["future_states"], dtype=np.float64)
            if horizon_steps is not None:
                horizon = int(horizon_steps)
                if future.shape[0] < horizon:
                    skipped_short += 1
                    continue
                future = future[:horizon]
            metrics = _compute_intrinsic_trajectory_metrics(
                target_trajectory=future[None, :, 1, :],
                initial_states=initial,
                dt=float(dt),
            )
            for name in collected:
                collected[name].append(float(metrics[name][0]))
        if not collected[INTRINSIC_TRAJECTORY_METRIC_NAMES[0]]:
            raise RuntimeError(
                "No real cut-in tail trajectory is long enough for the requested "
                f"{int(horizon_steps or 0)}-step manoeuvre comparison horizon"
            )
        metrics_arr = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in collected.items()
        }
        metrics_arr["_real_rows_used"] = np.asarray(
            [len(collected[INTRINSIC_TRAJECTORY_METRIC_NAMES[0]])],
            dtype=np.float64,
        )
        metrics_arr["_real_rows_skipped_shorter_than_horizon"] = np.asarray(
            [skipped_short],
            dtype=np.float64,
        )
        values = np.stack(
            [metrics_arr[name] for name in INTRINSIC_TRAJECTORY_METRIC_NAMES],
            axis=1,
        )
        return INTRINSIC_TRAJECTORY_METRIC_NAMES, values, metrics_arr

    dataset_path = Path(dataset_dir) / "dataset.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Cut-in diffusion dataset not found: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    idx_arr = _dataset_indices_for_tail_rows(data, rows)
    initial = data["initial_states"][idx_arr].astype(np.float64)
    future = data["future_states"][idx_arr].astype(np.float64)
    if horizon_steps is not None:
        future = future[:, : int(horizon_steps)]
    metrics = _compute_intrinsic_trajectory_metrics(
        target_trajectory=future[:, :, 1, :],
        initial_states=initial,
        dt=float(dt),
    )
    values = np.stack(
        [metrics[name] for name in INTRINSIC_TRAJECTORY_METRIC_NAMES], axis=1
    )
    return INTRINSIC_TRAJECTORY_METRIC_NAMES, values, metrics


def _dataset_indices_for_tail_rows(
    data: Any,
    rows: list[dict[str, Any]],
) -> np.ndarray:
    by_event_id: dict[str, list[int]] = {}
    for idx, event_id in enumerate(data["event_id"].tolist()):
        by_event_id.setdefault(str(event_id), []).append(int(idx))
    dataset_anchor = (
        data["anchor_frame"].astype(np.int64)
        if "anchor_frame" in data.files
        else None
    )
    indices: list[int] = []
    for row in rows:
        event_id = str(row["event_id"])
        candidates = by_event_id.get(event_id)
        if not candidates:
            raise KeyError(f"Tail event {event_id} not found in diffusion dataset")
        if dataset_anchor is not None and "anchor_frame" in row:
            anchor = int(row["anchor_frame"])
            chosen = min(
                candidates,
                key=lambda idx: abs(int(dataset_anchor[idx]) - anchor),
            )
        else:
            chosen = candidates[0]
        indices.append(int(chosen))
    return np.asarray(indices, dtype=np.int64)


def _real_tail_trajectory_arrays(
    rows: list[dict[str, Any]],
    *,
    dataset_dir: Path,
) -> tuple[np.ndarray, list[np.ndarray]]:
    if rows and all("future_states" in row for row in rows):
        initial = np.asarray(
            [row["initial_states"] for row in rows],
            dtype=np.float64,
        )
        target = [
            np.asarray(row["future_states"], dtype=np.float64)[:, 1, :]
            for row in rows
        ]
        return initial, target

    dataset_path = Path(dataset_dir) / "dataset.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Cut-in diffusion dataset not found: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    idx_arr = _dataset_indices_for_tail_rows(data, rows)
    initial = data["initial_states"][idx_arr].astype(np.float64)
    target = [
        data["future_states"][idx, :, 1, :].astype(np.float64)
        for idx in idx_arr
    ]
    return initial, target


def _generated_semantic_matrix(
    metrics: dict[str, np.ndarray],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Extract intrinsic metric matrix from generated-scenario metric dict."""
    values = np.stack(
        [metrics[name] for name in INTRINSIC_TRAJECTORY_METRIC_NAMES], axis=1
    )
    return INTRINSIC_TRAJECTORY_METRIC_NAMES, values


def _plot_lateral_trajectory_comparison(
    *,
    empirical_rows: list[dict[str, Any]],
    generated: Any,
    dataset_dir: Path,
    path: Path,
    dt: float,
    max_traces: int = 120,
) -> dict[str, Any]:
    real_initial, real_target = _real_tail_trajectory_arrays(
        empirical_rows,
        dataset_dir=dataset_dir,
    )
    gen_initial = generated["initial_states"].astype(np.float64)
    gen_target = generated["target_trajectory"].astype(np.float64)
    real_direction = np.sign(real_initial[:, 0, 1] - real_initial[:, 1, 1])
    gen_direction = np.sign(gen_initial[:, 0, 1] - gen_initial[:, 1, 1])
    real_direction = np.where(real_direction == 0.0, 1.0, real_direction)
    gen_direction = np.where(gen_direction == 0.0, 1.0, gen_direction)
    real_horizon = min(len(item) for item in real_target) if real_target else 0
    if real_horizon <= 0:
        raise RuntimeError("Real cut-in tail trajectories are empty")
    real_y = np.stack(
        [
            (target[:real_horizon, 1] - real_initial[idx, 1, 1]) * real_direction[idx]
            for idx, target in enumerate(real_target)
        ],
        axis=0,
    )
    gen_y = (gen_target[:, :, 1] - gen_initial[:, 1, 1:2]) * gen_direction[:, None]
    horizon = min(real_y.shape[1], gen_y.shape[1])
    real_y = real_y[:, :horizon]
    gen_y = gen_y[:, :horizon]
    t = np.arange(horizon, dtype=np.float64) * float(dt)

    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    rng = np.random.default_rng(42)
    for values, color, label in (
        (real_y, REAL_COLOR, "EVT tail"),
        (gen_y, GENERATED_COLOR, "Diffusion"),
    ):
        sample_count = min(max_traces, values.shape[0])
        if sample_count > 0:
            sample_idx = rng.choice(values.shape[0], size=sample_count, replace=False)
            ax.plot(
                t,
                values[sample_idx].T,
                color=color,
                alpha=0.045,
                linewidth=0.7,
            )
        median = np.nanmedian(values, axis=0)
        lower = np.nanquantile(values, 0.25, axis=0)
        upper = np.nanquantile(values, 0.75, axis=0)
        ax.fill_between(t, lower, upper, color=color, alpha=0.16, linewidth=0.0)
        ax.plot(t, median, color=color, linewidth=2.1, label=f"{label} median")
    ax.axhline(0.0, color="#333333", linestyle=":", linewidth=1.0, alpha=0.75)
    ax.set_xlabel(r"$t$ from anchor (s)")
    ax.set_ylabel(r"$\Delta y_{\mathrm{ego}}$ (m)")
    ax.set_title("Cut-in lateral trajectories")
    style_axes(ax)
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "real_num_trajectories": int(real_y.shape[0]),
        "generated_num_trajectories": int(gen_y.shape[0]),
        "horizon_steps": int(horizon),
        "real_final_lateral_displacement_mean": float(np.nanmean(real_y[:, -1])),
        "generated_final_lateral_displacement_mean": float(np.nanmean(gen_y[:, -1])),
    }


def _write_visualizations(
    *,
    empirical_rows: list[dict[str, Any]],
    generated_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    dt: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_paths = {
        "cutin_manoeuvre_tail_vs_generated": output_dir / "cutin_manoeuvre_tail_vs_generated.png",
        "cutin_lateral_trajectory_tail_vs_generated": output_dir / "cutin_lateral_trajectory_tail_vs_generated.png",
        "scenario_condition_tail_vs_copula_sampled": output_dir / "scenario_condition_tail_vs_copula_sampled.png",
    }

    generated = np.load(generated_path, allow_pickle=True)
    comparison_horizon_steps = int(generated["target_trajectory"].shape[1])
    generated_metrics = _generated_semantic_metrics(generated_path, dt)
    semantic_names, real_semantic, real_metrics = _real_semantic_matrix(
        empirical_rows,
        dataset_dir=dataset_dir,
        dt=dt,
        horizon_steps=comparison_horizon_steps,
    )
    _, generated_semantic = _generated_semantic_matrix(generated_metrics)
    semantic_metrics = _plot_hist_grid(
        real_semantic,
        generated_semantic,
        semantic_names,
        condition_paths["cutin_manoeuvre_tail_vs_generated"],
        real_label="EVT tail",
        generated_label="Diffusion",
    )
    lane_change_rates = {
        name: {
            "real_rate": float(np.mean(real_metrics[name])),
            "generated_rate": float(np.mean(generated_metrics[name])),
            "rate_delta": float(
                np.mean(generated_metrics[name]) - np.mean(real_metrics[name])
            ),
        }
        for name in LANE_CHANGE_RATE_NAMES
    }
    trajectory_family_metrics = _plot_lateral_trajectory_comparison(
        empirical_rows=empirical_rows,
        generated=generated,
        dataset_dir=dataset_dir,
        path=condition_paths["cutin_lateral_trajectory_tail_vs_generated"],
        dt=dt,
    )
    condition_metrics = _plot_condition_distribution_comparison(
        np.asarray(
            [row["scenario_conditions"] for row in empirical_rows],
            dtype=np.float64,
        ),
        generated["scenario_conditions"].astype(np.float64),
        _generated_realized_conditions(generated, dt=dt),
        histogram_path=condition_paths["scenario_condition_tail_vs_copula_sampled"],
    )
    report = {
        "figures": {key: str(value) for key, value in condition_paths.items()},
        "manoeuvre_comparison_horizon": {
            "aligned_to_generated_steps": comparison_horizon_steps,
            "aligned_to_generated_seconds": float(
                comparison_horizon_steps * float(dt)
            ),
            "real_rows_used": int(
                real_metrics.get("_real_rows_used", [len(empirical_rows)])[0]
            ),
            "real_rows_skipped_shorter_than_horizon": int(
                real_metrics.get("_real_rows_skipped_shorter_than_horizon", [0])[0]
            ),
            "endpoint_metrics": [
                "longitudinal_displacement",
                "total_lateral_displacement",
                "lateral_progress_toward_ego_lane",
                "final_abs_lateral_offset",
                "target_speed_change",
            ],
        },
        "cutin_manoeuvre_metrics": semantic_metrics,
        "cutin_lateral_trajectory_family": trajectory_family_metrics,
        "lane_change_event_rates": lane_change_rates,
        "scenario_condition_distribution": condition_metrics,
        "note": (
            "The main similarity assessment is cut-in specific: lane entry, "
            "post-entry retention, lateral progress/displacement, final lane "
            "position, entry timing, lateral motion, and target speed change. "
            "Real EVT-tail manoeuvre endpoint metrics are computed on the first "
            f"{comparison_horizon_steps} frames to align with the generated "
            "diffusion horizon. "
            "The scenario-condition histogram compares empirical EVT-tail "
            "conditions, Gaussian-copula sampled diffusion inputs, and conditions "
            "realized by diffusion trajectories. "
            "Interaction-dependent gap/TTC metrics are excluded from this plot; "
            "playback scripts generate highway-env IDM ego trajectories for "
            "closed-loop replay against the scripted adversary trajectory."
        ),
    }
    write_json(output_dir / "distribution_similarity_summary.json", report)
    return report


def run_cutin_tail_generation(config: dict[str, Any]) -> None:
    required = {
        "event_context_cache_path",
        "condition_distribution_path",
        "independent_tail_peaks_path",
        "evt_model_path",
        "evt_summary_path",
        "num_condition_samples",
        "num_diffusion_scenarios",
        "diffusion_dataset_dir",
        "diffusion_checkpoint_path",
        "diffusion_config_path",
        "generated_scenarios_path",
        "diffusion_batch_size",
        "diffusion_inference_steps",
        "diffusion_device",
        "diffusion_seed",
        "selection_random_seed",
        "copula_marginal_clip_quantile",
        "copula_correlation_regularization",
    }
    missing = sorted(key for key in required if key not in config)
    if missing:
        raise KeyError(f"Cut-in tail generation config missing keys: {missing}")

    rows = _load_semantic_contexts(_path(config, "event_context_cache_path"))
    evt_meta = _score_rows_with_evt(
        rows,
        model_path=_path(config, "evt_model_path"),
        summary_path=_path(config, "evt_summary_path"),
    )
    empirical_tail = _select_independent_tail_peaks(
        rows,
        _path(config, "independent_tail_peaks_path"),
    )
    rng = np.random.default_rng(int(config["selection_random_seed"]))
    condition_sample_count = int(config["num_condition_samples"])
    if condition_sample_count < int(config["num_diffusion_scenarios"]):
        raise ValueError(
            "num_condition_samples must be at least num_diffusion_scenarios; "
            "the cut-in tail pipeline samples scenario conditions directly from "
            "the fitted joint distribution and decodes each selected condition once."
        )
    sampled_conditions, corr, variable_mask = _sample_condition_distribution(
        empirical_tail,
        count=condition_sample_count,
        rng=rng,
        clip_quantile=float(config["copula_marginal_clip_quantile"]),
        regularization=float(config["copula_correlation_regularization"]),
    )
    next_condition_index = int(len(sampled_conditions))

    def _refill_condition_rows(count: int) -> list[dict[str, Any]]:
        nonlocal next_condition_index
        extra_rows, _, _ = _sample_condition_distribution(
            empirical_tail,
            count=int(count),
            rng=rng,
            clip_quantile=float(config["copula_marginal_clip_quantile"]),
            regularization=float(config["copula_correlation_regularization"]),
            start_index=next_condition_index,
        )
        next_condition_index += int(len(extra_rows))
        return extra_rows

    condition_distribution_path = _path(config, "condition_distribution_path")
    _save_condition_distribution(
        condition_distribution_path,
        empirical_rows=empirical_tail,
        corr=corr,
        variable_mask=variable_mask,
        evt_meta=evt_meta,
        config=config,
    )
    tail_context_path = Path(
        config.get(
            "tail_context_path",
            condition_distribution_path.with_name("tail_contexts.npz"),
        )
    ).resolve()
    _save_tail_contexts(
        tail_context_path,
        rows=[*empirical_tail, *sampled_conditions],
        evt_meta=evt_meta,
        dt=0.04,
    )
    generated_path, diffusion_summary = _generate_diffusion_scenarios(
        sampled_conditions,
        config=config,
        refill_sampler=_refill_condition_rows,
    )
    visual_summary = _write_visualizations(
        empirical_rows=empirical_tail,
        generated_path=generated_path,
        dataset_dir=_path(config, "diffusion_dataset_dir"),
        output_dir=generated_path.parent / "figures",
        dt=0.04,
    )
    summary = {
        **evt_meta,
        "condition_distribution": str(condition_distribution_path),
        "tail_contexts": str(tail_context_path),
        "num_evt_tail_conditions": int(len(empirical_tail)),
        "num_initial_condition_samples": int(len(sampled_conditions)),
        "num_condition_samples": int(
            diffusion_summary["rejection_candidates_evaluated"]
        ),
        "num_diffusion_scenarios": int(diffusion_summary["num_generated_scenarios"]),
        "num_requested_diffusion_scenarios": int(config["num_diffusion_scenarios"]),
        "condition_keys": list(CONDITION_KEYS),
        "tail_feature_names": list(TAIL_FEATURE_NAMES),
        "dynamic_tail_feature_names": [
            name for name, keep in zip(TAIL_FEATURE_NAMES, variable_mask) if bool(keep)
        ],
        "constant_tail_feature_values": {
            name: float(_feature_matrix(empirical_tail)[0, idx])
            for idx, (name, keep) in enumerate(zip(TAIL_FEATURE_NAMES, variable_mask))
            if not bool(keep)
        },
        "condition_distribution_model": (
            "Gaussian copula fitted on EVT declustered independent cut-in tail "
            "scenario_conditions"
        ),
        "model_condition_inputs": ["scenario_conditions"],
        "initial_states_role": "trajectory integration initial state, not denoiser input",
        "diffusion_generation": diffusion_summary,
        "visualization": visual_summary,
        "selection_random_seed": int(config["selection_random_seed"]),
        "copula_marginal_clip_quantile": float(config["copula_marginal_clip_quantile"]),
        "copula_correlation_regularization": float(
            config["copula_correlation_regularization"]
        ),
        "diffusion_rejection": dict(config.get("diffusion_rejection", {})),
        "diffusion_guidance": dict(config.get("diffusion_guidance", {})),
    }
    write_json(
        condition_distribution_path.with_name("scenario_condition_distribution_summary.json"),
        summary,
    )
    logger.info(
        "Wrote scenario-condition distribution from %d EVT cut-in tail events, "
        "%d sampled conditions, and %d diffusion scenarios",
        len(empirical_tail),
        len(sampled_conditions),
        int(diffusion_summary["num_generated_scenarios"]),
    )

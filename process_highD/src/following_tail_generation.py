"""highD car-following tail context generation and diffusion rollout engine."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from diffusion.src.features import FOLLOWING_SCENARIO_CONDITION_KEYS
from tools.evt import load_evt_model
from tools.io import write_json
from tools.plot_style import (
    GENERATED_COLOR,
    REAL_COLOR,
    get_pyplot,
    label_for,
    style_axes,
)


SOURCE_INDEPENDENT_TAIL_PEAK = "highd_independent_tail_peak"
SOURCE_TAIL_GAUSSIAN_COPULA = "highd_tail_gaussian_copula"
CONTEXT_METHOD_EMPIRICAL = "empirical"
CONTEXT_METHOD_GAUSSIAN_COPULA = "gaussian_copula"
FOLLOWING_EGO_VX_IDX = 0
FOLLOWING_LOG_GAP_IDX = 1
FOLLOWING_DELTA_V_IDX = 2
FOLLOWING_LEAD_AX_IDX = 3
CUTIN_EGO_VX_IDX = 0
CUTIN_LOG_GAP_IDX = 1
CUTIN_LATERAL_OFFSET_IDX = 2
CUTIN_DELTA_VX_IDX = 3
CUTIN_TARGET_AX_IDX = 4
CUTIN_TARGET_VY_IDX = 5
CUTIN_TARGET_AY_IDX = 6
CUTIN_FINAL_LATERAL_OFFSET_IDX = 7
CUTIN_TIME_TO_CROSS_IDX = 8
CUTIN_TARGET_SPEED_CHANGE_IDX = 9
VARIABLE_EPS = 1.0e-8


COMMON_SELECTION_DEFAULTS: dict[str, Any] = {
    "evt_target_mode": "collision_critical_level",
    "empirical_context_limit": None,
    "context_generation_method": CONTEXT_METHOD_GAUSSIAN_COPULA,
    "num_synthetic_contexts": 500,
    "include_empirical_contexts": True,
    "selection_random_seed": 42,
    "evt_return_period": 100,
    "min_future_steps": 125,
    "copula_correlation_regularization": 1.0e-4,
    "copula_marginal_clip_quantile": 0.01,
    "generate_diffusion_rollouts": False,
    "num_diffusion_scenarios": 0,
    "diffusion_checkpoint_path": "checkpoints/best_noise_mse_train_val_test.pt",
    "generated_scenarios_path": None,
    "condition_distribution_path": None,
    "diffusion_batch_size": 256,
    "diffusion_inference_steps": None,
    "diffusion_device": "auto",
    "diffusion_seed": 42,
}
logger = logging.getLogger(__name__)

_COMMON_CONTEXT_KEYS: tuple[str, ...] = (
    "recording_id",
    "event_id",
    "ego_id",
    "target_id",
    "anchor_frame",
    "context_anchor_frame",
    "event_anchor_frame",
    "ego_length",
    "adv_length",
    "initial_gap",
    "initial_closing_speed",
    "recorded_min_gap",
    "recorded_min_ttc",
    "collision",
    "near_collision",
    "risk_score",
    "evt_tail_probability",
    "peak_id",
    "representative_event_id",
    "base_context_index",
    "base_event_id",
    "synthetic_context",
    "context_model_method",
    "context_feature_distance",
)

_COMMON_CONTEXT_KEY_DTYPES = {
    "recording_id": "int",
    "event_id": "str",
    "ego_id": "int",
    "target_id": "int",
    "anchor_frame": "int",
    "context_anchor_frame": "int",
    "event_anchor_frame": "int",
    "ego_length": "float",
    "adv_length": "float",
    "initial_gap": "float",
    "initial_closing_speed": "float",
    "recorded_min_gap": "float",
    "recorded_min_ttc": "float",
    "collision": "float",
    "near_collision": "float",
    "risk_score": "float",
    "evt_tail_probability": "float",
    "peak_id": "str",
    "representative_event_id": "str",
    "base_context_index": "int",
    "base_event_id": "str",
    "synthetic_context": "int",
    "context_model_method": "str",
    "context_feature_distance": "float",
}

_DTYPE_ALIASES = {
    "int": np.int32,
    "float": np.float32,
    "str": object,
    "object": object,
}

_SYNTHETIC_CONTEXT_FIELDS = {
    "base_context_index": -1,
    "base_event_id": "",
    "synthetic_context": 0,
    "context_model_method": CONTEXT_METHOD_EMPIRICAL,
    "context_feature_distance": 0.0,
}

_SYNTHETIC_CONTEXT_DTYPES = {
    "base_context_index": "int",
    "base_event_id": "str",
    "synthetic_context": "int",
    "context_model_method": "str",
    "context_feature_distance": "float",
}


def _unique_keys(keys: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return tuple(out)


def _resolve_np_dtype(label: str | type | None) -> object | type | None:
    if label is None:
        return None
    if isinstance(label, str):
        return _DTYPE_ALIASES.get(label, label)
    return label


def _context_key_dtypes(config: dict[str, Any]) -> dict[str, object | type | None]:
    configured = dict(config.get("context_key_dtypes", {}))
    labels = {
        **_COMMON_CONTEXT_KEY_DTYPES,
        **_SYNTHETIC_CONTEXT_DTYPES,
        **configured,
    }
    return {key: _resolve_np_dtype(value) for key, value in labels.items()}


def _context_output_keys(config: dict[str, Any]) -> tuple[str, ...]:
    risk_key = str(config["risk_value_key"])
    configured = tuple(str(key) for key in config.get("context_output_keys", ()))
    return _unique_keys(_COMMON_CONTEXT_KEYS + (risk_key,) + configured)


FOLLOWING_TAIL_FEATURE_NAMES: tuple[str, ...] = (
    "ego_vx_0",
    "log_initial_gap",
    "initial_delta_v",
    "lead_ax_0",
    "lead_speed_change",
    "lead_min_ax",
    "lead_braking_duration",
)

CUTIN_TAIL_FEATURE_NAMES: tuple[str, ...] = (
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


def _tail_feature_names(config: dict[str, Any]) -> tuple[str, ...]:
    return (
        CUTIN_TAIL_FEATURE_NAMES
        if str(config["scenario"]) == "cut_in"
        else FOLLOWING_TAIL_FEATURE_NAMES
    )


def _merged_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = {**COMMON_SELECTION_DEFAULTS, **config}
    required = {
        "event_context_cache_path",
        "tail_context_path",
        "independent_tail_peaks_path",
        "evt_model_path",
        "evt_summary_path",
        "scenario",
        "risk_value_key",
        "context_loader",
    }
    missing = sorted(key for key in required if key not in cfg)
    if missing:
        raise KeyError(f"Tail context selection config missing keys: {missing}")
    return cfg


def _read_evt_summary(config: dict[str, Any]) -> dict[str, Any]:
    summary_path = Path(config["evt_summary_path"])
    if not summary_path.exists():
        raise FileNotFoundError(
            "EVT summary is required before tail context selection: "
            f"{summary_path}"
        )
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_evt_scoring(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Score rows with the EVT model and return per-row scores + dataset constants."""
    evt_model_path = Path(config["evt_model_path"])
    if not evt_model_path.exists():
        raise FileNotFoundError(
            "EVT model is required before tail context selection: "
            f"{evt_model_path}. Run "
            f"{config.get('fit_evt_hint', 'the scenario EVT fitting script')} first."
        )
    model = load_evt_model(evt_model_path)
    return_period = int(config["evt_return_period"])
    evt_summary = _read_evt_summary(config)
    if "collision_critical_level" not in evt_summary:
        raise KeyError(
            f"{config['evt_summary_path']} is missing collision_critical_level"
        )
    collision_critical_level = float(evt_summary["collision_critical_level"])
    if str(config["evt_target_mode"]) == "collision_critical_level":
        target = collision_critical_level
    else:
        target = float(model.return_level(return_period))
    failure_threshold = float(model.score(target))
    tail_threshold_u = float(model.u)
    tail_threshold_score = float(model.score(tail_threshold_u))
    exceedance_rate = float(model.exceedance_rate)

    risk_key = str(config["risk_value_key"])
    values = np.asarray([row[risk_key] for row in rows], dtype=np.float64)
    risk_score = np.asarray(model.score(values), dtype=np.float64)
    tail_probability = np.asarray(model.survival(values), dtype=np.float64)

    for idx, row in enumerate(rows):
        row["risk_score"] = float(risk_score[idx])
        row["evt_tail_probability"] = float(tail_probability[idx])

    # Return dataset-level constants separately (written once in summary, not per row).
    evt_meta = {
        "evt_tail_threshold_u": tail_threshold_u,
        "evt_tail_threshold_score": tail_threshold_score,
        "evt_exceedance_rate": exceedance_rate,
        "evt_return_period": return_period,
        "evt_return_level_target": target,
        "evt_failure_threshold": failure_threshold,
        "evt_model_path": str(evt_model_path),
        "evt_target_mode": str(config["evt_target_mode"]),
        "collision_critical_level": collision_critical_level,
        "collision_critical_level_mode": evt_summary.get(
            "collision_critical_level_mode"
        ),
        "human_calibrated_safety_threshold": evt_summary.get(
            "human_calibrated_safety_threshold"
        ),
        "risk_value_key": risk_key,
        "scenario": str(config["scenario"]),
    }
    return evt_meta, evt_model_path


def _load_cached_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    cache_path = Path(config["event_context_cache_path"])
    if not cache_path.exists():
        raise FileNotFoundError(
            "highD event context cache is required before tail selection: "
            f"{cache_path}. Run process_highD/scripts/extract_highd_events.py first."
        )
    loader = config["context_loader"]
    if not callable(loader):
        raise TypeError("Tail context selection config context_loader must be callable")
    rows = loader(cache_path)
    if not rows:
        raise RuntimeError(f"highD event context cache is empty: {cache_path}")
    row_filter = config.get("row_filter")
    if row_filter is not None:
        if not callable(row_filter):
            raise TypeError("Tail context selection config row_filter must be callable")
        before = len(rows)
        rows = row_filter(rows)
        if not rows:
            raise RuntimeError(
                f"No highD {config['scenario']} contexts remain after row_filter: "
                f"{cache_path}"
            )
        removed = before - len(rows)
        if removed:
            logger.info(
                "Filtered %d cached highD %s contexts before tail selection",
                removed,
                config["scenario"],
            )
    if "scenario_conditions" not in rows[0] or "initial_states" not in rows[0]:
        raise KeyError(
            f"{config['scenario']} context cache is not anchor-scenario. "
            f"Rebuild it first with python process_highD/scripts/extract_highd_events.py: "
            f"{cache_path}"
        )
    logger.info(
        "Loaded %d highD %s contexts from %s",
        len(rows),
        config["scenario"],
        cache_path,
    )
    return rows


def _load_rows(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    rows = _load_cached_rows(config)
    evt_meta, _ = _apply_evt_scoring(rows, config)
    return rows, evt_meta, f"{config['scenario']}_event_context_cache"


def _tail_feature(row: dict[str, Any], scenario: str) -> np.ndarray:
    conditions = np.asarray(row["scenario_conditions"], dtype=np.float64)
    gap = max(float(conditions[1]), 0.2)
    if str(scenario) == "cut_in":
        return np.asarray(
            [
                float(conditions[0]),
                np.log(gap),
                float(conditions[2]),
                float(conditions[3]),
                float(conditions[4]),
                float(conditions[5]),
                float(conditions[6]),
                float(conditions[7]),
                float(conditions[8]),
                float(conditions[9]),
            ],
            dtype=np.float64,
        )
    return np.asarray(
        [
            float(conditions[0]),
            np.log(gap),
            float(conditions[2]),
            float(conditions[3]),
            float(conditions[4]),
            float(conditions[5]),
            float(conditions[6]),
        ],
        dtype=np.float64,
    )


def _feature_matrix(rows: list[dict[str, Any]], scenario: str) -> np.ndarray:
    return np.stack([_tail_feature(row, scenario) for row in rows], axis=0)


def _reconstruct_initial_from_feature(
    base_row: dict[str, Any],
    target_feature: np.ndarray,
    scenario: str,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(base_row["initial_states"], dtype=np.float32).copy()
    ego_length = float(base_row["ego_length"])
    adv_length = float(base_row["adv_length"])

    if str(scenario) == "cut_in":
        target_gap = float(np.exp(float(target_feature[CUTIN_LOG_GAP_IDX])))
        ego_vx = max(float(target_feature[CUTIN_EGO_VX_IDX]), 0.0)
        delta_vx = float(target_feature[CUTIN_DELTA_VX_IDX])
        states[1, 0] = np.float32(
            states[0, 0] + 0.5 * (ego_length + adv_length) + target_gap
        )
        states[0, 2] = np.float32(ego_vx)
        states[1, 2] = np.float32(max(ego_vx - delta_vx, 0.0))
        states[1, 1] = np.float32(
            states[0, 1] + float(target_feature[CUTIN_LATERAL_OFFSET_IDX])
        )
        states[1, 4] = np.float32(
            np.clip(float(target_feature[CUTIN_TARGET_AX_IDX]), -8.0, 4.0)
        )
        states[1, 3] = np.float32(float(target_feature[CUTIN_TARGET_VY_IDX]))
        states[1, 5] = np.float32(
            np.clip(float(target_feature[CUTIN_TARGET_AY_IDX]), -4.0, 4.0)
        )
        scenario_conditions = np.asarray(
            [
                ego_vx,
                target_gap,
                float(target_feature[CUTIN_LATERAL_OFFSET_IDX]),
                delta_vx,
                float(target_feature[CUTIN_TARGET_AX_IDX]),
                float(target_feature[CUTIN_TARGET_VY_IDX]),
                float(target_feature[CUTIN_TARGET_AY_IDX]),
                float(target_feature[CUTIN_FINAL_LATERAL_OFFSET_IDX]),
                float(target_feature[CUTIN_TIME_TO_CROSS_IDX]),
                float(target_feature[CUTIN_TARGET_SPEED_CHANGE_IDX]),
            ],
            dtype=np.float32,
        )
        return scenario_conditions, states.astype(np.float32)

    target_gap = float(np.exp(float(target_feature[FOLLOWING_LOG_GAP_IDX])))
    ego_vx = max(float(target_feature[FOLLOWING_EGO_VX_IDX]), 0.0)
    delta_v = float(target_feature[FOLLOWING_DELTA_V_IDX])
    states[1, 0] = np.float32(states[0, 0] + 0.5 * (ego_length + adv_length) + target_gap)
    states[0, 2] = np.float32(ego_vx)
    states[1, 2] = np.float32(max(ego_vx - delta_v, 0.0))
    states[1, 4] = np.float32(
        np.clip(float(target_feature[FOLLOWING_LEAD_AX_IDX]), -8.0, 4.0)
    )
    scenario_conditions = np.asarray(
        [
            ego_vx,
            target_gap,
            delta_v,
            float(states[1, 4]),
            float(target_feature[4]),
            float(target_feature[5]),
            max(float(target_feature[6]), 0.0),
        ],
        dtype=np.float32,
    )
    return scenario_conditions, states.astype(np.float32)


def _normal_score_pseudo_observations(features: np.ndarray) -> np.ndarray:
    from scipy.special import ndtri

    ranks = np.empty_like(features, dtype=np.float64)
    n = int(features.shape[0])
    for col in range(int(features.shape[1])):
        order = np.argsort(features[:, col], kind="mergesort")
        ranks[order, col] = np.arange(1, n + 1, dtype=np.float64)
    u = ranks / float(n + 1)
    return ndtri(np.clip(u, 1.0e-6, 1.0 - 1.0e-6))


def _fit_gaussian_copula_model(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scenario = str(config["scenario"])
    features = _feature_matrix(rows, scenario)
    variable = np.std(features, axis=0) > VARIABLE_EPS
    if not np.any(variable):
        raise RuntimeError("Gaussian copula has no variable tail-feature dimensions")
    z = _normal_score_pseudo_observations(features[:, variable])
    corr_variable = np.corrcoef(z, rowvar=False)
    corr_variable = np.nan_to_num(
        np.atleast_2d(corr_variable),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(corr_variable, 1.0)
    reg = max(float(config["copula_correlation_regularization"]), 0.0)
    corr_variable += np.eye(corr_variable.shape[0], dtype=np.float64) * reg
    eigvals, eigvecs = np.linalg.eigh(corr_variable)
    eigvals = np.clip(eigvals, 1.0e-8, None)
    corr_variable = (eigvecs * eigvals[None, :]) @ eigvecs.T
    denom = np.sqrt(np.clip(np.diag(corr_variable), 1.0e-12, None))
    corr_variable = corr_variable / denom[:, None] / denom[None, :]

    corr = np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
    corr[np.ix_(variable, variable)] = corr_variable
    np.fill_diagonal(corr, 1.0)
    return features, corr, variable


def _save_condition_distribution(
    path: Path,
    *,
    empirical_rows: list[dict[str, Any]],
    evt_meta: dict[str, Any],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    features, corr, variable = _fit_gaussian_copula_model(
        empirical_rows,
        config=config,
    )
    payload: dict[str, np.ndarray] = {
        "scenario_conditions": np.asarray(
            [row["scenario_conditions"] for row in empirical_rows],
            dtype=np.float32,
        ),
        "tail_features": features.astype(np.float32),
        "condition_keys": np.asarray(FOLLOWING_SCENARIO_CONDITION_KEYS, dtype=object),
        "tail_feature_names": np.asarray(_tail_feature_names(config), dtype=object),
        "copula_correlation": corr.astype(np.float32),
        "copula_variable_mask": np.asarray(variable, dtype=bool),
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
            [SOURCE_INDEPENDENT_TAIL_PEAK for _ in empirical_rows],
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
        "tail_threshold": np.asarray(
            evt_meta["evt_tail_threshold_u"],
            dtype=np.float32,
        ),
        "collision_critical_level": np.asarray(
            evt_meta["collision_critical_level"],
            dtype=np.float32,
        ),
    }
    np.savez_compressed(path, **payload)


def _sample_gaussian_copula_contexts(
    rows: list[dict[str, Any]],
    *,
    count: int,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if len(rows) < 2:
        raise RuntimeError(
            "Gaussian copula tail context sampling requires at least 2 rows"
        )

    from scipy.special import ndtr

    scenario = str(config["scenario"])
    features = _feature_matrix(rows, scenario)
    z = _normal_score_pseudo_observations(features)
    corr = np.corrcoef(z, rowvar=False)
    corr = np.atleast_2d(corr).astype(np.float64)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    reg = max(float(config["copula_correlation_regularization"]), 0.0)
    corr = corr + np.eye(corr.shape[0], dtype=np.float64) * reg
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1.0e-8, None)
    corr = (eigvecs * eigvals[None, :]) @ eigvecs.T
    denom = np.sqrt(np.clip(np.diag(corr), 1.0e-12, None))
    corr = corr / denom[:, None] / denom[None, :]

    q = float(config["copula_marginal_clip_quantile"])
    q = min(max(q, 0.0), 0.49)
    lower = np.quantile(features, q, axis=0)
    upper = np.quantile(features, 1.0 - q, axis=0)
    center = np.median(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    standardized = (features - center) / scale

    sampled_z = rng.multivariate_normal(
        np.zeros(features.shape[1], dtype=np.float64),
        corr,
        size=int(count),
        check_valid="ignore",
    )
    sampled_u = np.clip(ndtr(sampled_z), 1.0e-6, 1.0 - 1.0e-6)

    sampled: list[dict[str, Any]] = []
    for idx in range(int(count)):
        target_feature = np.asarray(
            [
                np.quantile(features[:, col], sampled_u[idx, col])
                for col in range(features.shape[1])
            ],
            dtype=np.float64,
        )
        target_feature = np.clip(target_feature, lower, upper)
        if str(scenario) == "cut_in":
            target_feature[CUTIN_FINAL_LATERAL_OFFSET_IDX] = np.clip(
                target_feature[CUTIN_FINAL_LATERAL_OFFSET_IDX], -1.0, 1.0
            )
        target_standardized = (target_feature - center) / scale
        distance = np.sum(
            (standardized - target_standardized[None, :]) ** 2,
            axis=1,
        )
        base_idx = int(np.argmin(distance))
        base = rows[base_idx]
        item = dict(base)
        scenario_conditions, initial_states = _reconstruct_initial_from_feature(
            base,
            target_feature,
            scenario,
        )
        item["scenario_conditions"] = scenario_conditions
        item["initial_states"] = initial_states
        item["source_type"] = SOURCE_TAIL_GAUSSIAN_COPULA
        item["event_id"] = (
            f"gaussian_copula_tail_{idx:05d}_base_{base['event_id']}"
        )
        item["base_context_index"] = base_idx
        item["base_event_id"] = str(base["event_id"])
        item["synthetic_context"] = 1
        item["context_model_method"] = CONTEXT_METHOD_GAUSSIAN_COPULA
        item["context_feature_distance"] = float(np.sqrt(distance[base_idx]))
        new_feature = _tail_feature(item, scenario)
        if scenario == "cut_in":
            item["initial_gap"] = float(np.exp(new_feature[CUTIN_LOG_GAP_IDX]))
            item["initial_closing_speed"] = float(new_feature[CUTIN_DELTA_VX_IDX])
        else:
            item["initial_gap"] = float(np.exp(new_feature[FOLLOWING_LOG_GAP_IDX]))
            item["initial_closing_speed"] = float(new_feature[FOLLOWING_DELTA_V_IDX])
        sampled.append(item)
    return sampled


def _diffusion_generated_path(config: dict[str, Any], tail_context_path: Path) -> Path:
    configured = config.get("generated_scenarios_path")
    if configured is not None:
        return Path(configured)
    return tail_context_path.parent / "diffusion_generated_scenarios.npz"


def _project_following_jerk_actions(
    action_array: np.ndarray,
    initial_states: np.ndarray,
    conditions: np.ndarray,
    action_cfg: dict[str, Any],
    *,
    dt: float,
    iterations: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(action_array, dtype=np.float32).copy()
    initial = np.asarray(initial_states, dtype=np.float32)
    cond = np.asarray(conditions, dtype=np.float32)
    ax_min = float(action_cfg.get("ax_min", -8.0))
    ax_max = float(action_cfg.get("ax_max", 4.0))
    jerk_abs_max = float(action_cfg.get("jerk_abs_max", 12.0))
    dt_safe = max(float(dt), 1.0e-6)
    prev_ax = initial[:, 1, 4].astype(np.float32)
    actions[:, :, 0] = np.clip(actions[:, :, 0], -jerk_abs_max, jerk_abs_max)
    ax = prev_ax[:, None] + np.cumsum(actions[:, :, 0], axis=1) * dt_safe
    ax = np.clip(ax, ax_min, ax_max).astype(np.float32)

    target_speed_change = cond[:, 4].astype(np.float32)
    target_min_ax = cond[:, 5].astype(np.float32)
    ax = np.maximum(ax, target_min_ax[:, None]).astype(np.float32)
    horizon = int(ax.shape[1])
    ramp = np.linspace(0.0, 1.0, horizon, dtype=np.float32)[None, :]
    denom = float(np.sum(ramp) * dt_safe)
    if denom > 1.0e-6:
        for _ in range(max(int(iterations), 0)):
            current = np.sum(ax, axis=1) * dt_safe
            correction = ((target_speed_change - current) / denom)[:, None] * ramp
            ax = np.clip(ax + correction, ax_min, ax_max).astype(np.float32)

    jerk = np.diff(
        np.concatenate([prev_ax[:, None], ax], axis=1),
        axis=1,
    ) / dt_safe
    jerk = np.clip(jerk, -jerk_abs_max, jerk_abs_max).astype(np.float32)
    ax = prev_ax[:, None] + np.cumsum(jerk, axis=1) * dt_safe
    ax = np.clip(ax, ax_min, ax_max).astype(np.float32)
    return jerk[:, :, None], ax


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p95": float(np.percentile(arr, 95.0)),
    }


def _comparison_stats(real: np.ndarray, generated: np.ndarray) -> dict[str, Any]:
    real_arr = np.asarray(real, dtype=np.float64)
    gen_arr = np.asarray(generated, dtype=np.float64)
    real_arr = real_arr[np.isfinite(real_arr)]
    gen_arr = gen_arr[np.isfinite(gen_arr)]
    out: dict[str, Any] = {
        "real": _summary_stats(real_arr),
        "generated": _summary_stats(gen_arr),
    }
    if real_arr.size and gen_arr.size:
        try:
            from scipy.stats import ks_2samp, wasserstein_distance
        except ImportError:
            out["wasserstein"] = float("nan")
            out["ks"] = float("nan")
        else:
            out["wasserstein"] = float(wasserstein_distance(real_arr, gen_arr))
            out["ks"] = float(ks_2samp(real_arr, gen_arr).statistic)
    else:
        out["wasserstein"] = float("nan")
        out["ks"] = float("nan")
    return out


def _string_array(values: np.ndarray) -> np.ndarray:
    return np.asarray([str(item) for item in values.tolist()], dtype=object)


def _following_lead_intrinsic_metrics(
    initial_states: np.ndarray,
    lead_trajectory: np.ndarray,
    *,
    dt: float,
) -> dict[str, np.ndarray]:
    init = np.asarray(initial_states, dtype=np.float64)
    lead = np.asarray(lead_trajectory, dtype=np.float64)
    if init.ndim != 3 or init.shape[1:] != (2, 6):
        raise ValueError(f"initial_states must have shape [N, 2, 6], got {init.shape}")
    if lead.ndim != 3 or lead.shape[0] != init.shape[0] or lead.shape[2] != 6:
        raise ValueError(
            "lead_trajectory must have shape [N, H, 6] matching initial_states, "
            f"got {lead.shape}"
        )
    lead0 = init[:, 1]
    ax = lead[:, :, 4]
    ax_with_initial = np.concatenate([lead0[:, None, 4], ax], axis=1)
    jerk = np.diff(ax_with_initial, axis=1) / max(float(dt), 1.0e-6)
    negative_ax = np.minimum(ax_with_initial, 0.0)
    return {
        "lead_initial_speed": lead0[:, 2],
        "lead_speed_change": lead[:, -1, 2] - lead0[:, 2],
        "lead_min_speed": np.min(lead[:, :, 2], axis=1),
        "lead_max_speed": np.max(lead[:, :, 2], axis=1),
        "lead_final_speed": lead[:, -1, 2],
        "lead_displacement": lead[:, -1, 0] - lead0[:, 0],
        "lead_min_ax": np.min(ax_with_initial, axis=1),
        "lead_max_ax": np.max(ax_with_initial, axis=1),
        "lead_mean_ax": np.mean(ax_with_initial, axis=1),
        "lead_accel_std": np.std(ax_with_initial, axis=1),
        "lead_mean_abs_ax": np.mean(np.abs(ax_with_initial), axis=1),
        "lead_braking_duration": np.sum(ax_with_initial < 0.0, axis=1) * float(dt),
        "lead_braking_impulse": -np.sum(negative_ax, axis=1) * float(dt),
        "max_abs_jerk": np.max(np.abs(jerk), axis=1),
        "mean_abs_jerk": np.mean(np.abs(jerk), axis=1),
    }


def _load_real_following_tail_lead_trajectories(
    *,
    tail_context_path: Path,
    segment_cache_path: Path,
    horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load empirical independent-tail lead trajectories aligned to generated horizon."""
    if not tail_context_path.exists():
        raise FileNotFoundError(f"Tail context file not found: {tail_context_path}")
    if not segment_cache_path.exists():
        raise FileNotFoundError(
            f"Following segment cache not found: {segment_cache_path}"
        )
    contexts = np.load(tail_context_path, allow_pickle=True)
    source_type = _string_array(contexts["source_type"])
    empirical_mask = source_type == SOURCE_INDEPENDENT_TAIL_PEAK
    event_ids = _string_array(contexts["event_id"])
    anchor_key = (
        "context_anchor_frame"
        if "context_anchor_frame" in contexts.files
        else "anchor_frame"
    )
    anchor_frames = np.asarray(contexts[anchor_key], dtype=np.int64)

    segments = np.load(segment_cache_path, allow_pickle=True)
    seg_event_ids = _string_array(segments["event_id"])
    seg_index = {event_id: idx for idx, event_id in enumerate(seg_event_ids)}
    offsets = np.asarray(segments["offset"], dtype=np.int64)
    lengths = np.asarray(segments["length"], dtype=np.int64)
    frames_all = np.asarray(segments["frames"], dtype=np.int64)
    states_all = np.asarray(segments["world_states"], dtype=np.float32)

    initial_rows: list[np.ndarray] = []
    lead_rows: list[np.ndarray] = []
    condition_rows: list[np.ndarray] = []
    conditions = np.asarray(contexts["scenario_conditions"], dtype=np.float32)
    horizon = int(horizon_steps)
    for idx in np.flatnonzero(empirical_mask):
        event_id = str(event_ids[int(idx)])
        seg_idx = seg_index.get(event_id)
        if seg_idx is None:
            continue
        offset = int(offsets[seg_idx])
        length = int(lengths[seg_idx])
        segment_frames = frames_all[offset : offset + length]
        anchor = int(anchor_frames[int(idx)])
        start_pos = int(np.searchsorted(segment_frames, anchor))
        if (
            start_pos < 0
            or start_pos >= length
            or int(segment_frames[start_pos]) != anchor
            or start_pos + horizon >= length
        ):
            continue
        world = states_all[offset + start_pos : offset + start_pos + horizon + 1]
        initial_rows.append(world[0].astype(np.float32))
        lead_rows.append(world[1:, 1].astype(np.float32))
        condition_rows.append(conditions[int(idx)].astype(np.float32))
    if not lead_rows:
        raise RuntimeError(
            "No empirical following tail trajectories could be aligned to the "
            f"{horizon}-step generated horizon"
        )
    return (
        np.asarray(initial_rows, dtype=np.float32),
        np.asarray(lead_rows, dtype=np.float32),
        np.asarray(condition_rows, dtype=np.float32),
    )


def _write_metric_hist_grid(
    *,
    real_metrics: dict[str, np.ndarray],
    generated_metrics: dict[str, np.ndarray],
    path: Path,
) -> None:
    plt = get_pyplot()

    names = [
        "lead_speed_change",
        "lead_min_ax",
        "lead_braking_duration",
        "lead_final_speed",
        "lead_displacement",
        "lead_mean_abs_ax",
        "lead_accel_std",
        "lead_braking_impulse",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.2))
    for ax, name in zip(axes.ravel(), names, strict=True):
        real = np.asarray(real_metrics[name], dtype=np.float64)
        generated = np.asarray(generated_metrics[name], dtype=np.float64)
        values = np.concatenate([real[np.isfinite(real)], generated[np.isfinite(generated)]])
        if values.size == 0:
            continue
        lo, hi = np.percentile(values, [1.0, 99.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = float(np.min(values)), float(np.max(values))
        bins = np.linspace(lo, hi, 36)
        ax.hist(
            real,
            bins=bins,
            density=True,
            histtype="bar",
            alpha=0.58,
            color=REAL_COLOR,
            label="highD tail",
        )
        ax.hist(
            generated,
            bins=bins,
            density=True,
            histtype="bar",
            alpha=0.50,
            color=GENERATED_COLOR,
            label="Diffusion",
        )
        ax.set_title(label_for(name))
        ax.set_ylabel("Density")
        style_axes(ax)
    axes.ravel()[0].legend(loc="best", frameon=False)
    fig.suptitle("Following lead metrics")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_lead_trajectory_family_plot(
    *,
    real_initial: np.ndarray,
    real_lead: np.ndarray,
    generated_initial: np.ndarray,
    generated_lead: np.ndarray,
    dt: float,
    path: Path,
) -> None:
    plt = get_pyplot()

    def displacement(initial: np.ndarray, lead: np.ndarray) -> np.ndarray:
        return np.asarray(lead[:, :, 0] - initial[:, None, 1, 0], dtype=np.float64)

    real_disp = displacement(real_initial, real_lead)
    gen_disp = displacement(generated_initial, generated_lead)
    horizon = min(real_disp.shape[1], gen_disp.shape[1])
    time = np.arange(horizon, dtype=np.float64) * float(dt)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), sharey=True)
    for ax, values, title, color in (
        (axes[0], real_disp[:, :horizon], "highD tail", REAL_COLOR),
        (axes[1], gen_disp[:, :horizon], "Diffusion", GENERATED_COLOR),
    ):
        q05, q25, q50, q75, q95 = np.percentile(values, [5, 25, 50, 75, 95], axis=0)
        ax.fill_between(time, q05, q95, color=color, alpha=0.14, label="5-95%")
        ax.fill_between(time, q25, q75, color=color, alpha=0.25, label="25-75%")
        ax.plot(time, q50, color=color, linewidth=2.0, label="median")
        ax.set_title(title)
        ax.set_xlabel(r"$t$ from anchor (s)")
        style_axes(ax)
    axes[0].set_ylabel(r"$\Delta x_{\mathrm{lead}}$ (m)")
    axes[0].legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_condition_hist_grid(
    *,
    real_conditions: np.ndarray,
    sampled_conditions: np.ndarray,
    condition_keys: list[str],
    path: Path,
) -> None:
    plt = get_pyplot()

    count = int(min(len(condition_keys), real_conditions.shape[1], sampled_conditions.shape[1]))
    cols = 3
    rows = int(np.ceil(count / cols))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(rows, cols, figsize=(13.0, 3.0 * rows))
    axes_flat = np.asarray(axes).ravel()
    for idx in range(count):
        ax = axes_flat[idx]
        real = np.asarray(real_conditions[:, idx], dtype=np.float64)
        sampled = np.asarray(sampled_conditions[:, idx], dtype=np.float64)
        values = np.concatenate([real[np.isfinite(real)], sampled[np.isfinite(sampled)]])
        if values.size == 0:
            continue
        lo, hi = np.percentile(values, [1.0, 99.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = float(np.min(values)), float(np.max(values))
        bins = np.linspace(lo, hi, 32)
        ax.hist(
            real,
            bins=bins,
            density=True,
            histtype="bar",
            alpha=0.58,
            color=REAL_COLOR,
            label="highD tail",
        )
        ax.hist(
            sampled,
            bins=bins,
            density=True,
            histtype="bar",
            alpha=0.50,
            color=GENERATED_COLOR,
            label="Copula input",
        )
        ax.set_title(label_for(str(condition_keys[idx])))
        ax.set_ylabel("Density")
        style_axes(ax)
    for ax in axes_flat[count:]:
        ax.axis("off")
    axes_flat[0].legend(loc="best", frameon=False)
    fig.suptitle("Following conditions")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _following_condition_consistency(
    *,
    generated_initial: np.ndarray,
    generated_metrics: dict[str, np.ndarray],
    generated_conditions: np.ndarray,
    condition_keys: list[str],
) -> dict[str, dict[str, float]]:
    key_to_idx = {str(key): idx for idx, key in enumerate(condition_keys)}
    checks: dict[str, np.ndarray] = {}
    init = np.asarray(generated_initial, dtype=np.float64)
    cond = np.asarray(generated_conditions, dtype=np.float64)

    def add(metric_name: str, condition_name: str, values: np.ndarray) -> None:
        idx = key_to_idx.get(condition_name)
        if idx is None or idx >= cond.shape[1]:
            return
        checks[f"{metric_name}_minus_condition"] = (
            np.asarray(values, dtype=np.float64) - cond[:, idx]
        )

    add("lead_ax_0", "lead_ax_0", init[:, 1, 4])
    add("lead_speed_change", "lead_speed_change", generated_metrics["lead_speed_change"])
    add("lead_min_ax", "lead_min_ax", generated_metrics["lead_min_ax"])
    add(
        "lead_braking_duration",
        "lead_braking_duration",
        generated_metrics["lead_braking_duration"],
    )
    return {
        key: {
            "mean_error": float(np.mean(value)),
            "mean_abs_error": float(np.mean(np.abs(value))),
            "p95_abs_error": float(np.percentile(np.abs(value), 95.0)),
        }
        for key, value in checks.items()
        if np.asarray(value).size > 0
    }


def _write_following_visualizations(
    *,
    tail_context_path: Path,
    generated_path: Path,
    config: dict[str, Any],
    dt: float,
) -> dict[str, Any]:
    if not generated_path.exists():
        return {}
    generated = np.load(generated_path, allow_pickle=True)
    if "lead_trajectory" not in generated.files:
        return {}
    output_dir = generated_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_lead = np.asarray(generated["lead_trajectory"], dtype=np.float32)
    generated_initial = np.asarray(generated["initial_states"], dtype=np.float32)
    horizon = int(generated_lead.shape[1])
    segment_path = Path(
        config.get(
            "following_segment_cache_path",
            Path(__file__).resolve().parents[2]
            / "results"
            / "highd_events"
            / "following_event_segments.npz",
        )
    )
    real_initial, real_lead, real_conditions = _load_real_following_tail_lead_trajectories(
        tail_context_path=tail_context_path,
        segment_cache_path=segment_path,
        horizon_steps=horizon,
    )
    real_metrics = _following_lead_intrinsic_metrics(real_initial, real_lead, dt=dt)
    generated_metrics = _following_lead_intrinsic_metrics(
        generated_initial,
        generated_lead,
        dt=dt,
    )
    metric_report = {
        key: _comparison_stats(real_metrics[key], generated_metrics[key])
        for key in sorted(real_metrics)
    }
    condition_keys = (
        [str(item) for item in generated["condition_keys"].tolist()]
        if "condition_keys" in generated.files
        else list(FOLLOWING_TAIL_FEATURE_NAMES)
    )
    generated_conditions = np.asarray(generated["scenario_conditions"], dtype=np.float32)
    condition_report = {
        key: _comparison_stats(real_conditions[:, idx], generated_conditions[:, idx])
        for idx, key in enumerate(condition_keys)
        if idx < real_conditions.shape[1] and idx < generated_conditions.shape[1]
    }
    condition_consistency = _following_condition_consistency(
        generated_initial=generated_initial,
        generated_metrics=generated_metrics,
        generated_conditions=generated_conditions,
        condition_keys=condition_keys,
    )
    paths = {
        "following_manoeuvre_tail_vs_generated": output_dir
        / "following_manoeuvre_tail_vs_generated.png",
        "following_longitudinal_trajectory_tail_vs_generated": output_dir
        / "following_longitudinal_trajectory_tail_vs_generated.png",
        "scenario_condition_tail_vs_copula_sampled": output_dir
        / "scenario_condition_tail_vs_copula_sampled.png",
    }
    _write_metric_hist_grid(
        real_metrics=real_metrics,
        generated_metrics=generated_metrics,
        path=paths["following_manoeuvre_tail_vs_generated"],
    )
    _write_lead_trajectory_family_plot(
        real_initial=real_initial,
        real_lead=real_lead,
        generated_initial=generated_initial,
        generated_lead=generated_lead,
        dt=dt,
        path=paths["following_longitudinal_trajectory_tail_vs_generated"],
    )
    _write_condition_hist_grid(
        real_conditions=real_conditions,
        sampled_conditions=generated_conditions,
        condition_keys=condition_keys,
        path=paths["scenario_condition_tail_vs_copula_sampled"],
    )
    report = {
        "scenario": "following",
        "real_reference": (
            "highD independent tail peak lead trajectories from "
            "following_event_segments.npz"
        ),
        "generated_reference": (
            "diffusion-generated scripted lead trajectories conditioned on "
            "sampled following scenario_conditions"
        ),
        "num_real_tail_samples": int(real_lead.shape[0]),
        "num_generated_samples": int(generated_lead.shape[0]),
        "horizon_steps": horizon,
        "horizon_seconds": float(horizon * float(dt)),
        "intrinsic_lead_trajectory_metrics": metric_report,
        "scenario_condition_metrics": condition_report,
        "generated_condition_consistency": condition_consistency,
        "figures": {key: str(value) for key, value in paths.items()},
        "interaction_risk_metrics": {},
        "note": (
            "Following select/generation only scripts the adversary lead "
            "trajectory. Gap, TTC, THW, final_gap, collision, and other "
            "ego-interaction metrics are intentionally excluded here; they "
            "must be computed only during playback/closed-loop evaluation "
            "after the highway-env IDM ego response is rolled out."
        ),
    }
    write_json(output_dir / "distribution_similarity_summary.json", report)
    return report


def _generate_diffusion_rollouts(
    selected: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    tail_context_path: Path,
) -> dict[str, Any] | None:
    if not bool(config["generate_diffusion_rollouts"]):
        return None
    if "diffusion_dataset_dir" not in config:
        raise KeyError(
            "Tail context config requires diffusion_dataset_dir when "
            "generate_diffusion_rollouts=true"
        )

    import torch

    from diffusion.src.kinematics import integrate_cutin_acceleration_actions
    from diffusion.src.utils import set_seed
    from tools.diffusion_adapter import DiffusionPriorAdapter
    from tools.normalization import denormalize_torch, normalize_numpy

    output_path = _diffusion_generated_path(config, tail_context_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    natural_dir = Path(config["diffusion_dataset_dir"])
    checkpoint = Path(config["diffusion_checkpoint_path"])
    set_seed(int(config["diffusion_seed"]))
    adapter = DiffusionPriorAdapter.load(
        natural_dir,
        checkpoint,
        device=str(config["diffusion_device"]),
    )
    schema = adapter.schema
    event_type = str(schema.get("event_type", "")).lower()
    if event_type not in {"cut_in", "following"}:
        raise RuntimeError(
            "Configured diffusion checkpoint is not a supported prior: "
            f"{schema.get('event_type')}"
        )
    if str(config["scenario"]) != event_type:
        raise ValueError(
            "Tail context scenario and diffusion checkpoint event_type disagree: "
            f"{config['scenario']} vs {event_type}"
        )

    requested = int(config["num_diffusion_scenarios"])
    if requested <= 0:
        requested = int(len(selected))
    rng = np.random.default_rng(int(config["diffusion_seed"]))
    replace = requested > len(selected)
    context_indices = rng.choice(
        np.arange(len(selected)),
        size=requested,
        replace=replace,
    )
    conditions = np.asarray(
        [selected[int(idx)]["scenario_conditions"] for idx in context_indices],
        dtype=np.float32,
    )
    initial_states = np.asarray(
        [selected[int(idx)]["initial_states"] for idx in context_indices],
        dtype=np.float32,
    )
    normalized_conditions = normalize_numpy(
        conditions,
        adapter.stats,
        "scenario_conditions",
    )
    batch_size = max(int(config["diffusion_batch_size"]), 1)
    inference_steps = config["diffusion_inference_steps"]
    if inference_steps is not None:
        inference_steps = int(inference_steps)

    actions: list[np.ndarray] = []
    guidance_scale = float(config.get("diffusion_guidance_scale", 0.0))
    adapter.model.eval()
    with torch.no_grad():
        for start in range(0, requested, batch_size):
            end = min(start + batch_size, requested)
            cond = torch.from_numpy(normalized_conditions[start:end]).float().to(
                adapter.device
            )
            if guidance_scale > 0.0:
                sample = adapter.model.sample_ddim_with_guidance(
                    int(end - start),
                    cond,
                    inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                )
            else:
                sample = adapter.model.sample_ddim(
                    int(end - start),
                    cond,
                    inference_steps=inference_steps,
                )
            decoded = denormalize_torch(sample, adapter.stats, "actions")
            actions.append(decoded.detach().cpu().numpy().astype(np.float32))
    action_array = np.concatenate(actions, axis=0)
    action_cfg = adapter.config.get("action", {})
    dt = float(schema["dt"])
    ego_lengths = np.asarray(
        [selected[int(idx)]["ego_length"] for idx in context_indices],
        dtype=np.float32,
    )
    adv_lengths = np.asarray(
        [selected[int(idx)]["adv_length"] for idx in context_indices],
        dtype=np.float32,
    )
    payload = {
        "context_index": context_indices.astype(np.int64),
        "scenario_conditions": conditions.astype(np.float32),
        "initial_states": initial_states.astype(np.float32),
        "actions": action_array.astype(np.float32),
        "ego_length": ego_lengths,
        "adv_length": adv_lengths,
        "source_type": np.asarray(
            [selected[int(idx)].get("source_type", "") for idx in context_indices],
            dtype=object,
        ),
        "base_event_id": np.asarray(
            [
                selected[int(idx)].get("base_event_id")
                or selected[int(idx)].get("event_id", "")
                for idx in context_indices
            ],
            dtype=object,
        ),
        "event_type": np.asarray(event_type, dtype=object),
        "condition_keys": np.asarray(
            [str(key) for key in schema.get("condition_keys", [])],
            dtype=object,
        ),
    }
    if event_type == "cut_in":
        projection_cfg = adapter.config.get("trajectory_projection", {})
        trajectories = integrate_cutin_acceleration_actions(
            initial_states,
            action_array,
            dt,
            ax_min=float(action_cfg.get("ax_min", -8.0)),
            ax_max=float(action_cfg.get("ax_max", 4.0)),
            ay_abs_max=float(action_cfg.get("ay_abs_max", 4.0)),
            speed_min=float(projection_cfg.get("speed_min", 0.0)),
            speed_max=float(projection_cfg.get("speed_max", 50.0)),
        )
        payload["target_trajectory"] = trajectories.astype(np.float32)
    else:
        rep = str(schema.get("action_representation", "")).lower()
        if rep == "jerk":
            action_array, ax = _project_following_jerk_actions(
                action_array,
                initial_states,
                conditions,
                action_cfg,
                dt=dt,
            )
            payload["actions"] = action_array.astype(np.float32)
        else:
            ax = action_array[:, :, 0]
        ax = np.clip(
            ax,
            float(action_cfg.get("ax_min", -8.0)),
            float(action_cfg.get("ax_max", 4.0)),
        ).astype(np.float32)
        lead0 = initial_states[:, 1].astype(np.float32)
        x = lead0[:, 0].copy()
        y = lead0[:, 1].copy()
        vx = np.maximum(lead0[:, 2], 0.0)
        vy = lead0[:, 3].copy()
        ay = lead0[:, 5].copy()
        lead_trajectory = np.zeros((requested, ax.shape[1], 6), dtype=np.float32)
        for step in range(ax.shape[1]):
            ax_step = ax[:, step]
            x = x + vx * dt + 0.5 * ax_step * dt * dt
            vx = np.maximum(vx + ax_step * dt, 0.0)
            lead_trajectory[:, step] = np.stack(
                [x, y, vx, vy, ax_step, ay],
                axis=-1,
            )
        payload["acceleration"] = ax.astype(np.float32)
        payload["lead_trajectory"] = lead_trajectory
    np.savez_compressed(output_path, **payload)
    summary = {
        "generated_scenarios": str(output_path),
        "num_generated_scenarios": int(requested),
        "diffusion_dataset_dir": str(natural_dir),
        "diffusion_checkpoint_path": str(checkpoint),
        "diffusion_inference_steps": inference_steps,
        "diffusion_batch_size": batch_size,
        "diffusion_seed": int(config["diffusion_seed"]),
        "sampler": "ddim",
        "following_action_projection": (
            "min_ax_floor_then_ramp_speed_change_projection_with_jerk_and_ax_clipping"
            if event_type == "following"
            and str(schema.get("action_representation", "")).lower() == "jerk"
            else None
        ),
        "ego_policy": "playback_highway_env_idm",
        "ego_trajectory_output": False,
    }
    write_json(
        output_path.with_name("diffusion_generated_scenarios_summary.json"),
        summary,
    )
    logger.info(
        "Wrote %d %s diffusion generated scenarios to %s",
        requested,
        event_type,
        output_path,
    )
    return summary


def _independent_peak_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    peaks_path = Path(config["independent_tail_peaks_path"])
    if not peaks_path.exists():
        estimate_script = str(
            config.get(
                "estimate_exposure_hint",
                "the scenario exposure estimation script",
            )
        )
        raise FileNotFoundError(
            "Independent highD tail peaks are required for strict tail context "
            f"selection: {peaks_path}. Run "
            f"{estimate_script} first."
        )
    import pandas as pd

    peaks = pd.read_csv(peaks_path)
    required = {"representative_event_id", "peak_id"}
    missing = sorted(required - set(peaks.columns))
    if missing:
        raise KeyError(f"{peaks_path} is missing required columns: {missing}")

    by_event_id = {str(row["event_id"]): row for row in rows}
    selected: list[dict[str, Any]] = []
    missing_events: list[str] = []
    for _, peak in peaks.iterrows():
        event_id = str(peak["representative_event_id"])
        source = by_event_id.get(event_id)
        if source is None:
            missing_events.append(event_id)
            continue
        item = dict(source)
        for key, value in peak.to_dict().items():
            if hasattr(value, "item"):
                value = value.item()
            item[key] = value
        selected.append(item)
    if missing_events:
        raise KeyError(
            "Independent peaks reference events not found in the context cache: "
            f"{missing_events[:10]} (total={len(missing_events)})"
        )
    if not selected:
        raise RuntimeError(
            "No independent tail peak contexts could be matched from "
            f"{peaks_path}"
        )
    return selected


def _save_outputs(
    rows: list[dict[str, Any]],
    evt_meta: dict[str, Any],
    input_source: str,
    config: dict[str, Any],
) -> None:
    tail_context_path = Path(config["tail_context_path"])
    tail_context_path.parent.mkdir(parents=True, exist_ok=True)

    empirical_context_limit = config["empirical_context_limit"]
    if empirical_context_limit is not None:
        empirical_context_limit = int(empirical_context_limit)
        if empirical_context_limit <= 0:
            raise ValueError("empirical_context_limit must be positive or None")
    score = np.asarray(
        [row["risk_score"] for row in rows],
        dtype=np.float32,
    )
    finite_score = score[np.isfinite(score)]
    if finite_score.size == 0:
        raise RuntimeError("No finite highD tail risk scores were produced")

    tail_threshold_u = float(evt_meta["evt_tail_threshold_u"])
    context_source = "independent_tail_peaks"
    candidate_rows = _independent_peak_rows(rows, config)
    source_type = SOURCE_INDEPENDENT_TAIL_PEAK
    tail_selection_method = "evt_pot_threshold_declustered_peaks"
    context_distribution = "uniform over selected highD independent tail peaks"

    num_available_tail_contexts = int(len(candidate_rows))
    for row in candidate_rows:
        row["source_type"] = source_type
        row.update(_SYNTHETIC_CONTEXT_FIELDS)
    if empirical_context_limit is not None:
        rng = np.random.default_rng(int(config["selection_random_seed"]))
        sample_size = min(empirical_context_limit, num_available_tail_contexts)
        chosen = rng.choice(
            np.arange(num_available_tail_contexts),
            size=sample_size,
            replace=False,
        )
        candidate_rows = [candidate_rows[int(idx)] for idx in chosen]
    if not candidate_rows:
        raise RuntimeError(
            "No tail contexts found above EVT POT threshold "
            f"u={tail_threshold_u:.6g}"
        )
    configured_distribution_path = config.get("condition_distribution_path")
    condition_distribution_path = (
        Path(configured_distribution_path)
        if configured_distribution_path is not None
        else tail_context_path.with_name("scenario_condition_distribution.npz")
    )
    _save_condition_distribution(
        condition_distribution_path,
        empirical_rows=candidate_rows,
        evt_meta=evt_meta,
        config=config,
    )
    selected = candidate_rows
    context_generation_method = str(
        config["context_generation_method"]
    )
    num_synthetic_contexts = int(config["num_synthetic_contexts"])
    if context_generation_method == CONTEXT_METHOD_GAUSSIAN_COPULA:
        rng = np.random.default_rng(int(config["selection_random_seed"]))
        synthetic_rows = _sample_gaussian_copula_contexts(
            candidate_rows,
            count=num_synthetic_contexts,
            rng=rng,
            config=config,
        )
        selected = (
            candidate_rows
            if bool(config["include_empirical_contexts"])
            else []
        ) + synthetic_rows
        context_distribution = (
            "empirical highD independent tail peaks plus samples from a "
            "Gaussian-copula joint distribution over diffusion scenario "
            "condition variables"
        )
    elif context_generation_method != CONTEXT_METHOD_EMPIRICAL:
        raise ValueError(
            f"Unsupported context_generation_method: {context_generation_method}"
        )
    tail_sampling_method = (
        "uniform_random_without_replacement"
        if empirical_context_limit is not None
        else f"all_{context_source}"
    )
    selected_score = np.asarray(
        [row["risk_score"] for row in selected],
        dtype=np.float32,
    )

    collision_critical_level = evt_meta["collision_critical_level"]
    if str(config["scenario"]) == "following":
        for row in selected:
            context_anchor = row.get("context_anchor_frame")
            if context_anchor is None:
                continue
            row["event_anchor_frame"] = int(row["anchor_frame"])
            row["anchor_frame"] = int(context_anchor)

    payload: dict[str, np.ndarray] = {
        "scenario_conditions": np.asarray(
            [row["scenario_conditions"] for row in selected],
            dtype=np.float32,
        ),
        "initial_states": np.asarray(
            [row["initial_states"] for row in selected],
            dtype=np.float32,
        ),
        "source_type": np.asarray(
            [row["source_type"] for row in selected],
            dtype=object,
        ),
        "tail_threshold": np.asarray(tail_threshold_u, dtype=np.float32),
        "collision_critical_level": np.asarray(
            collision_critical_level,
            dtype=np.float32,
        ),
    }
    context_key_dtypes = _context_key_dtypes(config)
    for key in _context_output_keys(config):
        if all(key in row for row in selected):
            payload[key] = np.asarray(
                [row[key] for row in selected],
                dtype=context_key_dtypes.get(key),
            )
    np.savez_compressed(tail_context_path, **payload)
    diffusion_summary = _generate_diffusion_rollouts(
        selected,
        config=config,
        tail_context_path=tail_context_path,
    )
    visualization_summary = None
    if diffusion_summary is not None and str(config["scenario"]) == "following":
        schema_path = Path(config["diffusion_dataset_dir"]) / "feature_schema.json"
        dt = 1.0 / 25.0
        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            dt = float(schema.get("dt", dt))
        visualization_summary = _write_following_visualizations(
            tail_context_path=tail_context_path,
            generated_path=Path(diffusion_summary["generated_scenarios"]),
            config=config,
            dt=dt,
        )

    num_output_contexts = int(len(selected))
    num_output_synthetic_contexts = int(
        sum(int(row["synthetic_context"]) for row in selected)
    )
    num_output_empirical_contexts = num_output_contexts - num_output_synthetic_contexts
    selected_fraction = len(candidate_rows) / max(num_available_tail_contexts, 1)
    write_json(
        tail_context_path.with_name("tail_context_summary.json"),
        {
            **evt_meta,
            "input_source": input_source,
            "context_source": context_source,
            "tail_contexts": str(tail_context_path),
            "condition_distribution": str(condition_distribution_path),
            "context_distribution": context_distribution,
            "num_scored_events": int(len(rows)),
            "num_available_tail_contexts": num_available_tail_contexts,
            "num_output_contexts": num_output_contexts,
            "empirical_context_limit": empirical_context_limit,
            "num_empirical_contexts": int(len(candidate_rows)),
            "num_output_empirical_contexts": num_output_empirical_contexts,
            "num_synthetic_contexts": num_output_synthetic_contexts,
            "selected_tail_fraction": float(selected_fraction),
            "tail_selection_method": tail_selection_method,
            "tail_sampling_method": tail_sampling_method,
            "context_generation_method": context_generation_method,
            "tail_feature_names": list(_tail_feature_names(config)),
            "copula_correlation_regularization": float(
                config["copula_correlation_regularization"]
            ),
            "copula_marginal_clip_quantile": float(
                config["copula_marginal_clip_quantile"]
            ),
            "diffusion_generation": diffusion_summary,
            "visualization_summary": visualization_summary,
            "selection_random_seed": int(config["selection_random_seed"]),
            "scenario": str(config["scenario"]),
            "risk_value_key": str(config["risk_value_key"]),
            "tail_threshold": tail_threshold_u,
            "score_min": float(np.min(selected_score)),
            "score_mean": float(np.mean(selected_score)),
            "score_p95": float(np.percentile(selected_score, 95.0)),
            "score_max": float(np.max(selected_score)),
            "min_future_steps": int(config["min_future_steps"]),
        },
    )
    logger.info(
        (
            "Wrote %d %s tail contexts to %s | scored_events=%d, "
            "available_real_tail_peaks=%d, output_real=%d, output_synthetic=%d"
        ),
        num_output_contexts,
        config["scenario"],
        tail_context_path,
        len(rows),
        num_available_tail_contexts,
        num_output_empirical_contexts,
        num_output_synthetic_contexts,
    )


def run_following_tail_generation(config: dict[str, Any]) -> None:
    """Build long-tail context space from an explicit scenario config."""
    cfg = _merged_config(config)
    rows, evt_meta, input_source = _load_rows(cfg)
    _save_outputs(rows, evt_meta, input_source, cfg)

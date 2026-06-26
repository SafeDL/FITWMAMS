"""Shared evaluation utilities for diffusion action priors.

Used by both ``evaluate_cutin_prior.py`` and ``evaluate_following_prior.py``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from .kinematics import integrate_following_actions
from .types import VehicleState
from tools.highd_longitudinal import highd_risk_config
from tools.risk import resolve_risk_scoring

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path / config helpers
# ---------------------------------------------------------------------------


def _resolve_output_dir(config: dict, config_dir: Path) -> Path:
    paths = config.get("paths", {})
    if "output_dir" not in paths:
        raise KeyError("Config paths.output_dir is required")
    return (config_dir / paths["output_dir"]).resolve()


def _resolve_checkpoint_path(checkpoint: str | None, output_dir: Path) -> Path:
    DEFAULT_CHECKPOINT_PATH = "checkpoints/best_noise_mse_train_val_test.pt"
    path = Path(checkpoint or DEFAULT_CHECKPOINT_PATH)
    if path.is_absolute():
        return path
    return (output_dir / path).resolve()


# ---------------------------------------------------------------------------
# Action encoding / decoding
# ---------------------------------------------------------------------------


def _decode_actions(x: np.ndarray, stats: dict) -> np.ndarray:
    norm = stats["actions"]
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    return (x * std + mean).astype(np.float32)


def _actions_to_ax(
    actions: np.ndarray,
    initial_states: np.ndarray,
    schema: dict,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    action_cfg = config["action"]
    rep = str(schema["action_representation"]).lower()
    ax_min = float(action_cfg["ax_min"])
    ax_max = float(action_cfg["ax_max"])
    dt = float(schema["dt"])
    if rep == "jerk":
        prev_ax = initial_states[:, 1, 4].astype(np.float32)
        ax = prev_ax[:, None] + np.cumsum(actions[:, :, 0], axis=1) * dt
    else:
        ax = actions[:, :, 0]
    ax = ax.astype(np.float32)
    return np.clip(ax, ax_min, ax_max).astype(np.float32), ax


def _actions_to_jerk(
    actions: np.ndarray,
    ax: np.ndarray,
    initial_states: np.ndarray,
    schema: dict,
    config: dict,
) -> np.ndarray:
    rep = str(schema["action_representation"]).lower()
    if rep == "jerk":
        return actions[:, :, 0].astype(np.float32)
    dt = float(schema["dt"])
    prev_ax = initial_states[:, 1, 4].astype(np.float32)
    return (np.diff(np.concatenate([prev_ax[:, None], ax], axis=1), axis=1) / max(dt, 1e-6)).astype(np.float32)


# ---------------------------------------------------------------------------
# Statistical summaries
# ---------------------------------------------------------------------------


def _summary(x: np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {f"{prefix}_{k}": float("nan") for k in ("mean", "std", "p05", "p50", "p95")}
    q05, q50, q95 = np.quantile(arr, [0.05, 0.50, 0.95])
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_p05": float(q05),
        f"{prefix}_p50": float(q50),
        f"{prefix}_p95": float(q95),
    }


def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    n = max(len(x), len(y))
    q = (np.arange(n, dtype=np.float64) + 0.5) / n
    xp = np.interp(q, (np.arange(len(x), dtype=np.float64) + 0.5) / len(x), x)
    yp = np.interp(q, (np.arange(len(y), dtype=np.float64) + 0.5) / len(y), y)
    return float(np.mean(np.abs(xp - yp)))


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    values = np.sort(np.unique(np.concatenate([x, y])))
    cdf_x = np.searchsorted(x, values, side="right") / len(x)
    cdf_y = np.searchsorted(y, values, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _dtw_distance(a: np.ndarray, b: np.ndarray, window: int | None = None) -> float:
    """Dynamic Time Warping distance with optional Sakoe-Chiba band constraint.

    Parameters
    ----------
    a, b : 1-D arrays of equal or different lengths.
    window : int or None
        Sakoe-Chiba band half-width. If None, uses max(len(a), len(b)).

    Returns
    -------
    float
        Normalized DTW distance (divided by path length).
    """
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return float("nan")
    if n == 1 and m == 1:
        return float(abs(x[0] - y[0]))
    w = int(window) if window is not None else max(n, m)
    w = max(w, abs(n - m))
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - w)
        hi = min(m, i + w)
        for j in range(lo, hi + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    path_len = float(n + m)
    return float(dtw[n, m]) / path_len


def _histogram_l1(a: np.ndarray, b: np.ndarray, bins: int = 60) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    if hi <= lo:
        return 0.0
    hx, edges = np.histogram(x, bins=bins, range=(lo, hi), density=True)
    hy, _ = np.histogram(y, bins=edges, density=True)
    width = float(edges[1] - edges[0])
    return float(np.sum(np.abs(hx - hy)) * width)


def _spectral_l1(real: np.ndarray, gen: np.ndarray) -> float:
    real_arr = np.asarray(real, dtype=np.float64)
    gen_arr = np.asarray(gen, dtype=np.float64)
    if real_arr.shape != gen_arr.shape or real_arr.ndim != 2:
        return float("nan")
    real_amp = np.abs(np.fft.rfft(real_arr - real_arr.mean(axis=1, keepdims=True), axis=1))
    gen_amp = np.abs(np.fft.rfft(gen_arr - gen_arr.mean(axis=1, keepdims=True), axis=1))
    real_amp /= np.maximum(real_amp.sum(axis=1, keepdims=True), 1.0e-12)
    gen_amp /= np.maximum(gen_amp.sum(axis=1, keepdims=True), 1.0e-12)
    return float(np.mean(np.abs(real_amp - gen_amp)))


def _distribution_distance_metrics(real: np.ndarray, gen: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_wasserstein": _wasserstein_1d(real, gen),
        f"{prefix}_ks": _ks_statistic(real, gen),
        f"{prefix}_histogram_l1": _histogram_l1(real, gen),
    }


# ---------------------------------------------------------------------------
# Trajectory integration (following)
# ---------------------------------------------------------------------------


def _integrate_lead_batch(
    ax: np.ndarray,
    initial_states: np.ndarray,
    meta: dict[str, np.ndarray],
    schema: dict,
) -> np.ndarray:
    dt = float(schema["dt"])
    trajectories: list[np.ndarray] = []
    for i in range(ax.shape[0]):
        lead0 = initial_states[i, 1]
        lead_state = VehicleState(
            x=float(lead0[0]),
            y=float(lead0[1]),
            vx=float(lead0[2]),
            vy=float(lead0[3]),
            ax=float(lead0[4]),
            ay=float(lead0[5]),
        )
        lead = integrate_following_actions(lead_state, ax[i, :, None], dt)[1:]
        trajectories.append(lead)
    return np.stack(trajectories, axis=0)


def _integrate_target_batch(
    ax: np.ndarray,
    initial_states: np.ndarray,
    meta: dict[str, np.ndarray],
    schema: dict,
) -> np.ndarray:
    if str(schema.get("event_type", "")).lower() == "cut_in":
        raise ValueError("Cut-in evaluation expects direct maneuver trajectories")
    return _integrate_lead_batch(ax, initial_states, meta, schema)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _sample_actions(
    model,
    arrays: dict,
    idx: np.ndarray,
    device: torch.device,
    batch_size: int = 0,
) -> np.ndarray:
    batch = int(batch_size)
    if batch <= 0:
        batch = len(idx)
    chunks: list[np.ndarray] = []
    for start in range(0, len(idx), batch):
        sub_idx = idx[start:start + batch]
        scenario_conditions = torch.from_numpy(
            arrays["scenario_conditions"][sub_idx]
        ).float().to(device)
        sample = model.sample_ddim(
            len(sub_idx),
            scenario_conditions,
        )
        chunks.append(sample.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# Distribution metrics
# ---------------------------------------------------------------------------


def _distribution_metrics(
    real_ax: np.ndarray,
    gen_ax: np.ndarray,
    real_j: np.ndarray,
    gen_j: np.ndarray,
) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(_summary(real_ax, "real_ax"))
    out.update(_summary(gen_ax, "gen_ax"))
    out.update(_summary(real_j, "real_jerk"))
    out.update(_summary(gen_j, "gen_jerk"))
    out["ax_wasserstein"] = _wasserstein_1d(real_ax, gen_ax)
    out["jerk_wasserstein"] = _wasserstein_1d(real_j, gen_j)
    out["ax_ks"] = _ks_statistic(real_ax, gen_ax)
    out["jerk_ks"] = _ks_statistic(real_j, gen_j)
    out["ax_histogram_l1"] = _histogram_l1(real_ax, gen_ax)
    out["jerk_histogram_l1"] = _histogram_l1(real_j, gen_j)
    return out


# ---------------------------------------------------------------------------
# Feasibility metrics
# ---------------------------------------------------------------------------


def _feasibility_metrics(
    gen_unclipped_ax: np.ndarray,
    gen_jerk: np.ndarray,
    trajectories: np.ndarray,
    config: dict,
    dt: float = 0.04,
    is_cutin: bool = False,
) -> dict[str, float]:
    action_cfg = config["action"]
    ax_min = float(action_cfg["ax_min"])
    ax_max = float(action_cfg["ax_max"])
    jerk_abs_max = float(action_cfg["jerk_abs_max"])
    jumps = np.abs(np.diff(trajectories[:, :, 0], axis=1))
    out = {
        "action_clip_rate": float(
            np.mean((gen_unclipped_ax < ax_min) | (gen_unclipped_ax > ax_max))
        ),
        "speed_negative_rate": float(np.mean(trajectories[:, :, 2] < 0.0)),
        "jerk_violation_rate": float(np.mean(np.abs(gen_jerk) > jerk_abs_max)),
        "ax_violation_rate": float(
            np.mean((gen_unclipped_ax < ax_min) | (gen_unclipped_ax > ax_max))
        ),
        "trajectory_discontinuity_rate": float(
            np.mean(
                jumps
                > float(config["filters"]["max_position_jump"])
            )
        ),
    }
    if is_cutin:
        gen_ay = trajectories[:, :, 5].astype(np.float32)
        ay_abs_max = float(action_cfg.get("ay_abs_max", float("inf")))
        out["lateral_accel_violation_rate"] = float(np.mean(np.abs(gen_ay) > ay_abs_max))
        lateral_jerk_abs_max = float(action_cfg.get("lateral_jerk_abs_max", float("inf")))
        lateral_jerk = np.diff(gen_ay, axis=1) / max(float(dt), 1.0e-6)
        out["lateral_jerk_violation_rate"] = float(
            np.mean(np.abs(lateral_jerk) > lateral_jerk_abs_max)
        )
        # yaw rate from heading changes
        vx = trajectories[:, :, 2].astype(np.float64)
        vy = trajectories[:, :, 3].astype(np.float64)
        heading = np.unwrap(np.arctan2(vy, np.maximum(vx, 1.0e-6)), axis=1)
        yaw_rate = np.diff(heading, axis=1) / max(float(dt), 1.0e-6)
        max_yaw_rate = float(action_cfg.get("max_yaw_rate", float("inf")))
        out["yaw_rate_violation_rate"] = float(np.mean(np.abs(yaw_rate) > max_yaw_rate))
        # lateral displacement exceeding typical lane width
        lateral_disp = np.abs(trajectories[:, -1, 1] - trajectories[:, 0, 1])
        max_lane_width = float(action_cfg.get("max_lane_width", float("inf")))
        out["lateral_displacement_violation_rate"] = float(
            np.mean(lateral_disp > max_lane_width)
        )
    return out


# ---------------------------------------------------------------------------
# Trajectory naturalness metrics
# ---------------------------------------------------------------------------


def _trajectory_metrics(
    real_traj: np.ndarray,
    gen_traj: np.ndarray,
    target_anchor_x: np.ndarray,
    target_anchor_y: np.ndarray,
    is_cutin: bool = False,
) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(_summary(real_traj[:, :, 2], "real_lead_speed"))
    out.update(_summary(gen_traj[:, :, 2], "gen_lead_speed"))
    anchor_x = np.asarray(target_anchor_x, dtype=np.float32)
    anchor_y = np.asarray(target_anchor_y, dtype=np.float32)
    real_disp = real_traj[:, -1, 0] - anchor_x
    gen_disp = gen_traj[:, -1, 0] - anchor_x
    out.update(_summary(real_traj[:, -1, 2], "real_lead_final_speed"))
    out.update(_summary(gen_traj[:, -1, 2], "gen_lead_final_speed"))
    out.update(_summary(real_disp, "real_lead_displacement"))
    out.update(_summary(gen_disp, "gen_lead_displacement"))
    out.update(_distribution_distance_metrics(real_traj[:, :, 2], gen_traj[:, :, 2], "lead_speed"))
    out.update(_distribution_distance_metrics(real_traj[:, -1, 2], gen_traj[:, -1, 2], "lead_final_speed"))
    out.update(_distribution_distance_metrics(real_disp, gen_disp, "lead_displacement"))
    out["lead_speed_spectral_l1"] = _spectral_l1(real_traj[:, :, 2], gen_traj[:, :, 2])
    if is_cutin:
        real_lateral_disp = real_traj[:, -1, 1] - anchor_y
        gen_lateral_disp = gen_traj[:, -1, 1] - anchor_y
        out.update(_summary(real_traj[:, :, 3], "real_target_lateral_speed"))
        out.update(_summary(gen_traj[:, :, 3], "gen_target_lateral_speed"))
        out.update(_summary(real_traj[:, :, 5], "real_target_lateral_accel"))
        out.update(_summary(gen_traj[:, :, 5], "gen_target_lateral_accel"))
        out.update(_summary(real_lateral_disp, "real_final_lateral_offset"))
        out.update(_summary(gen_lateral_disp, "gen_final_lateral_offset"))
        out.update(_distribution_distance_metrics(real_traj[:, :, 3], gen_traj[:, :, 3], "target_lateral_speed"))
        out.update(_distribution_distance_metrics(real_traj[:, :, 5], gen_traj[:, :, 5], "target_lateral_accel"))
        out.update(_distribution_distance_metrics(real_lateral_disp, gen_lateral_disp, "final_lateral_offset"))
        out["target_lateral_position_spectral_l1"] = _spectral_l1(real_traj[:, :, 1], gen_traj[:, :, 1])
        out["target_lateral_speed_spectral_l1"] = _spectral_l1(real_traj[:, :, 3], gen_traj[:, :, 3])
    return out


# ---------------------------------------------------------------------------
# Interaction metrics
# ---------------------------------------------------------------------------


def _interaction_series(
    ego_traj: np.ndarray,
    lead_traj: np.ndarray,
    meta: dict[str, np.ndarray],
    config: dict,
) -> dict[str, np.ndarray]:
    half_lengths = 0.5 * (
        np.asarray(meta["ego_length"], dtype=np.float32)[:, None]
        + np.asarray(meta["adv_length"], dtype=np.float32)[:, None]
    )
    gap = lead_traj[:, :, 0] - ego_traj[:, :, 0] - half_lengths
    relative_speed = ego_traj[:, :, 2] - lead_traj[:, :, 2]
    lateral_offset = lead_traj[:, :, 1] - ego_traj[:, :, 1]
    relative_lateral_speed = ego_traj[:, :, 3] - lead_traj[:, :, 3]
    closing_speed = np.maximum(relative_speed, 0.0)
    eps = 1e-6
    ttc_cap = float(config["evaluation"]["ttc_cap"])
    thw_cap = float(config["evaluation"]["thw_cap"])
    ttc = np.where(closing_speed > eps, gap / np.maximum(closing_speed, eps), ttc_cap)
    thw = gap / np.maximum(ego_traj[:, :, 2], eps)
    return {
        "gap": gap.astype(np.float32),
        "ttc": np.clip(ttc, 0.0, ttc_cap).astype(np.float32),
        "thw": np.clip(thw, 0.0, thw_cap).astype(np.float32),
        "relative_speed": relative_speed.astype(np.float32),
        "lateral_offset": lateral_offset.astype(np.float32),
        "abs_lateral_offset": np.abs(lateral_offset).astype(np.float32),
        "relative_lateral_speed": relative_lateral_speed.astype(np.float32),
        "target_lateral_speed": lead_traj[:, :, 3].astype(np.float32),
        "closing_speed": closing_speed.astype(np.float32),
    }


def _interaction_metrics(
    real_interaction: dict[str, np.ndarray],
    gen_interaction: dict[str, np.ndarray],
    config: dict,
    is_cutin: bool = False,
) -> dict[str, float]:
    out: dict[str, float] = {}
    keys = [
        "gap",
        "ttc",
        "thw",
        "relative_speed",
        "closing_speed",
    ]
    if is_cutin:
        keys.extend(
            [
                "lateral_offset",
                "abs_lateral_offset",
                "relative_lateral_speed",
                "target_lateral_speed",
            ]
        )
    for key in keys:
        out.update(_summary(real_interaction[key], f"real_{key}"))
        out.update(_summary(gen_interaction[key], f"gen_{key}"))
        out.update(_distribution_distance_metrics(real_interaction[key], gen_interaction[key], key))
    real_min_gap = np.min(real_interaction["gap"], axis=1)
    gen_min_gap = np.min(gen_interaction["gap"], axis=1)
    real_final_gap = real_interaction["gap"][:, -1]
    gen_final_gap = gen_interaction["gap"][:, -1]
    real_min_ttc = np.min(real_interaction["ttc"], axis=1)
    gen_min_ttc = np.min(gen_interaction["ttc"], axis=1)
    rows = [
        (real_min_gap, gen_min_gap, "min_gap"),
        (real_final_gap, gen_final_gap, "final_gap"),
        (real_min_ttc, gen_min_ttc, "min_ttc"),
    ]
    if is_cutin:
        real_min_abs_lateral = np.min(real_interaction["abs_lateral_offset"], axis=1)
        gen_min_abs_lateral = np.min(gen_interaction["abs_lateral_offset"], axis=1)
        real_final_lateral = real_interaction["lateral_offset"][:, -1]
        gen_final_lateral = gen_interaction["lateral_offset"][:, -1]
        real_max_abs_lateral_speed = np.max(
            np.abs(real_interaction["target_lateral_speed"]),
            axis=1,
        )
        gen_max_abs_lateral_speed = np.max(
            np.abs(gen_interaction["target_lateral_speed"]),
            axis=1,
        )
        rows.extend(
            [
                (real_min_abs_lateral, gen_min_abs_lateral, "min_abs_lateral_offset"),
                (real_final_lateral, gen_final_lateral, "final_lateral_offset"),
                (real_max_abs_lateral_speed, gen_max_abs_lateral_speed, "max_abs_target_lateral_speed"),
            ]
        )
    for real, gen, key in rows:
        out.update(_summary(real, f"real_{key}"))
        out.update(_summary(gen, f"gen_{key}"))
        out.update(_distribution_distance_metrics(real, gen, key))
    near_gap = float(config["evaluation"]["near_collision_gap"])
    out["real_collision_rate"] = float(np.mean(real_interaction["gap"] <= 0.0))
    out["gen_collision_rate"] = float(np.mean(gen_interaction["gap"] <= 0.0))
    out["real_near_collision_rate"] = float(np.mean(real_interaction["gap"] < near_gap))
    out["gen_near_collision_rate"] = float(np.mean(gen_interaction["gap"] < near_gap))
    return out


# ---------------------------------------------------------------------------
# Rollout risk
# ---------------------------------------------------------------------------


def _softmax_pool_rows(value: np.ndarray, beta: float) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    scaled = float(beta) * arr
    scaled -= np.max(scaled, axis=1, keepdims=True)
    weights = np.exp(scaled)
    denom = np.maximum(np.sum(weights, axis=1), 1.0e-12)
    return np.sum(weights * arr, axis=1) / denom


def _rollout_risk_series(
    ego_traj: np.ndarray,
    lead_traj: np.ndarray,
    meta: dict[str, np.ndarray],
    config: dict,
    *,
    is_cutin: bool = False,
) -> dict[str, np.ndarray]:
    risk_cfg = highd_risk_config()
    risk_cfg["longitudinal_risk_scoring"].update(config.get("longitudinal_risk_scoring", {}))
    risk_cfg["closed_loop_risk"].update(config.get("closed_loop_risk", {}))
    scoring = resolve_risk_scoring(risk_cfg, "longitudinal_risk_scoring")
    closed_loop = risk_cfg["closed_loop_risk"]

    half_lengths = 0.5 * (
        np.asarray(meta["ego_length"], dtype=np.float64)[:, None]
        + np.asarray(meta["adv_length"], dtype=np.float64)[:, None]
    )
    gap = lead_traj[:, :, 0].astype(np.float64) - ego_traj[:, :, 0].astype(np.float64) - half_lengths
    ego_speed = ego_traj[:, :, 2].astype(np.float64)
    lead_speed = lead_traj[:, :, 2].astype(np.float64)
    ego_accel = ego_traj[:, :, 4].astype(np.float64)
    closing = ego_speed - lead_speed
    positive_closing = closing > 1.0e-6
    valid_gap = gap > 1.0e-6
    ttc = np.where(valid_gap & positive_closing, gap / np.maximum(closing, 1.0e-6), 1000.0)
    thw = np.where(valid_gap & (ego_speed > 1.0e-6), gap / np.maximum(ego_speed, 1.0e-6), 1000.0)
    drac = np.where(valid_gap & positive_closing, np.square(closing) / np.maximum(2.0 * gap, 1.0e-6), 0.0)

    ttc_raw = _softmax_pool_rows(1.0 / np.maximum(ttc, scoring["ttc_eps"]), scoring["pool_beta"])
    thw_raw = _softmax_pool_rows(1.0 / np.maximum(thw, scoring["thw_eps"]), scoring["pool_beta"])
    gap_raw = _softmax_pool_rows(1.0 / np.maximum(gap, scoring["gap_eps"]), scoring["pool_beta"])
    drac_raw = _softmax_pool_rows(np.maximum(drac, 0.0), scoring["pool_beta"])
    proxy = (
        scoring["ttc_weight"] * ttc_raw / max(scoring["ttc_scale"], 1.0e-6)
        + scoring["thw_weight"] * thw_raw / max(scoring["thw_scale"], 1.0e-6)
        + scoring["gap_weight"] * gap_raw / max(scoring["gap_scale"], 1.0e-6)
        + scoring["drac_weight"] * drac_raw / max(scoring["drac_scale"], 1.0e-6)
    )

    min_gap = np.min(gap, axis=1)
    min_ttc = np.min(np.clip(ttc, 0.0, 1000.0), axis=1)
    min_ego_accel = np.min(ego_accel, axis=1)
    hard_brake_threshold = float(closed_loop.get("hard_brake_threshold", -4.0))
    hard_brake = np.maximum(0.0, hard_brake_threshold - min_ego_accel) / max(abs(hard_brake_threshold), 1.0e-6)
    near_gap = float(config["evaluation"]["near_collision_gap"])
    y_long = (
        proxy
        + float(closed_loop.get("collision_bonus", 5.0)) * (min_gap <= 0.0)
        + float(closed_loop.get("near_collision_weight", 1.0)) * (min_gap < near_gap)
        + float(closed_loop.get("hard_brake_weight", 1.0)) * hard_brake
    )
    out = {
        "y_long": y_long.astype(np.float64),
        "proxy_risk_score": proxy.astype(np.float64),
        "min_gap": min_gap.astype(np.float64),
        "min_ttc": min_ttc.astype(np.float64),
        "final_gap": gap[:, -1].astype(np.float64),
        "final_lead_speed": lead_traj[:, -1, 2].astype(np.float64),
        "collision": (min_gap <= 0.0).astype(np.float64),
        "near_collision": (min_gap < near_gap).astype(np.float64),
    }
    if not is_cutin:
        return out

    cutin_cfg = config.get("cutin_risk", {})
    lateral = lead_traj[:, :, 1].astype(np.float64) - ego_traj[:, :, 1].astype(np.float64)
    if lead_traj.shape[-1] > 3 and ego_traj.shape[-1] > 3:
        lateral_velocity = (
            lead_traj[:, :, 3].astype(np.float64)
            - ego_traj[:, :, 3].astype(np.float64)
        )
    else:
        dt = 1.0 / max(float(config.get("sampling", {}).get("target_fps", 25.0)), 1.0e-6)
        lateral_velocity = np.diff(lateral, prepend=lateral[:, :1], axis=1) / max(dt, 1.0e-6)
    min_abs_lateral = np.min(np.abs(lateral), axis=1)
    final_abs_lateral = np.abs(lateral[:, -1])
    max_abs_lateral_velocity = np.max(np.abs(lateral_velocity), axis=1)
    lateral_offset_scale = max(float(cutin_cfg.get("lateral_offset_scale", 1.0)), 1.0e-6)
    lateral_offset_eps = max(float(cutin_cfg.get("lateral_offset_eps", 0.25)), 1.0e-6)
    lateral_velocity_scale = max(float(cutin_cfg.get("lateral_velocity_scale", 1.0)), 1.0e-6)
    lateral_weight = float(cutin_cfg.get("lateral_intrusion_weight", 1.5))
    duration_scale = max(float(cutin_cfg.get("cutin_duration_scale", 2.0)), 1.0e-6)
    dt = 1.0 / max(float(config.get("sampling", {}).get("target_fps", 25.0)), 1.0e-6)
    duration = max(float(lead_traj.shape[1]) * dt, float(cutin_cfg.get("cutin_duration_min_seconds", 0.1)))
    lateral_objective = (
        1.0 / np.maximum(min_abs_lateral / lateral_offset_scale, lateral_offset_eps)
        + max_abs_lateral_velocity / lateral_velocity_scale
        + duration_scale / duration
    )
    lateral_score = lateral_weight * lateral_objective
    y_cutin = y_long + lateral_score
    out.update(
        {
            "y_cutin": y_cutin.astype(np.float64),
            "lateral_intrusion_risk_score": lateral_score.astype(np.float64),
            "min_abs_lateral_offset": min_abs_lateral.astype(np.float64),
            "final_abs_lateral_offset": final_abs_lateral.astype(np.float64),
            "max_abs_lateral_velocity": max_abs_lateral_velocity.astype(np.float64),
        }
    )
    return out


def _rollout_shift_metrics(real_rollout: dict[str, np.ndarray], gen_rollout: dict[str, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {}
    risk_key = "y_cutin" if "y_cutin" in real_rollout and "y_cutin" in gen_rollout else "y_long"
    keys = [
        risk_key,
        "y_long",
        "proxy_risk_score",
        "min_gap",
        "min_ttc",
        "final_gap",
        "final_lead_speed",
    ]
    if risk_key == "y_cutin":
        keys.extend(
            [
                "lateral_intrusion_risk_score",
                "min_abs_lateral_offset",
                "final_abs_lateral_offset",
                "max_abs_lateral_velocity",
            ]
        )
    for key in dict.fromkeys(keys):
        if key not in real_rollout or key not in gen_rollout:
            continue
        out.update(_summary(real_rollout[key], f"real_{key}"))
        out.update(_summary(gen_rollout[key], f"gen_{key}"))
        out.update(_distribution_distance_metrics(real_rollout[key], gen_rollout[key], key))
    tail_threshold = float(np.quantile(real_rollout[risk_key], 0.90))
    out[f"real_{risk_key}_q90_threshold"] = tail_threshold
    out[f"gen_{risk_key}_above_real_q90_rate"] = float(np.mean(gen_rollout[risk_key] >= tail_threshold))
    out["real_collision_rate"] = float(np.mean(real_rollout["collision"]))
    out["gen_collision_rate"] = float(np.mean(gen_rollout["collision"]))
    out["real_near_collision_rate"] = float(np.mean(real_rollout["near_collision"]))
    out["gen_near_collision_rate"] = float(np.mean(gen_rollout["near_collision"]))
    return out


# ---------------------------------------------------------------------------
# Conditional sample quality
# ---------------------------------------------------------------------------


def _ensemble_crps(samples: np.ndarray, truth: np.ndarray, chunk_size: int = 256) -> float:
    total = 0.0
    count = 0
    for start in range(0, samples.shape[0], chunk_size):
        s = np.asarray(samples[start:start + chunk_size], dtype=np.float64)
        y = np.asarray(truth[start:start + chunk_size], dtype=np.float64)
        term1 = np.mean(np.abs(s - y[:, None, ...]), axis=1)
        pairwise = np.abs(s[:, :, None, ...] - s[:, None, :, ...])
        crps = term1 - 0.5 * np.mean(pairwise, axis=(1, 2))
        total += float(np.sum(crps))
        count += int(crps.size)
    return total / max(count, 1)


def _ensemble_interval_metrics(samples: np.ndarray, truth: np.ndarray, levels: tuple[float, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    arr = np.asarray(samples, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    for level in levels:
        lo_q = 0.5 * (1.0 - level)
        hi_q = 1.0 - lo_q
        lo = np.quantile(arr, lo_q, axis=1)
        hi = np.quantile(arr, hi_q, axis=1)
        key = int(round(level * 100))
        out[f"coverage_p{key}"] = float(np.mean((target >= lo) & (target <= hi)))
        out[f"interval_width_p{key}"] = float(np.mean(hi - lo))
    return out


def _conditional_sample_metrics(
    model,
    arrays: dict,
    idx: np.ndarray,
    device: torch.device,
    eval_cfg: dict,
) -> dict[str, float | int]:
    samples_per_context = int(eval_cfg.get("conditional_samples_per_context", 16))
    if samples_per_context < 2:
        raise ValueError("evaluation.conditional_samples_per_context must be at least 2")
    cond_idx = idx
    repeated = np.repeat(cond_idx, samples_per_context)
    batch_size = int(eval_cfg.get("sample_batch_size", 512))
    gen = _sample_actions(model, arrays, repeated, device, batch_size=batch_size)
    gen = gen.reshape(len(cond_idx), samples_per_context, *gen.shape[1:])
    truth = arrays["actions"][cond_idx].astype(np.float64)
    mean = np.mean(gen, axis=1)
    var = np.maximum(np.var(gen, axis=1), float(eval_cfg.get("conditional_nll_min_variance", 1.0e-4)))
    nll = 0.5 * (np.log(2.0 * np.pi * var) + np.square(truth - mean) / var)
    sample_mse = np.mean(np.square(gen - truth[:, None, ...]), axis=(2, 3))
    sample_l1 = np.mean(np.abs(gen - truth[:, None, ...]), axis=(2, 3))
    out: dict[str, float | int] = {
        "num_conditional_contexts": int(len(cond_idx)),
        "samples_per_context": int(samples_per_context),
        "conditional_diag_gaussian_nll": float(np.mean(nll)),
        "conditional_crps_action_norm": float(_ensemble_crps(gen, truth)),
        "ensemble_mean_mse_action_norm": float(np.mean(np.square(mean - truth))),
        "ensemble_mean_l1_action_norm": float(np.mean(np.abs(mean - truth))),
        "best_of_m_mse_action_norm": float(np.mean(np.min(sample_mse, axis=1))),
        "best_of_m_l1_action_norm": float(np.mean(np.min(sample_l1, axis=1))),
    }
    out.update(_ensemble_interval_metrics(gen, truth, (0.50, 0.80, 0.90, 0.95)))
    return out


# ---------------------------------------------------------------------------
# Trajectory reconstruction
# ---------------------------------------------------------------------------


def _trajectory_reconstruction_metrics(
    real_traj: np.ndarray,
    gen_traj: np.ndarray,
    dt: float = 0.04,
) -> dict[str, float]:
    """Compute per-sample MSE and DTW between real and generated trajectories.

    Evaluates both longitudinal and lateral trajectory reconstruction quality.
    """
    n_samples = real_traj.shape[0]
    out: dict[str, float] = {"num_reconstructed_samples": n_samples}
    # per-sample position MSE (longitudinal)
    x_mse = np.mean(np.square(gen_traj[:, :, 0] - real_traj[:, :, 0]), axis=1)
    out.update(_summary(x_mse, "per_sample_longitudinal_position_mse"))
    # per-sample lateral position MSE
    y_mse = np.mean(np.square(gen_traj[:, :, 1] - real_traj[:, :, 1]), axis=1)
    out.update(_summary(y_mse, "per_sample_lateral_position_mse"))
    # per-sample speed MSE
    vx_mse = np.mean(np.square(gen_traj[:, :, 2] - real_traj[:, :, 2]), axis=1)
    out.update(_summary(vx_mse, "per_sample_speed_mse"))
    # per-sample lateral speed MSE
    vy_mse = np.mean(np.square(gen_traj[:, :, 3] - real_traj[:, :, 3]), axis=1)
    out.update(_summary(vy_mse, "per_sample_lateral_speed_mse"))
    # DTW distances for longitudinal and lateral position
    dtw_x = np.array([
        _dtw_distance(gen_traj[i, :, 0], real_traj[i, :, 0])
        for i in range(n_samples)
    ])
    dtw_y = np.array([
        _dtw_distance(gen_traj[i, :, 1], real_traj[i, :, 1])
        for i in range(n_samples)
    ])
    out.update(_summary(dtw_x, "per_sample_longitudinal_dtw"))
    out.update(_summary(dtw_y, "per_sample_lateral_dtw"))
    # final position errors
    final_x_err = np.abs(gen_traj[:, -1, 0] - real_traj[:, -1, 0])
    final_y_err = np.abs(gen_traj[:, -1, 1] - real_traj[:, -1, 1])
    out.update(_summary(final_x_err, "per_sample_final_longitudinal_error"))
    out.update(_summary(final_y_err, "per_sample_final_lateral_error"))
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _write_plots(
    output_dir: Path,
    eval_cfg: dict,
    real_ax: np.ndarray,
    gen_ax: np.ndarray,
    real_j: np.ndarray,
    gen_j: np.ndarray,
    real_traj: np.ndarray,
    gen_traj: np.ndarray,
    real_gaps: np.ndarray,
    gen_gaps: np.ndarray,
    real_lateral_offsets: np.ndarray,
    gen_lateral_offsets: np.ndarray,
    real_relative_speed: np.ndarray,
    gen_relative_speed: np.ndarray,
    schema: dict,
) -> list[str]:
    plot_dir = output_dir / str(eval_cfg.get("plot_dir", "natural_prior_plots"))
    plot_dir.mkdir(parents=True, exist_ok=True)
    from tools.plot_style import (
        GENERATED_COLOR,
        REAL_COLOR,
        get_pyplot,
        style_axes,
    )

    plt = get_pyplot()

    written: list[Path] = []
    is_cutin = str(schema.get("event_type", "")).lower() == "cut_in"
    enabled_plots = (
        {
            "ax_distribution_real_vs_generated",
            "target_lateral_accel_distribution_real_vs_generated",
            "lateral_offset_distribution_real_vs_generated",
            "speed_distribution_real_vs_generated",
            "example_rollouts",
            "trajectory_reconstruction_errors",
        }
        if is_cutin
        else None
    )

    def should_plot(stem: str) -> bool:
        return enabled_plots is None or stem in enabled_plots

    if should_plot("ax_distribution_real_vs_generated"):
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.hist(real_ax.reshape(-1), bins=60, alpha=0.58, density=True, color=REAL_COLOR, label="highD")
        ax.hist(gen_ax.reshape(-1), bins=60, alpha=0.48, density=True, color=GENERATED_COLOR, label="Diffusion")
        ax.set_title(r"$a_x$ distribution")
        ax.set_xlabel(r"$a_x$ (m/s$^2$)")
        ax.set_ylabel("Density")
        style_axes(ax)
        ax.legend(frameon=False)
        path = plot_dir / "ax_distribution_real_vs_generated.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    if (
        should_plot("jerk_distribution_real_vs_generated")
        and not is_cutin
    ):
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.hist(real_j.reshape(-1), bins=60, alpha=0.58, density=True, color=REAL_COLOR, label="highD")
        ax.hist(gen_j.reshape(-1), bins=60, alpha=0.48, density=True, color=GENERATED_COLOR, label="Diffusion")
        ax.set_title(r"$j_x$ distribution")
        ax.set_xlabel(r"$j_x$ (m/s$^3$)")
        ax.set_ylabel("Density")
        style_axes(ax)
        ax.legend(frameon=False)
        path = plot_dir / "jerk_distribution_real_vs_generated.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    if should_plot("speed_distribution_real_vs_generated"):
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.hist(real_traj[:, :, 2].reshape(-1), bins=60, alpha=0.58, density=True, color=REAL_COLOR, label="highD")
        ax.hist(gen_traj[:, :, 2].reshape(-1), bins=60, alpha=0.48, density=True, color=GENERATED_COLOR, label="Diffusion")
        ax.set_title(r"$v_x$ distribution")
        ax.set_xlabel(r"$v_x$ (m/s)")
        ax.set_ylabel("Density")
        style_axes(ax)
        ax.legend(frameon=False)
        path = plot_dir / "speed_distribution_real_vs_generated.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    if is_cutin and should_plot("lateral_offset_distribution_real_vs_generated"):
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.hist(real_lateral_offsets.reshape(-1), bins=60, alpha=0.58, density=True, color=REAL_COLOR, label="highD")
        ax.hist(gen_lateral_offsets.reshape(-1), bins=60, alpha=0.48, density=True, color=GENERATED_COLOR, label="Diffusion")
        ax.set_title(r"$\Delta y$ distribution")
        ax.set_xlabel(r"$y_{\mathrm{tar}}-y_{\mathrm{ego}}$ (m)")
        ax.set_ylabel("Density")
        style_axes(ax)
        ax.legend(frameon=False)
        path = plot_dir / "lateral_offset_distribution_real_vs_generated.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    if should_plot("phase_space_vx_ax"):
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.scatter(
            real_traj[:, :, 2].reshape(-1),
            real_ax.reshape(-1),
            s=4,
            alpha=0.16,
            color=REAL_COLOR,
            label="highD",
        )
        ax.scatter(
            gen_traj[:, :, 2].reshape(-1),
            gen_ax.reshape(-1),
            s=4,
            alpha=0.16,
            color=GENERATED_COLOR,
            label="Diffusion",
        )
        ax.set_title(r"$v_x$-$a_x$ phase space")
        ax.set_xlabel(r"$v_x$ (m/s)")
        ax.set_ylabel(r"$a_x$ (m/s$^2$)")
        style_axes(ax)
        ax.legend(markerscale=3, frameon=False)
        path = plot_dir / "phase_space_vx_ax.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    if is_cutin and should_plot("phase_space_lateral_offset_vy"):
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.scatter(
            real_lateral_offsets.reshape(-1),
            real_traj[:, :, 3].reshape(-1),
            s=4,
            alpha=0.16,
            color=REAL_COLOR,
            label="highD",
        )
        ax.scatter(
            gen_lateral_offsets.reshape(-1),
            gen_traj[:, :, 3].reshape(-1),
            s=4,
            alpha=0.16,
            color=GENERATED_COLOR,
            label="Diffusion",
        )
        ax.set_title(r"$\Delta y$-$v_y$ phase space")
        ax.set_xlabel(r"$\Delta y$ (m)")
        ax.set_ylabel(r"$v_y$ (m/s)")
        style_axes(ax)
        ax.legend(markerscale=3, frameon=False)
        path = plot_dir / "phase_space_lateral_offset_vy.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    if should_plot("phase_space_gap_delta_v"):
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.scatter(
            real_gaps.reshape(-1),
            real_relative_speed.reshape(-1),
            s=4,
            alpha=0.16,
            color=REAL_COLOR,
            label="highD",
        )
        ax.scatter(
            gen_gaps.reshape(-1),
            gen_relative_speed.reshape(-1),
            s=4,
            alpha=0.16,
            color=GENERATED_COLOR,
            label="Diffusion",
        )
        ax.set_title(r"$g$-$\Delta v$ phase space")
        ax.set_xlabel(r"$g$ (m)")
        ax.set_ylabel(r"$\Delta v$ (m/s)")
        style_axes(ax)
        ax.legend(markerscale=3, frameon=False)
        path = plot_dir / "phase_space_gap_delta_v.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    if should_plot("example_rollouts"):
        n = min(6, gen_traj.shape[0])
        dt = float(schema["dt"])
        t = np.arange(gen_traj.shape[1], dtype=np.float32) * dt
        num_cols = 4 if is_cutin else 2
        fig, axes = plt.subplots(
            n,
            num_cols,
            figsize=(4 * num_cols, max(2.2 * n, 3)),
            constrained_layout=True,
            squeeze=False,
        )
        for i in range(n):
            if is_cutin:
                real_x_disp = real_traj[i, :, 0] - real_traj[i, 0, 0]
                gen_x_disp = gen_traj[i, :, 0] - gen_traj[i, 0, 0]
                real_y_disp = real_traj[i, :, 1] - real_traj[i, 0, 1]
                gen_y_disp = gen_traj[i, :, 1] - gen_traj[i, 0, 1]
                real_ay_i = real_traj[i, :, 5]
                gen_ay_i = gen_traj[i, :, 5]
                panels = (
                    (real_ax[i], gen_ax[i], r"$a_x$", r"$a_x$ (m/s$^2$)"),
                    (real_x_disp, gen_x_disp, r"$\Delta x$", r"$\Delta x$ (m)"),
                    (real_ay_i, gen_ay_i, r"$a_y$", r"$a_y$ (m/s$^2$)"),
                    (real_y_disp, gen_y_disp, r"$\Delta y$", r"$\Delta y$ (m)"),
                )
                for col, (real_series, gen_series, title, ylabel) in enumerate(panels):
                    axes[i, col].plot(t, real_series, color=REAL_COLOR, label="highD")
                    axes[i, col].plot(t, gen_series, color=GENERATED_COLOR, label="Diffusion")
                    axes[i, col].set_title(title if i == 0 else "")
                    axes[i, col].set_ylabel(ylabel)
                    style_axes(axes[i, col])
            else:
                axes[i, 0].plot(t, real_traj[i, :, 2], color=REAL_COLOR, label="highD")
                axes[i, 0].plot(t, gen_traj[i, :, 2], color=GENERATED_COLOR, label="Diffusion")
                axes[i, 0].set_ylabel(r"$v_x$ (m/s)")
                axes[i, 1].plot(t, real_gaps[i], color=REAL_COLOR, label="highD")
                axes[i, 1].plot(t, gen_gaps[i], color=GENERATED_COLOR, label="Diffusion")
                axes[i, 1].set_ylabel(r"$g$ (m)")
                style_axes(axes[i, 0])
                style_axes(axes[i, 1])
        axes[0, 0].legend(frameon=False)
        axes[0, 1].legend(frameon=False)
        if is_cutin:
            axes[0, 2].legend(frameon=False)
            axes[0, 3].legend(frameon=False)
        axes[-1, 0].set_xlabel(r"$t$ (s)")
        axes[-1, 1].set_xlabel(r"$t$ (s)")
        if is_cutin:
            axes[-1, 2].set_xlabel(r"$t$ (s)")
            axes[-1, 3].set_xlabel(r"$t$ (s)")
        path = plot_dir / "example_rollouts.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    if is_cutin:
        real_ay = real_traj[:, :, 5]
        gen_ay = gen_traj[:, :, 5]
        if should_plot("target_lateral_accel_distribution_real_vs_generated"):
            fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
            ax.hist(real_ay.reshape(-1), bins=60, alpha=0.58, density=True, color=REAL_COLOR, label="highD")
            ax.hist(gen_ay.reshape(-1), bins=60, alpha=0.48, density=True, color=GENERATED_COLOR, label="Diffusion")
            ax.set_title(r"$a_y$ distribution")
            ax.set_xlabel(r"$a_y$ (m/s$^2$)")
            ax.set_ylabel("Density")
            style_axes(ax)
            ax.legend(frameon=False)
            path = plot_dir / "target_lateral_accel_distribution_real_vs_generated.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            written.append(path)

        if should_plot("phase_space_vy_ay"):
            fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
            ax.scatter(
                real_traj[:, :, 3].reshape(-1),
                real_ay.reshape(-1),
                s=4, alpha=0.16, color=REAL_COLOR, label="highD",
            )
            ax.scatter(
                gen_traj[:, :, 3].reshape(-1),
                gen_ay.reshape(-1),
                s=4, alpha=0.16, color=GENERATED_COLOR, label="Diffusion",
            )
            ax.set_title(r"$v_y$-$a_y$ phase space")
            ax.set_xlabel(r"$v_y$ (m/s)")
            ax.set_ylabel(r"$a_y$ (m/s$^2$)")
            style_axes(ax)
            ax.legend(markerscale=3, frameon=False)
            path = plot_dir / "phase_space_vy_ay.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            written.append(path)

    # per-sample reconstruction error summary (both event types)
    if should_plot("trajectory_reconstruction_errors"):
        per_sample_x_rmse = np.sqrt(np.mean(np.square(gen_traj[:, :, 0] - real_traj[:, :, 0]), axis=1))
        per_sample_y_rmse = np.sqrt(np.mean(np.square(gen_traj[:, :, 1] - real_traj[:, :, 1]), axis=1))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        axes[0].hist(per_sample_x_rmse, bins=40, alpha=0.72, color=REAL_COLOR)
        axes[0].axvline(np.mean(per_sample_x_rmse), color="#333333", linestyle="--",
                        label=f"mean={np.mean(per_sample_x_rmse):.2f}")
        axes[0].set_title(r"$x$ RMSE")
        axes[0].set_xlabel(r"RMSE$_x$ (m)")
        axes[0].set_ylabel("Count")
        style_axes(axes[0])
        axes[0].legend(frameon=False)
        axes[1].hist(per_sample_y_rmse, bins=40, alpha=0.72, color=GENERATED_COLOR)
        axes[1].axvline(np.mean(per_sample_y_rmse), color="#333333", linestyle="--",
                        label=f"mean={np.mean(per_sample_y_rmse):.2f}")
        axes[1].set_title(r"$y$ RMSE")
        axes[1].set_xlabel(r"RMSE$_y$ (m)")
        axes[1].set_ylabel("Count")
        style_axes(axes[1])
        axes[1].legend(frameon=False)
        path = plot_dir / "trajectory_reconstruction_errors.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    return [str(p) for p in written]

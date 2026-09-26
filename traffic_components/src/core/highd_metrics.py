"""Model-agnostic highD factual, stochastic, and replay-audit metrics."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from .dynamics import KinematicTrafficDynamics


def semantic_cutin_agents(
    states: np.ndarray,
    valid: np.ndarray,
    *,
    lane_overlap_m: float = 1.0,
    initial_lane_offset_m: float = 1.8,
    minimum_post_steps: int = 50,
) -> np.ndarray:
    """Shared strict completed front cut-in detector for canonical highD windows."""
    values, present = np.asarray(states, np.float32), np.asarray(valid, bool)
    single = values.ndim == 3
    if single:
        values, present = values[None], present[None]
    if values.ndim != 4 or values.shape[2:] != (7, 6) or present.shape != values.shape[:3]:
        raise ValueError("states/valid must be [batch,time,7,6]/[batch,time,7]")
    relative_y = values[:, :, 1:, 1] - values[:, :, :1, 1]
    relative_x = values[:, :, 1:, 0] - values[:, :, :1, 0]
    overlap = np.abs(relative_y) <= float(lane_overlap_m)
    entered, first_entry = overlap.any(1), np.argmax(overlap, axis=1)
    frame = np.arange(values.shape[1])[None, :, None]
    after_entry = frame >= first_entry[:, None]
    retained = (overlap & after_entry).sum(1) / after_entry.sum(1) >= .7
    front_at_entry = np.take_along_axis(relative_x, first_entry[:, None], axis=1)[:, 0] > 0
    approached = np.min(np.abs(relative_y), axis=1) < (np.abs(relative_y[:, 0]) - .5)
    enough_post = first_entry <= values.shape[1] - int(minimum_post_steps)
    target_x, other_x = relative_x[:, :, :, None], relative_x[:, :, None, :]
    other_valid = present[:, :, None, 1:]
    candidate = np.arange(6)[None, None, :, None]
    other = np.arange(6)[None, None, None, :]
    post_window = (frame >= first_entry[:, None]) & (frame < first_entry[:, None] + int(minimum_post_steps))
    blocked = ((other_x > 0) & (other_x < target_x) & other_valid & (candidate != other) & post_window[..., None]).any((1, 3))
    semantic = present.all(1)[:, 1:] & (np.abs(relative_y[:, 0]) > float(initial_lane_offset_m)) & entered & retained & overlap[:, -1] & front_at_entry & approached & enough_post & ~blocked
    return semantic[0] if single else semantic


def factual_metrics(generated: np.ndarray, target: np.ndarray, active: np.ndarray) -> dict[str, float]:
    distance = np.linalg.norm(generated[..., 1:, :2] - target[..., 1:, :2], axis=-1)
    speed = np.abs(np.linalg.norm(generated[..., 1:, 2:4], axis=-1) - np.linalg.norm(target[..., 1:, 2:4], axis=-1))
    mask = np.broadcast_to(active[:, None], distance.shape)
    values = distance[mask]
    return {"ADE_m": float(values.mean()), "FDE_m": float(distance[:, -1][active].mean()), "P50_displacement_error_m": float(np.quantile(values, .5)), "P90_displacement_error_m": float(np.quantile(values, .9)), "P95_displacement_error_m": float(np.quantile(values, .95)), "P99_displacement_error_m": float(np.quantile(values, .99)), "speed_MAE_mps": float(speed[mask].mean()), "sequences": int(len(generated)), "frames": int(generated.shape[1])}


def temporal_factual_metrics(generated: np.ndarray, target: np.ndarray, active: np.ndarray) -> dict[str, list[float]]:
    distance = np.linalg.norm(generated[..., 1:, :2] - target[..., 1:, :2], axis=-1)
    speed = np.abs(np.linalg.norm(generated[..., 1:, 2:4], axis=-1) - np.linalg.norm(target[..., 1:, 2:4], axis=-1))
    return {"time_s": (0.04 * np.arange(1, generated.shape[1] + 1)).tolist(), "ADE_m": [float(distance[:, t][active].mean()) for t in range(distance.shape[1])], "P95_displacement_error_m": [float(np.quantile(distance[:, t][active], .95)) for t in range(distance.shape[1])], "speed_MAE_mps": [float(speed[:, t][active].mean()) for t in range(speed.shape[1])], "active_agent_frames": [int(active.sum()) for _ in range(distance.shape[1])]}


def ego_replay_metrics(generated: np.ndarray, target: np.ndarray) -> dict[str, float]:
    distance = np.linalg.norm(generated[..., 0, :2] - target[..., 0, :2], axis=-1)
    speed = np.abs(np.linalg.norm(generated[..., 0, 2:4], axis=-1) - np.linalg.norm(target[..., 0, 2:4], axis=-1))
    return {"ego_ADE_m": float(distance.mean()), "ego_FDE_m": float(distance[:, -1].mean()), "ego_FDE_P95_m": float(np.quantile(distance[:, -1], .95)), "ego_speed_MAE_mps": float(speed.mean())}


def ego_replay_gate(metrics: dict[str, float], limits: dict[str, float]) -> bool:
    return all(float(metrics[key]) <= float(value) for key, value in limits.items())


def stochastic_metrics(samples: list[np.ndarray], target: np.ndarray, active: np.ndarray) -> dict[str, float]:
    worlds = np.stack(samples, axis=1)
    error = np.linalg.norm(worlds[..., 1:, :2] - target[:, None, ..., 1:, :2], axis=-1)
    mask = np.broadcast_to(active[:, None, None], error.shape)
    per_ade = (error * mask).sum((2, 3)) / mask.sum((2, 3)).clip(1)
    per_fde = (error[:, :, -1] * active[:, None]).sum(-1) / active.sum(-1, keepdims=True).clip(1)
    pairwise = []
    for left in range(len(samples)):
        for right in range(left + 1, len(samples)):
            delta = np.linalg.norm(worlds[:, left, ..., 1:, :2] - worlds[:, right, ..., 1:, :2], axis=-1)
            pairwise.append(float((delta * np.broadcast_to(active[:, None], delta.shape)).sum() / np.broadcast_to(active[:, None], delta.shape).sum().clip(1)))
    return {"sample_mean_ADE_m": float(per_ade.mean()), "min_ADE_m": float(per_ade.min(1).mean()), "sample_mean_FDE_m": float(per_fde.mean()), "min_FDE_m": float(per_fde.min(1).mean()), "mean_pairwise_trajectory_distance_m": float(np.mean(pairwise) if pairwise else 0.0), "terminal_pairwise_distance_m": float(np.linalg.norm(worlds[:, :, -1, 1:, :2] - worlds[:, :1, -1, 1:, :2], axis=-1).mean()), "energy_score_proxy_m": float(per_ade.mean() - .5 * np.mean(pairwise) if pairwise else per_ade.mean())}


def paired_intervention_metrics(factual_actions: np.ndarray, intervention_actions: np.ndarray, active: np.ndarray) -> dict[str, Any]:
    delta = np.linalg.norm(intervention_actions - factual_actions, axis=-1)
    mask = np.broadcast_to(active[:, None], delta.shape)
    return {"background_action_change_mean": float(delta[mask].mean()), "background_action_change_p95": float(np.quantile(delta[mask], .95)), "response_nonzero_rate": float((delta[mask] > 1.0e-6).mean()), "longitudinal_wasserstein": float(wasserstein_distance(factual_actions[..., 0][mask], intervention_actions[..., 0][mask]))}


def _distribution_distance(real: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    """Shared, sample-weighted distribution diagnostic used by both models."""
    from scipy.stats import ks_2samp
    if not len(real) or not len(generated):
        return {"KS": float("nan"), "wasserstein": float("nan")}
    return {"KS": float(ks_2samp(real, generated).statistic), "wasserstein": float(wasserstein_distance(real, generated))}


def _risk_variables(states: np.ndarray, active: np.ndarray) -> dict[str, np.ndarray]:
    ego, background = states[..., :1, :], states[..., 1:, :]
    dx = background[..., 0] - ego[..., 0]
    same_lane_front = (dx > 0.0) & (np.abs(background[..., 1] - ego[..., 1]) < 1.8) & active[:, None]
    gap = np.maximum(dx - 4.8, 0.0)
    closing = np.maximum(ego[..., 2] - background[..., 2], 0.0)
    ttc = np.where(closing > 1.0e-3, gap / np.maximum(closing, 1.0e-3), 10.0)
    return {"gap_m": gap[same_lane_front], "TTC_s": np.minimum(ttc, 10.0)[same_lane_front]}


def _histogram_summary(real: np.ndarray, generated: np.ndarray, bounds: tuple[float, float], bins: int = 32) -> dict[str, Any]:
    edges = np.linspace(*bounds, bins + 1)
    real_counts, _ = np.histogram(np.clip(real, *bounds), bins=edges)
    generated_counts, _ = np.histogram(np.clip(generated, *bounds), bins=edges)
    r, g = real_counts / max(real_counts.sum(), 1), generated_counts / max(generated_counts.sum(), 1)
    return {"bin_edges": edges.tolist(), "real_probability": r.tolist(), "generated_probability": g.tolist(), "total_variation": float(.5 * np.abs(r - g).sum())}


def _windowed_jerk(
    actions: np.ndarray,
    time_axis: int,
    window_frames: int = 5,
    dt_s: float = 0.04,
) -> np.ndarray:
    values = np.moveaxis(np.asarray(actions), time_axis, -2)
    usable = values.shape[-2] // int(window_frames) * int(window_frames)
    blocks = values[..., :usable, :].reshape(
        *values.shape[:-2], usable // int(window_frames), int(window_frames), 2
    ).mean(-2)
    return np.moveaxis(
        np.diff(blocks, axis=-2) / (float(window_frames) * float(dt_s)),
        -2,
        time_axis,
    )


def _nearest_object_distance(states: np.ndarray, active: np.ndarray) -> np.ndarray:
    positions = states[..., :2]
    valid = np.concatenate((np.ones((len(active), 1), bool), active), 1)
    distance = np.linalg.norm(positions[..., :, None, :] - positions[..., None, :, :], axis=-1)
    pairs = valid[:, None, :, None] & valid[:, None, None, :]
    distance[np.broadcast_to(~pairs | np.eye(7, dtype=bool)[None, None], distance.shape)] = np.inf
    nearest = distance.min(-1)
    return nearest[np.broadcast_to(valid[:, None], nearest.shape)]


def _collision_indicator(states: np.ndarray, active: np.ndarray) -> np.ndarray:
    valid = np.concatenate((np.ones((len(active), 1), bool), active), 1)
    dx = np.abs(states[..., :, None, 0] - states[..., None, :, 0])
    dy = np.abs(states[..., :, None, 1] - states[..., None, :, 1])
    pairs = valid[:, None, :, None] & valid[:, None, None, :]
    collision = (dx < 4.8) & (dy < 1.8) & np.broadcast_to(pairs & np.triu(np.ones((7, 7), bool), 1)[None, None], dx.shape)
    return collision.any((-1, -2)).astype(np.float32).reshape(-1)


def distribution_metrics(
    samples: list[Any], initial_states: np.ndarray, target_states: np.ndarray,
    target_actions: np.ndarray, target_highd_actions: np.ndarray, active: np.ndarray,
) -> dict[str, Any]:
    """Common 16-sample generation, motion, and risk metrics.

    ``samples`` is deliberately duck-typed to the two models' rollout
    dataclasses; only ``states`` and ``background_actions`` are required.
    """
    worlds = np.stack([np.asarray(item.states) for item in samples], axis=1)
    actions = np.stack([np.asarray(item.background_actions) for item in samples], axis=1)
    distance = np.linalg.norm(worlds[..., 1:, :2] - target_states[:, None, ..., 1:, :2], axis=-1)
    valid = np.broadcast_to(active[:, None, None], distance.shape)
    ade = (distance * valid).sum((2, 3)) / valid.sum((2, 3)).clip(1)
    fde = (distance[:, :, -1] * active[:, None]).sum(2) / active.sum(1, keepdims=True).clip(1)
    pairwise, terminal = [], []
    trajectory_mask = np.broadcast_to(active[:, None], distance.shape[0:1] + distance.shape[2:])
    for left in range(len(samples)):
        for right in range(left + 1, len(samples)):
            delta = np.linalg.norm(worlds[:, left, :, 1:, :2] - worlds[:, right, :, 1:, :2], axis=-1)
            pairwise.append((delta * trajectory_mask).sum((1, 2)) / trajectory_mask.sum((1, 2)).clip(1))
            terminal.append((delta[:, -1] * active).sum(1) / active.sum(1).clip(1))
    pairwise_values = np.stack(pairwise) if pairwise else np.zeros((1, len(worlds)), np.float32)
    source = np.concatenate((np.broadcast_to(initial_states[:, None, None, 1:], (len(worlds), len(samples), 1, 6, 6)), worlds[:, :, :-1, 1:]), axis=2)
    generated_highd = KinematicTrafficDynamics.highd_actions(torch.from_numpy(actions), torch.from_numpy(source.copy())).numpy()
    real_mask = np.broadcast_to(active[:, None, :, None], target_highd_actions.shape)
    generated_mask = np.broadcast_to(active[:, None, None, :, None], generated_highd.shape)
    real_speed = np.linalg.norm(target_states[..., 1:, 2:4], axis=-1)
    generated_speed = np.linalg.norm(worlds[..., 1:, 2:4], axis=-1)
    speed_mask = np.broadcast_to(active[:, None], real_speed.shape)
    generated_speed_mask = np.broadcast_to(active[:, None, None], generated_speed.shape)
    real_risk = _risk_variables(target_states, active)
    generated_risk = _risk_variables(worlds.reshape(-1, 149, 7, 6), np.repeat(active, len(samples), axis=0))
    real_jerk, generated_jerk = np.diff(target_highd_actions, axis=1) / .04, np.diff(generated_highd, axis=2) / .04
    real_jerk_mask = np.broadcast_to(active[:, None, :, None], real_jerk.shape)
    generated_jerk_mask = np.broadcast_to(active[:, None, None, :, None], generated_jerk.shape)
    real_windowed, generated_windowed = _windowed_jerk(target_highd_actions, 1), _windowed_jerk(generated_highd, 2)
    real_windowed_mask = np.broadcast_to(active[:, None, :, None], real_windowed.shape)
    generated_windowed_mask = np.broadcast_to(active[:, None, None, :, None], generated_windowed.shape)
    real_yaw_accel, generated_yaw_accel = np.diff(target_actions[..., 1], axis=1) / .04, np.diff(actions[..., 1], axis=2) / .04
    real_yaw_mask = np.broadcast_to(active[:, None], real_yaw_accel.shape)
    generated_yaw_mask = np.broadcast_to(active[:, None, None], generated_yaw_accel.shape)
    real_nearest = _nearest_object_distance(target_states, active)
    generated_nearest = _nearest_object_distance(worlds.reshape(-1, 149, 7, 6), np.repeat(active, len(samples), 0))
    real_collision, generated_collision = _collision_indicator(target_states, active), _collision_indicator(worlds.reshape(-1, 149, 7, 6), np.repeat(active, len(samples), 0))
    real_accel = np.linalg.norm(target_highd_actions[real_mask].reshape(-1, 2), axis=-1)
    generated_accel = np.linalg.norm(generated_highd[generated_mask].reshape(-1, 2), axis=-1)
    return {
        "samples_per_condition": len(samples), "sample_mean_ADE_m": float(ade.mean()), "min_ADE_m": float(ade.min(1).mean()),
        "sample_mean_FDE_m": float(fde.mean()), "min_FDE_m": float(fde.min(1).mean()),
        "energy_score_m": float(ade.mean() - 0.5 * pairwise_values.mean()),
        "mean_pairwise_trajectory_distance_m": float(pairwise_values.mean() if pairwise else 0.0),
        "terminal_pairwise_distance_m": float(np.mean(terminal) if terminal else 0.0),
        "motion_distribution": {
            "speed": _distribution_distance(real_speed[speed_mask], generated_speed[generated_speed_mask]),
            "ax": _distribution_distance(target_highd_actions[..., 0][real_mask[..., 0]], generated_highd[..., 0][generated_mask[..., 0]]),
            "ay": _distribution_distance(target_highd_actions[..., 1][real_mask[..., 0]], generated_highd[..., 1][generated_mask[..., 0]]),
            "yaw_rate": _distribution_distance(target_actions[..., 1][np.broadcast_to(active[:, None], target_actions[..., 1].shape)], actions[..., 1][np.broadcast_to(active[:, None, None], actions[..., 1].shape)]),
            "jx": _distribution_distance(real_jerk[..., 0][real_jerk_mask[..., 0]], generated_jerk[..., 0][generated_jerk_mask[..., 0]]),
            "jy": _distribution_distance(real_jerk[..., 1][real_jerk_mask[..., 0]], generated_jerk[..., 1][generated_jerk_mask[..., 0]]),
        },
        "angular_distribution": {"yaw_rate": _distribution_distance(target_actions[..., 1][np.broadcast_to(active[:, None], target_actions[..., 1].shape)], actions[..., 1][np.broadcast_to(active[:, None, None], actions[..., 1].shape)]), "yaw_acceleration": _distribution_distance(real_yaw_accel[real_yaw_mask], generated_yaw_accel[generated_yaw_mask])},
        "jerk_resolution_diagnostic": {"raw_0p04s_highd_zero_mass": {"jx": float(np.isclose(real_jerk[..., 0][real_jerk_mask[..., 0]], 0).mean()), "jy": float(np.isclose(real_jerk[..., 1][real_jerk_mask[..., 0]], 0).mean())}, "windowed_0p2s": {"jx": _distribution_distance(real_windowed[..., 0][real_windowed_mask[..., 0]], generated_windowed[..., 0][generated_windowed_mask[..., 0]]), "jy": _distribution_distance(real_windowed[..., 1][real_windowed_mask[..., 0]], generated_windowed[..., 1][generated_windowed_mask[..., 0]])}},
        "risk_distribution": {name: _distribution_distance(real_risk[name], generated_risk[name]) for name in real_risk if len(real_risk[name]) and len(generated_risk[name])},
        "highd_adapted_realism": {"definition": "fixed-bin highD distribution diagnostics; not an official WOSAC score", "components": {"speed_mps": _histogram_summary(real_speed[speed_mask], generated_speed[generated_speed_mask], (0, 50)), "acceleration_magnitude_mps2": _histogram_summary(real_accel, generated_accel, (0, 10)), "yaw_rate_rps": _histogram_summary(target_actions[..., 1][np.broadcast_to(active[:, None], target_actions[..., 1].shape)], actions[..., 1][np.broadcast_to(active[:, None, None], actions[..., 1].shape)], (-.8, .8)), "yaw_acceleration_rps2": _histogram_summary(real_yaw_accel[real_yaw_mask], generated_yaw_accel[generated_yaw_mask], (-2, 2)), "nearest_object_distance_m": _histogram_summary(real_nearest, generated_nearest, (0, 80)), "gap_m": _histogram_summary(real_risk["gap_m"], generated_risk["gap_m"], (0, 80)), "TTC_s": _histogram_summary(real_risk["TTC_s"], generated_risk["TTC_s"], (0, 10)), "collision_incidence": _histogram_summary(real_collision, generated_collision, (0, 1), 2)}},
    }


def intervention_metrics(
    baseline: Any, mild: Any, strong: Any, initial: np.ndarray, active: np.ndarray,
    kind: str, natural_effects: np.ndarray | None = None,
) -> dict[str, Any]:
    """Common paired-CRN response/locality/dose diagnostic."""
    onset, committed = 25, 1
    current = initial[:, 24 + onset]
    ego, background = current[:, :1], current[:, 1:]
    relevance = np.exp(-np.abs(background[..., 0] - ego[..., 0]) / 35.0) * np.exp(-np.square(np.abs(background[..., 1] - ego[..., 1]) / 3.6)) * (0.6 + 0.4 * np.exp(-np.abs(ego[..., 2] - background[..., 2]) / 12.0)) * active
    near, far = relevance > 0.35, relevance < 0.1
    mild_delta = mild.background_actions[:, onset:] - baseline.background_actions[:, onset:]
    strong_delta = strong.background_actions[:, onset:] - baseline.background_actions[:, onset:]
    magnitude = np.linalg.norm(mild_delta, axis=-1)[:, committed:]
    near_mask, far_mask = np.broadcast_to(near[:, None], magnitude.shape), np.broadcast_to(far[:, None], magnitude.shape)
    near_value = float(magnitude[near_mask].mean()) if near_mask.any() else 0.0
    far_value = float(magnitude[far_mask].mean()) if far_mask.any() else 0.0
    near_profile = (magnitude * near[:, None]).sum((0, 2)) / near_mask.sum((0, 2)).clip(1)
    far_profile = (magnitude * far[:, None]).sum((0, 2)) / far_mask.sum((0, 2)).clip(1)
    detected = np.flatnonzero(near_profile > max(1.0e-6, .05 * float(near_profile.max())))
    response_frame = int(committed + detected[0]) if len(detected) else -1
    baseline_near = baseline.background_actions[:, onset + committed:, :, 0][near_mask]
    mild_near = mild.background_actions[:, onset + committed:, :, 0][near_mask]
    result: dict[str, Any] = {
        "committed_response_max_change": float(np.abs(mild_delta[:, :committed]).max()),
        "committed_response_invariant": bool(np.abs(mild_delta[:, :committed]).max() < 1e-8),
        "response_onset_frame_offset": response_frame,
        "response_latency_s": float(response_frame * .04) if response_frame >= 0 else float("nan"),
        "near_response_magnitude": float(near_value), "far_response_magnitude": float(far_value),
        "locality_ratio_far_to_near": float(far_value / max(near_value, 1e-8)),
        "strong_to_mild_response_ratio": float((np.linalg.norm(strong_delta[:, committed:], axis=-1)[near_mask].mean() if near_mask.any() else 0.0) / max(near_value, 1e-8)),
        "near_longitudinal_action_wasserstein": float(wasserstein_distance(baseline_near, mild_near)) if near_mask.any() else 0.0,
        "response_magnitude_profile": {"time_s": (.04 * np.arange(committed, mild_delta.shape[1])).tolist(), "near": near_profile.tolist(), "far": far_profile.tolist()},
    }
    if kind in {"brake", "accelerate"}:
        following = (current[:, 1:, 0] < current[:, :1, 0]) & (np.abs(current[:, 1:, 1] - current[:, :1, 1]) < 1.8) & active
        expected = -1.0 if kind == "brake" else 1.0
        mild_long, strong_long = mild_delta[:, committed:25, :, 0].mean(1), strong_delta[:, committed:25, :, 0].mean(1)
        result["direction_success_rate"] = float((expected * mild_long[following] > 0).mean()) if following.any() else 0.0
        result["dose_monotonicity_rate"] = float((expected * (strong_long - mild_long)[following] > 0).mean()) if following.any() else 0.0
        horizon = 20
        effect = expected * (mild.states[:, onset + horizon, 1:, 2] - baseline.states[:, onset + horizon, 1:, 2]) / (horizon * .04)
        if natural_effects is not None and following.any():
            natural = np.asarray(natural_effects, np.float32)
            lower, upper = np.quantile(natural, (.1, .9))
            result["response_distribution_wasserstein_mps2"] = float(wasserstein_distance(natural, effect[following]))
            result["response_within_natural_p10_p90_rate"] = float(((effect[following] >= lower) & (effect[following] <= upper)).mean())
        frame = onset + 24
        speed_base = np.linalg.norm(baseline.states[:, frame, 1:, 2:4], axis=-1)
        speed_mild = np.linalg.norm(mild.states[:, frame, 1:, 2:4], axis=-1)
        result["signed_follower_speed_response_mps"] = float((expected * (speed_mild - speed_base))[following].mean()) if following.any() else 0.0
    else:
        frame = onset + 24
        base_sep = np.abs(baseline.states[:, frame, 1:, 1] - baseline.states[:, frame, :1, 1])
        mild_sep = np.abs(mild.states[:, frame, 1:, 1] - mild.states[:, frame, :1, 1])
        result["separation_non_decrease_rate"] = float((mild_sep[near] >= base_sep[near]).mean()) if near.any() else 0.0
    return result


def intervention_dose_response(
    baseline: Any,
    treatments: dict[float, Any],
    initial: np.ndarray,
    active: np.ndarray,
    kind: str,
    natural_calibration: dict[str, Any],
) -> dict[str, Any]:
    """Shared multi-dose highD response curve at 0.2/0.4/0.8 s."""
    onset, horizons = 25, (0.2, 0.4, 0.8)
    current = initial[:, 24 + onset]
    if kind in {"brake", "accelerate"}:
        following = (current[:, 1:, 0] < current[:, :1, 0]) & (np.abs(current[:, 1:, 1] - current[:, :1, 1]) < 1.8) & active
        expected = -1.0 if kind == "brake" else 1.0
        metric = "signed_background_acceleration_effect_mps2"
        natural = {f"{h:.1f}s": natural_calibration["horizon_diagnostics"][f"{h:.1f}s"][kind]["effect_p10_p50_p90_mps2"] for h in horizons}
        def effect(rollout: Any, frames: int) -> np.ndarray:
            value = expected * (rollout.states[:, onset + frames, 1:, 2] - baseline.states[:, onset + frames, 1:, 2]) / (frames * .04)
            return value[following]
    else:
        dx = np.abs(current[:, 1:, 0] - current[:, :1, 0])
        dy = np.abs(current[:, 1:, 1] - current[:, :1, 1])
        closing = np.abs(current[:, :1, 2] - current[:, 1:, 2])
        near = (np.exp(-dx / 35) * np.exp(-np.square(dy / 3.6)) * (.6 + .4 * np.exp(-closing / 12)) * active) > .35
        metric, natural = "near_lateral_separation_change_m", None
        def effect(rollout: Any, frames: int) -> np.ndarray:
            base = np.abs(baseline.states[:, onset + frames, 1:, 1] - baseline.states[:, onset + frames, :1, 1])
            value = np.abs(rollout.states[:, onset + frames, 1:, 1] - rollout.states[:, onset + frames, :1, 1]) - base
            return value[near]
    doses: dict[str, Any] = {}
    for dose, rollout in treatments.items():
        values: dict[str, Any] = {}
        for horizon in horizons:
            result = effect(rollout, int(round(horizon / .04)))
            values[f"{horizon:.1f}s"] = {"p10_p50_p90": np.quantile(result, (.1, .5, .9)).tolist() if len(result) else [float("nan")] * 3, "mean": float(result.mean()) if len(result) else float("nan")}
        doses[f"{dose:g}"] = values
    return {"metric": metric, "horizons_s": list(horizons), "natural_p10_p50_p90": natural, "doses": doses}

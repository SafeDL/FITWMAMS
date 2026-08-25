"""Factual, stochastic and paired-intervention evaluation."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance

from diffusion.src.data import ANCHOR_INDEX, semantic_cutin_agents
from world_model.src.core.dynamics import KinematicTrafficDynamics
from world_model.src.core.utils import ensure_dir, load_json, save_json, select_device, set_seed

from .data import ego_controls, prepare_experiment_data
from .calibration import matched_response_calibration
from .planner import frozen_diffusion_plans, stochastic_diffusion_plan_samples
from .reference import response_relevance


@dataclass(frozen=True)
class Rollout:
    """Numpy rollout tensors and controls returned by the offline evaluator."""

    states: np.ndarray
    background_actions: np.ndarray
    ego_actions: np.ndarray
    reference_actions: np.ndarray


EgoActionPolicy = Callable[[dict[str, torch.Tensor | int]], torch.Tensor | np.ndarray]


def _logged_ego_actions(
    states: np.ndarray, valid: np.ndarray | None = None
) -> np.ndarray:
    actions = ego_controls(
        states[:, ANCHOR_INDEX:173, 0],
        states[:, ANCHOR_INDEX + 1 : 174, 0],
        0.04,
    )
    if valid is not None:
        present = valid[:, ANCHOR_INDEX:173, 0] & valid[:, ANCHOR_INDEX + 1 : 174, 0]
        actions = actions.copy()
        actions[~present] = 0.0
    return actions


def _intervene(
    actions: torch.Tensor,
    kind: str | None,
    dose: float,
) -> torch.Tensor:
    result = actions.clone()
    if kind is None:
        return result
    start, stop = 25, 50
    if kind == "brake":
        result[:, start:stop, 0] = (result[:, start:stop, 0] - dose).clamp_min(-8.0)
    elif kind == "accelerate":
        result[:, start:stop, 0] = (result[:, start:stop, 0] + dose).clamp_max(4.0)
    elif kind == "left":
        result[:, start:stop, 1] = (result[:, start:stop, 1] + dose).clamp_max(0.6)
    else:
        raise ValueError(f"unknown intervention {kind!r}")
    return result


@torch.no_grad()
def rollout(
    model,
    logged_states: np.ndarray,
    logged_valid: np.ndarray,
    soft_plans: np.ndarray,
    map_polylines: np.ndarray,
    map_polyline_valid: np.ndarray,
    *,
    device: torch.device,
    history_frames: int,
    motion_seed: int | None,
    intervention: str | None = None,
    dose: float = 0.0,
    ads_policy: EgoActionPolicy | None = None,
) -> Rollout:
    """Run one causal offline response rollout for a matched batch."""
    states = torch.from_numpy(logged_states[:, ANCHOR_INDEX].copy()).to(device)
    valid = torch.from_numpy(logged_valid[:, ANCHOR_INDEX].copy()).to(device)
    history = torch.from_numpy(
        logged_states[:, ANCHOR_INDEX - history_frames + 1 : ANCHOR_INDEX + 1].copy()
    ).to(device)
    history_valid = torch.from_numpy(
        logged_valid[:, ANCHOR_INDEX - history_frames + 1 : ANCHOR_INDEX + 1].copy()
    ).to(device)
    reference = torch.from_numpy(np.asarray(soft_plans, np.float32)).to(device)
    maps = torch.from_numpy(np.asarray(map_polylines, np.float32)).to(device)
    map_valid = torch.from_numpy(np.asarray(map_polyline_valid, bool)).to(device)
    initial_reference = states[:, 1:, :2].clone()
    logged_ego = torch.from_numpy(
        _logged_ego_actions(logged_states, logged_valid)
    ).to(device)
    scheduled_ego = _intervene(logged_ego, intervention, dose)
    historical_start = max(0, ANCHOR_INDEX - history_frames + 1)
    historical_ego_values = ego_controls(
        logged_states[:, historical_start:ANCHOR_INDEX, 0],
        logged_states[:, historical_start + 1 : ANCHOR_INDEX + 1, 0],
        0.04,
    )
    historical_present = (
        logged_valid[:, historical_start:ANCHOR_INDEX, 0]
        & logged_valid[:, historical_start + 1 : ANCHOR_INDEX + 1, 0]
    )
    historical_ego_values[~historical_present] = 0.0
    historical_ego = torch.from_numpy(historical_ego_values).to(device)
    generator = None
    if motion_seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(motion_seed))
    generated: list[torch.Tensor] = []
    background_actions: list[torch.Tensor] = []
    reference_actions: list[torch.Tensor] = []
    filter_state = None
    slow_scene = None
    slow_scene_noise = None
    agent_noise_state = None
    agent_style_state = None
    previous_current = None
    committed_ego_controls = historical_ego
    executed_ego: list[torch.Tensor] = []
    intervention_memory = None
    lateral_intervention_memory = None
    execute = model.cfg.execute_frames
    for start in range(0, 149, execute):
        count = min(execute, 149 - start)
        preview = reference[:, start : start + model.cfg.preview_frames]
        if preview.shape[1] < model.cfg.preview_frames:
            preview = torch.cat(
                (
                    preview,
                    preview[:, -1:].expand(
                        -1, model.cfg.preview_frames - preview.shape[1], -1, -1
                    ),
                ),
                dim=1,
            )
        base = initial_reference if start == 0 else reference[:, start - 1]
        ego_block = scheduled_ego[:, start : start + execute]
        if ads_policy is not None:
            proposed = torch.as_tensor(
                ads_policy(
                    {
                        "agent_states": states.detach().clone(),
                        "agent_valid": valid.detach().clone(),
                        "reference_index": start,
                    }
                ),
                dtype=states.dtype,
                device=device,
            )
            if proposed.shape == (len(states), 2):
                ego_block = proposed[:, None].expand(-1, execute, -1)
            elif proposed.shape == (len(states), execute, 2):
                ego_block = proposed
            else:
                raise ValueError(
                    "ads_policy must return [batch,2] or "
                    "[batch,execute_frames,2] controls"
                )
        if count < execute:
            ego_block = torch.cat(
                (
                    ego_block,
                    ego_block[:, -1:].expand(-1, execute - count, -1),
                ),
                dim=1,
            )
        scene_noise = torch.zeros(
            (len(states), model.cfg.scene_latent_dim),
            device=device,
            dtype=states.dtype,
        )
        agent_noise = torch.zeros(
            (len(states), 7, model.cfg.agent_latent_dim),
            device=device,
            dtype=states.dtype,
        )
        if generator is not None:
            scene_noise.normal_(generator=generator)
            agent_noise.normal_(generator=generator)
        response = model(
            history,
            history_valid,
            states,
            valid,
            preview,
            base,
            maps,
            map_valid,
            filter_state=filter_state,
            previous_current=previous_current,
            slow_scene=slow_scene,
            slow_scene_noise=slow_scene_noise,
            agent_noise_state=agent_noise_state,
            agent_style_state=agent_style_state,
            committed_ego_controls=committed_ego_controls,
            intervention_memory=intervention_memory,
            lateral_intervention_memory=lateral_intervention_memory,
            response_index=start // execute,
            scene_standard_normal=scene_noise,
            agent_standard_normal=agent_noise,
            deterministic=motion_seed is None,
        )
        filter_state = response.filter_state
        slow_scene = response.slow_scene
        slow_scene_noise = response.slow_scene_noise
        agent_noise_state = response.agent_noise_state
        agent_style_state = response.agent_style_state
        intervention_memory = response.intervention_memory
        lateral_intervention_memory = response.lateral_intervention_memory
        previous_current = states
        new_frames: list[torch.Tensor] = []
        for frame in range(count):
            controls = torch.cat(
                (ego_block[:, frame, None], response.actions[:, frame]), dim=1
            )
            states = model.dynamics.step(states, controls, valid, model.cfg.dt_s)
            new_frames.append(states)
        executed_ego.append(ego_block[:, :count])
        committed_ego_controls = torch.cat(
            (committed_ego_controls, ego_block[:, :count]), dim=1
        )[:, -model.cfg.intervention_trigger_history_frames - 1 :]
        block = torch.stack(new_frames, dim=1)
        generated.append(block)
        background_actions.append(response.actions[:, :count])
        reference_actions.append(response.reference_actions[:, :count])
        block_valid = valid[:, None].expand(-1, count, -1)
        history = torch.cat((history, block), dim=1)[:, -history_frames:]
        history_valid = torch.cat((history_valid, block_valid), dim=1)[
            :, -history_frames:
        ]
    return Rollout(
        torch.cat(generated, dim=1).cpu().numpy(),
        torch.cat(background_actions, dim=1).cpu().numpy(),
        torch.cat(executed_ego, dim=1).cpu().numpy(),
        torch.cat(reference_actions, dim=1).cpu().numpy(),
    )


def _factual_metrics(
    generated: np.ndarray,
    target: np.ndarray,
    active: np.ndarray,
) -> dict[str, float]:
    distance = np.linalg.norm(generated[..., 1:, :2] - target[..., 1:, :2], axis=-1)
    mask = np.broadcast_to(active[:, None], distance.shape)
    speed_error = np.abs(
        np.linalg.norm(generated[..., 1:, 2:4], axis=-1)
        - np.linalg.norm(target[..., 1:, 2:4], axis=-1)
    )
    valid_distance = distance[mask]
    return {
        "ADE_m": float(distance[mask].mean()),
        "FDE_m": float(distance[:, -1][active].mean()),
        "P50_displacement_error_m": float(np.quantile(valid_distance, 0.50)),
        "P90_displacement_error_m": float(np.quantile(valid_distance, 0.90)),
        "P95_displacement_error_m": float(np.quantile(valid_distance, 0.95)),
        "P99_displacement_error_m": float(np.quantile(valid_distance, 0.99)),
        "speed_MAE_mps": float(speed_error[mask].mean()),
        "sequences": int(len(generated)),
        "frames": int(generated.shape[1]),
    }


def _temporal_factual_metrics(
    generated: np.ndarray,
    target: np.ndarray,
    active: np.ndarray,
) -> dict[str, list[float]]:
    """Return per-horizon errors for drift rather than only end-point summaries."""
    distance = np.linalg.norm(generated[..., 1:, :2] - target[..., 1:, :2], axis=-1)
    speed_error = np.abs(
        np.linalg.norm(generated[..., 1:, 2:4], axis=-1)
        - np.linalg.norm(target[..., 1:, 2:4], axis=-1)
    )
    mask = np.broadcast_to(active[:, None], distance.shape)
    return {
        "time_s": (0.04 * np.arange(1, generated.shape[1] + 1)).tolist(),
        "ADE_m": [
            float(distance[:, step][active].mean()) for step in range(distance.shape[1])
        ],
        "P95_displacement_error_m": [
            float(np.quantile(distance[:, step][active], 0.95))
            for step in range(distance.shape[1])
        ],
        "speed_MAE_mps": [
            float(speed_error[:, step][active].mean())
            for step in range(speed_error.shape[1])
        ],
        "active_agent_frames": [
            int(mask[:, step].sum()) for step in range(mask.shape[1])
        ],
    }


def _factual_event_strata(
    generated: np.ndarray,
    target: np.ndarray,
    active: np.ndarray,
    evt_tail: np.ndarray,
    semantic_cutin: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Report conditional reconstruction errors on disjoint, declared test strata."""
    strata = {
        "all_natural": np.ones(len(generated), dtype=bool),
        "evt_labelled": np.asarray(evt_tail, bool),
        "semantic_cutin": np.asarray(semantic_cutin, bool),
    }
    report = {}
    for name, selected in strata.items():
        if not selected.any():
            raise RuntimeError(f"held-out factual stratum is empty: {name}")
        report[name] = _factual_metrics(
            generated[selected], target[selected], active[selected]
        )
    return report


def _distribution_distance(real: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    return {
        "KS": float(ks_2samp(real, generated).statistic),
        "wasserstein": float(wasserstein_distance(real, generated)),
    }


def _histogram_summary(
    real: np.ndarray,
    generated: np.ndarray,
    *,
    bounds: tuple[float, float],
    bins: int = 32,
) -> dict[str, Any]:
    """Fixed-bin densities make distribution figures comparable between runs."""
    edges = np.linspace(*bounds, bins + 1)
    real_counts, _ = np.histogram(np.clip(real, *bounds), bins=edges)
    generated_counts, _ = np.histogram(np.clip(generated, *bounds), bins=edges)
    real_density = real_counts / max(real_counts.sum(), 1)
    generated_density = generated_counts / max(generated_counts.sum(), 1)
    return {
        "bin_edges": edges.tolist(),
        "real_probability": real_density.tolist(),
        "generated_probability": generated_density.tolist(),
        "total_variation": float(0.5 * np.abs(real_density - generated_density).sum()),
    }


def _nearest_object_distance(states: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Nearest-centre distance for the ego plus valid background vehicles."""
    positions = np.asarray(states[..., :2], np.float32)
    valid = np.concatenate((np.ones((len(active), 1), bool), active), axis=1)
    delta = positions[..., :, None, :] - positions[..., None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    pair_valid = valid[:, None, :, None] & valid[:, None, None, :]
    diagonal = np.eye(7, dtype=bool)[None, None]
    invalid_pairs = np.broadcast_to(~pair_valid | diagonal, distance.shape)
    distance[invalid_pairs] = np.inf
    nearest = distance.min(axis=-1)
    return nearest[np.broadcast_to(valid[:, None], nearest.shape)]


def _collision_indicator(states: np.ndarray, active: np.ndarray) -> np.ndarray:
    """HighD-adapted footprint-overlap diagnostic; not an official WOSAC metric."""
    positions = np.asarray(states[..., :2], np.float32)
    valid = np.concatenate((np.ones((len(active), 1), bool), active), axis=1)
    dx = np.abs(positions[..., :, None, 0] - positions[..., None, :, 0])
    dy = np.abs(positions[..., :, None, 1] - positions[..., None, :, 1])
    pair_valid = valid[:, None, :, None] & valid[:, None, None, :]
    upper_triangle = np.triu(np.ones((7, 7), bool), k=1)[None, None]
    collision = (
        (dx < 4.8) & (dy < 1.8) & np.broadcast_to(pair_valid & upper_triangle, dx.shape)
    )
    return collision.any(axis=(-1, -2)).astype(np.float32).reshape(-1)


def _windowed_jerk(
    actions: np.ndarray,
    *,
    time_axis: int,
    window_frames: int = 5,
    dt_s: float = 0.04,
) -> np.ndarray:
    """Differentiate short action averages to reduce highD frame quantization."""
    values = np.moveaxis(np.asarray(actions), time_axis, -2)
    usable = values.shape[-2] // window_frames * window_frames
    values = values[..., :usable, :]
    blocks = values.reshape(
        *values.shape[:-2], usable // window_frames, window_frames, 2
    ).mean(axis=-2)
    jerk = np.diff(blocks, axis=-2) / (float(window_frames) * float(dt_s))
    return np.moveaxis(jerk, -2, time_axis)


def _risk_variables(states: np.ndarray, active: np.ndarray) -> dict[str, np.ndarray]:
    ego = states[..., :1, :]
    background = states[..., 1:, :]
    dx = background[..., 0] - ego[..., 0]
    dy = np.abs(background[..., 1] - ego[..., 1])
    same_lane_front = (dx > 0.0) & (dy < 1.8) & active[:, None]
    gap = np.maximum(dx - 4.8, 0.0)
    closing = np.maximum(ego[..., 2] - background[..., 2], 0.0)
    ttc = np.where(closing > 1.0e-3, gap / np.maximum(closing, 1.0e-3), 10.0)
    return {
        "gap_m": gap[same_lane_front],
        "TTC_s": np.minimum(ttc, 10.0)[same_lane_front],
    }


def _distribution_metrics(
    samples: list[Rollout],
    initial_states: np.ndarray,
    target_states: np.ndarray,
    target_actions: np.ndarray,
    target_highd_actions: np.ndarray,
    active: np.ndarray,
) -> dict[str, Any]:
    generated_states = np.stack([item.states for item in samples], axis=1)
    generated_actions = np.stack([item.background_actions for item in samples], axis=1)
    pairwise_trajectory = []
    pairwise_terminal = []
    trajectory_mask = np.broadcast_to(
        active[:, None], (len(active), generated_states.shape[2], active.shape[1])
    )
    for left in range(len(samples)):
        for right in range(left + 1, len(samples)):
            distance = np.linalg.norm(
                generated_states[:, left, :, 1:, :2]
                - generated_states[:, right, :, 1:, :2],
                axis=-1,
            )
            pairwise_trajectory.append(
                (distance * trajectory_mask).sum((1, 2))
                / trajectory_mask.sum((1, 2)).clip(min=1)
            )
            pairwise_terminal.append(
                (distance[:, -1] * active).sum(1) / active.sum(1).clip(min=1)
            )
    target_future = target_states
    distance = np.linalg.norm(
        generated_states[..., 1:, :2] - target_future[:, None, ..., 1:, :2],
        axis=-1,
    )
    valid = np.broadcast_to(active[:, None, None], distance.shape)
    per_sample_ade = (distance * valid).sum((2, 3)) / valid.sum((2, 3)).clip(min=1)
    final_distance = distance[:, :, -1]
    final_valid = np.broadcast_to(active[:, None], final_distance.shape)
    per_sample_fde = (final_distance * final_valid).sum(2) / final_valid.sum(2).clip(
        min=1
    )
    pairwise_values = (
        np.stack(pairwise_trajectory)
        if pairwise_trajectory
        else np.zeros((1,), np.float32)
    )
    action_mask = np.broadcast_to(
        active[:, None, None, :, None], generated_actions.shape
    )
    real_action_mask = np.broadcast_to(active[:, None, :, None], target_actions.shape)
    real_yaw = target_actions[..., 1][real_action_mask[..., 0]]
    generated_yaw = generated_actions[..., 1][action_mask[..., 0]]
    generated_source = np.concatenate(
        (
            np.broadcast_to(
                initial_states[:, None, None, 1:],
                (len(initial_states), len(samples), 1, 6, 6),
            ),
            generated_states[:, :, :-1, 1:],
        ),
        axis=2,
    )
    generated_highd = KinematicTrafficDynamics.highd_actions(
        torch.from_numpy(generated_actions),
        torch.from_numpy(generated_source.copy()),
    ).numpy()
    real_cartesian_mask = np.broadcast_to(
        active[:, None, :, None], target_highd_actions.shape
    )
    generated_cartesian_mask = np.broadcast_to(
        active[:, None, None, :, None], generated_highd.shape
    )
    real_jerk = np.diff(target_highd_actions, axis=1) / 0.04
    generated_jerk = np.diff(generated_highd, axis=2) / 0.04
    real_jerk_mask = np.broadcast_to(active[:, None, :, None], real_jerk.shape)
    generated_jerk_mask = np.broadcast_to(
        active[:, None, None, :, None], generated_jerk.shape
    )
    real_windowed_jerk = _windowed_jerk(target_highd_actions, time_axis=1)
    generated_windowed_jerk = _windowed_jerk(generated_highd, time_axis=2)
    real_windowed_mask = np.broadcast_to(
        active[:, None, :, None], real_windowed_jerk.shape
    )
    generated_windowed_mask = np.broadcast_to(
        active[:, None, None, :, None], generated_windowed_jerk.shape
    )
    real_yaw_acceleration = np.diff(target_actions[..., 1], axis=1) / 0.04
    generated_yaw_acceleration = np.diff(generated_actions[..., 1], axis=2) / 0.04
    real_yaw_acceleration_mask = np.broadcast_to(
        active[:, None], real_yaw_acceleration.shape
    )
    generated_yaw_acceleration_mask = np.broadcast_to(
        active[:, None, None], generated_yaw_acceleration.shape
    )
    real_speed = np.linalg.norm(target_states[..., 1:, 2:4], axis=-1)
    generated_speed = np.linalg.norm(generated_states[..., 1:, 2:4], axis=-1)
    speed_mask = np.broadcast_to(active[:, None], real_speed.shape)
    generated_speed_mask = np.broadcast_to(active[:, None, None], generated_speed.shape)
    target_risk = _risk_variables(target_states, active)
    generated_risk = _risk_variables(
        generated_states.reshape(-1, 149, 7, 6),
        np.repeat(active, len(samples), axis=0),
    )
    real_acceleration_magnitude = np.linalg.norm(
        target_highd_actions[real_cartesian_mask].reshape(-1, 2), axis=-1
    )
    generated_acceleration_magnitude = np.linalg.norm(
        generated_highd[generated_cartesian_mask].reshape(-1, 2), axis=-1
    )
    real_nearest = _nearest_object_distance(target_states, active)
    generated_nearest = _nearest_object_distance(
        generated_states.reshape(-1, 149, 7, 6),
        np.repeat(active, len(samples), axis=0),
    )
    real_collision = _collision_indicator(target_states, active)
    generated_collision = _collision_indicator(
        generated_states.reshape(-1, 149, 7, 6),
        np.repeat(active, len(samples), axis=0),
    )
    highd_adapted_histograms = {
        "speed_mps": _histogram_summary(
            real_speed[speed_mask],
            generated_speed[generated_speed_mask],
            bounds=(0.0, 50.0),
        ),
        "acceleration_magnitude_mps2": _histogram_summary(
            real_acceleration_magnitude,
            generated_acceleration_magnitude,
            bounds=(0.0, 10.0),
        ),
        "yaw_rate_rps": _histogram_summary(real_yaw, generated_yaw, bounds=(-0.8, 0.8)),
        "yaw_acceleration_rps2": _histogram_summary(
            real_yaw_acceleration[real_yaw_acceleration_mask],
            generated_yaw_acceleration[generated_yaw_acceleration_mask],
            bounds=(-2.0, 2.0),
        ),
        "nearest_object_distance_m": _histogram_summary(
            real_nearest, generated_nearest, bounds=(0.0, 80.0)
        ),
        "gap_m": _histogram_summary(
            target_risk["gap_m"], generated_risk["gap_m"], bounds=(0.0, 80.0)
        ),
        "TTC_s": _histogram_summary(
            target_risk["TTC_s"], generated_risk["TTC_s"], bounds=(0.0, 10.0)
        ),
        "collision_incidence": _histogram_summary(
            real_collision, generated_collision, bounds=(0.0, 1.0), bins=2
        ),
    }
    return {
        "samples_per_condition": len(samples),
        "sample_mean_ADE_m": float(per_sample_ade.mean()),
        "min_ADE_m": float(per_sample_ade.min(axis=1).mean()),
        "sample_mean_FDE_m": float(per_sample_fde.mean()),
        "min_FDE_m": float(per_sample_fde.min(axis=1).mean()),
        "energy_score_m": float(per_sample_ade.mean() - 0.5 * pairwise_values.mean()),
        "mean_pairwise_trajectory_distance_m": float(
            pairwise_values.mean() if pairwise_trajectory else 0.0
        ),
        "terminal_pairwise_distance_m": float(
            np.mean(pairwise_terminal) if pairwise_terminal else 0.0
        ),
        "motion_distribution": {
            "speed": _distribution_distance(
                real_speed[speed_mask], generated_speed[generated_speed_mask]
            ),
            "ax": _distribution_distance(
                target_highd_actions[..., 0][real_cartesian_mask[..., 0]],
                generated_highd[..., 0][generated_cartesian_mask[..., 0]],
            ),
            "ay": _distribution_distance(
                target_highd_actions[..., 1][real_cartesian_mask[..., 0]],
                generated_highd[..., 1][generated_cartesian_mask[..., 0]],
            ),
            "jx": _distribution_distance(
                real_jerk[..., 0][real_jerk_mask[..., 0]],
                generated_jerk[..., 0][generated_jerk_mask[..., 0]],
            ),
            "jy": _distribution_distance(
                real_jerk[..., 1][real_jerk_mask[..., 0]],
                generated_jerk[..., 1][generated_jerk_mask[..., 0]],
            ),
        },
        "angular_distribution": {
            "yaw_rate": _distribution_distance(real_yaw, generated_yaw),
            "yaw_acceleration": _distribution_distance(
                real_yaw_acceleration[real_yaw_acceleration_mask],
                generated_yaw_acceleration[generated_yaw_acceleration_mask],
            ),
        },
        "jerk_resolution_diagnostic": {
            "raw_0p04s_highd_zero_mass": {
                "jx": float(
                    np.isclose(real_jerk[..., 0][real_jerk_mask[..., 0]], 0.0).mean()
                ),
                "jy": float(
                    np.isclose(real_jerk[..., 1][real_jerk_mask[..., 0]], 0.0).mean()
                ),
            },
            "windowed_0p2s": {
                "jx": _distribution_distance(
                    real_windowed_jerk[..., 0][real_windowed_mask[..., 0]],
                    generated_windowed_jerk[..., 0][generated_windowed_mask[..., 0]],
                ),
                "jy": _distribution_distance(
                    real_windowed_jerk[..., 1][real_windowed_mask[..., 0]],
                    generated_windowed_jerk[..., 1][generated_windowed_mask[..., 0]],
                ),
            },
        },
        "risk_distribution": {
            name: _distribution_distance(target_risk[name], generated_risk[name])
            for name in target_risk
            if len(target_risk[name]) and len(generated_risk[name])
        },
        "highd_adapted_realism": {
            "definition": (
                "fixed-bin highD distribution diagnostics inspired by the WOSAC "
                "kinematic/interactive components; not an official WOSAC score"
            ),
            "components": highd_adapted_histograms,
        },
    }


def _following_agents(
    initial: np.ndarray, active: np.ndarray, onset: int
) -> np.ndarray:
    current = initial[:, ANCHOR_INDEX + onset]
    return (
        (current[:, 1:, 0] < current[:, :1, 0])
        & (np.abs(current[:, 1:, 1] - current[:, :1, 1]) < 1.8)
        & active
    )


def _dose_response_curve(
    baseline: Rollout,
    treatments: dict[float, Rollout],
    initial: np.ndarray,
    active: np.ndarray,
    kind: str,
    natural_calibration: dict[str, Any],
) -> dict[str, Any]:
    """Multi-dose response curves at the same horizons as natural matching."""
    onset = 25
    horizons = (0.2, 0.4, 0.8)
    if kind in {"brake", "accelerate"}:
        following = _following_agents(initial, active, onset)
        expected = -1.0 if kind == "brake" else 1.0
        metric = "signed_background_acceleration_effect_mps2"
        natural = {
            f"{horizon:.1f}s": natural_calibration["horizon_diagnostics"][
                f"{horizon:.1f}s"
            ][kind]["effect_p10_p50_p90_mps2"]
            for horizon in horizons
        }

        def effect(rollout: Rollout, frames: int) -> np.ndarray:
            value = (
                expected
                * (
                    rollout.states[:, onset + frames, 1:, 2]
                    - baseline.states[:, onset + frames, 1:, 2]
                )
                / (frames * 0.04)
            )
            return value[following]

    else:
        current = torch.from_numpy(initial[:, ANCHOR_INDEX + onset].copy())
        valid = torch.from_numpy(active)
        relation = response_relevance(
            current, torch.cat((torch.ones(len(valid), 1, dtype=torch.bool), valid), 1)
        ).numpy()
        near = relation > 0.35
        metric = "near_lateral_separation_change_m"
        natural = None

        def effect(rollout: Rollout, frames: int) -> np.ndarray:
            baseline_separation = np.abs(
                baseline.states[:, onset + frames, 1:, 1]
                - baseline.states[:, onset + frames, :1, 1]
            )
            separation = np.abs(
                rollout.states[:, onset + frames, 1:, 1]
                - rollout.states[:, onset + frames, :1, 1]
            )
            return (separation - baseline_separation)[near]

    doses: dict[str, Any] = {}
    for dose, rollout_value in treatments.items():
        by_horizon = {}
        for horizon in horizons:
            values = effect(rollout_value, int(round(horizon / 0.04)))
            by_horizon[f"{horizon:.1f}s"] = {
                "p10_p50_p90": np.quantile(values, (0.1, 0.5, 0.9)).tolist(),
                "mean": float(values.mean()),
            }
        doses[f"{dose:g}"] = by_horizon
    return {
        "metric": metric,
        "horizons_s": list(horizons),
        "natural_p10_p50_p90": natural,
        "doses": doses,
    }


def _intervention_metrics(
    baseline: Rollout,
    mild: Rollout,
    strong: Rollout,
    initial: np.ndarray,
    active: np.ndarray,
    kind: str,
    natural_effects: np.ndarray | None = None,
) -> dict[str, float]:
    onset = 25
    committed_frames = 1
    current = torch.from_numpy(initial[:, ANCHOR_INDEX + onset].copy())
    valid = torch.from_numpy(active)
    all_valid = torch.cat((torch.ones(len(valid), 1, dtype=torch.bool), valid), 1)
    relevance = response_relevance(current, all_valid).numpy()
    mild_delta = (
        mild.background_actions[:, onset:] - baseline.background_actions[:, onset:]
    )
    strong_delta = (
        strong.background_actions[:, onset:] - baseline.background_actions[:, onset:]
    )
    near = relevance > 0.35
    far = relevance < 0.1
    magnitude = np.linalg.norm(mild_delta, axis=-1)
    post_magnitude = magnitude[:, committed_frames:]
    near_mask = np.broadcast_to(near[:, None], post_magnitude.shape)
    far_mask = np.broadcast_to(far[:, None], post_magnitude.shape)
    near_value = post_magnitude[near_mask].mean()
    far_value = post_magnitude[far_mask].mean()
    near_profile = np.divide(
        (post_magnitude * near[:, None]).sum((0, 2)),
        np.broadcast_to(near[:, None], post_magnitude.shape).sum((0, 2)).clip(min=1),
    )
    far_profile = np.divide(
        (post_magnitude * far[:, None]).sum((0, 2)),
        np.broadcast_to(far[:, None], post_magnitude.shape).sum((0, 2)).clip(min=1),
    )
    threshold = max(1.0e-6, 0.05 * float(near_profile.max()))
    detected = np.flatnonzero(near_profile > threshold)
    response_frame = committed_frames + int(detected[0]) if len(detected) else -1
    committed_change = float(np.abs(mild_delta[:, :committed_frames]).max())
    baseline_near = baseline.background_actions[:, onset + committed_frames :]
    baseline_near = baseline_near[..., 0][near_mask]
    mild_near = mild.background_actions[:, onset + committed_frames :, :, 0][near_mask]
    strong_post = np.linalg.norm(strong_delta[:, committed_frames:], axis=-1)
    strong_near_value = strong_post[near_mask].mean()
    result = {
        "committed_response_max_change": committed_change,
        "committed_response_invariant": bool(committed_change < 1.0e-8),
        "response_onset_frame_offset": response_frame,
        "response_latency_s": (
            float(response_frame * 0.04) if response_frame >= 0 else float("nan")
        ),
        "near_response_magnitude": float(near_value),
        "far_response_magnitude": float(far_value),
        "locality_ratio_far_to_near": float(far_value / max(near_value, 1.0e-8)),
        "strong_to_mild_response_ratio": float(
            strong_near_value / max(near_value, 1.0e-8)
        ),
        "near_longitudinal_action_wasserstein": float(
            wasserstein_distance(baseline_near, mild_near)
        ),
        "response_magnitude_profile": {
            "time_s": (
                0.04 * np.arange(committed_frames, mild_delta.shape[1])
            ).tolist(),
            "near": near_profile.tolist(),
            "far": far_profile.tolist(),
        },
    }
    if kind in {"brake", "accelerate"}:
        following = _following_agents(initial, active, onset)
        expected = -1.0 if kind == "brake" else 1.0
        mild_long = mild_delta[:, committed_frames:25, :, 0].mean(1)
        strong_long = strong_delta[:, committed_frames:25, :, 0].mean(1)
        result["direction_success_rate"] = float(
            (expected * mild_long[following] > 0.0).mean()
        )
        result["dose_monotonicity_rate"] = float(
            (expected * (strong_long - mild_long)[following] > 0.0).mean()
        )
        # ``matched_response_calibration`` defines a human effect as the
        # background longitudinal velocity change over 0.8 s, relative to a
        # matched neutral response.  Compare exactly that physical quantity;
        # an action-space mean is useful for direction/monotonicity but is not
        # commensurate with the natural acceleration-effect distribution.
        horizon_frames = 20
        velocity_effect = (
            expected
            * (
                mild.states[:, onset + horizon_frames, 1:, 2]
                - baseline.states[:, onset + horizon_frames, 1:, 2]
            )
            / (horizon_frames * 0.04)
        )
        simulated_effect = velocity_effect[following]
        if natural_effects is not None and len(simulated_effect):
            natural = np.asarray(natural_effects, np.float32)
            lower, upper = np.quantile(natural, (0.10, 0.90))
            result["response_distribution_wasserstein_mps2"] = float(
                wasserstein_distance(natural, simulated_effect)
            )
            result["response_within_natural_p10_p90_rate"] = float(
                ((simulated_effect >= lower) & (simulated_effect <= upper)).mean()
            )
        evaluation_frame = onset + 24
        baseline_speed = np.linalg.norm(
            baseline.states[:, evaluation_frame, 1:, 2:4], axis=-1
        )
        mild_speed = np.linalg.norm(mild.states[:, evaluation_frame, 1:, 2:4], axis=-1)
        result["signed_follower_speed_response_mps"] = float(
            expected * (mild_speed - baseline_speed)[following].mean()
        )
    else:
        separation_base = np.abs(
            baseline.states[:, onset + 24, 1:, 1]
            - baseline.states[:, onset + 24, :1, 1]
        )
        separation_left = np.abs(
            mild.states[:, onset + 24, 1:, 1] - mild.states[:, onset + 24, :1, 1]
        )
        result["separation_non_decrease_rate"] = float(
            (separation_left[near] >= separation_base[near]).mean()
        )
    return result


def evaluate_world_model(config: dict[str, Any], *, config_dir: Path) -> dict[str, Any]:
    """Evaluate the maintained model and persist its complete JSON report."""
    from .calibration import fit_natural_response_calibrator
    from .data import split_rows
    from .model import DiffusionGuidedHiQR
    from .train import _model_config, load_checkpoint

    output = ensure_dir(config["paths"]["output_dir"])
    device = select_device(config["training"].get("device", "auto"))
    seed = int(config["training"]["seed"])
    set_seed(seed)
    experiment = prepare_experiment_data(config, config_dir)
    scope = str(config["training"].get("experiment_scope", "full"))
    if scope not in {"full", "pilot"}:
        raise ValueError("experiment_scope must be 'full' or 'pilot'")
    checkpoint_path = Path(
        config["paths"].get(
            "evaluation_checkpoint",
            output / "checkpoints/best_hierarchical_world_model.pt",
        )
    )
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    evaluation = config.get("evaluation", {})
    if evaluation.get("override_model_config", False):
        overridden = DiffusionGuidedHiQR(_model_config(config)).to(device)
        overridden.load_state_dict(model.state_dict())
        model = overridden
    if "intervention_adapter_logit" in evaluation:
        if not model.cfg.intervention_adapter_enabled:
            raise ValueError("adapter logit override requires intervention adapter")
        with torch.no_grad():
            model.decoder.intervention_logit.fill_(
                float(evaluation["intervention_adapter_logit"])
            )
    if model.cfg.natural_response_kernel_enabled:
        # This optional controller is calibrated only on the training
        # recording split.  In particular, a pilot must not fit it from its
        # small held-out test cohort just because that cohort is convenient.
        reference_path = config["paths"].get("response_calibration_reference")
        if reference_path:
            calibration = load_json(reference_path)
            response_bounds = np.asarray(
                [
                    calibration[name]["effect_p10_p50_p90_mps2"][::2]
                    for name in ("brake", "accelerate")
                ],
                np.float32,
            )
            sensitivity_bounds = np.asarray(
                calibration["dose_sensitivity_p10_p90_per_mps2"], np.float32
            )
        else:
            calibration_rows = split_rows(
                experiment.bundle.arrays, "train", seed=seed
            )
            calibrator, _ = fit_natural_response_calibrator(
                experiment.bundle.arrays,
                calibration_rows,
                minimum_events=int(
                    config["training"].get("response_calibration_minimum_events", 100)
                ),
                method=str(
                    config["training"].get("response_calibration_method", "exact")
                ),
            )
            response_bounds = calibrator.global_bounds
            sensitivity_bounds = calibrator.global_sensitivity_bounds
        model.set_matched_response_bounds(torch.from_numpy(response_bounds).to(device))
        model.set_response_sensitivity_bounds(
            torch.from_numpy(sensitivity_bounds).to(device)
        )
    model.eval()
    rows = experiment.test_rows
    states = np.asarray(experiment.bundle.arrays["agent_states"][rows], np.float32)
    valid = np.asarray(experiment.bundle.arrays["agent_valid"][rows], bool)
    maps = np.asarray(experiment.bundle.arrays["map_polylines"][rows], np.float32)
    map_valid = np.asarray(experiment.bundle.arrays["map_polyline_valid"][rows], bool)
    active = valid[:, ANCHOR_INDEX, 1:]
    target = states[:, ANCHOR_INDEX + 1 : 174]
    with tempfile.TemporaryDirectory(prefix="hierarchical_wm_eval_") as cache:
        diffusion = frozen_diffusion_plans(
            experiment.bundle,
            rows,
            checkpoint=config["paths"]["diffusion_checkpoint"],
            output_dir=cache,
            device=device,
            batch_size=32,
            ddim_steps=20,
            experiment_scope=scope,
        )
    generated: list[Rollout] = []
    for start in range(0, len(rows), 64):
        generated.append(
            rollout(
                model,
                states[start : start + 64],
                valid[start : start + 64],
                diffusion[start : start + 64],
                maps[start : start + 64],
                map_valid[start : start + 64],
                device=device,
                history_frames=25,
                motion_seed=None,
            )
        )
    diffusion_distance = np.linalg.norm(diffusion - target[..., 1:, :2], axis=-1)
    diffusion_mask = np.broadcast_to(active[:, None], diffusion_distance.shape)
    generated_states = np.concatenate([item.states for item in generated])
    diffusion_states = target.copy()
    diffusion_states[..., 1:, :2] = diffusion
    diffusion_positions = np.concatenate(
        (states[:, ANCHOR_INDEX : ANCHOR_INDEX + 1, 1:, :2], diffusion), axis=1
    )
    diffusion_states[..., 1:, 2:4] = np.diff(diffusion_positions, axis=1) / 0.04
    factual: dict[str, Any] = {
        "open_loop_diffusion": {
            "ADE_m": float(diffusion_distance[diffusion_mask].mean()),
            "FDE_m": float(diffusion_distance[:, -1][active].mean()),
            "P50_displacement_error_m": float(
                np.quantile(diffusion_distance[diffusion_mask], 0.50)
            ),
            "P90_displacement_error_m": float(
                np.quantile(diffusion_distance[diffusion_mask], 0.90)
            ),
            "P95_displacement_error_m": float(
                np.quantile(diffusion_distance[diffusion_mask], 0.95)
            ),
            "P99_displacement_error_m": float(
                np.quantile(diffusion_distance[diffusion_mask], 0.99)
            ),
            "sequences": int(len(rows)),
            "frames": 149,
        },
        "diffusion_guided_hiqr": _factual_metrics(generated_states, target, active),
        "temporal_error": {
            "open_loop_diffusion": _temporal_factual_metrics(
                diffusion_states, target, active
            ),
            "diffusion_guided_hiqr": _temporal_factual_metrics(
                generated_states, target, active
            ),
        },
    }
    factual["event_strata"] = _factual_event_strata(
        generated_states,
        target,
        active,
        np.asarray(experiment.bundle.arrays["is_evt_tail"])[rows],
        semantic_cutin_agents(
            states[:, ANCHOR_INDEX:174], valid[:, ANCHOR_INDEX:174]
        ).any(axis=1),
    )
    factual["history_ablation"] = {}
    for frames in (5, 10, 15):
        generated = [
            rollout(
                model,
                states[start : start + 64],
                valid[start : start + 64],
                diffusion[start : start + 64],
                maps[start : start + 64],
                map_valid[start : start + 64],
                device=device,
                history_frames=frames,
                motion_seed=None,
            ).states
            for start in range(0, len(rows), 64)
        ]
        factual["history_ablation"][str(frames)] = _factual_metrics(
            np.concatenate(generated), target, active
        )
    factual["history_ablation"]["25"] = factual["diffusion_guided_hiqr"]
    ablation_count = min(256, len(rows))
    initial = states[:ablation_count, ANCHOR_INDEX, 1:]
    time = np.arange(1, 150, dtype=np.float32)[None, :, None, None] * 0.04
    constant_velocity = initial[:, None, :, :2] + time * initial[:, None, :, 2:4]
    without_constraint = rollout(
        model,
        states[:ablation_count],
        valid[:ablation_count],
        constant_velocity,
        maps[:ablation_count],
        map_valid[:ablation_count],
        device=device,
        history_frames=25,
        motion_seed=None,
    )
    factual["without_long_horizon_constraint"] = _factual_metrics(
        without_constraint.states,
        target[:ablation_count],
        active[:ablation_count],
    )
    # This is only a bounded evaluation cohort.  It is deliberately not the
    # AMS/subset-simulation population; that estimator lives in IDM_subset.
    stochastic_cohort_size = min(1024, len(rows))
    stochastic_rows = slice(0, stochastic_cohort_size)
    motion_seeds = tuple(seed + sample for sample in range(16))
    sampled_plans = stochastic_diffusion_plan_samples(
        experiment.bundle,
        rows[stochastic_rows],
        checkpoint=config["paths"]["diffusion_checkpoint"],
        device=device,
        batch_size=32,
        ddim_steps=20,
        motion_seeds=motion_seeds,
    )
    sample_rollouts = [
        rollout(
            model,
            states[stochastic_rows],
            valid[stochastic_rows],
            plan,
            maps[stochastic_rows],
            map_valid[stochastic_rows],
            device=device,
            history_frames=25,
            motion_seed=motion_seed + 100_000,
        )
        for plan, motion_seed in zip(sampled_plans, motion_seeds)
    ]
    source_states = states[stochastic_rows, ANCHOR_INDEX:173, 1:]
    target_highd = np.asarray(
        experiment.bundle.arrays["actions_highd"][rows[stochastic_rows]], np.float32
    )
    target_actions = KinematicTrafficDynamics.controls_from_highd_actions(
        torch.from_numpy(target_highd.copy()), torch.from_numpy(source_states.copy())
    ).numpy()
    distribution = _distribution_metrics(
        sample_rollouts,
        states[stochastic_rows, ANCHOR_INDEX],
        target[stochastic_rows],
        target_actions,
        target_highd,
        active[stochastic_rows],
    )
    intervention_count = min(512, len(rows))
    selected = slice(0, intervention_count)
    common = seed + 1000
    baseline = rollout(
        model,
        states[selected],
        valid[selected],
        diffusion[selected],
        maps[selected],
        map_valid[selected],
        device=device,
        history_frames=25,
        motion_seed=common,
    )
    interventions: dict[str, Any] = {}
    # A 1,024-sequence pilot is sufficient for model rollouts but not for all
    # sparse matched human-response cells, especially ego acceleration at
    # 0.8 s.  Its diagnostic reference therefore uses the complete held-out
    # test-recording pool.  No candidate is selected on this reference; full
    # evaluation continues to use exactly its own complete test split.
    natural_reference_rows = (
        split_rows(experiment.bundle.arrays, "test", seed=seed)
        if scope == "pilot"
        else experiment.test_rows
    )
    _, test_response_scale = matched_response_calibration(
        experiment.bundle.arrays,
        natural_reference_rows,
        minimum_events=30,
    )
    doses = {
        "brake": (1.5, 2.25, 3.0),
        "accelerate": (1.0, 1.5, 2.0),
        "left": (0.08, 0.12, 0.16),
    }
    for kind, dose_values in doses.items():
        treatments = {
            dose: rollout(
                model,
                states[selected],
                valid[selected],
                diffusion[selected],
                maps[selected],
                map_valid[selected],
                device=device,
                history_frames=25,
                motion_seed=common,
                intervention=kind,
                dose=dose,
            )
            for dose in dose_values
        }
        mild, strong = treatments[dose_values[0]], treatments[dose_values[-1]]
        interventions[kind] = _intervention_metrics(
            baseline,
            mild,
            strong,
            states[selected],
            active[selected],
            kind,
            (
                np.asarray(test_response_scale[kind]["effect_samples_mps2"])
                if kind in {"brake", "accelerate"}
                else None
            ),
        )
        interventions[kind]["dose_response"] = _dose_response_curve(
            baseline,
            treatments,
            states[selected],
            active[selected],
            kind,
            test_response_scale,
        )
    report = {
        "evaluation_schema_version": 2,
        "experiment_scope": scope,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test_sequences": len(rows),
        "factual_fidelity": factual,
        "distribution_stochasticity": distribution,
        "randomness_contract": {
            "z_scenario": (
                "Flow base randomness (u_M,z_C0,z_K); with fixed C0/M only "
                "z_K varies"
            ),
            "z_motion": (
                "one motion seed deterministically addresses epsilon_diff "
                "[149,6,2], the 16-D scene innovation and six 16-D agent "
                "innovations"
            ),
            "paired_interventions_use_common_random_numbers": True,
        },
        "model_contract": {
            "architecture": (
                "Flow p(M)p(C0|M)p(K|C0,M); diffusion (C0,M,K) -> soft plan; "
                "observation-filtered, two-time-scale HiQR jerk response"
            ),
            # One response is committed per simulated frame.  The previous
            # report incorrectly called the 0.2 s diagnostic window the
            # replanning interval, which made the 25 Hz contract look like a
            # 5 Hz policy.
            "response_frequency_hz": float(1.0 / model.cfg.dt_s),
            "response_interval_s": float(model.cfg.execute_frames * model.cfg.dt_s),
            "response_commit_frames": int(model.cfg.execute_frames),
            "preview_horizon_s": float(model.cfg.preview_frames * model.cfg.dt_s),
            "scene_latent_refresh_s": float(
                model.cfg.scene_refresh_responses * model.cfg.dt_s
            ),
            "soft_plan_is_hard_trajectory": False,
            "soft_plan_policy": "rebase plan increments to the realized state",
            "future_ego_action_is_model_input": False,
            "history_training_frames": [5, 10, 15, 25],
            "scene_latent_dim": int(model.cfg.scene_latent_dim),
            "agent_latent_dim_per_vehicle": int(model.cfg.agent_latent_dim),
            "causal_response_scale": float(model.cfg.causal_response_scale),
            "intervention_adapter_enabled": bool(
                model.cfg.intervention_adapter_enabled
            ),
            "natural_response_kernel_enabled": bool(
                model.cfg.natural_response_kernel_enabled
            ),
            "intervention_trigger_threshold_mps2": float(
                model.cfg.intervention_trigger_threshold_mps2
            ),
            "evt_role": "external human-risk scale; excluded from training loss",
        },
        "evaluation_protocol": {
            "factual_reconstruction": (
                "held-out C0 and held-out long-horizon condition with a frozen "
                "diffusion preview; this is conditional reconstruction, not "
                "unconditional scenario-generation accuracy"
            ),
            # Keep the established result schema; these fields are bounded
            # evaluation cohorts, not subset-simulation populations.
            "stochasticity_subset_sequences": int(stochastic_cohort_size),
            "intervention_subset_sequences": int(intervention_count),
            "natural_response_reference_sequences": int(len(natural_reference_rows)),
            "intervention_common_random_numbers": True,
            "jerk_protocol": (
                "raw 0.04 s action jerk is reported together with 0.2 s "
                "windowed jerk because highD raw jerk has a large quantized "
                "zero mass"
            ),
        },
        "limitations": [
            "Within a fixed K constraint, motion diversity must be interpreted "
            "together with its proper score.",
            "Raw 0.04 s jerk KS is dominated by highD's quantized zero mass; "
            "the 0.2 s windowed diagnostic is more representative of continuous "
            "motion but does not replace the raw metric.",
            "Observed highD data and structural intervention tests do not prove "
            "counterfactual correctness for arbitrary ADS policies.",
        ],
        "intervention_effectiveness": interventions,
        "natural_response_calibration": test_response_scale,
        "claims": {
            "counterfactual_correctness_proven": False,
            "official_WOSAC_score": False,
            "interpretation": "highD factual and structural intervention evidence only",
        },
    }
    save_json(report, output / "evaluation.json")
    return report

"""B3 position-block generation followed by a common waypoint tracker."""

from __future__ import annotations

import numpy as np
import torch

from interactive_behavior_world_model.policies.interface import RandomKey
from interactive_behavior_world_model.policies.rolling_action_policy import RollingActionDiffusion
from world_model.src.core.dynamics import KinematicTrafficDynamics
from .action_rollout import RolloutBatch
from .rollout import _torch_features


def _tracker(
    current: torch.Tensor, target_xy: torch.Tensor, lookahead_s: float = 1.0
) -> torch.Tensor:
    """Pure-pursuit-style speed/heading tracking with a fixed lookahead."""
    delta = target_xy - current[..., :2]
    desired_velocity = delta / float(lookahead_s)
    desired_speed = torch.linalg.vector_norm(desired_velocity, dim=-1)
    speed = torch.linalg.vector_norm(current[..., 2:4], dim=-1)
    heading = torch.atan2(current[..., 3], current[..., 2])
    desired_heading = torch.atan2(desired_velocity[..., 1], desired_velocity[..., 0])
    error = torch.atan2(
        torch.sin(desired_heading - heading), torch.cos(desired_heading - heading)
    )
    return torch.stack(
        ((desired_speed - speed) / float(lookahead_s), error / float(lookahead_s)), -1
    )


def rollout_position_diffusion(
    model: RollingActionDiffusion,
    checkpoint: dict,
    initial_history: np.ndarray,
    history_valid: np.ndarray,
    ego_future: np.ndarray,
    lengths_m: np.ndarray,
    widths_m: np.ndarray,
    map_polylines: np.ndarray,
    map_valid: np.ndarray,
    scenario_ids: np.ndarray,
    *,
    benchmark_id: str,
    fit_seed: int,
    futures: int,
    inference_steps: int,
    warm_start: bool,
    device: torch.device,
    tracker_lookahead_steps: int = 5,
    exogenous_future: np.ndarray | None = None,
    exogenous_mask: np.ndarray | None = None,
) -> RolloutBatch:
    batch, k = len(initial_history), int(futures)
    expand = lambda value: np.repeat(np.asarray(value), k, axis=0)
    history = torch.from_numpy(expand(initial_history).copy()).to(device)
    valid_history = torch.from_numpy(expand(history_valid).copy()).to(device)
    ego_log = torch.from_numpy(expand(ego_future).copy()).to(device)
    horizon = int(ego_future.shape[1])
    decisions = (horizon + 4) // 5
    external = (
        None
        if exogenous_future is None
        else torch.from_numpy(expand(exogenous_future).copy()).to(device)
    )
    external_mask = (
        None
        if exogenous_mask is None
        else torch.from_numpy(expand(exogenous_mask).copy()).to(device)
    )
    lengths = torch.from_numpy(expand(lengths_m).copy()).to(device)
    widths = torch.from_numpy(expand(widths_m).copy()).to(device)
    active = valid_history[:, -1]
    lines = torch.from_numpy(expand(map_polylines).copy()).to(device)
    line_valid = torch.from_numpy(expand(map_valid).copy()).to(device)
    lane_valid = line_valid.any(-1)
    lane_centers = (lines[..., 1] * line_valid).sum(-1) / line_valid.sum(-1).clamp_min(
        1
    )
    lane_width = (lines[..., 4] * line_valid).sum(-1) / line_valid.sum(-1).clamp_min(1)
    feature_mean = torch.as_tensor(checkpoint["feature_mean"], device=device)
    feature_std = torch.as_tensor(checkpoint["feature_std"], device=device)
    noise = np.empty((batch * k, decisions, 15, 6, 2), np.float32)
    for scene in range(batch):
        for rollout_id in range(k):
            noise[scene * k + rollout_id] = np.random.default_rng(
                RandomKey(
                    benchmark_id,
                    str(scenario_ids[scene]),
                    fit_seed,
                    rollout_id,
                    "position_generation_noise",
                ).seed()
            ).standard_normal((decisions, 15, 6, 2))
    noise_t = torch.from_numpy(noise).to(device)
    block_valid = active[:, None, 1:].expand(-1, 15, -1)
    dynamics = KinematicTrafficDynamics()
    future_states = torch.zeros((batch * k, horizon, 7, 6), device=device)
    requested = torch.zeros((batch * k, decisions, 6, 2), device=device)
    applied = torch.zeros_like(requested)
    rewritten = torch.zeros((batch * k, decisions, 6), dtype=torch.bool, device=device)
    current = history[:, -1].clone()
    cached_absolute = None
    physical_frame = 0
    model.eval()
    with torch.no_grad():
        for decision in range(decisions):
            feature = (
                _torch_features(
                    history,
                    valid_history,
                    lengths,
                    widths,
                    lane_centers,
                    lane_width,
                    lane_valid,
                )
                - feature_mean
            ) / feature_std
            if warm_start and cached_absolute is not None:
                shifted_abs = torch.cat(
                    (cached_absolute[:, 1:], cached_absolute[:, -1:]), 1
                )
                shifted = shifted_abs - current[:, None, 1:, :2]
                block = model.sample_warm(
                    feature,
                    valid_history,
                    block_valid,
                    shifted,
                    noise_t[:, decision],
                    inference_steps=inference_steps,
                )
            else:
                block = model.sample(
                    feature,
                    valid_history,
                    block_valid,
                    noise_t[:, decision],
                    inference_steps=inference_steps,
                )
            cached_absolute = current[:, None, 1:, :2] + block
            lookahead_steps = max(
                1, min(int(tracker_lookahead_steps), cached_absolute.shape[1])
            )
            request = _tracker(
                current[:, 1:],
                cached_absolute[:, lookahead_steps - 1],
                0.2 * lookahead_steps,
            )
            action = request.clone()
            action[..., 0].clamp_(-8.0, 4.0)
            speed = torch.linalg.vector_norm(current[:, 1:, 2:4], dim=-1).clamp_min(
                1e-3
            )
            yaw_limit = torch.minimum(torch.full_like(speed, 0.6), 4.0 / speed)
            action[..., 1] = torch.maximum(
                torch.minimum(action[..., 1], yaw_limit), -yaw_limit
            )
            action *= active[:, 1:, None]
            requested[:, decision] = request
            applied[:, decision] = action
            rewritten[:, decision] = ((action - request).abs() > 1e-6).any(-1) & active[
                :, 1:
            ]
            frames = min(5, horizon - physical_frame)
            full_control = torch.zeros((batch * k, 7, 2), device=device)
            full_control[:, 1:] = action
            for _ in range(frames):
                stepped = dynamics.step(current, full_control, active, 0.04)
                stepped[:, 0] = ego_log[:, physical_frame]
                if external is not None and external_mask is not None:
                    stepped = torch.where(
                        external_mask[..., None], external[:, physical_frame], stepped
                    )
                future_states[:, physical_frame] = stepped
                current = stepped
                history = torch.cat((history[:, 1:], current[:, None]), 1)
                valid_history = torch.cat((valid_history[:, 1:], active[:, None]), 1)
                physical_frame += 1
    return RolloutBatch(
        future_states.reshape(batch, k, horizon, 7, 6).cpu().numpy(),
        requested.reshape(batch, k, decisions, 6, 2).cpu().numpy(),
        applied.reshape(batch, k, decisions, 6, 2).cpu().numpy(),
        rewritten.reshape(batch, k, decisions, 6).cpu().numpy(),
    )

"""Batched T1 rollout through the one common kinematic backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from traffic_components.src.core.dynamics import KinematicTrafficDynamics

from npc_behavior_benchmark.policies.interface import RandomKey
from npc_behavior_benchmark.policies.shared_bc import SharedBCModel
from .pnc import fixed_pnc_action, initial_pnc_targets, structured_bc_pnc_action


@dataclass
class RolloutBatch:
    states: np.ndarray
    requested_actions: np.ndarray
    applied_actions: np.ndarray
    rewrite: np.ndarray


def _torch_features(
    history, valid, lengths, widths, lane_centers, lane_widths, lane_valid
):
    current_ego = history[:, -1:, :1]
    position = history[..., :2] - current_ego[..., :2]
    relative_velocity = history[..., 2:4] - current_ego[..., 2:4]
    size = torch.stack((lengths, widths), -1)[:, None].expand(
        -1, history.shape[1], -1, -1
    )
    difference = torch.abs(history[..., 1, None] - lane_centers[:, None, None])
    difference = difference.masked_fill(~lane_valid[:, None, None], float("inf"))
    nearest = difference.argmin(-1)
    centers = lane_centers[:, None, None].expand(
        -1, history.shape[1], history.shape[2], -1
    )
    widths_all = lane_widths[:, None, None].expand_as(centers)
    chosen_center = torch.gather(centers, -1, nearest[..., None])[..., 0]
    chosen_width = torch.gather(widths_all, -1, nearest[..., None])[..., 0]
    lane = torch.stack((history[..., 1] - chosen_center, chosen_width), -1)
    feature = torch.cat(
        (position, history[..., 2:6], relative_velocity, size, lane), -1
    )
    return feature * valid[..., None]


def rollout_shared_bc(
    model: SharedBCModel,
    checkpoint: dict,
    initial_history: np.ndarray,
    history_valid: np.ndarray,
    ego_future: np.ndarray,
    agent_ids: np.ndarray,
    lengths_m: np.ndarray,
    widths_m: np.ndarray,
    map_polylines: np.ndarray,
    map_valid: np.ndarray,
    scenario_ids: np.ndarray,
    *,
    benchmark_id: str,
    fit_seed: int,
    futures: int,
    device: torch.device,
    exogenous_future: np.ndarray | None = None,
    exogenous_mask: np.ndarray | None = None,
    ego_policy_id: str | None = None,
    ego_policy_model: torch.nn.Module | None = None,
    ego_policy_checkpoint: dict | None = None,
) -> RolloutBatch:
    """Roll out all scenes and ensemble members with synchronous controls."""
    batch = len(initial_history)
    k = int(futures)
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
    noise = np.empty((batch * k, decisions, 6, 2), np.float32)
    style_steps = (
        decisions if checkpoint.get("style_resampling") == "per_decision" else 1
    )
    style_noise = np.zeros((batch * k, style_steps, 7, model.style_dim), np.float32)
    for scene in range(batch):
        for rollout_id in range(k):
            key = RandomKey(
                benchmark_id,
                str(scenario_ids[scene]),
                fit_seed,
                rollout_id,
                "action_innovation",
            )
            noise[scene * k + rollout_id] = np.random.default_rng(
                key.seed()
            ).standard_normal((decisions, 6, 2))
            if model.style_dim:
                style_key = RandomKey(
                    benchmark_id,
                    str(scenario_ids[scene]),
                    fit_seed,
                    rollout_id,
                    "driver_style",
                )
                style_noise[scene * k + rollout_id] = np.random.default_rng(
                    style_key.seed()
                ).standard_normal((style_steps, 7, model.style_dim))
    noise_t = torch.from_numpy(noise).to(device)
    style_t = torch.from_numpy(style_noise).to(device)
    dynamics = KinematicTrafficDynamics()
    future_states = torch.zeros((batch * k, horizon, 7, 6), device=device)
    requested = torch.zeros((batch * k, decisions, 6, 2), device=device)
    applied = torch.zeros_like(requested)
    rewritten = torch.zeros((batch * k, decisions, 6), dtype=torch.bool, device=device)
    current = history[:, -1].clone()
    target_speed, target_lane_y = initial_pnc_targets(current, lane_centers, lane_valid)
    model.eval()
    physical_frame = 0
    with torch.no_grad():
        for decision in range(decisions):
            raw_feature = _torch_features(
                history,
                valid_history,
                lengths,
                widths,
                lane_centers,
                lane_width,
                lane_valid,
            )
            feature = (raw_feature - feature_mean) / feature_std
            style_index = decision if style_steps > 1 else 0
            mean, log_std = model(
                feature,
                valid_history,
                style_t[:, style_index] if model.style_dim else None,
            )
            raw = mean + log_std.exp() * noise_t[:, decision]
            request = raw.clone()
            action = raw.clone()
            action[..., 0].clamp_(-8.0, 4.0)
            speed = torch.linalg.vector_norm(current[:, 1:, 2:4], dim=-1).clamp_min(
                1.0e-3
            )
            yaw_limit = torch.minimum(torch.full_like(speed, 0.6), 4.0 / speed)
            action[..., 1] = torch.maximum(
                torch.minimum(action[..., 1], yaw_limit), -yaw_limit
            )
            action = action * active[:, 1:, None]
            requested[:, decision] = request
            applied[:, decision] = action
            rewritten[:, decision] = ((action - request).abs() > 1.0e-6).any(
                -1
            ) & active[:, 1:]
            frames = min(5, horizon - physical_frame)
            full_control = torch.zeros((batch * k, 7, 2), device=device)
            full_control[:, 1:] = action
            if ego_policy_id in {"structured_bc", "structured_bc_lane_stable"}:
                if ego_policy_model is None or ego_policy_checkpoint is None:
                    raise ValueError("structured_bc requires model and checkpoint")
                stable = ego_policy_id == "structured_bc_lane_stable"
                full_control[:, 0] = structured_bc_pnc_action(
                    ego_policy_model,
                    ego_policy_checkpoint,
                    raw_feature,
                    valid_history,
                    current,
                    active,
                    target_lane_y=target_lane_y if stable else None,
                    learned_yaw_weight=0.25 if stable else 1.0,
                )
            elif ego_policy_id is not None:
                full_control[:, 0] = fixed_pnc_action(
                    current,
                    active,
                    lengths,
                    target_speed,
                    target_lane_y,
                    ego_policy_id,
                    widths=widths,
                )
            for _ in range(frames):
                stepped = dynamics.step(current, full_control, active, 0.04)
                if ego_policy_id is None:
                    stepped[:, 0] = ego_log[:, physical_frame]
                if external is not None and external_mask is not None:
                    stepped = torch.where(
                        external_mask[..., None], external[:, physical_frame], stepped
                    )
                future_states[:, physical_frame] = stepped
                current = stepped
                history = torch.cat((history[:, 1:], current[:, None]), dim=1)
                valid_history = torch.cat(
                    (valid_history[:, 1:], active[:, None]), dim=1
                )
                physical_frame += 1
    return RolloutBatch(
        states=future_states.reshape(batch, k, horizon, 7, 6).cpu().numpy(),
        requested_actions=requested.reshape(batch, k, decisions, 6, 2).cpu().numpy(),
        applied_actions=applied.reshape(batch, k, decisions, 6, 2).cpu().numpy(),
        rewrite=rewritten.reshape(batch, k, decisions, 6).cpu().numpy(),
    )

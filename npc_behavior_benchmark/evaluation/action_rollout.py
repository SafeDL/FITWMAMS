"""Batched cold/warm action-diffusion rollout for B0 and B1."""

from __future__ import annotations

import numpy as np
import torch

from traffic_components.src.core.dynamics import KinematicTrafficDynamics

from npc_behavior_benchmark.policies.interface import RandomKey
from npc_behavior_benchmark.data.behavior_condition import BEHAVIOR_CONDITION_MODES
from npc_behavior_benchmark.policies.rolling_action_policy import RollingActionDiffusion
from .rollout import RolloutBatch, _torch_features
from .pnc import fixed_pnc_action, initial_pnc_targets, structured_bc_pnc_action


def _append_t4a_condition(
    raw_feature: torch.Tensor,
    feature_dim: int,
    behavior_conditions: list[dict] | None,
    futures: int,
) -> torch.Tensor:
    """Append explicit per-scene T4a goals to a batched causal feature tensor."""
    if feature_dim == raw_feature.shape[-1]:
        if behavior_conditions is not None:
            raise ValueError(
                "an unconditional action model cannot receive behavior_conditions"
            )
        return raw_feature
    if feature_dim != raw_feature.shape[-1] + len(BEHAVIOR_CONDITION_MODES):
        raise ValueError(f"unsupported action model feature_dim={feature_dim}")
    batch = len(raw_feature) // int(futures)
    if behavior_conditions is not None and len(behavior_conditions) != batch:
        raise ValueError("behavior_conditions must have one entry per unexpanded scene")
    extra = raw_feature.new_zeros(
        (*raw_feature.shape[:-1], len(BEHAVIOR_CONDITION_MODES))
    )
    if behavior_conditions is not None:
        for scene, condition in enumerate(behavior_conditions):
            if condition is None:
                continue
            target = int(condition["target_agent_index"])
            mode = str(condition["mode"])
            if (
                not 0 < target < raw_feature.shape[-2]
                or mode not in BEHAVIOR_CONDITION_MODES
            ):
                raise ValueError("invalid T4a behavior condition")
            extra[
                scene * futures : (scene + 1) * futures,
                :,
                target,
                BEHAVIOR_CONDITION_MODES.index(mode),
            ] = 1.0
    return torch.cat((raw_feature, extra), dim=-1)


def rollout_action_diffusion(
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
    exogenous_future: np.ndarray | None = None,
    exogenous_mask: np.ndarray | None = None,
    ego_policy_id: str | None = None,
    ego_policy_model: torch.nn.Module | None = None,
    ego_policy_checkpoint: dict | None = None,
    behavior_conditions: list[dict] | None = None,
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
            key = RandomKey(
                benchmark_id,
                str(scenario_ids[scene]),
                fit_seed,
                rollout_id,
                "generation_noise",
            )
            noise[scene * k + rollout_id] = np.random.default_rng(
                key.seed()
            ).standard_normal((decisions, 15, 6, 2))
    noise_t = torch.from_numpy(noise).to(device)
    block_valid = active[:, None, 1:].expand(-1, 15, -1)
    dynamics = KinematicTrafficDynamics()
    future_states = torch.zeros((batch * k, horizon, 7, 6), device=device)
    requested = torch.zeros((batch * k, decisions, 6, 2), device=device)
    applied = torch.zeros_like(requested)
    rewritten = torch.zeros((batch * k, decisions, 6), dtype=torch.bool, device=device)
    current = history[:, -1].clone()
    cached = None
    physical_frame = 0
    target_speed, target_lane_y = initial_pnc_targets(current, lane_centers, lane_valid)
    model.eval()
    with torch.no_grad():
        for decision in range(decisions):
            pnc_raw_feature = _torch_features(
                history,
                valid_history,
                lengths,
                widths,
                lane_centers,
                lane_width,
                lane_valid,
            )
            raw_feature = _append_t4a_condition(
                pnc_raw_feature, model.config.feature_dim, behavior_conditions, k
            )
            feature = (raw_feature - feature_mean) / feature_std
            if warm_start and cached is not None:
                shifted = torch.cat((cached[:, 1:], cached[:, -1:]), dim=1)
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
            cached = block
            request = block[:, 0]
            action = request.clone()
            action[..., 0].clamp_(-8.0, 4.0)
            speed = torch.linalg.vector_norm(current[:, 1:, 2:4], dim=-1).clamp_min(
                1.0e-3
            )
            yaw_limit = torch.minimum(torch.full_like(speed, 0.6), 4.0 / speed)
            action[..., 1] = torch.maximum(
                torch.minimum(action[..., 1], yaw_limit), -yaw_limit
            )
            action *= active[:, 1:, None]
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
                    pnc_raw_feature,
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
                history = torch.cat((history[:, 1:], current[:, None]), 1)
                valid_history = torch.cat((valid_history[:, 1:], active[:, None]), 1)
                physical_frame += 1
    return RolloutBatch(
        future_states.reshape(batch, k, horizon, 7, 6).cpu().numpy(),
        requested.reshape(batch, k, decisions, 6, 2).cpu().numpy(),
        applied.reshape(batch, k, decisions, 6, 2).cpu().numpy(),
        rewritten.reshape(batch, k, decisions, 6).cpu().numpy(),
    )

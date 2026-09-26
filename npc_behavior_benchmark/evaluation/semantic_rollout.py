"""Idea C: natural softmin selection over frozen B0 joint-action candidates."""

from __future__ import annotations

import numpy as np
import torch

from traffic_components.src.core.dynamics import KinematicTrafficDynamics

from npc_behavior_benchmark.policies.interface import RandomKey
from npc_behavior_benchmark.policies.rolling_action_policy import RollingActionDiffusion
from .rollout import RolloutBatch, _torch_features
from .pnc import fixed_pnc_action, initial_pnc_targets, structured_bc_pnc_action


def _limit(actions, states, active):
    value = actions.clone()
    value[..., 0].clamp_(-8.0, 4.0)
    speed = torch.linalg.vector_norm(states[..., 1:, 2:4], dim=-1).clamp_min(1.0e-3)
    limit = torch.minimum(torch.full_like(speed, 0.6), 4.0 / speed)
    value[..., 1] = torch.maximum(torch.minimum(value[..., 1], limit), -limit)
    return value * active[..., 1:, None]


def _proxy_rollout(current, actions, active):
    dynamics = KinematicTrafficDynamics()
    state = current
    endpoints = []
    for decision in range(actions.shape[1]):
        control = torch.zeros(
            (len(state), 7, 2), device=state.device, dtype=state.dtype
        )
        control[:, 1:] = _limit(actions[:, decision], state, active)
        for _ in range(5):
            state = dynamics.step(state, control, active, 0.04)
        endpoints.append(state)
    return torch.stack(endpoints, 1)


def _natural_cost(
    states, actions, active, lengths, widths, lane_centers, lane_widths, lane_valid
):
    # states [B,D,7,6], actions [B,D,6,2]
    position = states[..., :2]
    dx = torch.abs(position[..., :, None, 0] - position[..., None, :, 0])
    dy = torch.abs(position[..., :, None, 1] - position[..., None, :, 1])
    long_scale = (lengths[:, None, :, None] + lengths[:, None, None, :]) * 0.5
    lat_scale = (widths[:, None, :, None] + widths[:, None, None, :]) * 0.5
    proximity = torch.relu(
        1.0
        - torch.maximum(dx / long_scale.clamp_min(0.5), dy / lat_scale.clamp_min(0.5))
    )
    pair = active[:, None, :, None] & active[:, None, None, :]
    upper = torch.triu(
        torch.ones((7, 7), dtype=torch.bool, device=states.device), diagonal=1
    )
    collision = (proximity * (pair & upper)).sum((-1, -2)).mean(-1)
    difference = torch.abs(states[..., 1, None] - lane_centers[:, None, None, :])
    difference = difference.masked_fill(~lane_valid[:, None, None], float("inf"))
    nearest = difference.argmin(-1)
    widths_all = lane_widths[:, None, None].expand_as(difference)
    chosen_width = torch.gather(widths_all, -1, nearest[..., None])[..., 0]
    lateral = difference.min(-1).values
    offroad = torch.relu(lateral - chosen_width * 0.5).mean((-1, -2))
    speed = torch.linalg.vector_norm(states[..., 2:4], dim=-1)
    desired = torch.linalg.vector_norm(states[:, :1, ..., 2:4], dim=-1).detach()
    speed_cost = (torch.abs(speed - desired) * active[:, None]).sum((-1, -2)) / (
        active.sum(-1) * states.shape[1]
    ).clamp_min(1)
    smooth = torch.diff(actions, dim=1).square().sum(-1)
    smooth_cost = (smooth * active[:, None, 1:]).sum((-1, -2)) / (
        active[:, 1:].sum(-1) * max(actions.shape[1] - 1, 1)
    ).clamp_min(1)
    return 5.0 * collision + 2.0 * offroad + 0.2 * speed_cost + 0.1 * smooth_cost


def rollout_semantic_policy(
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
    candidates: int = 8,
    response_samples: int = 2,
    response_aware: bool = True,
    stochastic_selection: bool = True,
    temperature: float = 0.5,
    device: torch.device,
    exogenous_future: np.ndarray | None = None,
    exogenous_mask: np.ndarray | None = None,
    ego_policy_id: str | None = None,
    ego_policy_model: torch.nn.Module | None = None,
    ego_policy_checkpoint: dict | None = None,
) -> RolloutBatch:
    batch, k, horizon = len(initial_history), int(futures), int(ego_future.shape[1])
    decisions = (horizon + 4) // 5
    base = batch * k
    c = int(candidates)
    r = int(response_samples)
    expand = lambda value: np.repeat(np.asarray(value), k, axis=0)
    history = torch.from_numpy(expand(initial_history).copy()).to(device)
    valid_history = torch.from_numpy(expand(history_valid).copy()).to(device)
    ego_log = torch.from_numpy(expand(ego_future).copy()).to(device)
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
    candidate_noise = np.empty((base, decisions, c, 15, 6, 2), np.float32)
    response_noise = np.empty((base, decisions, c, r, 15, 6, 2), np.float32)
    selection_noise = np.empty((base, decisions, c), np.float32)
    for scene in range(batch):
        for rollout_id in range(k):
            index = scene * k + rollout_id
            rng = np.random.default_rng(
                RandomKey(
                    benchmark_id,
                    str(scenario_ids[scene]),
                    fit_seed,
                    rollout_id,
                    "search_candidates",
                ).seed()
            )
            candidate_noise[index] = rng.standard_normal(candidate_noise[index].shape)
            response_noise[index] = rng.standard_normal(response_noise[index].shape)
            selection_noise[index] = rng.uniform(
                1.0e-6, 1.0 - 1.0e-6, selection_noise[index].shape
            )
    candidate_noise_t = torch.from_numpy(candidate_noise).to(device)
    response_noise_t = torch.from_numpy(response_noise).to(device)
    gumbel = -torch.log(-torch.log(torch.from_numpy(selection_noise).to(device)))
    model.eval()
    dynamics = KinematicTrafficDynamics()
    current = history[:, -1].clone()
    physical_frame = 0
    target_speed, target_lane_y = initial_pnc_targets(current, lane_centers, lane_valid)
    future_states = torch.zeros((base, horizon, 7, 6), device=device)
    requested = torch.zeros((base, decisions, 6, 2), device=device)
    applied = torch.zeros_like(requested)
    rewritten = torch.zeros((base, decisions, 6), dtype=torch.bool, device=device)
    target_valid = active[:, None, 1:].expand(-1, 15, -1)
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
            feature_c = (
                feature[:, None].expand(-1, c, -1, -1, -1).reshape(base * c, 25, 7, -1)
            )
            valid_c = (
                valid_history[:, None].expand(-1, c, -1, -1).reshape(base * c, 25, 7)
            )
            mask_c = (
                target_valid[:, None].expand(-1, c, -1, -1).reshape(base * c, 15, 6)
            )
            blocks = model.sample(
                feature_c,
                valid_c,
                mask_c,
                candidate_noise_t[:, decision].reshape(base * c, 15, 6, 2),
                inference_steps,
            ).reshape(base, c, 15, 6, 2)
            current_c = current[:, None].expand(-1, c, -1, -1).reshape(base * c, 7, 6)
            active_c = active[:, None].expand(-1, c, -1).reshape(base * c, 7)
            lengths_c = lengths[:, None].expand(-1, c, -1).reshape(base * c, 7)
            widths_c = widths[:, None].expand(-1, c, -1).reshape(base * c, 7)
            centers_c = lane_centers[:, None].expand(-1, c, -1).reshape(base * c, -1)
            lane_width_c = lane_width[:, None].expand(-1, c, -1).reshape(base * c, -1)
            lane_valid_c = lane_valid[:, None].expand(-1, c, -1).reshape(base * c, -1)
            if response_aware:
                first_states = _proxy_rollout(
                    current_c, blocks.reshape(base * c, 15, 6, 2)[:, :1], active_c
                )[:, 0]
                proxy_history = (
                    history[:, None]
                    .expand(-1, c, -1, -1, -1)
                    .reshape(base * c, 25, 7, 6)
                )
                # Approximate five new proxy frames by linear interpolation to
                # the actual one-decision endpoint, preserving causal history.
                interpolation = torch.stack(
                    [
                        current_c + (first_states - current_c) * (step / 5.0)
                        for step in range(1, 6)
                    ],
                    1,
                )
                proxy_history = torch.cat((proxy_history[:, 5:], interpolation), 1)
                proxy_feature = (
                    _torch_features(
                        proxy_history,
                        valid_c,
                        lengths_c,
                        widths_c,
                        centers_c,
                        lane_width_c,
                        lane_valid_c,
                    )
                    - feature_mean
                ) / feature_std
                proxy_feature = (
                    proxy_feature[:, None]
                    .expand(-1, r, -1, -1, -1)
                    .reshape(base * c * r, 25, 7, -1)
                )
                proxy_valid = (
                    valid_c[:, None].expand(-1, r, -1, -1).reshape(base * c * r, 25, 7)
                )
                proxy_mask = (
                    mask_c[:, None].expand(-1, r, -1, -1).reshape(base * c * r, 15, 6)
                )
                continuation = model.sample(
                    proxy_feature,
                    proxy_valid,
                    proxy_mask,
                    response_noise_t[:, decision].reshape(base * c * r, 15, 6, 2),
                    inference_steps,
                )
                # A candidate denotes the semantic plan of the currently most
                # relevant NPC.  Keep that plan fixed while the other NPCs
                # recondition on its realised first action; otherwise the
                # sampled continuation erases the candidate after 0.2 s and
                # C2 degenerates to C1/B0 selection.
                ego_distance = torch.linalg.vector_norm(
                    current_c[:, 1:, :2] - current_c[:, :1, :2], dim=-1
                )
                focal = ego_distance.masked_fill(~active_c[:, 1:], float("inf")).argmin(
                    -1
                )
                focal_plan = torch.cat(
                    (
                        blocks.reshape(base * c, 15, 6, 2)[:, 1:],
                        blocks.reshape(base * c, 15, 6, 2)[:, -1:],
                    ),
                    1,
                )
                focal_plan = (
                    focal_plan[:, None]
                    .expand(-1, r, -1, -1, -1)
                    .reshape(base * c * r, 15, 6, 2)
                )
                focal_r = focal[:, None].expand(-1, r).reshape(-1)
                continuation[
                    torch.arange(len(continuation), device=device), :, focal_r, :
                ] = focal_plan[
                    torch.arange(len(continuation), device=device), :, focal_r, :
                ]
                response_states = _proxy_rollout(
                    first_states[:, None]
                    .expand(-1, r, -1, -1)
                    .reshape(base * c * r, 7, 6),
                    continuation,
                    active_c[:, None].expand(-1, r, -1).reshape(base * c * r, 7),
                )
                costs = (
                    _natural_cost(
                        response_states,
                        continuation,
                        active_c[:, None].expand(-1, r, -1).reshape(base * c * r, 7),
                        lengths_c[:, None].expand(-1, r, -1).reshape(base * c * r, 7),
                        widths_c[:, None].expand(-1, r, -1).reshape(base * c * r, 7),
                        centers_c[:, None].expand(-1, r, -1).reshape(base * c * r, -1),
                        lane_width_c[:, None]
                        .expand(-1, r, -1)
                        .reshape(base * c * r, -1),
                        lane_valid_c[:, None]
                        .expand(-1, r, -1)
                        .reshape(base * c * r, -1),
                    )
                    .reshape(base, c, r)
                    .mean(-1)
                )
            else:
                proxy_states = _proxy_rollout(
                    current_c, blocks.reshape(base * c, 15, 6, 2), active_c
                )
                costs = _natural_cost(
                    proxy_states,
                    blocks.reshape(base * c, 15, 6, 2),
                    active_c,
                    lengths_c,
                    widths_c,
                    centers_c,
                    lane_width_c,
                    lane_valid_c,
                ).reshape(base, c)
            choice = torch.argmin(
                costs / float(temperature)
                - (gumbel[:, decision] if stochastic_selection else 0.0),
                dim=1,
            )
            selected = blocks[torch.arange(base, device=device), choice]
            request = selected[:, 0]
            action = _limit(request, current, active)
            requested[:, decision] = request
            applied[:, decision] = action
            rewritten[:, decision] = ((action - request).abs() > 1.0e-6).any(
                -1
            ) & active[:, 1:]
            frames = min(5, horizon - physical_frame)
            control = torch.zeros((base, 7, 2), device=device)
            control[:, 1:] = action
            if ego_policy_id in {"structured_bc", "structured_bc_lane_stable"}:
                if ego_policy_model is None or ego_policy_checkpoint is None:
                    raise ValueError("structured_bc requires model and checkpoint")
                stable = ego_policy_id == "structured_bc_lane_stable"
                control[:, 0] = structured_bc_pnc_action(
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
                control[:, 0] = fixed_pnc_action(
                    current,
                    active,
                    lengths,
                    target_speed,
                    target_lane_y,
                    ego_policy_id,
                    widths=widths,
                )
            for _ in range(frames):
                state = dynamics.step(current, control, active, 0.04)
                if ego_policy_id is None:
                    state[:, 0] = ego_log[:, physical_frame]
                if external is not None and external_mask is not None:
                    state = torch.where(
                        external_mask[..., None], external[:, physical_frame], state
                    )
                future_states[:, physical_frame] = state
                current = state
                history = torch.cat((history[:, 1:], current[:, None]), 1)
                valid_history = torch.cat((valid_history[:, 1:], active[:, None]), 1)
                physical_frame += 1
    return RolloutBatch(
        future_states.reshape(batch, k, horizon, 7, 6).cpu().numpy(),
        requested.reshape(batch, k, decisions, 6, 2).cpu().numpy(),
        applied.reshape(batch, k, decisions, 6, 2).cpu().numpy(),
        rewritten.reshape(batch, k, decisions, 6).cpu().numpy(),
    )

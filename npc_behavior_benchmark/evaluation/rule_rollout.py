"""Deterministic full-population IDM plus lane-keeping baseline."""

from __future__ import annotations

import numpy as np
import torch

from traffic_components.src.core.dynamics import KinematicTrafficDynamics

from .rollout import RolloutBatch


def _rule_action(current, active, lengths, target_speed, target_lane_y):
    speed = torch.linalg.vector_norm(current[..., 2:4], dim=-1)
    dx = current[:, None, :, 0] - current[:, :, None, 0]
    dy = torch.abs(current[:, None, :, 1] - current[:, :, None, 1])
    pair = active[:, :, None] & active[:, None, :] & (dx > 0.0) & (dy < 2.0)
    gap = dx - 0.5 * (lengths[:, :, None] + lengths[:, None, :])
    ranked = gap.masked_fill(~pair, float("inf"))
    nearest, index = ranked.min(-1)
    lead_speed = torch.gather(speed, 1, index)
    closing = speed - lead_speed
    maximum_accel = 1.2
    comfortable_brake = 1.5
    desired_gap = 2.0 + torch.clamp(
        speed * 1.5
        + speed * closing / (2.0 * (maximum_accel * comfortable_brake) ** 0.5),
        min=0.0,
    )
    interaction = torch.where(
        torch.isfinite(nearest),
        (desired_gap / nearest.clamp_min(0.2)).square(),
        torch.zeros_like(nearest),
    )
    acceleration = maximum_accel * (
        1.0 - (speed / target_speed.clamp_min(1.0)).pow(4) - interaction
    )
    heading = torch.atan2(current[..., 3], current[..., 2].clamp(min=1.0e-4))
    desired_heading = torch.atan((target_lane_y - current[..., 1]) / 12.0)
    heading_error = torch.atan2(
        torch.sin(desired_heading - heading), torch.cos(desired_heading - heading)
    )
    yaw_rate = heading_error / 0.8
    yaw_limit = torch.minimum(
        torch.full_like(speed, 0.6), 4.0 / speed.clamp_min(1.0e-3)
    )
    return (
        torch.stack(
            (
                acceleration.clamp(-8.0, 4.0),
                torch.maximum(torch.minimum(yaw_rate, yaw_limit), -yaw_limit),
            ),
            -1,
        )
        * active[..., None]
    )


def rollout_idm_lane_keep(
    initial_history: np.ndarray,
    history_valid: np.ndarray,
    ego_future: np.ndarray,
    lengths_m: np.ndarray,
    widths_m: np.ndarray,
    map_polylines: np.ndarray,
    map_valid: np.ndarray,
    *,
    futures: int,
    device: torch.device,
    exogenous_future: np.ndarray | None = None,
    exogenous_mask: np.ndarray | None = None,
) -> RolloutBatch:
    batch, k, horizon = len(initial_history), int(futures), int(ego_future.shape[1])
    decisions = (horizon + 4) // 5
    expand = lambda value: np.repeat(np.asarray(value), k, axis=0)
    history = torch.from_numpy(expand(initial_history).copy()).to(device)
    valid_history = torch.from_numpy(expand(history_valid).copy()).to(device)
    ego_log = torch.from_numpy(expand(ego_future).copy()).to(device)
    lengths = torch.from_numpy(expand(lengths_m).copy()).to(device)
    lines = torch.from_numpy(expand(map_polylines).copy()).to(device)
    line_valid = torch.from_numpy(expand(map_valid).copy()).to(device)
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
    active = valid_history[:, -1]
    lane_valid = line_valid.any(-1)
    centers = (lines[..., 1] * line_valid).sum(-1) / line_valid.sum(-1).clamp_min(1)
    current = history[:, -1].clone()
    distance = torch.abs(current[..., 1, None] - centers[:, None]).masked_fill(
        ~lane_valid[:, None], float("inf")
    )
    target_lane_y = torch.gather(
        centers[:, None].expand(-1, 7, -1), 2, distance.argmin(-1, keepdim=True)
    )[..., 0]
    target_speed = torch.linalg.vector_norm(current[..., 2:4], dim=-1).clamp_min(1.0)
    dynamics = KinematicTrafficDynamics()
    future_states = torch.zeros((batch * k, horizon, 7, 6), device=device)
    requested = torch.zeros((batch * k, decisions, 6, 2), device=device)
    applied = torch.zeros_like(requested)
    rewrite = torch.zeros((batch * k, decisions, 6), dtype=torch.bool, device=device)
    physical_frame = 0
    with torch.no_grad():
        for decision in range(decisions):
            full_action = _rule_action(
                current, active, lengths, target_speed, target_lane_y
            )
            requested[:, decision] = full_action[:, 1:]
            applied[:, decision] = full_action[:, 1:]
            frames = min(5, horizon - physical_frame)
            for _ in range(frames):
                stepped = dynamics.step(current, full_action, active, 0.04)
                stepped[:, 0] = ego_log[:, physical_frame]
                if external is not None and external_mask is not None:
                    stepped = torch.where(
                        external_mask[..., None], external[:, physical_frame], stepped
                    )
                future_states[:, physical_frame] = stepped
                current = stepped
                physical_frame += 1
    return RolloutBatch(
        future_states.reshape(batch, k, horizon, 7, 6).cpu().numpy(),
        requested.reshape(batch, k, decisions, 6, 2).cpu().numpy(),
        applied.reshape(batch, k, decisions, 6, 2).cpu().numpy(),
        rewrite.reshape(batch, k, decisions, 6).cpu().numpy(),
    )

"""Fixed, future-blind ego controllers for the T3 closed-loop suite."""

from __future__ import annotations

import torch

PNC_IDS = (
    "cruise",
    "idm_cautious",
    "idm_assertive",
    "trajectory_mpc",
    "structured_bc",
    "structured_bc_lane_stable",
)


def initial_pnc_targets(
    current: torch.Tensor, lane_centers: torch.Tensor, lane_valid: torch.Tensor
):
    speed = torch.linalg.vector_norm(current[:, 0, 2:4], dim=-1)
    distance = torch.abs(current[:, 0, 1, None] - lane_centers).masked_fill(
        ~lane_valid, float("inf")
    )
    lane_y = torch.gather(lane_centers, 1, distance.argmin(-1, keepdim=True))[:, 0]
    return speed, lane_y


def fixed_pnc_action(
    current: torch.Tensor,
    active: torch.Tensor,
    lengths: torch.Tensor,
    target_speed: torch.Tensor,
    target_lane_y: torch.Tensor,
    policy_id: str,
    widths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return causal ``[acceleration, yaw_rate]`` from the realised state."""
    if policy_id not in PNC_IDS:
        raise ValueError(f"unknown fixed PNC {policy_id!r}")
    if policy_id in {"structured_bc", "structured_bc_lane_stable"}:
        raise ValueError("structured_bc requires its frozen model checkpoint")
    ego = current[:, 0]
    speed = torch.linalg.vector_norm(ego[:, 2:4], dim=-1).clamp_min(0.0)
    if policy_id == "trajectory_mpc":
        return _trajectory_mpc_action(
            current,
            active,
            lengths,
            widths if widths is not None else torch.full_like(lengths, 1.9),
            target_speed,
            target_lane_y,
        )
    if policy_id == "cruise":
        desired_speed = target_speed
        acceleration = 0.8 * (desired_speed - speed)
    else:
        assertive = policy_id == "idm_assertive"
        desired_speed = (target_speed + (4.0 if assertive else 1.0)).clamp_min(1.0)
        time_headway = 0.8 if assertive else 1.8
        minimum_gap = 1.0 if assertive else 2.0
        comfortable_brake = 2.5 if assertive else 1.5
        maximum_accel = 2.0 if assertive else 1.2
        dx = current[:, 1:, 0] - ego[:, None, 0]
        same_lane = torch.abs(current[:, 1:, 1] - ego[:, None, 1]) < 2.0
        candidate = active[:, 1:] & same_lane & (dx > 0.0)
        gap = dx - 0.5 * (lengths[:, 0, None] + lengths[:, 1:])
        ranked = gap.masked_fill(~candidate, float("inf"))
        nearest, index = ranked.min(-1)
        lead_speed_all = torch.linalg.vector_norm(current[:, 1:, 2:4], dim=-1)
        lead_speed = torch.gather(lead_speed_all, 1, index[:, None])[:, 0]
        closing = speed - lead_speed
        desired_gap = minimum_gap + torch.clamp(
            speed * time_headway
            + speed * closing / (2.0 * (maximum_accel * comfortable_brake) ** 0.5),
            min=0.0,
        )
        interaction = torch.where(
            torch.isfinite(nearest),
            (desired_gap / nearest.clamp_min(0.2)).square(),
            torch.zeros_like(nearest),
        )
        acceleration = maximum_accel * (
            1.0 - (speed / desired_speed.clamp_min(1.0)).pow(4) - interaction
        )
    heading = torch.atan2(ego[:, 3], ego[:, 2].clamp(min=1.0e-4))
    lateral_error = target_lane_y - ego[:, 1]
    desired_heading = torch.atan(lateral_error / 12.0)
    heading_error = torch.atan2(
        torch.sin(desired_heading - heading), torch.cos(desired_heading - heading)
    )
    yaw_rate = (heading_error / 0.8).clamp(-0.35, 0.35)
    return (
        torch.stack((acceleration.clamp(-8.0, 4.0), yaw_rate), -1)
        * active[:, :1].float()
    )


def _trajectory_mpc_action(
    current: torch.Tensor,
    active: torch.Tensor,
    lengths: torch.Tensor,
    widths: torch.Tensor,
    target_speed: torch.Tensor,
    target_lane_y: torch.Tensor,
) -> torch.Tensor:
    """Finite-trajectory MPC with a causal constant-velocity traffic model.

    The controller replans at 5 Hz.  It scores 35 constant-control ego
    trajectories over 1.6 seconds while extrapolating the currently observed
    NPC states.  No logged future state or model-specific hidden state enters
    the planner.
    """
    ego = current[:, 0]
    batch = ego.shape[0]
    dtype, device = ego.dtype, ego.device
    speed = torch.linalg.vector_norm(ego[:, 2:4], dim=-1).clamp_min(1.0e-3)
    heading = torch.atan2(ego[:, 3], ego[:, 2].clamp(min=1.0e-4))

    lateral_error = target_lane_y - ego[:, 1]
    desired_heading = torch.atan(lateral_error / 12.0)
    heading_error = torch.atan2(
        torch.sin(desired_heading - heading), torch.cos(desired_heading - heading)
    )
    base_yaw = (heading_error / 0.8).clamp(-0.30, 0.30)
    base_acceleration = (0.8 * (target_speed - speed)).clamp(-4.0, 2.0)

    acceleration_offsets = torch.tensor(
        (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0), dtype=dtype, device=device
    )
    # Small offsets permit lane-centering alternatives without turning a
    # lane-keeping PNC into an uncommanded evasive lane-change planner.
    yaw_offsets = torch.tensor(
        (-0.04, -0.02, 0.0, 0.02, 0.04), dtype=dtype, device=device
    )
    acceleration = (
        base_acceleration[:, None, None] + acceleration_offsets[None, :, None]
    ).expand(-1, -1, 5)
    yaw_rate = (base_yaw[:, None, None] + yaw_offsets[None, None, :]).expand(-1, 7, -1)
    acceleration = acceleration.reshape(batch, -1).clamp(-8.0, 4.0)
    yaw_rate = yaw_rate.reshape(batch, -1)
    yaw_limit = torch.minimum(torch.full_like(speed, 0.6), 4.0 / speed)
    yaw_rate = torch.maximum(
        torch.minimum(yaw_rate, yaw_limit[:, None]), -yaw_limit[:, None]
    )

    candidates = acceleration.shape[1]
    x = ego[:, 0, None].expand(-1, candidates).clone()
    y = ego[:, 1, None].expand(-1, candidates).clone()
    candidate_speed = speed[:, None].expand(-1, candidates).clone()
    candidate_heading = heading[:, None].expand(-1, candidates).clone()
    cost = 0.04 * acceleration.square() + 0.20 * yaw_rate.square()

    npc = current[:, None, 1:]
    npc_active = active[:, None, 1:]
    npc_ahead_at_start = npc[..., 0] > ego[:, 0, None, None]
    half_length = 0.5 * (lengths[:, 0, None, None] + lengths[:, None, 1:])
    half_width = 0.5 * (widths[:, 0, None, None] + widths[:, None, 1:])
    dt = 0.2
    for step in range(1, 9):
        x = (
            x
            + candidate_speed * torch.cos(candidate_heading) * dt
            + 0.5 * acceleration * torch.cos(candidate_heading) * dt**2
        )
        y = (
            y
            + candidate_speed * torch.sin(candidate_heading) * dt
            + 0.5 * acceleration * torch.sin(candidate_heading) * dt**2
        )
        candidate_speed = (candidate_speed + acceleration * dt).clamp(0.0, 50.0)
        candidate_heading = candidate_heading + yaw_rate * dt

        elapsed = float(step) * dt
        npc_x = npc[..., 0] + npc[..., 2] * elapsed
        npc_y = npc[..., 1] + npc[..., 3] * elapsed
        longitudinal_gap = torch.abs(npc_x - x[..., None]) - half_length
        lateral_gap = torch.abs(npc_y - y[..., None]) - half_width
        overlap = torch.relu(-longitudinal_gap) * torch.relu(-lateral_gap)
        near_lane = torch.exp(-torch.square(lateral_gap.clamp_min(0.0) / 1.0))
        near_longitudinal = torch.relu(8.0 - longitudinal_gap.clamp_min(0.0)).square()
        front_gap = npc_x - x[..., None] - half_length
        front_buffer = torch.relu(12.0 - front_gap).square() * npc_ahead_at_start
        collision_cost = (
            600.0 * overlap + near_lane * (2.0 * near_longitudinal + 3.0 * front_buffer)
        ) * npc_active

        cost = cost + collision_cost.sum(-1)
        cost = cost + 0.08 * torch.square(candidate_speed - target_speed[:, None])
        cost = cost + 2.00 * torch.square(y - target_lane_y[:, None])
        cost = cost + 0.50 * torch.square(candidate_heading)

    previous_longitudinal_acceleration = ego[:, 4] * torch.cos(heading) + ego[
        :, 5
    ] * torch.sin(heading)
    cost = cost + 0.08 * torch.square(
        acceleration - previous_longitudinal_acceleration[:, None]
    )
    best = cost.argmin(-1)
    action = torch.stack(
        (
            acceleration.gather(1, best[:, None])[:, 0],
            yaw_rate.gather(1, best[:, None])[:, 0],
        ),
        -1,
    )
    return action * active[:, :1].float()


@torch.no_grad()
def structured_bc_pnc_action(
    model: torch.nn.Module,
    checkpoint: dict,
    features: torch.Tensor,
    history_valid: torch.Tensor,
    current: torch.Tensor,
    active: torch.Tensor,
    target_lane_y: torch.Tensor | None = None,
    learned_yaw_weight: float = 1.0,
) -> torch.Tensor:
    """Run the frozen learned PNC and apply only the common control bounds."""
    feature_mean = torch.as_tensor(checkpoint["feature_mean"], device=features.device)
    feature_std = torch.as_tensor(checkpoint["feature_std"], device=features.device)
    action_mean = torch.as_tensor(checkpoint["action_mean"], device=features.device)
    action_std = torch.as_tensor(checkpoint["action_std"], device=features.device)
    normalized = model((features - feature_mean) / feature_std, history_valid)
    action = normalized * action_std + action_mean
    acceleration = action[:, 0].clamp(-8.0, 4.0)
    speed = torch.linalg.vector_norm(current[:, 0, 2:4], dim=-1).clamp_min(1.0e-3)
    yaw_limit = torch.minimum(torch.full_like(speed, 0.6), 4.0 / speed)
    yaw_rate = action[:, 1]
    if target_lane_y is not None:
        ego = current[:, 0]
        heading = torch.atan2(ego[:, 3], ego[:, 2].clamp(min=1.0e-4))
        desired_heading = torch.atan((target_lane_y - ego[:, 1]) / 12.0)
        heading_error = torch.atan2(
            torch.sin(desired_heading - heading), torch.cos(desired_heading - heading)
        )
        lane_stabilizer = (heading_error / 0.8).clamp(-0.35, 0.35)
        yaw_rate = lane_stabilizer + float(learned_yaw_weight) * yaw_rate
    yaw_rate = torch.maximum(torch.minimum(yaw_rate, yaw_limit), -yaw_limit)
    return torch.stack((acceleration, yaw_rate), -1) * active[:, :1].float()

"""Shared conditional driving actions and competing lane-change hazards."""

import torch

from .idm_population import torch_idm


def physical_leader_observation(
    states, valid, dimensions, *, horizon_s=1.5, target_y=None
):
    """Nearest or imminently unsafe leader in the actors' swept body corridors."""
    speed = torch.linalg.vector_norm(states[..., 2:4], dim=-1)
    heading = torch.atan2(states[..., 3], states[..., 2])
    extent = 0.5 * (
        dimensions[..., 1] * heading.cos().abs()
        + dimensions[..., 0] * heading.sin().abs()
    )
    dx = states[..., None, :, 0] - states[..., :, None, 0]
    dy = states[..., None, :, 1] - states[..., :, None, 1]
    future_y = states[..., 1] + states[..., 3] * horizon_s
    if target_y is not None:
        future_y = torch.where(torch.isfinite(target_y), target_y, future_y)
    future_dy = future_y[..., None, :] - future_y[..., :, None]
    overlap = (
        torch.minimum(dy.abs(), future_dy.abs())
        < extent[..., :, None] + extent[..., None, :]
    )
    overlap |= dy * future_dy <= 0
    candidate = overlap & valid[..., :, None] & valid[..., None, :] & (dx > 0)
    gap = dx - 0.5 * (dimensions[..., :, None, 0] + dimensions[..., None, :, 0])
    projected = gap + (speed[..., None, :] - speed[..., :, None]) * horizon_s
    unsafe = candidate & (
        projected
        < torch.maximum(
            torch.full_like(speed[..., :, None], 2.0), 0.1 * speed[..., :, None]
        )
    )
    nearest = gap.masked_fill(~candidate, torch.inf)
    risky = projected.masked_fill(~unsafe, torch.inf)
    leader = torch.where(unsafe.any(-1), risky.argmin(-1), nearest.argmin(-1))
    present = candidate.any(-1)
    selected_gap = torch.gather(gap, -1, leader[..., None]).squeeze(-1)
    selected_speed = torch.gather(speed, -1, leader)
    return (
        torch.where(present, selected_gap, 1e6),
        speed,
        torch.where(present, selected_speed, speed),
        torch.where(present, leader, -1),
    )


def causal_maneuver_features(current, source_y, nearest_center, lane_width):
    """Update a source anchor only after settling, using realized motion."""
    settled = (current[..., 3].abs() < 0.15) & (
        (current[..., 1] - nearest_center).abs() < 0.6
    )
    source_y = torch.where(settled, nearest_center, source_y)
    offset = (current[..., 1] - source_y).abs()
    changing = (current[..., 3].abs() >= 0.15) | (offset >= 0.6)
    features = torch.stack(
        (
            changing.to(current.dtype),
            (offset / lane_width).clamp(0, 1),
            current[..., 3].abs(),
        ),
        -1,
    )
    return features, source_y


def longitudinal_features(
    gap, speed, leader_speed, history_features=None, maneuver_features=None
):
    current = torch.stack(
        (
            speed,
            torch.log1p(gap.clamp(0, 120)),
            speed - leader_speed,
            (gap >= 1e5).to(speed.dtype),
        ),
        -1,
    )
    if history_features is None:
        history_features = torch.zeros(
            (*speed.shape, 2), device=speed.device, dtype=speed.dtype
        )
    if maneuver_features is None:
        maneuver_features = torch.zeros(
            (*speed.shape, 3), device=speed.device, dtype=speed.dtype
        )
    return torch.cat((current, history_features, maneuver_features), -1)


def response_history_features(
    current, history, history_valid, leader_index, dimensions
):
    """Observed leader motion; own action persistence belongs to the residual process."""
    own = current[:, 1:]
    if history.shape[1] < 6:
        return torch.zeros(
            (*own.shape[:2], 2), device=current.device, dtype=current.dtype
        )
    past = history[:, -6]
    index = leader_index.clamp_min(0)
    old_leader = torch.gather(past, 1, index[..., None].expand(-1, -1, 6))
    leader = torch.gather(current, 1, index[..., None].expand(-1, -1, 6))
    leader_valid = leader_index.ge(0) & torch.gather(history_valid[:, -6], 1, index)
    _, _, _, past_leader = physical_leader_observation(
        past, history_valid[:, -6], dimensions
    )
    past_leader = past_leader[:, 1:]
    leader_valid &= past_leader.eq(leader_index)
    acceleration = (
        torch.linalg.vector_norm(leader[..., 2:4], dim=-1)
        - torch.linalg.vector_norm(old_leader[..., 2:4], dim=-1)
    ) / 0.2
    return torch.stack(
        (
            torch.where(leader_valid, acceleration, 0),
            leader_valid.to(current.dtype),
        ),
        -1,
    )


def _design(features, parameters):
    mean = features.new_tensor(parameters["feature_mean"])
    scale = features.new_tensor(parameters["feature_std"])
    normalized = (features - mean) / scale
    return torch.cat((torch.ones_like(normalized[..., :1]), normalized), -1)


def corrected_acceleration(
    gap,
    speed,
    leader_speed,
    theta,
    *,
    correction=None,
    residual_state=None,
    innovation=None,
    initial_innovation=None,
    history_features=None,
    maneuver_features=None,
    initialize=None,
    dt=0.04
):
    """Execute one persistent driver's IDM action and optional fitted residual."""
    acceleration = torch_idm(gap, speed, leader_speed, theta).clamp(-8, 4)
    parameters = None if correction is None else correction.get("longitudinal")
    if parameters is None:
        return acceleration, residual_state
    design = _design(
        longitudinal_features(
            gap, speed, leader_speed, history_features, maneuver_features
        ),
        parameters,
    )
    mean = design @ design.new_tensor(parameters["mean_coef"])
    standard_deviation = torch.exp(
        0.5 * (design @ design.new_tensor(parameters["variance_coef"])).clamp(-16, 4)
    )
    if innovation is None:
        innovation = torch.randn_like(speed)
    temporal = parameters["autoregression"]
    if float(dt) != temporal["dt_seconds"]:
        raise ValueError("Residual parameters use a different physical time step")
    a, b = temporal["coefficients"]
    correlation = temporal["initial_correlation"]
    if initial_innovation is None:
        initial_innovation = torch.randn(
            (*speed.shape, 2), device=speed.device, dtype=speed.dtype
        )
    previous = (
        correlation * innovation
        + (1 - correlation**2) ** 0.5 * initial_innovation[..., 0]
    )
    following = (
        a * innovation
        + b * previous
        + temporal["innovation_std"] * initial_innovation[..., 1]
    )
    initial = torch.stack(
        (innovation, previous, following, torch.zeros_like(speed)), -1
    )
    if residual_state is None:
        residual_state = initial
    else:
        age = residual_state[..., 3] + dt
        advance = age >= temporal["interval_seconds"] - 1e-7
        current = torch.where(advance, residual_state[..., 2], residual_state[..., 0])
        previous = torch.where(advance, residual_state[..., 0], residual_state[..., 1])
        following = torch.where(
            advance,
            a * current + b * previous + temporal["innovation_std"] * innovation,
            residual_state[..., 2],
        )
        age = torch.where(advance, age - temporal["interval_seconds"], age).clamp_min(0)
        residual_state = torch.stack((current, previous, following, age), -1)
        if initialize is not None:
            residual_state = torch.where(initialize[..., None], initial, residual_state)
    fraction = residual_state[..., 3] / temporal["interval_seconds"]
    variance = (
        (1 - fraction) ** 2 + fraction**2 + 2 * fraction * (1 - fraction) * correlation
    )
    colored = (
        (1 - fraction) * residual_state[..., 0] + fraction * residual_state[..., 2]
    ) / variance.sqrt()
    corrected = (
        acceleration
        + parameters["mean_strength"] * mean
        + parameters["noise_strength"] * standard_deviation * colored
    )
    immediate_gap = gap < torch.maximum(torch.full_like(speed, 2.0), 0.1 * speed)
    closing_brake = (speed > leader_speed) & acceleration.lt(0)
    corrected = torch.where(
        immediate_gap | closing_brake, torch.minimum(corrected, acceleration), corrected
    )
    return corrected.clamp(-8, 4), residual_state


def exclusive_lane_probabilities(
    incentive, safe, mobil, *, features=None, correction=None, style_offset=None
):
    """One-second keep/lower/higher probabilities with deterministic double-choice ties."""
    parameters = None if correction is None else correction.get("lane")
    side = torch.sigmoid(
        float(mobil["bias"]) + float(mobil["slope"]) * incentive.clamp(-12, 12)
    )

    def exclusive(mask):
        lower, higher = torch.where(mask, side, 0).unbind(-1)
        both = lower * higher
        lower_mass = lower * (1 - higher) + torch.where(lower >= higher, both, 0)
        higher_mass = higher * (1 - lower) + torch.where(higher > lower, both, 0)
        return torch.stack(((1 - lower) * (1 - higher), lower_mass, higher_mass), -1)

    probabilities = exclusive(safe)
    if parameters is not None:
        shift = _design(features, parameters) @ features.new_tensor(parameters["coef"])
        logits = exclusive(safe).clamp_min(1e-12).log()
        logits[..., 1:] += shift
        logits[..., 1:] += float(parameters.get("logit_bias", 0.0))
        logits[..., 1:] += logits.new_tensor([-1, 1]) * float(
            parameters.get("direction_bias", 0.0)
        )
        logits[..., 1:] = logits[..., 1:].masked_fill(~safe, -torch.inf)
        strength = float(parameters["strength"])
        probabilities = (1 - strength) * probabilities + strength * torch.softmax(
            logits, -1
        )
    if style_offset is not None:
        logits = probabilities.clamp_min(1e-12).log()
        logits[..., 1:] += style_offset[..., None]
        logits[..., 1:] = logits[..., 1:].masked_fill(~safe, -torch.inf)
        probabilities = torch.softmax(logits, -1)
    return probabilities


def lane_frame_probabilities(probabilities, dt=0.04):
    """Convert mutually exclusive one-second choices to constant competing hazards."""
    keep = probabilities[..., 0].clamp(1e-12, 1)
    move = -torch.expm1(torch.log(keep) * float(dt))
    side_share = probabilities[..., 1:] / (1 - keep).clamp_min(1e-12)[..., None]
    return torch.cat(((1 - move)[..., None], move[..., None] * side_share), -1)


def sample_lane_choice(probabilities, uniform, dt=0.04):
    frame = lane_frame_probabilities(probabilities, dt)
    index = (uniform[..., None] >= frame.cumsum(-1)).sum(-1).clamp_max(2)
    return torch.where(
        index == 1,
        -torch.ones_like(index),
        torch.where(index == 2, torch.ones_like(index), torch.zeros_like(index)),
    )

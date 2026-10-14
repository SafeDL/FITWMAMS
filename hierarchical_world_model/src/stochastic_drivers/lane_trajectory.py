"""Train-conditioned lane paths and curvature-feedforward Frenet tracking."""

import json
from pathlib import Path

import torch
from torch import nn

from ..core.dynamics import HIGHD_INTEGRATION_SPEED_MAX_MPS, KinematicTrafficDynamics

MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "results/hierarchical_world_model/model_assets/lane_trajectory.json"
)
DT = 0.04
SHAPE_BASIS = (
    (0, 0, 0, 64, -192, 192, -64, 0),
    (0, 0, 0, -64, 320, -576, 448, -128),
)


def path_coefficients(state, target_y, distance, shape):
    """Match current position, slope and curvature; settle at the target centre."""
    y, vx, vy, ax, ay = state[..., 1:6].unbind(-1)
    vx = vx.clamp_min(0.5)
    delta = target_y - y
    slope = vy / vx
    second = (ay * vx - vy * ax) / vx.pow(3)
    first = slope * distance
    half_second = 0.5 * second * distance.square()
    zeros = torch.zeros_like(y)
    coefficients = torch.stack(
        (
            y,
            first,
            half_second,
            10 * delta - 6 * first - 3 * half_second,
            -15 * delta + 8 * first + 3 * half_second,
            6 * delta - 3 * first - half_second,
            zeros,
            zeros,
        ),
        -1,
    )
    # Both learned shape terms and their first two derivatives vanish at endpoints.
    basis = state.new_tensor(SHAPE_BASIS)
    return coefficients + delta[..., None] * (shape @ basis)


def path_at(coefficients, origin_x, distance, x):
    """Evaluate d(s), d'(s), d''(s) on the straight highD road."""
    u = ((x - origin_x) / distance).clamp(0, 1)
    position = torch.zeros_like(u)
    slope = torch.zeros_like(u)
    second = torch.zeros_like(u)
    for power in range(7, -1, -1):
        position = position * u + coefficients[..., power]
        if power >= 1:
            slope = slope * u + power * coefficients[..., power]
        if power >= 2:
            second = second * u + power * (power - 1) * coefficients[..., power]
    return position, slope / distance, second / distance.square()


def frenet_yaw(
    state,
    coefficients,
    origin_x,
    distance,
    *,
    frequency=1.2,
    acceleration_limit=1.5,
    jerk_limit=2.0,
    dt=DT,
):
    """Curvature feedforward plus cross-track/heading feedback, in rad/s."""
    reference_y, slope, second = path_at(
        coefficients, origin_x, distance, state[..., 0] + state[..., 2] * dt * 0.5
    )
    return frenet_feedback(
        state,
        reference_y,
        slope,
        second,
        frequency=frequency,
        acceleration_limit=acceleration_limit,
        jerk_limit=jerk_limit,
        dt=dt,
    )


def frenet_feedback(
    state,
    reference_y,
    slope,
    second,
    *,
    frequency,
    acceleration_limit,
    jerk_limit,
    dt=DT,
):
    speed = torch.linalg.vector_norm(state[..., 2:4], dim=-1).clamp_min(0.5)
    heading = KinematicTrafficDynamics.heading_from_velocity(
        state[..., 3], state[..., 2]
    )
    reference_heading = torch.atan(slope)
    heading_error = torch.atan2(
        torch.sin(heading - reference_heading), torch.cos(heading - reference_heading)
    )
    error_d = (state[..., 1] - reference_y) / torch.sqrt(1 + slope.square())
    curvature = second / (1 + slope.square()).pow(1.5)
    feedforward = speed * curvature * torch.cos(heading_error)
    feedforward /= (1 - curvature * error_d).clamp_min(0.5)
    yaw = feedforward - 2 * frequency * torch.sin(heading_error)
    yaw -= frequency**2 * error_d / speed
    heading_cos, heading_sin = torch.cos(heading), torch.sin(heading)
    previous_lateral = -state[..., 4] * heading_sin + state[..., 5] * heading_cos
    lateral = speed * yaw
    lateral = lateral.clamp(-acceleration_limit, acceleration_limit)
    lateral = lateral.clamp(
        previous_lateral - jerk_limit * dt, previous_lateral + jerk_limit * dt
    )
    # State acceleration contains longitudinal acceleration projected onto y as well.
    return lateral / speed


def limit_realized_lateral_acceleration(
    state, yaw, longitudinal_acceleration, *, acceleration_limit, jerk_limit, dt=DT
):
    """Bound next-state normal acceleration, using the actually selected longitudinal action."""
    speed = torch.linalg.vector_norm(state[..., 2:4], dim=-1)
    next_speed = (speed + longitudinal_acceleration * dt).clamp(
        0, HIGHD_INTEGRATION_SPEED_MAX_MPS
    )
    heading = KinematicTrafficDynamics.heading_from_velocity(
        state[..., 3], state[..., 2]
    )
    previous = -state[..., 4] * heading.sin() + state[..., 5] * heading.cos()
    lateral = (next_speed * yaw).clamp(-acceleration_limit, acceleration_limit)
    lateral = lateral.clamp(previous - jerk_limit * dt, previous + jerk_limit * dt)
    return torch.where(next_speed > 0.5, lateral / next_speed.clamp_min(0.5), 0.0)


class LearnedLaneTrajectory(nn.Module):
    """Low-dimensional regression with jointly resampled Train residuals."""

    def __init__(self, path=MODEL_PATH, *, correction=None, model=None):
        super().__init__()
        report = (
            json.loads(Path(path).read_text(encoding="utf-8"))
            if model is None
            else model
        )
        for key in (
            "feature_mean",
            "feature_std",
            "feature_min",
            "feature_max",
            "coefficients",
            "residuals",
            "parameter_min",
            "parameter_max",
        ):
            self.register_buffer(
                key, torch.tensor(report["fit"][key], dtype=torch.float32)
            )
        self.frequency = float(report["tracking"]["frequency"])
        self.acceleration_limit = float(report["tracking"]["acceleration_limit_mps2"])
        self.jerk_limit = float(report["tracking"]["jerk_limit_mps3"])
        self.correction = correction

    def parameters_for(self, state, dimensions, target_y, noise=None):
        direction = torch.sign(target_y - state[..., 1])
        features = torch.stack(
            (
                state[..., 2],
                dimensions[..., 0].log(),
                dimensions[..., 1].log(),
                (target_y - state[..., 1]).abs(),
                direction * state[..., 3],
                direction * state[..., 5],
            ),
            -1,
        )
        features = features.clamp(self.feature_min, self.feature_max)
        design = torch.cat(
            (
                torch.ones_like(features[..., :1]),
                (features - self.feature_mean) / self.feature_std,
            ),
            -1,
        )
        parameters = design @ self.coefficients
        if noise is not None:
            probability = 0.5 * (1 + torch.erf(noise[..., 0] / 2**0.5))
            index = (
                (probability * len(self.residuals))
                .long()
                .clamp_max(len(self.residuals) - 1)
            )
            parameters = parameters + self.residuals[index]
        if self.correction is not None:
            shift = parameters.new_tensor(
                [self.correction["log_duration_scale"], *self.correction["shape_shift"]]
            )
            shift[1:] *= self.residuals.std(0)[1:]
            parameters = parameters + shift
        return parameters.clamp(self.parameter_min, self.parameter_max)

    def plan(self, state, dimensions, target_y, noise=None, duration_scale=1.0):
        parameters = self.parameters_for(state, dimensions, target_y, noise)
        duration = parameters[..., 0].exp() * duration_scale
        fraction = ((target_y - state[..., 1]).abs() / self.feature_mean[3]).clamp(
            0.2, 1
        )
        duration = duration * fraction.sqrt()
        distance = duration * state[..., 2].clamp_min(0.5)
        distance += 0.5 * state[..., 4].clamp(0, 2) * duration.square()
        coefficients = path_coefficients(state, target_y, distance, parameters[..., 1:])
        return coefficients, state[..., 0].clone(), distance

    def track(self, state, coefficients, origin_x, distance):
        return frenet_yaw(
            state,
            coefficients,
            origin_x,
            distance,
            frequency=self.frequency,
            acceleration_limit=self.acceleration_limit,
            jerk_limit=self.jerk_limit,
        )

    def safe_plan(
        self, states, valid, dimensions, target_y, noise, acceleration, road_bounds
    ):
        paths = [
            self.plan(states[:, 1:], dimensions[:, 1:], target_y, noise, scale)
            for scale in (1.0, 1.4, 1.8)
        ]
        coefficients = torch.stack([path[0] for path in paths], -2)
        origins = torch.stack([path[1] for path in paths], -1)
        distances = torch.stack([path[2] for path in paths], -1)
        clear = path_clearance(
            states,
            valid,
            dimensions,
            coefficients,
            origins,
            distances,
            acceleration,
            road_bounds=road_bounds,
            acceleration_limit=2 * self.acceleration_limit,
        )
        choice = clear.to(torch.int32).argmax(-1)
        path = (
            torch.gather(
                coefficients, -2, choice[..., None, None].expand(-1, -1, 1, 8)
            ).squeeze(-2),
            torch.gather(origins, -1, choice[..., None]).squeeze(-1),
            torch.gather(distances, -1, choice[..., None]).squeeze(-1),
        )
        return path, clear.any(-1)


def path_clearance(
    states,
    valid,
    dimensions,
    coefficients,
    origin,
    distance,
    acceleration,
    *,
    margin=0.5,
    road_bounds=None,
    acceleration_limit=None,
):
    """Preview measured-acceleration motion, with oriented footprint extents."""
    own = states[:, 1:]
    candidates = distance.shape[-1]
    times = states.new_tensor([0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6])

    def progress(vx, ax):
        stop = torch.where(ax < 0, vx / (-ax).clamp_min(1e-3), torch.full_like(vx, 2))
        accelerated_time = torch.minimum(times, stop[..., None].clamp(0, 2))
        remaining_speed = (vx[..., None] + ax[..., None] * accelerated_time).clamp_min(
            0
        )
        return (
            vx[..., None] * accelerated_time
            + 0.5 * ax[..., None] * accelerated_time.square()
            + remaining_speed * (times - accelerated_time)
        )

    own_progress = progress(own[..., 2], acceleration)
    own_x = own[..., 0, None] + own_progress
    other_x = states[..., 0, None] + progress(states[..., 2], states[..., 4])
    other_y = states[..., 1, None] + states[..., 3, None] * times
    query = own_x[:, :, None].expand(-1, -1, candidates, -1)
    y, slope, second = path_at(
        coefficients[..., None, :], origin[..., None], distance[..., None], query
    )
    heading = torch.atan(slope)
    own_length = dimensions[:, 1:, 0, None, None] * heading.cos().abs()
    own_length += dimensions[:, 1:, 1, None, None] * heading.sin().abs()
    own_width = dimensions[:, 1:, 1, None, None] * heading.cos().abs()
    own_width += dimensions[:, 1:, 0, None, None] * heading.sin().abs()
    other_heading = KinematicTrafficDynamics.heading_from_velocity(
        states[..., 3], states[..., 2]
    )
    other_length = (
        dimensions[..., 0] * other_heading.cos().abs()
        + dimensions[..., 1] * other_heading.sin().abs()
    )
    other_width = (
        dimensions[..., 1] * other_heading.cos().abs()
        + dimensions[..., 0] * other_heading.sin().abs()
    )
    length = (
        0.5 * (own_length[:, :, :, None] + other_length[:, None, None, :, None])
        + margin
    )
    width = (
        0.5 * (own_width[:, :, :, None] + other_width[:, None, None, :, None]) + margin
    )
    actors = states.shape[1]
    identity = torch.arange(actors, device=states.device)[
        None, None, None, :, None
    ] == (torch.arange(actors - 1, device=states.device)[None, :, None, None, None] + 1)
    overlap = valid[:, None, None, :, None] & ~identity
    overlap = overlap & (other_x[:, None, None] - query[:, :, :, None]).abs().lt(length)
    overlap = overlap & (other_y[:, None, None] - y[:, :, :, None]).abs().lt(width)
    clear = ~overlap.flatten(-2).any(-1)
    if road_bounds is not None:
        lower, upper = road_bounds
        road_clear = (y - own_width / 2 >= lower[:, None, None, None]) & (
            y + own_width / 2 <= upper[:, None, None, None]
        )
        clear &= road_clear.all(-1)
    if acceleration_limit is not None:
        future_vx = (
            own[..., 2, None] + acceleration[..., None] * times.clamp_max(2)
        ).clamp_min(0)
        lateral = (
            future_vx[:, :, None].square() * second / torch.sqrt(1 + slope.square())
        )
        clear &= lateral.abs().amax(-1).le(acceleration_limit)
    return clear

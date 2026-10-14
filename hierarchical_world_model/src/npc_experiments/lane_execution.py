"""Lane execution with observed target and longitudinal speed, not decision validation."""

import numpy as np
import torch

from ..core.dynamics import (
    DynamicsConfig,
    HIGHD_INTEGRATION_SPEED_MAX_MPS,
    KinematicTrafficDynamics,
)
from ..stochastic_drivers.lane_trajectory import (
    LearnedLaneTrajectory,
    limit_realized_lateral_acceleration,
)

DT = 0.04


def recording_interval(records, metric):
    """Bootstrap whole recordings; frames and repeated styles are not new events."""
    _, index = np.unique([row["recording_id"] for row in records], return_inverse=True)
    values = np.asarray([row[metric] for row in records], float)
    totals, counts = np.bincount(index, weights=values), np.bincount(index)
    draws = np.random.default_rng(20261002).integers(
        len(counts), size=(2000, len(counts))
    )
    means = totals[draws].sum(1) / counts[draws].sum(1)
    return np.quantile(means, [0.025, 0.975]).tolist()


@torch.no_grad()
def trajectory_replay(events, observations, model_path, *, seed=None):
    """Use the runtime planner/tracker, with measured longitudinal motion as input."""
    model = LearnedLaneTrajectory(model_path)
    records = []
    generator = torch.Generator().manual_seed(0 if seed is None else seed)
    for event in events.itertuples():
        trajectory = observations["states"][
            observations["offsets"][event.trajectory_index] : observations["offsets"][
                event.trajectory_index + 1
            ]
        ]
        data = trajectory[25:]
        # Complete-event caches store y/vx/vy/ay, so estimate initial ax from past vx.
        initial_ax = (trajectory[25, 1] - trajectory[24, 1]) / DT
        initial = [0, data[0, 0], data[0, 1], data[0, 2], initial_ax, data[0, 3]]
        target = event.target_y_m - event.source_y_m
        noise = None if seed is None else torch.randn(2, generator=generator)
        predicted, clipped = lane_path_rollout(
            model, initial, [event.length_m, event.width_m], target, data[:, 1], noise
        )
        positions, velocities, accelerations = (
            predicted[:, index] for index in (1, 3, 5)
        )
        completion = settled_time(predicted, target)
        peak = float(np.abs(velocities).max())
        duration = float((np.abs(velocities) > 0.15).sum() * DT)
        records.append(
            {
                "event_id": event.event_id,
                "recording_id": int(event.recording_id),
                "direction": int(event.direction),
                "speed_mps": event.speed_mps,
                "length_m": event.length_m,
                "width_m": event.width_m,
                "lateral_RMSE_m": float(
                    np.sqrt(np.mean((positions - data[:, 0]) ** 2))
                ),
                "final_target_error_m": float(abs(positions[-1] - float(target))),
                "peak_vy_error_mps": abs(peak - event.peak_lateral_speed_mps),
                "duration_error_s": abs(duration - event.duration_s),
                "model_peak_vy_mps": peak,
                "observed_peak_vy_mps": event.peak_lateral_speed_mps,
                "model_duration_s": duration,
                "observed_duration_s": event.duration_s,
                "completion_s": completion,
                "observed_exposure_s": (len(data) - 1) * DT,
                "longitudinal_condition_clipped_steps": clipped,
                "max_vx_condition_error_mps": float(
                    np.max(np.abs(predicted[:, 2] - data[:, 1]))
                ),
                "peak_ay_mps2": float(np.abs(accelerations).max()),
                "lateral_jerk_q99_mps3": float(
                    np.quantile(np.abs(np.diff(accelerations)) / DT, 0.99)
                ),
                "logged_jerk_q99_mps3": float(
                    np.quantile(np.abs(np.diff(data[:, 3])) / DT, 0.99)
                ),
                "completion_within_observed_exposure": completion is not None,
            }
        )
    return records


def lane_path_rollout(
    model, initial, dimensions, target, longitudinal_speed, noise=None
):
    """Supply only longitudinal speed; integrate all lateral motion without truth resets."""
    state = torch.tensor(initial, dtype=torch.float32)
    dimensions = torch.tensor(dimensions, dtype=state.dtype)
    target = torch.as_tensor(target, dtype=state.dtype)
    coefficients, origin, distance = model.plan(state, dimensions, target, noise)
    dynamics = KinematicTrafficDynamics(
        DynamicsConfig(speed_max_mps=HIGHD_INTEGRATION_SPEED_MAX_MPS)
    )
    states = [state.numpy().copy()]
    clipped = 0
    for vx in longitudinal_speed[1:]:
        yaw = model.track(state, coefficients, origin, distance)
        heading = dynamics.heading_from_velocity(state[3], state[2]) + yaw * DT
        speed = torch.linalg.vector_norm(state[2:4])
        # Fix next vx without importing the observed next vy or heading.
        next_speed = float(vx) / heading.cos().clamp_min(0.1)
        acceleration = (next_speed - speed) / DT
        clipped += int(not -8 <= float(acceleration) <= 4)
        yaw = limit_realized_lateral_acceleration(
            state,
            yaw,
            acceleration.clamp(-8, 4),
            acceleration_limit=model.acceleration_limit,
            jerk_limit=model.jerk_limit,
        )
        state = dynamics.step(
            state,
            torch.stack((acceleration, yaw)),
            torch.tensor(True),
            DT,
        )
        states.append(state.numpy().copy())
    return np.asarray(states), clipped


def settled_time(states, target):
    """First ten observed/generated samples close to the target; not a success label."""
    heading = np.arctan2(states[:, 3], states[:, 2])
    stable = 0
    for tick, close in enumerate(
        (np.abs(states[:, 1] - target) < 0.4) & (np.abs(heading) < 0.05)
    ):
        stable = stable + 1 if close else 0
        if stable >= 10:
            return tick * DT
    return None

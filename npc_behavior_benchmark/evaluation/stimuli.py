"""Fixed evaluator-owned stimulus trajectories and paired-response metrics."""

from __future__ import annotations

import numpy as np
import torch

from traffic_components.src.core.dynamics import KinematicTrafficDynamics


def longitudinal_stimulus_trajectories(
    initial_states: np.ndarray,
    active: np.ndarray,
    stimulus_indices: np.ndarray,
    signed_acceleration_mps2: np.ndarray,
    *,
    horizon_frames: int = 75,
    pulse_frames: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Return constant-speed and acceleration-pulse branches from one backend."""
    initial = torch.as_tensor(np.asarray(initial_states), dtype=torch.float32)
    valid = torch.as_tensor(np.asarray(active), dtype=torch.bool)
    indices = torch.as_tensor(np.asarray(stimulus_indices), dtype=torch.long)
    signed = torch.as_tensor(np.asarray(signed_acceleration_mps2), dtype=torch.float32)
    dynamics = KinematicTrafficDynamics()
    branches = []
    for intervention in (False, True):
        current = initial.clone()
        future = torch.zeros((len(initial), horizon_frames, 7, 6), dtype=torch.float32)
        for frame in range(horizon_frames):
            control = torch.zeros((len(initial), 7, 2), dtype=torch.float32)
            if intervention and frame < pulse_frames:
                control[torch.arange(len(initial)), indices, 0] = signed
            current = dynamics.step(current, control, valid, 0.04)
            future[:, frame] = current
        branches.append(future.numpy())
    return branches[0], branches[1]


def lateral_stimulus_trajectories(
    initial_states: np.ndarray,
    active: np.ndarray,
    stimulus_indices: np.ndarray,
    target_lane_y_m: np.ndarray,
    controller_rate_rps: np.ndarray,
    *,
    horizon_frames: int = 75,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lane-keeping and critically damped lane-transition branches."""
    initial = torch.as_tensor(np.asarray(initial_states), dtype=torch.float32)
    valid = torch.as_tensor(np.asarray(active), dtype=torch.bool)
    indices = torch.as_tensor(np.asarray(stimulus_indices), dtype=torch.long)
    target_y = torch.as_tensor(
        np.array(target_lane_y_m, dtype=np.float32, copy=True, order="C"),
        dtype=torch.float32,
    )
    controller_rate = torch.as_tensor(
        np.array(controller_rate_rps, dtype=np.float32, copy=True, order="C"),
        dtype=torch.float32,
    )
    batch_index = torch.arange(len(initial))
    dynamics = KinematicTrafficDynamics()
    branches = []
    for intervention in (False, True):
        current = initial.clone()
        future = torch.zeros((len(initial), horizon_frames, 7, 6), dtype=torch.float32)
        for frame in range(horizon_frames):
            control = torch.zeros((len(initial), 7, 2), dtype=torch.float32)
            if intervention:
                stimulus = current[batch_index, indices]
                speed = torch.linalg.vector_norm(stimulus[:, 2:4], dim=-1).clamp_min(
                    1.0
                )
                heading = torch.atan2(stimulus[:, 3], stimulus[:, 2].clamp(min=1.0e-4))
                lateral_error = target_y - stimulus[:, 1]
                yaw_rate = (
                    controller_rate.square() * lateral_error / speed
                    - 2.0 * controller_rate * heading
                )
                shared_limit = torch.minimum(torch.full_like(speed, 0.6), 4.0 / speed)
                bounded_yaw = torch.maximum(
                    torch.minimum(yaw_rate, shared_limit), -shared_limit
                )
                control[batch_index, indices, 1] = bounded_yaw
            current = dynamics.step(current, control, valid, 0.04)
            future[:, frame] = current
        branches.append(future.numpy())
    return branches[0], branches[1]


def sustained_response_latency(
    paired_acceleration_delta: np.ndarray,
    expected_sign: np.ndarray,
    *,
    threshold_mps2: float = 0.2,
    consecutive_decisions: int = 2,
    decision_dt_s: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect signed response onset; nonresponses are right-censored."""
    delta = np.asarray(paired_acceleration_delta)
    sign = np.asarray(expected_sign)
    aligned = delta * sign[..., None]
    above = aligned > float(threshold_mps2)
    sustained = above.copy()
    for offset in range(1, consecutive_decisions):
        shifted = np.zeros_like(above)
        shifted[..., :-offset] = above[..., offset:]
        sustained &= shifted
    responded = sustained.any(-1)
    first = sustained.argmax(-1)
    latency = first.astype(np.float64) * float(decision_dt_s)
    latency[~responded] = np.nan
    return responded, latency

"""Closed-loop stochastic highD evaluation at a 25 Hz plant rate."""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .model import ACCELERATION_OBSERVATION_NOISE, sample_driver_joint
from .reference_kernels import GPHistory, idm, se_kernel


def crps_ensemble(samples: np.ndarray, observation: np.ndarray) -> float:
    """Empirical CRPS: E|X-y| - 1/2 E|X-X'|, averaged over all entries."""
    values = np.asarray(samples, float)
    obs = np.asarray(observation, float)
    if values.ndim != 3 or values.shape[0] != obs.shape[0] or values.shape[2] != obs.shape[1]:
        raise ValueError("samples must be [segment, ensemble, time] and observation [segment, time]")
    first = np.mean(np.abs(values - obs[:, None, :]), axis=1)
    ordered = np.sort(values, axis=1)
    n = values.shape[1]
    weights = 2.0 * np.arange(1, n + 1) - n - 1.0
    second = np.sum(weights[None, :, None] * ordered, axis=1) / (n * n)
    return float(np.mean(first - second))


def _prefix_history(segment: dict[str, np.ndarray], theta: np.ndarray, prefix_index: int, memory_s: float | None,
                    observation_noise: float = ACCELERATION_OBSERVATION_NOISE) -> GPHistory:
    dt, stride = 0.04, 5  # plant is 25 Hz; stochastic driver action is causally refreshed at 5 Hz.
    begin = 0 if memory_s is None else max(0, prefix_index - int(round(memory_s / dt)))
    # Each observed residual must describe the same held 0.2 s action as the
    # simulator.  A one-frame velocity difference at every fifth frame mixes
    # 25 Hz measurement noise with a 5 Hz GP and was the source of a hidden
    # fit/simulation mismatch in the full-cohort route.
    first = begin + (-begin) % stride
    indices = np.arange(first, prefix_index - stride + 1, stride)
    v, lv, gap = segment["follower_v"], segment["leader_v"], segment["gap"]
    closing = v[indices] - lv[indices]
    residual = ((v[indices + stride] - v[indices]) / (stride * dt)
                - idm(gap[indices], v[indices], closing, theta))
    return GPHistory(times=(indices * dt).tolist(), residuals=np.asarray(residual, float).tolist(), sigma=1.0, lengthscale=1.0,
                     observation_noise=float(observation_noise))


def _completed_decision_actions(segment: dict[str, np.ndarray], prefix_index: int, memory_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return completed 5 Hz action observations and their decision-frame IDs.

    Calibration, GP conditioning and the plant adapter all treat a driver
    decision as a value held for five native frames.  Prefix individualisation
    must use that same object rather than one-frame finite differences.
    """
    dt, stride = .04, 5
    start = max(0, prefix_index - int(round(memory_s / dt)))
    first = start + (-start) % stride
    # The action starting at ``prefix_index - stride`` ends exactly at the
    # observed prediction origin.  It is a completed causal transition (not
    # a future action) and must be available to the parameter filter, just as
    # it is to the GP residual history.
    indices = np.arange(first, prefix_index - stride + 1, stride, dtype=int)
    speed = np.asarray(segment["follower_v"], float)
    return indices, (speed[indices + stride] - speed[indices]) / (stride * dt)


def condition_driver_candidates(posterior: dict[str, np.ndarray], segment: dict[str, np.ndarray], prefix_index: int,
                                rng: np.random.Generator, candidates: int = 128,
                                memory_s: float = 5.) -> tuple[np.ndarray, np.ndarray]:
    """Build a causal importance-weighted joint-parameter pool at one origin.

    The hierarchical population posterior is the prior for an unseen driver.
    At a prediction origin, observed speed/gap/leader states identify which
    styles can explain the preceding actions.  This inexpensive importance
    update deliberately uses no sample after ``prefix_index``; GP residual
    conditioning remains in :func:`_rollout`.
    """
    if candidates < 2:
        raise ValueError("at least two parameter candidates are required")
    draws = np.asarray([sample_driver_joint(posterior, rng) for _ in range(candidates)])
    indices, observed = _completed_decision_actions(segment, prefix_index, memory_s)
    if len(indices) < 3:
        return draws, np.full(candidates, 1. / candidates)
    speed, leader, gap = segment["follower_v"], segment["leader_v"], segment["gap"]
    closing = speed[indices] - leader[indices]
    predicted = np.asarray([idm(gap[indices], speed[indices], closing, draw[:5]) for draw in draws])
    residual = observed - predicted
    # Use the same SE-GP + iid covariance as the paper likelihood.  Treating
    # correlated actions as conditionally independent makes a 5 s prefix look
    # like 25 independent parameter observations and collapses the importance
    # distribution unrealistically.  128 Cholesky factorizations of at most a
    # 25 x 25 matrix are negligible beside closed-loop rollouts.
    times = indices.astype(float) * .04
    iid_sigma = draws[:, 7] if draws.shape[1] > 7 else np.full(candidates, ACCELERATION_OBSERVATION_NOISE)
    log_weight = np.empty(candidates)
    for row, (draw, value, iid) in enumerate(zip(draws, residual, iid_sigma)):
        covariance = se_kernel(times, times, draw[5], draw[6])
        covariance.flat[:: len(indices) + 1] += float(iid) ** 2
        try:
            factor = cho_factor(covariance, lower=True, check_finite=False)
            solved = cho_solve(factor, value, check_finite=False)
            log_weight[row] = -.5 * (value @ solved + 2. * np.log(np.diag(factor[0])).sum())
        except np.linalg.LinAlgError:
            log_weight[row] = -np.inf
    log_weight -= np.max(log_weight)
    weight = np.exp(log_weight); weight /= np.sum(weight)
    return draws, weight


def condition_driver_joint(posterior: dict[str, np.ndarray], segment: dict[str, np.ndarray], prefix_index: int,
                           rng: np.random.Generator, candidates: int = 128, memory_s: float = 5.) -> np.ndarray:
    """Draw one joint parameter vector conditional on a causal action prefix."""
    draws, weight = condition_driver_candidates(posterior, segment, prefix_index, rng, candidates, memory_s)
    return draws[int(rng.choice(len(draws), p=weight))]


def _rollout(segment: dict[str, np.ndarray], parameters: np.ndarray, *, prefix_index: int, horizon_frames: int,
             model: str, rng: np.random.Generator,
             memory_s: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt, update_frames = 0.04, 5
    theta, sigma, ell = parameters[:5], float(parameters[5]), float(parameters[6])
    iid_sigma = float(parameters[7]) if len(parameters) > 7 else ACCELERATION_OBSERVATION_NOISE
    v = float(segment["follower_v"][prefix_index])
    x = float(segment["follower_x"][prefix_index])
    positions, speeds, accelerations, discrepancies = [], [], [], []
    if model == "ma_idm":
        history = _prefix_history(segment, theta, prefix_index, memory_s, iid_sigma)
        history.sigma, history.lengthscale, history.memory_seconds = sigma, ell, memory_s
    noise = 0.0
    for step in range(horizon_frames):
        absolute_index = prefix_index + step
        if model == "ma_idm" and step % update_frames == 0:
            # The released paper simulator draws both the temporally
            # correlated GP discrepancy and the independent ``s2_a`` action
            # residual at each decision instant.  The latter is not a sensor
            # error and must therefore enter the generated acceleration.
            noise = history.step(absolute_index * dt, rng.standard_normal()) + iid_sigma * rng.standard_normal()
        elif model == "b_idm" and step % update_frames == 0:
            noise = sigma * rng.standard_normal()
        leader_x = float(segment["leader_x"][absolute_index])
        leader_v = float(segment["leader_v"][absolute_index])
        gap = leader_x - x - float(segment["length_sum"])
        acceleration = float(idm(gap, v, v - leader_v, theta) + noise)
        # Ballistic plant integration at native 25 Hz.  Non-negative speed is the only physical projection.
        x += v * dt + 0.5 * acceleration * dt * dt
        v = max(0.0, v + acceleration * dt)
        positions.append(x); speeds.append(v); accelerations.append(acceleration); discrepancies.append(noise)
    return np.asarray(positions), np.asarray(speeds), np.asarray(accelerations), np.asarray(discrepancies)

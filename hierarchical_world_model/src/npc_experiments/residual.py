"""Residual time structure fitted within contiguous physical-driver histories."""

import numpy as np
from scipy.optimize import minimize


def driver_residuals(data, residual):
    identity = data["recording"].astype(np.int64) * 100000 + data["vehicle"]
    _, index = np.unique(identity, return_inverse=True)
    means = np.bincount(index, weights=residual) / np.bincount(index)
    return residual - means[index], means


def residual_correlations(data, residual, lags=(1, 2, 5, 10, 20)):
    order = np.lexsort((data["frame"], data["vehicle"], data["recording"]))
    recording, vehicle, frame = (
        data[key][order] for key in ("recording", "vehicle", "frame")
    )
    values = residual[order]
    rows = []
    for lag in lags:
        valid = (
            (recording[lag:] == recording[:-lag])
            & (vehicle[lag:] == vehicle[:-lag])
            & (frame[lag:] - frame[:-lag] == 5 * lag)
        )
        a, b = values[:-lag][valid], values[lag:][valid]
        covariance = np.dot(a, b)
        correlation = covariance / max(np.sqrt(np.dot(a, a) * np.dot(b, b)), 1e-12)
        rows.append(
            {
                "lag_seconds": lag * 0.2,
                "pairs": int(valid.sum()),
                "correlation": float(correlation),
            }
        )
    return rows


def autoregression_parameters(coefficients, *, time_scale=1.0, dt=0.04):
    """Keep fitted coarse AR(2) covariance and set a physical sampling interval."""
    poles = np.roots([1, -coefficients[0], -coefficients[1]]).astype(complex)
    a, b = map(float, coefficients)
    rho = a / (1 - b)
    rho2 = a * rho + b
    variance = 1 - a * rho - b * rho2
    if variance <= 0 or abs(rho) >= 1 or np.max(np.abs(poles)) >= 1:
        raise ValueError("Residual autoregression must have a stationary covariance")
    return {
        "source_coefficients": list(map(float, coefficients)),
        "coefficients": [a, b],
        "innovation_std": float(np.sqrt(variance)),
        "initial_correlation": rho,
        "dt_seconds": dt,
        "interval_seconds": 0.2 * time_scale,
        "time_scale": float(time_scale),
    }


def fit_autoregression(data, standardized):
    correlations = residual_correlations(data, standardized)
    order = np.lexsort((data["frame"], data["vehicle"], data["recording"]))
    recording, vehicle, frame = (
        data[key][order] for key in ("recording", "vehicle", "frame")
    )
    valid = (
        (recording[2:] == recording[:-2])
        & (vehicle[2:] == vehicle[:-2])
        & (frame[2:] - frame[1:-1] == 5)
        & (frame[1:-1] - frame[:-2] == 5)
    )
    values = np.asarray(standardized)[order]
    x = np.column_stack((values[1:-1][valid], values[:-2][valid]))
    target = values[2:][valid]
    gram, rhs = x.T @ x, x.T @ target

    def objective(partial):
        p, q = np.tanh(partial)
        coefficients = np.array([p * (1 - q), q])
        error = gram @ coefficients - rhs
        jacobian = np.array([[(1 - p * p) * (1 - q), -p * (1 - q * q)], [0, 1 - q * q]])
        return float(
            (0.5 * coefficients @ gram @ coefficients - coefficients @ rhs)
            / max(len(target), 1)
        ), jacobian.T @ error / max(len(target), 1)

    fit = minimize(
        objective, [0.0, 0.0], jac=True, method="L-BFGS-B", bounds=[(-4, 4), (-4, 4)]
    )
    p, q = np.tanh(fit.x)
    coefficients = [float(p * (1 - q)), float(q)]
    parameters = autoregression_parameters(coefficients)
    return parameters, correlations


def temporal_statistics(states, valid, cohort, population):
    """Common position derivatives and per-vehicle residual covariance sums."""
    from scipy.ndimage import minimum_filter1d

    from ..stochastic_drivers.idm_population import idm_acceleration
    from .metrics import (
        diagnostic_states,
        longitudinal_acceleration,
    )
    from .road import neighbors

    if states.ndim == 3:
        states, valid = states[None], valid[None]
    derived = diagnostic_states(states, 25)[:, ::5]
    inside = (states[..., 0] >= cohort["bounds"][0]) & (
        states[..., 0] <= cohort["bounds"][1]
    )
    present = minimum_filter1d(
        (valid & inside).astype(np.uint8), 25, axis=1, mode="constant"
    ).astype(bool)[:, ::5]
    acceleration = longitudinal_acceleration(derived)
    speed = np.linalg.norm(derived[..., 2:4], axis=-1)
    residual = np.zeros_like(acceleration)
    for repeat in range(len(states)):
        for begin in range(0, derived.shape[1], 32):
            end = min(begin + 32, derived.shape[1])
            current = derived[repeat, begin:end]
            mask = present[repeat, begin:end]
            _, leader, gap, _, _ = neighbors(
                current, mask, cohort["dimensions"], cohort["lane_centers"]
            )
            own = speed[repeat, begin:end]
            front = np.take_along_axis(own, leader.clip(0), axis=-1)
            front = np.where(leader >= 0, front, own)
            expected = idm_acceleration(
                gap, own, front, np.exp(population["log_mean"])
            ).clip(-8, 4)
            residual[repeat, begin:end] = acceleration[repeat, begin:end] - expected
    count = present.sum(1)
    mean = np.divide(
        (residual * present).sum(1),
        count,
        out=np.zeros_like(count, dtype=float),
        where=count > 0,
    )
    residual -= mean[:, None]
    sums = []
    for lag in (1, 2, 5, 10, 20):
        uninterrupted = minimum_filter1d(
            present.astype(np.uint8), lag + 1, axis=1, origin=lag // 2, mode="constant"
        ).astype(bool)
        mask = present[:, :-lag] & uninterrupted[:, lag:]
        a, b = residual[:, :-lag], residual[:, lag:]
        sums.append(
            np.stack(
                [
                    (a * b * mask).sum((1, 2)),
                    (a * a * mask).sum((1, 2)),
                    (b * b * mask).sum((1, 2)),
                    mask.sum((1, 2)),
                ],
                axis=-1,
            )
        )
    jerk = (acceleration[:, 1:] - acceleration[:, :-1]) / 0.2
    jerk_valid = present[:, 1:] & present[:, :-1]
    edges = np.r_[-np.inf, np.linspace(-20, 20, 81), np.inf]
    hist = np.stack(
        [
            np.histogram(jerk[index][jerk_valid[index]], edges)[0]
            for index in range(len(states))
        ]
    )
    return np.stack(sums, axis=1), hist

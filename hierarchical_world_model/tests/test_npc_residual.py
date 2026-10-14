import numpy as np
import torch
import pytest

from hierarchical_world_model.src.npc_experiments.residual import (
    autoregression_parameters,
    driver_residuals,
    residual_correlations,
)
from hierarchical_world_model.src.stochastic_drivers.driver_policy import (
    corrected_acceleration,
)


def test_residual_correlations_do_not_join_gaps_or_physical_drivers():
    data = {
        "recording": np.array([1, 1, 1, 1, 2, 2]),
        "vehicle": np.array([1, 1, 1, 2, 1, 1]),
        "frame": np.array([0, 5, 15, 20, 0, 5]),
    }
    residual = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    centered, means = driver_residuals(data, residual)
    np.testing.assert_allclose(means, [2, 4, 5.5])
    np.testing.assert_allclose(centered, [-1, 0, 1, 0, -0.5, 0.5])
    assert residual_correlations(data, residual, lags=(1,))[0]["pairs"] == 2


@pytest.mark.parametrize("coefficients", [[1.88, -0.90], [0.92, 0.05]])
def test_ar2_has_stationary_variance_and_reproduces_serial_covariance(coefficients):
    parameters = autoregression_parameters(coefficients)
    a, b = parameters["coefficients"]
    rho = parameters["initial_correlation"]
    rng = np.random.default_rng(11)
    count = 12000
    noise = rng.standard_normal((180, count)).astype(np.float64)
    state = None
    speed = torch.full((count,), 20.0, dtype=torch.float64)
    theta = torch.tensor([33.0, 2.0, 1.2, 1.0, 2.0], dtype=torch.float64)
    correction = {
        "longitudinal": {
            "feature_mean": [0] * 9,
            "feature_std": [1] * 9,
            "mean_coef": [0] * 10,
            "variance_coef": [0] * 10,
            "autoregression": parameters,
            "mean_strength": 0.0,
            "noise_strength": 0.1,
        }
    }
    traces = []
    initial = torch.from_numpy(rng.standard_normal((count, 2)))
    from hierarchical_world_model.src.stochastic_drivers.idm_population import torch_idm

    base = torch_idm(torch.full_like(speed, 80), speed, speed, theta)
    for innovation in noise:
        action, state = corrected_acceleration(
            torch.full_like(speed, 80),
            speed,
            speed,
            theta,
            correction=correction,
            residual_state=state,
            innovation=torch.from_numpy(innovation),
            initial_innovation=initial,
        )
        traces.append(((action - base) / 0.1).numpy().copy())
    traces = np.asarray(traces)
    np.testing.assert_allclose(traces.var(axis=1), 1, atol=0.04)
    expected = [1.0, rho]
    for lag in range(2, 21):
        expected.append(a * expected[-1] + b * expected[-2])
    knots = traces[::5]
    for lag in (1, 2, 5, 10, 20):
        empirical = np.mean(knots[:-lag] * knots[lag:])
        assert abs(empirical - expected[lag]) < 0.035


def test_stationary_fit_recovers_ar_process_without_crossing_a_frame_gap():
    from hierarchical_world_model.src.npc_experiments.residual import fit_autoregression

    rng = np.random.default_rng(16)
    values = np.zeros(10000)
    noise = rng.standard_normal(len(values))
    for tick in range(2, len(values)):
        values[tick] = 0.7 * values[tick - 1] - 0.2 * values[tick - 2] + noise[tick]
    frames = np.arange(len(values)) * 5
    frames[5000:] += 100
    data = {
        "frame": frames,
        "vehicle": np.zeros(len(values), int),
        "recording": np.ones(len(values), int),
    }
    fitted, observed = fit_autoregression(data, values)
    np.testing.assert_allclose(fitted["coefficients"], [0.7, -0.2], atol=0.035)
    assert observed[0]["pairs"] == len(values) - 2
    assert np.max(abs(np.roots([1, *(-np.asarray(fitted["coefficients"]))]))) < 1

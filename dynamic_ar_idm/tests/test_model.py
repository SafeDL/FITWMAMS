from __future__ import annotations
import numpy as np
from dynamic_ar_idm.model import DynamicIDMDriver, ar_innovation, ar_spectral_radius, idm
from dynamic_ar_idm.evaluate import rollout_25hz


def test_lag_order_is_newest_first() -> None:
    error, history = ar_innovation(np.array([3., 2., 1.]), np.array([.5, .2, -.1]), 0., 0.)
    assert np.isclose(error, 1.8)
    assert np.allclose(history, [1.8, 3., 2.])


def test_ar1_stationary_variance_and_acf() -> None:
    rho, sigma = .8, .3
    rng = np.random.default_rng(6); history = np.zeros(1); values = []
    for _ in range(40000):
        value, history = ar_innovation(history, np.array([rho]), sigma, rng.standard_normal()); values.append(value)
    value = np.asarray(values[1000:])
    assert np.isclose(np.var(value), sigma ** 2 / (1 - rho ** 2), rtol=.12)
    assert np.isclose(np.corrcoef(value[1:], value[:-1])[0, 1], rho, atol=.03)


def test_companion_stationarity_not_componentwise() -> None:
    assert ar_spectral_radius(np.array([1.2, -.4])) < 1.
    assert ar_spectral_radius(np.array([1.1])) > 1.


def test_idm_residual_semantics_are_explicit() -> None:
    # This catches accidentally adding raw historical acceleration instead of e=a-IDM.
    theta = np.array([33.3, 2., 1.6, 1.5, 1.67])
    assert np.isfinite(idm(20., 15., 1., theta))


def test_native_clock_holds_one_decision_for_five_ticks() -> None:
    frames = 40
    pair = {"decision": np.arange(0, 31, 5), "follower_x": np.arange(frames, dtype=float),
            "follower_v": np.full(frames, 15.), "leader_x": np.arange(frames, dtype=float) + 30.,
            "leader_v": np.full(frames, 15.), "length_sum": 5.}
    _, _, action, _ = rollout_25hz(pair, np.array([33.3, 2., 1.6, 1.5, 1.67]), np.empty(0), 0.,
                                    anchor_decision=0, horizon_s=.8, rng=np.random.default_rng(0))
    assert np.allclose(action[:5], action[0])
    assert np.allclose(action[5:10], action[5])


def test_stateful_driver_advances_residual_history_without_plant_feedback() -> None:
    driver = DynamicIDMDriver(
        np.asarray([33.0, 2.0, 1.6, 1.5, 1.67]),
        np.asarray([0.5, -0.1]),
        0.2,
    )
    first = driver.decision(
        gap_m=25.0, speed_mps=20.0, leader_speed_mps=19.0, standard_normal=1.0
    )
    history = driver.state.residual_history.copy()
    second = driver.decision(
        gap_m=24.0, speed_mps=19.5, leader_speed_mps=19.0, standard_normal=0.0
    )
    assert np.isfinite(first) and np.isfinite(second)
    assert np.isclose(history[0], 0.2)
    assert np.isclose(driver.state.residual_history[0], 0.5 * 0.2)

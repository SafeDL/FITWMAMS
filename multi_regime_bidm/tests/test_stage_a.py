from __future__ import annotations

import numpy as np

from multi_regime_bidm.src.hsmm import fit_hsmm
from multi_regime_bidm.src.hsmm import FiniteHSMM
from multi_regime_bidm.src.online_filter import filter_step, initialize_filter, predict_filter, regime_posterior
from multi_regime_bidm.src.style import decision_grid, style_features
from multi_regime_bidm.src.evaluation import _initial_filter, _style_from_prefix
from multi_regime_bidm.src.driver import MultiRegimeIDMDriver


def test_decision_grid_uses_complete_held_action_intervals() -> None:
    pair = {"follower_v": np.arange(16, dtype=float), "leader_v": np.zeros(16), "gap": np.ones(16) * 20.}
    observation, action = decision_grid(pair)
    assert observation.shape == (3, 3)
    assert np.allclose(action, 25.)  # five m/s over 0.2 s
    assert style_features(pair).shape == (3,)


def test_finite_hsmm_returns_complete_offline_segmentations() -> None:
    rng = np.random.default_rng(3)
    sequences = [np.r_[rng.normal((-2., 0., 0.), .15, size=(15, 3)), rng.normal((2., 1., -1.), .15, size=(15, 3))]
                 for _ in range(6)]
    model, labels, durations = fit_hsmm(sequences, states=3, duration_max=10, iterations=3)
    assert model.transition.shape == (3, 3)
    assert np.allclose(np.diag(model.transition), 0.)
    assert all(len(a) == len(b) and np.sum(d) == len(a) for a, b, d in zip(sequences, labels, durations))


def test_online_duration_filter_conserves_mass_and_is_causal() -> None:
    model = FiniteHSMM(means=np.zeros((2, 3)), variances=np.ones((2, 3)),
                       transition=np.array([[0., 1.], [1., 0.]]), initial=np.array([.6, .4]),
                       duration_lambda=np.array([1.5, 2.]), duration_max=3,
                       observation_mean=np.zeros(3), observation_scale=np.ones(3))
    at_t = initialize_filter(model, np.zeros(3))
    predicted = predict_filter(model, at_t)
    assert np.isclose(predicted.sum(), 1.)
    assert np.all(predicted >= 0.)
    at_next = filter_step(model, at_t, np.array([30., 30., 30.]))
    # Immutable state ensures a later observation cannot rewrite time-t mass.
    assert np.allclose(regime_posterior(at_t), np.array([.6, .4]))
    assert at_next.observations_seen == 2 and at_next.low_observation_likelihood


def test_online_duration_prediction_matches_explicit_two_state_enumeration() -> None:
    """A tiny R=2, D=3 case guards duration indexing and no-self transitions."""
    model = FiniteHSMM(means=np.zeros((2, 3)), variances=np.ones((2, 3)),
                       transition=np.array([[0., 1.], [1., 0.]]), initial=np.array([.5, .5]),
                       duration_lambda=np.array([1.2, 2.4]), duration_max=3,
                       observation_mean=np.zeros(3), observation_scale=np.ones(3))
    state = initialize_filter(model, np.zeros(3))
    mass = np.array([[.10, .20, .15], [.05, .30, .20]])
    state = type(state)(mass=mass, observations_seen=state.observations_seen)
    predicted = predict_filter(model, state)
    duration = np.exp(model._duration_logpmf().T)
    expected = np.zeros_like(mass)
    for previous in range(2):
        for remaining in range(3):
            if remaining:
                expected[previous, remaining - 1] += mass[previous, remaining]
            else:
                for current in range(2):
                    if current != previous:
                        expected[current] += mass[previous, remaining] * model.transition[previous, current] * duration[current]
    assert np.allclose(predicted, expected, atol=1.e-12)


def test_prefix_style_assignment_does_not_read_future_frames() -> None:
    pair = {"follower_v": np.linspace(10., 16., 201), "leader_v": np.full(201, 17.), "gap": np.full(201, 30.)}
    centers = np.asarray(((1., 1., 1.), (2., 2., 2.), (3., 3., 3.)))
    mean, scale = np.zeros(3), np.ones(3)
    before = _style_from_prefix(pair, centers, mean, scale, prefix_frames=125)
    changed = {key: value.copy() for key, value in pair.items()}
    changed["follower_v"][126:] = 1000.
    assert _style_from_prefix(changed, centers, mean, scale, prefix_frames=125) == before


def test_prefix_filter_includes_current_frame_without_future() -> None:
    model = FiniteHSMM(
        means=np.zeros((2, 3)), variances=np.ones((2, 3)),
        transition=np.array([[0., 1.], [1., 0.]]), initial=np.array([.6, .4]),
        duration_lambda=np.array([1.5, 2.]), duration_max=3,
        observation_mean=np.zeros(3), observation_scale=np.ones(3),
    )
    pair = {
        "follower_v": np.linspace(10., 12., 131),
        "leader_v": np.full(131, 12.),
        "gap": np.full(131, 25.),
    }
    state = _initial_filter(model, pair, prefix_frames=125)
    assert state.observations_seen == 26  # frames 0, 5, ..., 125


def test_online_driver_uses_causal_filter_and_injected_randomness() -> None:
    model = FiniteHSMM(
        means=np.zeros((2, 3)),
        variances=np.ones((2, 3)),
        transition=np.array([[0.0, 1.0], [1.0, 0.0]]),
        initial=np.array([0.6, 0.4]),
        duration_lambda=np.array([1.5, 2.0]),
        duration_max=3,
        observation_mean=np.zeros(3),
        observation_scale=np.ones(3),
    )
    theta = np.tile(np.asarray([33.0, 2.0, 1.6, 1.5, 1.67]), (2, 1))
    driver = MultiRegimeIDMDriver(model, theta, np.asarray([0.1, 0.2]))
    driver.reset(np.asarray([[10.0, 5.0, 0.0], [10.2, 5.0, 0.0]]), seed=4)
    decision = driver.decision(
        gap_m=10.2,
        speed_mps=5.0,
        leader_speed_mps=5.0,
        standard_normal=0.0,
        regime_uniform=0.25,
    )
    seen = driver.state.observations_seen
    driver.observe(gap_m=10.1, speed_mps=5.0, leader_speed_mps=4.9)
    assert np.isfinite(decision.requested_acceleration_mps2)
    assert np.isclose(np.sum(decision.regime_probability), 1.0)
    assert driver.state.observations_seen == seen + 1

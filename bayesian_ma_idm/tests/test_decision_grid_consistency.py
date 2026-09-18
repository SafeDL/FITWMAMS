from __future__ import annotations

import numpy as np

from bayesian_ma_idm.src.evaluation import _completed_decision_actions, _prefix_history
from bayesian_ma_idm.src.full_model import DECISION_STRIDE, _residual, _starts
from bayesian_ma_idm.src.reference_kernels import idm


def _pair(count: int = 31) -> dict[str, np.ndarray]:
    time = np.arange(count) * .04
    return {
        "follower_v": 12. + .3 * time,
        "leader_v": 14. - .1 * time,
        "gap": 24. + .2 * time,
    }


def test_full_fit_residual_uses_held_driver_decision_grid() -> None:
    pair = _pair()
    theta = np.array([30., 2., 1.3, 1.5, 1.7])
    indices = np.arange(0, len(pair["follower_v"]) - DECISION_STRIDE, DECISION_STRIDE)
    expected = ((pair["follower_v"][indices + DECISION_STRIDE] - pair["follower_v"][indices]) / .2
                - idm(pair["gap"][indices], pair["follower_v"][indices],
                      pair["follower_v"][indices] - pair["leader_v"][indices], theta))
    assert np.allclose(_residual(pair, theta), expected)


def test_full_fit_uses_the_author_20_point_window_with_10_point_hop() -> None:
    assert np.array_equal(_starts(50), np.array([0, 10, 20, 30]))


def test_prefix_history_conditions_only_completed_5hz_actions() -> None:
    pair = _pair(251)
    pair.update({"follower_x": np.cumsum(pair["follower_v"]) * .04,
                 "leader_x": np.cumsum(pair["leader_v"]) * .04,
                 "length_sum": np.asarray(4.5)})
    history = _prefix_history(pair, np.array([30., 2., 1.3, 1.5, 1.7]), 250, 5.)
    # The last observed action spans [9.8, 10.0] seconds.  It may condition
    # the first simulated action at 10.0 s, but no future interval is read.
    assert np.isclose(history.times[-1], 9.8)
    assert len(history.times) == 25


def test_prefix_personalisation_uses_the_same_completed_5hz_actions() -> None:
    pair = _pair(251)
    index, action = _completed_decision_actions(pair, 250, 5.)
    assert np.isclose(index[-1] * .04, 9.8)
    expected = (pair["follower_v"][index + 5] - pair["follower_v"][index]) / .2
    assert np.allclose(action, expected)

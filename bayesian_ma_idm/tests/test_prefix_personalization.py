from __future__ import annotations

import numpy as np

from bayesian_ma_idm.src.evaluation import condition_driver_candidates, condition_driver_joint


def test_prefix_personalization_never_reads_prediction_future() -> None:
    """Changing states after the origin must not change a causal style draw."""
    length, origin = 220, 150
    time = np.arange(length) * .04
    speed = 12. + .2 * time
    segment = {"follower_v": speed.copy(), "leader_v": speed + 1., "gap": np.full(length, 20.)}
    altered = {key: value.copy() for key, value in segment.items()}
    # The state at the origin is observed and completes the last held action.
    # Only states strictly after that observed origin are prediction future.
    altered["follower_v"][origin + 1:] += 100.
    altered["leader_v"][origin + 1:] -= 100.
    altered["gap"][origin + 1:] = .01
    posterior = {
        "population_log_mean": np.tile(np.log([30., 2., 1.4, 1.4, 1.8, .3, 1.5]), (3, 1)),
        "population_log_cov": np.tile(np.eye(7)[None] * .01, (3, 1, 1)),
    }
    first = condition_driver_joint(posterior, segment, origin, np.random.default_rng(41), candidates=16)
    second = condition_driver_joint(posterior, altered, origin, np.random.default_rng(41), candidates=16)
    assert np.array_equal(first, second)


def test_conditioned_candidate_pool_is_normalized_and_joint() -> None:
    length, origin = 220, 150
    speed = 12. + .02 * np.arange(length)
    segment = {"follower_v": speed, "leader_v": speed + 1., "gap": np.full(length, 20.)}
    posterior = {"population_log_mean": np.tile(np.log([30., 2., 1.4, 1.4, 1.8, .3, 1.5]), (3, 1)),
                 "population_log_cov": np.tile(np.eye(7)[None] * .01, (3, 1, 1))}
    draws, weight = condition_driver_candidates(posterior, segment, origin, np.random.default_rng(9), candidates=16)
    assert draws.shape == (16, 7)
    assert np.isclose(np.sum(weight), 1.) and np.all(weight >= 0.)

from __future__ import annotations

import numpy as np

from bayesian_ma_idm.src.model import sample_driver_joint


def test_retained_pymc_driver_draw_is_not_clipped_or_population_resampled() -> None:
    """Fig. 10 mode must select an actual draw/driver parameter vector."""
    retained = np.array([[[91., 2., 1.5, .7, 2.1], [92., 3., 1.2, .8, 2.2]]])
    posterior = {
        "driver_parameter_draws": retained,
        "driver_sigma_draws": np.array([.3]),
        "driver_ell_draws": np.array([1.4]),
    }
    sampled = sample_driver_joint(posterior, np.random.default_rng(3))
    assert sampled[0] in {91., 92.}  # notably above empirical-Bayes v0 upper bound 60
    assert sampled[5] == .3 and sampled[6] == 1.4

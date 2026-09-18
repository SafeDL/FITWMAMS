from __future__ import annotations

import numpy as np

from bayesian_ma_idm.src.reference_kernels import conditional_gp


def test_source_zero_context_has_the_published_2ell_time_support() -> None:
    """The source protocol's zero context is finite and uses native SI time."""
    dt, ell = .04, 1.32
    points = int(2 * ell / dt)
    time = np.linspace(-points * dt, 0., points)
    mean, variance = conditional_gp(time, np.zeros(points), .2, .19, ell, 1.e-8)
    assert np.isclose(time[0], -2 * ell, atol=dt)
    assert np.isfinite(mean) and variance >= 0.

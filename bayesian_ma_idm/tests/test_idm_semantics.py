from __future__ import annotations

import numpy as np

from bayesian_ma_idm.src.reference_kernels import idm


def test_donor_ring_gap_clip_is_explicit_and_matches_its_definition() -> None:
    theta = np.array([33.3, 2.0, 1.6, 1.5, 1.67])
    # A much faster leader makes the dynamic part negative.  The paper equation
    # retains it; the author's ring-script variant clips it to zero.
    equation = idm(20., 10., -20., theta)
    donor = idm(20., 10., -20., theta, donor_ring_clip=True)
    # The un-clipped negative desired gap is squared by IDM and therefore
    # produces a spurious, much larger braking magnitude.
    assert donor > equation
    expected = theta[3] * (1. - (10. / theta[0]) ** 4 - (theta[1] / 20.) ** 2)
    assert np.isclose(donor, expected)

from __future__ import annotations

import numpy as np

from tools.evt import threshold_stability


def test_threshold_scan_keeps_unique_thresholds_and_corrected_scale() -> None:
    values = np.repeat(np.linspace(0.0, 1.0, 30), 4)
    rows = threshold_stability(
        values,
        min_exceedances=8,
        max_tail_fraction=0.8,
        max_threshold_candidates=60,
    )
    thresholds = [row["u"] for row in rows]
    assert len(thresholds) == len(set(thresholds))
    for row in rows:
        assert np.isclose(row["modified_scale"], row["beta"] - row["xi"] * row["u"])

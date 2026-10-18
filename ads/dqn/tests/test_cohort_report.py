from __future__ import annotations

import numpy as np
import pandas as pd

from ads.dqn.cohort_report import bootstrap_summary


def test_bootstrap_summary_reports_event_level_intervals() -> None:
    frame = pd.DataFrame(
        {
            "failure": [False, True, False, True],
            "progress_m": [10.0, 2.0, 8.0, 4.0],
            "reward": [1.0, -1.0, 0.5, -0.5],
            "minimum_distance_m": [5.0, 1.0, 4.0, 2.0],
            "collision": [False, True, False, False],
            "offroad": [False, False, False, True],
            "stalled": [False] * 4,
            "mean_abs_acceleration_mps2": [1.0, 2.0, 1.0, 2.0],
            "action_switches": [0, 1, 2, 1],
        }
    )
    result = bootstrap_summary(frame, seed=3, replicates=200)
    assert result["failure_rate"] == 0.5
    assert result["mean_progress_m"] == 6.0
    assert len(result["failure_rate_ci95"]) == 2
    assert np.isfinite(result["mean_minimum_distance_m_ci95"]).all()

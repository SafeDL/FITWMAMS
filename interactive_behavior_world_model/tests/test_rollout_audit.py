import numpy as np
import pandas as pd
import pytest

from interactive_behavior_world_model.scripts.evaluate_action import validate_action_metrics
from interactive_behavior_world_model.scripts.evaluate_events import validate_event_metrics


def test_t1_audit_accepts_and_rejects_bad_completion_contracts():
    frame = pd.DataFrame(
        {
            "scenario_id": ["a", "b"],
            "requested_futures": [16, 16],
            "completed_futures": [16, 16],
            "fair_energy_score": [0.1, 0.2],
            "failure_type": ["", ""],
        }
    )
    validate_action_metrics(frame, expected_scenarios=2)
    broken = frame.copy()
    broken.loc[1, "fair_energy_score"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_action_metrics(broken)


def test_t2a_audit_rejects_duplicate_or_incomplete_events():
    frame = pd.DataFrame(
        {
            "event_id": ["a", "b"],
            "requested_futures": [16, 16],
            "completed_futures": [16, 16],
            "fair_energy_score": [0.1, 0.2],
            "failure_type": ["", ""],
        }
    )
    validate_event_metrics(frame, expected_events=2)
    duplicate = pd.concat((frame, frame.iloc[[0]]), ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_event_metrics(duplicate)
    incomplete = frame.copy()
    incomplete.loc[1, "completed_futures"] = 15
    with pytest.raises(ValueError, match="incomplete"):
        validate_event_metrics(incomplete)

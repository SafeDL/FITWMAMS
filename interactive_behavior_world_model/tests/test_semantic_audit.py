import numpy as np
import pandas as pd
import pytest

from interactive_behavior_world_model.scripts.evaluate_events_semantic import (
    validate_semantic_event_metrics,
)
from interactive_behavior_world_model.scripts.evaluate_semantic import validate_semantic_scene_metrics


def test_semantic_t1_audit_accepts_a_complete_scene_artifact():
    frame = pd.DataFrame(
        {
            "scenario_id": ["a", "b"],
            "requested_futures": [4, 4],
            "completed_futures": [4, 4],
            "fair_energy_score": [0.2, 0.3],
            "failure_type": ["", ""],
        }
    )
    validate_semantic_scene_metrics(frame, expected_scenarios=2)


def test_semantic_t2a_audit_rejects_nonfinite_or_duplicate_events():
    frame = pd.DataFrame(
        {
            "event_id": ["a", "b"],
            "requested_futures": [4, 4],
            "completed_futures": [4, 4],
            "fair_energy_score": [0.2, 0.3],
            "failure_type": ["", ""],
        }
    )
    validate_semantic_event_metrics(frame, expected_events=2)
    duplicate = pd.concat((frame, frame.iloc[[0]]), ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_semantic_event_metrics(duplicate)
    nonfinite = frame.copy()
    nonfinite.loc[0, "fair_energy_score"] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_semantic_event_metrics(nonfinite)

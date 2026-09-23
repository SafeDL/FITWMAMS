import numpy as np
import pandas as pd
import pytest

from interactive_behavior_world_model.scripts.evaluate_pnc import validate_pnc_metrics


def _frame():
    return pd.DataFrame(
        {
            "scenario_id": ["a", "b"],
            "requested_futures": [4, 4],
            "completed_futures": [4, 4],
            "ego_collision_probability": [0.0, 0.25],
            "failure_type": ["", ""],
        }
    )


def test_t3_audit_accepts_complete_finite_unique_scene_metrics():
    validate_pnc_metrics(_frame(), event_cohort=False, expected_scenarios=2)


def test_t3_audit_rejects_nonfinite_and_wrong_identity_key():
    nonfinite = _frame()
    nonfinite.loc[0, "ego_collision_probability"] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_pnc_metrics(nonfinite, event_cohort=False)
    with pytest.raises(ValueError, match="event_id"):
        validate_pnc_metrics(_frame(), event_cohort=True)

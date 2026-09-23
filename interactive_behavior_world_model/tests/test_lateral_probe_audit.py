import numpy as np
import pandas as pd
import pytest

from interactive_behavior_world_model.scripts.evaluate_lateral_probes_action import (
    validate_lateral_probe_metrics,
)


def _frame():
    return pd.DataFrame(
        {
            "probe_id": ["a", "b"],
            "controller_rate_rps": [0.6, 0.9],
            "requested_pairs": [8, 8],
            "completed_pairs": [8, 8],
            "response_probability": [0.25, 0.5],
            "failure_type": ["", ""],
        }
    )


def test_lateral_probe_audit_accepts_complete_finite_unique_metrics():
    validate_lateral_probe_metrics(_frame(), expected_conditions=2)


def test_lateral_probe_audit_rejects_duplicate_or_nonfinite_metrics():
    duplicated = pd.concat((_frame(), _frame().iloc[[0]]), ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_lateral_probe_metrics(duplicated)
    nonfinite = _frame()
    nonfinite.loc[1, "response_probability"] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_lateral_probe_metrics(nonfinite)

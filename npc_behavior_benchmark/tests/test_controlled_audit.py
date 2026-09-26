import pandas as pd
import pytest

from npc_behavior_benchmark.scripts.evaluate_controlled_action import (
    validate_controlled_metrics,
)


def _frame():
    return pd.DataFrame(
        {
            "event_id": ["a", "b"],
            "requested_futures": [8, 8],
            "completed_futures": [8, 8],
            "completion_probability": [0.25, 0.5],
            "failure_type": ["", ""],
        }
    )


def test_t4a_audit_accepts_complete_finite_unique_events():
    validate_controlled_metrics(_frame(), expected_events=2)


def test_t4a_audit_rejects_incomplete_or_duplicate_events():
    incomplete = _frame()
    incomplete.loc[0, "completed_futures"] = 7
    with pytest.raises(ValueError, match="incomplete"):
        validate_controlled_metrics(incomplete)
    duplicate = pd.concat((_frame(), _frame().iloc[[0]]), ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_controlled_metrics(duplicate)

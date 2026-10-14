from __future__ import annotations

import pandas as pd

from hierarchical_world_model.scripts.collect_response_gate_data import (
    select_event_indices,
)


def test_response_gate_event_selection_is_stable_and_recording_balanced() -> None:
    frame = pd.DataFrame(
        {
            "event_id": [f"e{index}" for index in range(12)],
            "source_event_id": [f"s{index}" for index in range(12)],
            "recording_id": [1] * 4 + [2] * 4 + [3] * 4,
            "interaction_group": [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    first = select_event_indices(frame, max_events=3, seed=17)
    second = select_event_indices(frame, max_events=3, seed=17)
    assert first == second
    assert len(first) == 3
    assert frame.iloc[first]["recording_id"].nunique() == 3
    assert frame.iloc[first]["interaction_group"].is_unique
    shuffled = frame.sample(frac=1.0, random_state=41).reset_index(drop=True)
    shuffled_selected = select_event_indices(shuffled, max_events=3, seed=17)
    assert set(frame.iloc[first]["event_id"]) == set(
        shuffled.iloc[shuffled_selected]["event_id"]
    )


def test_response_gate_event_selection_rejects_empty_request() -> None:
    frame = pd.DataFrame({"event_id": ["e0"], "source_event_id": ["s0"]})
    try:
        select_event_indices(frame, max_events=0, seed=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("non-positive max-events should fail")

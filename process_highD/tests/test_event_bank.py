from __future__ import annotations

import pandas as pd

from process_highD.src.event_bank import ACTOR_COLUMNS, build_testing_event_bank


def _segments() -> pd.DataFrame:
    rows = []
    for index in range(12):
        row = {
            "segment_id": f"e{index}",
            "recording_id": 1 + index // 6,
            "ego_id": 10 + index,
            "window_start_frame": index * 100,
            "window_end_frame": index * 100 + 149,
            "event_risk": index / 11.0,
            "num_untracked_candidates": 0,
            "max_untracked_pair_risk": 0.0,
            "untracked_risk_exceeds_tracked_peak": False,
        }
        row.update({name: -1 for name in ACTOR_COLUMNS})
        rows.append(row)
    # e0/e1 overlap and share two of three actors, so they form one group.
    rows[0]["same_front_id"] = 99
    rows[1]["same_rear_id"] = 99
    rows[0]["left_front_id"] = 98
    rows[1]["left_rear_id"] = 98
    rows[1]["window_start_frame"] = 50
    return pd.DataFrame(rows)


def test_event_bank_has_stable_identity_groups_and_equal_role_budgets() -> None:
    frame = _segments()
    splits = {f"e{i}": "train" if i < 6 else "test" for i in range(12)}
    result = build_testing_event_bank(
        frame,
        evt_threshold=0.5,
        exceedance_rate=1.0 / 3.0,
        split_by_event=splits,
        seed=7,
    )
    events = result.events.set_index("event_id")
    assert events.loc["e0", "interaction_group"] == events.loc["e1", "interaction_group"]
    assert (events["source_event_id"] == events.index).all()
    assert events.groupby("split")["empirical_selected"].sum().to_dict() == {
        "test": 2,
        "train": 2,
    }
    assert result.summary["events"] == 12


def test_event_bank_rejects_missing_split_assignment() -> None:
    try:
        build_testing_event_bank(
            _segments(), evt_threshold=0.5, exceedance_rate=0.2,
            split_by_event={"e0": "train"},
        )
    except ValueError as error:
        assert "split" in str(error)
    else:
        raise AssertionError("missing split assignment was accepted")


def test_interaction_groups_do_not_merge_through_a_long_chain() -> None:
    frame = _segments().iloc[:3].copy()
    frame["recording_id"] = 1
    frame["window_start_frame"] = [0, 100, 200]
    frame["window_end_frame"] = [149, 249, 349]
    for name in ACTOR_COLUMNS:
        frame[name] = -1
    frame.loc[0, ["same_front_id", "left_front_id"]] = [90, 91]
    frame.loc[1, ["same_front_id", "left_front_id"]] = [90, 91]
    frame.loc[2, ["same_front_id", "left_front_id"]] = [90, 91]
    result = build_testing_event_bank(
        frame,
        evt_threshold=0.5,
        exceedance_rate=1.0 / 3.0,
        split_by_event={"e0": "train", "e1": "train", "e2": "train"},
        seed=7,
    )
    groups = result.events.set_index("event_id")["interaction_group"]
    assert groups["e0"] == groups["e1"]
    assert groups["e2"] != groups["e0"]

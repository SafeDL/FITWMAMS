from __future__ import annotations

import numpy as np

from hierarchical_world_model.src.collision_attribution import (
    collision_events,
    paired_collision_summary,
)


def _approach() -> tuple[np.ndarray, np.ndarray]:
    states = np.zeros((1, 3, 7, 6), np.float32)
    states[0, :, 0, 0] = [0.0, 5.0, 8.0]
    states[0, :, 0, 2] = 20.0
    states[0, :, 1, 0] = 11.0
    states[0, :, 1, 2] = 10.0
    active = np.zeros((1, 6), bool)
    active[0, 0] = True
    return states, active


def test_accelerating_ads_rear_approach_is_separated_from_npc_overlap() -> None:
    states, active = _approach()
    events = collision_events(states, active, intervention_start_frame=1, ads_acceleration_setpoint_mps2=4)
    assert events["raw"].tolist() == [True]
    assert events["ads_pursuit"].tolist() == [True]
    assert events["adjusted"].tolist() == [False]


def test_npc_cut_in_is_not_attributed_to_ads_pursuit() -> None:
    states, active = _approach()
    states[0, 0, 1, 1] = 3.6
    states[0, 1:, 1, 1] = 0.0
    events = collision_events(states, active, intervention_start_frame=1, ads_acceleration_setpoint_mps2=4)
    assert events["ads_pursuit"].tolist() == [False]
    assert events["adjusted"].tolist() == [True]


def test_ads_lane_change_exposure_is_labelled_but_not_excluded() -> None:
    states = np.zeros((1, 4, 7, 6), np.float32)
    states[0, :, 0, 0] = 0.0
    states[0, :, 0, 1] = [0.0, 0.0, 1.0, 2.0]
    states[0, :, 1, 0] = 1.0
    states[0, :, 1, 1] = 3.6
    active = np.zeros((1, 6), bool)
    active[0, 0] = True
    events = collision_events(states, active, intervention_start_frame=1)
    assert events["ads_cutin_exposure"].tolist() == [True]
    assert events["raw"].tolist() == [True]
    assert events["adjusted"].tolist() == [True]


def test_ads_cutin_exposure_accepts_nonstandard_adjacent_lane_width() -> None:
    states = np.zeros((1, 4, 7, 6), np.float32)
    states[0, :, 0, 1] = [0.0, 0.0, 2.5, 4.0]
    states[0, :, 1, 1] = 5.0
    states[0, :, 1, 0] = 1.0
    active = np.zeros((1, 6), bool)
    active[0, 0] = True
    events = collision_events(states, active, intervention_start_frame=1)
    assert events["ads_cutin_exposure"].tolist() == [True]
    assert events["raw"].tolist() == [True]


def test_other_pair_collision_remains_in_adjusted_scene_rate() -> None:
    states, active = _approach()
    states[0, :, 2, 0] = [30.0, 30.0, 30.0]
    states[0, :, 3, 0] = [40.0, 32.0, 32.0]
    active[0, 1:3] = True
    events = collision_events(states, active, intervention_start_frame=1, ads_acceleration_setpoint_mps2=4)
    assert events["ads_pursuit"].tolist() == [True]
    assert events["npc_npc"].tolist() == [True]
    assert events["adjusted"].tolist() == [True]


def test_paired_summary_counts_response_avoided_overlap() -> None:
    states, active = _approach()
    passive = collision_events(states, active, intervention_start_frame=1)
    responsive_states = states.copy()
    responsive_states[0, :, 1, 0] = 30.0
    responsive = collision_events(responsive_states, active, intervention_start_frame=1)
    baseline_states = states.copy()
    baseline_states[0, :, 0, 0] = 0.0
    baseline = collision_events(baseline_states, active, intervention_start_frame=1)
    summary = paired_collision_summary(passive, responsive, logged_baseline=baseline)
    assert summary["raw_avoided_with_npc_response"] == 1
    assert summary["raw_introduced_with_npc_response"] == 0
    assert summary["raw_ads_intervention_added_pair_scene_count"] == 1


def test_pair_audit_detects_new_npc_collision_inside_already_colliding_scene() -> None:
    states, active = _approach()
    states[0, :, 2, 0] = 30.0
    states[0, :, 3, 0] = 40.0
    active[0, 1:3] = True
    responsive_states = states.copy()
    responsive_states[0, 2, 3, 0] = 32.0
    passive = collision_events(states, active, intervention_start_frame=1)
    responsive = collision_events(responsive_states, active, intervention_start_frame=1)
    summary = paired_collision_summary(passive, responsive)
    assert summary["raw_introduced_with_npc_response"] == 0
    assert summary["raw_responsive_overlap_npc_response_added_pair_scene_count"] == 1

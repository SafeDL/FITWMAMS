import numpy as np
from types import SimpleNamespace

from counterfactual_response.experiment import (
    ROOT, Scenario, frozen_action_branch, leader_trajectory, load_highd_anchor,
    select_common_support_anchor_row,
)


def test_counterfactual_leader_brakes_only_in_declared_window():
    scenario = Scenario()
    anchor = load_highd_anchor(scenario)
    factual = leader_trajectory(scenario, 0.0, anchor)
    counterfactual = leader_trajectory(scenario, 3.0, anchor)
    active = (np.arange(scenario.frames) * scenario.dt_s >= scenario.brake_start_s) & (
        np.arange(scenario.frames) * scenario.dt_s < scenario.brake_start_s + scenario.brake_duration_s
    )
    difference = counterfactual["a"] - factual["a"]
    assert np.allclose(difference[active], -3.0)
    assert np.allclose(difference[~active], 0.0)
    assert np.allclose(counterfactual["x"][:26], factual["x"][:26])
    assert np.allclose(counterfactual["v"][:26], factual["v"][:26])
    assert counterfactual["v"][-1] < factual["v"][-1]


def test_frozen_baseline_uses_same_braking_leader_and_preserves_actions():
    scenario = Scenario()
    anchor = load_highd_anchor(scenario)
    leader = leader_trajectory(scenario, 3.0, anchor)
    action = np.linspace(0.1, -0.2, scenario.frames)
    natural = {
        "acceleration": action,
        "requested_acceleration": action,
        "residual": np.zeros(scenario.frames),
        "regime": np.full(scenario.frames, -1),
    }
    frozen = frozen_action_branch(natural, leader, scenario, SimpleNamespace(anchor=anchor))
    assert np.array_equal(frozen["acceleration"], action)
    expected_gap = leader["x"] - frozen["position"] - scenario.vehicle_length_m
    assert np.allclose(frozen["gap"], expected_gap)
    expected_closing = frozen["speed"] - leader["v"]
    assert np.allclose(frozen["closing"], expected_closing)


def test_highd_anchor_matches_cih_response_event():
    evidence = ROOT / "results/hierarchical_world_model/cih_wm/evidence/reaction_events/validation/reaction_events.npz"
    assert select_common_support_anchor_row(evidence) == 55398
    anchor = load_highd_anchor(Scenario())
    assert anchor.row == 55398
    assert anchor.recording_id == 27
    assert anchor.leader_id == 2560
    assert anchor.follower_id == 2568
    assert np.isclose(anchor.logged_gap[0], 22.64000244140625)


def test_native_and_decision_clocks_are_exactly_commensurate():
    scenario = Scenario()
    assert scenario.hold_frames == 5
    assert scenario.frames == 150
    assert scenario.decisions == 30

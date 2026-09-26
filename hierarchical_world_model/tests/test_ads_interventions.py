from __future__ import annotations

import numpy as np
import pytest
import torch

from hierarchical_world_model.src.ads_interventions import (
    AbsoluteAccelerationPolicy,
    OnlineAccelerationWindowPolicy,
    OnlineSemanticLaneChangePolicy,
    SemanticLaneChangePolicy,
)
from hierarchical_world_model.scripts.evaluate_online_ads import _screen
from hierarchical_world_model.scripts.ads_test_policies import MaintainEntrySpeedADS
from traffic_components.src.core.dynamics import KinematicTrafficDynamics


def _initial(speed_mps: float = 25.0) -> np.ndarray:
    states = np.zeros((1, 7, 6), np.float32)
    states[0, 0, 2] = speed_mps
    return states


def _context(states: torch.Tensor, frame: int) -> dict[str, torch.Tensor | int]:
    return {
        "agent_states": states,
        "agent_valid": torch.ones(1, 7, dtype=torch.bool),
        "reference_index": frame,
    }


def test_absolute_acceleration_policy_uses_physical_setpoint_not_delta() -> None:
    baseline = np.zeros((1, 149, 2), np.float32)
    baseline[..., 0] = 1.25
    policy = AbsoluteAccelerationPolicy(baseline, acceleration_mps2=-8.0)
    states = torch.from_numpy(_initial())
    assert policy(_context(states, 24))[0, 0].item() == pytest.approx(1.25)
    assert policy(_context(states, 25))[0, 0].item() == pytest.approx(-8.0)
    assert policy(_context(states, 49))[0, 0].item() == pytest.approx(-8.0)
    assert policy(_context(states, 50))[0, 0].item() == pytest.approx(1.25)


def test_absolute_acceleration_policy_rejects_out_of_range_setpoint() -> None:
    with pytest.raises(ValueError, match="outside physical bounds"):
        AbsoluteAccelerationPolicy(
            np.zeros((1, 149, 2), np.float32), acceleration_mps2=-8.01
        )


def test_lane_change_ads_holds_speed_at_maneuver_entry() -> None:
    policy = MaintainEntrySpeedADS(start_frame=25)
    states = torch.from_numpy(_initial(10.0))
    states[0, 0, 4] = -1.0
    assert policy(_context(states, 24))[0, 0].item() == pytest.approx(-1.0)
    assert policy(_context(states, 25))[0, 0].item() == pytest.approx(0.0)
    states[0, 0, 2] = 9.0
    assert policy(_context(states, 26))[0, 0].item() == pytest.approx(1.5)


@pytest.mark.parametrize("speed_mps", [5.0, 10.0, 25.0, 40.0])
def test_semantic_lane_change_reaches_target_and_settles(speed_mps: float) -> None:
    baseline = np.zeros((1, 149, 2), np.float32)
    initial = _initial(speed_mps)
    policy = SemanticLaneChangePolicy(baseline, initial, direction="left")
    states = torch.from_numpy(initial)
    valid = torch.zeros(1, 7, dtype=torch.bool)
    valid[:, 0] = True
    dynamics = KinematicTrafficDynamics()
    generated = []
    actions = []
    for frame in range(149):
        action = policy(_context(states, frame))
        controls = torch.zeros(1, 7, 2)
        controls[:, 0] = action
        states = dynamics.step(states, controls, valid, 0.04)
        generated.append(states.numpy())
        actions.append(action.numpy())
    rollout = np.stack(generated, axis=1)
    controls = np.stack(actions, axis=1)
    diagnostics = policy.diagnostics(rollout)
    assert diagnostics["completed"][0]
    assert abs(diagnostics["final_lateral_error_m"][0]) <= 0.15
    assert abs(diagnostics["final_heading_error_rad"][0]) <= 0.02
    assert diagnostics["maximum_target_overshoot_m"][0] < 0.15
    assert np.abs(controls[..., 1]).max() <= 0.35 + 1.0e-6


def test_semantic_lane_change_has_no_lateral_command_before_start() -> None:
    baseline = np.zeros((1, 149, 2), np.float32)
    policy = SemanticLaneChangePolicy(baseline, _initial(), start_frame=25)
    action = policy(_context(torch.from_numpy(_initial()), 24))
    assert action[0, 1].item() == 0.0


def test_semantic_lane_change_targets_actual_mapped_centre() -> None:
    baseline = np.zeros((1, 149, 2), np.float32)
    initial = _initial()
    initial[0, 0, 1] = 0.2
    lanes = np.zeros((1, 2, 8, 6), np.float32)
    lanes[:, 0, :, 1] = 1.1
    lanes[:, 1, :, 1] = 5.0
    lane_valid = np.ones((1, 2, 8), bool)
    policy = SemanticLaneChangePolicy(
        baseline, initial, map_polylines=lanes,
        map_polyline_valid=lane_valid, direction="left",
    )
    assert policy.source_y[0] == pytest.approx(1.1)
    assert policy.target_y[0] == pytest.approx(5.0)


def test_left_rear_screen_uses_mapped_target_lane() -> None:
    states = np.zeros((1, 25, 7, 6), np.float32)
    valid = np.zeros((1, 25, 7), bool)
    valid[0, 24, :3] = True
    states[0, 24, 1, :2] = (-8.0, 3.0)
    states[0, 24, 2, :2] = (-15.0, 5.0)
    lanes = np.zeros((1, 2, 8, 6), np.float32)
    lanes[:, 1, :, 1] = 5.0
    lane_valid = np.ones((1, 2, 8), bool)
    selected, receiver = _screen(states, valid, "left", lanes, lane_valid)
    assert selected.tolist() == [True]
    assert receiver.tolist() == [1]


def test_right_rear_screen_uses_mapped_target_lane() -> None:
    states = np.zeros((1, 174, 7, 6), np.float32)
    valid = np.ones((1, 174, 7), bool)
    states[0, 24, 1, :2] = (-8.0, -3.8)
    states[0, 24, 2, :2] = (-15.0, 3.9)
    lanes = np.zeros((1, 3, 8, 6), np.float32)
    lanes[:, 0, :, 1] = -3.8
    lanes[:, 1, :, 1] = 0.0
    lanes[:, 2, :, 1] = 3.9
    lane_valid = np.ones((1, 3, 8), bool)
    selected_right, receiver_right = _screen(
        states, valid, "right", lanes, lane_valid
    )
    selected_left, receiver_left = _screen(
        states, valid, "left", lanes, lane_valid
    )
    assert selected_right.tolist() == [True]
    assert receiver_right.tolist() == [0]
    assert selected_left.tolist() == [True]
    assert receiver_left.tolist() == [1]


def test_online_acceleration_window_uses_current_policy_without_trace() -> None:
    base = lambda context: torch.full((1, 2), 0.5)
    policy = OnlineAccelerationWindowPolicy(base, -8.0, start_frame=2, stop_frame=4)
    state = torch.from_numpy(_initial())
    assert policy(_context(state, 1))[0, 0].item() == 0.5
    assert policy(_context(state, 2))[0, 0].item() == -8.0
    assert policy(_context(state, 4))[0, 0].item() == 0.5


@pytest.mark.parametrize("speed_mps", [5.0, 10.0, 25.0, 40.0])
@pytest.mark.parametrize("direction, sign", [("left", 1.0), ("right", -1.0)])
def test_online_semantic_lane_change_completes_without_nominal_trace(
    speed_mps: float, direction: str, sign: float
) -> None:
    base = lambda context: torch.zeros((1, 2))
    policy = OnlineSemanticLaneChangePolicy(base, direction=direction)
    states = torch.from_numpy(_initial(speed_mps))
    valid = torch.zeros(1, 7, dtype=torch.bool)
    valid[:, 0] = True
    dynamics = KinematicTrafficDynamics()
    for frame in range(149):
        action = policy(_context(states, frame))
        controls = torch.zeros(1, 7, 2)
        controls[:, 0] = action
        states = dynamics.step(states, controls, valid, 0.04)
    assert policy.target_y is not None
    assert abs(states[0, 0, 1].item() - policy.target_y[0].item()) < 0.15
    assert policy.target_y[0].item() == pytest.approx(sign * 3.6)
    assert abs(torch.atan2(states[0, 0, 3], states[0, 0, 2]).item()) < 0.02


def test_online_semantic_right_change_targets_mapped_lane_centre() -> None:
    policy = OnlineSemanticLaneChangePolicy(
        lambda context: torch.zeros((1, 2)), direction="right", start_frame=0
    )
    state = torch.from_numpy(_initial())
    context = _context(state, 0)
    lanes = torch.zeros((1, 3, 8, 6))
    lanes[:, 0, :, 1] = -3.8
    lanes[:, 1, :, 1] = 0.0
    lanes[:, 2, :, 1] = 3.9
    context["map_polylines"] = lanes
    context["map_polyline_valid"] = torch.ones((1, 3, 8), dtype=torch.bool)
    policy(context)
    assert policy.target_y is not None
    assert policy.target_y[0].item() == pytest.approx(-3.8)


def test_online_semantic_lane_change_rejects_unmapped_target_lane() -> None:
    policy = OnlineSemanticLaneChangePolicy(
        lambda context: torch.zeros((1, 2)), direction="left", start_frame=0
    )
    state = torch.from_numpy(_initial())
    context = _context(state, 0)
    context["map_polylines"] = torch.zeros((1, 1, 8, 6))
    context["map_polyline_valid"] = torch.ones((1, 1, 8), dtype=torch.bool)
    with pytest.raises(ValueError, match="no mapped target lane"):
        policy(context)


def test_online_semantic_target_uses_mapped_lane_centre_not_drifted_ego_y() -> None:
    policy = OnlineSemanticLaneChangePolicy(
        lambda context: torch.zeros((1, 2)), direction="left", start_frame=0
    )
    state = torch.from_numpy(_initial())
    state[0, 0, 1] = 0.6
    context = _context(state, 0)
    lanes = torch.zeros((1, 2, 8, 6))
    lanes[:, 1, :, 1] = 3.6
    context["map_polylines"] = lanes
    context["map_polyline_valid"] = torch.ones((1, 2, 8), dtype=torch.bool)
    policy(context)
    assert policy.target_y is not None
    assert policy.target_y[0].item() == pytest.approx(3.6)


def test_online_semantic_target_uses_nonstandard_mapped_lane_width() -> None:
    policy = OnlineSemanticLaneChangePolicy(
        lambda context: torch.zeros((1, 2)), direction="left", start_frame=0
    )
    state = torch.from_numpy(_initial())
    state[0, 0, 1] = 0.2
    context = _context(state, 0)
    lanes = torch.zeros((1, 2, 8, 6))
    lanes[:, 0, :, 1] = 1.1
    lanes[:, 1, :, 1] = 5.0
    context["map_polylines"] = lanes
    context["map_polyline_valid"] = torch.ones((1, 2, 8), dtype=torch.bool)
    policy(context)
    assert policy.target_y is not None
    assert policy.target_y[0].item() == pytest.approx(5.0)

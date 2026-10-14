from __future__ import annotations

import numpy as np
import pytest
import torch

from hierarchical_world_model.src.ads_interventions import (
    OnlineAccelerationWindowPolicy,
    OnlineSemanticLaneChangePolicy,
)
from hierarchical_world_model.src.core.dynamics import KinematicTrafficDynamics


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


def test_online_lane_policy_keeps_batched_target_centers_one_dimensional() -> None:
    states = torch.from_numpy(np.repeat(_initial(), 3, axis=0))
    maps = torch.zeros(3, 3, 8, 6)
    maps[..., 1] = torch.tensor([-3.6, 0.0, 3.6])[None, :, None]
    maps[..., 4] = 3.6
    context = {
        "agent_states": states,
        "reference_index": 25,
        "map_polylines": maps,
        "map_polyline_valid": torch.ones(3, 3, 8, dtype=torch.bool),
    }
    policy = OnlineSemanticLaneChangePolicy(
        lambda context: torch.zeros(3, 2), direction="left"
    )
    action = policy(context)
    assert action.shape == (3, 2)
    assert policy.target_y.shape == (3,)
    assert (action[:, 1] > 0.0).all()


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

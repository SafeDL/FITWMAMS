"""highD vehicles above 50 m/s must not be truncated by either rollout plant."""

import numpy as np
import torch

from hierarchical_world_model.src.core.dynamics import (
    HIGHD_INTEGRATION_SPEED_MAX_MPS,
    DynamicsConfig,
    KinematicTrafficDynamics,
)
from hierarchical_world_model.src.highway import HighwayEnvTraffic


def test_unicycle_plant_preserves_highd_speed_above_50_mps() -> None:
    dynamics = KinematicTrafficDynamics(
        DynamicsConfig(speed_max_mps=HIGHD_INTEGRATION_SPEED_MAX_MPS)
    )
    state = torch.zeros((1, 7, 6), dtype=torch.float32)
    state[0, 1, 2] = 56.0
    control = torch.zeros((1, 7, 2), dtype=torch.float32)
    control[0, 1, 0] = 1.0
    valid = torch.zeros((1, 7), dtype=torch.bool)
    valid[0, 1] = True
    advanced = dynamics.step(state, control, valid, 0.04)
    assert torch.isclose(advanced[0, 1, 2], torch.tensor(56.04), atol=1.0e-4)


def test_highway_plant_matches_high_speed_unicycle_step() -> None:
    states = np.zeros((7, 6), np.float32)
    states[0, :3] = [0.0, 0.0, 56.0]
    states[1, :3] = [50.0, 3.6, 56.0]
    valid = np.zeros(7, bool)
    valid[:2] = True
    world = HighwayEnvTraffic()
    world.reset(states, valid)
    background = np.zeros((6, 2), np.float32)
    background[0, 0] = 1.0
    advanced = world.step(background, ego_action=np.zeros(2, np.float32)).states
    assert abs(float(advanced[1, 2]) - 56.04) < 1.0e-4
    assert abs(float(advanced[0, 2]) - 56.0) < 1.0e-4


def test_highway_idm_ego_is_not_forced_below_50_mps() -> None:
    states = np.zeros((7, 6), np.float32)
    states[0, :3] = [0.0, 0.0, 56.0]
    valid = np.zeros(7, bool)
    valid[0] = True
    world = HighwayEnvTraffic()
    world.reset(states, valid, idm_config={"target_speed": 56.0})
    advanced = world.step(np.zeros((6, 2), np.float32)).states
    assert advanced[0, 2] > 50.0


def test_stop_time_integration_matches_highway_and_never_moves_backwards():
    states = np.zeros((7, 6), np.float32)
    states[0, :3] = [0, 0, 0.1]
    states[1, :3] = [50, 3.6, 0]
    valid = np.zeros(7, bool)
    valid[:2] = True
    controls = np.zeros((7, 2), np.float32)
    controls[:2, 0] = -8
    dynamics = KinematicTrafficDynamics(DynamicsConfig(speed_max_mps=100))
    tensor = dynamics.step(
        torch.from_numpy(states)[None],
        torch.from_numpy(controls)[None],
        torch.from_numpy(valid)[None],
        0.04,
    )[0].numpy()
    world = HighwayEnvTraffic()
    world.reset(states, valid)
    result = world.step(controls[1:], ego_action=controls[0]).states
    np.testing.assert_allclose(result, tensor, atol=1e-6)
    assert abs(result[0, 0] - 0.1**2 / 16) < 1e-7
    assert result[1, 0] == 50
    for _ in range(25):
        result = world.step(controls[1:], ego_action=controls[0]).states
    assert abs(result[0, 0] - 0.1**2 / 16) < 1e-7
    assert result[1, 0] == 50


def test_crashed_vehicle_overrides_cached_acceleration_and_applies_impact():
    states = np.zeros((7, 6), np.float32)
    states[0, :3] = [0, 0, 20]
    states[1, :3] = [50, 3.6, 20]
    valid = np.zeros(7, bool)
    valid[:2] = True
    world = HighwayEnvTraffic()
    world.reset(states, valid)
    vehicle = world.background[0]
    vehicle.crashed = True
    vehicle.impact = np.array([0.25, 0.0])
    background = np.zeros((6, 2), np.float32)
    background[0] = [4, 0.5]
    result = world.step(background, ego_action=np.zeros(2)).states
    assert result[1, 2] < 20
    assert result[1, 3] == 0
    assert vehicle.impact is None
    assert result[1, 0] > 51.0


def test_collision_transition_replays_after_snapshot_restore():
    states = np.zeros((7, 6), np.float32)
    states[0, :3] = [0, 0, 20]
    states[1, :3] = [50, 3.6, 20]
    valid = np.zeros(7, bool)
    valid[:2] = True
    world = HighwayEnvTraffic()
    world.reset(states, valid)
    world.background[0].impact = np.array([0.1, -0.02])
    snapshot = world.snapshot()
    control = np.zeros((6, 2), np.float32)
    first = world.step(control, ego_action=np.zeros(2)).states
    world.restore(snapshot)
    replay = world.step(control, ego_action=np.zeros(2)).states
    np.testing.assert_array_equal(first, replay)

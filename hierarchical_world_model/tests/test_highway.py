from __future__ import annotations

import numpy as np
import torch

from hierarchical_world_model.src.highway import (
    HighwayEnvClosedLoopWorld,
    HighwayEnvTraffic,
    steering_to_yaw_rate,
    yaw_rate_to_steering,
)
from hierarchical_world_model.src.model import DiffusionGuidedHiQR
from hierarchical_world_model.src.randomness import WorldExogenousState
from world_model.src.core.dynamics import KinematicTrafficDynamics


def _initial_world() -> tuple[np.ndarray, np.ndarray]:
    states = np.zeros((7, 6), np.float32)
    states[0, 2] = 25.0
    states[1] = (20.0, 0.0, 15.0, 0.0, 0.0, 0.0)
    states[2] = (-15.0, 0.0, 26.0, 0.0, 0.0, 0.0)
    valid = np.asarray((True, True, True, False, False, False, False))
    return states, valid


def _idm_config() -> dict[str, float | bool]:
    return {
        "target_speed": 40.0,
        "enable_lane_change": False,
        "ACC_MAX": 6.0,
        "COMFORT_ACC_MAX": 3.0,
        "COMFORT_ACC_MIN": -5.0,
        "DISTANCE_WANTED": 10.0,
        "TIME_WANTED": 1.5,
        "DELTA": 4.0,
    }


def test_yaw_rate_and_highway_steering_are_consistent() -> None:
    steering = yaw_rate_to_steering(0.08, 20.0, 4.8, np.pi / 3.0)
    restored = steering_to_yaw_rate(steering, 20.0, 4.8)
    np.testing.assert_allclose(restored, 0.08, rtol=1.0e-6, atol=1.0e-6)


def test_idm_and_background_actions_are_advanced_by_highway_env() -> None:
    states, valid = _initial_world()
    world = HighwayEnvTraffic()
    world.reset(states, valid, idm_config=_idm_config())
    action = world.idm_action()
    assert action[0] < 0.0

    result = world.step(np.zeros((6, 2), np.float32))
    assert result.ego_action[0] < 0.0
    assert result.states[0, 2] < states[0, 2]
    np.testing.assert_allclose(result.states[1, 0], states[1, 0] + 0.6)
    assert not result.collision


def test_hiqr_background_command_is_committed_before_highway_step() -> None:
    states, valid = _initial_world()
    world = HighwayEnvTraffic()
    world.reset(states, valid)
    background = np.zeros((6, 2), np.float32)
    background[0] = (1.5, 0.0)
    result = world.step(background, ego_action=np.zeros(2, np.float32))
    np.testing.assert_allclose(result.background_actions[0, 0], 1.5)
    np.testing.assert_allclose(result.states[1, 2], states[1, 2] + 1.5 * 0.04)


def test_hiqr_controls_use_the_offline_unicycle_plant_on_highway_road() -> None:
    states, valid = _initial_world()
    world = HighwayEnvTraffic()
    world.reset(states, valid)
    current = torch.from_numpy(world.states()[None])
    ego_control = np.asarray((0.4, -0.03), np.float32)
    background = np.zeros((6, 2), np.float32)
    background[0] = (-0.7, 0.12)
    background[1] = (0.2, -0.04)
    controls = torch.from_numpy(
        np.concatenate((ego_control[None], background), axis=0)[None]
    )
    expected = KinematicTrafficDynamics().step(
        current,
        controls,
        torch.from_numpy(valid[None]),
        dt=0.04,
    )[0].numpy()
    actual = world.step(background, ego_action=ego_control).states
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-5)


def test_snapshot_restore_replays_a_highway_env_branch_exactly() -> None:
    states, valid = _initial_world()
    world = HighwayEnvTraffic()
    world.reset(states, valid)
    first_background = np.zeros((6, 2), np.float32)
    first_background[0] = (0.5, 0.03)
    world.step(first_background, ego_action=np.asarray((0.2, -0.02), np.float32))
    snapshot = world.snapshot()

    second_background = np.zeros((6, 2), np.float32)
    second_background[0] = (-0.4, -0.02)
    first = world.step(second_background, ego_action=np.asarray((0.1, 0.01), np.float32))
    world.restore(snapshot)
    replay = world.step(second_background, ego_action=np.asarray((0.1, 0.01), np.float32))
    np.testing.assert_allclose(replay.states, first.states, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(replay.ego_action, first.ego_action, rtol=0.0, atol=1.0e-6)


def test_hiqr_background_action_can_be_executed_after_highway_idm_action() -> None:
    states, valid = _initial_world()
    world = HighwayEnvTraffic()
    world.reset(states, valid, idm_config=_idm_config())
    ego_action = world.idm_action()
    current = torch.from_numpy(world.states()[None])
    present = torch.from_numpy(valid[None])
    history = current[:, None].expand(-1, 25, -1, -1).clone()
    history_valid = present[:, None].expand(-1, 25, -1).clone()
    reference = torch.zeros(1, 25, 6, 2)
    maps = torch.zeros(1, 8, 8, 6)
    maps[..., 0] = torch.linspace(-100.0, 100.0, 8)[None, None]
    maps[..., 1] = torch.arange(-3, 5)[None, :, None] * 3.6
    maps[..., 2] = 1.0
    maps[..., 4] = 3.6
    result = DiffusionGuidedHiQR().eval()(
        history,
        history_valid,
        current,
        present,
        reference,
        current[:, 1:, :2],
        maps,
        torch.ones(1, 8, 8, dtype=torch.bool),
        committed_ego_controls=torch.from_numpy(ego_action[None, None]),
        deterministic=True,
    )
    step = world.step(result.actions[0, 0].detach().numpy())
    assert step.states.shape == (7, 6)
    np.testing.assert_allclose(step.ego_action, ego_action, atol=1.0e-6)


def _maps(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    maps = torch.zeros(batch, 8, 8, 6)
    maps[..., 0] = torch.linspace(-100.0, 100.0, 8)[None, None]
    maps[..., 1] = torch.arange(-3, 5)[None, :, None] * 3.6
    maps[..., 2] = 1.0
    maps[..., 4] = 3.6
    return maps, torch.ones(batch, 8, 8, dtype=torch.bool)


def test_batched_highway_world_replays_the_same_exogenous_branch() -> None:
    states, valid = _initial_world()
    initial = torch.from_numpy(np.stack((states, states.copy())))
    initial[1, 0, 1] = 3.6
    present = torch.from_numpy(np.stack((valid, valid)))
    reference = torch.zeros(2, 149, 6, 2)
    maps, map_valid = _maps(2)
    exogenous = WorldExogenousState.sample(2, seed=17, response_steps=3)
    world = HighwayEnvClosedLoopWorld(DiffusionGuidedHiQR().eval())
    world.reset(initial, present, reference, maps, map_valid, exogenous_state=exogenous)
    first = world.advance_response(torch.zeros(2, 2))
    snapshot = world.snapshot()
    second = world.advance_response(torch.zeros(2, 2))
    world.restore(snapshot)
    replay = world.advance_response(torch.zeros(2, 2))
    torch.testing.assert_close(replay["agent_state_frames"], second["agent_state_frames"])
    assert first["background_actions"].shape == (2, 1, 6, 2)


def test_highway_world_accepts_offline_factual_history_and_controls() -> None:
    states, valid = _initial_world()
    initial = torch.from_numpy(states[None])
    present = torch.from_numpy(valid[None])
    maps, map_valid = _maps(1)
    history = initial[:, None].expand(-1, 25, -1, -1).clone()
    history[:, :, 0, 0] = torch.arange(25, dtype=torch.float32)
    controls = torch.zeros(1, 24, 2)
    controls[:, -1, 0] = -1.0
    world = HighwayEnvClosedLoopWorld(DiffusionGuidedHiQR().eval())
    world.reset(
        initial,
        present,
        torch.zeros(1, 149, 6, 2),
        maps,
        map_valid,
        exogenous_state=WorldExogenousState.sample(1, seed=29, response_steps=2),
        initial_history=history,
        initial_history_valid=present[:, None].expand(-1, 25, -1),
        committed_ego_controls=controls,
        deterministic_response=True,
    )
    assert world.history is not None
    assert world.committed_ego_controls is not None
    torch.testing.assert_close(world.history, history)
    torch.testing.assert_close(world.committed_ego_controls, controls)
    assert world.deterministic_response


def test_highway_world_uses_real_idm_actions_for_hiqr_execution() -> None:
    states, valid = _initial_world()
    initial = torch.from_numpy(states[None])
    present = torch.from_numpy(valid[None])
    maps, map_valid = _maps(1)
    world = HighwayEnvClosedLoopWorld(DiffusionGuidedHiQR().eval(), idm_config=_idm_config())
    world.reset(
        initial,
        present,
        torch.zeros(1, 149, 6, 2),
        maps,
        map_valid,
        exogenous_state=WorldExogenousState.sample(1, seed=23, response_steps=2),
    )
    action = world.idm_actions()
    transition = world.advance_response(action)
    assert float(action[0, 0]) < 0.0
    torch.testing.assert_close(transition["ego_actions"][:, 0], action)

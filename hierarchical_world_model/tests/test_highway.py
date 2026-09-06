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
from hierarchical_world_model.src.influence_graph import (
    CausalInfluenceGraph,
    ROLE_SAME_LANE_FOLLOWER,
    ROLE_SECONDARY_FOLLOWER,
)
from hierarchical_world_model.src.reaction_controller import (
    CalibratedResidualReactionController, RLResidualReactionController,
)
from hierarchical_world_model.src.randomness import WorldExogenousState
from hierarchical_world_model.src.execution import trajectory_event_risk
from hierarchical_world_model.src.rule_models import RuleModelBundle
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


def test_constant_speed_post_step_state_aligns_with_next_logged_frame() -> None:
    """Guard the 25 Hz x[t+1] versus x[t] factual-audit contract."""
    states, valid = _initial_world()
    states[0, 2] = 25.0
    world = HighwayEnvTraffic(dt_s=0.04)
    anchor = world.reset(states, valid)
    post_step = world.step(
        np.zeros((6, 2), np.float32),
        ego_action=np.zeros(2, np.float32),
    ).states
    np.testing.assert_allclose(post_step[0, 0], anchor[0, 0] + 1.0, atol=1.0e-6)
    # Comparing the post-step state with the anchor is precisely the former
    # one-frame audit error; the correct target is the next logged frame.
    assert np.linalg.norm(post_step[0, :2] - anchor[0, :2]) == 1.0


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
    assert first["crashed"].shape == (2, 7)
    assert first["crashed"].dtype == torch.bool


def test_snapshot_restores_autonomous_graph_and_policy_sample() -> None:
    states, valid = _initial_world()
    initial = torch.from_numpy(states[None])
    present = torch.from_numpy(valid[None])
    maps, map_valid = _maps(1)
    rule = RuleModelBundle((32.0, 2.5, 3.0, 2.0, 1.2, 4.0), 0.3, 0.2, 3.0)
    world = HighwayEnvClosedLoopWorld(
        DiffusionGuidedHiQR().eval(),
        controller=CalibratedResidualReactionController(rule),
    )
    world.reset(
        initial, present, torch.zeros(1, 149, 6, 2), maps, map_valid,
        exogenous_state=WorldExogenousState.sample(1, seed=41, response_steps=3),
    )
    world.advance_response(torch.zeros(1, 2))
    snapshot = world.snapshot()
    first = world.advance_response(torch.zeros(1, 2))
    world.restore(snapshot)
    replay = world.advance_response(torch.zeros(1, 2))
    torch.testing.assert_close(replay["background_actions"], first["background_actions"])
    torch.testing.assert_close(replay["influence_authority"], first["influence_authority"])


def test_autonomous_world_has_no_experiment_event_registration_api() -> None:
    world = HighwayEnvClosedLoopWorld(DiffusionGuidedHiQR().eval())
    assert not hasattr(world, "register_executed_ego_intervention")


def test_new_risk_scope_includes_same_rear() -> None:
    states = np.zeros((1, 2, 7, 6), np.float32)
    valid = np.zeros((1, 7), bool)
    valid[:, (0, 2)] = True
    states[:, :, 0, 2] = 20.0
    states[:, :, 2, 0] = -1.0
    states[:, :, 2, 2] = 25.0
    included = trajectory_event_risk(states, valid)
    excluded = trajectory_event_risk(states, valid, excluded_slots=("same_rear",))
    assert included[0] > excluded[0]


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


def test_autonomous_reaction_scope_replays_without_an_event_label() -> None:
    """Scope follows realised geometry and remains replayable without labels."""
    states, valid = _initial_world()
    initial, present = torch.from_numpy(states[None]), torch.from_numpy(valid[None])
    maps, map_valid = _maps(1)
    world = HighwayEnvClosedLoopWorld(
        DiffusionGuidedHiQR().eval(), controller="none", reaction_min_frames=1,
        reaction_max_frames=8, reaction_recovery_frames=2, reaction_safety_ttc_s=.1,
        reaction_release_ttc_s=.1, influence_stable_release_frames=1,
    )
    world.reset(initial, present, torch.zeros(1, 149, 6, 2), maps, map_valid,
        exogenous_state=WorldExogenousState.sample(1, seed=31, response_steps=6), deterministic_response=True)
    world.advance_response(torch.tensor([[-6.0, 0.0]]))
    engaged = world.advance_response(torch.zeros(1, 2))
    assert int(engaged["controller_phase"][0, 1]) == 1
    assert int(engaged["influence_role"][0, 1]) == 1
    snapshot = world.snapshot()
    expected = world.advance_response(torch.zeros(1, 2))
    world.restore(snapshot)
    replay = world.advance_response(torch.zeros(1, 2))
    torch.testing.assert_close(replay["agent_state_frames"], expected["agent_state_frames"])
    torch.testing.assert_close(replay["influence_authority"], expected["influence_authority"])


def test_causal_authority_does_not_expire_while_same_rear_is_still_closing() -> None:
    """A short ego command may not create a short NPC authority window."""
    states, valid = _initial_world()
    initial, present = torch.from_numpy(states[None]), torch.from_numpy(valid[None])
    maps, map_valid = _maps(1)
    world = HighwayEnvClosedLoopWorld(
        DiffusionGuidedHiQR().eval(), controller="none", reaction_min_frames=1,
        # Deliberately tiny legacy normalizer: it is not an authority timeout.
        reaction_max_frames=2, reaction_recovery_frames=1, reaction_safety_ttc_s=2.,
        reaction_release_ttc_s=4.,
    )
    world.reset(initial, present, torch.zeros(1, 149, 6, 2), maps, map_valid,
        exogenous_state=WorldExogenousState.sample(1, seed=37, response_steps=12), deterministic_response=True)
    # Directly fix an already-realized same-lane following state: 10 m gap,
    # 5 m/s closing, hence TTC=2 s. No external intervention label is needed.
    world.states[0, 0, 0], world.states[0, 0, 2] = 20., 20.
    world.states[0, 2, 0], world.states[0, 2, 2] = 10., 25.
    for _ in range(8):
        transition = world.advance_response(torch.zeros(1, 2))
        assert int(transition["controller_phase"][0, 1]) == 1


def test_causal_monitor_latch_reengages_after_a_brief_safe_interval() -> None:
    """A later re-closing relation must not be forgotten after recovery."""
    states, valid = _initial_world()
    initial, present = torch.from_numpy(states[None]), torch.from_numpy(valid[None])
    maps, map_valid = _maps(1)
    world = HighwayEnvClosedLoopWorld(
        DiffusionGuidedHiQR().eval(), controller="none", reaction_min_frames=1,
        reaction_max_frames=2, reaction_recovery_frames=2, reaction_safety_ttc_s=2.,
        reaction_release_ttc_s=4., influence_stable_release_frames=1,
    )
    world.reset(initial, present, torch.zeros(1, 149, 6, 2), maps, map_valid,
        exogenous_state=WorldExogenousState.sample(1, seed=41, response_steps=5), deterministic_response=True)
    # The observed relation starts authority, then a safe state drains the
    # finite correction envelope while preserving the autonomous monitor.
    engaged = world.advance_response(torch.zeros(1, 2))
    assert int(engaged["controller_phase"][0, 1]) == 1
    assert world.history is not None
    world.states[0, 2, 0], world.states[0, 2, 2] = -60., 15.
    world.history[0, -2, 2, 0] = -50.
    recovery = world.advance_response(torch.zeros(1, 2))
    assert int(recovery["controller_phase"][0, 1]) == 2
    assert int(recovery["influence_role"][0, 1]) == 1
    # A later realised same-lane re-closing relation immediately re-engages.
    world.states[0, 0, 0], world.states[0, 0, 2] = 20., 20.
    world.states[0, 2, 0], world.states[0, 2, 2] = 10., 25.
    reengaged = world.advance_response(torch.zeros(1, 2))
    assert int(reengaged["controller_phase"][0, 1]) == 1


def test_pending_ego_command_cannot_change_current_background_response() -> None:
    """The controller gets no access to the action passed to this tick."""
    states, valid = _initial_world()
    initial, present = torch.from_numpy(states[None]), torch.from_numpy(valid[None])
    maps, map_valid = _maps(1)
    model = DiffusionGuidedHiQR().eval()
    worlds = []
    for seed in (41, 41):
        world = HighwayEnvClosedLoopWorld(model, controller="none")
        world.reset(initial, present, torch.zeros(1, 149, 6, 2), maps, map_valid,
            exogenous_state=WorldExogenousState.sample(1, seed=seed, response_steps=2), deterministic_response=True)
        worlds.append(world)
    no_change = worlds[0].advance_response(torch.zeros(1, 2))
    hard_brake = worlds[1].advance_response(torch.tensor([[-8.0, 0.0]]))
    torch.testing.assert_close(no_change["background_actions"], hard_brake["background_actions"])


def test_dynamic_influence_graph_propagates_exactly_one_secondary_hop() -> None:
    current = torch.zeros(1, 7, 6)
    valid = torch.zeros(1, 7, dtype=torch.bool)
    valid[:, :4] = True
    current[0, 0, 2] = 20.0
    # Direct candidate in the ego rear semicircle.
    current[0, 1, 0], current[0, 1, 2] = -41.0, 25.0
    # Its close follower is outside the ego 50 m region and can therefore be
    # reached only through the direct parent's one-hop edge.
    current[0, 2, 0], current[0, 2, 2] = -51.0, 30.0
    # A follower behind the secondary car must not receive a depth-2 edge.
    current[0, 3, 0], current[0, 3, 2] = -61.0, 35.0
    history = current[:, None].expand(-1, 2, -1, -1).clone()
    graph = CausalInfluenceGraph()
    result = graph.update(current, valid, history, None)
    assert bool(result.direct[0, 0])
    assert int(result.role[0, 0]) == ROLE_SAME_LANE_FOLLOWER
    assert bool(result.secondary[0, 1])
    assert int(result.role[0, 1]) == ROLE_SECONDARY_FOLLOWER
    assert int(result.parent[0, 1]) == 1
    assert not bool(result.secondary[0, 2])

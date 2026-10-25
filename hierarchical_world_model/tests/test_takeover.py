"""Physical tracking, irreversible handover and actual-state continuation."""

from dataclasses import replace

import pytest
import torch

from hierarchical_world_model.src.config import WorldModelConfig
from hierarchical_world_model.src.highway import HighwayEnvClosedLoopWorld
from hierarchical_world_model.src.randomness import WorldExogenousState
from hierarchical_world_model.src.takeover import TakeoverConfig, TakeoverController
from hierarchical_world_model.src.tracking import TrackingConfig
from hierarchical_world_model.tests.traffic_context import (
    THETA,
    change_context,
    context,
)


def make_tracking_world(monkeypatch, controller=None, *, ego_distance=30.0):
    original = context()
    original.current[:, 0, 0] = ego_distance
    world = HighwayEnvClosedLoopWorld(
        controller=controller, tracking_config=TrackingConfig()
    )
    world.reset(
        original.current,
        original.current_valid,
        original.planned_path_xy,
        original.map_polylines,
        original.map_polyline_valid,
        vehicle_dimensions=original.vehicle_dimensions,
        exogenous_state=WorldExogenousState.sample(1, seed=37, response_steps=50),
    )
    return world


def test_tracking_runs_without_hiqr_or_state_overwrite(monkeypatch):
    world = make_tracking_world(monkeypatch)
    initial = world.states.clone()
    for _ in range(20):
        result = world.advance_response(torch.zeros(1, 2))
    torch.testing.assert_close(
        result["agent_states"][..., 0], initial[..., 0] + 16, atol=1e-3, rtol=0
    )
    assert result["base_background_actions"].abs().max() < 0.01


def test_takeover_occurs_once_and_never_releases():
    original = context()
    policy = TakeoverController(THETA, takeover_config=TakeoverConfig())
    states = original.current.clone()
    states[:, 0, 0], states[:, 0, 2] = 14, 12
    original = change_context(original, states)
    first = policy(original)
    assert first.takeover_started[0, 0]
    total = first.takeover_started.long()
    for frame in range(1, 8):
        states = original.current.clone()
        states[:, 0, 0] += 1000
        result = policy(replace(change_context(original, states), response_index=frame))
        total += result.takeover_started
        assert result.takeover_owned[0, 0]
    assert total.max() == 1


def test_owned_actions_ignore_old_timetable_and_old_lateral_goal():
    original = context()
    policy = TakeoverController(THETA, takeover_config=TakeoverConfig())
    policy(replace(original, response_index=50))
    snapshot = policy.snapshot_runtime()
    expected = policy(replace(original, response_index=1))
    policy.restore_runtime(snapshot)
    changed = replace(
        original,
        response_index=1,
        base_actions=torch.full_like(original.base_actions, 3),
        planned_path_xy=original.planned_path_xy + 20,
    )
    actual = policy(changed)
    mask = snapshot["owned"][:, None, :, None].expand_as(actual.actions)
    torch.testing.assert_close(
        actual.actions[mask], expected.actions[mask], atol=0, rtol=0
    )


def test_mid_maneuver_is_adopted_without_a_second_switch():
    original = context()
    states = original.current.clone()
    states[:, 1, 1] = 0.8
    states[:, 1, 3] = 0.5
    plan = original.planned_path_xy.clone()
    plan[:, :, 0, 1] = torch.linspace(0.82, 3.6, plan.shape[1])
    actual = replace(
        change_context(original, states),
        planned_current_xy=states[:, 1:, :2],
        planned_path_xy=plan,
    )
    policy = TakeoverController(THETA, takeover_config=TakeoverConfig())
    policy(actual)
    first = policy(replace(actual, response_index=50))
    assert first.takeover_started[0, 0]
    assert policy._autonomous_lane_target_y[0, 0] == pytest.approx(3.6)
    assert policy._autonomous_lane_ever[0, 0]
    second = policy(replace(actual, response_index=1))
    assert not second.takeover_started[0, 0]
    assert policy._autonomous_lane_target_y[0, 0] == pytest.approx(3.6)


def test_world_snapshot_restores_ownership_and_random_policy(monkeypatch):
    policy = TakeoverController(THETA)
    world = make_tracking_world(monkeypatch, policy)
    world.advance_response(torch.tensor([[-2.0, 0.0]]))
    snapshot = world.snapshot()
    first = world.advance_response(torch.tensor([[0.0, 0.0]]))
    world.restore(snapshot)
    second = world.advance_response(torch.tensor([[0.0, 0.0]]))
    torch.testing.assert_close(
        first["agent_states"], second["agent_states"], atol=0, rtol=0
    )
    torch.testing.assert_close(first["takeover_started"], second["takeover_started"])


def test_not_yet_executed_ads_action_is_not_visible(monkeypatch):
    one = make_tracking_world(monkeypatch, TakeoverController(THETA))
    two = make_tracking_world(monkeypatch, TakeoverController(THETA))
    left = one.advance_response(torch.tensor([[-5.0, 0.1]]))
    right = two.advance_response(torch.tensor([[2.0, -0.1]]))
    torch.testing.assert_close(
        left["response_background_actions"],
        right["response_background_actions"],
        atol=0,
        rtol=0,
    )


def test_npc_reacts_after_ads_braking_without_another_handover(monkeypatch):
    braking = make_tracking_world(
        monkeypatch, TakeoverController(THETA), ego_distance=12
    )
    cruising = make_tracking_world(
        monkeypatch, TakeoverController(THETA), ego_distance=12
    )
    counts = torch.zeros((1, 6), dtype=torch.long)
    for _ in range(25):
        left = braking.advance_response(torch.tensor([[-4.0, 0.0]]))
        right = cruising.advance_response(torch.zeros(1, 2))
        counts += left["takeover_started"]
    assert left["agent_states"][0, 1, 2] < right["agent_states"][0, 1, 2] - 0.1
    assert counts[0, 0] == 1


def test_admission_is_not_revisited_after_every_present_actor_is_owned(monkeypatch):
    import hierarchical_world_model.src.takeover as module

    original = context()
    valid = original.current_valid.clone()
    valid[:, 3:] = False
    original = replace(original, current_valid=valid)
    policy = TakeoverController(THETA, takeover_config=TakeoverConfig())
    policy(replace(original, response_index=50))
    policy(replace(original, response_index=1))
    monkeypatch.setattr(
        module,
        "conflict_scores",
        lambda *args: (_ for _ in ()).throw(AssertionError("admission was repeated")),
    )
    assert not policy(replace(original, response_index=1)).takeover_started.any()


def test_reference_end_hands_over_once_and_continues_without_hiqr():
    original = context()
    policy = TakeoverController(
        THETA, takeover_config=TakeoverConfig(support_threshold=1)
    )
    world = HighwayEnvClosedLoopWorld(
        cfg=WorldModelConfig(),
        tracking_config=TrackingConfig(),
        controller=policy,
    )
    world.reset(
        original.current,
        original.current_valid,
        original.planned_path_xy[:, :3],
        original.map_polylines,
        original.map_polyline_valid,
        exogenous_state=WorldExogenousState.sample(1, seed=83, response_steps=12),
        vehicle_dimensions=original.vehicle_dimensions,
    )
    counts = torch.zeros((1, 6), dtype=torch.long)
    for _ in range(10):
        result = world.advance_response(torch.zeros(1, 2))
        counts += result["takeover_started"]
    assert counts.eq(1).all()
    assert result["takeover_owned"].all()


def test_default_ads_factory_uses_shared_population_without_hiqr(monkeypatch):
    from types import SimpleNamespace

    import ads.common.world as runtime
    from hierarchical_world_model.src.stochastic_drivers.idm_population import (
        sample_theta,
    )

    factory = runtime.FrozenWorldFactory()
    randomness = WorldExogenousState.sample(1, seed=19)
    world = factory(SimpleNamespace(exogenous_state=randomness))
    assert not hasattr(world, "model")
    assert world.controller.driver_correction is not None
    torch.testing.assert_close(
        world.controller.theta,
        torch.as_tensor(sample_theta(factory.population, randomness.driver_latent)),
    )


def test_nearby_front_vehicle_keeps_its_plan_despite_incompatible_driver_draw():
    original = context()
    states = original.current.clone()
    states[:, 0, 0] = -20
    # The ego is behind, so the front vehicle has no reason to react to it.
    actual = change_context(original, states)
    theta = THETA.clone()
    theta[:, 0] = 8
    controller = TakeoverController(theta, takeover_config=TakeoverConfig())
    result = controller(actual)
    assert not result.takeover_owned[0, 0]
    torch.testing.assert_close(result.actions[:, :, 0], actual.base_actions[:, :, 0])

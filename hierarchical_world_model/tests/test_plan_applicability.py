"""Plan forecasts, empirical conflicts and physical interaction contracts."""

from dataclasses import replace
import numpy as np
import torch
from highway_env.utils import are_polygons_intersecting
from hierarchical_world_model.src.error_trajectories import ErrorTrajectoryBank
from hierarchical_world_model.src.plan_applicability import conflict_scores
from hierarchical_world_model.src.motion import motion_forecast
from hierarchical_world_model.src.scene_geometry import (
    rectangle_separation,
    paired_rectangle_separation,
)
from hierarchical_world_model.src.stochastic_drivers.driver_policy import (
    anticipated_entry_features,
    physical_leader_observation,
)
from hierarchical_world_model.src.takeover import TakeoverConfig, TakeoverController
from hierarchical_world_model.tests.traffic_context import (
    THETA,
    context,
    change_context,
)


def scores(c, owned=None, horizon=2):
    if owned is None:
        owned = torch.zeros(1, 6, dtype=torch.bool)
    return conflict_scores(
        c, owned, ErrorTrajectoryBank.nominal(int(horizon / 0.04)), horizon_s=horizon
    )


def closing_scene():
    c = context()
    states = c.current.clone()
    states[:, 0, 0] = 14
    states[:, 0, 2] = 12
    valid = c.current_valid.clone()
    valid[:, 2:] = False
    return replace(change_context(c, states), current_valid=valid)


def test_oriented_sat_agrees_with_independent_highway_polygon_collision():
    torch.manual_seed(3)
    position = torch.randn(100, 1, 2, 2) * 3
    angle = torch.rand(100, 1, 2) * 6
    dimensions = torch.rand(100, 2, 2) * 4 + 1
    gap = rectangle_separation(position, angle, dimensions, torch.zeros_like(position))
    for k in range(100):
        polygons = []
        for j in range(2):
            a = angle[k, 0, j].item()
            rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
            polygons.append(
                (corners * dimensions[k, j].numpy() / 2) @ rotation.T
                + position[k, 0, j].numpy()
            )
        expected = are_polygons_intersecting(*polygons, np.zeros(2), np.zeros(2))[0]
        assert (gap[k, 0, 0, 1] <= 0).item() == expected


def test_equal_speed_close_follower_does_not_trigger_on_distance():
    c = closing_scene()
    states = c.current.clone()
    states[:, 0, 2] = 20
    assert scores(change_context(c, states))[0].max() == 0


def test_future_conflict_triggers_before_physical_contact_and_only_once():
    c = closing_scene()
    policy = TakeoverController(THETA)
    first = policy(c)
    assert first.takeover_started[0, 0]
    assert first.takeover_conflict_time_s[0, 0] > 0
    assert c.current[0, 0, 0] - c.current[0, 1, 0] > 4.8
    for frame in range(1, 5):
        out = policy(replace(c, response_index=frame))
        assert out.takeover_owned[0, 0]
        assert not out.takeover_started.any()


def test_existing_plan_braking_is_used_instead_of_constant_npc_speed():
    c = closing_scene()
    path = c.planned_path_xy.clone()
    time = torch.arange(1, 51) * 0.04
    path[:, :, 0, 0] = 20 * time - 2 * time.square()
    assert scores(replace(c, planned_path_xy=path))[0].max() == 0


def test_only_rear_vehicle_is_handed_over_for_longitudinal_conflict():
    c = closing_scene()
    states = c.current.clone()
    states[:, 0, 0] = -10
    states[:, 0, 2] = 30
    assert not TakeoverController(THETA)(
        change_context(c, states)
    ).takeover_started.any()


def test_lane_change_plan_is_checked_against_target_lane_rear_actor():
    c = context()
    states = c.current.clone()
    states[:, 0, 0] = -5
    states[:, 0, 1] = 3.6
    states[:, 0, 2] = 24
    valid = c.current_valid.clone()
    valid[:, 2:] = False
    path = c.planned_path_xy.clone()
    path[:, :, 0, 1] = torch.linspace(0, 3.6, 50)
    c = replace(change_context(c, states), current_valid=valid, planned_path_xy=path)
    assert scores(c)[0][0, 0] > 0


def test_within_lane_correction_does_not_admit_leader_of_a_closing_ads():
    c = closing_scene()
    states = c.current.clone()
    states[:, 0, 0] = -10
    states[:, 0, 2] = 30
    path = c.planned_path_xy.clone()
    path[:, :, 0, 1] = torch.linspace(0, 0.6, path.shape[1])
    c = replace(change_context(c, states), planned_path_xy=path)
    assert scores(c)[0].max() == 0
    assert not TakeoverController(THETA)(c).takeover_started.any()


def test_ownership_flag_does_not_flood_unaffected_followers():
    c = context()
    owned = torch.tensor([[True, False, False, False, False, False]])
    assert scores(c, owned)[0].max() == 0


def test_lane_changer_is_not_rejected_because_other_actor_changes_corridor():
    c = context()
    states = c.current.clone()
    states[:, 0, 0] = -5
    states[:, 0, 3] = 1.8
    states[:, 0, 2] = 24
    valid = c.current_valid.clone()
    valid[:, 2:] = False
    path = c.planned_path_xy.clone()
    path[:, :, 0, 1] = torch.arange(1, 51) * 0.04 * 1.8
    c = replace(change_context(c, states), current_valid=valid, planned_path_xy=path)
    assert scores(c)[0][0, 0] > 0


def test_ca_forecast_stops_without_reversing_and_cold_start_ignores_ax_label():
    current = torch.tensor([[[0.0, 0.0, 2.0, 0.0, -100.0, 0.0]]])
    times = torch.tensor([0.5, 1.0, 2.0])
    cold, _ = motion_forecast(current, None, None, times)
    torch.testing.assert_close(cold[0, :, 0, 0], 2 * times)
    previous = current.clone()
    previous[..., 2] += 0.16
    braking, _ = motion_forecast(
        current, previous, torch.ones(1, 1, dtype=torch.bool), times
    )
    torch.testing.assert_close(braking[0, :, 0, 0], torch.full((3,), 0.5))


def test_expiration_is_separate_from_interaction_and_snapshot_restores_reason():
    c = context()
    policy = TakeoverController(THETA)
    out = policy(replace(c, response_index=50))
    assert out.takeover_reason.eq(3).all()
    snapshot = policy.snapshot_runtime()
    policy.restore_runtime({})
    policy.restore_runtime(snapshot)
    assert policy.admission_reason.eq(3).all()


def test_missing_and_expired_predictions_are_not_filled_with_fake_positions():
    c = closing_scene()
    valid = torch.zeros_like(c.current_valid)
    assert scores(replace(c, current_valid=valid))[0].max() == 0
    assert scores(replace(c, response_index=50))[0].max() == 0


def test_query_horizon_is_truncated_at_plan_end():
    c = closing_scene()
    assert scores(replace(c, planned_path_xy=c.planned_path_xy[:, :2]))[0].max() == 0


def test_native_plan_gate_does_not_observe_unexecuted_ads_control(monkeypatch):
    from hierarchical_world_model.tests.test_takeover import make_tracking_world

    one = make_tracking_world(monkeypatch, TakeoverController(THETA))
    two = make_tracking_world(monkeypatch, TakeoverController(THETA))
    left = one.advance_response(torch.tensor([[-8.0, 0.1]]))
    right = two.advance_response(torch.tensor([[4.0, -0.1]]))
    for key in ("takeover_score", "takeover_started", "response_background_actions"):
        torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)


def test_external_npc_stimulus_is_applied_after_the_gate(monkeypatch):
    from hierarchical_world_model.tests.test_takeover import make_tracking_world

    one = make_tracking_world(monkeypatch, TakeoverController(THETA))
    two = make_tracking_world(monkeypatch, TakeoverController(THETA))
    delta = torch.zeros(1, 1, 6, 2)
    delta[:, :, 4, 0] = -2
    left = one.advance_response(torch.zeros(1, 2), background_action_delta=delta)
    right = two.advance_response(torch.zeros(1, 2))
    for key in ("takeover_score", "takeover_started"):
        torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)
    assert left["agent_states"][0, 5, 2] < right["agent_states"][0, 5, 2]


def test_ownership_alone_cannot_expand_uncertainty_and_activate_a_follower():
    c = closing_scene()
    states = c.current.clone()
    states[:, 2, 0] = -6
    path = c.planned_path_xy.clone()
    path[:, :, 1, 0] = -6 + torch.arange(1, 51) * 0.8
    c = replace(change_context(c, states), planned_path_xy=path)
    policy = TakeoverController(THETA)
    policy.error_bank.samples[:6, :, :, :, 0] = -3
    assert policy(c).takeover_started[0, 0]
    # The proposed NPC action was externally cancelled. Only tracking motion
    # was actually executed; its follower must not react to the label change.
    states = states.clone()
    states[..., 0] += 0.8
    states[:, 0, 0] = 1000
    states[:, 0, 2] = 20
    valid = c.current_valid.clone()
    valid[:, 2] = True
    next_context = replace(
        change_context(c, states),
        response_index=1,
        current_valid=valid,
        previous_background_actions=torch.zeros(1, 6, 2),
        planned_current_xy=path[:, 0],
    )
    result = policy(next_context)
    assert not result.takeover_motion_changed[0, 0]
    assert not result.takeover_started[0, 1]
    unguarded, _, _ = conflict_scores(
        next_context, policy.owned, policy.error_bank, horizon_s=1.5
    )
    assert unguarded[0, 1] > 0


def test_snapshot_before_first_controller_call_restores_empty_execution_latch():
    policy = TakeoverController(THETA)
    snapshot = policy.snapshot_runtime()
    policy(closing_scene())
    policy.restore_runtime(snapshot)
    assert policy.owned is None
    assert policy.response_executed is None


def test_public_generated_rollout_can_continue_after_plan_expiration():
    from types import SimpleNamespace
    from hierarchical_world_model.src.execution import _rollout_sample
    from hierarchical_world_model.src.highway import HighwayEnvClosedLoopWorld
    from hierarchical_world_model.src.randomness import WorldExogenousState

    c = context()
    sample = SimpleNamespace(
        initial_states=c.current.numpy(),
        initial_valid=c.current_valid.numpy(),
        soft_plan=c.planned_path_xy[:, :3].numpy(),
        vehicle_dimensions=c.vehicle_dimensions.numpy(),
        exogenous_state=WorldExogenousState.sample(1, seed=7, response_steps=12),
    )

    class Sampler:
        def create_world(self, sample, **kwargs):
            world = HighwayEnvClosedLoopWorld(controller=TakeoverController(THETA))
            world.reset(
                c.current,
                c.current_valid,
                c.planned_path_xy[:, :3],
                c.map_polylines,
                c.map_polyline_valid,
                exogenous_state=sample.exogenous_state,
                vehicle_dimensions=c.vehicle_dimensions,
            )
            return world

    result = _rollout_sample(
        Sampler(),
        sample,
        lambda observation: torch.zeros(1, 2),
        reaction_controller="auto",
        steps=8,
    )
    assert result.states.shape[1] == 9
    reasons = result.takeover_diagnostics["takeover_reason"]
    assert (reasons[:, 3] == 3).all()
    assert (result.takeover_diagnostics["takeover_started"].sum(1) <= 1).all()


def test_pair_sat_matches_full_sat_with_rotated_actual_bodies():
    torch.manual_seed(12)
    p = torch.randn(200, 1, 2, 2) * 4
    a = torch.randn(200, 1, 2)
    d = torch.rand(200, 2, 2) * 5 + 1
    actual = paired_rectangle_separation(
        p[:, :, 0], a[:, :, 0], d[:, None, 0], p[:, :, 1], a[:, :, 1], d[:, None, 1]
    )
    expected = rectangle_separation(p, a, d, torch.zeros_like(p))[:, :, 0, 1]
    torch.testing.assert_close(actual, expected)


def test_uncertainty_envelope_contact_needs_empirical_support():
    c = closing_scene()
    states = c.current.clone()
    states[:, 0, 0] = 12
    states[:, 0, 2] = 20
    c = change_context(c, states)
    bank = ErrorTrajectoryBank.nominal()
    bank.samples[4, 1, :1, :, 0] = -10
    q, _, _ = conflict_scores(c, torch.zeros(1, 6, dtype=torch.bool), bank)
    assert q[0, 0] == 1 / 16
    policy = TakeoverController(THETA)
    policy.takeover_config = replace(policy.takeover_config, support_threshold=0.5)
    policy.error_bank = bank
    assert not policy(c).takeover_started.any()
    bank.samples[4, 1, :, :, 0] = -10
    assert policy(c).takeover_started[0, 0]


def test_known_tracking_offset_is_anchored_at_actual_position():
    c = closing_scene()
    states = c.current.clone()
    states[:, 1, 0] = 8
    c = change_context(c, states)
    q, _, _ = conflict_scores(
        c, torch.zeros(1, 6, dtype=torch.bool), ErrorTrajectoryBank.nominal()
    )
    assert q[0, 0] == 1


def test_side_by_side_future_entry_is_not_current_zero_gap_following():
    states = torch.tensor(
        [
            [
                [0.0, 0.0, 30.0, 0.0, 0.0, 0.0],
                [3.2, 4.0, 27.0, -2.0, 0.0, 0.0],
                [100.0, 0.0, 30.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    valid = torch.ones(1, 3, dtype=torch.bool)
    sizes = torch.tensor([4.8, 1.8]).expand(1, 3, 2)
    gap, _, _, leader = physical_leader_observation(states, valid, sizes)
    assert leader[0, 0] == 2
    assert gap[0, 0] > 90
    feature = anticipated_entry_features(states, valid, sizes)
    assert feature[0, 0, 0] == 1
    time = feature[0, 0, 1]
    torch.testing.assert_close(feature[0, 0, 2], 3.2 - 3 * time)
    assert feature[0, 0, 3] == 3


def test_actual_history_acceleration_and_cold_start_have_distinct_entry_geometry():
    states = torch.tensor(
        [[[0.0, 0.0, 30.0, 0.0, 0.0, 0.0], [3.2, 4.0, 27.0, -2.0, -8.0, 0.0]]]
    )
    valid = torch.ones(1, 2, dtype=torch.bool)
    sizes = torch.tensor([4.8, 1.8]).expand(1, 2, 2)
    prior = states.clone()
    prior[:, 1, 2] += 0.32
    cold = anticipated_entry_features(states, valid, sizes)
    history = anticipated_entry_features(states, valid, sizes, prior, valid)
    assert history[0, 0, 2] < cold[0, 0, 2]
    assert history[0, 0, 3] > cold[0, 0, 3]


def test_no_candidate_is_zero_and_invalid_actor_cannot_create_an_entry():
    c = context()
    assert (
        anticipated_entry_features(c.current, c.current_valid, c.vehicle_dimensions)
        .eq(0)
        .all()
    )
    states = c.current.clone()
    states[:, 0, 1], states[:, 0, 3] = 4.0, -2.0
    valid = c.current_valid.clone()
    valid[:, 0] = False
    assert (
        anticipated_entry_features(states, valid, c.vehicle_dimensions)[:, 1:]
        .eq(0)
        .all()
    )


def test_driver_state_advances_before_admission_and_is_independent_of_ownership():
    from hierarchical_world_model.tests.test_driver_policy import correction

    class AdmitAll(TakeoverController):
        def _admission(self, c):
            _, q, t, j = super()._admission(c)
            return c.current_valid[:, 1:] & ~self.owned, q, t, j

    c = context()
    passive = TakeoverController(THETA, driver_correction=correction())
    active = AdmitAll(THETA, driver_correction=correction())
    for frame in range(7):
        actual = replace(
            c,
            response_index=frame,
            policy_standard_normal=torch.full((1, 6, 2), frame / 10),
        )
        unchanged = passive(actual)
        active(actual)
        assert not unchanged.takeover_owned.any()
        torch.testing.assert_close(unchanged.actions, actual.base_actions)
        torch.testing.assert_close(
            passive._driver_residual_state,
            active._driver_residual_state,
            atol=0,
            rtol=0,
        )


def test_empirical_quadrature_consumes_no_driver_randomness():
    c = closing_scene()
    bank = ErrorTrajectoryBank.nominal()
    before = torch.random.get_rng_state().clone()
    first = conflict_scores(c, torch.zeros(1, 6, dtype=torch.bool), bank)
    second = conflict_scores(c, torch.zeros(1, 6, dtype=torch.bool), bank)
    assert torch.equal(before, torch.random.get_rng_state())
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_generated_auto_controller_and_ads_factory_share_current_parameters():
    from hierarchical_world_model.src.composition import load_selected_components
    from hierarchical_world_model.src.shared_evaluation import load_shared_components

    selection, population = load_selected_components()
    shared_population, fit = load_shared_components()
    assert selection["mobil"] == fit["mobil"]
    for name in ("log_mean", "log_cov"):
        np.testing.assert_array_equal(population[name], shared_population[name])

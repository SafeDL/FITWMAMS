import numpy as np
import torch

from hierarchical_world_model.src.highway import _highway_classes
from hierarchical_world_model.src.npc_experiments import road as traffic
from hierarchical_world_model.src.npc_experiments.road import (
    entry_lane_intent,
    neighbors,
    packed_candidates,
    rectangular_contacts,
    simulate_road,
    traffic_statistics,
)
from hierarchical_world_model.src.stochastic_drivers.lane_trajectory import (
    LearnedLaneTrajectory,
)
from hierarchical_world_model.src.stochastic_drivers.mobil_response import (
    evaluate_mobil_candidates,
)


def cohort():
    ticks, actors = 51, 3
    states = np.zeros((ticks, actors, 6), np.float32)
    states[..., 2] = 25
    states[:, :, 0] = np.asarray([-20, 30, -100]) + np.arange(ticks)[:, None]
    states[:, 2, 1] = 4
    entry_history = np.repeat(states[0, :, None], 25, axis=1)
    entry_history[..., 0] -= np.arange(25, 0, -1)[None] * 0.04 * entry_history[..., 2]
    return {
        "states": states,
        "valid": np.ones((ticks, actors), bool),
        "recording_id": np.asarray(4),
        "direction": np.asarray(2),
        "vehicle_ids": np.array([11, 12, 13]),
        "dimensions": np.tile([4.8, 1.8], (actors, 1)).astype(np.float32),
        "entry_ticks": np.array([0, 0, 10]),
        "entry_states": states[0].copy(),
        "entry_history": entry_history,
        "entry_history_valid": np.ones((actors, 25), bool),
        "lane_centers": np.array([-4, 0, 4], np.float32),
        "lane_widths": np.ones(3, np.float32) * 4,
        "bounds": np.array([-150, 500]),
    }


def test_neighbors_resolve_physical_leaders_and_ignore_inactive_actors():
    road = cohort()
    states = road["states"][0:1]
    present = np.array([[True, False, True]])
    _, leader, gap, _, _ = neighbors(
        states, present, road["dimensions"], road["lane_centers"]
    )
    assert leader[0, 0] == -1 and gap[0, 0] == 1e6


def test_packed_driver_equation_uses_the_focal_actor_and_real_sizes():
    road = cohort()
    inputs = packed_candidates(
        road["states"][0:1],
        road["valid"][0:1],
        road["dimensions"],
        road["lane_centers"],
        road["lane_widths"],
        np.tile([32, 2, 1.2, 1, 2], (1, 3, 1)),
    )
    current, present, sizes, theta, relation = inputs
    decision = evaluate_mobil_candidates(
        current, present, theta, relation, vehicle_lengths_m=sizes[..., 0]
    )
    assert current.shape == (3, 7, 6)
    np.testing.assert_array_equal(current[:, 1].numpy(), road["states"][0])
    assert decision.incentive_mps2.shape == (3, 6, 2)


def test_continuous_driver_probe_never_reads_interior_ground_truth():
    torch.set_num_threads(1)
    road = cohort()
    population = {"log_mean": np.log([32, 2, 1.2, 1, 2]), "log_cov": np.eye(5) * 0.01}
    mobil = {"politeness": 0, "bias": -4, "slope": 0.02}
    baseline, active, metadata = simulate_road(road, population, mobil, repeats=2)
    changed = {**road, "states": road["states"] + 1000}
    replay, replay_active, _ = simulate_road(changed, population, mobil, repeats=2)
    np.testing.assert_array_equal(baseline, replay)
    np.testing.assert_array_equal(active, replay_active)
    assert not active[:, :10, 2].any() and active[:, 10, 2].all()
    assert metadata["backend"] == "highway_env.road.road.Road.step"
    assert all(run["road_step_calls"] == 50 for run in metadata["runs"])


def test_boundary_demand_queues_instead_of_spawning_on_a_vehicle(monkeypatch):
    road = cohort()
    road["entry_ticks"] = np.array([0, 1000, 1])
    road["entry_states"][:, :4] = np.array(
        [[0, 0, 0, 0], [100, 0, 0, 0], [0, 0, 25, 0]]
    )
    population = {"log_mean": np.log([32, 2, 1.2, 1, 2]), "log_cov": np.eye(5) * 0.01}
    calls = []
    road_class = _highway_classes()[1]
    original = road_class.step

    def checked_step(self, dt):
        calls.append(len(self.vehicles))
        original(self, dt)

    monkeypatch.setattr(road_class, "step", checked_step)
    _, active, metadata = simulate_road(
        road, population, {"politeness": 0, "bias": -20, "slope": 0}, repeats=1
    )
    assert len(calls) == 50
    assert not active[0, :10, 2].any()
    assert metadata["runs"][0]["entry_wait_seconds"][2] > 0.2
    assert metadata["runs"][0]["queue_vehicles"].max() >= 1


def test_causal_ongoing_lane_change_continues_toward_the_next_center():
    state = np.array([0, 1.2, 25, 0.6, 0, 0], np.float32)
    history = np.repeat(state[None], 25, 0)
    history[:, 1] = state[1] - np.arange(25, 0, -1) * 0.04 * state[3]
    valid = np.ones(25, bool)
    target, moving, crossing = entry_lane_intent(
        state, history, valid, np.array([0, 4])
    )
    assert target == 4 and moving and crossing
    mirrored = state.copy()
    mirrored[[1, 3]] *= -1
    mirrored_history = history.copy()
    mirrored_history[:, [1, 3]] *= -1
    target, moving, crossing = entry_lane_intent(
        mirrored, mirrored_history, valid, np.array([-4, 0])
    )
    assert target == -4 and moving and crossing


def test_initial_ongoing_maneuver_uses_history_without_future_reading():
    road = cohort()
    road["entry_ticks"] = np.array([0, 1000, 1000])
    entry = road["entry_states"][0]
    entry[:] = [0, 1.2, 25, 0.6, 0, 0]
    history = np.repeat(entry[None], 25, 0)
    history[:, 0] -= np.arange(25, 0, -1) * 0.04 * 25
    history[:, 1] -= np.arange(25, 0, -1) * 0.04 * 0.6
    road["entry_history"][0] = history
    population = {"log_mean": np.log([32, 2, 1.2, 1, 2]), "log_cov": np.eye(5) * 0.01}
    mobil = {"politeness": 0, "bias": -20, "slope": 0}
    trajectory, present, _ = simulate_road(road, population, mobil, repeats=1)
    changed = {**road, "states": road["states"] + 1000}
    replay, active, _ = simulate_road(changed, population, mobil, repeats=1)
    np.testing.assert_array_equal(trajectory, replay)
    np.testing.assert_array_equal(present, active)
    assert trajectory[0, -1, 0, 1] > 2.0


def test_arrival_at_a_lane_center_does_not_start_a_second_change():
    state = np.array([0, 4.04, 25, 0.3, 0, 0], np.float32)
    history = np.repeat(state[None], 25, 0)
    history[:, 1] -= np.arange(25, 0, -1) * 0.04 * 0.3
    target, moving, crossing = entry_lane_intent(
        state, history, np.ones(25, bool), np.array([0, 4, 8])
    )
    assert target == 4 and moving and not crossing


def test_intersecting_vehicles_are_retained_and_observable():
    states = np.zeros((1, 2, 6))
    states[..., 2] = 25
    states[0, 1, 0] = 3
    sizes = np.tile([4.8, 1.8], (2, 1))
    contacts = rectangular_contacts(states, np.ones((1, 2), bool), sizes)
    assert contacts[0, 0, 1] and contacts[0, 1, 0]
    assert not contacts[0, 0, 0]


def test_statistics_expose_all_vehicles_and_filter_only_derivative_boundaries():
    road = cohort()
    hist, exposure = traffic_statistics(road["states"], road["valid"], road)
    assert exposure[..., 0].sum() == 6
    assert 0 < exposure[..., 5].sum() < exposure[..., 0].sum()
    np.testing.assert_allclose(hist["speed"].sum(), exposure[..., 5].sum() / 0.2)


def test_completion_in_last_partial_sampling_interval_is_counted():
    road = cohort()
    complete = np.zeros((1, *road["valid"].shape), bool)
    complete[0, -1, 0] = True
    _, exposure = traffic_statistics(
        road["states"], road["valid"], road, completed_changes=complete
    )
    assert exposure[..., 2].sum() == 1


def _assert_balanced_decisions(counts):
    assert counts["eligible_steps"] == sum(
        counts[name]
        for name in ("keep_choice_steps", "lower_choice_steps", "higher_choice_steps")
    )
    assert counts["requested_changes"] == sum(
        counts[name]
        for name in ("path_rejected", "reservation_rejected", "started_changes")
    )


def test_physical_vehicle_randomness_survives_actor_reordering_and_counters_balance():
    torch.set_num_threads(1)
    population = {"log_mean": np.log([32, 2, 1.2, 1, 2]), "log_cov": np.eye(5) * 0.01}
    states, valid, metadata = simulate_road(
        cohort(),
        population,
        {"politeness": 0, "bias": -20, "slope": 0},
        repeats=2,
        seed=20261201,
    )
    road = cohort()
    order = np.array([2, 0, 1])
    for field in (
        "dimensions",
        "vehicle_ids",
        "entry_ticks",
        "entry_states",
        "entry_history",
        "entry_history_valid",
    ):
        road[field] = road[field][order]
    road["states"] = road["states"][:, order]
    road["valid"] = road["valid"][:, order]
    replay, present, _ = simulate_road(
        road,
        population,
        {"politeness": 0, "bias": -20, "slope": 0},
        repeats=2,
        seed=20261201,
    )
    np.testing.assert_array_equal(replay, states[:, :, order])
    np.testing.assert_array_equal(present, valid[:, :, order])
    for run in metadata["runs"]:
        counts = run["lane_decision"]
        _assert_balanced_decisions(counts)
        assert counts["eligible_steps"] == 140
        assert counts["keep_choice_steps"] == 140


def test_no_target_lane_has_its_own_denominator():
    road = cohort()
    road["lane_centers"] = np.array([0], np.float32)
    road["lane_widths"] = np.array([4], np.float32)
    road["entry_states"][:, 1] = 0
    road["entry_history"][:, :, 1] = 0
    population = {"log_mean": np.log([32, 2, 1.2, 1, 2]), "log_cov": np.eye(5) * 0.01}
    _, _, metadata = simulate_road(
        road, population, {"politeness": 0, "bias": -4, "slope": 0.02}, repeats=1
    )
    counts = metadata["runs"][0]["lane_decision"]
    _assert_balanced_decisions(counts)
    assert counts["eligible_steps"] == 0
    assert counts["no_available_target_steps"] == 140


def _request_available_lane(probability, uniform, dt):
    return torch.where(
        probability[:, 1] > 0, -1, torch.where(probability[:, 2] > 0, 1, 0)
    )


def test_path_and_reservation_rejections_balance_actual_requests(monkeypatch):
    monkeypatch.setattr(traffic, "sample_lane_choice", _request_available_lane)
    original = LearnedLaneTrajectory.safe_plan

    def rejected_path(self, *args):
        planned, clear = original(self, *args)
        return planned, torch.zeros_like(clear)

    monkeypatch.setattr(LearnedLaneTrajectory, "safe_plan", rejected_path)
    population = {"log_mean": np.log([32, 2, 1.2, 1, 2]), "log_cov": np.eye(5) * 0.01}
    mobil = {"politeness": 0, "bias": -4, "slope": 0.02}
    _, _, metadata = simulate_road(cohort(), population, mobil, repeats=1)
    counts = metadata["runs"][0]["lane_decision"]
    _assert_balanced_decisions(counts)
    assert counts["path_rejected"] == counts["requested_changes"] > 0
    assert counts["started_changes"] == 0

    def cleared_path(self, *args):
        planned, clear = original(self, *args)
        return planned, torch.ones_like(clear)

    monkeypatch.setattr(LearnedLaneTrajectory, "safe_plan", cleared_path)
    road = cohort()
    road["entry_states"][1, 0] = -12
    _, _, metadata = simulate_road(road, population, mobil, repeats=1)
    counts = metadata["runs"][0]["lane_decision"]
    _assert_balanced_decisions(counts)
    assert counts["reservation_rejected"] > 0
    assert counts["started_changes"] > 0


def test_initial_continuation_is_retained_when_actor_exits_before_completion():
    road = cohort()
    road["entry_ticks"] = np.array([0, 1000, 1000])
    road["bounds"][1] = 10
    entry = road["entry_states"][0]
    entry[:] = [0, 1.2, 25, 0.6, 0, 0]
    history = np.repeat(entry[None], 25, 0)
    history[:, 0] -= np.arange(25, 0, -1) * 0.04 * 25
    history[:, 1] -= np.arange(25, 0, -1) * 0.04 * 0.6
    road["entry_history"][0] = history
    population = {"log_mean": np.log([32, 2, 1.2, 1, 2]), "log_cov": np.eye(5) * 0.01}
    _, _, metadata = simulate_road(
        road, population, {"politeness": 0, "bias": -20, "slope": 0}, repeats=1
    )
    counts = metadata["runs"][0]["lane_decision"]
    _assert_balanced_decisions(counts)
    assert counts["initial_continuations"] == 1
    assert counts["unfinished_on_exit"] == 1
    assert counts["unfinished_at_end"] == 0

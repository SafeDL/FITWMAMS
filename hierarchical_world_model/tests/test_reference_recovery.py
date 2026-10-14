"""A retired longitudinal timetable cannot pull an NPC back to its old speed."""

import pytest
import torch

from hierarchical_world_model.src.config import WorldModelConfig
from hierarchical_world_model.src.highway import HighwayEnvClosedLoopWorld
from hierarchical_world_model.src.model import FactualDynamicsModel
from hierarchical_world_model.src.randomness import WorldExogenousState
from hierarchical_world_model.src.reference import (
    continue_retired_reference,
    recover_reference,
    planned_lateral_position_at_x,
    unreachable_reference,
)


def reference(speed=10.0):
    state = torch.zeros(1, 6, 6)
    state[..., 2] = speed
    time = torch.arange(1, 26) * 0.04
    plan = torch.zeros(1, 25, 6, 2)
    plan[..., 0] = time[None, :, None] * speed
    return state, plan


@pytest.mark.parametrize("endpoint,retired", [(5.0, True), (10.0, False), (13.0, True)])
def test_reference_feasibility_uses_acceleration_and_braking_bounds(endpoint, retired):
    state, plan = reference()
    plan[:, -1, :, 0] = endpoint
    result = unreachable_reference(state, plan, torch.full((1, 6), 25))
    assert result.eq(retired).all()


def test_reference_feasibility_ignores_unavailable_padded_tail():
    state, plan = reference()
    plan[:, 2:, :, 0] = 100.0
    result = unreachable_reference(state, plan, torch.full((1, 6), 2))
    assert not result.any()


def test_feasible_reference_is_unchanged_and_invalid_slots_do_not_retire():
    state, plan = reference()
    base = state[..., :2].clone()
    valid = torch.ones(1, 6, dtype=torch.bool)
    valid[:, -1] = False
    plan[:, :, -1, 0] = 100.0
    actual, actual_base, retired = recover_reference(
        state,
        valid,
        plan,
        base,
        torch.full((1, 6), 25),
        torch.zeros_like(valid),
        WorldModelConfig(),
        reference=plan,
        reference_base=base,
    )
    torch.testing.assert_close(actual, plan)
    torch.testing.assert_close(actual_base, base)
    assert not retired.any()


def test_retired_reference_ignores_longitudinal_timetable_for_an_unchanged_lane():
    state, plan = reference()
    state[..., 0] = 30.0
    state[..., 1] = 3.6
    plan[..., 1] = 3.6
    retired = torch.ones(1, 6, dtype=torch.bool)
    actual, base = continue_retired_reference(
        state,
        plan,
        plan[:, 0],
        retired,
        reference=plan,
        reference_base=plan[:, 0],
    )
    alternative = plan.clone()
    alternative[..., 0] = 200.0 + alternative[..., 0] * 3
    second, second_base = continue_retired_reference(
        state,
        alternative,
        alternative[:, 0],
        retired,
        reference=alternative,
        reference_base=alternative[:, 0],
    )
    torch.testing.assert_close(actual, second)
    torch.testing.assert_close(base, second_base)
    torch.testing.assert_close(actual[..., 1], plan[..., 1])
    torch.testing.assert_close(actual[..., 0], plan[..., 0] + 30)
    torch.testing.assert_close(base[..., 0], state[..., 0])


@pytest.mark.parametrize("speed", [0.0, 10.0])
def test_retired_reference_advances_lateral_path_by_actual_longitudinal_distance(speed):
    state, plan = reference(20.0)
    state[..., 2] = speed
    plan[..., 1] = plan[..., 0] * (3.6 / 20.0)
    retired = torch.zeros(1, 6, dtype=torch.bool)
    retired[:, 0] = True
    actual, base = continue_retired_reference(
        state,
        plan,
        state[..., :2],
        retired,
        reference=plan,
        reference_base=state[..., :2],
    )
    torch.testing.assert_close(actual[:, :, 0, 1], actual[:, :, 0, 0] * (3.6 / 20.0))
    torch.testing.assert_close(actual[:, :, 1:], plan[:, :, 1:])
    torch.testing.assert_close(base, state[..., :2])
    assert actual[0, -1, 0, 1] == pytest.approx(3.6 * speed / 20.0)


def test_spatial_lateral_interpolation_handles_multiple_queries_and_stopped_points():
    origin = torch.zeros(1, 1, 2)
    path = torch.tensor([[[[0.0, 0.0]], [[2.0, 1.0]], [[4.0, 2.0]]]])
    query = torch.tensor([[[-1.0], [0.0], [1.0], [4.0], [5.0]]])
    actual = planned_lateral_position_at_x(origin, path, query)
    torch.testing.assert_close(
        actual, torch.tensor([[[0.0], [0.0], [0.5], [2.0], [2.0]]])
    )


def test_retirement_stays_latched_when_original_reference_becomes_reachable():
    state, plan = reference()
    valid = torch.ones(1, 6, dtype=torch.bool)
    retired = valid.clone()
    plan[..., 0] += 0.05
    actual, _, retired = recover_reference(
        state,
        valid,
        plan,
        state[..., :2],
        torch.full((1, 6), 25),
        retired,
        WorldModelConfig(),
        reference=plan,
        reference_base=state[..., :2],
    )
    assert retired.all()
    torch.testing.assert_close(actual[..., 0], reference()[1][..., 0])


def test_retired_continuation_relaxes_acceleration_with_bounded_jerk():
    state, plan = reference()
    state[..., 4] = 2.0
    actual, _ = continue_retired_reference(
        state,
        plan,
        state[..., :2],
        torch.ones(1, 6, dtype=torch.bool),
        reference=plan,
        reference_base=state[..., :2],
    )
    velocity = (actual[:, 1:, :, 0] - actual[:, :-1, :, 0]) / 0.04
    acceleration = torch.diff(velocity, dim=1) / 0.04
    jerk = torch.diff(acceleration, dim=1).abs() / 0.04
    assert jerk.max() < 2.05
    assert acceleration.max() < 2.0
    stopped = state.clone()
    stopped[..., 2] = 0.0
    stopped[..., 4] = -4.0
    actual, _ = continue_retired_reference(
        stopped,
        plan,
        stopped[..., :2],
        torch.ones(1, 6, dtype=torch.bool),
        reference=plan,
        reference_base=stopped[..., :2],
    )
    assert actual[..., 0].eq(0).all()


def test_retired_lateral_path_does_not_change_when_preview_window_moves_ahead():
    state, full_path = reference(20.0)
    origin = state[..., :2].clone()
    full_path[..., 1] = full_path[..., 0] * (3.6 / 20.0)
    state[..., 0] = 4.0
    state[..., 1] = 0.72
    state[..., 2] = 10.0
    retired = torch.ones(1, 6, dtype=torch.bool)
    continuations = [
        continue_retired_reference(
            state,
            full_path[:, start : start + 10],
            full_path[:, start - 1],
            retired,
            reference=full_path,
            reference_base=origin,
        )
        for start in (10, 14)
    ]
    for actual, base in continuations:
        torch.testing.assert_close(actual[..., 1], actual[..., 0] * (3.6 / 20.0))
        torch.testing.assert_close(base, state[..., :2])
    torch.testing.assert_close(continuations[0][0], continuations[1][0])


def make_world(model, speed, *, npc_speed=10.0, lateral_target=0.0):
    state = torch.zeros(1, 7, 6)
    state[..., 2] = 10.0
    state[:, 1, 2] = npc_speed
    state[:, 1, 0] = 30.0
    valid = torch.zeros(1, 7, dtype=torch.bool)
    valid[:, :2] = True
    _, plan = reference(speed)
    plan[:, :, 0, 0] += 30.0
    plan[:, :, 0, 1] = torch.linspace(0, lateral_target, 25)
    maps = torch.zeros(1, 3, 8, 6)
    maps[0, :, :, 1] = torch.tensor([-3.6, 0.0, 3.6])[:, None]
    world = HighwayEnvClosedLoopWorld(
        model,
        reference_recovery_enabled=True,
    )
    world.reset(
        state,
        valid,
        plan,
        maps,
        torch.ones(1, 3, 8, dtype=torch.bool),
        exogenous_state=WorldExogenousState.sample(1, seed=91, response_steps=2),
        deterministic_response=True,
    )
    return world


def test_stopped_retired_npc_does_not_receive_a_clock_driven_lane_change_action():
    world = make_world(FactualDynamicsModel(), 20.0, npc_speed=0.0, lateral_target=3.6)
    result = world.advance_response(torch.zeros(1, 1, 2))
    assert result["reference_retired"][0, 0]
    assert abs(float(result["background_actions"][0, 0, 0, 1])) < 1e-6
    assert abs(float(result["agent_states"][0, 1, 3])) < 1e-6


def test_actual_hiqr_controls_are_independent_of_retired_longitudinal_timetable():
    torch.manual_seed(91)
    model = FactualDynamicsModel()
    torch.nn.init.normal_(model.decoder.residual_jerk.weight, std=0.1)
    slow = make_world(model, 20.0)
    fast = make_world(model, 30.0)
    action = torch.zeros(1, 1, 2)
    first = slow.advance_response(action)
    second = fast.advance_response(action)
    assert first["reference_retired"][0, 0]
    assert not first["reference_retired"][0, 1:].any()
    torch.testing.assert_close(
        first["background_actions"], second["background_actions"]
    )
    torch.testing.assert_close(first["agent_states"], second["agent_states"])
    assert first["background_actions"][0, 0, 0, 0].abs() <= 0.081


def test_snapshot_restores_retired_mask_and_reset_clears_it():
    world = make_world(FactualDynamicsModel(), 30.0)
    world.advance_response(torch.zeros(1, 1, 2))
    snapshot = world.snapshot()
    world.reference_retired.zero_()
    world.restore(snapshot)
    torch.testing.assert_close(world.reference_retired, snapshot.reference_retired)
    assert world.reference_retired[0, 0]
    world.reset(
        snapshot.states,
        world.valid,
        snapshot.reference,
        world.map_polylines,
        world.map_polyline_valid,
        exogenous_state=WorldExogenousState.sample(1, seed=91, response_steps=2),
    )
    assert not world.reference_retired.any()

"""Causal single-pass NPC response contract."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from hierarchical_world_model.src.reaction_controller import ReactionControllerContext
from hierarchical_world_model.src.stochastic_drivers.online import (
    OnlineMAIDMController, nearest_leader_observation, planned_adjacent_lane_intent,
    planned_lateral_position_at_x,
)
from traffic_components.src.core.dynamics import KinematicTrafficDynamics


THETA = np.tile(np.array([33.0, 2.0, 1.2, 1.4, 2.0], np.float32), (6, 1))


def test_map_lane_intent_uses_actual_lane_centres_and_ignores_partial_move() -> None:
    origin = torch.tensor([[0.1, 0.1]])
    terminal = torch.tensor([[-3.5, -1.5]])
    maps = torch.zeros(1, 4, 8, 6)
    maps[0, :, :, 1] = torch.tensor([-3.9, 0.0, 3.9, 13.8])[:, None]
    present = torch.ones(1, 4, 8, dtype=torch.bool)
    intent, source, target = planned_adjacent_lane_intent(
        origin, terminal, maps, present
    )
    assert intent.tolist() == [[True, False]]
    torch.testing.assert_close(source, torch.zeros_like(source))
    torch.testing.assert_close(target, torch.tensor([[-3.9, 0.0]]))


def test_generated_lateral_path_uses_longitudinal_progress_not_clock_time() -> None:
    origin = torch.tensor([[[0.0, 0.0]]])
    path = torch.tensor([[[[20.0, 0.0]], [[40.0, 0.0]], [[60.0, 3.6]]]])
    query = torch.tensor([[10.0]])
    torch.testing.assert_close(planned_lateral_position_at_x(origin, path, query), torch.zeros_like(query))
    torch.testing.assert_close(
        planned_lateral_position_at_x(origin, path, torch.tensor([[50.0]])),
        torch.tensor([[1.8]]),
    )
    torch.testing.assert_close(
        planned_lateral_position_at_x(origin, path, torch.tensor([[-5.0]])),
        torch.zeros_like(query),
    )
    torch.testing.assert_close(
        planned_lateral_position_at_x(origin, path, torch.tensor([[80.0]])),
        torch.tensor([[3.6]]),
    )


def test_slowed_npc_waits_for_spatial_lane_change_location() -> None:
    context = _context(ego_x=16.0)
    context.base_actions[0, 0, 0, 1] = 0.1
    origin = context.current[:, 1:, :2].clone()
    path = origin[:, None].expand(-1, 3, -1, -1).clone()
    path[0, :, 0, 0] = torch.tensor([20.0, 40.0, 60.0])
    path[0, :, 0, 1] = torch.tensor([0.0, 0.0, 3.6])
    planned = path[:, -1]
    slowed = context.current.clone()
    slowed[0, 1, 0] = 5.0
    context = replace(
        context, current=slowed, planned_origin_xy=origin,
        planned_current_xy=planned, planned_terminal_xy=planned,
        planned_path_xy=path,
    )
    controller = OnlineMAIDMController(THETA)
    before_location = controller(context)
    assert before_location.policy_active[0, 0]
    assert before_location.actions[0, 0, 0, 1].item() == 0.0
    late = controller(replace(context, response_index=80))
    assert late.actions[0, 0, 0, 1] > 0.0
    crossing = slowed.clone()
    crossing[0, 0, 1] = 0.0
    crossing[0, 0, 3] = 1.0
    blocked_late = controller(replace(
        context, current=crossing, response_index=80,
    ))
    assert blocked_late.actions[0, 0, 0, 1] == 0.0
    progressed = slowed.clone()
    progressed[0, 1, 0] = 55.0
    snapshot = controller.snapshot_runtime()
    at_location = controller(replace(context, current=progressed))
    restored = OnlineMAIDMController(THETA)
    restored.restore_runtime(snapshot)
    replay = restored(replace(context, current=progressed))
    torch.testing.assert_close(replay.actions, at_location.actions)
    assert at_location.policy_active[0, 0]
    assert at_location.actions[0, 0, 0, 1] > 0.0


def _context(*, ego_x: float, ego_ax: float = 0.0) -> ReactionControllerContext:
    states = torch.zeros(1, 7, 6)
    valid = torch.zeros(1, 7, dtype=torch.bool)
    valid[0, :3] = True
    states[0, 0, 0] = ego_x
    states[0, 0, 2] = 15.0
    states[0, 0, 4] = ego_ax
    states[0, 1, 2] = 20.0
    states[0, 2, 0] = -10.0
    states[0, 2, 1] = 3.6
    states[0, 2, 2] = 20.0
    base = torch.zeros(1, 1, 6, 2)
    base[0, 0, 1, 0] = 0.7
    base[0, 0, :, 1] = 0.02
    return ReactionControllerContext(
        history=states[:, None], history_valid=valid[:, None],
        current=states, current_valid=valid,
        committed_ego_controls=torch.zeros(1, 1, 2),
        base_actions=base, reference_actions=base,
        intervention_trigger=torch.zeros(1), intervention_memory=torch.zeros(1),
        lateral_intervention_memory=torch.zeros(1),
        agent_style_state=torch.zeros(1, 7, 1), response_field_gain=None,
        response_sensitivity_bounds=None, adapter_gain=None,
        reaction_enabled=None, cfg=SimpleNamespace(),
    )


def _autonomous_lane_context(*, block_left: bool = False, block_right: bool = False):
    context = _context(ego_x=90.0)
    current = context.current.clone()
    valid = context.current_valid.clone()
    current[0, 0, 1] = 0.0
    current[0, 2, :5] = torch.tensor([23.0, 0.0, 10.0, 0.0, -8.0])
    if block_left:
        valid[0, 3] = True
        current[0, 3, :4] = torch.tensor([5.0, -3.9, 20.0, 0.0])
    if block_right:
        valid[0, 4] = True
        current[0, 4, :4] = torch.tensor([5.0, 3.9, 20.0, 0.0])
    maps = torch.zeros(1, 3, 8, 6)
    maps[0, :, :, 1] = torch.tensor([-3.9, 0.0, 3.9])[:, None]
    origin = current[:, 1:, :2].clone()
    return replace(
        context, current=current, current_valid=valid,
        history=current[:, None].clone(), history_valid=valid[:, None].clone(),
        planned_origin_xy=origin, planned_current_xy=origin.clone(),
        planned_terminal_xy=origin.clone(),
        map_polylines=maps,
        map_polyline_valid=torch.ones(1, 3, 8, dtype=torch.bool),
    )


def test_new_npc_lane_intent_is_mapped_safe_and_snapshot_replayable() -> None:
    context = _autonomous_lane_context()
    controller = OnlineMAIDMController(THETA)
    first = controller(context)
    assert first.autonomous_lane_active[0, 0]
    assert first.policy_active[0, 0]
    assert first.actions[0, 0, 0, 1] < 0.0
    snapshot = controller.snapshot_runtime()
    torch.testing.assert_close(snapshot["autonomous_lane_target_y"][0, 0], torch.tensor(-3.9))
    progressed = context.current.clone()
    progressed[0, 1, 1] = -1.0
    progressed[0, 1, 3] = -0.8
    next_context = replace(context, current=progressed)
    continuing = controller(next_context)
    restored = OnlineMAIDMController(THETA)
    restored.restore_runtime(snapshot)
    replay = restored(next_context)
    torch.testing.assert_close(continuing.actions, replay.actions)
    assert continuing.autonomous_lane_active[0, 0]


def test_new_npc_lane_intent_uses_clear_adjacent_lane_only() -> None:
    blocked_left = OnlineMAIDMController(THETA)(_autonomous_lane_context(block_left=True))
    assert blocked_left.autonomous_lane_active[0, 0]
    assert blocked_left.actions[0, 0, 0, 1] > 0.0
    blocked_both = OnlineMAIDMController(THETA)(_autonomous_lane_context(
        block_left=True, block_right=True,
    ))
    assert not blocked_both.autonomous_lane_active[0, 0]
    both_occupied = _autonomous_lane_context(block_left=True, block_right=True)
    torch.testing.assert_close(
        blocked_both.actions[0, 0, 0, 1], both_occupied.base_actions[0, 0, 0, 1]
    )
    no_map_context = replace(
        _autonomous_lane_context(), map_polylines=None, map_polyline_valid=None,
    )
    assert not OnlineMAIDMController(THETA)(no_map_context).autonomous_lane_active.any()


def test_new_npc_lane_intent_completes_full_maneuver_after_leader_brake() -> None:
    # A late-started manoeuvre may cross the dataset's frame-149 boundary.
    # Controller commitment must survive that boundary even though the fixed
    # evaluation rollout itself currently stops there.
    for start_frame in (0, 145):
        context = _autonomous_lane_context()
        state = context.current.clone()
        state[0, 2, 4] = 0.0
        history = state[:, None].expand(-1, 25, -1, -1).clone()
        controller = OnlineMAIDMController(THETA)
        dynamics = KinematicTrafficDynamics()
        previous = None
        ever = False
        for relative_frame in range(149):
            frame = start_frame + relative_frame
            observed = replace(
                context, current=state, history=history,
                history_valid=context.current_valid[:, None].expand(-1, 25, -1),
                previous_background_actions=previous, response_index=frame,
            )
            output = controller(observed)
            ever |= bool(output.autonomous_lane_active[0, 0])
            controls = torch.zeros(1, 7, 2)
            controls[:, 1:] = output.actions[:, 0]
            controls[0, 2, 0] = -8.0 if relative_frame < 25 else 0.0
            state = dynamics.step(state, controls, context.current_valid, 0.04)
            history = torch.cat((history[:, 1:], state[:, None]), dim=1)
            previous = controls[:, 1:].clone()
            assert -5.7 < state[0, 1, 1] < 5.7
            if frame == 149 and start_frame == 145:
                assert output.autonomous_lane_active[0, 0]
                assert abs(float(state[0, 1, 1]) + 3.9) > 0.75
        assert ever
        assert abs(float(state[0, 1, 1]) + 3.9) < 0.75
        assert abs(float(torch.atan2(state[0, 1, 3], state[0, 1, 2]))) < 0.05


def test_unaffected_npcs_execute_exact_hiqr_actions() -> None:
    context = _context(ego_x=90.0)
    output = OnlineMAIDMController(THETA)(context)
    torch.testing.assert_close(output.actions, context.base_actions)
    assert not output.active.any()


def test_npc_clear_pass_before_ads_enters_mapped_lane_keeps_hiqr_action() -> None:
    context = _context(ego_x=5.0)
    context.current[0, 0, 2:4] = torch.tensor([20.0, 1.0])
    context.current[0, 1, 1] = 3.6
    context.current[0, 1, 2] = 30.0
    context = replace(
        context,
        history=context.current[:, None].clone(),
        map_polylines=torch.zeros(1, 2, 8, 6),
        map_polyline_valid=torch.ones(1, 2, 8, dtype=torch.bool),
    )
    context.map_polylines[0, 1, :, 1] = 3.6
    no_map = OnlineMAIDMController(THETA)(replace(
        context, map_polylines=None, map_polyline_valid=None,
    ))
    mapped = OnlineMAIDMController(THETA)(context)
    assert no_map.actions[0, 0, 0, 0] < context.base_actions[0, 0, 0, 0]
    torch.testing.assert_close(
        mapped.actions[0, 0, 0, 0], context.base_actions[0, 0, 0, 0]
    )
    torch.testing.assert_close(
        mapped.actions[0, 0, 1:], no_map.actions[0, 0, 1:]
    )


def test_clear_pass_uses_mapped_width_and_stays_committed_until_ahead() -> None:
    context = _context(ego_x=8.8)
    context.current[0, 0, 2:4] = torch.tensor([27.2, 1.7])
    context.current[0, 1, 1:4] = torch.tensor([4.85, 36.0, 0.0])
    maps = torch.zeros(1, 2, 8, 6)
    maps[0, 0, :, 1] = 0.26
    maps[0, 1, :, 1] = 4.19
    context = replace(
        context,
        history=context.current[:, None].clone(),
        map_polylines=maps,
        map_polyline_valid=torch.ones(1, 2, 8, dtype=torch.bool),
    )
    controller = OnlineMAIDMController(THETA)
    initial = controller(context)
    assert controller._passing_cutin_latched[0, 0]
    torch.testing.assert_close(initial.actions[0, 0, 0, 0], context.base_actions[0, 0, 0, 0])

    # The instantaneous forecast becomes too short while the cars are nearly
    # abreast. Dropping the earlier safe-pass commitment here causes braking
    # into the crossing corridor in real Test scenes.
    progressed = context.current.clone()
    progressed[0, 0, 0:4] = torch.tensor([15.0, 0.6, 27.0, 1.7])
    progressed[0, 1, 0] = 10.0
    later = replace(context, current=progressed)
    snapshot = controller.snapshot_runtime()
    committed = controller(later)
    torch.testing.assert_close(committed.actions[0, 0, 0, 0], later.base_actions[0, 0, 0, 0])
    restored = OnlineMAIDMController(THETA)
    restored.restore_runtime(snapshot)
    replay = restored(later)
    torch.testing.assert_close(replay.actions, committed.actions)


def test_ads_pass_does_not_mask_a_more_dangerous_npc_leader() -> None:
    context = _context(ego_x=10.0)
    context.current[0, 0, 2:4] = torch.tensor([20.0, 1.0])
    context.current[0, 1, 1:4] = torch.tensor([3.6, 30.0, 0.0])
    context.current[0, 2, 0:4] = torch.tensor([8.0, 3.6, 5.0, 0.0])
    maps = torch.zeros(1, 2, 8, 6)
    maps[0, 1, :, 1] = 3.6
    context = replace(
        context,
        history=context.current[:, None].clone(),
        map_polylines=maps,
        map_polyline_valid=torch.ones(1, 2, 8, dtype=torch.bool),
    )
    _, _, _, leader_index = nearest_leader_observation(
        context.current, context.current_valid, prediction_horizon_s=2.0
    )
    assert leader_index[0, 0] == 2
    controller = OnlineMAIDMController(THETA)
    result = controller(context)
    assert result.actions[0, 0, 0, 0] < context.base_actions[0, 0, 0, 0]
    assert not controller._passing_cutin_latched[0, 0]


def test_lagging_follower_does_not_accelerate_to_chase_plan() -> None:
    context = _context(ego_x=45.0)
    context.base_actions[0, 0, 0, 0] = 3.0
    plan = context.current[:, 1:, :2].clone()
    plan[0, 0, 0] += 3.0
    no_plan = OnlineMAIDMController(THETA)(context)
    response = OnlineMAIDMController(THETA)(replace(context, planned_current_xy=plan))
    assert not no_plan.active[0, 0]
    assert response.active[0, 0]
    assert response.actions[0, 0, 0, 0] == 0.0
    torch.testing.assert_close(response.actions[0, 0, 1:], no_plan.actions[0, 0, 1:])
    # Lag is not itself a control command when there is no nearby leader.
    distant = _context(ego_x=90.0)
    distant_plan = distant.current[:, 1:, :2].clone()
    distant_plan[0, 0, 0] += 3.0
    distant_response = OnlineMAIDMController(THETA)(replace(
        distant, planned_current_xy=distant_plan,
    ))
    torch.testing.assert_close(distant_response.actions, distant.base_actions)


def test_realized_ego_and_npc_leaders_cause_only_at_risk_follower_to_brake() -> None:
    controller = OnlineMAIDMController(THETA)
    distant = controller(_context(ego_x=45.0))
    close = controller(_context(ego_x=16.0))
    assert close.active[0, 0]
    assert close.actions[0, 0, 0, 0] < distant.actions[0, 0, 0, 0]
    assert close.actions[0, 0, 0, 0] >= -8.0
    torch.testing.assert_close(close.actions[0, 0, 1], _context(ego_x=16.0).base_actions[0, 0, 1])
    # The same rule applies to an NPC leader; no ego-only trigger is needed.
    npc_leader = _context(ego_x=90.0)
    npc_leader.current[0, 2, :4] = torch.tensor([15.0, 3.6, 15.0, 0.0])
    npc_leader.current[0, 1, 1] = 3.6
    npc_leader.current[0, 1, 2] = 20.0
    response = controller(npc_leader)
    assert response.active[0, 0]
    assert response.actions[0, 0, 0, 0] < 0.0


def test_braking_current_lane_leader_is_not_masked_by_pulling_away_target_lane_car() -> None:
    context = _context(ego_x=10.0)
    context.current[0, 0, :5] = torch.tensor([10.0, 0.0, 16.0, 0.0, -8.0])
    context.current[0, 1, :5] = torch.tensor([0.0, 1.1, 23.8, 0.88, 3.2])
    context.current[0, 2, :5] = torch.tensor([8.0, 4.26, 28.3, 0.0, -0.4])
    _, _, _, leader = nearest_leader_observation(
        context.current, context.current_valid, prediction_horizon_s=2.0,
    )
    assert leader[0, 0] == 0
    response = OnlineMAIDMController(THETA)(context)
    assert response.actions[0, 0, 0, 0] < context.base_actions[0, 0, 0, 0]


def test_controller_never_needs_nominal_future_and_preserves_lateral_action() -> None:
    context = _context(ego_x=16.0)
    output = OnlineMAIDMController(THETA)(context)
    torch.testing.assert_close(output.actions[..., 1], context.base_actions[..., 1])


def test_realized_lateral_velocity_exposes_cutin_before_lane_overlap() -> None:
    context = _context(ego_x=14.0)
    context.current[0, 0, 1] = 3.6
    context.current[0, 0, 3] = -2.5
    # Current lane is still separate, but a constant-velocity swept corridor
    # crosses the follower lane within the local response horizon.
    output = OnlineMAIDMController(THETA)(context)
    assert output.active[0, 0]
    assert output.actions[0, 0, 0, 0] < context.base_actions[0, 0, 0, 0]


def test_clear_planned_lane_exit_avoids_false_longitudinal_brake() -> None:
    context = _context(ego_x=17.23)
    current = context.current.clone()
    current[0, 0, :6] = torch.tensor([17.23, 0.37, 24.1, 0.04, -2.0, 0.0])
    current[0, 1, :6] = torch.tensor([0.0, 0.0, 27.6, -0.2, 0.0, 0.0])
    valid = context.current_valid.clone()
    valid[0, 2] = False
    base = context.base_actions.clone()
    base[0, 0, 0, 0] = -0.33
    origin = current[:, 1:, :2].clone()
    path = origin[:, None].expand(-1, 149, -1, -1).clone()
    path[0, :, 0, 0] = torch.arange(1, 150) * 27.6 * 0.04
    path[0, :, 0, 1] = -3.6 * torch.clamp(torch.arange(1, 150) / 90.0, max=1.0)
    maps = torch.zeros(1, 3, 8, 6)
    maps[0, :, :, 1] = torch.tensor([-3.6, 0.0, 3.6])[:, None]
    map_valid = torch.ones(1, 3, 8, dtype=torch.bool)
    context = replace(
        context, current=current, current_valid=valid,
        base_actions=base, planned_origin_xy=origin,
        planned_current_xy=path[:, 0], planned_terminal_xy=path[:, -1],
        planned_path_xy=path, map_polylines=maps, map_polyline_valid=map_valid,
    )
    no_path = OnlineMAIDMController(THETA)(replace(
        context, planned_origin_xy=None, planned_current_xy=None,
        planned_terminal_xy=None, planned_path_xy=None,
    ))
    clear_exit = OnlineMAIDMController(THETA)(context)
    assert no_path.actions[0, 0, 0, 0] < base[0, 0, 0, 0]
    torch.testing.assert_close(clear_exit.actions[0, 0, 0, 0], base[0, 0, 0, 0])
    severe = current.clone()
    severe[0, 0, 4] = -8.0
    severe_response = OnlineMAIDMController(THETA)(replace(context, current=severe))
    assert severe_response.actions[0, 0, 0, 0] < base[0, 0, 0, 0]
    blocked = current.clone()
    blocked[0, 2, :4] = torch.tensor([6.0, -3.6, 25.0, 0.0])
    blocked_valid = valid.clone()
    blocked_valid[0, 2] = True
    blocked_response = OnlineMAIDMController(THETA)(replace(
        context, current=blocked, current_valid=blocked_valid,
    ))
    assert blocked_response.actions[0, 0, 0, 0] < base[0, 0, 0, 0]
    pulling_away = current.clone()
    pulling_away[0, 2, :4] = torch.tensor([5.5, -3.6, 30.0, 0.0])
    clear_at_entry = OnlineMAIDMController(THETA)(replace(
        context, current=pulling_away, current_valid=blocked_valid,
    ))
    torch.testing.assert_close(clear_at_entry.actions[0, 0, 0, 0], base[0, 0, 0, 0])
    # A vehicle currently in the source lane can still enter the target lane
    # during the local horizon. It must not be treated as a clear escape gap.
    crossing = current.clone()
    crossing[0, 0, :4] = torch.tensor([-2.0, 0.37, 30.0, -0.8])
    crossing[0, 2, :5] = torch.tensor([17.23, 0.37, 24.1, 0.0, -2.0])
    crossing_valid = valid.clone()
    crossing_valid[0, 2] = True
    crossing_response = OnlineMAIDMController(THETA)(replace(
        context, current=crossing, current_valid=crossing_valid,
    ))
    assert crossing_response.actions[0, 0, 0, 0] < base[0, 0, 0, 0]


def test_departing_leader_preserves_lane_keep_but_not_hard_brake() -> None:
    context = _context(ego_x=11.086)
    current = context.current.clone()
    current[0, 0, :5] = torch.tensor([11.086, -0.83, 31.19, -0.74, -0.5])
    current[0, 1, :5] = torch.tensor([0.0, 0.0, 32.16, -0.17, 0.0])
    valid = context.current_valid.clone()
    valid[0, 2] = False
    base = context.base_actions.clone()
    base[0, 0, 0, 0] = 0.084
    origin = current[:, 1:, :2].clone()
    maps = torch.zeros(1, 3, 8, 6)
    maps[0, :, :, 1] = torch.tensor([-3.6, 0.0, 3.6])[:, None]
    map_valid = torch.ones(1, 3, 8, dtype=torch.bool)
    context = replace(
        context, current=current, current_valid=valid, base_actions=base,
        planned_origin_xy=origin, planned_terminal_xy=origin,
        map_polylines=maps, map_polyline_valid=map_valid,
    )
    no_intent = OnlineMAIDMController(THETA)(replace(
        context, planned_origin_xy=None, planned_terminal_xy=None,
    ))
    departing = OnlineMAIDMController(THETA)(context)
    assert no_intent.actions[0, 0, 0, 0] < base[0, 0, 0, 0]
    torch.testing.assert_close(departing.actions[0, 0, 0, 0], base[0, 0, 0, 0])
    severe = current.clone()
    severe[0, 0, 4] = -8.0
    emergency = OnlineMAIDMController(THETA)(replace(context, current=severe))
    assert emergency.actions[0, 0, 0, 0] < base[0, 0, 0, 0]


def test_target_lane_gap_is_checked_at_planned_entry_time() -> None:
    context = _context(ego_x=90.0)
    current = context.current.clone()
    current[0, 1, :4] = torch.tensor([6.6, 0.2, 27.6, 0.1])
    current[0, 2, :4] = torch.tensor([12.0, 3.6, 30.0, 0.0])
    origin = current[:, 1:, :2].clone()
    origin[0, 0] = torch.tensor([0.0, 0.0])
    path = origin[:, None].expand(-1, 149, -1, -1).clone()
    path[0, :, 0, 0] = torch.arange(1, 150) * 27.6 * 0.04
    path[0, :, 0, 1] = 3.6 * torch.clamp(torch.arange(1, 150) / 90.0, max=1.0)
    maps = torch.zeros(1, 3, 8, 6)
    maps[0, :, :, 1] = torch.tensor([-3.6, 0.0, 3.6])[:, None]
    map_valid = torch.ones(1, 3, 8, dtype=torch.bool)
    context = replace(
        context, current=current, planned_origin_xy=origin,
        planned_current_xy=path[:, 5], planned_terminal_xy=path[:, -1],
        planned_path_xy=path, map_polylines=maps, map_polyline_valid=map_valid,
        response_index=5,
    )
    away = OnlineMAIDMController(THETA)(context)
    assert not away.policy_active[0, 0]
    torch.testing.assert_close(away.actions[0, 0, 0], context.base_actions[0, 0, 0])
    slow = current.clone()
    slow[0, 2, 2] = 25.0
    blocked = OnlineMAIDMController(THETA)(replace(context, current=slow))
    assert blocked.policy_active[0, 0]


def test_abrupt_realized_ads_acceleration_produces_causal_positive_response() -> None:
    context = _context(ego_x=30.0, ego_ax=4.0)
    history = context.current[:, None].expand(-1, 6, -1, -1).clone()
    history[:, :-1, 0, 4] = 0.0
    context = replace(context, history=history,
                      history_valid=context.current_valid[:, None].expand(-1, 6, -1))
    output = OnlineMAIDMController(THETA)(context)
    assert output.active[0, 0]
    assert output.actions[0, 0, 0, 0] > context.base_actions[0, 0, 0, 0]
    partial_valid = context.history_valid.clone()
    partial_valid[:, 0, 0] = False
    preserved_controller = OnlineMAIDMController(THETA)
    preserved = preserved_controller(replace(context, history_valid=partial_valid))
    assert preserved.actions[0, 0, 0, 0] > context.base_actions[0, 0, 0, 0]
    assert not preserved_controller.snapshot_runtime()["disturbance_latched"][0, 0]


def test_abrupt_npc_leader_braking_triggers_response_before_unsafe_gap() -> None:
    context = _context(ego_x=90.0)
    context.current[0, 2, :5] = torch.tensor([30.0, 0.0, 20.0, 0.0, -4.0])
    history = context.current[:, None].expand(-1, 6, -1, -1).clone()
    history[:, :-1, 2, 4] = 0.0
    context = replace(
        context, history=history,
        history_valid=context.current_valid[:, None].expand(-1, 6, -1),
    )
    controller = OnlineMAIDMController(THETA)
    response = controller(context)
    assert response.active[0, 0]
    assert response.actions[0, 0, 0, 0] < context.base_actions[0, 0, 0, 0]
    assert not response.active[0, 1]
    smooth_history = history.clone()
    smooth_history[:, :, 2, 4] = -4.0
    smooth = controller(replace(context, history=smooth_history))
    torch.testing.assert_close(smooth.actions[0, 0, 0], context.base_actions[0, 0, 0])
    partial_valid = context.history_valid.clone()
    partial_valid[:, 0, 2] = False
    newly_visible = controller(replace(context, history_valid=partial_valid))
    torch.testing.assert_close(newly_visible.actions[0, 0, 0], context.base_actions[0, 0, 0])


def test_adjacent_npc_shock_waits_for_lane_alignment_but_cut_in_risk_remains() -> None:
    context = _context(ego_x=90.0)
    context.current[0, 2, :5] = torch.tensor([30.0, 2.6, 20.0, -1.5, -4.0])
    history = context.current[:, None].expand(-1, 6, -1, -1).clone()
    history[:, :-1, 2, 4] = 0.0
    context = replace(
        context, history=history,
        history_valid=context.current_valid[:, None].expand(-1, 6, -1),
    )
    safe_crossing = OnlineMAIDMController(THETA)(context)
    torch.testing.assert_close(
        safe_crossing.actions[0, 0, 0], context.base_actions[0, 0, 0]
    )
    close = context.current.clone()
    close[0, 2, 0] = 12.0
    unsafe_crossing = OnlineMAIDMController(THETA)(replace(context, current=close))
    assert unsafe_crossing.actions[0, 0, 0, 0] < context.base_actions[0, 0, 0, 0]


def test_npc_braking_propagates_through_realized_traffic_history() -> None:
    def run(*, brake_leader: bool) -> tuple[float, float, float, float]:
        context = _context(ego_x=20.0)
        context.base_actions[0, 0, 1, 0] = 0.0
        context.base_actions[..., 1] = 0.0
        state = context.current.clone()
        state[0, 0, 1:3] = torch.tensor([3.6, 20.0])
        state[0, 2, :4] = torch.tensor([30.0, 0.0, 20.0, 0.0])
        history = state[:, None].expand(-1, 25, -1, -1).clone()
        valid_history = context.current_valid[:, None].expand(-1, 25, -1)
        controller = OnlineMAIDMController(THETA)
        dynamics = KinematicTrafficDynamics()
        first_response = 0.0
        minimum_gap = float("inf")
        previous_actions = None
        follower_actions = []
        for frame in range(149):
            observed = replace(
                context, current=state, history=history,
                history_valid=valid_history,
                previous_background_actions=previous_actions,
            )
            output = controller(observed)
            follower_actions.append(float(output.actions[0, 0, 0, 0]))
            if frame == 1:
                first_response = float(output.actions[0, 0, 0, 0])
            controls = torch.zeros(1, 7, 2)
            controls[:, 1:] = output.actions[:, 0]
            if brake_leader and frame < 25:
                controls[0, 2, 0] = -4.0
            state = dynamics.step(state, controls, context.current_valid, 0.04)
            history = torch.cat((history[:, 1:], state[:, None]), dim=1)
            previous_actions = controls[:, 1:].clone()
            minimum_gap = min(minimum_gap, float(state[0, 2, 0] - state[0, 1, 0] - 4.8))
        largest_action_step = float(np.abs(np.diff(follower_actions)).max())
        return first_response, float(state[0, 1, 0]), minimum_gap, largest_action_step

    treated_action, treated_x, treated_gap, largest_step = run(brake_leader=True)
    control_action, control_x, control_gap, _ = run(brake_leader=False)
    assert treated_action < control_action
    assert treated_x < control_x
    assert treated_gap > 0.0 and control_gap > 0.0
    assert largest_step <= 12.0 * 0.04 + 1.0e-5


def test_longitudinal_release_latch_snapshots_without_future_branch() -> None:
    context = _context(ego_x=90.0)
    context.current[0, 2, :4] = torch.tensor([30.0, 0.0, 20.0, 0.0])
    previous = torch.zeros(1, 6, 2)
    previous[0, 0, 0] = -2.0
    context = replace(context, previous_background_actions=previous)
    controller = OnlineMAIDMController(THETA)
    controller._longitudinal_response_latched = torch.tensor(
        [[True, False, False, False, False, False]]
    )
    controller._disturbance_latched = controller._longitudinal_response_latched.clone()
    controller._disturbance_leader_index = torch.tensor([[2, -1, -1, -1, -1, -1]])
    controller._previous_base_ax = context.base_actions[:, 0, :, 0].clone()
    snapshot = controller.snapshot_runtime()
    first = controller(context)
    restored = OnlineMAIDMController(THETA)
    restored.restore_runtime(snapshot)
    replay = restored(context)
    torch.testing.assert_close(first.actions, replay.actions)
    assert first.active[0, 0]
    torch.testing.assert_close(first.actions[0, 0, 0, 0], torch.tensor(-1.52))
    changed_leader = OnlineMAIDMController(THETA)
    changed_leader.restore_runtime(snapshot)
    changed_leader._disturbance_leader_index[0, 0] = 0
    unaffected = changed_leader(context)
    torch.testing.assert_close(
        unaffected.actions[0, 0, 0, 0], context.base_actions[0, 0, 0, 0]
    )


def test_displaced_npc_lane_change_has_semantic_target_and_gap_wait() -> None:
    context = _context(ego_x=90.0)
    context.base_actions[..., 1] = 0.0
    origin = context.current[:, 1:, :2].clone()
    planned = origin.clone()
    planned[0, 0, 1] = 2.0
    terminal = origin.clone()
    terminal[0, 0, 1] = 3.6
    context = replace(
        context, planned_origin_xy=origin,
        planned_current_xy=planned, planned_terminal_xy=terminal,
    )
    controller = OnlineMAIDMController(THETA)
    free = controller(context)
    assert free.policy_active[0, 0]
    assert free.actions[0, 0, 0, 1] > 0.0
    # Another car in the target lane blocks an uncommitted change.
    blocked_state = context.current.clone()
    blocked_state[0, 2, 0] = 5.0
    blocked = controller(replace(context, current=blocked_state))
    assert blocked.policy_active[0, 0]
    assert blocked.actions[0, 0, 0, 1].item() == 0.0
    committed = blocked_state.clone()
    committed[0, 1, 1] = 1.0
    delayed = controller(replace(context, current=committed))
    assert not delayed.policy_active[0, 0]
    assert delayed.actions[0, 0, 0, 1].item() == 0.0
    cleared = committed.clone()
    cleared[0, 2, 0] = -20.0
    resumed = controller(replace(context, current=cleared))
    assert resumed.policy_active[0, 0]
    assert resumed.actions[0, 0, 0, 1] > 0.0


def test_on_plan_npc_lateral_action_is_exact_hiqr_action() -> None:
    context = _context(ego_x=90.0)
    origin = context.current[:, 1:, :2].clone()
    terminal = origin.clone()
    terminal[0, 0, 1] = 3.6
    context = replace(
        context, planned_origin_xy=origin,
        planned_current_xy=origin.clone(), planned_terminal_xy=terminal,
    )
    output = OnlineMAIDMController(THETA)(context)
    torch.testing.assert_close(output.actions[..., 1], context.base_actions[..., 1])


def test_unstarted_lane_plan_does_not_trigger_early_gap_wait() -> None:
    context = _context(ego_x=90.0)
    origin = context.current[:, 1:, :2].clone()
    terminal = origin.clone()
    terminal[0, 0, 1] = 3.6
    path = origin[:, None].expand(-1, 3, -1, -1).clone()
    path[0, :, 0, 0] = torch.tensor([20.0, 40.0, 60.0])
    path[0, :, 0, 1] = torch.tensor([0.0, 0.0, 3.6])
    occupied = context.current.clone()
    occupied[0, 2, :4] = torch.tensor([5.0, 3.6, 20.0, 0.0])
    context = replace(
        context, current=occupied, planned_origin_xy=origin,
        planned_current_xy=origin, planned_terminal_xy=terminal,
        planned_path_xy=path,
    )
    output = OnlineMAIDMController(THETA)(context)
    assert not output.policy_active[0, 0]
    torch.testing.assert_close(output.actions[0, 0, 0, 1], context.base_actions[0, 0, 0, 1])


def test_on_plan_change_waits_for_fast_target_lane_rear_then_resumes() -> None:
    context = _context(ego_x=90.0)
    context.base_actions[0, 0, 0, 1] = 0.1
    origin = context.current[:, 1:, :2].clone()
    planned = origin.clone()
    planned[0, 0, 1] = 0.4
    terminal = origin.clone()
    terminal[0, 0, 1] = 3.6
    context = replace(
        context, planned_origin_xy=origin,
        planned_current_xy=planned, planned_terminal_xy=terminal,
    )
    context.current[0, 2, 0] = -20.0
    context.current[0, 2, 2] = 30.0
    controller = OnlineMAIDMController(THETA)
    blocked = controller(context)
    assert blocked.policy_active[0, 0]
    assert blocked.actions[0, 0, 0, 1].item() == 0.0
    cleared = context.current.clone()
    cleared[0, 2, 0] = -50.0
    resumed = controller(replace(context, current=cleared))
    assert resumed.policy_active[0, 0]
    assert resumed.actions[0, 0, 0, 1] > 0.0


def test_receding_target_lane_front_does_not_force_unnecessary_wait() -> None:
    context = _context(ego_x=90.0)
    context.base_actions[0, 0, 0, 1] = 0.1
    origin = context.current[:, 1:, :2].clone()
    planned = origin.clone()
    planned[0, 0, 1] = 0.4
    terminal = origin.clone()
    terminal[0, 0, 1] = 3.9
    maps = torch.zeros(1, 3, 8, 6)
    maps[0, :, :, 1] = torch.tensor([0.0, 3.9, 7.8])[:, None]
    context.current[0, 2, 0] = 7.4
    context.current[0, 2, 1] = 3.9
    context.current[0, 2, 2] = 22.0
    context = replace(
        context, planned_origin_xy=origin,
        planned_current_xy=planned, planned_terminal_xy=terminal,
        map_polylines=maps,
        map_polyline_valid=torch.ones(1, 3, 8, dtype=torch.bool),
    )
    output = OnlineMAIDMController(THETA)(context)
    assert not output.policy_active[0, 0]
    torch.testing.assert_close(output.actions[0, 0, 0, 1], context.base_actions[0, 0, 0, 1])


def test_blocked_planned_change_stays_in_source_then_completes() -> None:
    context = _context(ego_x=90.0)
    context.base_actions[..., 1] = 0.0
    origin = context.current[:, 1:, :2].clone()
    planned = origin.clone()
    planned[0, 0, 1] = 0.4
    terminal = origin.clone()
    terminal[0, 0, 1] = 3.9
    maps = torch.zeros(1, 3, 8, 6)
    maps[0, :, :, 1] = torch.tensor([0.0, 3.9, 7.8])[:, None]
    map_valid = torch.ones(1, 3, 8, dtype=torch.bool)
    context = replace(
        context, planned_origin_xy=origin,
        planned_current_xy=planned, planned_terminal_xy=terminal,
        map_polylines=maps, map_polyline_valid=map_valid,
    )
    context.current[0, 2, 1] = 3.9
    controller = OnlineMAIDMController(THETA)
    dynamics = KinematicTrafficDynamics()
    for frame in range(149):
        current = context.current.clone()
        current[0, 2, 0] = current[0, 1, 0] + (5.0 if frame < 50 else -50.0)
        current[0, 2, 2] = 20.0
        context = replace(context, current=current)
        output = controller(context)
        if frame == 49:
            assert abs(context.current[0, 1, 1].item()) < 0.05
        controls = torch.zeros(1, 7, 2)
        controls[:, 1:] = output.actions[:, 0]
        context = replace(
            context,
            current=dynamics.step(context.current, controls, context.current_valid, 0.04),
        )
    assert abs(context.current[0, 1, 1].item() - 3.9) < 0.15
    heading = torch.atan2(context.current[0, 1, 3], context.current[0, 1, 2])
    assert abs(heading.item()) < 0.02


def test_displaced_npc_lane_change_finishes_and_replays_latched_state() -> None:
    context = _context(ego_x=90.0)
    context.base_actions[..., 1] = 0.0
    origin = context.current[:, 1:, :2].clone()
    planned = origin.clone()
    planned[0, 0, 1] = 2.0
    terminal = origin.clone()
    terminal[0, 0, 1] = 3.6
    context = replace(
        context, planned_origin_xy=origin,
        planned_current_xy=planned, planned_terminal_xy=terminal,
    )
    controller = OnlineMAIDMController(THETA)
    dynamics = KinematicTrafficDynamics()
    for frame in range(149):
        output = controller(context)
        if frame == 50:
            restored = OnlineMAIDMController(THETA)
            restored.restore_runtime(controller.snapshot_runtime())
            replay = restored(context)
            torch.testing.assert_close(replay.actions, output.actions)
        controls = torch.zeros(1, 7, 2)
        controls[:, 1:] = output.actions[:, 0]
        state = dynamics.step(context.current, controls, context.current_valid, 0.04)
        context = replace(context, current=state)
    assert abs(context.current[0, 1, 1].item() - 3.6) < 0.15
    heading = torch.atan2(context.current[0, 1, 3], context.current[0, 1, 2])
    assert abs(heading.item()) < 0.02

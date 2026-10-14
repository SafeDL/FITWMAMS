from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from hierarchical_world_model.src.stochastic_drivers.online import OnlineMAIDMController

from hierarchical_world_model.src.response_gate import (
    MODE_LONGITUDINAL,
    MODE_MANEUVER,
    MODE_TRACK,
    ResponseGate,
    ResponseGateDecision,
    RESPONSE_GATE_FEATURE_DIM,
    ATTENTION_RESPONSE_FEATURE_DIM,
    response_gate_features,
)


def _decision(mode: int, gate: float, count: int = 3) -> ResponseGateDecision:
    logits = torch.full((1, count, 3), -10.0)
    logits[..., mode] = 10.0
    return ResponseGateDecision(
        logits=logits,
        mode=torch.full((1, count), mode, dtype=torch.long),
        gate=torch.full((1, count), gate),
    )


def test_response_gate_shapes_and_bounds() -> None:
    model = ResponseGate()
    result = model(torch.randn(2, 6, RESPONSE_GATE_FEATURE_DIM))
    assert result.logits.shape == (2, 6, 3)
    assert result.mode.shape == (2, 6)
    assert bool(((result.gate >= 0.0) & (result.gate <= 1.0)).all())


@pytest.mark.parametrize("mode", [MODE_LONGITUDINAL, MODE_MANEUVER])
def test_online_authority_blends_bounded_idm_as_in_training(mode):
    from dataclasses import replace
    from hierarchical_world_model.tests.test_online_ma_idm import _context
    from hierarchical_world_model.src.stochastic_drivers.idm_population import torch_idm

    context = _context(ego_x=50)
    current = context.current.clone()
    current[:, 0, 2] = 20
    context = replace(
        context,
        current=current,
        history=current[:, None],
        observation_tokens=torch.zeros(1, 7, 128),
    )
    theta = torch.tensor([33.0, 2.0, 4.0, 6.0, 0.2]).expand(6, -1)
    gate = ResponseGate(input_dim=ATTENTION_RESPONSE_FEATURE_DIM)
    with torch.no_grad():
        gate.mode_head.weight.zero_()
        gate.mode_head.bias.fill_(-10)
        gate.mode_head.bias[mode] = 10
        gate.gate_head.weight.zero_()
        gate.gate_head.bias.fill_(torch.logit(torch.tensor(0.05)))
    raw_idm = torch_idm(
        torch.tensor(45.2), torch.tensor(20.0), torch.tensor(20.0), theta[0]
    )
    assert raw_idm < -8
    output = OnlineMAIDMController(theta, response_gate=gate)(context)
    torch.testing.assert_close(output.actions[0, 0, 0, 0], torch.tensor(-0.4))
    torch.testing.assert_close(output.actions[..., 1], context.base_actions[..., 1])


def test_longitudinal_gradient_does_not_identify_lateral_permission():
    logits = torch.zeros(1, 3, requires_grad=True)
    gradients = []
    for mode in (MODE_LONGITUDINAL, MODE_MANEUVER):
        decision = ResponseGateDecision(
            logits, torch.tensor([mode]), torch.tensor([0.5])
        )
        gradients.append(
            torch.autograd.grad(decision.longitudinal_authority().sum(), logits)[0]
        )
    torch.testing.assert_close(gradients[0], gradients[1])
    torch.testing.assert_close(gradients[0][:, 1], gradients[0][:, 2])


def test_binary_gate_maps_response_to_full_rule_mode() -> None:
    gate = ResponseGate(mode_count=2)
    with torch.no_grad():
        gate.mode_head.weight.zero_()
        gate.mode_head.bias.copy_(torch.tensor((0.0, 1.0)))
    output = gate(torch.zeros(4, RESPONSE_GATE_FEATURE_DIM))
    assert output.logits.shape == (4, 2)
    assert torch.equal(output.mode, torch.full((4,), MODE_MANEUVER))
    assert torch.equal(output.gate, torch.ones(4))


def test_gate_features_use_realised_one_second_history_without_changing_current() -> (
    None
):
    current = torch.zeros(1, 7, 6)
    current[..., 2] = 20.0
    current[:, 1:, 0] = -10.0
    history = current[:, None].expand(-1, 25, -1, -1).clone()
    valid = torch.ones(1, 7, dtype=torch.bool)
    relations = SimpleNamespace(
        leader_index=torch.zeros(1, 7, dtype=torch.long),
        leader_gap_m=torch.full((1, 7), 10.0),
        leader_ttc_s=torch.full((1, 7), 5.0),
    )
    values = dict(
        current=current,
        current_valid=valid,
        history=history,
        history_valid=valid[:, None].expand(-1, 25, -1),
        base_actions=torch.zeros(1, 1, 6, 2),
        planned_current_xy=current[:, 1:, :2].clone(),
        influence_authority=torch.zeros(1, 6),
        reaction_phase=torch.zeros(1, 6),
        interaction_relations=relations,
        vehicle_dimensions=torch.tensor(
            [
                [
                    [5.2, 2.0],
                    [3.0, 1.5],
                    [4.0, 1.7],
                    [4.8, 1.8],
                    [4.8, 1.8],
                    [4.8, 1.8],
                    [4.8, 1.8],
                ]
            ]
        ),
    )
    theta = torch.ones(1, 6, 5)
    factual = response_gate_features(SimpleNamespace(**values), theta)
    changed_history = history.clone()
    changed_history[:, 5, 0, 4] = -4.0
    disturbed = response_gate_features(
        SimpleNamespace(**{**values, "history": changed_history}), theta
    )
    assert factual.shape == (1, 6, RESPONSE_GATE_FEATURE_DIM)
    torch.testing.assert_close(factual[..., :21], disturbed[..., :21])
    assert not torch.equal(factual[..., 21:], disturbed[..., 21:])


def test_learned_gate_uses_distinct_entry_and_stable_release() -> None:
    controller = OnlineMAIDMController(torch.ones(6, 5), response_gate_release_frames=3)
    active = _decision(MODE_LONGITUDINAL, 0.7, count=6)
    track = _decision(MODE_TRACK, 0.0, count=6)
    entered = controller._hysteretic_gate_decision(active)
    assert entered.mode.eq(MODE_LONGITUDINAL).all()
    first_release = controller._hysteretic_gate_decision(track)
    assert first_release.mode.eq(MODE_LONGITUDINAL).all()
    snapshot = controller.snapshot_runtime()
    second_release = controller._hysteretic_gate_decision(track)
    assert second_release.mode.eq(MODE_LONGITUDINAL).all()
    released = controller._hysteretic_gate_decision(track)
    assert released.mode.eq(MODE_TRACK).all()

    restored = OnlineMAIDMController(torch.ones(6, 5), response_gate_release_frames=3)
    restored.restore_runtime(snapshot)
    replay = restored._hysteretic_gate_decision(track)
    torch.testing.assert_close(replay.mode, second_release.mode)
    torch.testing.assert_close(replay.gate, second_release.gate)


def test_longitudinal_response_phase_cannot_keep_its_own_gate_active_forever():
    from hierarchical_world_model.src.reaction_controller import ManeuverPhase

    controller = OnlineMAIDMController(torch.ones(6, 5), response_gate_release_frames=3)
    controller._hysteretic_gate_decision(_decision(MODE_LONGITUDINAL, 0.7))
    controller._maneuver_phase = torch.full((1, 3), int(ManeuverPhase.RESPOND))
    track = _decision(MODE_TRACK, 0.0)
    for _ in range(3):
        decision = controller._hysteretic_gate_decision(track)
    assert decision.mode.eq(MODE_TRACK).all()
    assert not controller._learned_gate_latched_mode.any()


def test_committed_lane_execution_keeps_authority_until_path_is_completed():
    from hierarchical_world_model.src.reaction_controller import ManeuverPhase

    controller = OnlineMAIDMController(torch.ones(6, 5), response_gate_release_frames=3)
    controller._hysteretic_gate_decision(_decision(MODE_MANEUVER, 0.7))
    controller._maneuver_phase = torch.full((1, 3), int(ManeuverPhase.EXECUTE))
    track = _decision(MODE_TRACK, 0.0)
    for _ in range(5):
        decision = controller._hysteretic_gate_decision(track)
    assert decision.mode.eq(MODE_MANEUVER).all()
    controller._maneuver_phase.fill_(int(ManeuverPhase.REBASE))
    for _ in range(3):
        decision = controller._hysteretic_gate_decision(track)
    assert decision.mode.eq(MODE_TRACK).all()


def test_attention_selector_reuses_observed_tokens_not_rule_authority():
    from dataclasses import replace
    from hierarchical_world_model.tests.test_online_ma_idm import _context, THETA

    context = replace(_context(ego_x=100), observation_tokens=torch.randn(1, 7, 128))
    model = ResponseGate(input_dim=ATTENTION_RESPONSE_FEATURE_DIM)
    first = model.features(context, torch.as_tensor(THETA))
    changed = model.features(
        replace(
            context,
            influence_authority=torch.ones(1, 6),
            reaction_phase=torch.full((1, 6), 3),
        ),
        torch.as_tensor(THETA),
    )
    assert first.shape == (1, 6, ATTENTION_RESPONSE_FEATURE_DIM)
    torch.testing.assert_close(first, changed)
    torch.testing.assert_close(first[..., :128], context.observation_tokens[:, 1:])
    assert not torch.equal(
        first,
        model.features(
            replace(context, observation_tokens=context.observation_tokens + 1),
            torch.as_tensor(THETA),
        ),
    )


def test_learned_track_cannot_disable_clearance_braking():
    from dataclasses import replace
    from hierarchical_world_model.tests.test_online_ma_idm import _context, THETA

    gate = ResponseGate(input_dim=ATTENTION_RESPONSE_FEATURE_DIM)
    with torch.no_grad():
        gate.mode_head.weight.zero_()
        gate.mode_head.bias.copy_(torch.tensor([10.0, -10.0, -10.0]))
    context = replace(_context(ego_x=12), observation_tokens=torch.zeros(1, 7, 128))
    output = OnlineMAIDMController(THETA, response_gate=gate)(context)
    assert output.response_mode.eq(MODE_TRACK).all()
    assert output.actions[0, 0, 0, 0] < 0
    assert output.active[0, 0]


def test_learned_track_preserves_unaffected_hiqr_actions():
    from dataclasses import replace
    from hierarchical_world_model.tests.test_online_ma_idm import _context, THETA

    gate = ResponseGate(input_dim=ATTENTION_RESPONSE_FEATURE_DIM)
    with torch.no_grad():
        gate.mode_head.weight.zero_()
        gate.mode_head.bias.copy_(torch.tensor([10.0, -10.0, -10.0]))
    context = replace(_context(ego_x=100), observation_tokens=torch.zeros(1, 7, 128))
    output = OnlineMAIDMController(THETA, response_gate=gate)(context)
    torch.testing.assert_close(output.actions, context.base_actions)
    assert not output.active.any()


@pytest.mark.parametrize("mode", [MODE_TRACK, MODE_LONGITUDINAL, MODE_MANEUVER])
def test_emergency_braking_does_not_override_learned_new_lane_permission(mode):
    from dataclasses import replace
    from hierarchical_world_model.tests.test_online_ma_idm import (
        _autonomous_lane_context,
        THETA,
    )

    gate = ResponseGate(input_dim=ATTENTION_RESPONSE_FEATURE_DIM)
    with torch.no_grad():
        gate.mode_head.weight.zero_()
        gate.mode_head.bias.fill_(-10)
        gate.mode_head.bias[mode] = 10
    context = replace(
        _autonomous_lane_context(), observation_tokens=torch.zeros(1, 7, 128)
    )
    output = OnlineMAIDMController(THETA, response_gate=gate)(context)
    assert output.actions[0, 0, 0, 0] < 0
    assert bool(output.autonomous_lane_active[0, 0]) == (mode == MODE_MANEUVER)


def test_selector_cannot_cancel_a_committed_lane_path():
    from dataclasses import replace
    from hierarchical_world_model.tests.test_online_ma_idm import (
        _autonomous_lane_context,
        THETA,
    )

    gate = ResponseGate(input_dim=ATTENTION_RESPONSE_FEATURE_DIM)
    with torch.no_grad():
        gate.mode_head.weight.zero_()
        gate.mode_head.bias.copy_(torch.tensor([-10.0, -10.0, 10.0]))
    context = replace(
        _autonomous_lane_context(), observation_tokens=torch.zeros(1, 7, 128)
    )
    controller = OnlineMAIDMController(
        THETA, response_gate=gate, response_gate_release_frames=1
    )
    first = controller(context)
    assert first.autonomous_lane_active[0, 0]
    target = controller._autonomous_lane_target_y.clone()
    with torch.no_grad():
        gate.mode_head.bias.copy_(torch.tensor([10.0, -10.0, -10.0]))
    states = context.current.clone()
    states[:, 1, 0] += 5
    states[:, 2, 0] = 1000
    continued = controller(replace(context, current=states, response_index=10))
    assert continued.response_mode[0, 0] == MODE_MANEUVER
    assert continued.autonomous_lane_active[0, 0]
    torch.testing.assert_close(controller._autonomous_lane_target_y, target)
    assert continued.actions[0, 0, 0, 1] * target[0, 0] > 0

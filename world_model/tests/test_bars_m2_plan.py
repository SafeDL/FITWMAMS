"""Focused invariants for the BARS-M2 overlapping-plan extension."""
from __future__ import annotations

import torch

from world_model.src.intent_response_decoder import IntentResponseDecoder, IntentResponseDecoderConfig
from world_model.src.semi_markov_model import SemiMarkovRelationalWorldModel, SemiMarkovWorldModelConfig


def test_zero_initialized_m2_plan_repeats_m1_controls() -> None:
    torch.manual_seed(3)
    h, batch, agents = 16, 2, 7
    m1 = IntentResponseDecoder(IntentResponseDecoderConfig(hidden_dim=h, plan_horizon_frames=1, execute_frames=5))
    m2 = IntentResponseDecoder(IntentResponseDecoderConfig(hidden_dim=h, plan_horizon_frames=25, execute_frames=5))
    m2.load_state_dict(m1.state_dict(), strict=False)
    agent = torch.randn(batch, agents, h)
    scene = torch.randn(batch, h)
    state = torch.randn(batch, h)
    elapsed = torch.tensor([1.0, 3.0])
    valid = torch.ones(batch, agents, dtype=torch.bool)
    reference = torch.randn(batch, agents, 2).clamp(-1.0, 1.0)
    expected = m1(agent, scene, state, elapsed, valid, reference)["controls"]
    planned = m2(agent, scene, state, elapsed, valid, reference)
    assert planned["control_plan"].shape == (batch, 25, agents, 2)
    assert planned["applied_controls"].shape == (batch, 5, agents, 2)
    assert torch.allclose(planned["applied_controls"], expected[:, None].expand(-1, 5, -1, -1), atol=1.0e-6)


def test_b0_prefix_is_preserved_while_future_plan_remains_learnable() -> None:
    torch.manual_seed(7)
    h, batch, agents = 16, 1, 7
    decoder = IntentResponseDecoder(IntentResponseDecoderConfig(hidden_dim=h, plan_horizon_frames=25, execute_frames=5))
    anchor_residual = torch.randn(batch, 5, agents, 2).clamp(-0.5, 0.5)
    result = decoder(
        torch.randn(batch, agents, h), torch.randn(batch, h), torch.randn(batch, h),
        torch.ones(batch), torch.ones(batch, agents, dtype=torch.bool), anchor_residual,
        suppress_residual=True,
    )
    assert torch.allclose(result["applied_controls"], anchor_residual, atol=1.0e-6)
    result["control_plan"][:, 5:].sum().backward()
    assert decoder.intent_time[-1].weight.grad is not None


def test_temporal_forecast_heads_cannot_perturb_executed_roll_prefix() -> None:
    torch.manual_seed(11)
    h, batch, agents = 16, 2, 7
    m1 = IntentResponseDecoder(IntentResponseDecoderConfig(hidden_dim=h, plan_horizon_frames=1, execute_frames=5))
    m2 = IntentResponseDecoder(IntentResponseDecoderConfig(hidden_dim=h, plan_horizon_frames=25, execute_frames=5))
    m2.load_state_dict(m1.state_dict(), strict=False)
    # Simulate a trained temporal forecast head.  Its effect must begin only
    # after the five physical frames actually applied by this response.
    with torch.no_grad():
        m2.intent_time[-1].bias.fill_(1.0)
        m2.local_time[-1].bias.fill_(0.25)
    agent, scene, state = torch.randn(batch, agents, h), torch.randn(batch, h), torch.randn(batch, h)
    elapsed, valid = torch.tensor([2.0, 4.0]), torch.ones(batch, agents, dtype=torch.bool)
    reference = torch.randn(batch, agents, 2).clamp(-1.0, 1.0)
    m1_controls = m1(agent, scene, state, elapsed, valid, reference)["controls"]
    m2_result = m2(agent, scene, state, elapsed, valid, reference)
    assert torch.allclose(m2_result["applied_controls"], m1_controls[:, None].expand(-1, 5, -1, -1), atol=1.0e-6)
    assert not torch.allclose(m2_result["control_plan"][:, 5:], m2_result["control_plan"][:, :1].expand(-1, 20, -1, -1))


def test_execution_plan_mask_keeps_agent_and_control_axes_distinct() -> None:
    applied = torch.zeros(2, 25, 5, 6, 2)
    target = torch.ones_like(applied)
    valid = torch.ones(2, 25, 25, 6, dtype=torch.bool)
    error = (applied - target).abs() * valid[:, :, :5, :, None].float()
    assert error.shape == applied.shape


def test_clean_plan_carry_has_zero_safe_point_and_aligned_prefix() -> None:
    controls = torch.zeros(1, 5, 7, 2)
    plan = torch.zeros(1, 25, 7, 2)
    previous = torch.zeros_like(plan)
    previous[:, 5:10].fill_(0.4)
    safe = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(
        hidden_dim=16, variant="m2", plan_horizon_frames=25, plan_execute_frames=5, plan_carry_mix=0.0,
    ))
    safe_controls, safe_plan = safe._carry_clean_plan_prefix(controls, plan, previous)
    assert torch.equal(safe_controls, controls)
    assert torch.equal(safe_plan, plan)

    carried = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(
        hidden_dim=16, variant="m2", plan_horizon_frames=25, plan_execute_frames=5, plan_carry_mix=0.25,
    ))
    carried_controls, carried_plan = carried._carry_clean_plan_prefix(controls, plan, previous)
    assert torch.allclose(carried_controls, torch.full_like(controls, 0.1))
    assert torch.allclose(carried_plan[:, :5], carried_controls)
    assert torch.equal(carried_plan[:, 5:], plan[:, 5:])


def test_plan_carry_start_defaults_to_post_anchor_and_can_be_delayed() -> None:
    default = SemiMarkovWorldModelConfig(response_interval_s=0.2)
    delayed = SemiMarkovWorldModelConfig(response_interval_s=0.2, plan_carry_start_response_steps=10)
    assert default.effective_plan_carry_start_response_steps == 5
    assert delayed.effective_plan_carry_start_response_steps == 10


def test_plan_state_target_aligns_each_physical_prediction_and_masks_suffix() -> None:
    model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(
        hidden_dim=16, variant="m2", plan_horizon_frames=10, plan_execute_frames=5,
    ))
    states = torch.zeros(1, 150, 7, 6)
    valid = torch.ones(1, 150, 7, dtype=torch.bool)
    states[:, 25:35, 1:, 0] = torch.arange(10, dtype=torch.float32).view(1, 10, 1)
    target, target_valid = model._plan_state_target(states, valid, response=0)
    assert torch.equal(target[0, :, 0, 0], torch.arange(10, dtype=torch.float32))
    assert target_valid.all()

    suffix_target, suffix_valid = model._plan_state_target(states, valid, response=24)
    assert suffix_valid[:, :5].all()
    assert not suffix_valid[:, 5:].any()
    assert torch.equal(suffix_target[:, 5:], torch.zeros_like(suffix_target[:, 5:]))


def test_joint_plan_loss_is_pairwise_masked_and_translation_invariant() -> None:
    model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(
        hidden_dim=16, variant="m2", plan_horizon_frames=4, plan_execute_frames=2,
        joint_plan_min_separation_m=2.0,
    ))
    target = torch.zeros(1, 4, 3, 6)
    target[..., 0, 0] = 0.0
    target[..., 1, 0] = 4.0
    target[..., 2, 0] = 8.0
    valid = torch.ones(1, 4, 3, dtype=torch.bool)
    translated = target.clone()
    translated[..., :2] += torch.tensor([10.0, -3.0])
    assert torch.allclose(model._joint_plan_loss(translated, target, valid), torch.zeros(()))

    perturbed = translated.clone().requires_grad_()
    with torch.no_grad():
        perturbed[..., 1, 0] += 1.0
    loss = model._joint_plan_loss(perturbed, target, valid)
    assert loss > 0.0
    loss.backward()
    assert perturbed.grad is not None

    invalid = valid.clone()
    invalid[..., 2] = False
    masked = model._joint_plan_loss(translated, target, invalid)
    ignored = translated.clone()
    ignored[..., 2, :4] += 1_000.0
    assert torch.allclose(masked, model._joint_plan_loss(ignored, target, invalid))

from __future__ import annotations

import numpy as np
from pathlib import Path
import torch

from hierarchical_world_model.src.config import WorldModelConfig
from hierarchical_world_model.src.model import DiffusionGuidedHiQR
from hierarchical_world_model.src.reaction_controller import (
    HandcraftedReactionController,
    CalibratedResidualReactionController,
    IDMResidualReactionController,
    NoReactionController,
    RLResidualReactionController,
    ReactionControllerContext,
    REACTION_FEATURE_DIM,
    controller_features,
    jerk_limited_reaction_action,
)
from hierarchical_world_model.src.influence_graph import ROLE_SAME_LANE_FOLLOWER
from hierarchical_world_model.src.rule_models import RuleModelBundle, fit_rule_models
from hierarchical_world_model.src.reaction_training import (
    PolicyTrainingConfig, ReactionEpisode, ReactionTrainingEnvironment,
    _gae, train_reaction_policy,
)
from hierarchical_world_model.src.planner import complete_missing_background_plans


def _inputs(batch: int = 2):
    current = torch.zeros(batch, 7, 6)
    current[:, 0, 2] = 25.0
    current[:, 1, 0], current[:, 1, 2] = 15.0, 18.0
    current[:, 2, 0], current[:, 2, 2] = -12.0, 27.0
    valid = torch.tensor([[True, True, True, False, False, False, False]]).expand(batch, -1).clone()
    history = current[:, None].expand(-1, 25, -1, -1).clone()
    history_valid = valid[:, None].expand(-1, 25, -1).clone()
    reference = torch.zeros(batch, 25, 6, 2)
    maps = torch.zeros(batch, 8, 8, 6)
    maps[..., 0] = torch.linspace(-100, 100, 8)[None, None]
    maps[..., 1] = torch.arange(-3, 5)[None, :, None] * 3.6
    maps[..., 2], maps[..., 4] = 1.0, 3.6
    controls = torch.zeros(batch, 5, 2); controls[:, -1, 0] = -4.0
    return history, history_valid, current, valid, reference, maps, controls


def _context(model, response, values):
    history, history_valid, current, valid, reference, _, controls = values
    return ReactionControllerContext(
        history=history, history_valid=history_valid, current=current, current_valid=valid,
        committed_ego_controls=controls, base_actions=response.actions,
        reference_actions=response.reference_actions, intervention_trigger=response.intervention_trigger,
        intervention_memory=response.intervention_memory,
        lateral_intervention_memory=response.lateral_intervention_memory,
        agent_style_state=response.agent_style_state,
        response_field_gain=response.response_field_gain,
        response_sensitivity_bounds=model.response_sensitivity_bounds,
        adapter_gain=torch.sigmoid(model.decoder.intervention_logit), cfg=model.cfg,
        reaction_enabled=None,
    )


def test_none_controller_returns_base_actions_exactly():
    values = _inputs(); model = DiffusionGuidedHiQR().eval()
    history, history_valid, current, valid, reference, maps, controls = values
    response = model(history, history_valid, current, valid, reference, current[:, 1:, :2], maps, torch.ones(2, 8, 8, dtype=torch.bool), committed_ego_controls=controls, deterministic=True, apply_intervention_adapter=False)
    output = NoReactionController()(_context(model, response, values))
    torch.testing.assert_close(output.actions, response.actions)


def test_explicit_ego_response_can_be_excluded_without_changing_default_contract():
    values = _inputs(); model = DiffusionGuidedHiQR().eval()
    history, history_valid, current, valid, reference, maps, controls = values
    kwargs = dict(filter_state=None, committed_ego_controls=controls, deterministic=True,
        apply_intervention_adapter=False)
    default = model(history, history_valid, current, valid, reference, current[:, 1:, :2], maps,
        torch.ones(2, 8, 8, dtype=torch.bool), **kwargs)
    explicit = model(history, history_valid, current, valid, reference, current[:, 1:, :2], maps,
        torch.ones(2, 8, 8, dtype=torch.bool), apply_explicit_ego_response=True, **kwargs)
    nominal = model(history, history_valid, current, valid, reference, current[:, 1:, :2], maps,
        torch.ones(2, 8, 8, dtype=torch.bool), apply_explicit_ego_response=False, **kwargs)
    torch.testing.assert_close(default.actions, explicit.actions)
    assert torch.isfinite(nominal.actions).all()


def test_scope_masked_present_background_plan_is_completed_from_frozen_highd():
    states = torch.zeros(1, 174, 7, 6).numpy()
    valid = torch.zeros(1, 174, 7, dtype=torch.bool).numpy()
    states[0, 24, 2, 0] = -10.0
    valid[0, 24:, 2] = True
    states[0, 25:174, 2, 0] = torch.arange(149).numpy()
    plan = torch.zeros(1, 149, 6, 2).numpy()
    completed = complete_missing_background_plans(plan, states, valid)
    assert float(completed[0, 0, 1, 0]) == 0.0
    assert float(completed[0, 10, 1, 0]) == 10.0
    torch.testing.assert_close(torch.from_numpy(completed[..., 0, :]), torch.from_numpy(plan[..., 0, :]))


def test_handcrafted_hook_matches_legacy_decoder_adapter():
    cfg = WorldModelConfig(intervention_adapter_enabled=True, lateral_response_adapter_enabled=True)
    model = DiffusionGuidedHiQR(cfg).eval(); values = _inputs()
    history, history_valid, current, valid, reference, maps, controls = values
    kwargs = dict(filter_state=None, committed_ego_controls=controls, deterministic=True)
    legacy = model(history, history_valid, current, valid, reference, current[:, 1:, :2], maps, torch.ones(2, 8, 8, dtype=torch.bool), **kwargs)
    base = model(history, history_valid, current, valid, reference, current[:, 1:, :2], maps, torch.ones(2, 8, 8, dtype=torch.bool), apply_intervention_adapter=False, **kwargs)
    external = HandcraftedReactionController(model.decoder.intervention_logit)(_context(model, base, values))
    torch.testing.assert_close(external.actions, legacy.actions, rtol=0.0, atol=1.0e-6)


def test_rl_residual_keeps_yaw_and_respects_acceleration_limits():
    values = _inputs(); model = DiffusionGuidedHiQR().eval()
    history, history_valid, current, valid, reference, maps, controls = values
    response = model(history, history_valid, current, valid, reference, current[:, 1:, :2], maps, torch.ones(2, 8, 8, dtype=torch.bool), committed_ego_controls=controls, deterministic=True, apply_intervention_adapter=False)
    controller = RLResidualReactionController(); output = controller(_context(model, response, values))
    torch.testing.assert_close(output.actions[..., 1], response.actions[..., 1])
    assert float(output.actions[..., 0].min()) >= -8.0
    assert float(output.actions[..., 0].max()) <= 4.0
    assert torch.equal(output.actions[:, 0, 2:], response.actions[:, 0, 2:])


def test_unresolved_following_risk_cannot_rebound_to_hiqr_acceleration():
    """The persistent controller guard is physical, causal and longitudinal."""
    values = _inputs(); model = DiffusionGuidedHiQR().eval()
    history, history_valid, current, valid, reference, maps, controls = values
    # Same-rear target 10 m behind and closing at 5 m/s: realized TTC=2 s.
    current[:, 0, 0], current[:, 0, 2] = 20., 20.
    current[:, 2, 0], current[:, 2, 2] = 10., 25.
    history[:, -1] = current
    response = model(history, history_valid, current, valid, reference, current[:, 1:, :2], maps,
        torch.ones(2, 8, 8, dtype=torch.bool), committed_ego_controls=controls, deterministic=True,
        apply_intervention_adapter=False)
    base = response.actions.clone(); base[:, 0, 1, 0] = 3.0
    response = type(response)(**{**response.__dict__, "actions": base})
    context = ReactionControllerContext(**{
        **_context(model, response, values).__dict__, "reaction_enabled": torch.ones(2, dtype=torch.bool),
        "reaction_phase": torch.ones(2, dtype=torch.long), "reaction_release_ttc_s": 4.0,
    })
    output = RLResidualReactionController()(context, deterministic=True)
    assert torch.all(output.actions[:, 0, 1, 0] <= 0.0)
    torch.testing.assert_close(output.actions[..., 1], base[..., 1])


def test_stateful_jerk_layer_is_causal_smooth_and_exact_when_unarmed():
    values = _inputs(); model = DiffusionGuidedHiQR().eval()
    history, history_valid, current, valid, reference, maps, controls = values
    current[:, 0, 0], current[:, 0, 2] = 20., 20.
    current[:, 2, 0], current[:, 2, 2] = 10., 25.  # realised TTC = 2 s
    history[:, -1] = current
    response = model(history, history_valid, current, valid, reference, current[:, 1:, :2], maps,
        torch.ones(2, 8, 8, dtype=torch.bool), committed_ego_controls=controls, deterministic=True,
        apply_intervention_adapter=False)
    prior = torch.zeros(2, 6, 2)
    desired = response.actions[:, 0, :, 0].clone(); desired[:, 1] = -8.
    armed = ReactionControllerContext(**{
        **_context(model, response, values).__dict__, "reaction_enabled": torch.ones(2, dtype=torch.bool),
        "reaction_phase": torch.ones(2, dtype=torch.long), "reaction_release_ttc_s": 4.0,
        "previous_background_actions": prior,
    })
    executed = jerk_limited_reaction_action(desired, armed, target_slot_index=1)
    # At TTC=2 the physical brake change is bounded by (20 + 40*2/3)*0.04.
    assert torch.all(executed[:, 1] <= 0.)
    assert torch.all(executed[:, 1] >= -(20. + 40. * 2. / 3.) * .04 - 1.e-6)
    unarmed = ReactionControllerContext(**{**armed.__dict__, "reaction_phase": torch.zeros(2, dtype=torch.long)})
    torch.testing.assert_close(jerk_limited_reaction_action(desired, unarmed, target_slot_index=1), desired)


def test_rl_residual_only_changes_the_observed_same_rear_brake_case():
    values = _inputs(); model = DiffusionGuidedHiQR().eval()
    history, history_valid, current, valid, reference, maps, controls = values
    response = model(history, history_valid, current, valid, reference, current[:, 1:, :2], maps,
        torch.ones(2, 8, 8, dtype=torch.bool), committed_ego_controls=controls, deterministic=True,
        apply_intervention_adapter=False)
    context = _context(model, response, values)
    disabled = ReactionControllerContext(**{**context.__dict__, "reaction_enabled": torch.zeros(2, dtype=torch.bool)})
    output = RLResidualReactionController()(disabled, deterministic=True)
    torch.testing.assert_close(output.actions, response.actions)
    assert not output.active.any()


def test_ppo_evaluation_has_actor_and_critic_gradients():
    controller = RLResidualReactionController()
    features = torch.randn(8, REACTION_FEATURE_DIM); raw = torch.randn(8, 2)
    log_prob, entropy, value = controller.evaluate_raw_action(features, raw)
    loss = -log_prob.mean() - .01 * entropy.mean() + value.square().mean(); loss.backward()
    assert controller.actor_mean.weight.grad is not None
    assert controller.critic[-1].weight.grad is not None
    rewards, values = torch.ones(3, 2), torch.zeros(3, 2)
    advantages, returns = _gae(rewards, values, torch.zeros(3, 2, dtype=torch.bool), .99, .95)
    assert torch.isfinite(advantages).all() and torch.isfinite(returns).all()


def test_tiny_ppo_training_saves_a_reloadable_controller(tmp_path):
    values = _inputs(batch=1)
    _, _, current, valid, _, maps, _ = values
    states = current[:, None].expand(-1, 174, -1, -1).clone().numpy()
    arrays = {"agent_states": states, "agent_valid": valid[:, None].expand(-1, 174, -1).clone().numpy(),
              "map_polylines": maps.numpy(), "map_polyline_valid": torch.ones(1, 8, 8, dtype=torch.bool).numpy()}
    cfg = PolicyTrainingConfig(updates=1, rollout_steps=64, episodes_per_rollout=2,
        event_episodes=0, non_event_episodes=0, synthetic_episodes=2,
        epochs_per_update=1, minibatch_size=32)
    summary = train_reaction_policy(DiffusionGuidedHiQR().eval(), train_arrays=arrays,
        train_plans=torch.zeros(1, 149, 6, 2).numpy(), output_dir=tmp_path, config=cfg,
        device=torch.device("cpu"), controller_mode="rl_residual")
    payload = torch.load(summary["checkpoint"], map_location="cpu", weights_only=False)
    reloaded = RLResidualReactionController(); reloaded.load_state_dict(payload["state_dict"])


def test_synthetic_stream_never_receives_a_human_target():
    values = _inputs(batch=1)
    _, _, current, valid, _, maps, _ = values
    arrays = {
        "agent_states": current[:, None].expand(-1, 174, -1, -1).clone().numpy(),
        "agent_valid": valid[:, None].expand(-1, 174, -1).clone().numpy(),
        "map_polylines": maps.numpy(),
        "map_polyline_valid": torch.ones(1, 8, 8, dtype=torch.bool).numpy(),
    }
    config = PolicyTrainingConfig(
        rollout_steps=1, episodes_per_rollout=1,
        event_episodes=0, non_event_episodes=0, synthetic_episodes=1,
    )
    environment = ReactionTrainingEnvironment(
        DiffusionGuidedHiQR().eval(), arrays=arrays,
        soft_plans=torch.zeros(1, 149, 6, 2).numpy(),
        controller=RLResidualReactionController(), device=torch.device("cpu"),
        config=config,
    )
    environment.reset([ReactionEpisode(0, "synthetic")])
    _, _, _, info = environment.step()
    assert not info["human_target_gate"].any()


def test_tiny_ppo_uses_post_full_pass_validation_checkpoint(tmp_path):
    values = _inputs(batch=1)
    _, _, current, valid, _, maps, _ = values
    arrays = {
        "agent_states": current[:, None].expand(-1, 174, -1, -1).clone().numpy(),
        "agent_valid": valid[:, None].expand(-1, 174, -1).clone().numpy(),
        "map_polylines": maps.numpy(),
        "map_polyline_valid": torch.ones(1, 8, 8, dtype=torch.bool).numpy(),
    }
    plans = torch.zeros(1, 149, 6, 2).numpy()
    config = PolicyTrainingConfig(
        updates=1, rollout_steps=64, episodes_per_rollout=2,
        event_episodes=0, non_event_episodes=0, synthetic_episodes=2,
        epochs_per_update=1, minibatch_size=16,
    )
    summary = train_reaction_policy(
        DiffusionGuidedHiQR().eval(), train_arrays=arrays,
        train_plans=plans, output_dir=tmp_path, config=config,
        device=torch.device("cpu"), controller_mode="rl_residual",
    )
    assert summary["best_validation_energy_score"] is None


def test_tiny_idm_and_calibrated_ppo_training_smoke(tmp_path):
    values = _inputs(batch=1); _, _, current, valid, _, maps, _ = values
    states = current[:, None].expand(-1, 174, -1, -1).clone().numpy()
    arrays = {"agent_states": states, "agent_valid": valid[:, None].expand(-1, 174, -1).clone().numpy(),
              "map_polylines": maps.numpy(), "map_polyline_valid": torch.ones(1, 8, 8, dtype=torch.bool).numpy()}
    cfg = PolicyTrainingConfig(updates=1, rollout_steps=64, episodes_per_rollout=2,
        event_episodes=0, non_event_episodes=0, synthetic_episodes=2,
        epochs_per_update=1, minibatch_size=32)
    for mode in ("rl_residual_idm", "calibrated_residual"):
        summary = train_reaction_policy(DiffusionGuidedHiQR().eval(), train_arrays=arrays,
            train_plans=torch.zeros(1, 149, 6, 2).numpy(), output_dir=tmp_path / mode, config=cfg,
            device=torch.device("cpu"), controller_mode=mode, rule_model=_rules())
        assert Path(summary["checkpoint"]).is_file() and summary["controller_mode"] == mode


def test_authority_features_include_realized_phase_and_age():
    values = _inputs(); model = DiffusionGuidedHiQR().eval()
    history, history_valid, current, valid, reference, maps, controls = values
    response = model(history, history_valid, current, valid, reference, current[:, 1:, :2], maps,
        torch.ones(2, 8, 8, dtype=torch.bool), committed_ego_controls=controls, deterministic=True,
        apply_intervention_adapter=False)
    context = _context(model, response, values)
    active = ReactionControllerContext(**{
        **context.__dict__, "reaction_enabled": torch.ones(2, dtype=torch.bool),
        "reaction_phase": torch.tensor((1, 2)), "reaction_age_frames": torch.tensor((4, 12)),
        "reaction_max_frames": 75, "reaction_recovery_remaining": torch.tensor((0, 8)),
        "reaction_recovery_frames": 15,
    })
    features = controller_features(active)
    assert features.shape == (2, 6, REACTION_FEATURE_DIM)
    assert not torch.equal(features[0], features[1])


def test_recovery_envelope_continuously_tapers_residual_authority():
    values = _inputs(); model = DiffusionGuidedHiQR().eval()
    history, history_valid, current, valid, reference, maps, controls = values
    response = model(history, history_valid, current, valid, reference, current[:, 1:, :2], maps,
        torch.ones(2, 8, 8, dtype=torch.bool), committed_ego_controls=controls, deterministic=True,
        apply_intervention_adapter=False)
    context = _context(model, response, values)
    recovery = ReactionControllerContext(**{
        **context.__dict__, "reaction_enabled": torch.ones(2, dtype=torch.bool),
        "reaction_phase": torch.full((2,), 2), "reaction_age_frames": torch.tensor((6, 6)),
        "reaction_max_frames": 75, "reaction_recovery_remaining": torch.tensor((0, 1)),
        "reaction_recovery_frames": 15,
    })
    output = RLResidualReactionController()(recovery, deterministic=True)
    # Slot 1 is the only follower role.  A zero countdown has exact handoff;
    # one remaining frame can use at most 1/15 of the learned gate.
    torch.testing.assert_close(output.actions[0, :, 1], response.actions[0, :, 1])
    assert float(output.alpha[1, 1]) <= 1.0 / 15.0


def _rules():
    return RuleModelBundle((32., 2.5, 3.0, 2., 1.2, 4.), .3, .2, 3.)


def test_idm_and_calibrated_controllers_are_causal_legal_and_keep_yaw():
    values = _inputs(); model = DiffusionGuidedHiQR().eval()
    history, history_valid, current, valid, reference, maps, controls = values
    response = model(history, history_valid, current, valid, reference, current[:, 1:, :2], maps,
        torch.ones(2, 8, 8, dtype=torch.bool), committed_ego_controls=controls, deterministic=True, apply_intervention_adapter=False)
    context = ReactionControllerContext(**{**_context(model, response, values).__dict__, "reaction_enabled": torch.ones(2, dtype=torch.bool)})
    controller = CalibratedResidualReactionController(_rules())
    output = controller(context, deterministic=True)
    torch.testing.assert_close(output.actions[..., 1], response.actions[..., 1])
    assert float(output.actions[..., 0].min()) >= -8. and float(output.actions[..., 0].max()) <= 4.
    zero_authority = torch.zeros(2, 6)
    disabled = controller(ReactionControllerContext(**{**context.__dict__, "influence_authority": zero_authority}), deterministic=True)
    torch.testing.assert_close(disabled.actions, response.actions)


def test_calibrated_policy_has_exact_zero_residual_before_learning():
    values = _inputs(); model = DiffusionGuidedHiQR().eval()
    history, history_valid, current, valid, reference, maps, controls = values
    response = model(
        history, history_valid, current, valid, reference, current[:, 1:, :2], maps,
        torch.ones(2, 8, 8, dtype=torch.bool), committed_ego_controls=controls,
        deterministic=True, apply_intervention_adapter=False,
    )
    authority = torch.zeros(2, 6); authority[:, 1] = 1.0
    roles = torch.zeros(2, 6, dtype=torch.long); roles[:, 1] = ROLE_SAME_LANE_FOLLOWER
    context = ReactionControllerContext(**{
        **_context(model, response, values).__dict__,
        "influence_authority": authority,
        "influence_role": roles,
        "influence_predicted_ttc_s": torch.full((2, 6), 5.0),
        "previous_background_actions": response.actions[:, 0].detach().clone(),
    })
    output = CalibratedResidualReactionController(_rules())(context, deterministic=True)
    torch.testing.assert_close(output.actions, response.actions, rtol=0.0, atol=0.0)


def test_idm_feature_does_not_define_calibrated_action_bounds():
    base = torch.tensor(((-7.0, 3.0),), dtype=torch.float32)
    authority = torch.ones_like(base)
    active = torch.ones_like(base, dtype=torch.bool)
    raw = torch.tensor((((20.0, -20.0), (20.0, 20.0))), dtype=torch.float32)
    mapped, _ = CalibratedResidualReactionController.mapped_action(
        base, authority, active, raw, -8.0, 4.0,
    )
    assert mapped[0, 0] >= -8.0 and mapped[0, 1] <= 4.0
    # The mapping accepts no IDM value: changing IDM can only change the
    # actor feature upstream, never the executable interval itself.
    assert list(CalibratedResidualReactionController.mapped_action.__annotations__) == [
        "base", "authority", "active", "raw", "minimum", "maximum", "return",
    ]


def test_rule_calibration_returns_one_global_bounded_model():
    states = torch.zeros(2, 174, 7, 6).numpy(); valid = torch.zeros(2, 174, 7, dtype=torch.bool).numpy()
    valid[:, :, :3] = True; states[:, :, 0, 2] = 20.; states[:, :, 2, 0] = -12.; states[:, :, 2, 2] = 24.
    states[:, :, 0, 0] = torch.arange(174).numpy()[None] * .8; states[:, :, 2, 0] += torch.arange(174).numpy()[None] * .96
    rules, report = fit_rule_models({"agent_states": states, "agent_valid": valid}, np.asarray((0, 1)), epochs=1, batch_size=64)
    assert len(rules.idm_parameters) == 6 and report["following_observations"] >= 30

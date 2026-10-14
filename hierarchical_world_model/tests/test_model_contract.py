"""Contract tests for soft-reference causal response generation."""

from __future__ import annotations

import pytest
import torch

from hierarchical_world_model.src.calibration import NaturalResponseCalibrator
from hierarchical_world_model.src.config import WorldModelConfig
from hierarchical_world_model.src.core.highd_metrics import temporal_factual_metrics
from hierarchical_world_model.src.core.highd_metrics import (
    _collision_indicator,
    _histogram_summary,
    _nearest_object_distance,
    _windowed_jerk,
)
from hierarchical_world_model.src.highway import HighwayEnvClosedLoopWorld
from hierarchical_world_model.src.randomness import WorldExogenousState
from hierarchical_world_model.src.hiqr.encoder import UnifiedRelationalQueryEncoder
from hierarchical_world_model.src.losses import training_losses
from hierarchical_world_model.src.model import FactualDynamicsModel
from hierarchical_world_model.src.reference import rebase_soft_preview
from hierarchical_world_model.src.train import _apply_stage_trainable


@pytest.mark.parametrize("valid_frames", ((0, 1, 2, 3, 4), (3, 4), (0, 2, 4)))
def test_history_encoder_selects_latest_observed_frame(valid_frames):
    class PassThrough(torch.nn.Module):
        def forward(self, values, src_key_padding_mask):
            return values

    encoder = UnifiedRelationalQueryEncoder(WorldModelConfig())
    encoder.temporal_layers = PassThrough()
    encoder.state_mlp = torch.nn.Linear(10, 128, bias=False)
    with torch.no_grad():
        encoder.state_mlp.weight.zero_()
        encoder.state_mlp.weight[0, 0] = 1.0
        encoder.temporal_position.zero_()
        encoder.mode_embedding.weight.zero_()
    history = torch.zeros(1, 5, 7, 6)
    history[0, :, :, 0] = torch.arange(1, 6)[:, None] * 100.0
    valid = torch.zeros(1, 5, 7, dtype=torch.bool)
    valid[:, valid_frames] = True
    current_valid = torch.ones(1, 7, dtype=torch.bool)
    current_valid[:, -1] = False
    token = encoder._temporal_tokens(
        history, valid, current_valid, torch.zeros_like(current_valid), "roll"
    )
    torch.testing.assert_close(token[0, :-1, 0], torch.full((6,), 5.0))
    assert token[0, -1].eq(0).all()


def inputs(batch: int = 2):
    history = torch.zeros(batch, 25, 7, 6)
    history[..., 2] = 25.0
    valid = torch.ones(batch, 25, 7, dtype=torch.bool)
    current = history[:, -1].clone()
    current_valid = valid[:, -1].clone()
    reference = torch.zeros(batch, 25, 6, 2)
    reference[..., 0] = torch.arange(1, 26)[None, :, None]
    base = torch.zeros(batch, 6, 2)
    maps = torch.zeros(batch, 8, 8, 6)
    maps[..., 0] = torch.linspace(-100.0, 100.0, 8)[None, None]
    maps[..., 1] = torch.arange(-3, 5)[None, :, None] * 3.6
    maps[..., 2] = 1.0
    maps[..., 4] = 3.6
    map_valid = torch.ones(batch, 8, 8, dtype=torch.bool)
    return history, valid, current, current_valid, reference, base, maps, map_valid


def highway_world(model, seed):
    initial = torch.zeros(7, 6)
    initial[:, 0] = torch.tensor([0, 60, -60, 30, -30, 30, -30])
    initial[:, 1] = torch.tensor([0, 0, 0, 3.6, 3.6, -3.6, -3.6])
    initial[:, 2] = 25
    time = torch.arange(1, 51) * 0.04
    reference = initial[None, 1:, :2] + time[:, None, None] * initial[None, 1:, 2:4]
    maps = torch.zeros(3, 8, 6)
    maps[..., 0] = torch.linspace(-200, 200, 8)[None]
    maps[..., 1] = torch.tensor([-3.6, 0, 3.6])[:, None]
    maps[..., 2] = 1
    maps[..., 4] = 3.6
    world = HighwayEnvClosedLoopWorld(model)
    world.reset(
        initial[None],
        torch.ones(1, 7, dtype=torch.bool),
        reference[None],
        maps[None],
        torch.ones(1, 3, 8, dtype=torch.bool),
        exogenous_state=WorldExogenousState.sample(1, seed=seed),
        vehicle_dimensions=torch.tensor([4.8, 1.8]).expand(1, 7, 2),
    )
    return world


def test_missing_previous_frame_does_not_create_a_state_change():
    model = FactualDynamicsModel(WorldModelConfig())
    model.eval()
    values = list(inputs(batch=1))
    values[1][:, :-1] = False
    controls = torch.zeros(1, 1, 2)
    with torch.no_grad():
        first = model(*values, committed_ego_controls=controls, deterministic=True)
        values[0][:, :-1] = 10000.0
        second = model(*values, committed_ego_controls=controls, deterministic=True)
    torch.testing.assert_close(first.mean, second.mean, atol=0, rtol=0)


def test_invalid_control_padding_cannot_trigger_a_response():
    model = FactualDynamicsModel(WorldModelConfig())
    model.eval()
    values = inputs(batch=1)
    with torch.no_grad():
        first = model(
            *values,
            committed_ego_controls=torch.zeros(1, 24, 2),
            committed_ego_controls_valid=torch.zeros(1, 24, dtype=torch.bool),
            deterministic=True,
        )
        second = model(
            *values,
            committed_ego_controls=torch.full((1, 24, 2), 1000.0),
            committed_ego_controls_valid=torch.zeros(1, 24, dtype=torch.bool),
            deterministic=True,
        )
    torch.testing.assert_close(first.mean, second.mean, atol=0, rtol=0)


def test_current_time_embedding_is_independent_of_history_length():
    class PassThrough(torch.nn.Module):
        def forward(self, values, src_key_padding_mask):
            return values

    encoder = UnifiedRelationalQueryEncoder(WorldModelConfig())
    encoder.temporal_layers = PassThrough()
    encoder.state_mlp = torch.nn.Linear(10, 128, bias=False)
    with torch.no_grad():
        encoder.state_mlp.weight.zero_()
    valid = torch.ones(1, 7, dtype=torch.bool)
    tokens = []
    for frames in (1, 5, 25):
        history = torch.zeros(1, frames, 7, 6)
        mask = torch.zeros(1, frames, 7, dtype=torch.bool)
        mask[:, -1] = True
        tokens.append(
            encoder._temporal_tokens(
                history, mask, valid, torch.zeros_like(valid), "roll"
            )
        )
    torch.testing.assert_close(tokens[0], tokens[1], atol=0, rtol=0)
    torch.testing.assert_close(tokens[0], tokens[2], atol=0, rtol=0)


def test_rebase_keeps_the_soft_reference_as_a_long_horizon_anchor():
    values = inputs()
    current, reference, base = values[2], values[4], values[5]
    first = rebase_soft_preview(current[:, 1:], reference, base)
    shifted_current = current[:, 1:].clone()
    shifted_current[..., :2] += 1000.0
    translated = rebase_soft_preview(shifted_current, reference + 1000.0, base + 1000.0)
    torch.testing.assert_close(translated, first + 1000.0)


def test_rebase_matches_realized_velocity_without_moving_preview_endpoint():
    values = inputs(batch=1)
    current, reference, base = values[2], values[4], values[5]
    current[:, 1:, 2] = 20.0
    rebased = rebase_soft_preview(
        current[:, 1:], reference, base, endpoint_offset_weight=0.25
    )
    first_velocity = (rebased[:, 0] - current[:, 1:, :2]) / 0.04
    assert ((first_velocity[..., 0] - 20.0).abs() < 1.1).all()
    expected_endpoint = reference[:, -1:] + 0.25 * (
        current[:, None, 1:, :2] - base[:, None]
    )
    torch.testing.assert_close(rebased[:, -1:], expected_endpoint)


def test_reaction_reference_preserves_longitudinal_schedule_offset_only():
    values = inputs(batch=1)
    current, reference, base = values[2], values[4], values[5]
    current[:, 1:, 0] = -10.0
    current[:, 1:, 1] = 3.6
    current[:, 1:, 2] = 25.0
    rebased = rebase_soft_preview(
        current[:, 1:],
        reference,
        base,
        longitudinal_endpoint_offset_weight=1.0,
        lateral_endpoint_offset_weight=0.0,
    )
    torch.testing.assert_close(rebased[..., 0], reference[..., 0] - 10.0)
    assert torch.allclose(rebased[:, -1, :, 1], reference[:, -1, :, 1])


def test_fixed_motion_noise_is_exactly_reproducible():
    model = FactualDynamicsModel().eval()
    arguments = inputs()
    scene = torch.randn(2, 16, generator=torch.Generator().manual_seed(7))
    agent = torch.randn(2, 7, 16, generator=torch.Generator().manual_seed(8))
    first = model(
        *arguments,
        scene_standard_normal=scene,
        agent_standard_normal=agent,
    )
    second = model(
        *arguments,
        scene_standard_normal=scene,
        agent_standard_normal=agent,
    )
    torch.testing.assert_close(first.actions, second.actions)
    changed = model(
        *arguments,
        scene_standard_normal=scene + 1.0,
        agent_standard_normal=agent + 1.0,
    )
    assert not torch.equal(first.agent_latent, changed.agent_latent)
    assert not torch.equal(first.actions, changed.actions)
    torch.testing.assert_close(first.mean, changed.mean)


def test_interaction_tokens_exclude_the_soft_reference_and_random_intentions():
    model = FactualDynamicsModel().eval()
    values = list(inputs(batch=1))
    first = model(*values, deterministic=True)
    values[4] = values[4] + 50.0
    changed_plan = model(*values, deterministic=True)
    torch.testing.assert_close(
        first.observation_tokens, changed_plan.observation_tokens
    )
    changed_intention = model(
        *values,
        scene_standard_normal=torch.ones(1, model.cfg.scene_latent_dim),
        agent_standard_normal=torch.ones(1, 7, model.cfg.agent_latent_dim),
        deterministic=False,
    )
    torch.testing.assert_close(
        first.observation_tokens, changed_intention.observation_tokens
    )
    assert first.observation_tokens.shape == (1, 7, model.cfg.hidden_dim)
    values[2] = values[2].clone()
    values[2][:, 2, 2] += 8.0
    changed_state = model(*values, deterministic=True)
    assert not torch.equal(first.observation_tokens, changed_state.observation_tokens)


def test_graph_coupled_latent_is_persistent_replayable_and_observed_state_conditioned():
    cfg = WorldModelConfig(graph_coupled_latent_enabled=True)
    model = FactualDynamicsModel(cfg).eval()
    values = list(inputs(batch=1))
    scene = torch.randn(1, 16, generator=torch.Generator().manual_seed(17))
    agent = torch.randn(1, 7, 16, generator=torch.Generator().manual_seed(18))
    first = model(*values, scene_standard_normal=scene, agent_standard_normal=agent)
    repeated = model(*values, scene_standard_normal=scene, agent_standard_normal=agent)
    torch.testing.assert_close(first.agent_noise_state, repeated.agent_noise_state)
    assert torch.isfinite(first.agent_latent_log_prob).all()
    assert torch.count_nonzero(first.graph_latent_message[:, 1:]) > 0

    # At the next response boundary, changing another realized vehicle state
    # changes the target driver's transition through the graph, not a future
    # ego plan.  Use the same innovations and latent snapshot in both worlds.
    changed = list(values)
    changed[2] = values[2].clone()
    changed[2][:, 1, 0] += 12.0
    baseline_next = model(
        *values,
        filter_state=first.filter_state,
        previous_current=values[2],
        slow_scene=first.slow_scene,
        slow_scene_noise=first.slow_scene_noise,
        agent_noise_state=first.agent_noise_state,
        response_index=1,
        scene_standard_normal=scene,
        agent_standard_normal=agent,
    )
    changed_next = model(
        *changed,
        filter_state=first.filter_state,
        previous_current=values[2],
        slow_scene=first.slow_scene,
        slow_scene_noise=first.slow_scene_noise,
        agent_noise_state=first.agent_noise_state,
        response_index=1,
        scene_standard_normal=scene,
        agent_standard_normal=agent,
    )
    assert not torch.equal(
        baseline_next.agent_noise_state[:, 2], changed_next.agent_noise_state[:, 2]
    )


def test_causal_response_field_is_inert_without_an_observed_intervention():
    base_cfg = WorldModelConfig(intervention_adapter_enabled=True)
    field_cfg = WorldModelConfig(
        intervention_adapter_enabled=True,
        causal_response_field_enabled=True,
        graph_coupled_latent_enabled=True,
    )
    baseline = FactualDynamicsModel(base_cfg).eval()
    upgraded = FactualDynamicsModel(field_cfg).eval()
    upgraded.load_state_dict(baseline.state_dict(), strict=False)
    values = inputs(batch=1)
    committed = torch.zeros(1, 5, 2)
    reference = baseline(*values, committed_ego_controls=committed, deterministic=True)
    result = upgraded(*values, committed_ego_controls=committed, deterministic=True)
    torch.testing.assert_close(result.mean, reference.mean)


def test_behavior_mode_decoder_changes_only_stochastic_response_branches():
    cfg = WorldModelConfig(
        graph_coupled_latent_enabled=True,
        behavior_mode_decoder_enabled=True,
    )
    model = FactualDynamicsModel(cfg).eval()
    # Make the zero-initialized, trainable mode head observable without
    # changing its bounded contract.
    with torch.no_grad():
        model.decoder.behavior_mode[-1].weight.fill_(0.05)
    values = inputs(batch=1)
    noise = torch.ones(1, 7, 16)
    stochastic = model(*values, agent_standard_normal=noise)
    deterministic = model(*values, agent_standard_normal=noise, deterministic=True)
    assert not torch.equal(stochastic.actions, deterministic.actions)
    assert (stochastic.actions[..., 0] <= 4.0).all()
    assert (stochastic.actions[..., 0] >= -8.0).all()


def test_observed_ego_action_drives_local_response_in_the_same_direction():
    model = FactualDynamicsModel().eval()
    values = list(inputs(batch=1))
    values[0][:, :, 1, 0] = -10.0
    values[2][:, 1, 0] = -10.0
    values[0][:, :, 2, 0] = -10.0
    values[2][:, 2, 0] = -10.0
    values[0][:, :, 2, 1] = 10.0
    values[2][:, 2, 1] = 10.0
    baseline = model(*values, deterministic=True)
    braking = list(values)
    braking[0] = values[0].clone()
    braking[2] = values[2].clone()
    braking[0][:, :, 0, 4] = -2.0
    braking[2][:, 0, 4] = -2.0
    response = model(*braking, deterministic=True)
    near = (response.mean[:, :, 0, 0] - baseline.mean[:, :, 0, 0]).abs().mean()
    far = (response.mean[:, :, 1, 0] - baseline.mean[:, :, 1, 0]).abs().mean()
    assert (response.mean[:, :, 0, 0] < baseline.mean[:, :, 0, 0]).all()
    assert far < 0.2 * near


def test_intervention_adapter_is_exactly_inert_on_natural_history():
    baseline = FactualDynamicsModel().eval()
    adapted = FactualDynamicsModel(
        WorldModelConfig(intervention_adapter_enabled=True)
    ).eval()
    adapted.load_state_dict(baseline.state_dict())
    values = list(inputs(batch=1))
    committed = torch.zeros(1, 5, 2)
    unchanged = adapted(*values, committed_ego_controls=committed, deterministic=True)
    reference = baseline(*values, deterministic=True)
    torch.testing.assert_close(unchanged.mean, reference.mean)

    # A large control deviation from the observed history activates only the
    # local follower correction and does not require future ego input.
    values[0][:, :, 1, 0] = -10.0
    values[2][:, 1, 0] = -10.0
    values[0][:, :, 2, 0] = -10.0
    values[2][:, 2, 0] = -10.0
    values[0][:, :, 2, 1] = 10.0
    values[2][:, 2, 1] = 10.0
    values[2][:, 0, 4] = -2.0
    committed[:, -1, 0] = -2.0
    triggered = adapted(*values, committed_ego_controls=committed, deterministic=True)
    unadapted = baseline(*values, deterministic=True)
    near = triggered.mean[:, :, 0, 0] - unadapted.mean[:, :, 0, 0]
    far = triggered.mean[:, :, 1, 0] - unadapted.mean[:, :, 1, 0]
    assert (near < 0).all()
    assert far.abs().mean() < 0.2 * near.abs().mean()


def test_zero_residual_decoder_tracks_physical_soft_reference():
    model = FactualDynamicsModel().eval()
    values = inputs()
    result = model(*values, deterministic=True)
    torch.testing.assert_close(
        result.states[..., :2],
        values[4][:, :1],
        atol=1.0e-5,
        rtol=1.0e-5,
    )


def test_all_supported_history_lengths_and_invalid_slot_mask():
    model = FactualDynamicsModel().eval()
    values = inputs()
    for frames in (5, 10, 15, 25):
        arguments = (
            values[0][:, -frames:],
            values[1][:, -frames:],
            *values[2:],
        )
        current_valid = arguments[3].clone()
        current_valid[:, -1] = False
        arguments = (*arguments[:3], current_valid, *arguments[4:])
        result = model(*arguments, deterministic=True)
        assert result.actions.shape == (2, 1, 6, 2)
        assert torch.count_nonzero(result.actions[:, :, -1]) == 0


def test_empty_background_scene_is_finite_and_stationary():
    model = FactualDynamicsModel().eval()
    values = list(inputs())
    values[1][..., 1:] = False
    values[3][..., 1:] = False
    result = model(*values, deterministic=True)
    assert torch.isfinite(result.actions).all()
    assert torch.count_nonzero(result.actions) == 0


def test_conditional_response_targets_override_only_matching_valid_slots():
    global_bounds = torch.tensor(((0.04, 0.40), (0.03, 0.33))).numpy()
    # ego x=20 m, background x=0 m, equal velocity and zero acceleration.
    key = (1, 10, 5, 16)
    calibrator = NaturalResponseCalibrator(
        global_bounds,
        (
            {key: torch.tensor((0.10, 0.20)).numpy()},
            {key: torch.tensor((0.11, 0.21)).numpy()},
        ),
    )
    state = torch.zeros(7, 6)
    state[0, 0] = 20.0
    state[2, 0] = 100.0
    present = torch.ones(7, dtype=torch.bool)
    bounds = calibrator.bounds_for(state.numpy(), present.numpy())
    torch.testing.assert_close(
        torch.from_numpy(bounds[:, 0]), torch.tensor(((0.10, 0.20), (0.11, 0.21)))
    )
    torch.testing.assert_close(
        torch.from_numpy(bounds[:, 1]), torch.from_numpy(global_bounds)
    )
    present[-1] = False
    invalid = calibrator.bounds_for(state.numpy(), present.numpy())
    assert not invalid[:, -1].any()


def test_dose_sensitivity_targets_are_split_global_and_mask_invalid_slots():
    global_bounds = torch.tensor(((0.04, 0.40), (0.03, 0.33))).numpy()
    sensitivity = torch.tensor(((0.10, 0.20), (0.05, 0.15))).numpy()
    calibrator = NaturalResponseCalibrator(global_bounds, ({}, {}), sensitivity)
    state = torch.zeros(7, 6)
    present = torch.ones(7, dtype=torch.bool)
    values = calibrator.sensitivity_bounds_for(state.numpy(), present.numpy())
    torch.testing.assert_close(
        torch.from_numpy(values[:, 0]), torch.from_numpy(sensitivity)
    )
    present[-1] = False
    invalid = calibrator.sensitivity_bounds_for(state.numpy(), present.numpy())
    assert not invalid[:, -1].any()


def test_snapshot_restore_replays_identical_branch():
    model = FactualDynamicsModel().eval()
    world = highway_world(model, 19)
    ego_actions = torch.zeros(1, 2)
    world.advance_response(ego_actions)
    snapshot = world.snapshot()
    first = world.advance_response(ego_actions)
    world.restore(snapshot)
    second = world.advance_response(ego_actions)
    torch.testing.assert_close(
        first["agent_state_frames"], second["agent_state_frames"]
    )
    torch.testing.assert_close(
        first["background_actions"], second["background_actions"]
    )


def test_soft_plan_never_bypasses_physical_action_bounds():
    model = FactualDynamicsModel().eval()
    values = list(inputs())
    zero = values[4].clone().zero_()
    extreme = values[4].clone()
    extreme[..., 0] = 10_000.0
    for reference in (zero, extreme):
        arguments = (*values[:4], reference, *values[5:])
        result = model(*arguments, deterministic=True)
        assert (result.actions[..., 0] >= -8.0).all()
        assert (result.actions[..., 0] <= 4.0).all()
        assert (result.actions[..., 1].abs() <= 0.6).all()


def test_distribution_diagnostics_are_fixed_bin_and_mask_aware():
    values = torch.zeros(2, 3, 7, 6)
    values[..., 0, 0] = 20.0
    values[..., 1, 0] = 0.0
    values[..., 2, 0] = -10.0
    active = torch.tensor(((True, True, False, False, False, False),) * 2).numpy()
    states = values.numpy()
    nearest = _nearest_object_distance(states, active)
    assert nearest.shape == (2 * 3 * 3,)
    assert (nearest >= 10.0).all()
    assert not _collision_indicator(states, active).any()
    histogram = _histogram_summary(
        torch.tensor((0.0, 0.5, 1.0)).numpy(),
        torch.tensor((0.25, 0.75)).numpy(),
        bounds=(0.0, 1.0),
        bins=4,
    )
    assert len(histogram["bin_edges"]) == 5
    assert abs(sum(histogram["real_probability"]) - 1.0) < 1.0e-8
    assert abs(sum(histogram["generated_probability"]) - 1.0) < 1.0e-8


def test_temporal_factual_metrics_preserve_the_horizon_axis():
    target = torch.zeros(1, 3, 7, 6).numpy()
    generated = target.copy()
    generated[:, :, 1, 0] = torch.tensor((0.04, 0.08, 0.12)).numpy()
    active = torch.tensor(((True, False, False, False, False, False),)).numpy()
    temporal = temporal_factual_metrics(generated, target, active)
    assert temporal["time_s"] == [0.04, 0.08, 0.12]
    assert temporal["ADE_m"] == pytest.approx([0.04, 0.08, 0.12])


def test_future_ads_action_cannot_modify_already_committed_response():
    model = FactualDynamicsModel().eval()
    baseline = highway_world(model, 5)
    intervened = highway_world(model, 5)
    baseline_step = baseline.advance_response(torch.zeros(1, 2))
    braking = torch.zeros(1, 2)
    braking[:, 0] = -4.0
    intervention_step = intervened.advance_response(braking)
    torch.testing.assert_close(
        baseline_step["background_actions"],
        intervention_step["background_actions"],
    )
    torch.testing.assert_close(
        baseline_step["agent_state_frames"][:, :, 1:],
        intervention_step["agent_state_frames"][:, :, 1:],
    )
    assert not torch.equal(
        baseline_step["agent_state_frames"][:, :, 0],
        intervention_step["agent_state_frames"][:, :, 0],
    )
    assert baseline_step["reference_index"] == 1
    assert intervention_step["reference_index"] == 1


def test_25hz_world_commits_exactly_one_frame_per_response():
    model = FactualDynamicsModel().eval()
    world = highway_world(model, 17)
    first = world.advance_response(torch.zeros(1, 2))
    second = world.advance_response(torch.zeros(1, 2))
    assert first["agent_state_frames"].shape == (1, 1, 7, 6)
    assert second["agent_state_frames"].shape == (1, 1, 7, 6)
    assert first["reference_index"] == 1
    assert second["reference_index"] == 2


def test_windowed_jerk_averages_before_differentiating():
    actions = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1, 1)
    actions = torch.cat((actions, actions), dim=-1).numpy()
    jerk = _windowed_jerk(actions, time_axis=1, window_frames=5, dt_s=0.1)
    assert jerk.shape == (1, 1, 1, 2)
    assert float(jerk[0, 0, 0, 0]) == 10.0


@pytest.mark.parametrize("adapter_enabled", [False, True])
def test_closed_loop_intervention_loss_is_finite_and_differentiable(adapter_enabled):
    model = FactualDynamicsModel(
        WorldModelConfig(
            intervention_adapter_enabled=adapter_enabled,
            lateral_response_adapter_enabled=adapter_enabled,
            graph_coupled_latent_enabled=True,
            behavior_mode_decoder_enabled=True,
        )
    )
    model.set_matched_response_bounds(torch.tensor(((0.0375, 0.4), (0.025, 0.325))))
    values = inputs(batch=2)
    target_states = values[2][:, None, 1:].expand(-1, 1, -1, -1).clone()
    target_states[..., 0] = values[4][:, :1, :, 0]
    batch = {
        "history": values[0],
        "history_valid": values[1],
        "current": values[2],
        "current_valid": values[3],
        "target_actions": torch.zeros(2, 1, 6, 2),
        "target_states": target_states,
        "closed_target_states": target_states.expand(-1, 25, -1, -1).clone(),
        "closed_target_actions": torch.zeros(2, 25, 6, 2),
        "closed_ego_actions": torch.zeros(2, 25, 2),
        "previous_actions": torch.zeros(2, 6, 2),
        "soft_reference": values[4],
        "reference_base": values[5],
        "map_polylines": values[6],
        "map_polyline_valid": values[7],
    }
    terms = training_losses(
        model,
        batch,
        torch.zeros(2, 16),
        torch.zeros(2, 7, 16),
    )
    assert torch.isfinite(terms["loss"])
    terms["loss"].backward()
    assert model.decoder.response_gain.weight.grad is not None


def test_base_loss_uses_a_real_deterministic_forward_and_disables_energy():
    model = FactualDynamicsModel(WorldModelConfig(graph_coupled_latent_enabled=True))
    values = inputs(batch=2)
    target_states = values[2][:, None, 1:].expand(-1, 1, -1, -1).clone()
    batch = {
        "history": values[0],
        "history_valid": values[1],
        "current": values[2],
        "current_valid": values[3],
        "target_actions": torch.zeros(2, 1, 6, 2),
        "target_states": target_states,
        "closed_target_states": target_states.expand(-1, 25, -1, -1).clone(),
        "closed_target_actions": torch.zeros(2, 25, 6, 2),
        "closed_ego_actions": torch.zeros(2, 25, 2),
        "previous_actions": torch.zeros(2, 6, 2),
        "soft_reference": values[4],
        "reference_base": values[5],
        "map_polylines": values[6],
        "map_polyline_valid": values[7],
    }
    terms = training_losses(
        model,
        batch,
        torch.randn(2, 16),
        torch.randn(2, 7, 16),
        deterministic_forward=True,
    )
    assert float(terms["energy"]) == 0.0
    terms["loss"].backward()
    assert model.decoder.scene_innovation.weight.grad is None
    assert model.decoder.agent_innovation.weight.grad is None


def test_base_stage_trainable_freezes_stochastic_heads_only():
    cfg = WorldModelConfig(
        graph_coupled_latent_enabled=True,
        behavior_mode_decoder_enabled=True,
    )
    model = FactualDynamicsModel(cfg)
    stage = "base"
    stage_mode = _apply_stage_trainable(
        model,
        stage=stage,
        stage_config={"trainable": "backbone"},
    )
    assert stage_mode == "backbone"
    frozen = [
        "latent_transition.",
        "decoder.behavior_mode.",
        "decoder.scene_innovation.",
        "decoder.agent_innovation.",
    ]
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in frozen):
            assert not parameter.requires_grad, f"base should freeze {name}"
        else:
            assert parameter.requires_grad, f"base should train {name}"


def test_stochastic_heads_stage_trainable_only_updates_stochastic_heads():
    cfg = WorldModelConfig(
        graph_coupled_latent_enabled=True,
        behavior_mode_decoder_enabled=True,
    )
    model = FactualDynamicsModel(cfg)
    stage = "stochastic_heads"
    stage_mode = _apply_stage_trainable(
        model,
        stage=stage,
        stage_config={
            "trainable": [
                "latent_transition.",
                "decoder.behavior_mode.",
                "decoder.scene_innovation.",
                "decoder.agent_innovation.",
            ]
        },
    )
    assert (
        stage_mode
        == "prefixes:latent_transition.,decoder.behavior_mode.,decoder.scene_innovation.,decoder.agent_innovation."
    )
    frozen = [
        "latent_transition.",
        "decoder.behavior_mode.",
        "decoder.scene_innovation.",
        "decoder.agent_innovation.",
    ]
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in frozen):
            assert parameter.requires_grad, f"stochastic stage should train {name}"
        else:
            assert not parameter.requires_grad, f"stochastic stage should freeze {name}"

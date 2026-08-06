"""Causal and isolation regression tests for the HiQR-v2 contract."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from normalizing_flow.src.features import EGO_FEATURES, SLOT_NAMES, slot_feature_index, trajectory_feature_index
from world_model.src.hiqr.environment import HiQRFlowStartMetadata, HiQRWorldRandomness
from world_model.src.hiqr_v2 import HiQRV2Config, HiQRV2WorldModel
from world_model.src.hiqr_v2.data import cohort_manifest, stable_slot_mask
from world_model.src.hiqr_v2.diagnostics import (
    continuation_decision,
    decision_from_bootstrap,
)
from world_model.src.hiqr_v2.environment import HiQRV2WorldModelEnvironment
from world_model.src.hiqr_v2.evaluation import (
    _aggregate_risk,
    _bootstrap_difference,
    _summary_from_rollout,
)
from world_model.src.hiqr_v2.train import (
    _epoch_shuffle_seed,
    _load_state,
    _save_state,
    _set_stage_learning_rate,
    load_hiqr_v2_checkpoint,
    require_canonical_hiqr_v2_checkpoint,
)


def _model(**overrides) -> HiQRV2WorldModel:
    torch.manual_seed(7)
    cfg = HiQRV2Config(
        hidden_dim=32,
        scene_latent_dim=8,
        agent_residual_dim=8,
        temporal_layers=1,
        attention_layers=1,
        decoder_layers=1,
        num_heads=4,
        dropout=0.0,
        **overrides,
    )
    return HiQRV2WorldModel(cfg).eval()


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    frames = 174
    states = torch.randn(batch_size, frames, 7, 6)
    states[..., 2] += 20.0
    valid = torch.ones(batch_size, frames, 7, dtype=torch.bool)
    valid[:, :, -1] = False
    maps = torch.zeros(batch_size, 2, 4, 6)
    maps[..., 0] = torch.arange(4)
    maps[:, 1, :, 1] = 3.6
    maps[..., 2] = 1.0
    maps[..., 4] = 3.6
    return {
        "agent_states": states,
        "agent_valid": valid,
        "ego_index": torch.zeros(batch_size, dtype=torch.long),
        "map_polylines": maps,
        "map_polyline_valid": torch.ones(batch_size, 2, 4, dtype=torch.bool),
        "actions_highd": torch.randn(batch_size, 149, 6, 2),
        "behavior_anchor_raw": torch.randn(batch_size, 6, 6),
        "behavior_anchor_valid": valid[:, 24, 1:].clone(),
    }


def _metadata() -> HiQRFlowStartMetadata:
    maps = np.zeros((2, 4, 6), np.float32)
    maps[..., 0] = np.arange(4)
    maps[1, :, 1] = 3.6
    maps[..., 2] = 1.0
    maps[..., 4] = 3.6
    slots = np.ones(6, bool)
    event = np.zeros(12, np.float32)
    event[:6] = slots
    event[6] = 1.0
    return HiQRFlowStartMetadata(
        slot_valid=slots,
        map_polylines=maps,
        map_polyline_valid=np.ones((2, 4), bool),
        primary_slot_index=0,
        event_structure=event,
        mask_pattern=63,
        event_structure_id=0,
        event_structure_log_prob=-0.2,
        conditional_log_prob=-1.3,
        log_prob=-1.5,
        flow_checkpoint_sha256="a" * 64,
        flow_schema_sha256="b" * 64,
        sampling_seed=41,
        sampling_temperature=1.0,
        sampling_rejection={
            "reject_invalid": True,
            "max_rounds": 8,
            "oversample_factor": 1,
            "min_draw": 1,
            "num_rejected": 0,
            "rejection_rate": 0.0,
        },
    )


def _flow_start_from_scene(
    current: torch.Tensor, raw_b0: torch.Tensor, valid: torch.Tensor
) -> np.ndarray:
    """Encode one ego-relative test scene in the frozen C0+B0 layout."""
    assert current.shape == (1, 7, 6)
    assert torch.equal(current[:, 0, :2], torch.zeros_like(current[:, 0, :2]))
    feature = np.zeros(76, np.float32)
    feature[: len(EGO_FEATURES)] = current[0, 0, 2:].cpu().numpy()
    for index, slot in enumerate(SLOT_NAMES):
        if not valid[0, index + 1]:
            continue
        background = current[0, index + 1]
        values = (
            background[0],
            background[1],
            background[2] - current[0, 0, 2],
            background[3] - current[0, 0, 3],
            background[4],
            background[5],
        )
        for name, value in zip(
            ("rel_x_m", "rel_y_left_m", "rel_vx_mps", "rel_vy_left_mps", "other_ax_mps2", "other_ay_left_mps2"),
            values,
            strict=True,
        ):
            feature[slot_feature_index(slot, name)] = float(value)
        for column, value in enumerate(raw_b0[0, index]):
            feature[trajectory_feature_index(slot, (
                "delta_vx_1s_mps", "delta_vy_left_1s_mps", "mean_ax_1s_mps2",
                "min_ax_1s_mps2", "final_ax_1s_mps2", "mean_ay_left_1s_mps2",
            )[column])] = float(value)
    return feature


def test_prior_rollout_is_insensitive_to_future_labels() -> None:
    model, batch = _model(), _batch(1)
    changed = copy.deepcopy(batch)
    changed["agent_states"][:, 25:, 1:] += 10_000.0
    changed["actions_highd"] += 10_000.0
    left = model.rollout_reconstruction(batch, response_steps=3, deterministic=True)
    right = model.rollout_reconstruction(changed, response_steps=3, deterministic=True)
    assert torch.equal(left["predicted_states"], right["predicted_states"])


def test_plan_step_and_rollout_use_the_same_response_core() -> None:
    model, batch = _model(), _batch(1)
    current = batch["agent_states"][:, model.cfg.anchor_state_index]
    current_valid = batch["agent_valid"][:, model.cfg.anchor_state_index]
    ego_mask = model._ego_mask(batch)
    filter_state = model.initialize_start(
        current,
        current_valid,
        ego_mask,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        batch["behavior_anchor_raw"],
        batch["behavior_anchor_valid"],
    )
    online = model.plan_step(
        None,
        None,
        current,
        current_valid,
        ego_mask,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        filter_state=filter_state,
        deterministic=True,
    )
    offline = model.rollout_reconstruction(batch, response_steps=1, deterministic=True)

    assert torch.equal(
        online["background_future_actions"],
        offline["background_future_actions"][:, 0],
    )
    assert torch.equal(online["scene_latent"], offline["scene_latent"][:, 0])
    assert torch.equal(online["agent_residual"], offline["agent_residual"][:, 0])


def test_offline_rollout_and_online_environment_are_strictly_consistent() -> None:
    """Logged ego controls must produce the same prior world in both paths."""
    model, batch = _model(), _batch(1)
    # Use a fixed nonzero plan to avoid amplifying sub-ULP Flow adapter
    # rounding through an untrained random decoder.  This still exercises all
    # 149 physical ticks and response-boundary continuation plumbing.
    torch.nn.init.zeros_(model.decoder.action.weight)
    model.decoder.action.bias.data.copy_(torch.tensor((0.2, 0.1)))
    anchor = model.cfg.anchor_state_index
    # Use exactly representable C0 values so the test detects a path mismatch,
    # not harmless add/subtract rounding in Flow's relative-velocity adapter.
    batch["agent_states"][:, anchor] = 0.0
    batch["agent_states"][:, anchor, :, 2] = 20.0
    batch["agent_states"][:, anchor, 1:, 0] = torch.arange(1, 7) * 5.0
    ticks = torch.arange(150, dtype=batch["agent_states"].dtype)
    batch["agent_states"][:, anchor : anchor + 150, 0] = 0.0
    batch["agent_states"][:, anchor : anchor + 150, 0, 0] = ticks * 0.8
    batch["agent_states"][:, anchor : anchor + 150, 0, 2] = 20.0
    current, valid = batch["agent_states"][:, anchor], batch["agent_valid"][:, anchor]
    feature = _flow_start_from_scene(current, batch["behavior_anchor_raw"], valid)
    reconstructed, _, _ = model.flow_condition_to_scene(
        torch.from_numpy(feature)[None], valid[:, 1:]
    )
    batch["agent_states"][:, anchor] = reconstructed
    current = batch["agent_states"][:, anchor]
    metadata = _metadata()
    event = np.zeros(12, np.float32)
    event[:6] = valid[0, 1:].cpu().numpy()
    event[6] = 1.0
    metadata = HiQRFlowStartMetadata(
        **{
            **metadata.__dict__,
            "slot_valid": valid[0, 1:].cpu().numpy(),
            "event_structure": event,
            "mask_pattern": 31,
        }
    )
    offline = model.rollout_reconstruction(batch, deterministic=True)
    controls = offline["applied_ego_controls"][0].cpu().numpy()
    control_valid = offline["applied_ego_control_valid"][0].cpu().numpy()
    environment = HiQRV2WorldModelEnvironment(model)
    environment.reset_from_flow(feature[:40], feature[40:].reshape(6, 6), metadata)
    online = []
    for action, ego_valid in zip(controls, control_valid, strict=True):
        online.append(environment.step(action, bool(ego_valid))["agent_states"])
    online_states = np.stack(online)
    offline_states = offline["predicted_states"][0].detach().cpu().numpy()
    error = np.abs(online_states - offline_states)
    mismatch = np.flatnonzero(error.max(axis=(1, 2)) > 1.0e-5)
    assert error.max() < 1.0e-5, (
        "first mismatch at 25 Hz tick "
        f"{None if not len(mismatch) else int(mismatch[0])}"
    )


def test_logged_ego_controls_are_exogenous_to_each_response_plan() -> None:
    model, batch = _model(), _batch(1)
    changed = copy.deepcopy(batch)
    changed["agent_states"][:, 25:, 0, 2] += 5.0
    left = model.rollout_reconstruction(batch, response_steps=2, deterministic=True)
    right = model.rollout_reconstruction(changed, response_steps=2, deterministic=True)
    assert torch.equal(
        left["background_future_actions"][:, 0],
        right["background_future_actions"][:, 0],
    )
    assert torch.equal(
        left["predicted_states"][:, :5, 1:],
        right["predicted_states"][:, :5, 1:],
    )
    assert not torch.equal(
        left["predicted_states"][:, :5, 0],
        right["predicted_states"][:, :5, 0],
    )


def test_logged_ego_control_recovery_matches_speed_and_heading_change() -> None:
    model = _model()
    source = torch.zeros(1, 1, 6)
    target = torch.zeros_like(source)
    source[..., 2] = 10.0
    target_speed = 10.04
    target_heading = 0.004
    target[..., 2] = target_speed * torch.cos(torch.tensor(target_heading))
    target[..., 3] = target_speed * torch.sin(torch.tensor(target_heading))
    controls = model._logged_ego_controls(
        source, target, torch.ones(1, 1, dtype=torch.bool)
    )
    assert torch.allclose(controls, torch.tensor([[[1.0, 0.1]]]), atol=1.0e-4)


def test_posterior_cannot_mutate_filter_state() -> None:
    model, batch = _model(), _batch(1)
    current, valid = batch["agent_states"][:, 24], batch["agent_valid"][:, 24]
    ego = torch.zeros_like(valid)
    ego[:, 0] = True
    state = model.initialize_start(
        current,
        valid,
        ego,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        batch["behavior_anchor_raw"],
        batch["behavior_anchor_valid"],
    )
    agents, scene, _, _ = model.encoder(
        None,
        None,
        current,
        valid,
        ego,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        mode="start",
    )
    observed = model.filter.observe(state, agents, scene, current, None, valid)
    before_global, before_agent = (
        observed.global_hidden.clone(),
        observed.agent_hidden.clone(),
    )
    prior_scene = model.filter.prior_scene(observed, scene)
    model.filter.posterior(
        observed,
        agents,
        scene,
        current,
        valid,
        batch["agent_states"][:, 25:50],
        batch["agent_valid"][:, 25:50],
        prior_scene,
    )
    assert torch.equal(observed.global_hidden, before_global)
    assert torch.equal(observed.agent_hidden, before_agent)


def test_posterior_diagnostics_do_not_change_prior_trajectory() -> None:
    model, batch = _model(), _batch(1)
    prior = model.rollout_reconstruction(batch, response_steps=3, deterministic=True)
    diagnostic = model.diagnostic_rollout(batch, response_steps=3)
    assert torch.equal(prior["predicted_states"], diagnostic["predicted_states"])
    assert diagnostic["terms"]["posterior_position"].item() > 0.0


def test_scene_mode_is_held_for_five_responses_and_then_refreshes() -> None:
    rollout = _model().rollout_reconstruction(
        _batch(1), response_steps=6, deterministic=True
    )
    scene = rollout["scene_latent"]
    assert all(torch.equal(scene[:, 0], scene[:, index]) for index in range(1, 5))
    assert not torch.equal(scene[:, 4], scene[:, 5])


def test_posterior_scene_is_local_and_held_within_slow_scene_group() -> None:
    model, batch = _model(), _batch(1)
    current, valid = batch["agent_states"][:, 24], batch["agent_valid"][:, 24]
    ego = torch.zeros_like(valid)
    ego[:, 0] = True
    agents, scene, _, _ = model.encoder(
        None,
        None,
        current,
        valid,
        ego,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        mode="start",
    )
    state = model.initialize_start(
        current,
        valid,
        ego,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        batch["behavior_anchor_raw"],
        batch["behavior_anchor_valid"],
    )
    state = model.filter.observe(state, agents, scene, current, None, valid)
    prior_scene = model.filter.prior_scene(state, scene)
    future = batch["agent_states"][:, 25:50]
    future_valid = batch["agent_valid"][:, 25:50]
    first_scene, first_agent, first_terms = model.filter.posterior(
        state,
        agents,
        scene,
        current,
        valid,
        future,
        future_valid,
        prior_scene,
    )
    conditional_prior_agent = model.filter.prior_agents(state, agents, first_scene)[0]
    background = valid.clone()
    background[:, 0] = False
    expected_distillation = (first_scene - prior_scene[0]).square().mean()
    expected_distillation = expected_distillation + (
        (first_agent - conditional_prior_agent).square().mean(dim=-1)
        * background.float()
    ).sum() / background.float().sum().clamp_min(1.0)
    assert torch.allclose(first_terms["prior_distillation"], expected_distillation)
    posterior_future = future.clone()
    posterior_valid = future_valid.clone()
    posterior_future[:, :, 0] = current[:, None, 0]
    posterior_valid[:, :, 0] = valid[:, None, 0]
    future_agent = model.filter.future(
        model.filter._future_features(current, posterior_future, posterior_valid)
    )
    pooled = model.filter._pool(future_agent, background)
    inferred_scene, _ = model.filter.distribution_parameters(
        model.filter.scene_posterior(
            torch.cat((scene, state.global_hidden, pooled), dim=-1)
        )
    )
    shared = torch.cat((state.global_hidden, inferred_scene), dim=-1)[:, None].expand(
        -1, agents.shape[1], -1
    )
    inferred_agent, inferred_agent_log = model.filter.distribution_parameters(
        model.filter.agent_posterior(
            torch.cat((agents, state.agent_hidden, future_agent, shared), dim=-1)
        )
    )
    conditional_prior, conditional_prior_log = model.filter.prior_agents(
        state, agents, inferred_scene
    )
    ratio = torch.exp(2.0 * (inferred_agent_log - conditional_prior_log))
    expected_agent_kl = (
        conditional_prior_log
        - inferred_agent_log
        + 0.5
        * (
            ratio
            + (inferred_agent - conditional_prior).square()
            * torch.exp(-2.0 * conditional_prior_log)
            - 1.0
        )
    ).mean(dim=-1)
    expected_agent_kl = (
        expected_agent_kl * background.float()
    ).sum() / background.float().sum().clamp_min(1.0)
    assert torch.allclose(first_terms["agent_kl"], expected_agent_kl)
    held_scene, _, terms = model.filter.posterior(
        state,
        agents,
        scene,
        current,
        valid,
        future + 1000.0,
        future_valid,
        prior_scene,
        fixed_scene_latent=first_scene,
    )
    assert torch.equal(held_scene, first_scene)
    assert terms["scene_kl"].item() == 0.0
    assert terms["posterior_scene_std"].item() == 0.0


def test_start_uses_initialized_filter_without_duplicate_observe() -> None:
    model, batch = _model(), _batch(1)
    current, valid = batch["agent_states"][:, 24], batch["agent_valid"][:, 24]
    ego = torch.zeros_like(valid)
    ego[:, 0] = True
    initial = model.initialize_start(
        current,
        valid,
        ego,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        batch["behavior_anchor_raw"],
        batch["behavior_anchor_valid"],
    )
    output = model.plan_step(
        None,
        None,
        current,
        valid,
        ego,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        filter_state=initial,
        deterministic=True,
    )
    assert torch.equal(output["filter_state"].global_hidden, initial.global_hidden)
    assert torch.equal(output["filter_state"].agent_hidden, initial.agent_hidden)


def test_decoder_enforces_five_fifteen_five_continuation() -> None:
    model, batch = _model(), _batch(1)
    current, valid = batch["agent_states"][:, 24], batch["agent_valid"][:, 24]
    ego = torch.zeros_like(valid)
    ego[:, 0] = True
    agents, scene, _, _ = model.encoder(
        None,
        None,
        current,
        valid,
        ego,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        mode="start",
    )
    state = model.initialize_start(
        current,
        valid,
        ego,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        batch["behavior_anchor_raw"],
        batch["behavior_anchor_valid"],
    )
    state = model.filter.observe(state, agents, scene, current, None, valid)
    g = model.filter.prior_scene(state, scene)[0]
    z = model.filter.prior_agents(state, agents, g)[0]
    previous = 0.1 * torch.randn(1, 25, 6, 2)
    prior_states = model._integrate_background_actions(current, previous, valid)
    expected_ego = model._expected_ego(current)
    realized = current.clone()
    realized[:, 0] = expected_ego[:, 4, 0]
    realized[:, 1:] = prior_states[:, 4]
    output = model.decoder(
        agents,
        state.global_hidden,
        state.agent_hidden,
        g,
        z,
        realized,
        valid,
        previous,
        prior_states,
        expected_ego,
    )
    assert torch.equal(
        output["background_future_actions"][:, :5],
        previous[:, 5:10] * valid[:, None, 1:, None].float(),
    )
    assert torch.equal(
        output["background_future_action_masks"]["carried"][:, :20],
        valid[:, None, 1:].expand(-1, 20, -1),
    )
    assert not output["background_future_action_masks"]["carried"][:, 20:].any()
    replan = _model(continuation_mode="full_replan")
    output = replan.decoder(
        agents,
        state.global_hidden,
        state.agent_hidden,
        g,
        z,
        realized,
        valid,
        previous,
        prior_states,
        expected_ego,
    )
    assert not output["background_future_action_masks"]["carried"].any()


def test_decoder_invalidates_hard_carry_after_large_ads_deviation() -> None:
    model, batch = _model(), _batch(1)
    current, valid = batch["agent_states"][:, 24], batch["agent_valid"][:, 24]
    ego = torch.zeros_like(valid)
    ego[:, 0] = True
    agents, scene, _, _ = model.encoder(
        None,
        None,
        current,
        valid,
        ego,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        mode="start",
    )
    state = model.initialize_start(
        current,
        valid,
        ego,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        batch["behavior_anchor_raw"],
        batch["behavior_anchor_valid"],
    )
    state = model.filter.observe(state, agents, scene, current, None, valid)
    g = model.filter.prior_scene(state, scene)[0]
    z = model.filter.prior_agents(state, agents, g)[0]
    previous = 0.1 * torch.randn(1, 25, 6, 2)
    prior_states = model._integrate_background_actions(current, previous, valid)
    expected_ego = model._expected_ego(current)
    deviated = current.clone()
    deviated[:, 0] = expected_ego[:, 4, 0]
    deviated[:, 0, 0] += 10.0
    deviated[:, 1:] = prior_states[:, 4]
    output = model.decoder(
        agents,
        state.global_hidden,
        state.agent_hidden,
        g,
        z,
        deviated,
        valid,
        previous,
        prior_states,
        expected_ego,
    )
    assert output["background_future_action_masks"]["emergency"][:, :5].any()
    assert not torch.equal(
        output["background_future_actions"][:, :5], previous[:, 5:10]
    )


def test_b0_is_per_agent_filter_initialization() -> None:
    model, batch = _model(), _batch(1)
    current, valid = batch["agent_states"][:, 24], batch["agent_valid"][:, 24]
    ego = torch.zeros_like(valid)
    ego[:, 0] = True
    agents, scene, _, _ = model.encoder(
        None,
        None,
        current,
        valid,
        ego,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        mode="start",
    )
    baseline = model.filter.initialize(
        scene, agents, torch.zeros(1, 6, 6), torch.ones(1, 6, dtype=torch.bool)
    )
    raw = torch.zeros(1, 6, 6)
    raw[:, 2, 0] = 3.0
    changed = model.filter.initialize(
        scene, agents, raw, torch.ones(1, 6, dtype=torch.bool)
    )
    assert not torch.equal(baseline.agent_hidden[:, 3], changed.agent_hidden[:, 3])
    assert torch.equal(baseline.agent_hidden[:, 1], changed.agent_hidden[:, 1])


def test_fixed_cohort_rejects_only_c0_active_exits() -> None:
    valid = np.ones((3, 174, 7), bool)
    valid[2, 24, 2] = False
    valid[1, 80, 1] = False  # C0-active vehicle exits: excluded.
    valid[2, 80, 2] = False  # C0-inactive slot remains irrelevant.
    assert np.array_equal(
        stable_slot_mask({"agent_valid": valid}),
        np.asarray([True, False, True]),
    )


def test_fixed_cohort_rejects_c0_vehicle_that_exits_during_b0_second() -> None:
    valid = np.ones((1, 174, 7), bool)
    valid[:, 30:, 1] = False
    # Such a row has an invalid first-second B0 summary, but its C0 vehicle
    # must still disqualify the sequence.
    assert not stable_slot_mask({"agent_valid": valid})[0]


def test_cohort_manifest_audits_excluded_sequences_and_slots() -> None:
    valid = np.ones((2, 174, 7), bool)
    valid[1, 80:, 1] = False
    manifest = cohort_manifest(
        {"agent_valid": valid, "split_index": np.asarray([0, 0])}
    )
    assert manifest["excluded_sequences"] == 1
    assert manifest["exiting_background_slots"] == 1
    assert manifest["excluded_sequence_active_background_slots"] == 6


def test_environment_snapshot_restore_and_ams_branches_are_replayable() -> None:
    model, meta = _model(), _metadata()
    parent = HiQRV2WorldModelEnvironment(model)
    parent.reset_from_flow(
        np.zeros(40, np.float32),
        np.zeros((6, 6), np.float32),
        meta,
        deterministic=False,
        world_randomness=HiQRWorldRandomness(seed=13),
    )
    for _ in range(5):
        parent.step(np.zeros(2, np.float32))
    snapshot = parent.snapshot()
    first = HiQRV2WorldModelEnvironment(model)
    second = HiQRV2WorldModelEnvironment(model)
    first.branch_from_snapshot(snapshot, HiQRWorldRandomness(seed=31), level="scene")
    second.branch_from_snapshot(
        snapshot, HiQRWorldRandomness(seed=31), level="residual"
    )
    assert first.step(np.zeros(2, np.float32))["planner_updated"]
    assert second.step(np.zeros(2, np.float32))["planner_updated"]
    assert first.trace["world_randomness"]["response_innovations"][-1][
        "scene_refreshed"
    ]
    assert not second.trace["world_randomness"]["response_innovations"][-1][
        "scene_refreshed"
    ]
    assert first.trace["world_randomness"]["branch_resampling"][-1]["scene_seed"] == 31
    assert second.trace["world_randomness"]["branch_resampling"][-1]["scene_seed"] == 13
    replay = HiQRV2WorldModelEnvironment(model)
    replay.restore(snapshot)
    assert np.array_equal(
        replay.observe()["agent_states"], parent.observe()["agent_states"]
    )


def test_checkpoint_schema_is_v2_only_and_restores(tmp_path) -> None:
    model = _model()
    model.flow_schema_sha256 = "b" * 64
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    path = tmp_path / "last_hiqr_v2_training_state.pt"
    _save_state(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        stage_index=0,
        stage_epoch=1,
        global_step=2,
        best_fde=1.0,
        cohort={"stable": 1},
    )
    restored = _model()
    restored.flow_schema_sha256 = "b" * 64
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1.0e-4)
    info = _load_state(
        path,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(restored_optimizer),
        flow_schema_sha256="b" * 64,
    )
    assert info["model_type"] == model.model_type
    foreign = tmp_path / "v1.pt"
    torch.save(
        {"model_type": "hierarchical_interaction_query_refine_world_model"}, foreign
    )
    with pytest.raises(ValueError, match="not HiQR-v2"):
        load_hiqr_v2_checkpoint(foreign)


def test_resume_preserves_scheduler_lr_and_epoch_shuffle_order() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=2.5e-5)
    stage = {"learning_rate": 1.0e-4}
    _set_stage_learning_rate(optimizer, stage, completed=3)
    assert optimizer.param_groups[0]["lr"] == 2.5e-5
    _set_stage_learning_rate(optimizer, stage, completed=0)
    assert optimizer.param_groups[0]["lr"] == 1.0e-4
    assert _epoch_shuffle_seed(101, 7) == _epoch_shuffle_seed(101, 7)


def test_complete_149_transition_prior_rollout() -> None:
    rollout = _model().rollout_reconstruction(_batch(1), deterministic=True)
    assert rollout["predicted_states"].shape == (1, 149, 7, 6)


def test_risk_metrics_use_predicted_ego_for_predicted_backgrounds() -> None:
    predicted = torch.zeros(1, 1, 7, 6)
    target = torch.zeros_like(predicted)
    valid = torch.zeros(1, 1, 7, dtype=torch.bool)
    valid[:, :, :2] = True
    predicted[:, :, 0, 0] = 100.0
    predicted[:, :, 1, 0] = 101.0
    target[:, :, 1, 0] = 20.0
    _, _, risk = _summary_from_rollout(
        {
            "predicted_states": predicted,
            "target_states": target,
            "target_valid": valid,
        }
    )
    assert risk["collision_episode"].item()
    summary = _aggregate_risk([risk])
    assert summary["collision_episode_rate"] == 1.0
    assert summary["target_collision_episode_rate"] == 0.0
    assert summary["collision_pair_point_rate"] == 1.0
    assert summary["risk_variable_distribution"]["gap_m"]["available"]
    assert summary["risk_variable_summary"]["gap_m"]["target"]["count"] == 1


def test_diagnostic_scene_frequency_is_allowed_only_when_explicit() -> None:
    ablation = _model(scene_mode_responses=1)
    with pytest.raises(ValueError, match="not a canonical"):
        require_canonical_hiqr_v2_checkpoint(ablation)
    require_canonical_hiqr_v2_checkpoint(ablation, allow_diagnostic_ablation=True)


def test_bootstrap_admission_requires_significant_ten_percent_gain() -> None:
    baseline = np.ones(32)
    candidate = np.full(32, 0.8)
    comparison = _bootstrap_difference(candidate, baseline, samples=100, seed=9)
    decision = decision_from_bootstrap({"bootstrap_vs_baseline": comparison})
    assert decision["eligible_for_v2_default"]


def test_adaptive_continuation_uses_smoothness_and_safety_gate() -> None:
    baseline = {
        "deterministic_prior_mean": {
            "fde_5p96s_m": 1.0,
            "jerk": 1.0,
            "plan_discontinuity": 1.0,
            "emergency_rate": 0.0,
        },
        "interaction_metrics": {
            "collision_episode_rate": 0.0,
            "gap_mae_m": 1.0,
            "ttc_mae_s": 1.0,
            "drac_mae_mps2": 1.0,
        },
    }
    candidate = copy.deepcopy(baseline)
    candidate["deterministic_prior_mean"].update(
        {"fde_5p96s_m": 1.04, "jerk": 0.8, "plan_discontinuity": 0.5}
    )
    assert continuation_decision(candidate, baseline)["eligible_for_v2_default"]
    candidate["deterministic_prior_mean"]["fde_5p96s_m"] = 1.06
    assert not continuation_decision(candidate, baseline)["eligible_for_v2_default"]


def test_environment_runs_the_complete_149_transition_protocol() -> None:
    model, meta = _model(), _metadata()
    left, right = HiQRV2WorldModelEnvironment(model), HiQRV2WorldModelEnvironment(model)
    for environment in (left, right):
        environment.reset_from_flow(
            np.zeros(40, np.float32),
            np.zeros((6, 6), np.float32),
            meta,
            deterministic=False,
            world_randomness=HiQRWorldRandomness(seed=101),
        )
    for _ in range(149):
        left.step(np.zeros(2, np.float32))
        right.step(np.zeros(2, np.float32))
    assert left.physics_step_index == right.physics_step_index == 149
    # 149 transitions comprise 29 completed five-frame responses and the
    # four executed frames of the final (30th) response.
    assert left.response_index == right.response_index == 29
    assert np.array_equal(
        left.observe()["agent_states"], right.observe()["agent_states"]
    )

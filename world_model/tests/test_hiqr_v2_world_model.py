"""Regression tests for the compact accepted HiQR-v2 contract."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from world_model.src.hiqr.environment import HiQRFlowStartMetadata, HiQRWorldRandomness
from world_model.src.hiqr_v2 import HiQRV2Config, HiQRV2WorldModel
from world_model.src.hiqr_v2.data import cohort_manifest, stable_slot_mask
from world_model.src.hiqr_v2.environment import HiQRV2WorldModelEnvironment
from world_model.src.hiqr_v2.evaluation import (
    _aggregate_risk,
    _bootstrap_difference,
    _summary_from_rollout,
)
from world_model.src.hiqr_v2.flow_evaluation import _ego_control_recovery
from world_model.src.hiqr_v2.train import (
    _load_state,
    _save_state,
    load_hiqr_v2_checkpoint,
)


def _model(**overrides) -> HiQRV2WorldModel:
    torch.manual_seed(7)
    return HiQRV2WorldModel(
        HiQRV2Config(
            hidden_dim=32,
            scene_latent_dim=8,
            agent_residual_dim=8,
            attention_layers=1,
            decoder_layers=1,
            num_heads=4,
            dropout=0.0,
            **overrides,
        )
    ).eval()


def _batch(batch_size: int = 1) -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    states = torch.randn(batch_size, 174, 7, 6)
    states[..., 2] += 20.0
    valid = torch.ones(batch_size, 174, 7, dtype=torch.bool)
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


def test_prior_rollout_is_insensitive_to_future_background_labels() -> None:
    model, batch = _model(), _batch()
    changed = copy.deepcopy(batch)
    changed["agent_states"][:, 25:, 1:] += 10_000.0
    changed["actions_highd"] += 10_000.0
    left = model.rollout_reconstruction(batch, response_steps=3)
    right = model.rollout_reconstruction(changed, response_steps=3)
    assert torch.equal(left["predicted_states"], right["predicted_states"])


def test_plan_step_and_rollout_share_one_response_core() -> None:
    model, batch = _model(), _batch()
    current = batch["agent_states"][:, model.cfg.anchor_state_index]
    valid = batch["agent_valid"][:, model.cfg.anchor_state_index]
    ego = model._ego_mask(batch)
    state = model.initialize_start(
        current,
        valid,
        ego,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        batch["behavior_anchor_raw"],
        batch["behavior_anchor_valid"],
    )
    online = model.plan_step(
        None,
        None,
        current,
        valid,
        ego,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        filter_state=state,
    )
    offline = model.rollout_reconstruction(batch, response_steps=1)
    assert torch.equal(
        online["background_future_actions"],
        offline["background_future_actions"][:, 0],
    )


def test_zero_initialized_jerk_starts_from_observed_control() -> None:
    model, batch = _model().eval(), _batch()
    current = batch["agent_states"][:, model.cfg.anchor_state_index]
    plan = model.rollout_reconstruction(batch, response_steps=1)[
        "background_future_actions"
    ][:, 0]
    initial = model.dynamics.controls_from_highd_actions(
        current[:, 1:, 4:6], current[:, 1:]
    )
    expected = initial[:, None].expand_as(plan).clone()
    expected[..., 0].clamp_(model.cfg.min_acceleration, model.cfg.max_acceleration)
    expected[..., 1].clamp_(-model.cfg.max_yaw_rate, model.cfg.max_yaw_rate)
    valid = batch["agent_valid"][:, model.cfg.anchor_state_index, None, 1:, None]
    assert torch.allclose(plan, expected * valid.float(), atol=1.0e-6)


def test_plan_changes_obey_jerk_limits() -> None:
    model, batch = _model().eval(), _batch()
    torch.nn.init.normal_(model.decoder.jerk.weight, std=0.2)
    plan = model.rollout_reconstruction(batch, response_steps=1)[
        "background_future_actions"
    ][:, 0]
    limit = plan.new_tensor(
        (model.cfg.max_longitudinal_jerk, model.cfg.max_yaw_jerk)
    ) * float(model.cfg.simulation_dt_s)
    assert torch.all((plan[:, 1:] - plan[:, :-1]).abs() <= limit + 1.0e-6)


def test_start_does_not_observe_the_initialized_state_twice() -> None:
    model, batch = _model(), _batch()
    current, valid = batch["agent_states"][:, 24], batch["agent_valid"][:, 24]
    ego = model._ego_mask(batch)
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
    )
    assert torch.equal(output["filter_state"].global_hidden, initial.global_hidden)
    assert torch.equal(output["filter_state"].agent_hidden, initial.agent_hidden)


def test_scene_mode_is_held_for_five_responses_then_refreshes() -> None:
    scene = _model().rollout_reconstruction(_batch(), response_steps=6)["scene_latent"]
    assert all(torch.equal(scene[:, 0], scene[:, index]) for index in range(1, 5))
    assert not torch.equal(scene[:, 4], scene[:, 5])


def test_stochastic_innovations_are_replayable_and_hierarchical() -> None:
    model, batch = _model(), _batch()
    current, valid = batch["agent_states"][:, 24], batch["agent_valid"][:, 24]
    ego = model._ego_mask(batch)
    state = model.initialize_start(
        current,
        valid,
        ego,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        batch["behavior_anchor_raw"],
        batch["behavior_anchor_valid"],
    )
    scene_noise = torch.ones(1, model.cfg.scene_latent_dim)
    agent_noise = torch.ones(1, 7, model.cfg.agent_residual_dim)
    arguments = dict(
        filter_state=state,
        deterministic=False,
        scene_standard_normal=scene_noise,
        agent_standard_normal=agent_noise,
    )
    first = model.plan_step(
        None,
        None,
        current,
        valid,
        ego,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        **arguments,
    )
    second = model.plan_step(
        None,
        None,
        current,
        valid,
        ego,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        **arguments,
    )
    assert torch.equal(first["scene_latent"], second["scene_latent"])
    assert torch.equal(first["agent_residual"], second["agent_residual"])


def test_logged_ego_controls_are_exogenous_and_recovered_per_tick() -> None:
    model = _model()
    source = torch.zeros(1, 1, 6)
    target = torch.zeros_like(source)
    source[..., 2] = 10.0
    speed, heading = 10.04, 0.004
    target[..., 2] = speed * torch.cos(torch.tensor(heading))
    target[..., 3] = speed * torch.sin(torch.tensor(heading))
    controls = model._logged_ego_controls(
        source, target, torch.ones(1, 1, dtype=torch.bool)
    )
    assert torch.allclose(controls, torch.tensor([[[1.0, 0.1]]]), atol=1.0e-4)


def test_flow_ads_reports_logged_ego_control_recovery() -> None:
    model, frames = _model(), 4
    state = torch.zeros(1, 1, 6)
    state[..., 2] = 15.0
    rows = [state[:, 0].clone()]
    actions = np.zeros((1, frames - 1, 2), np.float32)
    valid = np.ones((1, frames - 1), bool)
    for frame in range(frames - 1):
        state = model.dynamics.step(
            state,
            torch.as_tensor(actions[:, frame : frame + 1]),
            torch.as_tensor(valid[:, frame : frame + 1]),
            model.cfg.simulation_dt_s,
        )
        rows.append(state[:, 0].clone())
    replay = np.zeros((1, frames, 7, 6), np.float32)
    replay[:, :, 0] = torch.stack(rows, 1).numpy()
    report = _ego_control_recovery(model, replay, actions, valid)
    assert report["acceptance"]
    assert report["fde_m"] == pytest.approx(0.0, abs=1.0e-6)


def test_b0_initializes_each_background_agent_independently() -> None:
    model, batch = _model(), _batch()
    current, valid = batch["agent_states"][:, 24], batch["agent_valid"][:, 24]
    ego = model._ego_mask(batch)
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
    zeros = torch.zeros(1, 6, 6)
    baseline = model.filter.initialize(scene, agents, zeros, valid[:, 1:])
    changed_b0 = zeros.clone()
    changed_b0[:, 2, 0] = 3.0
    changed = model.filter.initialize(scene, agents, changed_b0, valid[:, 1:])
    assert not torch.equal(baseline.agent_hidden[:, 3], changed.agent_hidden[:, 3])
    assert torch.equal(baseline.agent_hidden[:, 1], changed.agent_hidden[:, 1])


def test_fixed_cohort_rejects_c0_active_vehicle_exits() -> None:
    valid = np.ones((3, 174, 7), bool)
    valid[1, 80:, 1] = False
    valid[2, 24, 2] = False
    valid[2, 80:, 2] = False
    assert np.array_equal(
        stable_slot_mask({"agent_valid": valid}), np.asarray([True, False, True])
    )
    manifest = cohort_manifest(
        {"agent_valid": valid, "split_index": np.asarray([0, 0, 0])}
    )
    assert manifest["excluded_sequences"] == 1
    assert manifest["exiting_background_slots"] == 1


def test_snapshot_restore_and_future_ams_branches_are_replayable() -> None:
    model, metadata = _model(), _metadata()
    parent = HiQRV2WorldModelEnvironment(model)
    parent.reset_from_flow(
        np.zeros(40, np.float32),
        np.zeros((6, 6), np.float32),
        metadata,
        deterministic=False,
        world_randomness=HiQRWorldRandomness(seed=13),
    )
    for _ in range(5):
        parent.step(np.zeros(2, np.float32))
    snapshot = parent.snapshot()
    scene_branch = HiQRV2WorldModelEnvironment(model)
    residual_branch = HiQRV2WorldModelEnvironment(model)
    scene_branch.branch_from_snapshot(
        snapshot, HiQRWorldRandomness(seed=31), level="scene"
    )
    residual_branch.branch_from_snapshot(
        snapshot, HiQRWorldRandomness(seed=31), level="residual"
    )
    assert scene_branch.step(np.zeros(2, np.float32))["planner_updated"]
    assert residual_branch.step(np.zeros(2, np.float32))["planner_updated"]
    assert scene_branch.trace["world_randomness"]["response_innovations"][-1][
        "scene_refreshed"
    ]
    assert not residual_branch.trace["world_randomness"]["response_innovations"][-1][
        "scene_refreshed"
    ]


def test_checkpoint_contract_rejects_other_model_types(tmp_path) -> None:
    model = _model()
    model.flow_schema_sha256 = "b" * 64
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    state_path = tmp_path / "state.pt"
    _save_state(
        state_path,
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
        state_path,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(restored_optimizer),
        flow_schema_sha256="b" * 64,
    )
    assert info["model_type"] == model.model_type
    assert "checkpoint_schema_version" not in info
    old = tmp_path / "old.pt"
    torch.save(
        {
            "model_type": "prior_driven_hierarchical_interaction_query_world_model_v2_r2",
        },
        old,
    )
    with pytest.raises(ValueError, match="not HiQR-v2"):
        load_hiqr_v2_checkpoint(old)


def test_bootstrap_admission_requires_significant_ten_percent_gain() -> None:
    comparison = _bootstrap_difference(
        np.full(32, 0.8), np.ones(32), samples=100, seed=9
    )
    assert comparison["candidate_improvement_fraction"] >= 0.10
    assert comparison["ci95_high_m"] < 0.0


def test_risk_report_uses_predicted_ego_and_includes_distributions() -> None:
    predicted = torch.zeros(1, 1, 7, 6)
    target = torch.zeros_like(predicted)
    valid = torch.zeros(1, 1, 7, dtype=torch.bool)
    valid[:, :, :2] = True
    predicted[:, :, 0, 0], predicted[:, :, 1, 0] = 100.0, 101.0
    target[:, :, 1, 0] = 20.0
    _, _, risk = _summary_from_rollout(
        {"predicted_states": predicted, "target_states": target, "target_valid": valid}
    )
    summary = _aggregate_risk([risk])
    assert summary["collision_episode_rate"] == 1.0
    assert summary["target_collision_episode_rate"] == 0.0
    assert summary["risk_variable_distribution"]["gap_m"]["available"]


def test_complete_149_transition_rollout_and_environment_replay() -> None:
    assert _model().rollout_reconstruction(_batch())["predicted_states"].shape == (
        1,
        149,
        7,
        6,
    )
    model, metadata = _model(), _metadata()
    left = HiQRV2WorldModelEnvironment(model)
    right = HiQRV2WorldModelEnvironment(model)
    for environment in (left, right):
        environment.reset_from_flow(
            np.zeros(40, np.float32),
            np.zeros((6, 6), np.float32),
            metadata,
            deterministic=False,
            world_randomness=HiQRWorldRandomness(seed=101),
        )
    for _ in range(149):
        left.step(np.zeros(2, np.float32))
        right.step(np.zeros(2, np.float32))
    assert left.physics_step_index == right.physics_step_index == 149
    assert left.response_index == right.response_index == 29
    assert np.array_equal(
        left.observe()["agent_states"], right.observe()["agent_states"]
    )

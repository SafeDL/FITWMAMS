"""Focused isolation and causal-regression tests for HiQR-WM."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from world_model.src.core.initial_behavior_anchor import (
    behavior_anchor_from_flow_feature,
)
from world_model.src.hiqr import (
    BatchedHiQRWorldModelEnvironment,
    HierarchicalInteractionQueryRefineWorldModel,
    HiQRFlowStartMetadata,
    HiQRWorldModelConfig,
    HiQRWorldModelEnvironment,
    HiQRWorldRandomness,
)
from world_model.src.hiqr.data import (
    HIQR_SEQUENCE_FIELDS,
    HIQR_TRAINING_SIDECAR_ARRAYS,
    _validate_flow_tail_alignment,
    to_hiqr_batch,
)
from world_model.src.hiqr.encoder import UnifiedRelationalQueryEncoder
from world_model.src.hiqr.flow_evaluation import (
    compare_flow_rollouts,
    replay_states_to_ego_controls,
)
from world_model.src.hiqr.train import (
    _load_training_state,
    _save_training_state,
    _write_tensorboard_epoch,
)


def _model() -> HierarchicalInteractionQueryRefineWorldModel:
    torch.manual_seed(7)
    return HierarchicalInteractionQueryRefineWorldModel(
        HiQRWorldModelConfig(
            hidden_dim=32,
            scene_latent_dim=8,
            agent_residual_dim=8,
            temporal_layers=1,
            attention_layers=1,
            decoder_layers=1,
            num_heads=4,
            dropout=0.0,
            rollout_frames=15,
        )
    )


def _batch(batch: int = 2, frames: int = 42) -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    states = torch.randn(batch, frames, 7, 6)
    valid = torch.ones(batch, frames, 7, dtype=torch.bool)
    valid[:, :, -1] = False
    maps = torch.zeros(batch, 2, 4, 6)
    maps[..., 0] = torch.arange(4)
    maps[:, 1, :, 1] = 3.6
    maps[..., 2] = 1.0
    maps[..., 4] = 3.6
    return {
        "agent_states": states,
        "agent_valid": valid,
        "ego_index": torch.zeros(batch, dtype=torch.long),
        "map_polylines": maps,
        "map_polyline_valid": torch.ones(batch, 2, 4, dtype=torch.bool),
        "actions_highd": torch.randn(batch, frames - 25, 6, 2),
        "behavior_anchor_raw": torch.randn(batch, 6, 6),
        "behavior_anchor_valid": valid[:, 24, 1:].clone(),
    }


def _metadata() -> HiQRFlowStartMetadata:
    maps = np.zeros((2, 4, 6), np.float32)
    maps[..., 0] = np.arange(4)
    maps[1, :, 1] = 3.6
    maps[..., 2] = 1.0
    maps[..., 4] = 3.6
    return HiQRFlowStartMetadata(
        slot_valid=np.ones(6, bool),
        map_polylines=maps,
        map_polyline_valid=np.ones((2, 4), bool),
        primary_slot_index=0,
    )


def test_unified_encoder_has_no_qr_start_path_or_risk_inputs() -> None:
    model, batch = _model(), _batch()
    encoder = model.encoder
    assert not hasattr(encoder, "encode_start")
    assert "lane_graph_edges" not in inspect.signature(encoder.forward).parameters
    assert encoder.relation_feature_dim == 9
    assert encoder.relation_layers[0].bias[0].in_features == 9
    valid, ego = batch["agent_valid"][:, 24], torch.zeros(2, 7, dtype=torch.bool)
    ego[:, 0] = True
    start = encoder(
        None,
        None,
        batch["agent_states"][:, 24],
        valid,
        ego,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        mode="start",
    )
    roll = encoder(
        batch["agent_states"][:, :25],
        batch["agent_valid"][:, :25],
        batch["agent_states"][:, 24],
        valid,
        ego,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        mode="roll",
    )
    assert start[0].shape == roll[0].shape == (2, 7, model.cfg.hidden_dim)
    assert not start[0][:, -1].any() and not roll[0][:, -1].any()


def test_signed_headings_and_lane_relations_preserve_the_design_inputs() -> None:
    states = torch.zeros(1, 2, 6)
    states[0, 0, 2], states[0, 1, 2] = -10.0, 10.0
    valid = torch.ones(1, 2, dtype=torch.bool)
    features = UnifiedRelationalQueryEncoder._state_features(
        states, valid, torch.zeros_like(valid)
    )
    assert features[0, 0, 7] < -0.99 and features[0, 1, 7] > 0.99
    relation = _model().encoder._relation_features(
        states, torch.zeros(1, 2, 6), torch.zeros(1, 2, dtype=torch.bool)
    )
    assert not relation[..., 7:].any()


def test_b0_validity_is_independent_from_the_flow_event_slot_mask() -> None:
    model, batch = _model().eval(), _batch(batch=1)
    current, current_valid = batch["agent_states"][:, 24], batch["agent_valid"][:, 24]
    anchor_valid = batch["behavior_anchor_valid"].clone()
    anchor_valid[:, 0] = False
    raw = batch["behavior_anchor_raw"].clone()
    masked_raw = raw.clone()
    masked_raw[:, 0] = 0.0
    ego = torch.zeros(1, 7, dtype=torch.bool)
    ego[:, 0] = True
    first = model.initialize_start(
        current,
        current_valid,
        ego,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        raw,
        anchor_valid,
    )
    second = model.initialize_start(
        current,
        current_valid,
        ego,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        masked_raw,
        anchor_valid,
    )
    assert torch.allclose(first, second)


def test_primary_risk_slot_is_audit_only_not_an_h0_feature() -> None:
    model = _model()
    assert (
        "primary_slot_index" not in inspect.signature(model.initialize_start).parameters
    )
    assert "primary_slot_index" not in HIQR_TRAINING_SIDECAR_ARRAYS
    assert model.interaction_state.event[0].in_features == 6


def test_hierarchy_and_adaptive_continuation_are_joint_and_mask_aware() -> None:
    model, batch = _model(), _batch()
    rollout = model.rollout_reconstruction(batch, response_steps=3, deterministic=True)
    assert rollout["scene_latent"].shape == (2, 3, 8)
    assert rollout["agent_residual"].shape == (2, 3, 7, 8)
    assert not rollout["agent_residual"][:, :, -1].any()
    masks = rollout["background_future_action_masks"]
    assert masks["carried"].shape == (2, 3, 25, 6)
    assert (
        masks["carried"][:, 1, :20, :5].all()
        and not masks["carried"][:, 1, :, -1].any()
    )
    assert not masks["carried"][:, 1, 20:].any()
    expected_revised = (rollout["continuation_gate"].amax(dim=-1) > 1.0e-4) & masks[
        "valid"
    ]
    assert torch.equal(masks["revised"], expected_revised)
    assert not masks["revised"][:, 0].any()
    assert masks["revised"][:, 1, :20, :5].all()
    assert not masks["revised"][:, 1, 20:].any()
    assert not hasattr(model, "joint_refiner") and not hasattr(model, "scene_memory")
    assert "raw_b0" not in inspect.signature(model.plan_step).parameters
    assert "lane_graph_edges" not in inspect.signature(model.plan_step).parameters


def test_roll_gates_read_only_their_matching_carry_actions() -> None:
    model, batch = _model().eval(), _batch(batch=1)
    previous = torch.randn(1, 25, 6, 2, requires_grad=True)
    output = model.decoder(
        torch.randn(1, 7, model.cfg.hidden_dim),
        torch.randn(1, model.cfg.hidden_dim),
        torch.randn(1, model.cfg.scene_latent_dim),
        torch.randn(1, 7, model.cfg.agent_residual_dim),
        batch["agent_valid"][:, 24],
        previous,
    )
    output["continuation_gate"][:, :20].sum().backward()
    assert previous.grad is not None
    assert not previous.grad[:, :5].any()
    assert previous.grad[:, 5:].abs().sum() > 0
    assert model.decoder.new_tail_token.shape == (1, 5, 1, model.cfg.hidden_dim)


def test_hiqr_batching_avoids_qr_aliases_and_lane_graph_data() -> None:
    assert "lane_graph_edges" not in HIQR_SEQUENCE_FIELDS
    assert "primary_slot_index" not in HIQR_TRAINING_SIDECAR_ARRAYS
    batch = to_hiqr_batch(
        (torch.ones(1, 6, 6),), ("behavior_anchor_raw",), torch.device("cpu")
    )
    assert set(batch) == {"behavior_anchor_raw"}


def test_hiqr_keeps_the_30_response_start_roll_time_protocol() -> None:
    cfg = HiQRWorldModelConfig()
    assert cfg.start_reconstruction_frames * cfg.simulation_dt_s == 1.0
    assert (
        cfg.rollout_frames - cfg.start_reconstruction_frames
    ) * cfg.simulation_dt_s == 4.96
    assert cfg.response_steps == 30
    assert cfg.rollout_frames - (cfg.response_steps - 1) * cfg.execute_frames == 4


def test_decoder_uses_the_complete_configured_acceleration_range() -> None:
    model = _model()
    with torch.no_grad():
        model.decoder.action.weight.zero_()
        model.decoder.action.bias.copy_(torch.tensor((-20.0, -20.0)))
    low = model.decoder._direct_actions(torch.zeros(1, 1, 1, model.cfg.hidden_dim))
    with torch.no_grad():
        model.decoder.action.bias.copy_(torch.tensor((20.0, 20.0)))
    high = model.decoder._direct_actions(torch.zeros(1, 1, 1, model.cfg.hidden_dim))
    assert torch.allclose(low[..., 0], torch.full_like(low[..., 0], -8.0), atol=1.0e-4)
    assert torch.allclose(high[..., 0], torch.full_like(high[..., 0], 4.0), atol=1.0e-4)
    assert torch.allclose(low[..., 1], torch.full_like(low[..., 1], -0.6), atol=1.0e-4)
    assert torch.allclose(high[..., 1], torch.full_like(high[..., 1], 0.6), atol=1.0e-4)


def test_gap_ttc_loss_only_supervises_same_lane_front_followers() -> None:
    model = _model()
    target = torch.zeros(1, 1, 7, 6)
    target[:, :, 0, 2] = 20.0
    target[:, :, 1, 0], target[:, :, 1, 2] = 30.0, 10.0
    target[:, :, 2, 0], target[:, :, 2, 1] = 30.0, 3.6
    target[:, :, 3, 0], target[:, :, 3, 2] = -20.0, 5.0
    valid = torch.zeros(1, 1, 6, dtype=torch.bool)
    valid[:, :, :3] = True
    baseline = model._gap_ttc_loss(target.clone(), target, valid)
    outside = target.clone()
    outside[:, :, 2, 0] = -100.0
    outside[:, :, 2, 2] = 80.0
    outside[:, :, 3, 0] = 100.0
    outside[:, :, 3, 2] = -80.0
    assert torch.allclose(model._gap_ttc_loss(outside, target, valid), baseline)
    follower = target.clone()
    follower[:, :, 1, 0] = 45.0
    follower[:, :, 1, 2] = 0.0
    assert model._gap_ttc_loss(follower, target, valid) > baseline


def test_explicit_response_noise_stream_never_reuses_an_innovation() -> None:
    control = HiQRWorldRandomness(
        scene_standard_normal=np.stack((np.zeros(8), np.ones(8))),
        agent_standard_normal=np.stack((np.zeros((7, 8)), np.ones((7, 8)))),
    )
    first = control.resolve_response(
        response_index=1,
        scene_dim=8,
        agents=7,
        residual_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    second = control.resolve_response(
        response_index=2,
        scene_dim=8,
        agents=7,
        residual_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert not torch.equal(first[0], second[0])
    assert not torch.equal(first[1], second[1])
    one_response = HiQRWorldRandomness(
        scene_standard_normal=np.zeros(8),
        agent_standard_normal=np.zeros((7, 8)),
    )
    with pytest.raises(ValueError, match="requires a seed or explicit response noise"):
        one_response.resolve_response(
            response_index=2,
            scene_dim=8,
            agents=7,
            residual_dim=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_sidecar_b0_is_checked_against_the_frozen_flow_tail(tmp_path) -> None:
    features = np.zeros((1, 76), np.float32)
    slot_mask = np.ones((1, 6), bool)
    np.savez(
        tmp_path / "dataset.npz",
        segment_id=np.asarray(("segment-1",)),
        features=features,
        slot_mask=slot_mask,
    )
    raw, valid = behavior_anchor_from_flow_feature(features[0], slot_mask[0])
    summary = _validate_flow_tail_alignment(
        {"sequence_id": np.asarray(("segment-1",))},
        SimpleNamespace(source_path=tmp_path / "dataset_schema.json"),
        raw[None],
        valid[None],
    )
    assert summary["matched_flow_tail_sequences"] == 1
    broken = raw.copy()
    broken[0, 0] = 1.0
    with pytest.raises(ValueError, match="differs from frozen Flow"):
        _validate_flow_tail_alignment(
            {"sequence_id": np.asarray(("segment-1",))},
            SimpleNamespace(source_path=tmp_path / "dataset_schema.json"),
            broken[None],
            valid[None],
        )


def test_tensorboard_epoch_records_training_and_validation_scalars() -> None:
    class Recorder:
        def __init__(self) -> None:
            self.scalars: list[tuple[str, float, int]] = []
            self.flushed = False

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            self.scalars.append((tag, value, step))

        def flush(self) -> None:
            self.flushed = True

    writer = Recorder()
    _write_tensorboard_epoch(
        writer,
        {
            "epoch": 3,
            "rollout_seconds": 3.0,
            "selection_metric": 1.25,
            "train_loss": 0.4,
            "val_position": 0.3,
        },
    )
    assert writer.flushed
    assert ("batch/train/loss", 0.4, 3) not in writer.scalars
    assert ("epoch/train/loss", 0.4, 3) in writer.scalars
    assert ("epoch/validation/position", 0.3, 3) in writer.scalars
    assert ("selection/validation_fde_m", 1.25, 3) in writer.scalars


def test_training_state_restores_model_optimizer_scheduler_and_progress(
    tmp_path,
) -> None:
    model = _model()
    model.flow_schema_sha256 = "frozen-schema"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    parameter = next(model.parameters())
    parameter.sum().backward()
    optimizer.step()
    scheduler.step(1.0)
    expected = parameter.detach().clone()
    path = tmp_path / "last_training_state.pt"
    _save_training_state(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=9,
        stage_index=1,
        stage_epoch=3,
        global_step=27,
        best_validation_fde=4.5,
        training_protocol={"h0_event_structure": "slot_mask_only_causal"},
    )
    with torch.no_grad():
        parameter.add_(1.0)
    optimizer.param_groups[0]["lr"] = 0.5
    restored = _load_training_state(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        flow_schema_sha256="frozen-schema",
    )
    assert restored == {
        "epoch": 9,
        "stage_index": 1,
        "stage_epoch": 3,
        "global_step": 27,
        "best_validation_fde": 4.5,
    }
    assert torch.allclose(parameter, expected)
    assert optimizer.state and optimizer.param_groups[0]["lr"] == 1.0e-3


def test_posterior_excludes_future_ego_and_has_trainable_kl() -> None:
    model, batch = _model(), _batch()
    result = model.forward_training(batch, response_steps=2)
    result["loss"].backward()
    assert result["scene_kl"].isfinite() and result["agent_kl"].isfinite()
    assert model.interaction_state.scene_posterior[0].weight.grad is not None

    current, current_valid = batch["agent_states"][:, 24], batch["agent_valid"][:, 24]
    ego = torch.zeros_like(current_valid)
    ego[:, 0] = True
    hidden = model.initialize_start(
        current,
        current_valid,
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
        current_valid,
        ego,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        mode="start",
    )
    future, future_valid = (
        batch["agent_states"][:, 25:30],
        batch["agent_valid"][:, 25:30],
    )
    changed_ego = future.clone()
    changed_ego[:, :, 0] += 100.0
    original = model.interaction_state.sample(
        agents,
        scene,
        hidden,
        current,
        current_valid,
        deterministic=True,
        use_posterior=True,
        posterior_future=future,
        posterior_future_valid=future_valid,
    )
    masked = model.interaction_state.sample(
        agents,
        scene,
        hidden,
        current,
        current_valid,
        deterministic=True,
        use_posterior=True,
        posterior_future=changed_ego,
        posterior_future_valid=future_valid,
    )
    assert torch.allclose(original[0], masked[0])
    assert torch.allclose(original[1], masked[1])


def test_snapshot_restore_and_scene_vs_residual_branching_are_replayable() -> None:
    model = _model().eval()
    meta = _metadata()
    parent = HiQRWorldModelEnvironment(model)
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
    scene_child, residual_child = HiQRWorldModelEnvironment(
        model
    ), HiQRWorldModelEnvironment(model)
    scene_child.branch_from_snapshot(
        snapshot, HiQRWorldRandomness(seed=31), level="scene"
    )
    residual_child.branch_from_snapshot(
        snapshot, HiQRWorldRandomness(seed=31), level="residual"
    )
    scene_next, residual_next = scene_child.step(
        np.zeros(2, np.float32)
    ), residual_child.step(np.zeros(2, np.float32))
    assert scene_next["planner_updated"] and residual_next["planner_updated"]
    assert (
        scene_next["trace"]["world_randomness"]["branch_resampling"][-1]["level"]
        == "scene"
    )
    assert (
        residual_next["trace"]["world_randomness"]["branch_resampling"][-1]["level"]
        == "residual"
    )
    for _ in range(4):
        scene_child.step(np.zeros(2, np.float32))
        residual_child.step(np.zeros(2, np.float32))
    scene_child.step(np.zeros(2, np.float32))
    residual_child.step(np.zeros(2, np.float32))

    def roll_innovations(environment: HiQRWorldModelEnvironment) -> list[dict]:
        return [
            row
            for row in environment.trace["world_randomness"]["response_innovations"]
            if row["kind"] == "roll"
        ]

    def response_noise(seed: int, response_index: int) -> tuple[np.ndarray, np.ndarray]:
        scene, agent = HiQRWorldRandomness(seed=seed).resolve_response(
            response_index=response_index,
            scene_dim=model.cfg.scene_latent_dim,
            agents=7,
            residual_dim=model.cfg.agent_residual_dim,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        return scene.numpy(), agent.numpy()

    scene_rolls, residual_rolls = roll_innovations(scene_child), roll_innovations(
        residual_child
    )
    assert [row["response_index"] for row in scene_rolls] == [1, 2]
    assert [row["response_index"] for row in residual_rolls] == [1, 2]
    for index, response_index in enumerate((1, 2)):
        child_scene, child_agent = response_noise(31, response_index)
        parent_scene, _ = response_noise(13, response_index)
        assert np.allclose(scene_rolls[index]["scene_standard_normal"], child_scene)
        assert np.allclose(scene_rolls[index]["agent_standard_normal"], child_agent)
        assert np.allclose(residual_rolls[index]["scene_standard_normal"], parent_scene)
        assert np.allclose(residual_rolls[index]["agent_standard_normal"], child_agent)
    replay = HiQRWorldModelEnvironment(model)
    replay.restore(snapshot)
    restored = replay.observe()
    assert np.array_equal(restored["agent_states"], parent.observe()["agent_states"])


def test_batched_worlds_keep_independent_random_streams() -> None:
    model = _model().eval()
    features, slots = np.zeros((2, 76), np.float32), np.ones((2, 6), bool)
    maps = np.zeros((2, 2, 4, 6), np.float32)
    maps[..., 0] = np.arange(4)
    maps[:, 1, :, 1] = 3.6
    maps[..., 2] = 1.0
    maps[..., 4] = 3.6
    environment = BatchedHiQRWorldModelEnvironment(model)
    environment.reset_from_flow_batch(
        features,
        slots,
        maps,
        np.ones((2, 2, 4), bool),
        primary_slot_index=np.zeros(2, np.int64),
        deterministic=False,
        world_randomness=[HiQRWorldRandomness(seed=101), HiQRWorldRandomness(seed=202)],
    )
    response = environment.advance_response(np.zeros((2, 5, 2), np.float32))
    assert response["agent_state_frames"].shape == (2, 5, 7, 6)
    assert not torch.equal(
        response["agent_state_frames"][0], response["agent_state_frames"][1]
    )
    snapshot = environment.snapshot()
    child = BatchedHiQRWorldModelEnvironment(model)
    child.branch_from_snapshot(
        snapshot,
        [HiQRWorldRandomness(seed=303), HiQRWorldRandomness(seed=404)],
        level="scene",
    )
    next_step = child.step(np.zeros((2, 2), np.float32))
    assert next_step["response_index"] == 1
    with pytest.raises(ValueError, match="ego_valid"):
        child.step(np.zeros((2, 2), np.float32), ego_valid=np.ones(1, bool))


def test_flow_metrics_and_replayed_heading_use_signed_velocity() -> None:
    states = np.zeros((1, 3, 7, 6), np.float32)
    states[:, :, 0, 2] = -10.0
    states[:, :, 1, 0] = 20.0
    valid = np.ones((1, 3, 7), bool)
    controls, control_valid = replay_states_to_ego_controls(states, valid, dt_s=0.04)
    assert control_valid.all() and np.allclose(controls[..., 1], 0.0)
    generated = states[:, 1:].copy()
    generated[:, :, 1, 0] -= 1.0
    metrics = compare_flow_rollouts(
        generated, valid[:, 1:], states[:, 1:], valid[:, 1:]
    )
    assert metrics["risk_variable_distribution"]["gap_m"]["available"]
    assert metrics["risk_variable_distribution"]["ttc_s"]["available"]
    assert "drac_mae_mps2" in metrics["following_error"]
    assert "collision_episode_rate" in metrics["generated_collision"]

"""Architecture, jointness, Flow lifecycle, and causal-rollout tests for QR-WM."""

from __future__ import annotations

import torch

from normalizing_flow.src.features import slot_feature_index
from world_model.src.core.initial_behavior_anchor import start_state_from_flow_feature, start_state_from_flow_tensor
from world_model.src.qr.config import QRWorldModelConfig
from world_model.src.qr.environment import FlowStartMetadata, QRWorldModelEnvironment
from world_model.src.qr.model import QueryRefineWorldModel
from world_model.src.qr.train import _tensorboard_writer, _write_tensorboard_epoch, load_qr_checkpoint


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    frames, agents, maps, points = 150, 7, 8, 8
    state = torch.zeros(batch_size, frames, agents, 6)
    state[..., 0] = torch.arange(frames).view(1, frames, 1) * 0.8
    state[..., 0] += torch.arange(agents).view(1, 1, agents) * 12.0
    state[..., 1] = (torch.arange(agents).view(1, 1, agents) % 3 - 1) * 3.6
    state[..., 2] = 20.0 + torch.arange(agents).view(1, 1, agents) * 0.2
    state[..., 4] = 0.05
    state[..., 5] = 0.01
    return {
        "agent_states": state,
        "agent_valid": torch.ones(batch_size, frames, agents, dtype=torch.bool),
        "ego_index": torch.zeros(batch_size, dtype=torch.long),
        "map_polylines": torch.randn(batch_size, maps, points, 6),
        "map_polyline_valid": torch.ones(batch_size, maps, points, dtype=torch.bool),
        "lane_graph_edges": torch.tensor([[[0, 1, 0], [1, 2, 1]]], dtype=torch.long).expand(batch_size, -1, -1).clone(),
        "actions_highd": torch.zeros(batch_size, 125, agents - 1, 2),
        "is_evt_tail": torch.zeros(batch_size, dtype=torch.bool),
    }


def _model() -> QueryRefineWorldModel:
    return QueryRefineWorldModel(
        QRWorldModelConfig(
            hidden_dim=32, behavior_latent_dim=8, plan_frames=25, execute_frames=5, refinement_iterations=2,
            attention_layers=2, num_heads=4, refinement_noise_levels=(1.0,),
        )
    )


def test_qr_training_rollout_has_refined_receding_buffer_and_gradients() -> None:
    model, batch = _model(), _batch()
    result = model.forward_training(batch, response_steps=2, tbptt_steps=1)
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["denoising"])
    result["loss"].backward()
    assert model.joint_refiner.blocks[0].agents.in_proj_weight.grad is not None
    assert model.joint_refiner.noise_embedding[0].weight.grad is not None
    rollout = model.rollout_reconstruction(batch, response_steps=2, deterministic=True)
    assert rollout["predicted_states"].shape == (2, 10, 7, 6)
    assert rollout["control_buffers"].shape == (2, 2, 25, 6, 2)
    assert rollout["refined_plan_states"].shape == (2, 2, 25, 6, 6)
    assert not torch.equal(rollout["pre_refinement_buffers"], rollout["control_buffers"])
    masks = rollout["control_buffer_masks"]
    assert masks["refinable"].shape == (2, 2, 25, 6)
    assert rollout["executed_control_masks"].shape == (2, 2, 5, 6)
    assert not hasattr(model, "world_update")
    assert not hasattr(model, "memory")
    assert len(model.encoder.relation_blocks) == model.cfg.attention_layers
    assert len(model.joint_refiner.blocks) == model.cfg.attention_layers


def test_qr_inference_does_not_read_background_future() -> None:
    model, batch = _model(), _batch()
    model.eval()
    reference = model.rollout_reconstruction(batch, response_steps=2, deterministic=True)["predicted_states"]
    altered = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    altered["agent_states"][:, 25:, 1:] += 10_000.0
    altered["actions_highd"] += 1_000.0
    candidate = model.rollout_reconstruction(altered, response_steps=2, deterministic=True)["predicted_states"]
    assert torch.equal(reference, candidate)


def test_joint_refiner_has_cross_agent_control_gradients() -> None:
    model = _model().eval()
    refiner, h, z = model.joint_refiner, model.cfg.hidden_dim, model.cfg.behavior_latent_dim
    controls = torch.randn(1, 25, 6, 2, requires_grad=True)
    residual = refiner.residual(
        controls, torch.randn(1, 25, 6, 6), torch.randn(1, 7, h), torch.randn(1, h), torch.randn(1, h),
        torch.randn(1, 7, z), torch.randn(1, 4, h), torch.ones(1, 4, dtype=torch.bool),
        torch.randn(1, 25, 2), torch.randn(1, 25, 6), torch.ones(1, 25, 6, dtype=torch.bool), torch.ones(1),
    )
    residual[:, :, 0].square().mean().backward()
    assert controls.grad is not None
    assert controls.grad[:, :, 1].abs().sum() > 0


def test_b0_changes_only_start_initialization_and_flow_rollout() -> None:
    model, batch = _model().eval(), _batch()
    batch["behavior_anchor_raw"] = torch.zeros(2, 6, 6)
    batch["behavior_anchor_valid"] = torch.ones(2, 6, dtype=torch.bool)
    first = model.rollout_reconstruction(batch, response_steps=2, deterministic=True)
    changed = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    changed["behavior_anchor_raw"][:, 0, 0] = 3.0
    second = model.rollout_reconstruction(changed, response_steps=2, deterministic=True)
    assert not torch.equal(first["control_buffers"][:, 0], second["control_buffers"][:, 0])
    assert "raw_anchor" not in model.plan_step.__code__.co_varnames

    feature = torch.zeros(2, 76)
    maps = torch.randn(2, 4, 5, 6)
    keyword = {
        "slot_valid": torch.ones(2, 6, dtype=torch.bool), "map_polylines": maps,
        "map_polyline_valid": torch.ones(2, 4, 5, dtype=torch.bool),
        "lane_graph_edges": torch.zeros(2, 1, 3, dtype=torch.long),
        "ego_future_controls": torch.zeros(2, 10, 2), "response_steps": 2,
    }
    flow_first = model.rollout_from_flow(feature, **keyword)
    feature[:, 40] = 3.0
    flow_second = model.rollout_from_flow(feature, **keyword)
    assert flow_first["predicted_states"].shape == (2, 10, 7, 6)
    assert not torch.equal(flow_first["control_buffers"][:, 0], flow_second["control_buffers"][:, 0])


def test_ego_controls_condition_background_and_buffer_masks_protect_invalid_slots() -> None:
    model, batch = _model().eval(), _batch()
    batch["agent_valid"][:, :, 6] = False
    zeros = torch.zeros(2, 10, 2)
    altered = zeros.clone(); altered[..., 0] = 2.0
    left = model.rollout(batch, response_steps=2, deterministic=True, ego_future_controls=zeros)
    right = model.rollout(batch, response_steps=2, deterministic=True, ego_future_controls=altered)
    assert not torch.equal(left["predicted_states"][:, :, 1:], right["predicted_states"][:, :, 1:])
    assert not left["control_buffer_masks"]["refinable"][..., -1].any()
    assert torch.equal(left["control_buffers"][..., -1, :], torch.zeros_like(left["control_buffers"][..., -1, :]))


def test_start_mix_is_convex_decaying_and_start_loss_is_trained() -> None:
    model, batch = _model(), _batch()
    fresh = torch.ones(1, 25, 6, 2)
    anchor = torch.full_like(fresh, -3.0)
    fresh[..., 1] = 0.1
    anchor[..., 1] = -0.3
    mixed = model._mix_start_controls(fresh, anchor)
    assert torch.allclose(mixed[:, 0, :, 0], torch.full_like(mixed[:, 0, :, 0], -2.0))
    assert torch.allclose(mixed[:, 0, :, 1], torch.full_like(mixed[:, 0, :, 1], -0.2))
    assert torch.allclose(mixed[:, -1], fresh[:, -1])
    batch["behavior_anchor_raw"] = torch.zeros(2, 6, 6)
    batch["behavior_anchor_valid"] = torch.ones(2, 6, dtype=torch.bool)
    start = model.supervised_terms(batch, response_steps=5, start_mode=True, training=True)
    assert torch.isfinite(start["start_summary"])
    mixed_terms = model.forward_training(batch, response_steps=2, tbptt_steps=1)
    assert mixed_terms["start_fraction"] == 0.5
    assert torch.isfinite(mixed_terms["start_loss"])
    assert torch.isfinite(mixed_terms["roll_loss"])


def test_ads_rollout_integrates_ego_without_reading_logged_future() -> None:
    model, batch = _model().eval(), _batch()
    try:
        model.rollout(batch, response_steps=2)
    except ValueError as exc:
        assert "rollout_reconstruction" in str(exc)
    else:
        raise AssertionError("dynamic rollout must require ADS controls")
    controls = torch.zeros(2, 10, 2)
    reference = model.rollout(batch, response_steps=2, deterministic=True, ego_future_controls=controls)
    changed = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    changed["agent_states"][:, 25:, 0, :4] += 10_000.0
    candidate = model.rollout(changed, response_steps=2, deterministic=True, ego_future_controls=controls)
    assert torch.equal(reference["predicted_states"], candidate["predicted_states"])
    expected_ego = model.dynamics.step(batch["agent_states"][:, 24, :1], controls[:, 0, None], batch["agent_valid"][:, 24, :1], model.cfg.simulation_dt_s)
    assert torch.allclose(reference["predicted_states"][:, 0, :1], expected_ego)


def test_ego_future_state_token_and_online_environment() -> None:
    model, batch = _model().eval(), _batch()
    refiner, h, z = model.joint_refiner, model.cfg.hidden_dim, model.cfg.behavior_latent_dim
    ego_states = torch.randn(1, 25, 6, requires_grad=True)
    residual = refiner.residual(
        torch.randn(1, 25, 6, 2), torch.randn(1, 25, 6, 6), torch.randn(1, 7, h), torch.randn(1, h), torch.randn(1, h),
        torch.randn(1, 7, z), torch.randn(1, 4, h), torch.ones(1, 4, dtype=torch.bool),
        torch.randn(1, 25, 2), ego_states, torch.ones(1, 25, 6, dtype=torch.bool), torch.ones(1),
    )
    residual.square().mean().backward()
    assert ego_states.grad is not None and ego_states.grad.abs().sum() > 0

    metadata = FlowStartMetadata(
        slot_valid=torch.ones(6, dtype=torch.bool).numpy(), map_polylines=batch["map_polylines"][0].numpy(),
        map_polyline_valid=batch["map_polyline_valid"][0].numpy(), lane_graph_edges=batch["lane_graph_edges"][0].numpy(),
        primary_slot_index=0, event_structure=[1, 0], mask_pattern=63, event_structure_id=0, event_structure_log_prob=-0.2,
        conditional_log_prob=-1.3, log_prob=-1.5,
    )
    c0, b0 = torch.zeros(40), torch.zeros(6, 6)
    c0[0] = 20.0
    left, right = QRWorldModelEnvironment(model), QRWorldModelEnvironment(model)
    assert left.reset_from_flow(c0, b0, metadata)["flow_metadata"]["log_prob"] == -1.5
    right.reset_from_flow(c0, b0, metadata)
    quiet, braking = left.step(torch.zeros(5, 2)), right.step(torch.tensor([[-4.0, 0.0]] * 5))
    assert quiet["response_index"] == 1
    assert not torch.equal(torch.from_numpy(quiet["agent_states"][:1]), torch.from_numpy(braking["agent_states"][:1]))
    assert not torch.equal(torch.from_numpy(quiet["background_states"]), torch.from_numpy(braking["background_states"]))


def test_flow_start_has_no_synthetic_history_and_keeps_metadata() -> None:
    from unittest.mock import patch

    model, batch = _model().eval(), _batch()
    feature = torch.zeros(2, 76); feature[:, 0] = 20.0
    metadata = {"primary_slot_index": torch.tensor([0, 1]), "log_prob": torch.tensor([-2.0, -3.0]), "event_structure": torch.ones(2, 3)}
    keyword = {
        "slot_valid": torch.ones(2, 6, dtype=torch.bool), "map_polylines": batch["map_polylines"],
        "map_polyline_valid": batch["map_polyline_valid"], "lane_graph_edges": batch["lane_graph_edges"],
        "ego_future_controls": torch.zeros(2, 10, 2), "response_steps": 2, "flow_metadata": metadata,
    }
    with patch.object(model.encoder, "_temporal_tokens", wraps=model.encoder._temporal_tokens) as temporal:
        rollout = model.rollout_from_flow(feature, **keyword)
    assert [call.args[0].shape[1] for call in temporal.call_args_list] == [6]
    assert torch.equal(rollout["flow_metadata"]["log_prob"], metadata["log_prob"])


def test_qr_flow_adapter_restores_absolute_background_velocity_numerically() -> None:
    feature = torch.zeros(1, 76)
    feature[:, 0:4] = torch.tensor([[31.0, -2.0, 0.3, -0.4]])
    feature[:, slot_feature_index("same_front", "rel_x_m")] = 18.0
    feature[:, slot_feature_index("same_front", "rel_y_left_m")] = 3.6
    feature[:, slot_feature_index("same_front", "rel_vx_mps")] = -7.0
    feature[:, slot_feature_index("same_front", "rel_vy_left_mps")] = 1.5
    feature[:, slot_feature_index("same_front", "other_ax_mps2")] = -1.0
    feature[:, slot_feature_index("same_front", "other_ay_left_mps2")] = 0.5
    slots = torch.tensor([[True, False, False, False, False, False]])
    scene, valid, anchor = start_state_from_flow_tensor(feature, slots)[:3]
    numpy_scene, numpy_valid, numpy_anchor, _ = start_state_from_flow_feature(feature[0].numpy(), slots[0].numpy())
    assert torch.allclose(scene[0], torch.from_numpy(numpy_scene))
    assert torch.equal(valid[0], torch.from_numpy(numpy_valid))
    assert torch.allclose(anchor[0], torch.from_numpy(numpy_anchor))
    assert torch.equal(scene[0, 1, 2:4], torch.tensor([24.0, -0.5]))
    assert not valid[0, 2]
    qr_scene, _, _ = QueryRefineWorldModel.flow_condition_to_scene(feature, slots)
    assert torch.equal(qr_scene, scene)


def test_incompatible_checkpoint_is_rejected_with_retrain_message(tmp_path) -> None:
    checkpoint = tmp_path / "obsolete.pt"
    torch.save({"model_type": QueryRefineWorldModel.model_type, "architecture_version": 2}, checkpoint)
    try:
        load_qr_checkpoint(checkpoint)
    except ValueError as exc:
        assert "Retrain QR-WM" in str(exc)
    else:
        raise AssertionError("incompatible checkpoint must not load into QR-WM")


def test_qr_tensorboard_records_batch_loss_and_epoch_metrics(tmp_path) -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tensorboard_writer, log_dir = _tensorboard_writer(tmp_path, {"tensorboard": True})
    assert tensorboard_writer is not None
    assert log_dir == tmp_path / "tensorboard"
    tensorboard_writer.add_scalar("batch/train/loss", 1.25, 0)
    _write_tensorboard_epoch(
        tensorboard_writer,
        {
            "epoch": 1,
            "rollout_seconds": 5.0,
            "train_loss": 1.0, "train_start_summary": 0.25, "train_start_loss": 1.1, "train_roll_loss": 0.9,
            "val_position": 0.5,
            "selection_metric": 1.75,
        },
    )
    tensorboard_writer.close()

    events = EventAccumulator(str(log_dir))
    events.Reload()
    tags = events.Tags()["scalars"]
    assert {
        "batch/train/loss", "epoch/train/loss", "epoch/train/start_summary", "epoch/train/start_loss",
        "epoch/validation/position", "selection/validation_fde_m",
    } <= set(tags)

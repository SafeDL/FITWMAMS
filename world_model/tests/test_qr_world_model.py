"""Architecture and causal-rollout tests for the independent QR-WM."""

from __future__ import annotations

import torch

from world_model.src.qr.config import QRWorldModelConfig
from world_model.src.qr.model import QueryRefineWorldModel
from world_model.src.qr.train import _tensorboard_writer, _write_tensorboard_epoch


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
        QRWorldModelConfig(hidden_dim=32, behavior_latent_dim=8, plan_frames=25, execute_frames=5, refinement_iterations=2)
    )


def test_qr_training_rollout_has_refined_receding_buffer_and_gradients() -> None:
    model, batch = _model(), _batch()
    result = model.forward_training(batch, response_steps=2, tbptt_steps=1)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert model.encoder.agent_query.weight.grad is not None
    rollout = model.rollout(batch, response_steps=2, deterministic=True)
    assert rollout["predicted_states"].shape == (2, 10, 7, 6)
    assert rollout["control_buffers"].shape == (2, 2, 25, 6, 2)
    assert rollout["refined_plan_states"].shape == (2, 2, 25, 6, 6)
    assert not torch.equal(rollout["pre_refinement_buffers"], rollout["control_buffers"])


def test_qr_inference_does_not_read_background_future() -> None:
    model, batch = _model(), _batch()
    model.eval()
    reference = model.rollout(batch, response_steps=2, deterministic=True)["predicted_states"]
    altered = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    altered["agent_states"][:, 25:, 1:] += 10_000.0
    altered["actions_highd"] += 1_000.0
    candidate = model.rollout(altered, response_steps=2, deterministic=True)["predicted_states"]
    assert torch.equal(reference, candidate)


def test_qr_flow_adapter_returns_training_schema_scene() -> None:
    feature = torch.randn(3, 76)
    slots = torch.tensor([[True, True, False, True, False, True]]).expand(3, -1)
    scene, valid, anchor = QueryRefineWorldModel.flow_condition_to_scene(feature, slots)
    assert scene.shape == (3, 7, 6)
    assert valid.shape == (3, 7)
    assert anchor.shape == (3, 6, 6)
    assert torch.equal(scene[:, 0, 2:6], feature[:, :4])
    assert not valid[:, 3].any()


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
            "train_loss": 1.0,
            "val_position": 0.5,
            "selection_metric": 1.75,
        },
    )
    tensorboard_writer.close()

    events = EventAccumulator(str(log_dir))
    events.Reload()
    tags = events.Tags()["scalars"]
    assert {"batch/train/loss", "epoch/train/loss", "epoch/validation/position", "selection/validation_fde_m"} <= set(tags)

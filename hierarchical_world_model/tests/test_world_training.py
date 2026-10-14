"""World training, scene identity and fixed-condition sampling regressions."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hierarchical_world_model.src import data, train
from hierarchical_world_model.src.losses import training_losses
from hierarchical_world_model.src.model import FactualDynamicsModel
from hierarchical_world_model.src.config import WorldModelConfig


def test_full_split_uses_current_split_rows_without_removed_seed(monkeypatch):
    bundle = SimpleNamespace(arrays={"split_index": np.array([2, 0, 1, 0])})
    monkeypatch.setattr(data, "load_data_bundle", lambda *a: bundle)
    e = data.prepare_experiment_data(
        {"training": {"seed": 1, "experiment_scope": "full"}}, None
    )
    np.testing.assert_array_equal(e.train_rows, [1, 3])
    np.testing.assert_array_equal(e.validation_rows, [2])
    np.testing.assert_array_equal(e.test_rows, [0])


def test_training_provenance_survives_cache_reordering():
    e = SimpleNamespace(
        bundle=SimpleNamespace(arrays={"sequence_id": np.array(["a", "b", "c"])}),
        train_rows=np.array([0, 2]),
        validation_rows=np.array([1]),
    )
    moved = SimpleNamespace(
        bundle=SimpleNamespace(arrays={"sequence_id": np.array(["c", "a", "b"])}),
        train_rows=np.array([0, 1]),
        validation_rows=np.array([2]),
    )
    assert data.training_sequence_ids(e) == data.training_sequence_ids(moved)


def test_closed_loop_dataset_retains_future_sliding_previews():
    states = np.zeros((1, 174, 7, 6), np.float32)
    states[..., 2] = 25
    bundle = SimpleNamespace(
        arrays={
            "agent_states": states,
            "agent_valid": np.ones((1, 174, 7), bool),
            "actions_highd": np.zeros((1, 149, 6, 2), np.float32),
            "map_polylines": np.zeros((1, 8, 8, 6), np.float32),
            "map_polyline_valid": np.ones((1, 8, 8), bool),
        }
    )
    plans = np.zeros((1, 149, 6, 2), np.float32)
    plans[..., 0] = np.arange(149)[None, :, None]
    sample = data.ResponseDataset(
        bundle, np.array([0]), training=False, seed=1, soft_plans=plans
    )[0]
    assert sample["soft_reference"].shape == (49, 6, 2)
    assert sample["history"].shape[0] == 1
    assert sample["committed_ego_controls"].numel() == 0
    collated = data.response_collate([sample])
    assert not collated["committed_ego_controls_valid"].any()
    assert collated["committed_ego_controls"].eq(0).all()
    torch.testing.assert_close(
        sample["soft_reference"][24:49, 0, 0], torch.arange(24, 49).float()
    )


def test_early_stopping_has_minimum_training_and_stable_patience():
    assert not train.should_stop(8, 2, minimum_epochs=20, patience=10)
    assert not train.should_stop(20, 15, minimum_epochs=20, patience=10)
    assert train.should_stop(25, 15, minimum_epochs=20, patience=10)


def test_warmup_and_decay_reach_configured_rates():
    rates = [
        train.learning_rate_factor(i, epochs=50, warmup_epochs=3, minimum_ratio=0.1)
        for i in range(50)
    ]
    assert rates[:3] == pytest.approx([1 / 3, 2 / 3, 1])
    assert rates[-1] == pytest.approx(0.1)
    assert all(a >= b for a, b in zip(rates[3:], rates[4:]))


def test_stochastic_samples_use_identical_controls_and_independent_noise():
    model = FactualDynamicsModel(WorldModelConfig(graph_coupled_latent_enabled=True))
    model.eval()
    current = torch.zeros(1, 7, 6)
    current[..., 2] = 25
    history = current[:, None].expand(-1, 25, -1, -1).clone()
    reference = torch.zeros(1, 49, 6, 2)
    reference[..., 0] = torch.arange(1, 50)[None, :, None]
    batch = {
        "history": history,
        "history_valid": torch.ones(1, 25, 7, dtype=torch.bool),
        "current": current,
        "current_valid": torch.ones(1, 7, dtype=torch.bool),
        "target_actions": torch.zeros(1, 1, 6, 2),
        "target_states": current[:, None, 1:],
        "previous_actions": torch.zeros(1, 6, 2),
        "soft_reference": reference,
        "reference_base": torch.zeros(1, 6, 2),
        "map_polylines": torch.zeros(1, 8, 8, 6),
        "map_polyline_valid": torch.ones(1, 8, 8, dtype=torch.bool),
        "committed_ego_controls": torch.zeros(1, 5, 2),
    }
    calls = []
    original = model.forward

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    model.forward = capture
    training_losses(
        model,
        batch,
        torch.ones(1, 16),
        torch.ones(1, 7, 16),
        include_intervention=False,
        include_closed_loop=False,
        generator=torch.Generator().manual_seed(7),
    )
    assert len(calls) == 2
    assert all(
        call[1]["committed_ego_controls"] is batch["committed_ego_controls"]
        for call in calls
    )
    assert all(call[0][4].shape[1] == 25 for call in calls)
    assert not torch.equal(calls[1][1]["scene_standard_normal"], -torch.ones(1, 16))
    batch["closed_target_states"] = current[:, None, 1:].expand(-1, 25, -1, -1).clone()
    batch["closed_ego_actions"] = torch.zeros(1, 25, 2)
    scores = []
    for global_seed in (11, 99):
        torch.manual_seed(global_seed)
        scores.append(
            training_losses(
                model,
                batch,
                torch.ones(1, 16),
                torch.ones(1, 7, 16),
                include_intervention=False,
                generator=torch.Generator().manual_seed(7),
            )["closed_loop_factual"].detach()
        )
    torch.testing.assert_close(scores[0], scores[1], atol=0, rtol=0)


def test_validation_supervises_every_scene_without_synthetic_interventions(monkeypatch):
    model = torch.nn.Linear(1, 1)
    model.cfg = SimpleNamespace(scene_latent_dim=16, agent_latent_dim=16)
    calls = []

    def loss(model, batch, *args, **kwargs):
        calls.append(kwargs)
        return {"loss": torch.tensor(1.0)}

    monkeypatch.setattr(train, "training_losses", loss)
    train._mean_terms(
        model,
        [{"current": torch.zeros(7, 7, 6)}],
        torch.device("cpu"),
        optimizer=None,
        grad_clip=1.0,
        intervention_interval=8,
        intervention_sequences=4,
        noise_seed=1,
        deterministic_forward=True,
    )
    assert calls[0]["closed_loop_sequences"] == 7
    assert calls[0]["include_closed_loop"]
    assert not calls[0]["include_intervention"]


def test_plan_cache_cannot_reuse_changed_scene_identity_or_inputs(
    tmp_path, monkeypatch
):
    from hierarchical_world_model.src import planner

    source = tmp_path / "dataset.npz"
    source.write_bytes(b"data")
    checkpoint = tmp_path / "diffusion.pt"
    checkpoint.write_bytes(b"model")
    bundle = SimpleNamespace(
        arrays={"sequence_id": np.array(["a"])},
        flow_schema={"dataset_npz": str(source)},
    )
    calls = []

    class Model:
        def eval(self):
            pass

        def sample_ddim(self, condition, mask, **kwargs):
            calls.append(1)
            return torch.zeros_like(mask, dtype=torch.float32)

    monkeypatch.setattr(
        planner,
        "load_checkpoint",
        lambda *a, **k: (
            Model(),
            {
                "dataset_contract": {
                    "condition_dim": 68,
                    "target_representation": "residual",
                }
            },
        ),
    )
    monkeypatch.setattr(
        planner, "BackgroundTrajectoryDataset", lambda *a, **k: object()
    )
    batch = {
        "condition": torch.zeros(1, 68),
        "target_mask": torch.ones(1, 149, 12),
        "trajectory_reference": torch.zeros(1, 149, 6, 2),
    }
    monkeypatch.setattr(planner, "make_loader", lambda *a, **k: [batch])
    monkeypatch.setattr(
        planner,
        "decode_residual_trajectory",
        lambda normalized, reference, *a: reference,
    )

    def run():
        return planner.frozen_diffusion_plans(
            bundle,
            np.array([0]),
            checkpoint=checkpoint,
            output_dir=tmp_path / "plans",
            device=torch.device("cpu"),
            batch_size=1,
            ddim_steps=20,
        )

    run()
    run()
    assert len(calls) == 1
    bundle.arrays["sequence_id"] = np.array(["new_scene"])
    run()
    assert len(calls) == 2
    source.write_bytes(b"changed physical data")
    run()
    assert len(calls) == 3
    checkpoint.write_bytes(b"changed weights")
    run()
    assert len(calls) == 4

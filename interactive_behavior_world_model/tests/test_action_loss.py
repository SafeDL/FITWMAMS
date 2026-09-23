import numpy as np
import pytest
import torch

from interactive_behavior_world_model.policies.rolling_action_policy import (
    ActionDiffusionConfig,
    RollingActionDiffusion,
    runtime_action_method_id,
)
from interactive_behavior_world_model.scripts.evaluate_action import prepare_run_contract
from interactive_behavior_world_model.scripts.train_action import data_augmented_method_id


def test_lateral_trajectory_multiplier_strengthens_lateral_error():
    config = ActionDiffusionConfig(
        action_horizon=2, hidden_dim=16, num_layers=1, num_heads=4, dropout=0.0
    )
    model = RollingActionDiffusion(
        config, np.zeros(2, np.float32), np.ones(2, np.float32)
    ).eval()
    clean = torch.zeros((1, 2, 6, 2))
    features = torch.zeros((1, 25, 7, 12))
    history_valid = torch.ones((1, 25, 7), dtype=torch.bool)
    target_valid = torch.ones((1, 2, 6), dtype=torch.bool)
    current = torch.zeros((1, 6, 6))
    current[..., 2] = 10.0
    future = current[:, None].expand(-1, 2, -1, -1).clone()
    future[..., 1] = 2.0

    torch.manual_seed(17)
    ordinary = model.loss(clean, features, history_valid, target_valid, current, future)
    torch.manual_seed(17)
    lateral = model.loss(
        clean,
        features,
        history_valid,
        target_valid,
        current,
        future,
        lateral_trajectory_multiplier=5.0,
    )

    assert lateral["trajectory"] > ordinary["trajectory"]


def test_flow_matching_loss_and_samplers_are_finite_and_masked():
    config = ActionDiffusionConfig(
        action_horizon=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        training_objective="flow_matching",
    )
    model = RollingActionDiffusion(
        config, np.zeros(2, np.float32), np.ones(2, np.float32)
    ).eval()
    clean = torch.zeros((1, 2, 6, 2))
    features = torch.zeros((1, 25, 7, 12))
    history_valid = torch.ones((1, 25, 7), dtype=torch.bool)
    target_valid = torch.ones((1, 2, 6), dtype=torch.bool)
    target_valid[..., -1] = False
    current = torch.zeros((1, 6, 6))
    future = current[:, None].expand(-1, 2, -1, -1).clone()

    losses = model.loss(clean, features, history_valid, target_valid, current, future)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()

    noise = torch.randn_like(clean)
    cold = model.sample(features, history_valid, target_valid, noise, inference_steps=4)
    warm = model.sample_warm(
        features, history_valid, target_valid, cold, noise, inference_steps=4
    )
    assert torch.isfinite(cold).all() and torch.isfinite(warm).all()
    assert torch.equal(cold[..., -1, :], torch.zeros_like(cold[..., -1, :]))
    assert torch.equal(warm[..., -1, :], torch.zeros_like(warm[..., -1, :]))


def test_flow_euler_reaches_constant_velocity_endpoint_at_each_nfe_budget():
    class ConstantVelocity(torch.nn.Module):
        def forward(self, noisy, timestep, features, valid):
            del timestep, features, valid
            return torch.full_like(noisy, 0.5)

    config = ActionDiffusionConfig(
        action_horizon=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        training_objective="flow_matching",
    )
    model = RollingActionDiffusion(
        config, np.zeros(2, np.float32), np.ones(2, np.float32)
    ).eval()
    model.denoiser = ConstantVelocity()
    features = torch.zeros((1, 25, 7, 12))
    history_valid = torch.ones((1, 25, 7), dtype=torch.bool)
    target_valid = torch.ones((1, 2, 6), dtype=torch.bool)
    noise = torch.zeros((1, 2, 6, 2))
    target = torch.full_like(noise, 0.5)

    for evaluations in (4, 8, 16):
        cold = model.sample(
            features, history_valid, target_valid, noise, inference_steps=evaluations
        )
        warm = model.sample_warm(
            features,
            history_valid,
            target_valid,
            target,
            noise,
            inference_steps=evaluations,
            start_step=49,
        )
        assert torch.allclose(cold, target, atol=1.0e-6)
        assert torch.allclose(warm, target, atol=1.0e-6)


def test_runtime_method_id_preserves_data_and_objective_ablation_suffixes():
    assert (
        runtime_action_method_id("rolling_action_b0_d0d1", False)
        == "rolling_action_b0_d0d1"
    )
    assert (
        runtime_action_method_id("rolling_action_b0_d0d1", True)
        == "rolling_action_b1_d0d1"
    )
    assert (
        runtime_action_method_id("rolling_action_flow_b0_d0d1", True)
        == "rolling_action_flow_b1_d0d1"
    )
    assert (
        runtime_action_method_id("rolling_action_b2rlp_e3", True)
        == "rolling_action_b2rlp_e3"
    )


def test_data_augmented_method_id_encodes_d1_dose():
    assert (
        data_augmented_method_id("rolling_action_b0", False, 0.05)
        == "rolling_action_b0"
    )
    assert (
        data_augmented_method_id("rolling_action_b0", True, 1.0)
        == "rolling_action_b0_d0d1"
    )
    assert (
        data_augmented_method_id("rolling_action_b0", True, 0.05)
        == "rolling_action_b0_d0d1f005"
    )
    assert (
        data_augmented_method_id("rolling_action_b2r_d0d1", True, 0.05)
        == "rolling_action_b2r_d0d1"
    )


def test_resume_contract_rejects_changed_checkpoint_or_budget(tmp_path):
    path = tmp_path / "run_contract.json"
    contract = {"checkpoint_sha256": "abc", "futures": 4}
    prepare_run_contract(path, contract, resume_with_metrics=False)
    prepare_run_contract(path, contract, resume_with_metrics=True)
    with pytest.raises(ValueError, match="does not match"):
        prepare_run_contract(
            path, {**contract, "futures": 16}, resume_with_metrics=True
        )

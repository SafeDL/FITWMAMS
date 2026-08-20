"""Contracts for state-knot-conditioned position-residual diffusion."""

from __future__ import annotations

import numpy as np
import torch

from diffusion.src.data import (
    CONSTRAINT_FEATURE_DIM,
    CONDITION_DIM,
    c0_states_from_physical_features,
    condition_vector,
    extract_trajectory_constraint,
    prepare_external_condition,
    semantic_cutin_agents,
    split_rows,
    trajectory_reference_positions,
)
from diffusion.src.evaluation import (
    _accumulate_component_errors,
    _cohort_metrics,
    _component_error_metrics,
    _new_component_error_totals,
    states_from_actions,
)
from diffusion.src.model import BackgroundTrajectoryDiffusion, DiffusionModelConfig
from diffusion.src.sampling import decode_background_latents


def _model():
    return BackgroundTrajectoryDiffusion(
        DiffusionModelConfig(
            hidden_dim=16, num_layers=1, num_heads=4, dropout=0, diffusion_steps=8
        )
    ).eval()


def _condition(batch=2):
    value = torch.randn(batch, CONDITION_DIM)
    value[:, 40:46] = 1
    return value


def test_condition_is_shared_40d_c0_plus_mask_and_state_knots():
    c0 = np.arange(40, dtype=np.float32)
    mask = np.array([1, 1, 0, 1, 0, 1], bool)
    constraint = np.arange(6 * CONSTRAINT_FEATURE_DIM, dtype=np.float32).reshape(6, -1)
    value = condition_vector(c0, mask, constraint)
    assert value.shape == (CONDITION_DIM,)
    np.testing.assert_array_equal(value[:40], c0)
    assert not value[46:].reshape(6, -1)[~mask].any()


def _identity_flow_schema():
    from normalizing_flow.src.features import build_feature_schema

    names = list(build_feature_schema().feature_names)
    return {
        "feature_names": names,
        "model_feature_transforms": ["identity"] * len(names),
        "normalization": {"mean": [0.0] * len(names), "std": [1.0] * len(names)},
    }


def test_external_condition_accepts_physical_c0_and_explicit_state_knots():
    c0 = np.zeros(40, np.float32)
    c0[0] = 20.0
    c0[4:10] = [30.0, 0.2, -2.0, 0.1, -0.3, 0.05]
    slots = np.array([True, False, False, False, False, False])
    constraint = np.zeros((6, CONSTRAINT_FEATURE_DIM), np.float32)
    constraint[0, [0, 4, 8]] = [36.0, 68.0, 96.0]
    contract = {
        "trajectory_constraint": {
            "mean": [0.0] * CONSTRAINT_FEATURE_DIM,
            "std": [1.0] * CONSTRAINT_FEATURE_DIM,
        }
    }
    prepared = prepare_external_condition(
        c0,
        slots,
        constraint,
        flow_schema=_identity_flow_schema(),
        diffusion_contract=contract,
        inactive_seed=7,
    )
    assert prepared["condition"].shape == (118,)
    assert prepared["target_mask"].shape == (149, 12)
    assert prepared["trajectory_reference"].shape == (149, 6, 2)
    np.testing.assert_allclose(prepared["c0_states"][1], [30, 0.2, 18, 0.1, -0.3, 0.05])
    assert prepared["target_mask"][:, :2].all()
    assert not prepared["target_mask"][:, 2:].any()


def test_external_condition_rejects_flow_plan_as_state_knot_substitute():
    with np.testing.assert_raises_regex(ValueError, r"shape \[6, 12\]"):
        prepare_external_condition(
            np.zeros(40, np.float32),
            np.ones(6, bool),
            np.zeros((6, 8), np.float32),
            flow_schema=_identity_flow_schema(),
            diffusion_contract={
                "trajectory_constraint": {
                    "mean": [0.0] * CONSTRAINT_FEATURE_DIM,
                    "std": [1.0] * CONSTRAINT_FEATURE_DIM,
                }
            },
            inactive_seed=7,
        )


def test_c0_state_restoration_uses_ego_relative_slot_kinematics():
    c0 = np.zeros(40, np.float32)
    c0[:4] = [25.0, 0.4, -0.2, 0.1]
    c0[4:10] = [10.0, -0.5, 3.0, -0.2, 0.4, -0.1]
    states = c0_states_from_physical_features(
        c0, np.array([True, False, False, False, False, False])
    )
    np.testing.assert_allclose(states[0], [0, 0, 25, 0.4, -0.2, 0.1])
    np.testing.assert_allclose(states[1], [10, -0.5, 28, 0.2, 0.4, -0.1])
    assert not states[2:].any()


def test_external_condition_matches_the_training_dataset_contract():
    from pathlib import Path

    from diffusion.src.data import BackgroundTrajectoryDataset, load_data_bundle
    from world_model.src.core.utils import load_json, load_yaml

    root = Path(__file__).resolve().parents[2]
    config_path = root / "diffusion/configs/highd_background_diffusion.yaml"
    config = load_yaml(config_path)
    bundle = load_data_bundle(config, config_path.parent)
    row = int(np.flatnonzero(np.asarray(bundle.arrays["split_index"]) == 2)[0])
    contract = load_json(root / "results/background_diffusion/dataset_contract.json")
    item = BackgroundTrajectoryDataset(bundle, np.asarray([row]), contract)[0]
    flow_row = int(bundle.flow_row_for_sequence[row])
    prepared = prepare_external_condition(
        np.asarray(bundle.flow_arrays["features"])[flow_row],
        np.asarray(bundle.flow_arrays["slot_mask"])[flow_row],
        item["trajectory_constraint"].numpy(),
        flow_schema=bundle.flow_schema,
        diffusion_contract=contract,
        inactive_reference_normalized=np.asarray(
            bundle.flow_arrays["features_normalized"]
        )[flow_row],
    )
    np.testing.assert_allclose(prepared["condition"], item["condition"].numpy())
    np.testing.assert_allclose(
        prepared["c0_states"], item["c0_states"].numpy(), atol=2.0e-6
    )
    np.testing.assert_allclose(
        prepared["trajectory_reference"],
        item["trajectory_reference"].numpy(),
        atol=2.0e-6,
    )
    np.testing.assert_array_equal(prepared["target_mask"], item["target_mask"].numpy())


def test_external_condition_requires_an_explicit_inactive_reference():
    with np.testing.assert_raises_regex(ValueError, "inactive_seed"):
        prepare_external_condition(
            np.zeros(40, np.float32),
            np.array([True, False, False, False, False, False]),
            np.zeros((6, 12), np.float32),
            flow_schema=_identity_flow_schema(),
            diffusion_contract={
                "trajectory_constraint": {
                    "mean": [0.0] * CONSTRAINT_FEATURE_DIM,
                    "std": [1.0] * CONSTRAINT_FEATURE_DIM,
                }
            },
        )


def test_future_constraint_changes_do_not_change_c0_or_mask_context():
    c0 = np.arange(40, dtype=np.float32)
    mask = np.array([1, 0, 1, 0, 1, 0], bool)
    first = condition_vector(c0, mask, np.zeros((6, 12), np.float32))
    second = condition_vector(c0, mask, np.ones((6, 12), np.float32))
    np.testing.assert_array_equal(first[:46], second[:46])
    assert not np.array_equal(first[46:], second[46:])


def test_trajectory_reference_passes_through_declared_knots():
    states = np.zeros((150, 6, 6), np.float32)
    time = np.arange(150) * 0.04
    states[..., 0] = time[:, None] * 20.0
    states[..., 2] = 20.0
    states[:, 0, 1] = np.linspace(0.0, 3.6, 150)
    states[:, 0, 3] = 3.6 / (149 * 0.04)
    constraint = extract_trajectory_constraint(states)
    reference = trajectory_reference_positions(states[0], constraint)
    np.testing.assert_allclose(
        reference[[49, 99, 148]], states[[50, 100, 149], :, :2], atol=1e-5
    )


def test_semantic_cutin_requires_retention():
    states = np.zeros((150, 7, 6), np.float32)
    valid = np.zeros((150, 7), bool)
    valid[:, :2] = True
    states[:, 1, 0] = 15
    states[:81, 1, 1] = np.linspace(3, 0.5, 81)
    states[81:, 1, 1] = 0.5
    assert semantic_cutin_agents(states, valid)[0]
    states[-20:, 1, 1] = 3
    assert not semantic_cutin_agents(states, valid).any()


def test_split_rows_does_not_apply_a_secondary_filter():
    arrays = {"split_index": np.array([0, 0, 1, 2])}
    assert set(split_rows(arrays, "train", seed=4)) == {0, 1}


def test_masked_loss_ignores_inactive_targets():
    model = _model()
    clean = torch.randn(2, 149, 12)
    mask = torch.ones_like(clean, dtype=torch.bool)
    mask[..., 2:4] = False
    time = torch.tensor([2, 5])
    noise = torch.randn_like(clean)
    first = model.loss(clean, _condition(), mask, timesteps=time, noise=noise)["loss"]
    clean[..., 2:4] = 1e5
    second = model.loss(clean, _condition(), mask, timesteps=time, noise=noise)["loss"]
    assert torch.isfinite(first) and torch.isfinite(second)


def test_motion_latent_is_pointwise_replayable():
    model = _model()
    condition = _condition(1)
    mask = torch.ones(1, 149, 12, dtype=torch.bool)
    latent = torch.randn(1, 149, 12)
    first = model.sample_ddim(condition, mask, inference_steps=4, initial_noise=latent)
    second = model.sample_ddim(condition, mask, inference_steps=4, initial_noise=latent)
    torch.testing.assert_close(first, second)


def test_denoiser_preserves_joint_action_shape():
    model = _model()
    noisy = torch.randn(2, 149, 12)
    output = model.denoiser(noisy, torch.tensor([2, 5]), _condition())
    assert output.shape == noisy.shape and torch.isfinite(output).all()


def test_latent_decoder_adds_position_reference_after_residual_decode():
    model = _model()
    condition = _condition(1)
    mask = torch.ones(1, 149, 12, dtype=torch.bool)
    latent = torch.randn(2, 149, 12)
    reference = np.full((149, 6, 2), 0.05, np.float32)
    contract = {"position_residual": {"mean": [0, 0], "std": [1, 1]}}
    decoded = decode_background_latents(
        model,
        condition,
        mask,
        np.zeros((6, 6), np.float32),
        contract,
        latent,
        trajectory_reference=reference,
        inference_steps=4,
    )
    assert decoded["background_states"].shape == (2, 149, 6, 6)


def test_cartesian_integration_matches_constant_acceleration():
    initial = np.zeros((1, 6, 6), np.float32)
    initial[..., 2] = 10
    actions = np.zeros((1, 149, 6, 2), np.float32)
    actions[..., 0] = 1
    result = states_from_actions(initial, actions)
    duration = 149 * 0.04
    np.testing.assert_allclose(
        result[0, -1, :, 0], 10 * duration + 0.5 * duration**2, rtol=1e-5
    )


def test_pairwise_diversity_excludes_identical_self_pairs():
    samples = np.zeros((1, 2, 3, 1, 2), np.float32)
    samples[:, 1, :, 0, 0] = 2.0
    metrics = _cohort_metrics(
        samples,
        np.zeros((1, 3, 1, 2), np.float32),
        np.ones((1, 1), bool),
    )
    assert metrics["mean_pairwise_sample_distance_m"] == 2.0
    assert metrics["mean_pairwise_endpoint_distance_m"] == 2.0


def test_velocity_metrics_include_all_random_draws_and_active_slots():
    generated = np.zeros((1, 2, 3, 2, 2), np.float32)
    generated[:, 1, :, 0, 0] = 2.0
    active = np.array([[True, False]])
    totals = _new_component_error_totals(2)
    _accumulate_component_errors(
        totals,
        generated,
        np.zeros((1, 3, 2, 2), np.float32),
        active,
    )
    metrics = _component_error_metrics(totals, ("vx", "vy"))
    assert metrics["vx"]["MAE"] == 1.0
    assert np.isclose(metrics["vx"]["RMSE"], np.sqrt(2.0))
    assert metrics["vy"]["MAE"] == 0.0

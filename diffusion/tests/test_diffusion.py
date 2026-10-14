from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from diffusion.src.data import (
    CONDITION_DIM,
    CONSTRAINT_FEATURE_DIM,
    BackgroundTrajectoryDataset,
    c0_states_from_physical_features,
    condition_vector,
    decode_residual_trajectory,
    extract_trajectory_constraint,
    load_data_bundle,
    prepare_external_condition,
    semantic_cutin_agents,
    trajectory_reference_positions,
    trajectory_residual_target,
)
from diffusion.src.model import BackgroundTrajectoryDiffusion, DiffusionModelConfig
from hierarchical_world_model.src.core.utils import load_json, load_yaml
from normalizing_flow.src.features import (
    build_feature_schema,
    feature_index,
    feature_valid_from_slot_mask,
)


def test_compact_condition_layout() -> None:
    c0 = np.arange(26, dtype=np.float32)
    mask = np.array([1, 1, 0, 1, 0, 1], bool)
    knots = np.arange(6 * CONSTRAINT_FEATURE_DIM, dtype=np.float32).reshape(6, -1)
    value = condition_vector(c0, mask, knots)
    assert value.shape == (CONDITION_DIM,) == (68,)
    np.testing.assert_array_equal(value[:26], c0)
    np.testing.assert_array_equal(value[26:32], mask)
    assert not value[32:68].reshape(6, -1)[~mask].any()


def test_model_uses_position_only_condition() -> None:
    config = DiffusionModelConfig(hidden_dim=32, num_layers=1, num_heads=4)
    model = BackgroundTrajectoryDiffusion(config)
    condition = torch.zeros(2, 68)
    target = torch.zeros(2, 149, 12)
    output = model.denoiser(target, torch.zeros(2, dtype=torch.long), condition)
    assert output.shape == target.shape


def test_c0_restoration_sets_controller_acceleration_to_zero() -> None:
    schema = build_feature_schema()
    c0 = np.zeros(schema.num_features, np.float32)
    c0[feature_index(None, "ego_vx_mps")] = 25.0
    c0[feature_index(None, "ego_vy_left_mps")] = 0.4
    c0[feature_index("same_front", "rel_x_m")] = 10.0
    c0[feature_index("same_front", "rel_vx_mps")] = 3.0
    states = c0_states_from_physical_features(
        c0, np.array([True, False, False, False, False, False])
    )
    np.testing.assert_allclose(states[0], [0, 0, 25, 0.4, 0, 0])
    np.testing.assert_allclose(states[1], [10, 0, 28, 0.4, 0, 0])


def test_residual_decode_preserves_knots_and_does_not_resmooth() -> None:
    reference = np.full((2, 1, 149, 6, 2), 10.0, np.float32)
    active = np.array([[True, False, True, True, True, True]] * 2)[:, None]
    contract = {
        "position_residual": {
            "mean": [1.0, 2.0],
            "std": [3.0, 4.0],
        }
    }
    normalized = np.ones((2, 3, 149, 12), np.float32)
    normalized[:, :, 25, 0] = 2.0
    path = decode_residual_trajectory(normalized, reference, active, contract)
    assert path.shape == (2, 3, 149, 6, 2)
    np.testing.assert_array_equal(path[..., 1, :], 0.0)
    np.testing.assert_array_equal(path[0, :, [49, 99, 148], 0], 10.0)
    np.testing.assert_allclose(path[0, :, 24, 0], np.tile((14.0, 16.0), (3, 1)))
    np.testing.assert_allclose(path[0, :, 25, 0], np.tile((17.0, 16.0), (3, 1)))


def test_pchip_reference_passes_through_position_knots() -> None:
    initial = np.zeros((2, 6, 6), np.float32)
    initial[..., :2] = [10.0, 2.0]
    knots = np.tile([40.0, 0.5, 80.0, 1.0, 120.0, 1.5], (2, 6, 1))
    reference = trajectory_reference_positions(initial, knots)
    expected = initial[..., None, :, :2] + knots.reshape(2, 6, 3, 2).swapaxes(1, 2)
    np.testing.assert_allclose(reference[:, [49, 99, 148]], expected)


def test_extract_trajectory_constraint_returns_three_xy_knots() -> None:
    states = np.zeros((150, 6, 6), np.float32)
    for frame in range(150):
        states[frame, :, 0] = frame + np.arange(6)
        states[frame, :, 1] = -2 * frame + np.arange(6)
    constraint = extract_trajectory_constraint(states)
    assert constraint.shape == (6, 6)
    np.testing.assert_allclose(
        constraint[0],
        [50.0, -100.0, 100.0, -200.0, 149.0, -298.0],
    )


def test_external_condition_matches_dataset_item() -> None:
    root = Path(__file__).resolve().parents[2]
    config_path = root / "diffusion/config.yaml"
    config = load_yaml(config_path)
    bundle = load_data_bundle(config, config_path.parent)
    row = int(np.flatnonzero(np.asarray(bundle.arrays["split_index"]) == 2)[0])
    contract = load_json(root / "results/diffusion/dataset_contract.json")
    item = BackgroundTrajectoryDataset(bundle, np.asarray([row]), contract)[0]
    flow_row = int(bundle.flow_row_for_sequence[row])
    prepared = prepare_external_condition(
        np.asarray(bundle.flow_arrays["features"])[flow_row],
        np.asarray(bundle.flow_arrays["slot_mask"])[flow_row],
        extract_trajectory_constraint(
            np.concatenate(
                (
                    item["c0_states"].numpy()[None, 1:],
                    item["future_states"].numpy()[:, 1:],
                )
            )
        ),
        flow_schema=bundle.flow_schema,
        diffusion_contract=contract,
    )
    np.testing.assert_allclose(prepared["condition"], item["condition"].numpy())
    slots = np.asarray(bundle.flow_arrays["slot_mask"])[flow_row]
    valid = feature_valid_from_slot_mask(bundle.flow_schema, slots)[0]
    np.testing.assert_array_equal(prepared["condition"][:26][~valid], 0.0)
    np.testing.assert_allclose(
        prepared["trajectory_reference"][:, slots],
        item["trajectory_reference"].numpy()[:, slots],
        atol=3.0e-5,
    )
    decoded = decode_residual_trajectory(
        item["target"].numpy(), item["trajectory_reference"].numpy(), slots, contract
    )
    expected = item["trajectory_reference"].numpy() + trajectory_residual_target(
        item["future_states"].numpy()[:, 1:, :2],
        item["trajectory_reference"].numpy(),
    )
    np.testing.assert_allclose(
        decoded[:, slots],
        expected[:, slots],
        atol=3.0e-5,
    )


def test_external_condition_requires_position_knots() -> None:
    schema = build_feature_schema()
    with np.testing.assert_raises_regex(ValueError, r"shape \[6, 6\]"):
        prepare_external_condition(
            np.zeros(schema.num_features, np.float32),
            np.ones(6, bool),
            np.zeros((6, 12), np.float32),
            flow_schema={},
            diffusion_contract={},
        )


def test_sampler_is_deterministic_and_masks_absent_agents() -> None:
    config = DiffusionModelConfig(
        hidden_dim=32, num_layers=1, diffusion_steps=4, x0_clip_abs=2.5
    )
    model = BackgroundTrajectoryDiffusion(config).eval()
    condition = torch.zeros(1, 68)
    condition[:, 26] = 1.0
    mask = torch.zeros(1, 149, 12, dtype=torch.bool)
    mask[:, :, :2] = True
    noise = torch.randn(1, 149, 12)
    first = model.sample_ddim(condition, mask, initial_noise=noise, inference_steps=4)
    second = model.sample_ddim(condition, mask, initial_noise=noise, inference_steps=4)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert not first[:, :, 2:].any()
    assert first[:, :, :2].any()
    assert torch.isfinite(first).all()


def test_physical_diagnostics_use_decoded_speed() -> None:
    from diffusion.src.evaluation import _physical_rates

    actions = np.zeros((1, 149, 6, 2), np.float32)
    vx = np.full((1, 149, 6), 20.0, np.float32)
    vx[0, 0, 0] = -1.0
    active = np.array([[True, False, False, False, False, False]])
    rates = _physical_rates(actions, vx, active)
    np.testing.assert_allclose(rates["negative_vx_rate"], 1.0 / 149)


def test_cutin_mask_matches_single_and_batch_inputs() -> None:
    states = np.zeros((150, 7, 6), np.float32)
    states[:, 0, 0] = np.arange(150) * 0.8
    states[:, 3, 0] = states[:, 0, 0] + 30.0
    states[:, 3, 1] = 3.5 * (1 - np.clip((np.arange(150) - 10) / 70, 0, 1))
    valid = np.zeros((150, 7), bool)
    valid[:, [0, 3]] = True
    expected = np.array([False, False, True, False, False, False])
    np.testing.assert_array_equal(semantic_cutin_agents(states, valid), expected)
    np.testing.assert_array_equal(
        semantic_cutin_agents(states[None], valid[None]), expected[None]
    )


def test_cohort_metrics_keep_only_active_trajectory_targets() -> None:
    from diffusion.src.evaluation import _cohort_metrics

    target = np.zeros((3, 149, 6, 2), np.float32)
    samples = np.broadcast_to(target[:, None], (3, 2, 149, 6, 2)).copy()
    samples[:, 0, ..., 0] = 1.0
    samples[:, 1, ..., 1] = 2.0
    active = np.array([[1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0] * 6], bool)
    metrics = _cohort_metrics(samples, target, active, samples[:, 0])
    assert metrics["source_sequences"] == 3 and metrics["sequences"] == 2
    np.testing.assert_allclose(metrics["sample_mean_ADE_m"], 1.5)
    np.testing.assert_allclose(metrics["min_ADE_m"], 1.0)
    np.testing.assert_allclose(metrics["ensemble_mean_ADE_m"], np.sqrt(1.25))
    np.testing.assert_allclose(metrics["mean_pairwise_sample_distance_m"], np.sqrt(5))
    assert "coordinate_CRPS_m" not in metrics and "energy_score_m" not in metrics


def test_empty_scenes_do_not_dilute_training_loss() -> None:
    model = BackgroundTrajectoryDiffusion(
        DiffusionModelConfig(hidden_dim=32, num_layers=1, diffusion_steps=4)
    ).eval()
    condition = torch.zeros(2, 68)
    condition[0, 26] = 1.0
    mask = torch.zeros(2, 149, 12, dtype=torch.bool)
    mask[0, :, :2] = True
    target = torch.randn(2, 149, 12) * mask
    noise = torch.randn_like(target)
    timesteps = torch.tensor([2, 3])
    full = model.loss(target, condition, mask, timesteps=timesteps, noise=noise)
    single = model.loss(
        target[:1], condition[:1], mask[:1], timesteps=timesteps[:1], noise=noise[:1]
    )
    for key in full:
        assert torch.isfinite(full[key])
        torch.testing.assert_close(full[key], single[key], rtol=1.0e-5, atol=1.0e-6)
    full["loss"].backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_training_report_and_interrupted_resume(tmp_path, monkeypatch) -> None:
    from diffusion.src import train

    root = Path(__file__).resolve().parents[2]
    config = load_yaml(root / "diffusion/config.yaml")
    config["paths"]["output_dir"] = str(tmp_path)
    config["training"].update(device="cpu", epochs=2, num_workers=0)
    config["model"].update(hidden_dim=32, num_layers=1)
    config["diffusion"]["steps"] = 4
    contract = load_json(root / "results/diffusion/dataset_contract.json")
    bundle = SimpleNamespace(
        arrays={
            "split_index": np.array([0, 0, 1, 1]),
            "is_evt_tail": np.zeros(4, bool),
        },
        flow_row_for_sequence=np.arange(4),
        flow_arrays={
            "slot_mask": np.ones((4, 6), bool),
            "trajectory_constraint": np.zeros((4, 6, 6), np.float32),
        },
    )
    monkeypatch.setattr(
        train, "prepare_training_data", lambda *args: (bundle, contract)
    )
    monkeypatch.setattr(
        train, "BackgroundTrajectoryDataset", lambda bundle, rows, contract: list(rows)
    )
    monkeypatch.setattr(train, "save_training_curves", lambda *args: None)
    calls = 0

    def interrupted_epoch(model, ema, loader, optimizer, *args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("test interruption")
        optimizer.step()
        return {"loss": 1.0}

    monkeypatch.setattr(train, "_train_epoch", interrupted_epoch)
    monkeypatch.setattr(train, "_fixed_validation", lambda *args: {"loss": 0.1})
    with pytest.raises(RuntimeError, match="test interruption"):
        train.train_background_diffusion(config, config_dir=root / "diffusion")
    last = tmp_path / "checkpoints/last_training_state.pt"
    assert last.is_file()
    assert len(load_json(tmp_path / "training.json")["history"]) == 1
    assert "summary" not in load_json(tmp_path / "training.json")

    monkeypatch.setattr(train, "_fixed_validation", lambda *args: {"loss": 0.05})
    summary = train.train_background_diffusion(
        config, config_dir=root / "diffusion", resume=last
    )
    report = load_json(tmp_path / "training.json")
    assert report["summary"] == summary
    assert [row["epoch"] for row in report["history"]] == [1, 2]
    assert summary["best_epoch"] == 2 and not last.exists()
    assert not (tmp_path / "training_summary.json").exists()
    assert not (tmp_path / "training_history.json").exists()
    _, checkpoint = train.load_checkpoint(
        tmp_path / summary["best_checkpoint"], device=torch.device("cpu")
    )
    assert checkpoint["epoch"] == 2
    assert "optimizer_state" not in checkpoint and "ema_model_state" not in checkpoint

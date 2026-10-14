#!/usr/bin/env python3
"""Verify the published full-data diffusion result against saved evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import load_data_bundle  # noqa: E402
from diffusion.src.train import _model_config, load_checkpoint  # noqa: E402
from hierarchical_world_model.src.core.utils import load_json, load_yaml  # noqa: E402

CONFIG = ROOT / "diffusion/config.yaml"


def _close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(actual, expected, rtol=1.0e-7, atol=1.0e-9):
        raise AssertionError(f"{name}: {actual} != {expected}")


def verify(config_path: Path) -> dict[str, object]:
    config = load_yaml(config_path)
    output = Path(config["paths"]["output_dir"]).resolve()
    contract = load_json(output / "dataset_contract.json")
    training_report = load_json(output / "training.json")
    training = training_report["summary"]
    history = training_report["history"]
    evaluation = load_json(output / "evaluation_summary.json")
    per_sequence = np.load(output / "evaluation_per_sequence.npz")

    assert config["dataset"]["condition_mode"] == "c0_position_knots"
    assert config["dataset"]["include_ego_future"] is False
    assert contract["condition_dim"] == 26 + 6 + 6 * 6 == 68
    assert contract["ego_future_in_condition"] is False
    assert contract["target_representation"] == (
        "single_smoothed_pchip_position_residual"
    )
    assert (
        config["dataset"]["target_representation"] == contract["target_representation"]
    )
    assert evaluation["target_representation"] == contract["target_representation"]
    assert contract["c0_normalization"]["fit_split"] == "train"
    assert contract["c0_normalization"]["inactive_slot_policy"] == "zero"
    assert "vehicle_dimensions" not in contract
    assert contract["trajectory_reference"] == (
        "shape_preserving_cubic_position_2s_4s_end"
    )
    assert "position_trajectory" not in contract
    assert evaluation["trajectory_interpolation"] is True
    assert evaluation["knots_hard_enforced"] is True
    assert evaluation["condition_disclosure"] == {
        "endpoint_is_conditioned": True,
        "knots_hard_enforced": True,
        "interpretation": (
            "position-knot-conditioned reconstruction, not C0-only prediction"
        ),
        "knot_times_s": [2.0, 4.0, 5.96],
        "uses_future_background_position_knots": True,
        "uses_future_ego": False,
    }

    assert training["experiment_scope"] == evaluation["experiment_scope"] == "full"
    assert training["epochs"] == len(history) <= training["requested_epochs"]
    assert training["requested_epochs"] == config["training"]["epochs"]
    assert (
        history[training["best_epoch"] - 1]["val_loss"]
        == training["best_validation_loss"]
    )
    assert training["train_sequences"] == 72_771
    assert training["validation_sequences"] == 13_133
    assert training["train_sequences_with_background_targets"] == 71_698
    assert training["validation_sequences_with_background_targets"] == 12_857
    assert evaluation["metrics"]["all"]["source_sequences"] == 10_151
    assert evaluation["metrics"]["all"]["sequences"] == 9_841
    assert training["best_epoch"] == evaluation["checkpoint_epoch"]
    checkpoint_path = output / training["best_checkpoint"]
    assert checkpoint_path == output / "checkpoints/best_background_diffusion.pt"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == training["best_epoch"]
    assert checkpoint["dataset_contract"] == contract
    assert checkpoint["model_config"]["condition_dim"] == 68
    assert checkpoint["model_config"]["c0_dim"] == 26
    assert checkpoint["model_config"]["constraint_dim"] == 6
    assert checkpoint["model_config"] == _model_config(config, contract).to_dict()
    load_checkpoint(checkpoint_path, device=torch.device("cpu"))
    assert "ema_model_state" not in checkpoint
    assert "optimizer_state" not in checkpoint
    assert "scheduler_state" not in checkpoint

    all_metrics = evaluation["metrics"]["all"]
    sample_ade = np.asarray(per_sequence["sample_ade_m"], np.float64)
    draws = int(config["evaluation"]["samples_per_condition"])
    assert sample_ade.shape == (9_841, draws)
    assert np.isfinite(sample_ade).all()
    _close(float(sample_ade.mean()), all_metrics["sample_mean_ADE_m"], "sample ADE")
    _close(float(sample_ade.min(axis=1).mean()), all_metrics["min_ADE_m"], "min ADE")

    bundle = load_data_bundle(config, config_path.parent)
    rows = np.asarray(per_sequence["row_index"], np.int64)
    assert len(np.unique(rows)) == 9_841
    evaluated_ids = set(np.asarray(bundle.arrays["sequence_id"])[rows].astype(str))
    canonical_test = np.asarray(bundle.arrays["split_index"]) == 2
    flow_rows = bundle.flow_row_for_sequence
    has_background = np.asarray(bundle.flow_arrays["slot_mask"])[flow_rows].any(axis=1)
    canonical_ids = set(
        np.asarray(bundle.arrays["sequence_id"])[
            canonical_test & has_background
        ].astype(str)
    )
    flow_test = np.asarray(bundle.flow_arrays["split_index"]) == 2
    flow_ids = set(
        np.asarray(bundle.flow_arrays["segment_id"])[
            flow_test & np.asarray(bundle.flow_arrays["slot_mask"]).any(axis=1)
        ].astype(str)
    )
    assert evaluated_ids == canonical_ids == flow_ids
    np.testing.assert_array_equal(
        per_sequence["recording_id"],
        np.asarray(bundle.flow_arrays["recording_id"])[flow_rows[rows]],
    )
    tail = np.asarray(per_sequence["is_evt_tail"], bool)
    np.testing.assert_array_equal(tail, np.asarray(bundle.arrays["is_evt_tail"])[rows])
    _close(
        float(sample_ade[tail].mean()),
        evaluation["metrics"]["evt_tail"]["sample_mean_ADE_m"],
        "EVT-tail ADE",
    )
    cohort = np.load(output / "cohorts/cut_in.npz")
    cohort_manifest = load_json(output / "cohorts/cut_in.json")
    assert len(cohort["sequence_row"]) == cohort_manifest["window_count"]
    np.testing.assert_array_equal(
        cohort["split_index"],
        np.asarray(bundle.arrays["split_index"])[cohort["sequence_row"]],
    )
    assert int((cohort["split_index"] == 2).sum()) == (
        evaluation["metrics"]["crossing_aligned_cut_in"]["windows"]
    )
    endpoint_error = np.linalg.norm(
        np.asarray(per_sequence["generated_endpoint"])
        - np.asarray(per_sequence["target_endpoint"]),
        axis=-1,
    )
    active = np.asarray(per_sequence["active_background"], bool)
    assert endpoint_error.shape == active.shape == (9_841, 6)
    endpoint_fde = (endpoint_error * active).sum(axis=-1) / active.sum(axis=-1)
    _close(float(endpoint_fde.mean()), all_metrics["sample_mean_FDE_m"], "FDE")
    _close(float(endpoint_fde.mean()), all_metrics["min_FDE_m"], "min FDE")

    headline = evaluation["headline_metrics"]
    assert set(headline) == {
        "mean_single_draw_ADE_m",
        "longitudinal_velocity_MAE_mps",
        "lateral_velocity_MAE_mps",
        "mean_pairwise_sample_distance_m",
    }
    _close(
        headline["mean_single_draw_ADE_m"],
        all_metrics["sample_mean_ADE_m"],
        "headline single-draw ADE",
    )
    velocity_error = np.asarray(
        per_sequence["mean_velocity_vector_error_mps"], np.float64
    )
    assert velocity_error.shape == (9_841,)
    assert np.isfinite(velocity_error).all() and (velocity_error >= 0.0).all()
    pca_highd = np.asarray(per_sequence["trajectory_pca_highd"], np.float64)
    pca_diffusion = np.asarray(per_sequence["trajectory_pca_diffusion"], np.float64)
    pca_explained = np.asarray(
        per_sequence["trajectory_pca_explained_variance_ratio"], np.float64
    )
    assert pca_highd.shape == pca_diffusion.shape == (3_000, 3)
    assert pca_explained.shape == (3,)
    projection_rows = per_sequence["trajectory_pca_sequence_row"]
    projection_slots = per_sequence["trajectory_pca_agent_slot"]
    assert projection_rows.shape == projection_slots.shape == (3_000,)
    assert np.all((projection_slots >= 1) & (projection_slots <= 6))
    row_lookup = {int(row): local for local, row in enumerate(rows)}
    assert all(
        active[row_lookup[int(row)], int(slot) - 1]
        for row, slot in zip(projection_rows, projection_slots)
    )
    assert np.isfinite(pca_highd).all() and np.isfinite(pca_diffusion).all()
    assert np.isfinite(pca_explained).all() and (pca_explained > 0.0).all()
    assert "reconstruction_gate" not in evaluation
    assert "sparse_constraint_reference" not in evaluation["metrics"]
    assert "semantic_cut_in" not in evaluation["metrics"]
    assert "crossing_aligned_cut_in" in evaluation["metrics"]
    assert contract["name"] == evaluation["model"]
    physical_rates = evaluation["metrics"]["physical_feasibility"]["generated"]
    figure_names = {
        path.name for path in (output / "figures").iterdir() if path.is_file()
    }
    assert figure_names == {
        "training_curves.png",
        "evaluation_diagnostics.png",
    }

    return {
        "status": "PASS",
        "condition_dim": 68,
        "canonical_train_validation_test": [72_771, 13_133, 10_151],
        "trajectory_train_validation_test": [71_698, 12_857, 9_841],
        "checkpoint_epoch": training["best_epoch"],
        "mean_single_draw_ADE_m": headline["mean_single_draw_ADE_m"],
        "physical_violation_rates_all_zero": all(
            float(value) == 0.0 for value in physical_rates.values()
        ),
        "checkpoint_role": "compact EMA inference checkpoint",
        "future_ego_in_condition": False,
        "future_background_position_knots_disclosed": True,
        "trajectory_interpolation": True,
        "knots_hard_enforced": True,
    }


def main() -> None:
    report = verify(CONFIG)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

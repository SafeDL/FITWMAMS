#!/usr/bin/env python3
"""Verify the published full-data diffusion result against saved evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import load_data_bundle  # noqa: E402
from world_model.src.core.utils import load_json, load_yaml  # noqa: E402

DEFAULT_CONFIG = ROOT / "diffusion/configs/highd_background_diffusion.yaml"


def _close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(actual, expected, rtol=1.0e-7, atol=1.0e-9):
        raise AssertionError(f"{name}: {actual} != {expected}")


def _contract_hash(contract: dict[str, object]) -> str:
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify(config_path: Path) -> dict[str, object]:
    config = load_yaml(config_path)
    output = Path(config["paths"]["output_dir"]).resolve()
    contract = load_json(output / "dataset_contract.json")
    training = load_json(output / "training_summary.json")
    evaluation = load_json(output / "evaluation_summary.json")
    manifest = load_json(output / "manifest.json")
    per_sequence = np.load(output / "evaluation_per_sequence.npz")

    assert config["dataset"]["condition_mode"] == "c0_long_horizon_state_knots"
    assert config["dataset"]["include_ego_future"] is False
    assert int(config["dataset"]["max_train_sequences"]) == 0
    assert int(config["dataset"]["max_val_sequences"]) == 0
    assert int(config["evaluation"]["max_sequences"]) == 0
    assert contract["condition_dim"] == 40 + 6 + 6 * 12 == 118
    assert contract["ego_future_in_condition"] is False
    assert contract["target_representation"] == (
        "smooth_reference_relative_dx_dy_residual"
    )
    assert contract["trajectory_reference"] == "piecewise_cubic_hermite_2s_4s_end"
    assert evaluation["condition_disclosure"] == {
        "endpoint_is_conditioned": True,
        "interpretation": (
            "conditional trajectory reconstruction, not C0-only future prediction"
        ),
        "knot_times_s": [2.0, 4.0, 5.96],
        "uses_future_background_state_knots": True,
        "uses_future_ego": False,
    }

    assert training["experiment_scope"] == evaluation["experiment_scope"] == "full"
    assert training["train_sequences"] == 72_771
    assert training["validation_sequences"] == 13_133
    assert evaluation["metrics"]["all"]["sequences"] == 10_151
    assert training["best_epoch"] == evaluation["checkpoint_epoch"] == 50
    checkpoint_path = output / training["best_checkpoint"]
    assert checkpoint_path == output / "checkpoints/best_background_diffusion.pt"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == training["best_epoch"]
    assert checkpoint["dataset_contract"] == contract
    assert checkpoint["model_config"]["condition_dim"] == 118
    assert "ema_model_state" not in checkpoint
    assert "optimizer_state" not in checkpoint
    assert "scheduler_state" not in checkpoint

    all_metrics = evaluation["metrics"]["all"]
    sample_ade = np.asarray(per_sequence["sample_ade_m"], np.float64)
    sample_fde = np.asarray(per_sequence["sample_fde_m"], np.float64)
    assert sample_ade.shape == sample_fde.shape == (10_151, 4)
    assert np.isfinite(sample_ade).all() and np.isfinite(sample_fde).all()
    _close(float(sample_ade.mean()), all_metrics["sample_mean_ADE_m"], "sample ADE")
    _close(float(sample_fde.mean()), all_metrics["sample_mean_FDE_m"], "sample FDE")
    _close(float(sample_ade.min(axis=1).mean()), all_metrics["min_ADE_m"], "min ADE")
    _close(float(sample_fde.min(axis=1).mean()), all_metrics["min_FDE_m"], "min FDE")

    bundle = load_data_bundle(config, config_path.parent)
    rows = np.asarray(per_sequence["row_index"], np.int64)
    assert len(np.unique(rows)) == 10_151
    evaluated_ids = set(np.asarray(bundle.arrays["sequence_id"])[rows].astype(str))
    canonical_test = np.asarray(bundle.arrays["split_index"]) == 2
    canonical_ids = set(
        np.asarray(bundle.arrays["sequence_id"])[canonical_test].astype(str)
    )
    flow_test = np.asarray(bundle.flow_arrays["split_index"]) == 2
    flow_ids = set(np.asarray(bundle.flow_arrays["segment_id"])[flow_test].astype(str))
    assert evaluated_ids == canonical_ids == flow_ids

    headline = evaluation["headline_metrics"]
    historical = evaluation["reconstruction_gate"]["historical_target"]
    assert headline["ADE_m"] <= historical["ADE_m"]
    assert headline["FDE_m"] <= historical["FDE_m"]
    assert evaluation["reconstruction_gate"]["physical_feasibility_passed"]
    assert all(
        float(value) == 0.0
        for value in evaluation["metrics"]["physical_feasibility"]["generated"].values()
    )
    assert manifest["model"] == contract["name"] == evaluation["model"]
    assert manifest["contract_sha256"] == _contract_hash(contract)
    _close(manifest["evaluation"]["ensemble_ADE_m"], headline["ADE_m"], "manifest ADE")
    _close(manifest["evaluation"]["ensemble_FDE_m"], headline["FDE_m"], "manifest FDE")
    for relative in manifest["artifacts"].values():
        if not (output / relative).exists():
            raise FileNotFoundError(output / relative)

    return {
        "status": "PASS",
        "condition_dim": 118,
        "contract_sha256": manifest["contract_sha256"],
        "train_validation_test": [72_771, 13_133, 10_151],
        "checkpoint_epoch": 50,
        "ensemble_ADE_m": headline["ADE_m"],
        "ensemble_FDE_m": headline["FDE_m"],
        "physical_violation_rates_all_zero": True,
        "checkpoint_role": "compact EMA inference checkpoint",
        "future_ego_in_condition": False,
        "future_background_state_knots_disclosed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    report = verify(Path(args.config).resolve())
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

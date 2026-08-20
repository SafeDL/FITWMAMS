#!/usr/bin/env python3
"""Verify the published direct scenario-condition Flow artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import prepare_flow_condition  # noqa: E402
from normalizing_flow.src.data import load_natural_dataset  # noqa: E402
from normalizing_flow.src.sampling import (  # noqa: E402
    load_checkpoint_and_dataset,
    sample_constraints,
    sample_scenarios,
)
from normalizing_flow.src.utils import load_json  # noqa: E402

OUTPUT = ROOT / "results/highd_natural_driving_flow"
DIFFUSION_CONTRACT = ROOT / "results/background_diffusion/dataset_contract.json"


def main() -> None:
    arrays, schema = load_natural_dataset(OUTPUT)
    report = load_json(OUTPUT / "evaluation_summary.json")
    manifest = load_json(OUTPUT / "manifest.json")
    checkpoint = OUTPUT / "checkpoints/best_scenario_condition_flow.pt"
    model, _, _, _ = load_checkpoint_and_dataset(
        checkpoint, OUTPUT, repo_root=ROOT, device="cpu"
    )
    assert set(arrays) == {
        "features",
        "features_normalized",
        "feature_valid",
        "contexts",
        "slot_mask",
        "mask_pattern",
        "trajectory_constraint",
        "trajectory_constraint_normalized",
        "trajectory_constraint_valid",
        "split_index",
        "segment_id",
        "recording_id",
        "ego_id",
        "anchor_frame",
        "event_risk",
        "is_evt_tail",
    }
    assert len(arrays["split_index"]) == 96_055
    assert [int(np.sum(arrays["split_index"] == index)) for index in range(3)] == [
        72_771,
        13_133,
        10_151,
    ]
    assert schema["probability_factorization"] == "p(mask) p(C0|mask) p(K|C0,mask)"
    assert manifest["diffusion_condition"] == {
        "dimension": 118,
        "groups": {"C0": 40, "M": 6, "K": 72},
        "knot_times_s": [2.0, 4.0, 5.96],
        "contract_changed": False,
    }
    assert manifest["architecture"]["k_flow"] == {
        "num_bins": 8,
        "tail_bound": 5.0,
        "num_layers": 8,
        "hidden_features": 256,
        "num_blocks": 2,
        "dropout_probability": 0.02,
        "use_residual_blocks": True,
        "use_batch_norm": False,
    }
    assert report["all_quality_gates_passed"] is True
    expected_fields = {
        "c0",
        "slot_mask",
        "trajectory_constraint",
        "trajectory_constraint_valid",
        "c0_normalized_reference",
        "constraint_normalized_reference",
    }
    generated = np.load(report["generated_samples"])
    assert set(generated.files) == expected_fields

    first = sample_scenarios(model, 8, 1042)
    second = sample_scenarios(model, 8, 1042)
    np.testing.assert_allclose(first.c0, second.c0)
    np.testing.assert_allclose(first.trajectory_constraint, second.trajectory_constraint)
    changed = sample_constraints(
        model, first.c0[:1], first.slot_mask[:1], 8, 1043
    )
    assert not np.array_equal(
        np.repeat(first.trajectory_constraint[:1], 8, axis=0),
        changed.trajectory_constraint,
    )
    terms = model.log_prob(first)
    np.testing.assert_allclose(
        terms["joint_log_prob"],
        terms["mask_log_prob"] + terms["c0_log_prob"] + terms["k_log_prob"],
    )
    prepared = prepare_flow_condition(
        first,
        0,
        flow_schema=schema,
        diffusion_contract=load_json(DIFFUSION_CONTRACT),
    )
    assert prepared["condition"].shape == (118,)
    assert np.isfinite(prepared["condition"]).all()
    print(
        json.dumps(
            {
                "status": "PASS",
                "train_validation_test": [72_771, 13_133, 10_151],
                "joint_test_nll": report["held_out_nll"]["joint_nll"],
                "k_mean_ks": report["distribution"]["k"]["mean_ks"],
                "diffusion_condition_dim": 118,
                "seed_reproducible": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

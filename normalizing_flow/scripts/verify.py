"""Verify the maintained all-tail scenario Flow artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import prepare_flow_condition
from normalizing_flow.src.data import (
    dataset_dir_from_config,
    evt_tail_indices,
    load_scenario_dataset,
    output_dir_from_config,
)
from normalizing_flow.src.sampling import load_checkpoint_and_dataset
from normalizing_flow.src.utils import load_json, load_yaml

CONFIG = ROOT / "normalizing_flow/config.yaml"
DIFFUSION_CONTRACT = ROOT / "results/diffusion/dataset_contract.json"


def main() -> None:
    config = load_yaml(CONFIG)
    dataset_dir = dataset_dir_from_config(config, CONFIG.parent)
    output = output_dir_from_config(config, CONFIG.parent)
    arrays, schema = load_scenario_dataset(dataset_dir)
    rows = evt_tail_indices(arrays)
    report = load_json(output / "evaluation_summary.json")
    checkpoint = output / "checkpoints/scenario_flow.pt"
    model, _, _, payload = load_checkpoint_and_dataset(
        checkpoint,
        dataset_dir,
        repo_root=ROOT,
        device=report["sampling_device"],
    )
    assert len(arrays["segment_id"]) == 96_055
    assert len(rows) == 2_964
    assert arrays["features"].shape == (96_055, 26)
    assert arrays["trajectory_constraint"].shape == (96_055, 6, 6)
    assert schema["probability_factorization"] == "p(M) p(C0|M) p(K|C0,M)"
    assert payload["metrics"]["fit_scope"] == "all_evt_tail_events"
    assert payload["metrics"]["num_fit_events"] == 2_964
    assert model.dimension_bank.shape == (2_964, 7, 2)
    assert report["final_fit_events"] == report["target_events"] == 2_964
    assert report["cross_validation"]["protocol"].startswith("recording_group_5_fold")

    regenerated = model.sample_scenarios(report["target_events"], report["sample_seed"])
    saved = np.load(ROOT / report["generated_samples"], allow_pickle=False)
    for name, values in {
        "c0": regenerated.c0,
        "slot_mask": regenerated.slot_mask,
        "trajectory_constraint": regenerated.trajectory_constraint,
        "trajectory_constraint_valid": regenerated.trajectory_constraint_valid,
        "vehicle_dimensions": regenerated.vehicle_dimensions,
        "c0_normalized_reference": regenerated.c0_normalized_reference,
        "constraint_normalized_reference": regenerated.constraint_normalized_reference,
    }.items():
        np.testing.assert_array_equal(saved[name], values)

    first = model.sample_scenarios(8, 1042)
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
    assert prepared["condition"].shape == (68,)
    assert np.isfinite(prepared["condition"]).all()
    print(
        json.dumps(
            {
                "status": "PASS",
                "tail_events_used_for_final_fit": 2_964,
                "cross_validation_folds": 5,
                "diffusion_condition_dimensions": {"C0": 26, "M": 6, "K": 36},
                "empirical_vehicle_dimensions": 14,
                "saved_samples_reproduced": True,
                "sampling_device": report["sampling_device"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

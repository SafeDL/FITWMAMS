#!/usr/bin/env python3
"""Verify the maintained full model and its three-objective evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.core.utils import file_sha256, load_json, load_yaml  # noqa: E402

CONFIG = (
    ROOT
    / "hierarchical_traffic_world_model/configs/highd_hierarchical_world_model.yaml"
)


def main() -> None:
    config = load_yaml(CONFIG)
    output = Path(config["paths"]["output_dir"])
    evaluation = load_json(output / "evaluation.json")
    training = load_json(output / "training_summary.json")
    sampling = load_json(output / "sampling_hierarchy_evaluation.json")
    visual = load_json(output / "visualization_manifest.json")
    manifest = load_json(output / "manifest.json")
    full = load_json(output / "full_training_manifest.json")
    randomness = load_json(output / "randomness_ablation.json")

    assert config["training"]["experiment_scope"] == "full"
    assert training["experiment_scope"] == evaluation["experiment_scope"] == "full"
    assert evaluation["evaluation_schema_version"] == 2
    assert (training["train_sequences"], training["validation_sequences"]) == (
        72771,
        13133,
    )
    assert evaluation["test_sequences"] == full["test_sequences"] == 10151
    assert training["best_checkpoint_factual_gate_passed"] is True

    factual = evaluation["factual_fidelity"]
    guided = factual["diffusion_guided_hiqr"]
    assert guided["sequences"] == 10151 and guided["ADE_m"] < 0.05
    assert set(factual["event_strata"]) == {
        "all_natural",
        "evt_labelled",
        "semantic_cutin",
    }
    assert factual["without_long_horizon_constraint"]["ADE_m"] > 1.0
    distribution = evaluation["distribution_stochasticity"]
    assert distribution["mean_pairwise_trajectory_distance_m"] >= 0.02
    assert distribution["terminal_pairwise_distance_m"] > 0.0
    assert set(distribution["highd_adapted_realism"]["components"]) == {
        "speed_mps",
        "acceleration_magnitude_mps2",
        "yaw_rate_rps",
        "yaw_acceleration_rps2",
        "nearest_object_distance_m",
        "gap_m",
        "TTC_s",
        "collision_incidence",
    }
    assert randomness["cohort_sequences"] == 1024
    assert randomness["samples_per_condition"] == 16
    assert (
        randomness["quadrants"]["fully_deterministic"][
            "mean_pairwise_trajectory_distance_m"
        ]
        == 0.0
    )
    assert randomness["energy_improvement_fraction"] >= 0.05
    strict_intervention = True
    for name in ("brake", "accelerate"):
        value = evaluation["intervention_effectiveness"][name]
        assert value["direction_success_rate"] >= 0.90
        assert value["locality_ratio_far_to_near"] <= 0.20
        assert value["committed_response_invariant"] is True
        strict_intervention &= (
            value["dose_monotonicity_rate"] >= 0.95
            and value["response_within_natural_p10_p90_rate"] >= 0.50
        )
    assert evaluation["claims"]["counterfactual_correctness_proven"] is False
    assert evaluation["claims"]["official_WOSAC_score"] is False

    checkpoint_path = output / "checkpoints/best_hierarchical_world_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert manifest["checkpoint_sha256"] == file_sha256(checkpoint_path)
    assert checkpoint["experiment_scope"] == "full"
    assert (
        checkpoint["model_config"]["causal_response_scale"]
        == config["model"]["causal_response_scale"]
    )
    for name, value in full["selected_response_config"].items():
        assert checkpoint["model_config"][name] == value
        assert config["model"][name] == value
    assert sampling["experiment_scope"] == "full"
    assert sampling["seed_reproducible"] is True
    assert all(sampling["physical_validity"].values())
    assert sampling["test_c0_conditions"] == 64
    assert sampling["scenario_constraint_randomness"]["draws_per_condition"] == 16
    assert (
        sampling["scenario_constraint_randomness"][
            "closed_loop_terminal_pairwise_distance_m"
        ]["mean"]
        > 0.0
    )

    required = (
        checkpoint_path,
        output / "experiment_report.md",
        output / "randomness_ablation.json",
        output / "figures/training_diagnostics.png",
        output / "figures/three_objective_evaluation.png",
        output / "figures/event_fidelity.png",
        output / "figures/natural_response_calibration.png",
        output / "figures/factual_temporal_error.png",
        output / "figures/highd_adapted_distribution_realism.png",
        output / "figures/quality_diversity_tradeoff.png",
        output / "figures/intervention_dose_and_locality.png",
        output / "figures/trajectory_reconstruction.png",
    )
    assert all(path.is_file() for path in required)
    assert len(visual["playbacks"]) == 3
    assert all((output / path).is_file() for path in visual["playbacks"])
    playback = load_json(output / "playbacks/playback_manifest.json")
    for episode in playback["episodes"]:
        assert episode["playback_frames"] == 150
        image = Image.open(output / episode["gif"])
        assert image.n_frames == 150
        duration_ms = []
        for frame in range(image.n_frames):
            image.seek(frame)
            duration_ms.append(int(image.info.get("duration", 0)))
        assert sum(duration_ms) >= 6_000
    assert all((output / path).is_file() for path in manifest["artifacts"].values())
    assert manifest["three_objective_gates_passed"] is (
        strict_intervention and randomness["gates"]["all_passed"]
    )
    print(
        json.dumps(
            {
                "status": (
                    "PASS" if manifest["three_objective_gates_passed"] else "PARTIAL"
                ),
                "experiment_scope": "full",
                "train_validation_test": [72771, 13133, 10151],
                "checkpoint_epoch": evaluation["checkpoint_epoch"],
                "ADE_m": guided["ADE_m"],
                "FDE_m": guided["FDE_m"],
                "three_objective_gates_passed": manifest[
                    "three_objective_gates_passed"
                ],
                "strict_intervention_gate_passed": strict_intervention,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

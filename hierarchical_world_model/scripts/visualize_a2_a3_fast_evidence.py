#!/usr/bin/env python3
"""Summarize the focused A2/A3 V4 validation and test comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _w1(left: np.ndarray, right: np.ndarray) -> float:
    count = min(len(left), len(right))
    if not count:
        return float("nan")
    q = np.linspace(0., 1., count)
    return float(np.abs(np.quantile(left, q) - np.quantile(right, q)).mean())


def _metrics(root: Path, split: str) -> dict:
    evaluation = root / "evaluation"
    report = json.loads((evaluation / f"four_arm_comparison_{split}.json").read_text())
    with np.load(evaluation / f"counterfactual_telemetry_{split}.npz") as source:
        telemetry = {name: source[name].copy() for name in source.files}
    with np.load(evaluation / f"human_reference_telemetry_{split}.npz") as source:
        human = {name: source[name].copy() for name in source.files}
    result = {"rows": int(report["counterfactual_rows"]), "conditions": {}}
    for duration in (5, 15, 25):
        block = report["interventions"][f"duration_{duration}_frames"]["8"]
        condition = {}
        for code, arm in ((2, "A2"), (3, "A3")):
            aggregate = block[f"{arm}_rl_residual_{'idm' if arm == 'A2' else 'gail'}"]
            mask = (
                (telemetry["arm_code"] == code)
                & (telemetry["duration_frames"] == duration)
                & (telemetry["dose_mps2"] == 8.)
                & telemetry["active"].astype(bool)
            )
            condition[arm] = {
                "response_dose_mps2": float(aggregate["target_response_dose_mps2"]),
                "natural_kl": float(aggregate["target_natural_kl_mean"]),
                "collision_sequence_rate": float(aggregate["collision_sequence_rate"]),
                "rear_collision_sequence_rate": float(aggregate["target_rear_collision_sequence_rate"]),
                "positive_rebound_rate": float(aggregate["post_command_positive_acceleration_rebound_rate"]),
                "acceleration_w1_mps2": _w1(telemetry["final_ax_mps2"][mask], human["final_ax_mps2"]),
                "jerk_w1_mps3": _w1(telemetry["jerk_mps3"][mask], human["jerk_mps3"]),
                "telemetry_active_samples": int(mask.sum()),
            }
        condition["relative"] = {
            "response_ratio": condition["A3"]["response_dose_mps2"] / condition["A2"]["response_dose_mps2"],
            "kl_improvement": 1. - condition["A3"]["natural_kl"] / condition["A2"]["natural_kl"],
            "acceleration_w1_improvement": 1. - condition["A3"]["acceleration_w1_mps2"] / condition["A2"]["acceleration_w1_mps2"],
            "jerk_w1_improvement": 1. - condition["A3"]["jerk_w1_mps3"] / condition["A2"]["jerk_w1_mps3"],
        }
        result["conditions"][str(duration)] = condition
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_dir.resolve()
    evidence = root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    values = {split: _metrics(root, split) for split in ("validation", "test")}
    durations = np.asarray((.2, .6, 1.0))
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = {"A2": "#5e81ac", "A3": "#d08770"}
    styles = {"validation": "--", "test": "-"}
    for split in ("validation", "test"):
        for arm in ("A2", "A3"):
            response = [values[split]["conditions"][str(frame)][arm]["response_dose_mps2"] for frame in (5, 15, 25)]
            axes[0, 0].plot(durations, response, marker="o", linestyle=styles[split], color=colors[arm], label=f"{arm} {split}")
        kl = [100. * values[split]["conditions"][str(frame)]["relative"]["kl_improvement"] for frame in (5, 15, 25)]
        axes[0, 1].plot(durations, kl, marker="o", linestyle=styles[split], label=split)
    axes[0, 1].axhline(10., color="#bf616a", linestyle=":", label="required 10%")
    axes[0, 0].set(title="Strong-brake causal response", xlabel="ego brake duration (s)", ylabel="response dose (m/s²)")
    axes[0, 1].set(title="A3 improvement over A2: final-action KL", xlabel="ego brake duration (s)", ylabel="improvement (%)")

    x = np.arange(6)
    labels = []
    acceleration, jerk = [], []
    for split in ("validation", "test"):
        for frame in (5, 15, 25):
            labels.append(f"{split[:3]}\n{frame/25:.1f}s")
            relative = values[split]["conditions"][str(frame)]["relative"]
            acceleration.append(100. * relative["acceleration_w1_improvement"])
            jerk.append(100. * relative["jerk_w1_improvement"])
    axes[1, 0].bar(x - .18, acceleration, .36, label="acceleration W1")
    axes[1, 0].bar(x + .18, jerk, .36, label="jerk W1")
    axes[1, 0].axhline(10., color="#bf616a", linestyle=":", label="required 10%")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set(title="A3 improvement over A2: highD distance", ylabel="improvement (%)")

    for split in ("validation", "test"):
        for arm in ("A2", "A3"):
            collision = [100. * values[split]["conditions"][str(frame)][arm]["rear_collision_sequence_rate"] for frame in (5, 15, 25)]
            axes[1, 1].plot(durations, collision, marker="o", linestyle=styles[split], color=colors[arm], label=f"{arm} {split}")
    axes[1, 1].set(title="Controlled-NPC collision rate", xlabel="ego brake duration (s)", ylabel="sequences (%)")
    for axis in axes.flat:
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    figure.savefig(evidence / "a2_a3_v4_focused_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    acceptance = {
        split: {
            "kl_improvement_at_least_10_percent_all_conditions": all(
                item["relative"]["kl_improvement"] >= .10 for item in values[split]["conditions"].values()
            ),
            "jerk_w1_improvement_at_least_10_percent_all_conditions": all(
                item["relative"]["jerk_w1_improvement"] >= .10 for item in values[split]["conditions"].values()
            ),
            "response_at_least_85_percent_all_conditions": all(
                item["relative"]["response_ratio"] >= .85 for item in values[split]["conditions"].values()
            ),
        } for split in values
    }
    accepted = all(all(flags.values()) for flags in acceptance.values())
    payload = {
        "schema": "a2_a3_gail_v4_fast_validation_v1",
        "evaluation": values,
        "acceptance": acceptance,
        "accepted": accepted,
        "selected_controller": "A3_rl_residual_gail" if accepted else "A2_rl_residual_idm",
        "promotion": "candidate only; formal artifacts unchanged" if not accepted else "eligible for explicit promotion",
    }
    (evidence / "a2_a3_v4_acceptance.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(evidence / "a2_a3_v4_focused_comparison.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render the four-panel audit figure for the frozen HumanActionPriorV4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from world_model.src.core.utils import ensure_dir, save_json  # noqa: E402

DEFAULT = ROOT / "hierarchical_world_model/config/reaction_naturalistic.yaml"


def _conditional_mae(action: np.ndarray, prediction: np.ndarray, condition: np.ndarray,
                     edges: tuple[float, ...]) -> list[float]:
    result = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (condition >= low) & (condition < high)
        result.append(float(np.abs(action[mask] - prediction[mask]).mean()) if mask.any() else np.nan)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize formal highD longitudinal GAIL-V4 evidence.")
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    args = parser.parse_args()
    config = load_protocol_config(args.config.resolve())
    root = Path(config["paths"]["human_prior"]).parent if args.artifact_dir is None else args.artifact_dir
    summary = json.loads((root / "training_summary.json").read_text())
    output = ensure_dir(root / "evidence")
    bc = [item for item in summary["history"] if item["phase"] == "bc"]
    refine = summary["highway_gail_ppo"]
    if not refine:
        raise RuntimeError("V4 evidence requires at least one adversarial GAIL pass")
    passes = np.asarray([item["pass"] + 1 for item in refine])
    with np.load(root / "training_distribution_samples.npz") as values:
        distributions = {name: values[name].copy() for name in values.files}

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    loss_axis = axes[0, 0]
    loss_axis.plot(passes, [item["policy_loss"] for item in refine], marker="o", label="actor PPO loss")
    loss_axis.plot(passes, [item["value_loss"] for item in refine], marker="o", label="critic loss")
    loss_axis.plot(passes, [item["discriminator_loss"] for item in refine], marker="o", label="discriminator loss")
    loss_axis.plot(passes, [item["generator_reward"] for item in refine], marker="o", label="generator reward")
    loss_axis.set(title="GAIL optimization (complete highD passes)", xlabel="pass", ylabel="objective / reward")

    auc_axis = axes[0, 1]
    auc_axis.plot(passes, [item["discriminator_auc"] for item in refine], marker="o", label="expert vs generated AUC")
    auc_axis.plot(passes, [item["label_shuffle_auc"] for item in refine], marker=".", label="shuffled-label sanity AUC")
    auc_axis.axhline(.5, color="#4c566a", linestyle="--", label="chance")
    auc_axis.axhline(.75, color="#bf616a", linestyle=":", label="acceptance ceiling")
    auc_axis.set(title="Adversarial occupancy distinguishability", xlabel="pass", ylabel="AUC", ylim=(0., 1.))

    acceleration_axis = axes[1, 0]
    for key, label, color in (
        ("highd_action_mps2", "highD expert", "#2e3440"),
        ("bc_action_mps2", "BC rollout", "#5e81ac"),
        ("gail_action_mps2", "GAIL rollout", "#d08770"),
    ):
        acceleration_axis.hist(distributions[key], bins=60, range=(-8., 4.), density=True,
                               histtype="step", linewidth=1.8, label=label, color=color)
    acceleration_axis.set(title="Closed-loop longitudinal acceleration", xlabel="acceleration (m/s²)", ylabel="density")

    jerk_axis = axes[1, 1]
    for key, label, color in (
        ("highd_jerk_mps3", "highD expert", "#2e3440"),
        ("bc_jerk_mps3", "BC rollout", "#5e81ac"),
        ("gail_jerk_mps3", "GAIL rollout", "#d08770"),
    ):
        clipped = np.clip(distributions[key], 0., 100.)
        jerk_axis.hist(clipped, bins=60, range=(0., 100.), density=True,
                       histtype="step", linewidth=1.8, label=label, color=color)
    jerk_axis.set(title="Closed-loop jerk and conditional action error", xlabel="|jerk| (m/s³, clipped at 100)", ylabel="density")

    conditional = {}
    validation_path = root / "heldout_validation_prior_samples.npz"
    bc_path = root / "heldout_validation_prior_samples_bc.npz"
    if validation_path.exists() and bc_path.exists():
        with np.load(validation_path) as gail_values, np.load(bc_path) as bc_values:
            action = gail_values["action_mps2"]
            gap, ttc = gail_values["gap_m"], gail_values["ttc_s"]
            inset = jerk_axis.inset_axes([.49, .46, .48, .48])
            positions = np.arange(6)
            for source, label, color, shift in (
                (bc_values["predicted_mean_mps2"], "BC", "#5e81ac", -.12),
                (gail_values["predicted_mean_mps2"], "GAIL", "#d08770", .12),
            ):
                errors = _conditional_mae(action, source, ttc, (0., 2., 4., 10.01))
                errors += _conditional_mae(action, source, gap, (-1000., 10., 25., 1000.))
                inset.bar(positions + shift, errors, width=.24, color=color, alpha=.8, label=label)
                conditional[label] = errors
            inset.set_xticks(positions, ("T<2", "T2–4", "T≥4", "g<10", "g10–25", "g≥25"), rotation=45, fontsize=7)
            inset.set_ylabel("MAE (m/s²)", fontsize=7)
            inset.tick_params(labelsize=7)
            inset.legend(fontsize=7)

    for axis in axes.flat:
        axis.grid(alpha=.22)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    figure.savefig(output / "gail_v4_four_panel_evidence.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    selected_pass = int(distributions["selected_pass"])
    selected_entry = next(item for item in refine if int(item["pass"]) == selected_pass)
    save_json({
        "schema": "formal_longitudinal_gail_evidence_v4",
        "training_rows": summary["training_rows"],
        "expert_samples": summary["expert_samples"],
        "adversarial_passes": len(refine),
        "best_selected_pass": selected_pass,
        "best_discriminator_auc": float(selected_entry["discriminator_auc"]),
        "conditional_mean_action_mae_mps2": conditional,
        "bc_epochs": int(summary.get("bc_epochs", len(bc))),
    }, output / "gail_evidence_metrics.json")
    print(output / "gail_v4_four_panel_evidence.png")


if __name__ == "__main__":
    main()

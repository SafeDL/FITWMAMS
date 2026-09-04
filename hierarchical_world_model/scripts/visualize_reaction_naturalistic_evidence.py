#!/usr/bin/env python3
"""Create auditable training, causal-response, and naturalness evidence.

The reaction study deliberately keeps raw JSON/NPZ artifacts.  This script is
the presentation layer: it never changes a checkpoint, and every plotted
number is regenerated from an artifact produced by the training/evaluation
entry points.
"""

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
ARMS = ("A0_none", "A1_rl_residual", "A2_rl_residual_idm", "A3_rl_residual_gail")
ARM_DIR = {"A1_rl_residual": "rl_residual", "A2_rl_residual_idm": "rl_residual_idm", "A3_rl_residual_gail": "rl_residual_gail"}
COLORS = {"A0_none": "#4c566a", "A1_rl_residual": "#5e81ac", "A2_rl_residual_idm": "#a3be8c", "A3_rl_residual_gail": "#bf616a"}
ARM_CODE = {arm: index for index, arm in enumerate(ARMS)}


def _load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _arm_mask(data: dict[str, np.ndarray], arm: str) -> np.ndarray:
    """Support legacy string telemetry and compact full-split arm codes."""
    if "arm_code" in data:
        return data["arm_code"] == ARM_CODE[arm]
    return data["arm"] == arm


def _smooth(values: np.ndarray, window: int = 9) -> np.ndarray:
    if len(values) < window:
        return values
    radius = window // 2
    padded = np.pad(values, (radius, window - radius - 1), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _w1(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.sort(np.asarray(left, float)), np.sort(np.asarray(right, float))
    if not len(left) or not len(right):
        return float("nan")
    q = np.linspace(0., 1., min(len(left), len(right)))
    return float(np.mean(np.abs(np.quantile(left, q) - np.quantile(right, q))))


def _ks(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.sort(np.asarray(left, float)), np.sort(np.asarray(right, float))
    if not len(left) or not len(right):
        return float("nan")
    values = np.sort(np.concatenate((left, right)))
    return float(np.max(np.abs(np.searchsorted(left, values, side="right") / len(left) - np.searchsorted(right, values, side="right") / len(right))))


def _save(figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_training(root: Path, human_root: Path, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for arm, directory in ARM_DIR.items():
        history = _load(root / "controllers" / directory / "training_summary.json")["history"]
        update = np.asarray([item["update"] for item in history])
        axes[0, 0].plot(update, _smooth(np.asarray([item["loss"] for item in history])), label=arm, color=COLORS[arm])
        axes[0, 1].plot(update, _smooth(np.asarray([item["reward"] for item in history])), label=arm, color=COLORS[arm])
    axes[0, 0].set(title="PPO objective (9-update moving mean)", xlabel="PPO update", ylabel="loss")
    axes[0, 1].set(title="Rollout reward (9-update moving mean)", xlabel="PPO update", ylabel="reward")
    prior = _load(human_root / "training_summary.json")
    history = prior["history"]
    for phase, color in (("bc", "#5e81ac"), ("gail", "#bf616a")):
        rows = [item for item in history if item["phase"] == phase]
        axes[1, 0].plot([item["epoch"] for item in rows], [item["loss"] for item in rows], marker="o", label=phase.upper(), color=color)
    axes[1, 0].set(title="Human prior BC/GAIL objective", xlabel="epoch", ylabel="loss")
    refine = prior.get("highway_gail_ppo", [])
    if refine:
        x = [item.get("update", item.get("pass", index)) for index, item in enumerate(refine)]
        axes[1, 1].plot(x, [item["discriminator_loss"] for item in refine], marker="o", label="discriminator", color="#d08770")
        axes[1, 1].plot(x, [item["bc_anchor"] for item in refine], marker="o", label="BC anchor", color="#a3be8c")
    axes[1, 1].set(title="HighwayEnv GAIL refinement", xlabel="update", ylabel="loss")
    for axis in axes.flat:
        axis.grid(alpha=.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels)
    _save(figure, output / "training_dynamics.png")


def plot_rules(rule_root: Path, output: Path) -> None:
    rules = _load(rule_root / "highd_global_idm_mobil.json")
    names = ("v0", "a max", "comfort brake", "s0", "headway", "delta")
    values = np.asarray(rules["idm_parameters"], float)
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)
    positions = np.arange(len(names))
    axes[0].plot(positions, values, marker="o", label="one global calibrated IDM")
    axes[0].set(xticks=positions, xticklabels=names, title="highD global IDM parameters", ylabel="native parameter value")
    mobil_names = ("politeness", "incentive threshold", "safe brake")
    mobil_values = (rules["mobil_politeness"], rules["mobil_incentive_threshold_mps2"], rules["mobil_safe_brake_mps2"])
    axes[1].bar(mobil_names, mobil_values, color=("#5e81ac", "#a3be8c", "#d08770"))
    axes[1].set(title="MOBIL calibration (diagnostic only)", ylabel="calibrated value")
    axes[1].tick_params(axis="x", rotation=15)
    for axis in axes:
        axis.grid(alpha=.25, axis="y"); axis.legend() if axis is axes[0] else None
    _save(figure, output / "rule_calibration_evidence.png")


def plot_human_prior_fit(human_root: Path, output: Path) -> None:
    """Show a held-out prior fit, separate from A3's PPO reward signal."""
    path = human_root / "heldout_validation_prior_samples.npz"
    if not path.exists():
        return
    with np.load(path) as data:
        action, predicted, ttc = data["action_mps2"], data["predicted_mean_mps2"], data["ttc_s"]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    axes[0].hist(action, bins=60, range=(-8, 4), density=True, histtype="step", linewidth=2, label="held-out highD")
    axes[0].hist(predicted, bins=60, range=(-8, 4), density=True, histtype="step", linewidth=2, label="GAIL prior mean")
    axes[0].set(title="Held-out human acceleration distribution", xlabel="acceleration (m/s²)", ylabel="density")
    bins = ((0., 2.), (2., 4.), (4., 10.01))
    labels, mae = [], []
    for low, high in bins:
        mask = (ttc >= low) & (ttc < high)
        labels.append(f"{low:g}–{high if high < 10 else '∞'}")
        mae.append(float(np.abs(action[mask] - predicted[mask]).mean()) if mask.any() else np.nan)
    axes[1].bar(labels, mae, color="#5e81ac")
    axes[1].set(title="Prior action MAE by held-out TTC", xlabel="TTC bin (s)", ylabel="MAE (m/s²)")
    for axis in axes:
        axis.grid(alpha=.25, axis="y"); axis.legend() if axis is axes[0] else None
    _save(figure, output / "human_prior_heldout_fit.png")


def plot_causal(report: dict, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    doses = (2, 4, 6, 8)
    for duration, linestyle in ((5, ":"), (15, "--"), (25, "-")):
        block = report["interventions"][f"duration_{duration}_frames"]
        for arm in ARMS:
            dose_response = [block[str(dose)][arm]["target_response_dose_mps2"] for dose in doses]
            axes[0, 0].plot(doses, dose_response, marker="o", linestyle=linestyle, color=COLORS[arm], label=f"{arm}, {duration / 25:.1f}s")
        # Focus collision comparisons on the causal rear vehicle, excluding
        # the ego's independent front collision from the same-rear claim.
        for arm in ARMS:
            collision = [block[str(dose)][arm]["target_rear_collision_sequence_rate"] for dose in doses]
            axes[0, 1].plot(doses, collision, marker="o", linestyle=linestyle, color=COLORS[arm], label=f"{arm}, {duration / 25:.1f}s")
    final = report["interventions"]["duration_25_frames"]
    for arm in ARMS:
        delay = [final[str(dose)][arm]["first_effective_response_delay_frames"] / 25. for dose in doses]
        locality = [final[str(dose)][arm]["unrelated_max_abs_correction_mps2"] for dose in doses]
        axes[1, 0].plot(doses, delay, marker="o", color=COLORS[arm], label=arm)
        axes[1, 1].plot(doses, locality, marker="o", color=COLORS[arm], label=arm)
    axes[0, 0].set(title="Causal response dose", xlabel="observed ego brake magnitude (m/s²)", ylabel="NPC extra brake (m/s²)")
    axes[0, 1].set(title="Target rear collision rate", xlabel="observed ego brake magnitude (m/s²)", ylabel="sequence rate")
    axes[1, 0].set(title="First effective response latency", xlabel="observed ego brake magnitude (m/s²)", ylabel="seconds; −0.04 means no response")
    axes[1, 1].set(title="Locality: unrelated-vehicle correction", xlabel="observed ego brake magnitude (m/s²)", ylabel="max |Δa| (m/s²)")
    for axis in axes.flat:
        axis.grid(alpha=.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[0, 1].legend(fontsize=7, ncol=2)
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    _save(figure, output / "causal_response_evidence.png")


def plot_naturalness(counterfactual: np.lib.npyio.NpzFile, human: np.lib.npyio.NpzFile, output: Path) -> dict:
    data = {name: counterfactual[name] for name in counterfactual.files}
    expert = {name: human[name] for name in human.files}
    # Compare like-with-like.  The highD reference and generated actions are
    # selected within identical causal TTC bins rather than as an invalid
    # unconditional histogram comparison.
    bins = ((0., 2., "TTC 0–2 s"), (2., 4., "TTC 2–4 s"), (4., 10.01, "TTC ≥4 s"))
    figure, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    minimum_samples = 128
    for column, (low, high, title) in enumerate(bins):
        expert_mask = (expert["ttc_s"] >= low) & (expert["ttc_s"] < high)
        action_ref, jerk_ref = expert["final_ax_mps2"][expert_mask], expert["jerk_mps3"][expert_mask]
        axes[0, column].hist(action_ref, bins=40, density=True, histtype="step", linewidth=2, color="#2e3440", label=f"highD (n={len(action_ref)})")
        axes[1, column].hist(np.clip(jerk_ref, 0, 100), bins=40, density=True, histtype="step", linewidth=2, color="#2e3440", label=f"highD (n={len(jerk_ref)})")
        metrics[title] = {}
        for arm in ARMS:
            generated_mask = (_arm_mask(data, arm) & (data["duration_frames"] == 25) & (data["dose_mps2"] == 8.) &
                              (data["ttc_s"] >= low) & (data["ttc_s"] < high))
            # An A0 action exists even without control authority.  For
            # learned arms use all post-observation target actions too: this
            # evaluates what the traffic participant actually did, not only
            # the policy's selected subset.
            action, jerk = data["final_ax_mps2"][generated_mask], data["jerk_mps3"][generated_mask]
            axes[0, column].hist(action, bins=40, range=(-8, 4), density=True, histtype="step", linewidth=1.5, color=COLORS[arm], label=f"{arm} (n={len(action)})")
            axes[1, column].hist(np.clip(jerk, 0, 100), bins=40, range=(0, 100), density=True, histtype="step", linewidth=1.5, color=COLORS[arm], label=f"{arm} (n={len(jerk)})")
            sufficient = len(action_ref) >= minimum_samples and len(action) >= minimum_samples
            metrics[title][arm] = {
                "status": "ok" if sufficient else "insufficient_matched_samples",
                "minimum_samples_required": minimum_samples,
                "action_w1_mps2": _w1(action_ref, action) if sufficient else None,
                "action_ks": _ks(action_ref, action) if sufficient else None,
                "jerk_w1_mps3": _w1(jerk_ref, jerk) if sufficient else None,
                "jerk_ks": _ks(jerk_ref, jerk) if sufficient else None,
                "highd_samples": int(len(action_ref)), "generated_samples": int(len(action)),
            }
        axes[0, column].set(title=title, xlabel="longitudinal acceleration (m/s²)", ylabel="density")
        axes[1, column].set(xlabel="|jerk| (m/s³, clipped at 100)", ylabel="density")
        axes[0, column].grid(alpha=.2); axes[1, column].grid(alpha=.2)
    axes[0, 0].legend(fontsize=7); axes[1, 0].legend(fontsize=7)
    _save(figure, output / "conditional_human_distribution_evidence.png")
    return metrics


def plot_observed_brake_naturalness(counterfactual: np.lib.npyio.NpzFile, human: np.lib.npyio.NpzFile, output: Path) -> dict:
    """Compare against held-out rear responses to a real front-brake event."""
    data, expert = ({name: counterfactual[name] for name in counterfactual.files}, {name: human[name] for name in human.files})
    minimum, metrics = 128, {}
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.2), constrained_layout=True)
    for row, dose in enumerate((2, 4)):
        for ttc_band, axis in zip((1, 2), axes[row]):
            expert_mask = (expert["dose_band"] == dose) & (expert["ttc_band"] == ttc_band)
            reference = expert["final_ax_mps2"][expert_mask]
            label = ("2–4 s" if ttc_band == 1 else "≥4 s")
            if len(reference):
                axis.hist(reference, bins=40, range=(-8, 4), density=True, histtype="step", linewidth=2, color="#2e3440", label=f"highD observed brake (n={len(reference)})")
            key = f"ego_brake_{dose}_{dose + 2}_ttc_{label}"
            metrics[key] = {}
            for arm in ARMS:
                ttc_mask = ((data["ttc_s"] >= 2.) & (data["ttc_s"] < 4.)) if ttc_band == 1 else (data["ttc_s"] >= 4.)
                generated_mask = (_arm_mask(data, arm) & (data["duration_frames"] == 15) & (data["dose_mps2"] == dose) &
                                  (data["step"] >= 27) & (data["step"] <= 35) & ttc_mask)
                generated = data["final_ax_mps2"][generated_mask]
                enough = len(reference) >= minimum and len(generated) >= minimum
                if len(generated):
                    axis.hist(generated, bins=40, range=(-8, 4), density=True, histtype="step", linewidth=1.5, color=COLORS[arm], label=f"{arm} (n={len(generated)})")
                metrics[key][arm] = {"status": "ok" if enough else "insufficient_matched_samples", "highd_samples": int(len(reference)), "generated_samples": int(len(generated)), "action_w1_mps2": _w1(reference, generated) if enough else None, "action_ks": _ks(reference, generated) if enough else None}
            axis.set(title=f"highD front brake {dose}–{dose + 2} m/s²; TTC {label}", xlabel="rear acceleration (m/s²)", ylabel="density")
            axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=6)
    _save(figure, output / "observed_brake_human_distribution_evidence.png")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Render auditable evidence for the naturalistic NPC reaction study.")
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--artifact-dir", type=Path, default=None,
                        help="candidate/formal root containing controllers/ and evaluation/")
    parser.add_argument("--human-prior-dir", type=Path, default=None)
    parser.add_argument("--rule-dir", type=Path, default=None)
    args = parser.parse_args()
    config = load_protocol_config(args.config.resolve())
    root = (args.artifact_dir if args.artifact_dir is not None else Path(config["paths"]["output_dir"])).resolve()
    human_root = (args.human_prior_dir if args.human_prior_dir is not None else Path(config["paths"]["human_prior"]).parent).resolve()
    rule_root = (args.rule_dir if args.rule_dir is not None else Path(config["paths"]["rule_model"]).parent).resolve()
    evaluation = root / "evaluation"
    output = ensure_dir(root / "evidence" / args.split)
    report = _load(evaluation / f"four_arm_comparison_{args.split}.json")
    plot_training(root, human_root, output)
    plot_rules(rule_root, output)
    plot_human_prior_fit(human_root, output)
    plot_causal(report, output)
    reference = human_root / f"heldout_{args.split}_human_reference.npz"
    brake_reference = human_root / f"heldout_{args.split}_brake_response_reference.npz"
    if not reference.exists():
        raise FileNotFoundError(
            f"missing balanced held-out human reference {reference}; run evaluate_human_driving_prior.py --split {args.split} first"
        )
    if not brake_reference.exists():
        raise FileNotFoundError(f"missing observed-brake reference {brake_reference}; rerun evaluate_human_driving_prior.py")
    with np.load(evaluation / f"counterfactual_telemetry_{args.split}.npz") as counterfactual, np.load(reference) as human:
        metrics = plot_naturalness(counterfactual, human, output)
    with np.load(evaluation / f"counterfactual_telemetry_{args.split}.npz") as counterfactual, np.load(brake_reference) as human:
        observed_brake_metrics = plot_observed_brake_naturalness(counterfactual, human, output)
    save_json({
        "schema": "reaction_naturalness_distribution_evidence_v1", "split": args.split,
        "comparison": "held-out highD versus post-observation HighwayEnv target-NPC actions, matched by realised TTC bin",
        "counterfactual_condition": "-8 m/s² ego brake for 1.0 s", "minimum_matched_samples": 128,
        "metrics": metrics,
    }, output / "conditional_distribution_metrics.json")
    save_json({"schema": "reaction_observed_brake_distribution_evidence_v1", "split": args.split, "comparison": "held-out highD rear response following an observed front brake, versus the corresponding post-observation counterfactual response window", "minimum_matched_samples": 128, "metrics": observed_brake_metrics}, output / "observed_brake_distribution_metrics.json")
    print(output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Held-out factual validation and visualization for calibrated IDM/MOBIL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.rule_models import RuleModelBundle, _idm_numpy  # noqa: E402
from world_model.src.core.utils import ensure_dir, save_json  # noqa: E402


DEFAULT = ROOT / "hierarchical_world_model/config/reaction_naturalistic.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the highD IDM reference on a held-out split; MOBIL remains diagnostic-only.")
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--limit", type=int, default=None, help="debug only; formal validation must omit this")
    args = parser.parse_args()
    config = load_protocol_config(args.config.resolve())
    base = load_protocol_config(ROOT / config.get("base_config", "hierarchical_world_model/config/release.yaml"))
    experiment = prepare_experiment_data(base, ROOT)
    split_rows = experiment.validation_rows if args.split == "validation" else experiment.test_rows
    rows = split_rows if args.limit is None else split_rows[:args.limit]
    arrays = experiment.bundle.arrays
    states = np.asarray(arrays["agent_states"])[rows]
    valid = np.asarray(arrays["agent_valid"])[rows]
    rules = RuleModelBundle.load(config["paths"]["rule_model"])
    expected, observed, posterior = [], [], []
    with torch.no_grad():
        for frame in range(25, min(173, states.shape[1] - 1)):
            rule_action, style = rules.idm_reference(
                torch.from_numpy(states[:, frame - 24:frame + 1]), torch.from_numpy(states[:, frame]),
                torch.from_numpy(valid[:, frame]),
            )
            leader, rear = states[:, frame, 0], states[:, frame, 2]
            gap = leader[:, 0] - rear[:, 0] - 4.8
            mask = (valid[:, frame, 0] & valid[:, frame, 2] & valid[:, frame + 1, 2] &
                    (gap > .1) & (np.abs(leader[:, 1] - rear[:, 1]) < 1.8) & (leader[:, 0] > rear[:, 0]))
            expected.append(rule_action.numpy()[mask])
            observed.append(np.clip((states[:, frame + 1, 2, 2] - rear[:, 2]) / .04, -8., 4.)[mask])
            posterior.append(style.numpy()[mask])
    expected, observed, posterior = np.concatenate(expected), np.concatenate(observed), np.concatenate(posterior)
    error = expected - observed
    output = ensure_dir(Path(config["paths"]["rule_model"]).parent)
    report = {
        "schema": "highd_global_idm_heldout_validation_v2", "split": args.split, "rows": int(len(rows)), "samples": int(len(expected)),
        "full_split": args.limit is None,
        "acceleration_mae_mps2": float(np.abs(error).mean()), "acceleration_bias_mps2": float(error.mean()),
        "acceleration_rmse_mps2": float(np.sqrt(np.square(error).mean())),
        "correlation": float(np.corrcoef(expected, observed)[0, 1]) if np.std(expected) > 1.e-8 and np.std(observed) > 1.e-8 else 0.,
        "global_model_utilisation": int(len(posterior)),
        "mobil_scope": "calibrated diagnostic only; no lateral action is evaluated in this longitudinal study.",
    }
    save_json(report, output / f"heldout_{args.split}_rule_metrics.json")
    np.savez_compressed(output / f"heldout_{args.split}_idm_samples.npz", idm_ax_mps2=expected, highd_ax_mps2=observed, style_posterior=posterior)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    pick = np.linspace(0, len(expected) - 1, min(8000, len(expected))).astype(int)
    axes[0].scatter(observed[pick], expected[pick], s=2, alpha=.16, color="#5e81ac")
    axes[0].plot((-8, 4), (-8, 4), "--", color="#bf616a", label="perfect match")
    axes[0].set(title="Held-out IDM acceleration fit", xlabel="highD acceleration (m/s²)", ylabel="IDM reference (m/s²)")
    axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].hist(observed, bins=60, range=(-8, 4), density=True, histtype="step", linewidth=2, label="held-out highD")
    axes[1].hist(expected, bins=60, range=(-8, 4), density=True, histtype="step", linewidth=2, label="IDM")
    axes[1].set(title="Held-out IDM action distribution", xlabel="acceleration (m/s²)", ylabel="density")
    axes[1].legend(); axes[1].grid(alpha=.25)
    figure.savefig(output / f"heldout_{args.split}_idm_fit.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    # Paper-inspired calibration evidence: the optimizer trace and an
    # observed-vs-simulated one-second following example.  The latter feeds
    # the observed leader trace to IDM, so it tests the longitudinal response
    # law without conflating it with the frozen world-model rollout bridge.
    summary_path = output / "calibration_summary.json"
    summary = __import__("json").loads(summary_path.read_text()) if summary_path.exists() else {}
    trace = summary.get("report", {}).get("history", [])
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), constrained_layout=True)
    if trace:
        axes[0].plot([item["epoch"] for item in trace], [item["robust_acceleration_loss"] for item in trace], color="#5e81ac", marker="o")
    axes[0].set(title="Global IDM calibration objective", xlabel="full-data epoch", ylabel="robust acceleration loss")
    # Find three valid factual sequences and roll follower speed/gap for 1 s.
    shown = 0
    for sequence in range(len(states)):
        if shown == 3:
            break
        frame = 30
        leader, rear = states[sequence, frame, 0], states[sequence, frame, 2]
        if not (valid[sequence, frame:frame + 26, 0].all() and valid[sequence, frame:frame + 26, 2].all() and leader[0] > rear[0] and abs(leader[1] - rear[1]) < 1.8):
            continue
        leader_trace = states[sequence, frame:frame + 26, 0]
        rear_trace = states[sequence, frame:frame + 26, 2]
        gap_sim = [float(leader_trace[0, 0] - rear_trace[0, 0] - 4.8)]
        speed_sim = float(np.linalg.norm(rear_trace[0, 2:4])); follower_x = float(rear_trace[0, 0])
        for item in range(25):
            leader_speed = float(np.linalg.norm(leader_trace[item, 2:4]))
            action = float(np.clip(_idm_numpy(
                np.asarray(rules.idm_parameters), np.asarray((max(leader_trace[item, 0] - follower_x - 4.8, .1),)), np.asarray((speed_sim,)), np.asarray((leader_speed,)))[0], -8., 4.))
            speed_sim = max(0., speed_sim + .04 * action); follower_x += .04 * speed_sim
            gap_sim.append(float(leader_trace[item + 1, 0] - follower_x - 4.8))
        time = np.arange(26) * .04
        axes[1].plot(time, leader_trace[:, 0] - rear_trace[:, 0] - 4.8, color="#4c566a", alpha=.45)
        axes[1].plot(time, gap_sim, "--", color="#bf616a", alpha=.8)
        shown += 1
    axes[1].plot([], [], color="#4c566a", label="held-out highD gap")
    axes[1].plot([], [], "--", color="#bf616a", label="IDM simulated gap")
    axes[1].set(title="Held-out 1 s following rollouts", xlabel="time (s)", ylabel="space headway (m)")
    for axis in axes:
        axis.grid(alpha=.25); axis.legend()
    figure.savefig(output / f"heldout_{args.split}_idm_calibration_evidence.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(report)


if __name__ == "__main__":
    main()

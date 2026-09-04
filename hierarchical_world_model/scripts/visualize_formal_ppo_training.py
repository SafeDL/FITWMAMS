#!/usr/bin/env python3
"""Render separate auditable PPO convergence evidence for A1/A2/A3."""
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
ARMS = (("A1 pure Residual PPO", "rl_residual", "#5e81ac"), ("A2 + global IDM", "rl_residual_idm", "#a3be8c"), ("A3 + IDM + frozen GAIL", "rl_residual_gail", "#bf616a"))

def smooth(x: np.ndarray, width: int = 15) -> np.ndarray:
    """Moving mean without zero-padding artifacts at the plot boundaries."""
    if len(x) < width:
        return x
    radius = width // 2
    padded = np.pad(x, (radius, width - radius - 1), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--artifact-dir", type=Path, default=None,
                        help="candidate/formal artifact root containing controllers/")
    args = parser.parse_args()
    config = load_protocol_config(args.config.resolve())
    root = (args.artifact_dir if args.artifact_dir is not None else Path(config["paths"]["output_dir"])).resolve()
    histories = {label: json.loads((root / "controllers" / directory / "training_summary.json").read_text())["history"] for label, directory, _ in ARMS}
    output = ensure_dir(root / "evidence")
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    panels = (("reward", "rollout reward"), ("value_loss", "critic value loss"), ("policy_loss", "policy surrogate loss"), ("entropy", "policy entropy"), ("clip_fraction", "PPO clip fraction"), ("natural_kl_mean", "final-action KL to frozen GAIL"))
    for axis, (key, title) in zip(axes.flat, panels):
        for label, _, color in ARMS:
            values = np.asarray([row[key] for row in histories[label]], float)
            axis.plot(np.arange(1, len(values) + 1), smooth(values), label=label, color=color)
        axis.set(title=title, xlabel="PPO update"); axis.grid(alpha=.25)
    axes[0, 0].legend(fontsize=7); figure.savefig(output / "ppo_training_convergence.png", dpi=180, bbox_inches="tight"); plt.close(figure)
    validation_figure, validation_axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    validation_panels = (
        ("validation_reward", "validation reward"),
        ("validation_response_dose_mps2", "validation response dose (m/s²)"),
        ("validation_collision_sequence_rate", "validation collision sequence rate"),
    )
    for axis, (key, title) in zip(validation_axes, validation_panels):
        for label, _, color in ARMS:
            rows = [row for row in histories[label] if key in row]
            axis.plot([row["update"] + 1 for row in rows], [row[key] for row in rows], marker="o", color=color, label=label)
        axis.set(title=title, xlabel="PPO update after complete scene pass"); axis.grid(alpha=.25)
    validation_axes[0].legend(fontsize=7)
    validation_figure.savefig(output / "ppo_validation_checkpoint_selection.png", dpi=180, bbox_inches="tight")
    plt.close(validation_figure)
    save_json({"schema": "formal_reaction_ppo_training_evidence_v2", "arms": {
        label: {"updates": len(history), "eligible_scenes_seen": history[-1]["eligible_scenes_seen"],
                "validation_checks": int(sum("validation_reward" in row for row in history)),
                "finite": all(np.isfinite([row[key] for row in history]).all() for key, _ in panels)}
        for label, history in histories.items()
    }}, output / "ppo_training_evidence_metrics.json")
    print(output / "ppo_training_convergence.png")

if __name__ == "__main__": main()

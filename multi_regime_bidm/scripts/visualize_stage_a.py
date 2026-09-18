#!/usr/bin/env python3
"""Paper-oriented diagnostics for train-only style discovery and offline regimes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesian_ma_idm.src.data import load_ragged_pairs
from multi_regime_bidm.src.style import decision_grid

COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c")


def _segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    edge = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True])
    return [(int(start), int(stop), int(labels[start])) for start, stop in zip(edge[:-1], edge[1:])]


def render(dataset: str | Path, stage_dir: str | Path, output_dir: str | Path) -> list[Path]:
    stage, output = Path(stage_dir), Path(output_dir)
    report = json.loads((stage / "stage_a_report.json").read_text(encoding="utf-8"))
    archive = np.load(stage / "stage_a_offline_labels.npz", allow_pickle=False)
    features, style = np.asarray(archive["style_feature_raw"]), np.asarray(archive["style_id"])
    centers = np.asarray(report["style_centers_raw_sorted_by_mean_headway"])
    output.mkdir(parents=True, exist_ok=True); paths: list[Path] = []

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.25))
    pairs = ((0, 1, "mean headway [s]", "max acceleration [m/s²]"),
             (0, 2, "mean headway [s]", "max deceleration [m/s²]"),
             (1, 2, "max acceleration [m/s²]", "max deceleration [m/s²]"))
    for axis, (x, y, xlabel, ylabel) in zip(axes, pairs):
        for identifier, color in enumerate(COLORS):
            axis.scatter(features[style == identifier, x], features[style == identifier, y], s=16, alpha=.65,
                         color=color, label=f"style rank {identifier}")
        axis.scatter(centers[:, x], centers[:, y], marker="X", s=105, color="black", label="train center")
        axis.set(xlabel=xlabel, ylabel=ylabel)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("Train-only highD style discovery (paper Fig. 5/6 analogue; ranks are not semantic labels)", y=1.03)
    fig.tight_layout(); path = output / "fig_style_clusters_train_only.png"; fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig); paths.append(path)

    _, all_pairs = load_ragged_pairs(dataset)
    offsets, pair_index = np.asarray(archive["regime_offsets"]), np.asarray(archive["pair_index"])
    packed = np.asarray(archive["offline_regime"])
    # Pick the event with the most state transitions, then show the whole
    # offline segmentation explicitly as a retrospective diagnostic.
    labels_all = [packed[offsets[i]:offsets[i + 1]] for i in range(len(pair_index))]
    chosen = int(np.argmax([np.count_nonzero(np.diff(labels)) for labels in labels_all]))
    labels, pair = labels_all[chosen], all_pairs[int(pair_index[chosen])]
    observation, _ = decision_grid(pair); time = np.arange(len(labels)) * .2
    fig, axes = plt.subplots(3, 1, figsize=(10, 5.4), sharex=True)
    names = ("gap [m]", "speed [m/s]", "closing speed [m/s]")
    for axis, values, name in zip(axes, observation.T, names):
        for begin, end, state in _segments(labels):
            axis.axvspan(time[begin], time[min(end - 1, len(time) - 1)] + .2, color=COLORS[state], alpha=.18, lw=0)
        axis.plot(time, values, color="#202020", lw=.75); axis.set_ylabel(name)
    axes[0].legend([Line2D([], [], color=color, lw=6, alpha=.4) for color in COLORS],
                   [f"offline regime {state}" for state in range(3)], fontsize=8, loc="best")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"Offline explicit-duration segmentation: highD follower #{int(pair['follower_id'])} "
                 "(retrospective only; not online filtering)", y=.995)
    fig.tight_layout(); path = output / "fig_offline_regime_segmentation.png"; fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9), sharey=True)
    for state, axis in enumerate(axes):
        durations = [end - begin for labels in labels_all for begin, end, value in _segments(labels) if value == state]
        axis.hist(np.asarray(durations) * .2, bins=24, color=COLORS[state], alpha=.8)
        axis.set(title=f"offline regime {state}", xlabel="duration [s]")
    axes[0].set_ylabel("segments")
    fig.suptitle("Explicit-duration diagnostics (all train events; truncated at 20 s)", y=1.02)
    fig.tight_layout(); path = output / "fig_regime_duration_distribution.png"; fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig); paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "multi_regime_bidm/evidence/dataset/highd_multi_regime_cohort_25hz.npz"))
    parser.add_argument("--stage-dir", default=str(ROOT / "multi_regime_bidm/evidence/segmentation"))
    parser.add_argument("--output-dir", default=str(ROOT / "multi_regime_bidm/evidence/segmentation/figures"))
    args = parser.parse_args()
    print(*render(args.dataset, args.stage_dir, args.output_dir), sep="\n")


if __name__ == "__main__":
    main()

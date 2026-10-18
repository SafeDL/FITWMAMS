#!/usr/bin/env python3
"""Aggregate fixed-event ADS evaluations with event-level uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _percentile_interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, (0.025, 0.975))]


def bootstrap_summary(
    frame: pd.DataFrame,
    *,
    seed: int,
    replicates: int,
    cluster_column: str | None = None,
) -> dict[str, object]:
    if not len(frame):
        raise ValueError("evaluation frame is empty")
    rng = np.random.default_rng(seed)
    clusters = (
        frame[cluster_column].astype(str).to_numpy()
        if cluster_column is not None
        else np.arange(len(frame)).astype(str)
    )
    unique_clusters, cluster_index = np.unique(clusters, return_inverse=True)
    draws = rng.integers(
        0, len(unique_clusters), size=(replicates, len(unique_clusters))
    )

    def clustered_means(values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        sums = np.bincount(
            cluster_index[finite], weights=values[finite], minlength=len(unique_clusters)
        )
        counts = np.bincount(
            cluster_index[finite], minlength=len(unique_clusters)
        )
        return sums[draws].sum(axis=1) / counts[draws].sum(axis=1).clip(min=1)

    failure = frame["failure"].astype(float).to_numpy()
    progress = frame["progress_m"].astype(float).to_numpy()
    reward = frame["reward"].astype(float).to_numpy()
    distance = frame["minimum_distance_m"].astype(float).to_numpy()
    finite_distance = distance[np.isfinite(distance)]
    boot_failure = clustered_means(failure)
    boot_progress = clustered_means(progress)
    boot_reward = clustered_means(reward)
    boot_distance = clustered_means(distance)
    return {
        "events": int(len(frame)),
        "bootstrap_unit": cluster_column or "event_id",
        "bootstrap_cluster_count": int(len(unique_clusters)),
        "failures": int(failure.sum()),
        "failure_rate": float(failure.mean()),
        "failure_rate_ci95": _percentile_interval(boot_failure),
        "mean_progress_m": float(progress.mean()),
        "mean_progress_m_ci95": _percentile_interval(boot_progress),
        "mean_reward": float(reward.mean()),
        "mean_reward_ci95": _percentile_interval(boot_reward),
        "mean_minimum_distance_m": float(finite_distance.mean()),
        "minimum_distance_event_count": int(len(finite_distance)),
        "mean_minimum_distance_m_ci95": _percentile_interval(boot_distance),
        "collision_rate": float(frame["collision"].astype(float).mean()),
        "offroad_rate": float(frame["offroad"].astype(float).mean()),
        "stalled_rate": float(frame["stalled"].astype(float).mean()),
        "mean_abs_acceleration_mps2": float(
            frame["mean_abs_acceleration_mps2"].mean()
        ),
        "mean_action_switches": float(frame["action_switches"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation",
        action="append",
        nargs=2,
        metavar=("LABEL", "CSV"),
        required=True,
    )
    parser.add_argument(
        "--manifest",
        action="append",
        nargs=2,
        metavar=("LABEL", "CSV"),
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()
    frames: dict[str, pd.DataFrame] = {}
    manifests = {label: path for label, path in args.manifest}
    summaries = {}
    for index, (label, path) in enumerate(args.evaluation):
        if label in frames:
            raise ValueError(f"duplicate evaluation label {label!r}")
        frame = pd.read_csv(path)
        if frame["event_id"].duplicated().any():
            raise ValueError(f"{label} contains duplicate event IDs")
        cluster_column = None
        if label in manifests:
            manifest = pd.read_csv(manifests[label])
            frame = frame.merge(
                manifest[["event_id", "recording_id", "event_risk"]],
                on="event_id",
                how="left",
                validate="one_to_one",
            )
            if frame["recording_id"].isna().any():
                raise ValueError(f"{label} manifest does not cover every event")
            cluster_column = "recording_id"
        frames[label] = frame
        summaries[label] = bootstrap_summary(
            frame,
            seed=args.seed + index,
            replicates=args.bootstrap_replicates,
            cluster_column=cluster_column,
        )
    labels = list(frames)
    overlaps = {}
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            overlaps[f"{left}:{right}"] = len(
                set(frames[left]["source_event_id"].astype(str))
                & set(frames[right]["source_event_id"].astype(str))
            )
    result = {
        "bootstrap_seed": args.seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_resampling": (
            "recording_cluster" if manifests else "event"
        ),
        "summaries": summaries,
        "source_event_overlap": overlaps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure observed highD support for sustained leader-acceleration changes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_world_model.scripts.evaluate_online_ads import _screen  # noqa: E402
from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from traffic_components.src.core.utils import save_json  # noqa: E402


def main() -> None:
    experiment = prepare_experiment_data(
        load_protocol_config(ROOT / "hierarchical_world_model/config/world_model.yaml"),
        ROOT,
    )
    arrays = experiment.bundle.arrays
    states = np.asarray(arrays["agent_states"][experiment.test_rows], np.float32)
    valid = np.asarray(arrays["agent_valid"][experiment.test_rows], bool)
    selected, receiver = _screen(states, valid, "same")
    index = np.arange(len(states))
    gap = (
        states[index, ANCHOR_INDEX, 0, 0]
        - states[index, ANCHOR_INDEX, receiver + 1, 0]
        - 4.8
    )
    near = np.flatnonzero(selected & (gap < 45.0))
    ego = states[near, :, 0]
    follower = states[near, :, receiver[near] + 1]
    lane_stable = (
        (np.abs(ego[:, 100, 1] - ego[:, 25, 1]) < 0.5)
        & (np.abs(follower[:, 100, 1] - follower[:, 25, 1]) < 0.5)
    )
    leader_change = ego[:, 50:75, 4].mean(axis=1) - ego[:, 25:50, 4].mean(axis=1)
    follower_change = (
        follower[:, 55:80, 4].mean(axis=1)
        - follower[:, 25:50, 4].mean(axis=1)
    )
    thresholds = {}
    for threshold in (0.25, 0.5, 1.0, 2.0):
        groups = {}
        for direction, mask in (
            ("increase", lane_stable & (leader_change > threshold)),
            ("decrease", lane_stable & (leader_change < -threshold)),
        ):
            groups[direction] = {
                "scenes": int(mask.sum()),
                "median_follower_change_mps2": (
                    None if not mask.any() else float(np.median(follower_change[mask]))
                ),
            }
        thresholds[str(threshold)] = groups
    report = {
        "schema": "natural_leader_acceleration_support_v1",
        "full_test_rows": len(experiment.test_rows),
        "same_lane_rear_scenes_within_45m": int(len(near)),
        "stable_lane_scenes": int(lane_stable.sum()),
        "leader_change_windows": "mean ax frames 50:75 minus 25:50",
        "follower_change_windows": "mean ax frames 55:80 minus 25:50",
        "thresholds_mps2": thresholds,
        "interpretation": (
            "This stable-lane subset contains no one-second mean leader-acceleration "
            "increase above 1 m/s², so it does not calibrate the magnitude of forced "
            "+2/+4 m/s² ADS responses."
        ),
    }
    output = ROOT / "results/hierarchical_world_model/evaluation/natural_leader_acceleration_support.json"
    save_json(report, output)
    print(report)


if __name__ == "__main__":
    main()

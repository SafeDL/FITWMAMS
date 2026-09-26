"""Offline train-only MA-IDM paired response cache for Idea A."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from external_model_baselines.models.bayesian_ma_idm.src.model import (
    load_posterior,
    sample_driver_joint,
)
from external_model_baselines.models.bayesian_ma_idm.src.reference_kernels import idm

from .causal_cache import load_causal_cache
from .manifest import load_benchmark_config


def _stimulus_trajectory(
    current: np.ndarray, logged: np.ndarray, extra_acceleration: float = -3.0
) -> np.ndarray:
    output = np.asarray(logged, np.float32).copy()
    state = np.asarray(current, np.float64).copy()
    for frame in range(len(output)):
        acceleration = float(logged[frame, 4]) + (
            float(extra_acceleration) if frame < 25 else 0.0
        )
        state[0] += state[2] * 0.04 + 0.5 * acceleration * 0.04**2
        state[2] = max(0.0, state[2] + acceleration * 0.04)
        state[4] = acceleration
        output[frame, 0] = state[0]
        output[frame, 2] = state[2]
        output[frame, 4] = acceleration
    return output


def _teacher_rollout(
    current_follower, leader_future, leader_length, follower_length, theta
):
    follower = np.asarray(current_follower, np.float64).copy()
    features = []
    for decision, endpoint in enumerate(range(4, 75, 5)):
        leader = leader_future[min(decision * 5, 74)]
        gap = float(leader[0] - follower[0] - 0.5 * (leader_length + follower_length))
        acceleration = float(
            np.clip(
                idm(gap, follower[2], follower[2] - leader[2], theta[:5]), -8.0, 4.0
            )
        )
        for _ in range(5):
            follower[0] += follower[2] * 0.04 + 0.5 * acceleration * 0.04**2
            follower[2] = max(0.0, follower[2] + acceleration * 0.04)
        endpoint_leader = leader_future[endpoint]
        endpoint_gap = float(
            endpoint_leader[0] - follower[0] - 0.5 * (leader_length + follower_length)
        )
        features.append(
            (follower[2], endpoint_gap, acceleration, follower[1] - current_follower[1])
        )
    return np.asarray(features, np.float32)


def build_response_pair_cache(
    config_path: str | Path, teacher_futures: int = 4
) -> dict:
    config, _ = load_benchmark_config(config_path)
    output = Path(config["paths"]["output_dir"])
    path = output / "data/response_pairs_ma_idm_train_v3.npz"
    report_path = output / "data/response_pairs_ma_idm_train_v3.json"
    if path.exists() and report_path.exists():
        return json.loads(report_path.read_text())
    arrays, metadata, _ = load_causal_cache(config_path)
    events = pd.read_csv(output / "event_manifest.csv")
    events = events[
        (events.split_index == 0)
        & (events.event_type == "following_brake")
        & (events.response_agent_index > 0)
    ].reset_index(drop=True)
    posterior_path = (
        Path(__file__).resolve().parents[2]
        / "external_model_baselines/evaluation/driver_reproduction/artifacts/matched/ma_idm_train25.npz"
    )
    posterior = load_posterior(posterior_path)
    n = len(events)
    teacher = np.zeros((n, teacher_futures, 15, 4), np.float32)
    teacher_baseline = np.zeros_like(teacher)
    teacher_reactive = np.zeros_like(teacher)
    stimulus = np.zeros((n, 75, 6), np.float32)
    for index, event in enumerate(events.itertuples(index=False)):
        row, onset = int(event.scenario_row), int(event.local_onset_frame)
        leader, follower = int(event.stimulus_agent_index), int(
            event.response_agent_index
        )
        current = np.asarray(arrays["agent_states"][row, onset])
        logged = np.asarray(arrays["agent_states"][row, onset + 1 : onset + 76, leader])
        intervention = _stimulus_trajectory(current[leader], logged)
        stimulus[index] = intervention
        for future in range(teacher_futures):
            rng = np.random.default_rng(
                np.random.SeedSequence([20260919, index, future])
            )
            theta = sample_driver_joint(posterior, rng)
            baseline = _teacher_rollout(
                current[follower],
                logged,
                metadata["lengths_m"][row, leader],
                metadata["lengths_m"][row, follower],
                theta,
            )
            reactive = _teacher_rollout(
                current[follower],
                intervention,
                metadata["lengths_m"][row, leader],
                metadata["lengths_m"][row, follower],
                theta,
            )
            teacher_baseline[index, future] = baseline
            teacher_reactive[index, future] = reactive
            teacher[index, future] = reactive - baseline
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        event_id=np.asarray(events.event_id.astype(str).tolist(), dtype="U128"),
        scenario_row=events.scenario_row.to_numpy(np.int64),
        local_onset_frame=events.local_onset_frame.to_numpy(np.int64),
        stimulus_agent_index=events.stimulus_agent_index.to_numpy(np.int64),
        response_agent_index=events.response_agent_index.to_numpy(np.int64),
        intervention_stimulus_states=stimulus,
        teacher_response_delta=teacher,
        teacher_baseline_features=teacher_baseline,
        teacher_reactive_features=teacher_reactive,
    )
    report = {
        "cache_format": "ma_idm_response_delta_pairs_v3",
        "events": n,
        "teacher_futures_per_event": teacher_futures,
        "teacher": "MA-IDM train-only posterior",
        "teacher_posterior": str(posterior_path),
        "stimulus": "additional -3 m/s2 for 1.0 s",
        "response_features": [
            "speed_mps",
            "net_gap_m",
            "acceleration_mps2",
            "lateral_displacement_m",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report

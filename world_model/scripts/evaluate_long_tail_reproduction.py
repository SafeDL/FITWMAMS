#!/usr/bin/env python3
"""Replay held-out highD tail events and compare world-model risk dynamics.

The protocol fixes the logged 1 s history, B0 condition, map and observed ego
replay for every held-out EVT-tail sequence.  It then produces one deterministic
trajectory and K stochastic closed-loop futures from each model.  CAT-TopK is
kept as its archived, information-asymmetric baseline: its START call receives
the frozen future-action summary recorded in the legacy cache.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.scripts.compare_semi_markov_cat_topk import (
    _batch,
    _catk_multichunk_rollout,
    _legacy_sequences,
)
from world_model.src.core.data import dataset_dir_from_config, load_world_model_dataset
from world_model.src.core.initial_behavior_anchor import FrozenLegacyFlowSchema
from world_model.src.cat_topk.model import load_checkpoint as load_catk_checkpoint
from world_model.src.ramp.train import load_ramp_checkpoint
from world_model.src.semi_markov.train import load_semi_markov_checkpoint
from world_model.src.core.sequential_dataset import (
    ensure_frozen_flow_behavior_anchor_cache,
    load_sequential_dataset,
    sequence_cache_owner_dir,
)
from world_model.src.core.utils import (
    ensure_dir,
    load_yaml,
    save_json,
    select_device,
    set_seed,
)

DT = 0.04
EPS = 1.0e-6
QUANTILES = (0.90, 0.95, 0.99)
MAX_DISTRIBUTION_POINTS = 100_000
EVENTS = (
    "high_risk_following",
    "hard_braking",
    "high_speed_approach",
    "close_interaction",
    "strong_relative_speed_change",
)


def _finite(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, np.float64)[np.isfinite(values)]


def _distribution_sample(values: np.ndarray) -> np.ndarray:
    """Bound empirical comparisons without changing the scenario/branch set.

    Full K=32 rollout pools contain millions of highly correlated 40 ms
    points.  A fixed evenly spaced subsample is sufficient for empirical CDF
    metrics, makes the post-processing memory-bounded, and remains exactly
    reproducible.  Trajectory, event and tail metrics still use every sample.
    """
    values = _finite(values)
    if len(values) <= MAX_DISTRIBUTION_POINTS:
        return values
    index = np.linspace(0, len(values) - 1, MAX_DISTRIBUTION_POINTS, dtype=np.int64)
    return values[index]


def _mean(values: np.ndarray) -> float:
    values = _finite(values)
    return float(values.mean()) if len(values) else float("nan")


def _empirical_distance(real: np.ndarray, generated: np.ndarray) -> dict[str, Any]:
    """W1, KS and tail quantiles without a SciPy dependency."""
    real, generated = _distribution_sample(real), _distribution_sample(generated)
    if not len(real) or not len(generated):
        return {"available": False}
    grid = np.linspace(0.0, 1.0, max(len(real), len(generated)))
    rq = np.quantile(real, grid)
    gq = np.quantile(generated, grid)
    merged = np.sort(np.concatenate((real, generated)))
    cdf_real = np.searchsorted(np.sort(real), merged, side="right") / len(real)
    cdf_gen = np.searchsorted(np.sort(generated), merged, side="right") / len(generated)
    result: dict[str, Any] = {
        "available": True,
        "num_real": int(len(real)),
        "num_generated": int(len(generated)),
        "wasserstein_1": float(np.mean(np.abs(rq - gq))),
        "ks": float(np.max(np.abs(cdf_real - cdf_gen))),
    }
    result["quantiles"] = {
        f"q{int(q * 100)}": {
            "real": float(np.quantile(real, q)),
            "generated": float(np.quantile(generated, q)),
            "absolute_error": float(
                abs(np.quantile(real, q) - np.quantile(generated, q))
            ),
        }
        for q in QUANTILES
    }
    return result


def _risk_fields(
    background: np.ndarray, ego: np.ndarray, valid: np.ndarray
) -> dict[str, np.ndarray]:
    """Physical safety variables for [episode, time, background-slot, state]."""
    states = np.asarray(background, np.float32)
    ego = np.asarray(ego, np.float32)
    valid = np.asarray(valid, bool)
    rel_x = states[..., 0] - ego[:, :, None, 0]
    rel_vx = states[..., 2] - ego[:, :, None, 2]
    gap = np.maximum(np.abs(rel_x) - 4.5, 0.0)
    closing = np.maximum(-rel_vx, 0.0)
    ttc = np.where(closing > EPS, gap / np.maximum(closing, EPS), 10.0)
    ttc = np.clip(ttc, 0.0, 10.0)
    drac = np.where(closing > EPS, np.square(closing) / np.maximum(2.0 * gap, 0.1), 0.0)
    speed = np.linalg.norm(states[..., 2:4], axis=-1)
    acceleration = states[..., 4]
    delta_speed = np.zeros_like(speed)
    delta_speed[:, 1:] = np.diff(speed, axis=1)
    jerk = np.zeros_like(acceleration)
    jerk[:, 1:] = np.diff(acceleration, axis=1) / DT
    risk = (
        np.maximum(3.0 - ttc, 0.0) / 3.0
        + 0.5 * np.log1p(drac)
        + np.maximum(8.0 - gap, 0.0) / 8.0
        + np.maximum(-acceleration - 2.0, 0.0) / 4.0
    )
    return {
        "valid": valid,
        "speed_mps": speed,
        "lateral_speed_mps": states[..., 3],
        "delta_speed_mps": delta_speed,
        "acceleration_mps2": acceleration,
        "jerk_mps3": jerk,
        "gap_m": gap,
        "ttc_s": ttc,
        "drac_mps2": drac,
        "relative_speed_mps": rel_vx,
        "closing_speed_mps": closing,
        "risk": risk,
    }


def _episode_extrema(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    valid = fields["valid"]

    def reduce(name: str, method: str) -> np.ndarray:
        values = np.where(valid, fields[name], np.nan)
        fn = np.nanmin if method == "min" else np.nanmax
        with np.errstate(all="ignore"):
            return fn(values, axis=(1, 2))

    return {
        "min_acceleration_mps2": reduce("acceleration_mps2", "min"),
        "min_gap_m": reduce("gap_m", "min"),
        "min_ttc_s": reduce("ttc_s", "min"),
        "max_drac_mps2": reduce("drac_mps2", "max"),
        "max_closing_speed_mps": reduce("closing_speed_mps", "max"),
        "risk_score": reduce("risk", "max"),
    }


def _variable_values(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    valid = fields["valid"]
    result = {
        name: np.asarray(fields[name])[valid]
        for name in (
            "speed_mps",
            "delta_speed_mps",
            "acceleration_mps2",
            "jerk_mps3",
            "gap_m",
            "ttc_s",
            "drac_mps2",
            "relative_speed_mps",
            "closing_speed_mps",
        )
    }
    result.update(_episode_extrema(fields))
    return result


def _event_masks(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    extrema = _episode_extrema(fields)
    valid = fields["valid"]
    relative = fields["relative_speed_mps"]
    # Preserve slot identity: extrema from different vehicles do not
    # constitute a relative-speed change for either vehicle.
    slot_present = valid.any(axis=1)
    high = np.where(valid, relative, -np.inf).max(axis=1)
    low = np.where(valid, relative, np.inf).min(axis=1)
    per_slot_delta = np.where(slot_present, high - low, -np.inf)
    strongest_delta = per_slot_delta.max(axis=1)
    return {
        "high_risk_following": extrema["min_ttc_s"] < 3.0,
        "hard_braking": extrema["min_acceleration_mps2"] < -1.5,
        "high_speed_approach": extrema["max_closing_speed_mps"] > 5.0,
        "close_interaction": extrema["min_gap_m"] < 8.0,
        "strong_relative_speed_change": strongest_delta > 3.0,
    }


def _trajectory_metrics(
    samples: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> dict[str, Any]:
    distance = np.linalg.norm(samples[..., :2] - target[None, ..., :2], axis=-1)
    weight = valid[None].astype(np.float32)
    denom = weight.sum(axis=(2, 3)).clip(min=1.0)
    ade = (distance * weight).sum(axis=(2, 3)) / denom
    final = valid[:, -1]
    final_denom = final.sum(axis=1).clip(min=1)
    fde = (distance[:, :, -1] * final[None]).sum(axis=2) / final_denom[None]
    return {
        "ADE_m": _mean(ade[0]),
        "FDE_m": _mean(fde[0]),
        "sample_mean_ADE_m": _mean(ade),
        "sample_mean_FDE_m": _mean(fde),
        "minADE_at_K_m": _mean(np.min(ade, axis=0)),
        "minFDE_at_K_m": _mean(np.min(fde, axis=0)),
        "per_episode_min_fde_m": np.min(fde, axis=0),
    }


def _pairwise_mean(values: np.ndarray) -> float:
    count = len(values)
    if count < 2:
        return float("nan")
    distances = []
    for left in range(count):
        for right in range(left + 1, count):
            distances.append(np.mean(np.abs(values[left] - values[right])))
    return _mean(np.asarray(distances))


def _diversity(
    samples: np.ndarray, ego: np.ndarray, valid: np.ndarray, min_fde: np.ndarray
) -> dict[str, Any]:
    fields = [_risk_fields(sample, ego, valid) for sample in samples]
    endpoints = samples[:, :, -1, :, :2]
    endpoint_distance = np.linalg.norm(endpoints[:, None] - endpoints[None, :], axis=-1)
    upper = np.triu_indices(len(samples), k=1)
    pair_fde = (
        endpoint_distance[upper].mean(axis=(1, 2)) if len(upper[0]) else np.array([])
    )
    gap_paths = np.stack([item["gap_m"] for item in fields])
    ttc_paths = np.stack([item["ttc_s"] for item in fields])
    return {
        "average_pairwise_FDE_m": _mean(pair_fde),
        "average_pairwise_gap_distance_m": _pairwise_mean(gap_paths),
        "average_pairwise_ttc_distance_s": _pairwise_mean(ttc_paths),
        "coverage": {
            f"minFDE_le_{threshold:g}m": float(np.mean(min_fde <= threshold))
            for threshold in (1.0, 2.0, 5.0)
        },
    }


def _correlation(fields: dict[str, np.ndarray]) -> np.ndarray:
    valid = fields["valid"]
    values = np.stack(
        (fields["speed_mps"], fields["acceleration_mps2"], fields["gap_m"]), axis=-1
    )
    rows = values[valid]
    if len(rows) < 3:
        return np.full((3, 3), np.nan)
    return np.corrcoef(rows, rowvar=False)


def _conditional_brake(
    fields: dict[str, np.ndarray], states: np.ndarray
) -> dict[str, Any]:
    """Rear response conditioned on the nearest same-lane front vehicle brake."""
    valid = fields["valid"]
    acceleration = fields["acceleration_mps2"]
    x, y = states[..., 0], states[..., 1]
    pairs_front: list[float] = []
    pairs_rear: list[float] = []
    for episode in range(len(states)):
        for frame in range(states.shape[1]):
            active = np.flatnonzero(valid[episode, frame])
            for rear in active:
                front = active[
                    (x[episode, frame, active] > x[episode, frame, rear])
                    & (
                        np.abs(y[episode, frame, active] - y[episode, frame, rear])
                        < 1.8
                    )
                ]
                if len(front):
                    nearest = front[
                        np.argmin(x[episode, frame, front] - x[episode, frame, rear])
                    ]
                    pairs_front.append(float(acceleration[episode, frame, nearest]))
                    pairs_rear.append(float(acceleration[episode, frame, rear]))
    if not pairs_front:
        return {"available": False}
    front = np.asarray(pairs_front)
    rear = np.asarray(pairs_rear)
    bins = np.asarray((-8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0))
    return {
        "available": True,
        "num_pairs": int(len(front)),
        "pearson_correlation": (
            float(np.corrcoef(front, rear)[0, 1]) if len(front) > 1 else float("nan")
        ),
        "front_acceleration_bin_edges_mps2": bins.tolist(),
        "mean_rear_acceleration_mps2": [
            _mean(rear[(front >= lower) & (front < upper)])
            for lower, upper in zip(bins[:-1], bins[1:])
        ],
    }


def _interaction(
    fields: dict[str, np.ndarray],
    states: np.ndarray,
    reference: dict[str, np.ndarray],
    reference_states: np.ndarray,
) -> dict[str, Any]:
    correlation = _correlation(fields)
    reference_correlation = _correlation(reference)
    return {
        "delta_gap_distribution": _empirical_distance(
            np.diff(reference["gap_m"], axis=1)[
                reference["valid"][:, 1:] & reference["valid"][:, :-1]
            ],
            np.diff(fields["gap_m"], axis=1)[
                fields["valid"][:, 1:] & fields["valid"][:, :-1]
            ],
        ),
        "delta_relative_speed_distribution": _empirical_distance(
            np.diff(reference["relative_speed_mps"], axis=1)[
                reference["valid"][:, 1:] & reference["valid"][:, :-1]
            ],
            np.diff(fields["relative_speed_mps"], axis=1)[
                fields["valid"][:, 1:] & fields["valid"][:, :-1]
            ],
        ),
        "correlation_variables": ["speed_mps", "acceleration_mps2", "gap_m"],
        "correlation_real": reference_correlation.tolist(),
        "correlation_generated": correlation.tolist(),
        "correlation_mean_absolute_error": float(
            np.nanmean(np.abs(correlation - reference_correlation))
        ),
        "brake_response_real": _conditional_brake(reference, reference_states),
        "brake_response_generated": _conditional_brake(fields, states),
    }


def _acf(fields: dict[str, np.ndarray], max_lag: int = 25) -> list[float]:
    value, valid = fields["acceleration_mps2"], fields["valid"]
    mean = _mean(value[valid])
    centered = value - mean
    variance = _mean(centered[valid] ** 2)
    if not np.isfinite(variance) or variance < EPS:
        return [float("nan")] * max_lag
    result = []
    for lag in range(1, max_lag + 1):
        mask = valid[:, lag:] & valid[:, :-lag]
        result.append(
            _mean(centered[:, lag:][mask] * centered[:, :-lag][mask]) / variance
        )
    return result


def _run_lengths(active: np.ndarray, valid: np.ndarray) -> np.ndarray:
    durations: list[int] = []
    for episode in range(len(active)):
        for slot in range(active.shape[2]):
            count = 0
            for flag, present in zip(active[episode, :, slot], valid[episode, :, slot]):
                if bool(flag and present):
                    count += 1
                elif count:
                    durations.append(count)
                    count = 0
            if count:
                durations.append(count)
    return np.asarray(durations, np.float64) * DT


def _temporal(
    fields: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "acceleration_acf_real": _acf(reference),
        "acceleration_acf_generated": _acf(fields),
    }
    result["acceleration_acf_mean_absolute_error"] = float(
        np.nanmean(
            np.abs(
                np.asarray(result["acceleration_acf_real"])
                - np.asarray(result["acceleration_acf_generated"])
            )
        )
    )
    for name, condition in {
        "braking_duration_s": lambda item: item["acceleration_mps2"] < -0.5,
        "acceleration_duration_s": lambda item: item["acceleration_mps2"] > 0.5,
        "lateral_motion_duration_s": lambda item: np.abs(item["lateral_speed_mps"])
        > 0.1,
    }.items():
        result[name] = _empirical_distance(
            _run_lengths(condition(reference), reference["valid"]),
            _run_lengths(condition(fields), fields["valid"]),
        )
    return result


def _risk_tail(
    real: dict[str, np.ndarray], generated: dict[str, np.ndarray]
) -> dict[str, Any]:
    real_score = _episode_extrema(real)["risk_score"]
    generated_score = _episode_extrema(generated)["risk_score"]
    result = _empirical_distance(real_score, generated_score)
    result["risk_definition"] = (
        "max_t,slot[(3-TTC)_+/3 + 0.5*log(1+DRAC) + (8-gap)_+/8 + (-ax-2)_+/4]"
    )
    result["exceedance_at_real_quantiles"] = {
        f"q{int(q * 100)}": {
            "threshold": float(np.quantile(real_score, q)),
            "real": float(np.mean(real_score > np.quantile(real_score, q))),
            "generated": float(np.mean(generated_score > np.quantile(real_score, q))),
        }
        for q in QUANTILES
    }
    return result


def _physical_validity(
    fields: dict[str, np.ndarray], states: np.ndarray, ego: np.ndarray
) -> dict[str, float]:
    valid = fields["valid"]
    speed_bad = valid & ((fields["speed_mps"] < 0.0) | (fields["speed_mps"] > 75.0))
    accel_bad = valid & (np.abs(fields["acceleration_mps2"]) > 12.0)
    jerk_bad = valid & (np.abs(fields["jerk_mps3"]) > 40.0)
    overlap = (
        valid
        & (np.abs(states[..., 0] - ego[:, :, None, 0]) < 4.5)
        & (np.abs(states[..., 1] - ego[:, :, None, 1]) < 1.0)
    )
    per_episode = np.any(speed_bad | accel_bad | jerk_bad | overlap, axis=(1, 2))
    return {
        "invalid_trajectory_rate": float(np.mean(per_episode)),
        "speed_out_of_range_rate": (
            float(np.mean(speed_bad[valid])) if valid.any() else float("nan")
        ),
        "acceleration_out_of_range_rate": (
            float(np.mean(accel_bad[valid])) if valid.any() else float("nan")
        ),
        "jerk_out_of_range_rate": (
            float(np.mean(jerk_bad[valid])) if valid.any() else float("nan")
        ),
        "collision_overlap_rate": (
            float(np.mean(overlap[valid])) if valid.any() else float("nan")
        ),
    }


def _plot_ccdf(path: Path, real: np.ndarray, reports: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for name, values, style in [("highD real tail", real, "k-")] + [
        (name, value, "-") for name, value in reports.items()
    ]:
        values = np.sort(_finite(values))
        if len(values):
            axis.step(
                values,
                1.0 - np.arange(1, len(values) + 1) / len(values),
                style,
                where="post",
                label=name,
            )
    axis.set_yscale("log")
    axis.set_xlabel("trajectory risk score")
    axis.set_ylabel("P(R > x)")
    axis.set_ylim(1.0e-3, 1.0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _checkpoint_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report_from_samples(
    name: str,
    samples: np.ndarray,
    *,
    target: np.ndarray,
    ego: np.ndarray,
    valid: np.ndarray,
    event_masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    num_samples = int(samples.shape[0])
    real_fields = _risk_fields(target, ego, valid)
    generated_fields = [_risk_fields(sample, ego, valid) for sample in samples]
    pooled_fields = {
        key: np.concatenate([item[key] for item in generated_fields], axis=0)
        for key in generated_fields[0]
    }
    trajectory = _trajectory_metrics(samples, target, valid)
    report: dict[str, Any] = {
        "model": name,
        "trajectory_reproduction": {
            key: value
            for key, value in trajectory.items()
            if key != "per_episode_min_fde_m"
        },
        "risk_variable_distribution": {
            key: _empirical_distance(real_values, _variable_values(pooled_fields)[key])
            for key, real_values in _variable_values(real_fields).items()
        },
        "risk_tail": _risk_tail(real_fields, pooled_fields),
        "multi_vehicle_interaction": _interaction(
            generated_fields[0], samples[0], real_fields, target
        ),
        "temporal_dynamics": _temporal(generated_fields[0], real_fields),
        "diversity": _diversity(
            samples, ego, valid, trajectory["per_episode_min_fde_m"]
        ),
        "physical_validity": _physical_validity(
            pooled_fields,
            np.concatenate(samples, axis=0),
            np.tile(ego, (num_samples, 1, 1)),
        ),
        "events": {},
    }
    for event, mask in event_masks.items():
        if not np.any(mask):
            report["events"][event] = {"available": False, "num_sequences": 0}
            continue
        event_samples, event_target, event_valid = (
            samples[:, mask],
            target[mask],
            valid[mask],
        )
        metric = _trajectory_metrics(event_samples, event_target, event_valid)
        event_fields = [
            _risk_fields(sample[mask], ego[mask], valid[mask]) for sample in samples
        ]
        event_pooled = {
            key: np.concatenate([item[key] for item in event_fields], axis=0)
            for key in event_fields[0]
        }
        report["events"][event] = {
            "available": True,
            "num_sequences": int(mask.sum()),
            "trajectory_reproduction": {
                key: value
                for key, value in metric.items()
                if key != "per_episode_min_fde_m"
            },
            "risk_tail": _risk_tail(
                _risk_fields(target[mask], ego[mask], valid[mask]), event_pooled
            ),
        }
    return report


def _repeat_batch(batch: dict[str, Any], copies: int) -> dict[str, Any]:
    """Repeat each scene contiguously so K branches share its exact condition."""
    return {
        name: value.repeat_interleave(int(copies), dim=0)
        for name, value in batch.items()
    }


def _parallel_model_samples(
    model,
    batch: dict[str, Any],
    *,
    num_samples: int,
    branch_batch_size: int,
    seed: int,
) -> np.ndarray:
    """Generate K branches in small batched groups rather than serially."""
    deterministic = (
        model.rollout_roll_mode(batch, seed=seed, deterministic=True)[
            "predicted_states"
        ][:, :, 1:]
        .cpu()
        .numpy()
    )
    branches: list[np.ndarray] = []
    batch_size = int(deterministic.shape[0])
    for first in range(0, int(num_samples), int(branch_batch_size)):
        count = min(int(branch_batch_size), int(num_samples) - first)
        rollout = (
            model.rollout_roll_mode(
                _repeat_batch(batch, count),
                seed=seed + 10_000 + first,
                deterministic=False,
            )["predicted_states"][:, :, 1:]
            .cpu()
            .numpy()
        )
        branches.append(
            rollout.reshape(batch_size, count, *rollout.shape[1:]).transpose(
                1, 0, 2, 3, 4
            )
        )
    samples = np.concatenate(branches, axis=0)
    samples[0] = deterministic
    return samples


def _parallel_cat_samples(
    model,
    arrays: dict[str, np.ndarray],
    schema: dict[str, Any],
    legacy: np.ndarray,
    *,
    num_samples: int,
    branch_batch_size: int,
    device,
    seed: int,
) -> np.ndarray:
    """Equivalent batched stochastic branching for the frozen CAT-TopK API."""
    deterministic, _ = _catk_multichunk_rollout(
        model,
        arrays,
        schema,
        legacy,
        chunks=5,
        device=device,
        seed=seed,
        deterministic=True,
    )
    branches: list[np.ndarray] = []
    batch_size = int(len(legacy))
    for first in range(0, int(num_samples), int(branch_batch_size)):
        count = min(int(branch_batch_size), int(num_samples) - first)
        rollout, _ = _catk_multichunk_rollout(
            model,
            arrays,
            schema,
            np.repeat(legacy, count, axis=0),
            chunks=5,
            device=device,
            seed=seed + 10_000 + first,
            deterministic=False,
        )
        branches.append(
            rollout.reshape(batch_size, count, *rollout.shape[1:]).transpose(
                1, 0, 2, 3, 4
            )
        )
    samples = np.concatenate(branches, axis=0)
    samples[0] = deterministic
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ramp-config",
        default=str(ROOT / "world_model/scripts/configs/highd_ramp_world_model.yaml"),
    )
    parser.add_argument(
        "--semi-config",
        default=str(
            ROOT / "world_model/scripts/configs/highd_semi_markov_world_model.yaml"
        ),
    )
    parser.add_argument(
        "--catk-config",
        default=str(
            ROOT / "world_model/scripts/configs/highd_cat_topk_world_model.yaml"
        ),
    )
    parser.add_argument(
        "--ramp-checkpoint",
        default=str(
            ROOT
            / "results/highd_world_model/ramp_world_model/checkpoints/best_ramp_world_model.pt"
        ),
    )
    parser.add_argument(
        "--semi-checkpoint",
        default=str(
            ROOT
            / "results/highd_world_model/semi_markov_world_model/checkpoints/best_semi_markov_relational.pt"
        ),
    )
    parser.add_argument(
        "--catk-checkpoint",
        default=str(
            ROOT
            / "results/highd_world_model/cat_topk_world_model/checkpoints/best_world_model.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results/highd_world_model/long_tail_reproduction"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument(
        "--branch-batch-size",
        type=int,
        default=4,
        help="Number of stochastic futures generated together per scene.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=0,
        help="0 evaluates every held-out EVT-tail sequence.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.num_samples < 2:
        raise ValueError("--num-samples must be at least 2 for a diversity experiment")
    if args.branch_batch_size < 1:
        raise ValueError("--branch-batch-size must be positive")

    ramp_path, semi_path, cat_path = (
        Path(args.ramp_config).resolve(),
        Path(args.semi_config).resolve(),
        Path(args.catk_config).resolve(),
    )
    ramp_cfg, semi_cfg, cat_cfg = (
        load_yaml(ramp_path),
        load_yaml(semi_path),
        load_yaml(cat_path),
    )
    output = ensure_dir(Path(args.output_dir).resolve())
    device = select_device(str(ramp_cfg.get("evaluation", {}).get("device", "auto")))
    set_seed(args.seed)
    owner = sequence_cache_owner_dir(ramp_cfg, config_dir=ramp_path.parent)
    arrays, manifest = load_sequential_dataset(owner)
    sequence_index = np.flatnonzero(
        (np.asarray(arrays["split_index"]) == 2)
        & np.asarray(arrays["is_evt_tail"], bool)
    )
    if args.max_sequences:
        sequence_index = sequence_index[: args.max_sequences]
    if not len(sequence_index):
        raise RuntimeError("the held-out sequence cache contains no EVT-tail scenarios")
    schema_path = Path(ramp_cfg["paths"]["flow_schema"])
    schema = FrozenLegacyFlowSchema.load(
        schema_path
        if schema_path.is_absolute()
        else (ramp_path.parent / schema_path).resolve()
    )
    arrays.update(
        ensure_frozen_flow_behavior_anchor_cache(owner, arrays, manifest, schema)
    )
    source_arrays, source_schema = load_world_model_dataset(
        dataset_dir_from_config(cat_cfg, cat_path.parent)
    )
    legacy = _legacy_sequences(
        source_arrays,
        np.asarray(arrays["sequence_id"])[sequence_index],
        horizon_steps=int(source_schema["horizon_steps"]),
        chunks=5,
    )
    ramp = load_ramp_checkpoint(Path(args.ramp_checkpoint).resolve(), device=device)
    semi = load_semi_markov_checkpoint(
        Path(args.semi_checkpoint).resolve(), device=device
    )
    semi.set_frozen_flow_schema(schema)
    cat, _ = load_catk_checkpoint(str(Path(args.catk_checkpoint).resolve()), device)

    targets: list[np.ndarray] = []
    egos: list[np.ndarray] = []
    valids: list[np.ndarray] = []
    predicted: dict[str, list[np.ndarray]] = {
        "ramp": [],
        "semi_markov": [],
        "cat_topk": [],
    }
    import torch

    with torch.no_grad():
        for start in range(0, len(sequence_index), args.batch_size):
            stop = min(start + args.batch_size, len(sequence_index))
            rows = sequence_index[start:stop]
            print(
                f"Generating tail scenes {start + 1}-{stop}/{len(sequence_index)} "
                f"with K={args.num_samples}",
                flush=True,
            )
            batch = _batch(arrays, rows, device)
            target = np.asarray(arrays["agent_states"][rows, 25:150, 1:], np.float32)
            ego = np.asarray(arrays["agent_states"][rows, 25:150, 0], np.float32)
            valid = np.asarray(arrays["agent_valid"][rows, 25:150, 1:], bool)
            targets.append(target)
            egos.append(ego)
            valids.append(valid)
            for name, model in (("ramp", ramp), ("semi_markov", semi)):
                predicted[name].append(
                    _parallel_model_samples(
                        model,
                        batch,
                        num_samples=args.num_samples,
                        branch_batch_size=args.branch_batch_size,
                        seed=args.seed + 100_000 * (1 if name == "ramp" else 2) + start,
                    )
                )
            predicted["cat_topk"].append(
                _parallel_cat_samples(
                    cat,
                    source_arrays,
                    source_schema,
                    legacy[start:stop],
                    num_samples=args.num_samples,
                    branch_batch_size=args.branch_batch_size,
                    device=device,
                    seed=args.seed + 300_000 + start,
                )
            )

    target, ego, valid = (
        np.concatenate(targets),
        np.concatenate(egos),
        np.concatenate(valids),
    )
    all_samples = {
        name: np.concatenate(value, axis=1) for name, value in predicted.items()
    }
    real_fields = _risk_fields(target, ego, valid)
    event_masks = _event_masks(real_fields)
    report: dict[str, Any] = {
        "protocol": {
            "dataset": "highD held-out EVT-tail sequences",
            "split": "test",
            "horizon_seconds": 5.0,
            "fixed_conditions": [
                "logged initial traffic state",
                "frozen B0 behavior condition",
                "road graph",
                "ego history and observed closed-loop ego replay",
            ],
            "num_sequences": int(len(target)),
            "num_stochastic_futures": int(args.num_samples),
            "stochastic_branch_batch_size": int(args.branch_batch_size),
            "distribution_empirical_max_points": MAX_DISTRIBUTION_POINTS,
            "cat_topk_information_asymmetric": True,
            "cat_topk_start_condition": "archived future-action summary",
            "ramp_and_semi_start_condition": "frozen B0 behavior anchor",
        },
        "tail_event_selection": {
            name: int(mask.sum()) for name, mask in event_masks.items()
        },
        "checkpoints": {
            name: {
                "path": str(path.resolve()),
                "sha256": _checkpoint_hash(path.resolve()),
            }
            for name, path in {
                "ramp": Path(args.ramp_checkpoint),
                "semi_markov": Path(args.semi_checkpoint),
                "cat_topk": Path(args.catk_checkpoint),
            }.items()
        },
        "models": {},
    }
    risk_scores: dict[str, np.ndarray] = {}
    for name, samples in all_samples.items():
        model_report = _report_from_samples(
            name,
            samples,
            target=target,
            ego=ego,
            valid=valid,
            event_masks=event_masks,
        )
        report["models"][name] = model_report
        generated = [_risk_fields(sample, ego, valid) for sample in samples]
        pooled = {
            key: np.concatenate([item[key] for item in generated], axis=0)
            for key in generated[0]
        }
        risk_scores[name] = _episode_extrema(pooled)["risk_score"]
    _plot_ccdf(
        output / "risk_ccdf.png",
        _episode_extrema(real_fields)["risk_score"],
        risk_scores,
    )
    save_json(report, output / "long_tail_reproduction_summary.json")
    print(output / "long_tail_reproduction_summary.json")


if __name__ == "__main__":
    main()

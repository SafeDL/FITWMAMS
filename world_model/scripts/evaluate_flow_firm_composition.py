#!/usr/bin/env python3
"""Evaluate frozen EVT Flow × FIRM-WM with the formal 8 × 4 protocol."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from normalizing_flow.src.sampling import load_checkpoint_and_dataset
from world_model.scripts import evaluate_flow_ramp_composition as common
from world_model.scripts import evaluate_long_tail_reproduction as tail
from world_model.src.core.data import SPLIT_TO_INDEX
from world_model.src.firm.train import load_firm_checkpoint
from world_model.src.core.initial_behavior_anchor import start_state_from_flow_feature
from world_model.src.ramp.distribution_evaluation import (
    multivariate_feature_distance,
    trajectory_feature_rows,
)
from world_model.src.core.sequential_dataset import load_sequential_dataset
from world_model.src.core.utils import ensure_dir, save_json, select_device


OUTER_FLOW_SAMPLES = 8
INNER_FIRM_WORLDS = 4
ROLLOUT_FRAMES = 125
HISTORY_FRAME = 24
SEED = 20260727


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor(value: np.ndarray, device):
    import torch

    return torch.from_numpy(np.asarray(value).copy()).to(device)


def _absolute_quantiles(values: np.ndarray) -> dict[str, float]:
    """Report a unit-preserving tail audit for an executed control variable."""
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"q50": float("nan"), "q90": float("nan"), "q95": float("nan"), "q99": float("nan")}
    return {
        f"q{int(level * 100)}": float(np.quantile(np.abs(finite), level))
        for level in (0.50, 0.90, 0.95, 0.99)
    }


def _overlap_timing(
    background: np.ndarray, ego: np.ndarray, valid: np.ndarray
) -> dict[str, int]:
    """Separate first-second overlap from later closed-loop overlap."""
    overlap = (
        valid
        & (np.abs(background[..., 0] - ego[:, :, None, 0]) < 4.5)
        & (np.abs(background[..., 1] - ego[:, :, None, 1]) < 1.0)
    )
    episode = np.any(overlap, axis=(1, 2))
    first = np.any(overlap[:, :25], axis=(1, 2))
    later = np.any(overlap[:, 25:], axis=(1, 2))
    # At a background vehicle's first ego-overlap, classify it in the ego
    # travel direction.  This is a diagnostic only: it identifies whether a
    # future action coordinate must model ego-facing rather than merely
    # background-background relations.
    prior = np.concatenate(
        (np.zeros_like(overlap[:, :1]), np.cumsum(overlap, axis=1)[:, :-1]), axis=1
    )
    first_point = overlap & (prior == 0)
    direction = np.sign(ego[..., 2])[:, :, None]
    direction = np.where(np.abs(direction) > 1.0e-3, direction, 1.0)
    longitudinal = (background[..., 0] - ego[:, :, None, 0]) * direction
    ahead = first_point & (longitudinal > 0.0)
    behind = first_point & (longitudinal <= 0.0)
    return {
        "episodes": int(len(background)),
        "episodes_any_overlap": int(episode.sum()),
        "episodes_first_second_overlap": int(first.sum()),
        "episodes_after_first_second_overlap": int(later.sum()),
        "overlap_points_first_second": int(overlap[:, :25].sum()),
        "overlap_points_after_first_second": int(overlap[:, 25:].sum()),
        "first_overlap_background_ahead_of_ego": int(ahead.sum()),
        "first_overlap_background_behind_ego": int(behind.sum()),
    }


def _exceedance_at_real_quantiles(
    real: np.ndarray, generated: np.ndarray, *, seed: int, bootstrap_samples: int = 2000
) -> dict[str, dict[str, float | bool | list[float]]]:
    """Calibrate generated tail mass at fixed highD risk thresholds.

    The highD threshold is held fixed while bootstrap resamples estimate the
    finite-reference uncertainty of its exceedance probability.  This avoids
    refitting EVT or changing a generated trajectory during reporting.
    """
    reference = np.asarray(real, dtype=np.float64).reshape(-1)
    samples = np.asarray(generated, dtype=np.float64).reshape(-1)
    reference = reference[np.isfinite(reference)]
    samples = samples[np.isfinite(samples)]
    if not len(reference) or not len(samples):
        return {}
    rng = np.random.default_rng(int(seed))
    result: dict[str, dict[str, float | bool | list[float]]] = {}
    for level in (0.90, 0.95, 0.99):
        threshold = float(np.quantile(reference, level))
        real_probability = float(np.mean(reference > threshold))
        generated_probability = float(np.mean(samples > threshold))
        draws = np.empty(int(bootstrap_samples), dtype=np.float64)
        for index in range(len(draws)):
            draws[index] = np.mean(
                rng.choice(reference, size=len(reference), replace=True) > threshold
            )
        interval = np.quantile(draws, (0.025, 0.975))
        result[f"q{int(level * 100)}"] = {
            "threshold": threshold,
            "highd": real_probability,
            "flow_firm": generated_probability,
            "absolute_error": float(abs(generated_probability - real_probability)),
            "highd_bootstrap_95": [float(interval[0]), float(interval[1])],
            "within_highd_bootstrap_95": bool(interval[0] <= generated_probability <= interval[1]),
        }
    return result


def _composition_batch(
    starts: dict[str, np.ndarray], donors: np.ndarray, cache: dict[str, np.ndarray], *, device
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    count = len(donors)
    initial = np.zeros((count, 7, 6), np.float32)
    valid = np.zeros((count, 7), bool)
    anchor = np.zeros((count, 6, 6), np.float32)
    anchor_valid = np.zeros((count, 6), bool)
    for index, (feature, mask) in enumerate(zip(starts["features"], starts["slot_mask"])):
        initial[index], valid[index], anchor[index], anchor_valid[index] = start_state_from_flow_feature(feature, mask)
    # Only C0 exists at frame 24.  FIRM does not receive a copied 25-frame
    # history; the earlier frames are invalid padding and cannot reach its encoder.
    states = np.zeros((count, 150, 7, 6), np.float32)
    present = np.zeros((count, 150, 7), bool)
    states[:, HISTORY_FRAME] = initial
    present[:, HISTORY_FRAME] = valid
    ego = np.empty((count, ROLLOUT_FRAMES, 6), np.float32)
    for index, donor in enumerate(donors):
        ego[index] = common._translated_ego_replay(
            np.asarray(cache["agent_states"][donor], np.float32), initial[index, 0]
        )
    states[:, HISTORY_FRAME + 1 :, 0] = ego
    present[:, HISTORY_FRAME + 1 :, 0] = True
    batch = {
        "agent_states": _tensor(states, device),
        "agent_valid": _tensor(present, device),
        "ego_index": _tensor(np.zeros(count, np.int64), device),
        "actions_highd": _tensor(np.zeros((count, ROLLOUT_FRAMES, 6, 2), np.float32), device),
        "behavior_anchor_raw": _tensor(anchor, device),
        "behavior_anchor_valid": _tensor(anchor_valid, device),
        "flow_latent": _tensor(starts["features"], device),
    }
    return batch, initial, ego, valid[:, 1:]


def _repeat(batch: dict[str, Any], count: int) -> dict[str, Any]:
    return {key: value.repeat_interleave(count, dim=0) for key, value in batch.items()}


def evaluate_flow_firm_composition(
    *,
    output_dir: Path | None = None,
    firm_checkpoint: Path | None = None,
    deterministic: bool = False,
) -> dict[str, Any]:
    import torch

    output = ensure_dir(
        ROOT / "results/highd_world_model/firm_world_model/evaluation"
        if output_dir is None
        else output_dir
    )
    device = select_device("auto")
    flow_checkpoint = ROOT / "results/highd_tail_flow/checkpoints/best_tail_conditional_maf.pt"
    firm_checkpoint = (
        ROOT / "results/highd_world_model/firm_world_model/checkpoints/best_firm_world_model.pt"
        if firm_checkpoint is None
        else firm_checkpoint
    )
    if not firm_checkpoint.exists():
        raise FileNotFoundError(f"FIRM checkpoint is unavailable: {firm_checkpoint}")
    flow, flow_arrays, flow_schema, _ = load_checkpoint_and_dataset(
        flow_checkpoint, ROOT / "results/highd_tail_flow", repo_root=ROOT, device=device
    )
    cache, _ = load_sequential_dataset(
        ROOT / "results/highd_world_model/training_data/semi_markov_sequence_cache"
    )
    heldout_rows = np.flatnonzero(
        (np.asarray(cache["split_index"]) == SPLIT_TO_INDEX["test"])
        & np.asarray(cache["is_evt_tail"], bool)
    )
    groups_all = common._replay_structures(cache, heldout_rows)
    flow_train = np.flatnonzero(np.asarray(flow_arrays["split_index"]) == SPLIT_TO_INDEX["train"])
    supported = {
        (int(flow_arrays["mask_pattern"][row]), int(flow_arrays["primary_slot_index"][row]))
        for row in flow_train
    }
    groups = {key: rows for key, rows in groups_all.items() if key in supported}
    if not groups:
        raise RuntimeError("no held-out EVT replay structure is supported by the frozen Flow")
    starts, donors = common._sample_flow_starts(flow, flow_arrays, flow_schema, groups, device=device)
    model = load_firm_checkpoint(firm_checkpoint, device=device)
    reference_rows = np.unique(donors)
    reference = np.asarray(cache["agent_states"])[reference_rows, 25:, 1:]
    reference_ego = np.asarray(cache["agent_states"])[reference_rows, 25:, 0]
    reference_valid = np.asarray(cache["agent_valid"])[reference_rows, 25:, 1:]
    reference_fields = tail._risk_fields(reference, reference_ego, reference_valid)
    reference_values = tail._variable_values(reference_fields)
    risk_scores: list[np.ndarray] = []
    variable_values: dict[str, list[np.ndarray]] = defaultdict(list)
    physical_counts: list[dict[str, int]] = []
    overlap_timing: list[dict[str, int]] = []
    anchor_target: list[np.ndarray] = []
    anchor_generated: list[np.ndarray] = []
    anchor_valid: list[np.ndarray] = []
    feature_rows: list[dict[str, np.ndarray]] = []
    within_fde: list[float] = []
    within_risk_std: list[float] = []
    single_states: list[np.ndarray] = []
    single_ego: list[np.ndarray] = []
    single_valid: list[np.ndarray] = []
    executed_longitudinal_jerk: list[np.ndarray] = []
    executed_yaw_jerk: list[np.ndarray] = []
    branch_size = 32
    for start in range(0, len(donors), branch_size):
        stop = min(start + branch_size, len(donors))
        part = {key: value[start:stop] for key, value in starts.items()}
        batch, initial, ego, valid = _composition_batch(part, donors[start:stop], cache, device=device)
        with torch.no_grad():
            rollout = model.rollout_roll_mode(
                _repeat(batch, INNER_FIRM_WORLDS), seed=SEED + start, deterministic=deterministic
            )
        count = stop - start
        full = rollout["predicted_states"].cpu().numpy().reshape(
            count, INNER_FIRM_WORLDS, ROLLOUT_FRAMES, 7, 6
        )
        background = full[..., 1:, :]
        flattened_background = background.reshape(-1, ROLLOUT_FRAMES, 6, 6)
        flattened_ego = np.repeat(ego, INNER_FIRM_WORLDS, axis=0)
        flattened_valid = np.repeat(valid[:, None], INNER_FIRM_WORLDS, axis=1).reshape(-1, 6)
        flattened_valid = np.repeat(flattened_valid[:, None], ROLLOUT_FRAMES, axis=1)
        jerk_valid = flattened_valid.reshape(
            -1, model.cfg.response_steps, model.cfg.execute_frames, 6
        )
        executed = rollout["joint_jerk_plans"][:, :, : model.cfg.execute_frames].cpu().numpy()
        executed_longitudinal_jerk.append(executed[..., 0][jerk_valid])
        executed_yaw_jerk.append(executed[..., 1][jerk_valid])
        fields = tail._risk_fields(flattened_background, flattened_ego, flattened_valid)
        risk_scores.append(tail._episode_extrema(fields)["risk_score"])
        for name, value in tail._variable_values(fields).items():
            variable_values[name].append(common._cap(value))
        physical_counts.append(common._physical_counts(fields, flattened_background, flattened_ego))
        overlap_timing.append(_overlap_timing(flattened_background, flattened_ego, flattened_valid))
        anchor_target.append(np.repeat(batch["behavior_anchor_raw"].cpu().numpy(), INNER_FIRM_WORLDS, axis=0))
        anchor_generated.append(common._b0_summary(np.repeat(initial, INNER_FIRM_WORLDS, axis=0), full.reshape(-1, ROLLOUT_FRAMES, 7, 6)))
        anchor_valid.append(np.repeat(batch["behavior_anchor_valid"].cpu().numpy(), INNER_FIRM_WORLDS, axis=0))
        single_states.append(background[:, 0])
        single_ego.append(ego)
        single_valid.append(np.repeat(valid[:, None], ROLLOUT_FRAMES, axis=1))
        feature_rows.append(trajectory_feature_rows(background[:, 0], valid[:, None].repeat(ROLLOUT_FRAMES, axis=1), ego))
        endpoint, risk_std = common._within_start_diversity(background, ego, valid)
        within_fde.append(endpoint)
        within_risk_std.append(risk_std)
        print(f"Flow×FIRM starts {stop}/{len(donors)}", flush=True)
    generated_risk = np.concatenate(risk_scores)
    reference_risk = tail._episode_extrema(reference_fields)["risk_score"]
    sampled_values = {name: np.concatenate(values) for name, values in variable_values.items()}
    anchors_true = np.concatenate(anchor_target)
    anchors_generated = np.concatenate(anchor_generated)
    anchors_valid = np.concatenate(anchor_valid)
    anchor_error = np.abs(anchors_true - anchors_generated)
    single = np.concatenate(single_states)
    single_ego_array = np.concatenate(single_ego)
    single_valid_array = np.concatenate(single_valid)
    single_fields = tail._risk_fields(single, single_ego_array, single_valid_array)
    observed_rows = trajectory_feature_rows(reference, reference_valid, reference_ego)
    generated_rows = {
        key: np.concatenate([row[key] for row in feature_rows]) for key in feature_rows[0]
    }
    flow_test = np.flatnonzero(np.asarray(flow_arrays["split_index"]) == SPLIT_TO_INDEX["test"])
    reference_with_start = np.asarray(cache["agent_states"])[reference_rows, 24 : 25 + ROLLOUT_FRAMES, 1:]
    reference_with_start_valid = np.asarray(cache["agent_valid"])[reference_rows, 24 : 25 + ROLLOUT_FRAMES, 1:]
    reference_jerk = (reference_with_start[:, 1:, :, 4:6] - reference_with_start[:, :-1, :, 4:6]) / model.cfg.simulation_dt_s
    reference_jerk_valid = reference_with_start_valid[:, 1:] & reference_with_start_valid[:, :-1]
    c0 = common._mean_distances(
        starts["features"][:, :40], starts["feature_valid"][:, :40],
        np.asarray(flow_arrays["features"])[flow_test, :40], np.asarray(flow_arrays["feature_valid"])[flow_test, :40],
    )
    b0 = common._mean_distances(
        starts["features"][:, 40:], starts["feature_valid"][:, 40:],
        np.asarray(flow_arrays["features"])[flow_test, 40:], np.asarray(flow_arrays["feature_valid"])[flow_test, 40:],
    )
    report = {
        "protocol": {
            "name": (
                "held-out EVT Flow × FIRM-WM deterministic-centre replay"
                if deterministic
                else "held-out EVT Flow × FIRM-WM replay-controlled composition"
            ),
            "outer_flow_samples_per_replay": OUTER_FLOW_SAMPLES,
            "inner_firm_worlds_per_flow_start": INNER_FIRM_WORLDS,
            "total_futures_per_held_out_replay": OUTER_FLOW_SAMPLES * INNER_FIRM_WORLDS,
            "held_out_replay_conditions": int(len(reference_rows)),
            "flow_start_samples": int(len(donors)),
            "generated_closed_loop_trajectories": int(len(generated_risk)),
            "horizon_seconds": 5.0,
            "start_history": "single C0 frame with invalid causal padding; no copied static history",
            "world_source": (
                "zero action-flow residual with persistent zW"
                if deterministic
                else "persistent zW plus joint action-flow innovations"
            ),
        },
        "artifacts": {
            "flow_checkpoint": str(flow_checkpoint),
            "flow_checkpoint_sha256": _sha256(flow_checkpoint),
            "firm_checkpoint": str(firm_checkpoint),
            "firm_checkpoint_sha256": _sha256(firm_checkpoint),
        },
        "flow_input_fidelity_on_held_out_structure": {
            "c0": c0,
            "b0": b0,
            "physical_rejection_rate": float(starts["rejection_rate"][0]),
        },
        "b0_execution_fidelity": {
            "feature_names": ["delta_vx", "delta_vy", "mean_ax", "min_ax", "final_ax", "mean_ay"],
            "mean_absolute_error": float(anchor_error[anchors_valid].mean()),
            "per_feature_mean_absolute_error": [float(anchor_error[..., feature][anchors_valid].mean()) for feature in range(6)],
        },
        "closed_loop_distribution": {
            "risk_tail_all_inner_samples": {
                **tail._empirical_distance(reference_risk, generated_risk),
                "exceedance_at_real_quantiles": _exceedance_at_real_quantiles(
                    reference_risk, generated_risk, seed=SEED + 71
                ),
            },
            "risk_variable_distribution": {key: tail._empirical_distance(reference_values[key], sampled_values[key]) for key in reference_values},
            "multi_vehicle_interaction": tail._interaction(single_fields, single, reference_fields, reference),
            "temporal_dynamics": tail._temporal(single_fields, reference_fields),
            "physical_validity": common._merge_counts(physical_counts),
            "overlap_timing": {
                key: int(sum(item[key] for item in overlap_timing))
                for key in overlap_timing[0]
            },
            "executed_jerk_distribution": {
                "definition": "absolute jerk of the random action-flow prefix actually written at every 0.2 s response",
                "longitudinal_mps3": {
                    "flow_firm": _absolute_quantiles(np.concatenate(executed_longitudinal_jerk)),
                    "highd_reference": _absolute_quantiles(reference_jerk[..., 0][reference_jerk_valid]),
                },
                "yaw_rps3": {
                    "flow_firm": _absolute_quantiles(np.concatenate(executed_yaw_jerk)),
                    "highd_reference": _absolute_quantiles(reference_jerk[..., 1][reference_jerk_valid]),
                },
            },
            "within_flow_start_diversity": {
                "mean_pairwise_endpoint_distance_m": float(np.mean(within_fde)),
                "mean_risk_score_standard_deviation": float(np.mean(within_risk_std)),
            },
            "two_sample_trajectory_features": multivariate_feature_distance(generated_rows, observed_rows, seed=SEED),
        },
    }
    suffix = "flow_firm_deterministic_composition" if deterministic else "flow_firm_composition"
    save_json(report, output / f"{suffix}.json")
    np.savez_compressed(
        output / f"{suffix}_tail_scores.npz",
        highd_risk_scores=reference_risk.astype(np.float32),
        firm_risk_scores=generated_risk.astype(np.float32),
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="diagnose the zero-residual action-flow centre without overwriting stochastic composition results",
    )
    arguments = parser.parse_args()
    evaluate_flow_firm_composition(
        output_dir=None if arguments.output_dir is None else Path(arguments.output_dir).resolve(),
        firm_checkpoint=None if arguments.checkpoint is None else Path(arguments.checkpoint).resolve(),
        deterministic=bool(arguments.deterministic),
    )

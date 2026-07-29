#!/usr/bin/env python3
"""Evaluate the EVT Flow × RAMP-WM composition under held-out ego replay.

The Flow supplies an independently sampled C0 state and its one-second B0
behaviour anchor.  A held-out highD replay with the same discrete event
structure supplies only the road graph and already-observed ego trajectory.
This is a distribution-level closed-loop test, not a paired reconstruction:
there is deliberately no ADE/FDE-to-donor score for a sampled C0.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from normalizing_flow.src.sampling import (
    load_checkpoint_and_dataset,
    sample_tail_c0,
)
from world_model.scripts import evaluate_long_tail_reproduction as tail
from world_model.src.core.data import SPLIT_TO_INDEX
from world_model.src.core.initial_behavior_anchor import start_state_from_flow_feature
from world_model.src.ramp.distribution_evaluation import (
    multivariate_feature_distance,
    trajectory_feature_rows,
)
from world_model.src.ramp.train import load_ramp_checkpoint
from world_model.src.core.sequential_dataset import load_sequential_dataset
from world_model.src.core.utils import ensure_dir, save_json, select_device

OUTER_FLOW_SAMPLES_PER_REPLAY = 8
INNER_RAMP_SAMPLES_PER_FLOW_START = 4
ROLLOUT_FRAMES = 125
HISTORY_FRAMES = 25
SEED = 20260726


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor(value: np.ndarray, device):
    import torch

    return torch.from_numpy(np.asarray(value).copy()).to(device)


def _cap(values: np.ndarray, maximum: int = 768) -> np.ndarray:
    values = np.asarray(values)
    if len(values) <= maximum:
        return values
    return values[np.linspace(0, len(values) - 1, maximum, dtype=np.int64)]


def _mean_distances(
    generated: np.ndarray,
    generated_valid: np.ndarray,
    observed: np.ndarray,
    observed_valid: np.ndarray,
) -> dict[str, Any]:
    metrics = {}
    for index in range(generated.shape[1]):
        left = generated[:, index][generated_valid[:, index]]
        right = observed[:, index][observed_valid[:, index]]
        metrics[str(index)] = tail._empirical_distance(right, left)
    available = [value for value in metrics.values() if value.get("available")]
    return {
        "per_feature": metrics,
        "mean_wasserstein_1": float(
            np.mean([value["wasserstein_1"] for value in available])
        ),
        "mean_ks": float(np.mean([value["ks"] for value in available])),
    }


def _replay_structures(
    cache: dict[str, np.ndarray], cache_rows: np.ndarray
) -> dict[tuple[int, int], np.ndarray]:
    """Use observed START slot occupancy, never a future trajectory label."""
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    masks = np.asarray(cache["agent_valid"])[cache_rows, HISTORY_FRAMES - 1, 1:]
    patterns = np.sum(masks * (1 << np.arange(masks.shape[1])), axis=1)
    primary = np.where(masks[:, 0], 0, np.argmax(masks, axis=1))
    for row, pattern, primary_slot, mask in zip(cache_rows, patterns, primary, masks):
        if not np.any(mask):
            continue
        groups[(int(pattern), int(primary_slot))].append(int(row))
    if not groups:
        raise RuntimeError("held-out replay set has no active background slot")
    return {key: np.asarray(value, np.int64) for key, value in groups.items()}


def _sample_flow_starts(
    flow,
    flow_arrays: dict[str, np.ndarray],
    flow_schema: dict[str, Any],
    groups: dict[tuple[int, int], np.ndarray],
    *,
    device,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Draw eight C0/B0 starts for every matched held-out replay condition."""
    pieces: dict[str, list[np.ndarray]] = defaultdict(list)
    donors: list[np.ndarray] = []
    for offset, (key, rows) in enumerate(sorted(groups.items())):
        count = len(rows) * OUTER_FLOW_SAMPLES_PER_REPLAY
        sampled = sample_tail_c0(
            flow,
            flow_arrays,
            flow_schema,
            num_samples=count,
            device=device,
            seed=SEED + 1009 * offset,
            mask_pattern=key[0],
            primary_slot=key[1],
            event_structure_split="train",
            event_structure_sampling="quota",
            reject_invalid=True,
            max_rounds=80,
            oversample_factor=1,
            min_draw=1,
            temperature=1.0295,
        )
        for name, value in sampled.items():
            pieces[name].append(np.asarray(value))
        donors.append(np.repeat(rows, OUTER_FLOW_SAMPLES_PER_REPLAY))
    result = {name: np.concatenate(value) for name, value in pieces.items()}
    donor_rows = np.concatenate(donors)
    if len(result["features"]) != len(donor_rows):
        raise RuntimeError("Flow starts and held-out replay donors do not align")
    return result, donor_rows


def _translated_ego_replay(donor: np.ndarray, initial: np.ndarray) -> np.ndarray:
    """Preserve a held-out ego motion increment while starting at sampled C0."""
    anchor = donor[HISTORY_FRAMES - 1, 0]
    future = donor[HISTORY_FRAMES : HISTORY_FRAMES + ROLLOUT_FRAMES, 0].copy()
    return future - anchor + initial


def _composition_batch(
    starts: dict[str, np.ndarray],
    donors: np.ndarray,
    cache: dict[str, np.ndarray],
    *,
    device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    count = len(donors)
    initial = np.zeros((count, 7, 6), np.float32)
    valid = np.zeros((count, 7), bool)
    anchor = np.zeros((count, 6, 6), np.float32)
    anchor_valid = np.zeros((count, 6), bool)
    for index, (feature, mask) in enumerate(zip(starts["features"], starts["slot_mask"])):
        initial[index], valid[index], anchor[index], anchor_valid[index] = (
            start_state_from_flow_feature(feature, mask)
        )
    states = np.repeat(initial[:, None], HISTORY_FRAMES + ROLLOUT_FRAMES, axis=1)
    present = np.repeat(valid[:, None], HISTORY_FRAMES + ROLLOUT_FRAMES, axis=1)
    ego = np.empty((count, ROLLOUT_FRAMES, 6), np.float32)
    for index, row in enumerate(donors):
        ego[index] = _translated_ego_replay(
            np.asarray(cache["agent_states"][row], np.float32), initial[index, 0]
        )
    states[:, HISTORY_FRAMES:, 0] = ego
    present[:, :, 0] = True
    batch = {
        "agent_states": _tensor(states, device),
        "agent_valid": _tensor(present, device),
        "ego_index": _tensor(np.zeros(count, np.int64), device),
        "map_polylines": _tensor(cache["map_polylines"][donors], device),
        "map_polyline_valid": _tensor(cache["map_polyline_valid"][donors], device),
        "lane_graph_edges": _tensor(cache["lane_graph_edges"][donors], device),
        "actions_highd": _tensor(np.zeros((count, ROLLOUT_FRAMES, 6, 2), np.float32), device),
        "behavior_anchor_raw": _tensor(anchor, device),
        "behavior_anchor_valid": _tensor(anchor_valid, device),
    }
    return batch, initial, ego, valid[:, 1:]


def _repeat_batch(batch: dict[str, Any], count: int) -> dict[str, Any]:
    return {name: value.repeat_interleave(count, dim=0) for name, value in batch.items()}


def _physical_counts(fields: dict[str, np.ndarray], states: np.ndarray, ego: np.ndarray) -> dict[str, int]:
    valid = fields["valid"]
    speed = fields["speed_mps"]
    acceleration = fields["acceleration_mps2"]
    jerk = fields["jerk_mps3"]
    overlap = (
        valid
        & (np.abs(states[..., 0] - ego[:, :, None, 0]) < 4.5)
        & (np.abs(states[..., 1] - ego[:, :, None, 1]) < 1.0)
    )
    speed_bad = valid & ((speed < 0.0) | (speed > 75.0))
    acceleration_bad = valid & (np.abs(acceleration) > 12.0)
    jerk_bad = valid & (np.abs(jerk) > 40.0)
    invalid = np.any(speed_bad | acceleration_bad | jerk_bad | overlap, axis=(1, 2))
    return {
        "episodes": int(len(states)),
        "invalid_episodes": int(invalid.sum()),
        "valid_points": int(valid.sum()),
        "speed_bad": int(speed_bad.sum()),
        "acceleration_bad": int(acceleration_bad.sum()),
        "jerk_bad": int(jerk_bad.sum()),
        "overlap": int(overlap.sum()),
    }


def _merge_counts(parts: list[dict[str, int]]) -> dict[str, float]:
    total = {key: int(sum(item[key] for item in parts)) for key in parts[0]}
    points = max(total["valid_points"], 1)
    episodes = max(total["episodes"], 1)
    return {
        "invalid_trajectory_rate": total["invalid_episodes"] / episodes,
        "speed_out_of_range_rate": total["speed_bad"] / points,
        "acceleration_out_of_range_rate": total["acceleration_bad"] / points,
        "jerk_out_of_range_rate": total["jerk_bad"] / points,
        "collision_overlap_rate": total["overlap"] / points,
        "generated_trajectories": total["episodes"],
    }


def _b0_summary(initial: np.ndarray, generated: np.ndarray) -> np.ndarray:
    full = np.concatenate((initial[:, None], generated[:, :HISTORY_FRAMES]), axis=1)
    acceleration = full[..., 4]
    return np.stack(
        (
            full[:, -1, 1:, 2] - full[:, 0, 1:, 2],
            full[:, -1, 1:, 3] - full[:, 0, 1:, 3],
            acceleration[:, :, 1:].mean(axis=1),
            acceleration[:, :, 1:].min(axis=1),
            acceleration[:, -1, 1:],
            full[:, :, 1:, 5].mean(axis=1),
        ),
        axis=-1,
    )


def _within_start_diversity(
    samples: np.ndarray, ego: np.ndarray, valid: np.ndarray
) -> tuple[float, float]:
    if samples.shape[1] < 2:
        return float("nan"), float("nan")
    endpoint = samples[:, :, -1, :, :2]
    upper = np.triu_indices(samples.shape[1], k=1)
    distance = np.linalg.norm(endpoint[:, upper[0]] - endpoint[:, upper[1]], axis=-1)
    mask = valid[:, None, :]
    per_pair = (distance * mask).sum(axis=-1) / mask.sum(axis=-1).clip(min=1)
    risk = []
    for draw in range(samples.shape[1]):
        fields = tail._risk_fields(
            samples[:, draw], ego, valid[:, None].repeat(ROLLOUT_FRAMES, axis=1)
        )
        risk.append(tail._episode_extrema(fields)["risk_score"])
    return float(per_pair.mean()), float(np.std(np.stack(risk), axis=1).mean())


def evaluate_flow_ramp_composition(max_replays: int = 0) -> dict[str, Any]:
    import torch

    if max_replays:
        raise ValueError(
            "the Flow×RAMP composition report is formal-only; do not write bounded smoke results"
        )
    output = ensure_dir(
        ROOT / "results/highd_world_model/long_tail_reproduction/ramp_world_model"
    )
    device = select_device("auto")
    flow_checkpoint = ROOT / "results/highd_tail_flow/checkpoints/best_tail_conditional_maf.pt"
    flow_output = ROOT / "results/highd_tail_flow"
    ramp_checkpoint = (
        ROOT / "results/highd_world_model/ramp_world_model/checkpoints/best_ramp_world_model.pt"
    )
    flow, flow_arrays, flow_schema, _ = load_checkpoint_and_dataset(
        flow_checkpoint, flow_output, repo_root=ROOT, device=device
    )
    cache_owner = ROOT / "results/highd_world_model/training_data/semi_markov_sequence_cache"
    cache, _manifest = load_sequential_dataset(cache_owner)
    rows = np.flatnonzero(
        (np.asarray(cache["split_index"]) == SPLIT_TO_INDEX["test"])
        & np.asarray(cache["is_evt_tail"], bool)
    )
    if max_replays:
        rows = rows[:max_replays]
    raw_groups = _replay_structures(cache, rows)
    flow_train = np.flatnonzero(
        np.asarray(flow_arrays["split_index"]) == SPLIT_TO_INDEX["train"]
    )
    supported = {
        (int(flow_arrays["mask_pattern"][row]), int(flow_arrays["primary_slot_index"][row]))
        for row in flow_train
    }
    groups = {key: value for key, value in raw_groups.items() if key in supported}
    unsupported = {key: value for key, value in raw_groups.items() if key not in supported}
    if not groups:
        raise RuntimeError("no held-out replay event structures are supported by the frozen Flow")
    starts, donors = _sample_flow_starts(flow, flow_arrays, flow_schema, groups, device=device)
    model = load_ramp_checkpoint(ramp_checkpoint, device=device)

    reference_rows = np.unique(donors)
    reference_target = np.asarray(cache["agent_states"])[reference_rows, HISTORY_FRAMES:, 1:]
    reference_ego = np.asarray(cache["agent_states"])[reference_rows, HISTORY_FRAMES:, 0]
    reference_valid = np.asarray(cache["agent_valid"])[reference_rows, HISTORY_FRAMES:, 1:]
    reference_fields = tail._risk_fields(reference_target, reference_ego, reference_valid)
    reference_values = tail._variable_values(reference_fields)

    anchor_target, anchor_generated, anchor_valid = [], [], []
    risk_scores: list[np.ndarray] = []
    variable_values: dict[str, list[np.ndarray]] = defaultdict(list)
    physical_counts: list[dict[str, int]] = []
    single_states, single_ego, single_valid = [], [], []
    feature_rows: list[dict[str, np.ndarray]] = []
    within_fde, within_risk_std = [], []
    candidate_probabilities: list[np.ndarray] = []
    branch_size = 16
    for start in range(0, len(donors), branch_size):
        stop = min(start + branch_size, len(donors))
        part = {key: value[start:stop] for key, value in starts.items()}
        batch, initial, ego, valid = _composition_batch(part, donors[start:stop], cache, device=device)
        with torch.no_grad():
            rollout = model.rollout_roll_mode(
                _repeat_batch(batch, INNER_RAMP_SAMPLES_PER_FLOW_START),
                seed=SEED + start,
                deterministic=False,
            )
        count = stop - start
        full = rollout["predicted_states"].cpu().numpy().reshape(
            count, INNER_RAMP_SAMPLES_PER_FLOW_START, ROLLOUT_FRAMES, 7, 6
        )
        background = full[..., 1:, :]
        all_background = background.reshape(-1, ROLLOUT_FRAMES, 6, 6)
        all_ego = np.repeat(ego, INNER_RAMP_SAMPLES_PER_FLOW_START, axis=0)
        all_valid = np.repeat(valid[:, None], INNER_RAMP_SAMPLES_PER_FLOW_START, axis=1).reshape(-1, 6)
        all_valid = np.repeat(all_valid[:, None], ROLLOUT_FRAMES, axis=1)
        fields = tail._risk_fields(all_background, all_ego, all_valid)
        risk_scores.append(tail._episode_extrema(fields)["risk_score"])
        for name, value in tail._variable_values(fields).items():
            variable_values[name].append(_cap(value))
        physical_counts.append(_physical_counts(fields, all_background, all_ego))
        generated_anchor = _b0_summary(
            np.repeat(initial, INNER_RAMP_SAMPLES_PER_FLOW_START, axis=0),
            full.reshape(-1, ROLLOUT_FRAMES, 7, 6),
        )
        anchor_target.append(np.repeat(batch["behavior_anchor_raw"].cpu().numpy(), INNER_RAMP_SAMPLES_PER_FLOW_START, axis=0))
        anchor_generated.append(generated_anchor)
        anchor_valid.append(np.repeat(batch["behavior_anchor_valid"].cpu().numpy(), INNER_RAMP_SAMPLES_PER_FLOW_START, axis=0))
        single_states.append(background[:, 0])
        single_ego.append(ego)
        single_valid.append(np.repeat(valid[:, None], ROLLOUT_FRAMES, axis=1))
        feature_rows.append(trajectory_feature_rows(background[:, 0], valid[:, None].repeat(ROLLOUT_FRAMES, axis=1), ego))
        fde, risk_std = _within_start_diversity(background, ego, valid)
        within_fde.append(fde)
        within_risk_std.append(risk_std)
        candidate_probabilities.append(rollout["candidate_probabilities"].cpu().numpy())
        print(f"Flow×RAMP starts {stop}/{len(donors)}", flush=True)

    anchor_target = np.concatenate(anchor_target)
    anchor_generated = np.concatenate(anchor_generated)
    anchor_valid = np.concatenate(anchor_valid)
    anchor_error = np.abs(anchor_generated - anchor_target)
    feature_mae = [
        float(anchor_error[..., feature][anchor_valid].mean())
        for feature in range(anchor_error.shape[-1])
    ]
    generated_risk = np.concatenate(risk_scores)
    sampled_values = {name: np.concatenate(value) for name, value in variable_values.items()}
    one_state = np.concatenate(single_states)
    one_ego = np.concatenate(single_ego)
    one_valid = np.concatenate(single_valid)
    one_fields = tail._risk_fields(one_state, one_ego, one_valid)
    observed_rows = trajectory_feature_rows(reference_target, reference_valid, reference_ego)
    generated_rows = {
        key: np.concatenate([part[key] for part in feature_rows]) for key in feature_rows[0]
    }
    flow_test = np.flatnonzero(np.asarray(flow_arrays["split_index"]) == SPLIT_TO_INDEX["test"])
    c0 = _mean_distances(
        starts["features"][:, :40], starts["feature_valid"][:, :40],
        np.asarray(flow_arrays["features"])[flow_test, :40], np.asarray(flow_arrays["feature_valid"])[flow_test, :40],
    )
    b0 = _mean_distances(
        starts["features"][:, 40:], starts["feature_valid"][:, 40:],
        np.asarray(flow_arrays["features"])[flow_test, 40:], np.asarray(flow_arrays["feature_valid"])[flow_test, 40:],
    )
    report = {
        "protocol": {
            "name": "held-out EVT Flow × RAMP replay-controlled composition",
            "flow_outer_samples_per_replay": OUTER_FLOW_SAMPLES_PER_REPLAY,
            "ramp_inner_samples_per_flow_start": INNER_RAMP_SAMPLES_PER_FLOW_START,
            "total_futures_per_held_out_replay": OUTER_FLOW_SAMPLES_PER_REPLAY * INNER_RAMP_SAMPLES_PER_FLOW_START,
            "held_out_replay_conditions": int(len(reference_rows)),
            "unsupported_held_out_replay_conditions": int(sum(len(value) for value in unsupported.values())),
            "flow_start_samples": int(len(donors)),
            "generated_closed_loop_trajectories": int(len(donors) * INNER_RAMP_SAMPLES_PER_FLOW_START),
            "horizon_seconds": 5.0,
            "map_and_ego_source": "held-out highD replay matched by cache-derived START event structure",
            "primary_slot_rule": "same_front when present; otherwise the first active fixed slot",
            "flow_source": "frozen conditional EVT-tail MAF; sampled C0 plus B0",
            "world_source": "RAMP categorical joint candidate sequence",
            "not_a_paired_reconstruction": True,
        },
        "artifacts": {
            "flow_checkpoint": str(flow_checkpoint),
            "flow_checkpoint_sha256": _sha256(flow_checkpoint),
            "ramp_checkpoint": str(ramp_checkpoint),
            "ramp_checkpoint_sha256": _sha256(ramp_checkpoint),
        },
        "flow_input_fidelity_on_held_out_structure": {
            "c0": c0,
            "b0": b0,
            "physical_rejection_rate": float(starts["rejection_rate"][0]),
        },
        "b0_execution_fidelity": {
            "feature_names": ["delta_vx", "delta_vy", "mean_ax", "min_ax", "final_ax", "mean_ay"],
            "mean_absolute_error": float(anchor_error[anchor_valid].mean()),
            "per_feature_mean_absolute_error": feature_mae,
        },
        "closed_loop_distribution": {
            "risk_tail": tail._risk_tail(reference_fields, {**one_fields, "risk": one_fields["risk"]}),
            "risk_tail_all_inner_samples": tail._empirical_distance(
                tail._episode_extrema(reference_fields)["risk_score"], generated_risk
            ),
            "risk_variable_distribution": {
                key: tail._empirical_distance(reference_values[key], sampled_values[key])
                for key in reference_values
            },
            "multi_vehicle_interaction": tail._interaction(one_fields, one_state, reference_fields, reference_target),
            "temporal_dynamics": tail._temporal(one_fields, reference_fields),
            "physical_validity": _merge_counts(physical_counts),
            "within_flow_start_diversity": {
                "mean_pairwise_endpoint_distance_m": float(np.mean(within_fde)),
                "mean_risk_score_standard_deviation": float(np.mean(within_risk_std)),
            },
            "candidate_probability_mean": np.concatenate(candidate_probabilities).mean(axis=(0, 1)).tolist(),
            "two_sample_trajectory_features": multivariate_feature_distance(generated_rows, observed_rows, seed=SEED),
        },
        "interpretation": {
            "primary_metric": "conditional closed-loop safety and interaction distribution fidelity under a fixed external ego replay policy",
            "secondary_metrics": ["B0 execution fidelity", "physical validity", "within-start stochastic diversity"],
            "excluded_metric": "per-episode minADE/minFDE, because Flow C0 is intentionally sampled rather than copied from its replay donor",
            "limitation": "The Flow does not model map geometry or future ego policy. Their held-out replay pairing is an explicit external test condition, not a learned joint sample.",
        },
    }
    path = output / "flow_ramp_composition.json"
    save_json(report, path)
    print(path)
    return report


if __name__ == "__main__":
    evaluate_flow_ramp_composition()

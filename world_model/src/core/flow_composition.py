"""Shared EVT Flow start sampling for formal world-model composition tests."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from normalizing_flow.src.sampling import load_checkpoint_and_dataset, sample_tail_c0

from .data import SPLIT_TO_INDEX
from .initial_behavior_anchor import start_state_from_flow_feature
from .long_tail_metrics import (
    collision_metrics,
    distribution_values,
    empirical_distance,
    event_masks,
    feature_distribution_distance,
    speed_kl_divergence,
    traffic_fields,
)
from .sequential_dataset import load_sequential_dataset
from .utils import ensure_dir, file_sha256, save_json, select_device


OUTER_FLOW_SAMPLES = 8
INNER_WORLD_SAMPLES = 4
HISTORY_FRAMES = 25
ROLLOUT_FRAMES = 125
FLOW_COMPOSITION_SEED = 20260729


def tensor(value: np.ndarray, device):
    import torch

    return torch.from_numpy(np.asarray(value).copy()).to(device)


def repeat_batch(batch: dict[str, Any], copies: int) -> dict[str, Any]:
    return {key: value.repeat_interleave(int(copies), dim=0) for key, value in batch.items()}


def translated_ego_replay(
    donor: np.ndarray,
    initial_ego: np.ndarray,
    *,
    rollout_frames: int = ROLLOUT_FRAMES,
) -> np.ndarray:
    """Translate a logged ego replay to a sampled Flow C0 origin.

    ``rollout_frames`` is explicit because QR consumes every available highD
    transition (149), whereas older model families retain their 125-frame
    protocol.
    """
    frames = int(rollout_frames)
    if frames < 1:
        raise ValueError("rollout_frames must be positive")
    anchor = np.asarray(donor[HISTORY_FRAMES - 1, 0], np.float32)
    future = np.asarray(donor[HISTORY_FRAMES : HISTORY_FRAMES + frames, 0], np.float32)
    if future.shape[0] != frames:
        raise ValueError(
            f"donor replay has {future.shape[0]} future states, expected {frames}"
        )
    return future - anchor + initial_ego


def decode_flow_starts(starts: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode sampled Flow rows for models that consume NumPy scene batches."""
    features = np.asarray(starts["features"], np.float32)
    slot_masks = np.asarray(starts["slot_mask"], bool)
    if features.ndim != 2 or features.shape[1] != 76 or slot_masks.shape != (len(features), 6):
        raise ValueError("Flow starts require features [batch, 76] and slot_mask [batch, 6]")
    states = np.zeros((len(features), 7, 6), np.float32)
    valid = np.zeros((len(features), 7), bool)
    anchor = np.zeros((len(features), 6, 6), np.float32)
    anchor_valid = np.zeros((len(features), 6), bool)
    for index, (feature, slot_mask) in enumerate(zip(features, slot_masks)):
        states[index], valid[index], anchor[index], anchor_valid[index] = start_state_from_flow_feature(feature, slot_mask)
    return states, valid, anchor, anchor_valid


def write_flow_composition_report(
    *,
    checkpoint: Path,
    output_dir: Path,
    protocol: dict[str, Any],
    generated: np.ndarray,
    ego: np.ndarray,
    valid: np.ndarray,
    target: np.ndarray,
    target_ego: np.ndarray,
    target_valid: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the shared distribution report for a completed Flow composition."""
    generated_fields = traffic_fields(generated, ego, valid)
    target_fields = traffic_fields(target, target_ego, target_valid)
    generated_values = distribution_values(generated_fields)
    target_values = distribution_values(target_fields)

    def exceedance(
        real: np.ndarray, generated_value: np.ndarray, *, threshold: float, less_than: bool,
    ) -> dict[str, float]:
        """Compare a tail probability without treating either sample as paired."""
        left = np.asarray(real, np.float64).reshape(-1)
        right = np.asarray(generated_value, np.float64).reshape(-1)
        left, right = left[np.isfinite(left)], right[np.isfinite(right)]
        if not len(left) or not len(right):
            return {"real_probability": float("nan"), "generated_probability": float("nan"), "absolute_error": float("nan"), "relative_error": float("nan")}
        if less_than:
            real_probability = float(np.mean(left < threshold))
            generated_probability = float(np.mean(right < threshold))
        else:
            real_probability = float(np.mean(left > threshold))
            generated_probability = float(np.mean(right > threshold))
        absolute_error = abs(generated_probability - real_probability)
        return {
            "real_probability": real_probability,
            "generated_probability": generated_probability,
            "absolute_error": float(absolute_error),
            "relative_error": float(absolute_error / max(real_probability, 1.0e-6)),
        }

    def event_rates(background: np.ndarray, ego_state: np.ndarray, present: np.ndarray, fields: dict[str, np.ndarray]) -> dict[str, float]:
        labels = event_masks(background, ego_state, present)
        following = np.asarray(fields["following_valid"], bool)
        near_collision = np.any((np.asarray(fields["gap_m"]) < 2.0) & following, axis=(1, 2, 3))
        labels.update({
            "near_collision_gap_lt_2m": near_collision,
            "collision": np.asarray(fields["collision"], bool).any(axis=(1, 2, 3)),
        })
        return {name: float(np.mean(value)) for name, value in labels.items()}

    generated_events = event_rates(generated, ego, valid, generated_fields)
    target_events = event_rates(target, target_ego, target_valid, target_fields)
    event_distribution = {
        name: {
            "real_episode_rate": target_events[name],
            "generated_episode_rate": generated_events[name],
            "absolute_error": float(abs(generated_events[name] - target_events[name])),
            "relative_error": float(
                abs(generated_events[name] - target_events[name]) / max(target_events[name], 1.0e-6)
            ),
        }
        for name in target_events
    }
    report = {
        "protocol": protocol,
        "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
        "closed_loop_distribution": {
            "motion_variable_distribution": {
                key: {
                    **empirical_distance(target_values[key], generated_values[key]),
                    "kl_real_to_generated": speed_kl_divergence(target_values[key], generated_values[key]),
                }
                for key in ("speed_mps", "acceleration_mps2", "jerk_mps3", "curvature_m_inv")
            },
            "risk_variable_distribution": {
                key: empirical_distance(target_values[key], generated_values[key])
                for key in ("ttc_s", "drac_mps2", "gap_m", "relative_speed_mps")
            },
            "risk_tail_exceedance": {
                "ttc_lt_1s": exceedance(target_values["ttc_s"], generated_values["ttc_s"], threshold=1.0, less_than=True),
                "drac_gt_3mps2": exceedance(target_values["drac_mps2"], generated_values["drac_mps2"], threshold=3.0, less_than=False),
                "gap_lt_2m": exceedance(target_values["gap_m"], generated_values["gap_m"], threshold=2.0, less_than=True),
            },
            "semantic_event_distribution": event_distribution,
            "physical_validity": collision_metrics(generated_fields),
            **feature_distribution_distance(
                generated, ego, valid, target, target_ego, target_valid, seed=FLOW_COMPOSITION_SEED
            ),
        },
        **(extra or {}),
    }
    destination = ensure_dir(output_dir) / "flow_composition_evaluation.json"
    save_json(report, destination)
    print(destination)
    return report


def _event_groups(cache: dict[str, np.ndarray], rows: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    valid = np.asarray(cache["agent_valid"])[rows, HISTORY_FRAMES - 1, 1:].astype(bool)
    patterns = np.sum(valid * (1 << np.arange(valid.shape[1])), axis=1)
    primary = np.where(valid[:, 0], 0, np.argmax(valid, axis=1))
    for row, pattern, primary_slot, mask in zip(rows, patterns, primary, valid):
        if mask.any():
            groups[(int(pattern), int(primary_slot))].append(int(row))
    return {key: np.asarray(value, np.int64) for key, value in groups.items()}


def _match_reference_replays(
    cache: dict[str, np.ndarray],
    candidates: np.ndarray,
    sampled_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match sampled Flow starts to a compatible reference ego replay.

    The frozen Flow controls the discrete event structure (slot mask and
    primary risk slot).  Within that exact structure, the only continuous
    replay-side condition available in the sequential cache is the ego
    longitudinal speed at the history anchor.  Use deterministic nearest
    neighbour matching for it, rather than retaining the arbitrary source row
    that happened to request a Flow draw.
    """

    candidates = np.asarray(candidates, np.int64)
    sampled_speed = np.asarray(sampled_features, np.float32)[:, 0]
    replay_speed = np.asarray(
        cache["agent_states"][candidates, HISTORY_FRAMES - 1, 0, 2], np.float32
    )
    distances = np.abs(sampled_speed[:, None] - replay_speed[None, :])
    nearest = np.argmin(distances, axis=1)
    donors = candidates[nearest]
    return donors, replay_speed[nearest], distances[np.arange(len(nearest)), nearest].astype(np.float32)


def load_flow_tail_starts(
    repo_root: Path,
    *,
    device=None,
    replay_scope: str = "held_out_test",
    sequence_cache_owner: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Draw Flow C0/B0 starts and match compatible reference ego replays.

    Matching is exact for Flow's event structure (slot mask and primary risk
    slot), uses the single highD straight-lane road cohort, and minimizes the
    initial longitudinal ego-speed mismatch within that cohort.
    """
    device = select_device("auto") if device is None else device
    flow_checkpoint = repo_root / "results/highd_tail_flow/checkpoints/best_tail_conditional_maf.pt"
    flow_schema_path = repo_root / "results/highd_tail_flow/dataset_schema.json"
    flow, flow_arrays, flow_schema, _ = load_checkpoint_and_dataset(
        flow_checkpoint, repo_root / "results/highd_tail_flow", repo_root=repo_root, device=device
    )
    flow_checkpoint_sha256 = file_sha256(flow_checkpoint)
    flow_schema_sha256 = file_sha256(flow_schema_path)
    # Baseline model families retain their own default.  QR passes its
    # raw-150-state cache explicitly, so its 5.96 s protocol cannot read a
    # different model's replay cache.
    cache_owner = sequence_cache_owner or (
        repo_root / "results/highd_world_model/training_data/semi_markov_sequence_cache"
    )
    cache, _ = load_sequential_dataset(cache_owner)
    if replay_scope == "held_out_test":
        reference_rows = np.flatnonzero(
            (np.asarray(cache["split_index"]) == SPLIT_TO_INDEX["test"])
            & np.asarray(cache["is_evt_tail"], bool)
        )
    elif replay_scope == "all_evt_tail":
        reference_rows = np.flatnonzero(np.asarray(cache["is_evt_tail"], bool))
    else:
        raise ValueError("replay_scope must be 'held_out_test' or 'all_evt_tail'")
    groups = _event_groups(cache, reference_rows)
    trained = np.flatnonzero(np.asarray(flow_arrays["split_index"]) == SPLIT_TO_INDEX["train"])
    supported = {(int(flow_arrays["mask_pattern"][row]), int(flow_arrays["primary_slot_index"][row])) for row in trained}
    pieces: dict[str, list[np.ndarray]] = defaultdict(list)
    donors: list[np.ndarray] = []
    sample_fields = (
        "features",
        "features_normalized",
        "feature_valid",
        "contexts",
        "event_structure",
        "slot_mask",
        "mask_pattern",
        "primary_slot_index",
        "primary_slot_name",
        "event_structure_id",
        "event_structure_log_prob",
        "conditional_log_prob",
        "log_prob",
    )
    for offset, (key, rows) in enumerate(sorted(groups.items())):
        if key not in supported:
            continue
        sampling_seed = FLOW_COMPOSITION_SEED + 1009 * offset
        sampled = sample_tail_c0(
            flow, flow_arrays, flow_schema, num_samples=len(rows) * OUTER_FLOW_SAMPLES,
            device=device, seed=sampling_seed,
            mask_pattern=key[0], primary_slot=key[1], event_structure_split="train",
            event_structure_sampling="quota", reject_invalid=True, max_rounds=80,
            oversample_factor=1, min_draw=1, temperature=1.0295,
        )
        sample_count = len(sampled["features"])
        for name in sample_fields:
            pieces[name].append(np.asarray(sampled[name]))
        audit_values = {
            "flow_checkpoint_sha256": flow_checkpoint_sha256,
            "flow_schema_sha256": flow_schema_sha256,
            "sampling_seed": np.int64(sampling_seed),
            "sampling_temperature": np.float32(1.0295),
            "sampling_event_structure": "quota",
            "sampling_reject_invalid": True,
            "sampling_max_rounds": np.int64(80),
            "sampling_oversample_factor": np.int64(1),
            "sampling_min_draw": np.int64(1),
            "sampling_num_rejected": np.int64(sampled["num_rejected"][0]),
            "sampling_rejection_rate": np.float32(sampled["rejection_rate"][0]),
        }
        for name, value in audit_values.items():
            pieces[name].append(np.full(sample_count, value))
        matched_rows, matched_speed, speed_error = _match_reference_replays(
            cache, rows, np.asarray(sampled["features"], np.float32)
        )
        donors.append(matched_rows)
        pieces["matched_replay_ego_vx_mps"].append(matched_speed)
        pieces["matched_replay_ego_vx_abs_error_mps"].append(speed_error)
        pieces["road_type"].append(np.full(len(matched_rows), "highd_straight_lane"))
    if not donors:
        raise RuntimeError("no reference EVT-tail replay structure is supported by the frozen Flow")
    starts = {name: np.concatenate(parts) for name, parts in pieces.items()}
    return starts, cache, np.concatenate(donors)

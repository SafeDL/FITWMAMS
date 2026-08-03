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
    feature_distribution_distance,
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


def translated_ego_replay(donor: np.ndarray, initial_ego: np.ndarray) -> np.ndarray:
    anchor = np.asarray(donor[HISTORY_FRAMES - 1, 0], np.float32)
    future = np.asarray(donor[HISTORY_FRAMES : HISTORY_FRAMES + ROLLOUT_FRAMES, 0], np.float32)
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
    report = {
        "protocol": protocol,
        "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
        "closed_loop_distribution": {
            "risk_variable_distribution": {
                key: empirical_distance(target_values[key], generated_values[key])
                for key in ("ttc_s", "drac_mps2", "gap_m", "relative_speed_mps")
            },
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


def load_flow_tail_starts(
    repo_root: Path, *, device=None
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Load frozen Flow and draw eight valid C0/B0 starts per supported tail replay."""
    device = select_device("auto") if device is None else device
    flow_checkpoint = repo_root / "results/highd_tail_flow/checkpoints/best_tail_conditional_maf.pt"
    flow, flow_arrays, flow_schema, _ = load_checkpoint_and_dataset(
        flow_checkpoint, repo_root / "results/highd_tail_flow", repo_root=repo_root, device=device
    )
    cache, _ = load_sequential_dataset(repo_root / "results/highd_world_model/training_data/semi_markov_sequence_cache")
    heldout = np.flatnonzero(
        (np.asarray(cache["split_index"]) == SPLIT_TO_INDEX["test"])
        & np.asarray(cache["is_evt_tail"], bool)
    )
    groups = _event_groups(cache, heldout)
    trained = np.flatnonzero(np.asarray(flow_arrays["split_index"]) == SPLIT_TO_INDEX["train"])
    supported = {(int(flow_arrays["mask_pattern"][row]), int(flow_arrays["primary_slot_index"][row])) for row in trained}
    pieces: dict[str, list[np.ndarray]] = defaultdict(list)
    donors: list[np.ndarray] = []
    for offset, (key, rows) in enumerate(sorted(groups.items())):
        if key not in supported:
            continue
        sampled = sample_tail_c0(
            flow, flow_arrays, flow_schema, num_samples=len(rows) * OUTER_FLOW_SAMPLES,
            device=device, seed=FLOW_COMPOSITION_SEED + 1009 * offset,
            mask_pattern=key[0], primary_slot=key[1], event_structure_split="train",
            event_structure_sampling="quota", reject_invalid=True, max_rounds=80,
            oversample_factor=1, min_draw=1, temperature=1.0295,
        )
        for name, value in sampled.items():
            pieces[name].append(np.asarray(value))
        donors.append(np.repeat(rows, OUTER_FLOW_SAMPLES))
    if not donors:
        raise RuntimeError("no held-out EVT replay structure is supported by the frozen Flow")
    starts = {name: np.concatenate(parts) for name, parts in pieces.items()}
    return starts, cache, np.concatenate(donors)

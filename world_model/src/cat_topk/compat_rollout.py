"""Frozen CAT-TopK START/ROLL compatibility rollout.

This is an adapter for the archived CAT-TopK interface.  It is intentionally
kept with the baseline implementation, not in a cross-model comparison script.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from world_model.src.core.data import ROLL_MODE_INDEX, START_MODE_INDEX
from world_model.src.core.schema import SLOT_NAMES

from .evaluation import _model_actions_normalized
from .model import numpy_batch_to_torch
from .rollout import (
    build_relation_features_from_current,
    integrate_background_actions_batch,
    normalize_relation_features,
    normalize_states,
    unnormalize_actions,
)


def legacy_sequence_rows(
    arrays: dict[str, np.ndarray], sequence_ids: np.ndarray, *, horizon_steps: int, chunks: int
) -> np.ndarray:
    """Find one archived START row and all required ROLL rows per sequence."""
    start: dict[str, int] = {}
    roll: dict[tuple[str, int], int] = {}
    for index, segment in enumerate(arrays["segment_id"]):
        key = str(segment)
        mode, offset = int(arrays["mode_index"][index]), int(arrays["offset"][index])
        if mode == START_MODE_INDEX and offset == 0:
            start[key] = index
        elif mode == ROLL_MODE_INDEX:
            roll[(key, offset)] = index
    rows, missing = [], []
    for sequence_id in sequence_ids:
        key = str(sequence_id)
        sequence = [start.get(key, -1)] + [
            roll.get((key, chunk * int(horizon_steps)), -1)
            for chunk in range(1, int(chunks))
        ]
        if min(sequence) < 0:
            missing.append(key)
        rows.append(sequence)
    if missing:
        raise RuntimeError(
            f"{len(missing)} highD sequences lack a complete CAT-TopK START/ROLL chain "
            f"(first={missing[0]})"
        )
    return np.asarray(rows, dtype=np.int64)


def rollout_legacy_chunks(
    model,
    arrays: dict[str, np.ndarray],
    schema: dict[str, Any],
    sequences: np.ndarray,
    *,
    chunks: int,
    device,
    seed: int,
    deterministic: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll frozen CAT-TopK chunks in the initial global coordinate frame."""
    import torch

    horizon = int(schema["horizon_steps"])
    if horizon * int(chunks) > 125:
        raise ValueError("CAT-TopK compatibility rollout exceeds five seconds")
    batch_size = len(sequences)
    current_indices = sequences[:, 0]
    batch = numpy_batch_to_torch(arrays, current_indices, device)
    current = np.asarray(arrays["current_states"][current_indices], np.float32)
    current_valid = np.asarray(arrays["current_valid"][current_indices], bool)
    origins = np.zeros((batch_size, 2), np.float32)
    outputs, targets = [], []
    with torch.no_grad():
        for chunk in range(int(chunks)):
            normalized = _model_actions_normalized(
                model, batch, device=device, deterministic=deterministic,
                temperature=1.0, seed=int(seed) + chunk,
            )
            actions = unnormalize_actions(normalized, schema)
            generated, generated_valid = integrate_background_actions_batch(
                current, current_valid, actions, dt=1.0 / float(schema["fps"])
            )
            global_generated = generated.copy()
            global_generated[..., :2] += origins[:, None, None, :]
            outputs.append(global_generated.astype(np.float32))
            target = np.asarray(arrays["target_states"][current_indices], np.float32).copy()
            target[..., :2] += origins[:, None, None, :]
            targets.append(target)
            if chunk + 1 == int(chunks):
                break

            ego_history = np.asarray(arrays["ego_future_states"][current_indices], np.float32)
            ego_valid = np.asarray(arrays["ego_future_valid"][current_indices], bool)
            local_origin = ego_history[:, -1, :2].copy()
            origins += local_origin
            history = np.zeros((batch_size, horizon, 1 + len(SLOT_NAMES), generated.shape[-1]), np.float32)
            history_valid = np.zeros((batch_size, horizon, 1 + len(SLOT_NAMES)), bool)
            history[:, :, 0], history[:, :, 1:] = ego_history, generated
            history_valid[:, :, 0], history_valid[:, :, 1:] = ego_valid, generated_valid
            history[..., :2] -= local_origin[:, None, None, :]
            history[~history_valid] = 0.0
            current, current_valid = history[:, -1], history_valid[:, -1]
            relation = np.stack([
                build_relation_features_from_current(
                    current[row], current_valid[row],
                    primary_slot_index=int(arrays["primary_slot_index"][sequences[row, chunk + 1]]),
                )
                for row in range(batch_size)
            ]).astype(np.float32)
            next_indices = sequences[:, chunk + 1]
            batch = {
                "history_states": torch.from_numpy(normalize_states(history, history_valid, schema)).float().to(device),
                "history_valid": torch.from_numpy(history_valid).bool().to(device),
                "current_states": torch.from_numpy(normalize_states(current, current_valid, schema)).float().to(device),
                "current_valid": torch.from_numpy(current_valid).bool().to(device),
                "mode_index": torch.full((batch_size,), ROLL_MODE_INDEX, dtype=torch.long, device=device),
                "primary_slot_index": torch.from_numpy(arrays["primary_slot_index"][next_indices]).long().to(device),
                "flow_action_summary": torch.zeros(
                    (batch_size, len(SLOT_NAMES), len(schema["flow_action_summary_features"])),
                    dtype=torch.float32, device=device,
                ),
                "relation_features": torch.from_numpy(
                    normalize_relation_features(relation, current_valid[:, 1:], schema)
                ).float().to(device),
            }
            current_indices = next_indices
    return np.concatenate(outputs, axis=1), np.concatenate(targets, axis=1)

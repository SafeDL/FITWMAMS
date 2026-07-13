#!/usr/bin/env python3
"""以相同 highD 样本对比冻结基线与候选世界模型。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.data import (  # noqa: E402
    dataset_dir_from_config,
    load_world_model_dataset,
    output_dir_from_config,
    split_indices,
)
from world_model.src.evaluation import _model_actions_normalized  # noqa: E402
from world_model.src.model import load_checkpoint, numpy_batch_to_torch  # noqa: E402
from world_model.src.rollout import (  # noqa: E402
    build_relation_features_from_current,
    integrate_background_actions_batch,
    normalize_relation_features,
    normalize_states,
    unnormalize_actions,
)
from world_model.src.schema import ROLL_MODE_INDEX, START_MODE_INDEX, SLOT_NAMES  # noqa: E402
from world_model.src.utils import ensure_dir, load_yaml, save_json, select_device, set_seed  # noqa: E402


def _batched(indices: np.ndarray, batch_size: int):
    for start in range(0, len(indices), int(batch_size)):
        yield indices[start : start + int(batch_size)]


def _per_sample_metrics(
    states: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray,
    ego_future: np.ndarray,
) -> dict[str, np.ndarray]:
    mask = np.asarray(valid, dtype=bool)
    dist = np.linalg.norm(states[..., :2] - targets[..., :2], axis=-1)
    count = mask.sum(axis=(1, 2)).clip(min=1)
    ade = (dist * mask).sum(axis=(1, 2)) / count
    final_mask = mask[:, -1]
    fde = np.full(len(mask), np.nan, dtype=np.float64)
    final_count = final_mask.sum(axis=1)
    has_final = final_count > 0
    fde[has_final] = (dist[:, -1] * final_mask).sum(axis=1)[has_final] / final_count[has_final]
    gap_pred = np.abs(states[..., 0] - ego_future[:, :, None, 0])
    gap_target = np.abs(targets[..., 0] - ego_future[:, :, None, 0])
    gap_mae = (np.abs(gap_pred - gap_target) * mask).sum(axis=(1, 2)) / count
    return {"ADE_m": ade, "FDE_m": fde, "gap_mae_m": gap_mae}


def _predict_one_chunk(
    model,
    arrays: dict[str, np.ndarray],
    schema: dict[str, Any],
    indices: np.ndarray,
    *,
    device,
    batch_size: int,
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    dt = 1.0 / float(schema["fps"])
    for batch_indices in _batched(indices, batch_size):
        batch = numpy_batch_to_torch(arrays, batch_indices, device)
        action_norm = _model_actions_normalized(
            model,
            batch,
            device=device,
            deterministic=True,
            temperature=1.0,
            seed=seed_offset + int(batch_indices[0]),
            std_floor_normalized=None,
        )
        action_raw = unnormalize_actions(action_norm, schema)
        state_raw, _valid = integrate_background_actions_batch(
            arrays["current_states"][batch_indices],
            arrays["current_valid"][batch_indices],
            action_raw,
            dt=dt,
        )
        actions.append(action_raw)
        states.append(state_raw)
    return np.concatenate(actions), np.concatenate(states)


def _start_to_roll_predictions(
    model,
    arrays: dict[str, np.ndarray],
    schema: dict[str, Any],
    pairs: list[tuple[int, int]],
    *,
    device,
    batch_size: int,
    seed_offset: int,
) -> np.ndarray:
    import torch

    horizon = int(schema["horizon_steps"])
    dt = 1.0 / float(schema["fps"])
    output: list[np.ndarray] = []
    start_indices = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    roll_indices = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    for local_positions in _batched(np.arange(len(pairs), dtype=np.int64), batch_size):
        start_idx = start_indices[local_positions]
        roll_idx = roll_indices[local_positions]
        start_batch = numpy_batch_to_torch(arrays, start_idx, device)
        first_norm = _model_actions_normalized(
            model,
            start_batch,
            device=device,
            deterministic=True,
            temperature=1.0,
            seed=seed_offset + int(start_idx[0]),
            std_floor_normalized=None,
        )
        first_actions = unnormalize_actions(first_norm, schema)
        first_states, first_valid = integrate_background_actions_batch(
            arrays["current_states"][start_idx],
            arrays["current_valid"][start_idx],
            first_actions,
            dt=dt,
        )
        ego_history = arrays["ego_future_states"][start_idx].astype(np.float32)
        ego_valid = arrays["ego_future_valid"][start_idx].astype(bool)
        history = np.zeros((len(start_idx), horizon, 1 + len(SLOT_NAMES), len(schema["state_features"])), dtype=np.float32)
        history_valid = np.zeros((len(start_idx), horizon, 1 + len(SLOT_NAMES)), dtype=bool)
        history[:, :, 0] = ego_history
        history_valid[:, :, 0] = ego_valid
        history[:, :, 1:] = first_states
        history_valid[:, :, 1:] = first_valid
        origin = ego_history[:, -1, :2].copy()
        history[..., 0] -= origin[:, None, None, 0]
        history[..., 1] -= origin[:, None, None, 1]
        history[~history_valid] = 0.0
        current = history[:, -1]
        current_valid = history_valid[:, -1]
        relation = np.stack(
            [
                build_relation_features_from_current(
                    current[row],
                    current_valid[row],
                    primary_slot_index=int(arrays["primary_slot_index"][roll_idx[row]]),
                )
                for row in range(len(roll_idx))
            ]
        ).astype(np.float32)
        roll_batch = {
            "history_states": torch.from_numpy(normalize_states(history, history_valid, schema)).float().to(device),
            "history_valid": torch.from_numpy(history_valid).bool().to(device),
            "current_states": torch.from_numpy(normalize_states(current, current_valid, schema)).float().to(device),
            "current_valid": torch.from_numpy(current_valid).bool().to(device),
            "mode_index": torch.full((len(roll_idx),), ROLL_MODE_INDEX, dtype=torch.long, device=device),
            "primary_slot_index": torch.from_numpy(arrays["primary_slot_index"][roll_idx]).long().to(device),
            "flow_action_summary": torch.zeros(
                (len(roll_idx), len(SLOT_NAMES), len(schema["flow_action_summary_features"])),
                dtype=torch.float32,
                device=device,
            ),
            "relation_features": torch.from_numpy(
                normalize_relation_features(relation, current_valid[:, 1:], schema)
            ).float().to(device),
        }
        second_norm = _model_actions_normalized(
            model,
            roll_batch,
            device=device,
            deterministic=True,
            temperature=1.0,
            seed=seed_offset + 100000 + int(roll_idx[0]),
            std_floor_normalized=None,
        )
        second_actions = unnormalize_actions(second_norm, schema)
        second_states, _second_valid = integrate_background_actions_batch(
            current,
            current_valid,
            second_actions,
            dt=dt,
        )
        output.append(second_states)
    return np.concatenate(output)


def _bootstrap_difference(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    diff = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return {"num_pairs": 0, "candidate_minus_baseline": float("nan"), "upper_95": float("nan"), "passes": False}
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(repetitions), dtype=np.float64)
    for start in range(0, int(repetitions), 200):
        count = min(200, int(repetitions) - start)
        choice = rng.integers(0, len(diff), size=(count, len(diff)))
        means[start : start + count] = diff[choice].mean(axis=1)
    point = float(diff.mean())
    upper = float(np.quantile(means, 0.95))
    return {
        "num_pairs": int(len(diff)),
        "candidate_minus_baseline": point,
        "upper_95": upper,
        "passes": bool(point <= 0.0 and upper <= 0.0),
    }


def _mean_metrics(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    return {key: float(np.nanmean(value)) for key, value in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "world_model/scripts/configs/highd_world_model.yaml"))
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    config_dir = config_path.parent
    arrays, schema = load_world_model_dataset(dataset_dir_from_config(config, config_dir))
    device = select_device(config.get("evaluation", {}).get("device", "auto"))
    set_seed(int(args.seed))
    candidate, _candidate_payload = load_checkpoint(str(Path(args.candidate_checkpoint).resolve()), device)
    baseline, _baseline_payload = load_checkpoint(str(Path(args.baseline_checkpoint).resolve()), device)

    split_idx = split_indices(arrays, args.split)
    evt_start = split_idx[
        (arrays["mode_index"][split_idx] == START_MODE_INDEX)
        & arrays["is_evt_tail"][split_idx].astype(bool)
    ]
    if len(evt_start) == 0:
        raise RuntimeError(f"No EVT-tail START rows for split={args.split}")
    _candidate_actions, candidate_states = _predict_one_chunk(
        candidate, arrays, schema, evt_start, device=device, batch_size=args.batch_size, seed_offset=1100
    )
    _baseline_actions, baseline_states = _predict_one_chunk(
        baseline, arrays, schema, evt_start, device=device, batch_size=args.batch_size, seed_offset=1100
    )
    target_states = arrays["target_states"][evt_start]
    target_valid = arrays["target_valid"][evt_start]
    ego_future = arrays["ego_future_states"][evt_start]
    candidate_evt = _per_sample_metrics(candidate_states, target_states, target_valid, ego_future)
    baseline_evt = _per_sample_metrics(baseline_states, target_states, target_valid, ego_future)

    horizon = int(schema["horizon_steps"])
    start_idx = split_idx[arrays["mode_index"][split_idx] == START_MODE_INDEX]
    roll_idx = split_idx[
        (arrays["mode_index"][split_idx] == ROLL_MODE_INDEX)
        & (arrays["offset"][split_idx] == horizon)
    ]
    roll_by_segment = {str(arrays["segment_id"][index]): int(index) for index in roll_idx}
    pairs = [(int(index), roll_by_segment[str(arrays["segment_id"][index])]) for index in start_idx if str(arrays["segment_id"][index]) in roll_by_segment]
    if not pairs:
        raise RuntimeError(f"No logged-ego START/ROLL pairs for split={args.split}")
    candidate_roll = _start_to_roll_predictions(
        candidate, arrays, schema, pairs, device=device, batch_size=args.batch_size, seed_offset=2200
    )
    baseline_roll = _start_to_roll_predictions(
        baseline, arrays, schema, pairs, device=device, batch_size=args.batch_size, seed_offset=2200
    )
    roll_indices = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    candidate_closed = _per_sample_metrics(
        candidate_roll,
        arrays["target_states"][roll_indices],
        arrays["target_valid"][roll_indices],
        arrays["ego_future_states"][roll_indices],
    )
    baseline_closed = _per_sample_metrics(
        baseline_roll,
        arrays["target_states"][roll_indices],
        arrays["target_valid"][roll_indices],
        arrays["ego_future_states"][roll_indices],
    )

    comparisons = {
        "EVT_tail_START": {
            "candidate": _mean_metrics(candidate_evt),
            "baseline": _mean_metrics(baseline_evt),
            "paired_bootstrap": {
                key: _bootstrap_difference(
                    candidate_evt[key], baseline_evt[key], repetitions=args.bootstrap_repetitions, seed=args.seed + offset
                )
                for offset, key in enumerate(("ADE_m", "FDE_m", "gap_mae_m"))
            },
        },
        "logged_ego_START_to_ROLL": {
            "candidate": _mean_metrics(candidate_closed),
            "baseline": _mean_metrics(baseline_closed),
            "paired_bootstrap": {
                "ADE_m": _bootstrap_difference(
                    candidate_closed["ADE_m"], baseline_closed["ADE_m"], repetitions=args.bootstrap_repetitions, seed=args.seed + 10
                )
            },
        },
    }
    gates = [
        comparisons["EVT_tail_START"]["paired_bootstrap"][key]["passes"]
        for key in ("ADE_m", "FDE_m", "gap_mae_m")
    ]
    gates.append(comparisons["logged_ego_START_to_ROLL"]["paired_bootstrap"]["ADE_m"]["passes"])
    result = {
        "protocol": {
            "split": args.split,
            "candidate_temperature": 1.0,
            "integrator": "fixed_25Hz_background_integrator",
            "bootstrap_repetitions": int(args.bootstrap_repetitions),
            "criterion": "candidate_minus_baseline point estimate <= 0 and one-sided 95% bootstrap upper <= 0",
        },
        "candidate_checkpoint": str(Path(args.candidate_checkpoint).resolve()),
        "baseline_checkpoint": str(Path(args.baseline_checkpoint).resolve()),
        "comparisons": comparisons,
        "promotion_passes_primary_error_gate": bool(all(gates)),
    }
    out_dir = output_dir_from_config(config, config_dir)
    save_json(result, ensure_dir(out_dir) / "paired_bootstrap_comparison.json")
    print(result["promotion_passes_primary_error_gate"])


if __name__ == "__main__":
    main()

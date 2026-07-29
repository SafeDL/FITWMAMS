#!/usr/bin/env python3
"""Test the Semi-Markov World Model against frozen CAT-TopK on aligned highD segments.

The script aligns samples by immutable highD sequence id, then evaluates the
frozen CAT-K interface as released.  Each artifact explicitly records CAT-K's
future-action START summary instead of modifying that frozen baseline.
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

from world_model.src.core.data import (
    ROLL_MODE_INDEX,
    START_MODE_INDEX,
    dataset_dir_from_config,
    load_world_model_dataset,
    split_indices,
)
from world_model.src.cat_topk.evaluation import _model_actions_normalized
from world_model.src.core.initial_behavior_anchor import FrozenLegacyFlowSchema
from world_model.src.cat_topk.model import (
    load_checkpoint as load_catk_checkpoint,
    numpy_batch_to_torch,
)
from world_model.src.cat_topk.rollout import (
    build_relation_features_from_current,
    integrate_background_actions_batch,
    normalize_relation_features,
    normalize_states,
    unnormalize_actions,
)
from world_model.src.core.schema import SLOT_NAMES
from world_model.src.semi_markov.train import (
    FIELDS,
    _to_batch,
    load_semi_markov_checkpoint,
)
from world_model.src.core.sequential_dataset import (
    FLOW_ANCHOR_ARRAYS,
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(
    pred: np.ndarray, target: np.ndarray, valid: np.ndarray, ego: np.ndarray
) -> dict[str, np.ndarray]:
    mask = np.asarray(valid, bool)
    distance = np.linalg.norm(pred[..., :2] - target[..., :2], axis=-1)
    count = mask.sum(axis=(1, 2)).clip(min=1)
    ade = (distance * mask).sum(axis=(1, 2)) / count
    final = mask[:, -1]
    final_count = final.sum(axis=1).clip(min=1)
    fde = (distance[:, -1] * final).sum(axis=1) / final_count
    pred_gap = np.linalg.norm(pred[..., :2] - ego[:, :, None, :2], axis=-1)
    target_gap = np.linalg.norm(target[..., :2] - ego[:, :, None, :2], axis=-1)
    gap = (np.abs(pred_gap - target_gap) * mask).sum(axis=(1, 2)) / count
    return {"ADE_m": ade, "FDE_m": fde, "gap_mae_m": gap}


def _bootstrap(
    candidate: np.ndarray, baseline: np.ndarray, *, repetitions: int, seed: int
) -> dict[str, Any]:
    difference = np.asarray(candidate, np.float64) - np.asarray(baseline, np.float64)
    difference = difference[np.isfinite(difference)]
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(repetitions), np.float64)
    for start in range(0, int(repetitions), 100):
        count = min(100, int(repetitions) - start)
        sample = rng.integers(0, len(difference), size=(count, len(difference)))
        means[start : start + count] = difference[sample].mean(axis=1)
    point = float(difference.mean())
    upper = float(np.quantile(means, 0.95))
    return {
        "num_pairs": int(len(difference)),
        "candidate_minus_baseline": point,
        "upper_95": upper,
        "passes": bool(point <= 0.0 and upper <= 0.0),
    }


def _batch(arrays: dict[str, np.ndarray], index: np.ndarray, device):
    import torch

    names = tuple([*FIELDS, *[name for name in FLOW_ANCHOR_ARRAYS if name in arrays]])
    values = tuple(
        torch.from_numpy(np.asarray(arrays[field][index])) for field in names
    )
    return _to_batch(values, names, device)


def _relationship_counts(
    states: np.ndarray, valid: np.ndarray, *, lane_width_m: float = 3.6
) -> np.ndarray:
    """Count highD same/adjacent/unrelated pairs without retaining all rows."""
    counts = np.zeros(3, dtype=np.float64)
    for sample_states, sample_valid in zip(states, valid):
        for frame_states, frame_valid in zip(sample_states, sample_valid):
            active = np.flatnonzero(frame_valid)
            if len(active) < 2:
                continue
            lane = np.round(frame_states[:, 1] / float(lane_width_m)).astype(np.int64)
            for left in active:
                for right in active:
                    if left >= right:
                        continue
                    difference = abs(int(lane[left]) - int(lane[right]))
                    counts[0 if difference == 0 else 1 if difference == 1 else 2] += 1
    return counts


def _relationship_distribution(counts: np.ndarray) -> dict[str, float]:
    total = float(np.asarray(counts, np.float64).sum())
    values = (
        np.asarray(counts, np.float64) / total
        if total
        else np.zeros(3, dtype=np.float64)
    )
    return {
        "same_lane": float(values[0]),
        "adjacent_lane": float(values[1]),
        "unrelated": float(values[2]),
    }


def _total_variation(left: dict[str, float], right: dict[str, float]) -> float:
    return float(0.5 * sum(abs(left[key] - right[key]) for key in left))


def _legacy_sequences(
    source_arrays: dict[str, np.ndarray],
    sequence_ids: np.ndarray,
    *,
    horizon_steps: int,
    chunks: int,
) -> np.ndarray:
    """Look up the legacy START plus successive ROLL rows for every segment."""
    start_lookup: dict[str, int] = {}
    roll_lookup: dict[tuple[str, int], int] = {}
    for index in range(len(source_arrays["segment_id"])):
        segment_id = str(source_arrays["segment_id"][index])
        mode, offset = int(source_arrays["mode_index"][index]), int(
            source_arrays["offset"][index]
        )
        if mode == START_MODE_INDEX and offset == 0:
            start_lookup[segment_id] = int(index)
        elif mode == ROLL_MODE_INDEX:
            roll_lookup[(segment_id, offset)] = int(index)
    rows: list[list[int]] = []
    missing: list[str] = []
    for sequence_id in sequence_ids:
        segment_id = str(sequence_id)
        sequence = [start_lookup.get(segment_id, -1)]
        for chunk in range(1, int(chunks)):
            sequence.append(
                roll_lookup.get((segment_id, chunk * int(horizon_steps)), -1)
            )
        if min(sequence) < 0:
            missing.append(segment_id)
        rows.append(sequence)
    if missing:
        raise RuntimeError(
            f"{len(missing)} aligned semi-Markov sequences lack a legacy START/ROLL chain "
            f"of {chunks} chunks (first={missing[0]})"
        )
    return np.asarray(rows, dtype=np.int64)


def _catk_multichunk_rollout(
    model,
    source_arrays: dict[str, np.ndarray],
    source_schema: dict[str, Any],
    sequences: np.ndarray,
    *,
    chunks: int,
    device,
    seed: int,
    deterministic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay CAT-K START/ROLL while retaining the initial-sequence frame.

    The frozen cache stores every ROLL row in the then-current ego frame.  The
    legacy model therefore receives local states after every one-second
    transition; ``origins`` converts each generated chunk back to the initial
    sequence frame used by the semi-Markov sequence cache.
    """
    import torch

    horizon = int(source_schema["horizon_steps"])
    if horizon * int(chunks) > 125:
        raise ValueError(
            "comparison horizon exceeds the fixed five-second sequence target"
        )
    batch_size = len(sequences)
    current_indices = sequences[:, 0]
    batch = numpy_batch_to_torch(source_arrays, current_indices, device)
    current_raw = np.asarray(
        source_arrays["current_states"][current_indices], np.float32
    )
    current_valid = np.asarray(source_arrays["current_valid"][current_indices], bool)
    origins = np.zeros((batch_size, 2), dtype=np.float32)
    outputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for chunk in range(int(chunks)):
            action_norm = _model_actions_normalized(
                model,
                batch,
                device=device,
                deterministic=bool(deterministic),
                temperature=1.0,
                seed=int(seed) + chunk,
            )
            action_raw = unnormalize_actions(action_norm, source_schema)
            generated_local, generated_valid = integrate_background_actions_batch(
                current_raw,
                current_valid,
                action_raw,
                dt=1.0 / float(source_schema["fps"]),
            )
            generated_global = generated_local.copy()
            generated_global[..., :2] += origins[:, None, None, :]
            outputs.append(generated_global.astype(np.float32))
            target_local = np.asarray(
                source_arrays["target_states"][current_indices], np.float32
            )
            target_global = target_local.copy()
            target_global[..., :2] += origins[:, None, None, :]
            targets.append(target_global)
            if chunk + 1 >= int(chunks):
                break

            # Logged ego motion is deliberately supplied only *after* the
            # current transition, exactly as in CAT-K's model-state evaluator.
            ego_history = np.asarray(
                source_arrays["ego_future_states"][current_indices], np.float32
            )
            ego_valid = np.asarray(
                source_arrays["ego_future_valid"][current_indices], bool
            )
            local_origin = ego_history[:, -1, :2].copy()
            origins += local_origin
            history = np.zeros(
                (batch_size, horizon, 1 + len(SLOT_NAMES), generated_local.shape[-1]),
                np.float32,
            )
            history_valid = np.zeros((batch_size, horizon, 1 + len(SLOT_NAMES)), bool)
            history[:, :, 0] = ego_history
            history[:, :, 1:] = generated_local
            history_valid[:, :, 0] = ego_valid
            history_valid[:, :, 1:] = generated_valid
            history[..., 0] -= local_origin[:, None, None, 0]
            history[..., 1] -= local_origin[:, None, None, 1]
            history[~history_valid] = 0.0
            current_raw, current_valid = history[:, -1], history_valid[:, -1]
            relation = np.stack(
                [
                    build_relation_features_from_current(
                        current_raw[item],
                        current_valid[item],
                        primary_slot_index=int(
                            source_arrays["primary_slot_index"][
                                sequences[item, chunk + 1]
                            ]
                        ),
                    )
                    for item in range(batch_size)
                ]
            ).astype(np.float32)
            relation_valid = current_valid[:, 1:]
            next_indices = sequences[:, chunk + 1]
            batch = {
                "history_states": torch.from_numpy(
                    normalize_states(history, history_valid, source_schema)
                )
                .float()
                .to(device),
                "history_valid": torch.from_numpy(history_valid).bool().to(device),
                "current_states": torch.from_numpy(
                    normalize_states(current_raw, current_valid, source_schema)
                )
                .float()
                .to(device),
                "current_valid": torch.from_numpy(current_valid).bool().to(device),
                "mode_index": torch.full(
                    (batch_size,), ROLL_MODE_INDEX, dtype=torch.long, device=device
                ),
                "primary_slot_index": torch.from_numpy(
                    source_arrays["primary_slot_index"][next_indices]
                )
                .long()
                .to(device),
                "flow_action_summary": torch.zeros(
                    (
                        batch_size,
                        len(SLOT_NAMES),
                        len(source_schema["flow_action_summary_features"]),
                    ),
                    dtype=torch.float32,
                    device=device,
                ),
                "relation_features": torch.from_numpy(
                    normalize_relation_features(relation, relation_valid, source_schema)
                )
                .float()
                .to(device),
            }
            current_indices = next_indices
    return np.concatenate(outputs, axis=1), np.concatenate(targets, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--semi-checkpoint", required=True)
    parser.add_argument(
        "--catk-checkpoint",
        default=str(
            ROOT
            / "results/highd_world_model/cat_topk_world_model/checkpoints/best_world_model.pt"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--horizon-seconds", type=int, choices=(1, 5), default=1)
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=0,
        help="Bounded smoke-run only; 0 uses all held-out sequences.",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Append a label such as _smoke; bounded runs must not overwrite formal reports.",
    )
    args = parser.parse_args()

    semi_path, catk_path = (
        Path(args.semi_config).resolve(),
        Path(args.catk_config).resolve(),
    )
    semi_cfg, catk_cfg = load_yaml(semi_path), load_yaml(catk_path)
    device = select_device(str(semi_cfg.get("evaluation", {}).get("device", "auto")))
    set_seed(int(args.seed))
    # A full-data fine-tune writes checkpoints to its own output directory but
    # deliberately reuses the immutable cache owned by the base full-data run.
    # Resolve that owner just as training and evaluation do, rather than
    # assuming arrays are colocated with the checkpoint.
    semi_cache_owner = sequence_cache_owner_dir(semi_cfg, config_dir=semi_path.parent)
    semi_arrays, manifest = load_sequential_dataset(semi_cache_owner)
    if bool(manifest.get("bounded_development_cache", True)):
        raise RuntimeError(
            "paired comparison requires the complete highD sequence cache"
        )
    source_dir = dataset_dir_from_config(catk_cfg, catk_path.parent)
    source_arrays, source_schema = load_world_model_dataset(source_dir)
    horizon_steps = int(source_schema["horizon_steps"])
    chunks = int(args.horizon_seconds) * int(source_schema["fps"]) // horizon_steps
    if chunks * horizon_steps != int(args.horizon_seconds) * int(source_schema["fps"]):
        raise RuntimeError(
            "comparison horizon must be an exact multiple of the frozen CAT-K chunk length"
        )
    sequence_index = np.flatnonzero(np.asarray(semi_arrays["split_index"]) == 2)
    if int(args.max_sequences) > 0:
        sequence_index = sequence_index[: int(args.max_sequences)]
    source_sequences = _legacy_sequences(
        source_arrays,
        np.asarray(semi_arrays["sequence_id"])[sequence_index],
        horizon_steps=horizon_steps,
        chunks=chunks,
    )

    semi_checkpoint, catk_checkpoint = (
        Path(args.semi_checkpoint).resolve(),
        Path(args.catk_checkpoint).resolve(),
    )
    semi = load_semi_markov_checkpoint(semi_checkpoint, device=device)
    if semi.uses_behavior_anchor:
        flow_schema = semi_cfg.get("paths", {}).get("flow_schema")
        if not flow_schema:
            raise ValueError("Semi-Markov comparison requires paths.flow_schema")
        schema_path = Path(flow_schema)
        schema = FrozenLegacyFlowSchema.load(
            schema_path
            if schema_path.is_absolute()
            else (semi_path.parent / schema_path).resolve()
        )
        semi.set_frozen_flow_schema(schema)
        semi_arrays.update(
            ensure_frozen_flow_behavior_anchor_cache(
                semi_cache_owner, semi_arrays, manifest, schema
            )
        )
    catk, _ = load_catk_checkpoint(str(catk_checkpoint), device)
    import torch

    semi_rows: list[dict[str, np.ndarray]] = []
    catk_rows: list[dict[str, np.ndarray]] = []
    tail_rows: list[np.ndarray] = []
    parity_max_error = 0.0
    relation_counts = {
        "candidate": np.zeros(3, np.float64),
        "baseline": np.zeros(3, np.float64),
        "target": np.zeros(3, np.float64),
    }
    with torch.no_grad():
        for start in range(0, len(sequence_index), int(args.batch_size)):
            stop = min(start + int(args.batch_size), len(sequence_index))
            seq_idx, old_indices = (
                sequence_index[start:stop],
                source_sequences[start:stop],
            )
            rollout = semi.rollout_roll_mode(
                _batch(semi_arrays, seq_idx, device),
                seed=int(args.seed) + start,
                deterministic=True,
            )
            frames = chunks * horizon_steps
            semi_pred = rollout["predicted_states"][:, :frames, 1:].cpu().numpy()
            target = np.asarray(
                semi_arrays["agent_states"][seq_idx, 25 : 25 + frames, 1:], np.float32
            )
            valid = np.asarray(
                semi_arrays["agent_valid"][seq_idx, 25 : 25 + frames, 1:], bool
            )
            ego = np.asarray(
                semi_arrays["agent_states"][seq_idx, 25 : 25 + frames, 0], np.float32
            )
            catk_pred, source_target = _catk_multichunk_rollout(
                catk,
                source_arrays,
                source_schema,
                old_indices,
                chunks=chunks,
                device=device,
                seed=int(args.seed) + start,
            )
            source_valid = np.concatenate(
                [
                    np.asarray(
                        source_arrays["target_valid"][old_indices[:, chunk]], bool
                    )
                    for chunk in range(chunks)
                ],
                axis=1,
            )
            common = valid & source_valid
            if common.any():
                parity_max_error = max(
                    parity_max_error,
                    float(np.abs(target - source_target)[common].max()),
                )
            semi_rows.append(_metrics(semi_pred, target, valid, ego))
            catk_rows.append(_metrics(catk_pred, target, valid, ego))
            tail_rows.append(np.asarray(semi_arrays["is_evt_tail"][seq_idx], bool))
            if chunks > 1:
                relation_counts["candidate"] += _relationship_counts(semi_pred, valid)
                relation_counts["baseline"] += _relationship_counts(catk_pred, valid)
                relation_counts["target"] += _relationship_counts(target, valid)
    candidate = {
        key: np.concatenate([row[key] for row in semi_rows]) for key in semi_rows[0]
    }
    baseline = {
        key: np.concatenate([row[key] for row in catk_rows]) for key in catk_rows[0]
    }
    tail = np.concatenate(tail_rows)
    comparisons = {
        key: _bootstrap(
            candidate[key],
            baseline[key],
            repetitions=args.bootstrap_repetitions,
            seed=int(args.seed) + offset,
        )
        for offset, key in enumerate(("ADE_m", "FDE_m", "gap_mae_m"))
    }
    output_dir = ensure_dir(
        Path(semi_cfg["paths"]["output_dir"]).resolve()
        if Path(semi_cfg["paths"]["output_dir"]).is_absolute()
        else (semi_path.parent / semi_cfg["paths"]["output_dir"]).resolve()
    )
    relationship_report: dict[str, Any] = {}
    if chunks > 1:
        candidate_relation = _relationship_distribution(relation_counts["candidate"])
        baseline_relation = _relationship_distribution(relation_counts["baseline"])
        target_relation = _relationship_distribution(relation_counts["target"])
        candidate_tv = _total_variation(candidate_relation, target_relation)
        baseline_tv = _total_variation(baseline_relation, target_relation)
        relationship_report = {
            "candidate": {
                "predicted": candidate_relation,
                "target": target_relation,
                "total_variation": candidate_tv,
            },
            "baseline": {
                "predicted": baseline_relation,
                "target": target_relation,
                "total_variation": baseline_tv,
            },
            "candidate_minus_baseline_total_variation": float(
                candidate_tv - baseline_tv
            ),
        }
    report = {
        "protocol": {
            "same_sequence": True,
            "split": "test",
            "horizon_seconds": float(args.horizon_seconds),
            "catk_chunks": int(chunks),
            "bootstrap_repetitions": int(args.bootstrap_repetitions),
            "criterion": "candidate_minus_baseline point estimate <= 0 and one-sided 95% bootstrap upper <= 0",
            "baseline_uses_future_flow_action_summary": bool(
                catk_cfg.get("model", {}).get("use_start_flow_summary", False)
            ),
        },
        "candidate_checkpoint": str(semi_checkpoint),
        "candidate_checkpoint_sha256": _sha256(semi_checkpoint),
        "baseline_checkpoint": str(catk_checkpoint),
        "baseline_checkpoint_sha256": _sha256(catk_checkpoint),
        "num_paired_sequences": int(len(sequence_index)),
        "target_coordinate_parity_max_abs_error": float(parity_max_error),
        "candidate": {key: float(np.mean(value)) for key, value in candidate.items()},
        "baseline": {key: float(np.mean(value)) for key, value in baseline.items()},
        "paired_bootstrap": comparisons,
        "all_primary_error_gates_pass": bool(
            all(item["passes"] for item in comparisons.values())
        ),
        "evt_tail": {
            "num_paired_sequences": int(tail.sum()),
            "candidate": (
                {key: float(np.mean(value[tail])) for key, value in candidate.items()}
                if tail.any()
                else {}
            ),
            "baseline": (
                {key: float(np.mean(value[tail])) for key, value in baseline.items()}
                if tail.any()
                else {}
            ),
            "paired_bootstrap": (
                {
                    key: _bootstrap(
                        candidate[key][tail],
                        baseline[key][tail],
                        repetitions=args.bootstrap_repetitions,
                        seed=int(args.seed) + 100 + offset,
                    )
                    for offset, key in enumerate(("ADE_m", "FDE_m", "gap_mae_m"))
                }
                if tail.any()
                else {}
            ),
        },
        "relationship_distribution": relationship_report,
    }
    extra = str(args.output_suffix)
    if extra and not extra.startswith("_"):
        extra = "_" + extra
    path = output_dir / (
        f"paired_semi_markov_vs_catk{extra}.json"
        if chunks == 1
        else f"paired_semi_markov_vs_catk_{int(args.horizon_seconds)}s{extra}.json"
    )
    save_json(report, path)
    print(path)


if __name__ == "__main__":
    main()

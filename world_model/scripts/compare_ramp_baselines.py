#!/usr/bin/env python3
"""Paired full-highD comparison of RAMP-WM with frozen Semi-Markov and CAT-TopK.

CAT-TopK's archived START interface receives a future-action summary.  The
report therefore preserves it as a reproducibility comparison but explicitly
marks it as information-asymmetric and never presents it as a promotion proof.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.scripts.compare_semi_markov_cat_topk import (
    _batch,
    _bootstrap,
    _catk_multichunk_rollout,
    _legacy_sequences,
    _metrics,
    _relationship_counts,
    _relationship_distribution,
    _sha256,
    _total_variation,
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


def _background_metrics(
    pred: np.ndarray, target: np.ndarray, valid: np.ndarray, ego: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-sequence trajectory, interaction, and state-validity metrics."""
    mask = np.asarray(valid, bool)
    count = mask.sum(axis=(1, 2)).clip(min=1)
    final = mask[:, -1]
    final_count = final.sum(axis=1).clip(min=1)
    distance = np.linalg.norm(pred[..., :2] - target[..., :2], axis=-1)
    pred_gap = np.abs(pred[..., 0] - ego[:, :, None, 0])
    target_gap = np.abs(target[..., 0] - ego[:, :, None, 0])
    pred_rel_v = pred[..., 2] - ego[:, :, None, 2]
    target_rel_v = target[..., 2] - ego[:, :, None, 2]
    eps = 1.0e-3
    pred_ttc = np.where(
        pred_rel_v < -eps, pred_gap / np.maximum(-pred_rel_v, eps), np.inf
    )
    target_ttc = np.where(
        target_rel_v < -eps, target_gap / np.maximum(-target_rel_v, eps), np.inf
    )
    finite_ttc = mask & np.isfinite(pred_ttc) & np.isfinite(target_ttc)
    pred_drac = np.where(
        pred_rel_v < -eps,
        np.maximum(-pred_rel_v, 0.0) ** 2 / np.maximum(2.0 * pred_gap, eps),
        0.0,
    )
    target_drac = np.where(
        target_rel_v < -eps,
        np.maximum(-target_rel_v, 0.0) ** 2 / np.maximum(2.0 * target_gap, eps),
        0.0,
    )

    def mean(values: np.ndarray, weight: np.ndarray = mask) -> np.ndarray:
        denom = weight.sum(axis=(1, 2)).clip(min=1)
        return (values * weight).sum(axis=(1, 2)) / denom

    speed = np.linalg.norm(pred[..., 2:4], axis=-1)
    acceleration = np.linalg.norm(pred[..., 4:6], axis=-1)
    lateral_overlap = np.abs(pred[..., 1] - ego[:, :, None, 1]) < 1.0
    longitudinal_overlap = np.abs(pred[..., 0] - ego[:, :, None, 0]) < 4.5
    overlap = mask & lateral_overlap & longitudinal_overlap
    acceleration_bad = mask & (acceleration > 12.0)
    speed_bad = mask & ((speed < 0.0) | (speed > 75.0))
    jerk_bad = np.zeros_like(mask)
    jerk_bad[:, 1:] = mask[:, 1:] & (
        np.linalg.norm(np.diff(pred[..., 4:6], axis=1), axis=-1) / 0.04 > 40.0
    )
    pred_risk = mask & ((np.minimum(pred_ttc, 10.0) < 3.0) | (pred_drac > 2.0))
    target_risk = mask & ((np.minimum(target_ttc, 10.0) < 3.0) | (target_drac > 2.0))
    return {
        "ADE_m": mean(distance),
        "FDE_m": (distance[:, -1] * final).sum(axis=1) / final_count,
        "gap_mae_m": mean(np.abs(pred_gap - target_gap)),
        "relative_vx_mae_mps": mean(np.abs(pred_rel_v - target_rel_v)),
        "velocity_mae_mps": mean(
            np.linalg.norm(pred[..., 2:4] - target[..., 2:4], axis=-1)
        ),
        "acceleration_mae_mps2": mean(
            np.linalg.norm(pred[..., 4:6] - target[..., 4:6], axis=-1)
        ),
        "ttc_error_s": mean(
            np.abs(np.minimum(pred_ttc, 10.0) - np.minimum(target_ttc, 10.0)),
            finite_ttc,
        ),
        "drac_error_mps2": mean(np.abs(pred_drac - target_drac)),
        "small_ttc_rate": mean((pred_ttc < 3.0).astype(np.float32)),
        "risk_relaxation_rate": (
            (target_risk & ~pred_risk).sum(axis=(1, 2))
            / target_risk.sum(axis=(1, 2)).clip(min=1)
        ),
        "invalid_rate": np.any(
            overlap | acceleration_bad | speed_bad | jerk_bad, axis=(1, 2)
        ).astype(np.float32),
        "overlap_rate": mean(overlap.astype(np.float32)),
        "negative_gap_rate": mean(overlap.astype(np.float32)),
        "acceleration_out_of_range_rate": mean(acceleration_bad.astype(np.float32)),
        "speed_out_of_range_rate": mean(speed_bad.astype(np.float32)),
        "jerk_out_of_range_rate": mean(jerk_bad.astype(np.float32)),
    }


def _one_sided_report(
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    tail: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict:
    names = tuple(candidate)
    return {
        "candidate": {name: float(np.mean(candidate[name])) for name in names},
        "baseline": {name: float(np.mean(baseline[name])) for name in names},
        "paired_bootstrap": {
            name: _bootstrap(
                candidate[name],
                baseline[name],
                repetitions=repetitions,
                seed=seed + index,
            )
            for index, name in enumerate(names)
        },
        "evt_tail": {
            "num_paired_sequences": int(tail.sum()),
            "candidate": (
                {name: float(np.mean(candidate[name][tail])) for name in names}
                if tail.any()
                else {}
            ),
            "baseline": (
                {name: float(np.mean(baseline[name][tail])) for name in names}
                if tail.any()
                else {}
            ),
            "paired_bootstrap": (
                {
                    name: _bootstrap(
                        candidate[name][tail],
                        baseline[name][tail],
                        repetitions=repetitions,
                        seed=seed + 100 + index,
                    )
                    for index, name in enumerate(names)
                }
                if tail.any()
                else {}
            ),
        },
    }


def _relation_report(
    candidate_states: list[np.ndarray],
    semi_states: list[np.ndarray],
    cat_states: list[np.ndarray],
    targets: list[np.ndarray],
    valids: list[np.ndarray],
) -> dict:
    values = {
        "ramp": np.zeros(3),
        "semi_markov": np.zeros(3),
        "cat_topk": np.zeros(3),
        "target": np.zeros(3),
    }
    for ramp, semi, cat, target, valid in zip(
        candidate_states, semi_states, cat_states, targets, valids
    ):
        values["ramp"] += _relationship_counts(ramp, valid)
        values["semi_markov"] += _relationship_counts(semi, valid)
        values["cat_topk"] += _relationship_counts(cat, valid)
        values["target"] += _relationship_counts(target, valid)
    target = _relationship_distribution(values["target"])
    result = {"target": target}
    for name in ("ramp", "semi_markov", "cat_topk"):
        prediction = _relationship_distribution(values[name])
        result[name] = {
            "predicted": prediction,
            "total_variation": _total_variation(prediction, target),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ramp-config",
        default=str(ROOT / "world_model/scripts/configs/highd_ramp_world_model.yaml"),
    )
    parser.add_argument(
        "--ramp-checkpoint",
        default=str(
            ROOT
            / "results/highd_world_model/ramp_world_model/checkpoints/best_ramp_world_model.pt"
        ),
    )
    parser.add_argument(
        "--semi-config",
        default=str(
            ROOT / "world_model/scripts/configs/highd_semi_markov_world_model.yaml"
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
        "--catk-config",
        default=str(
            ROOT / "world_model/scripts/configs/highd_cat_topk_world_model.yaml"
        ),
    )
    parser.add_argument(
        "--catk-checkpoint",
        default=str(
            ROOT
            / "results/highd_world_model/cat_topk_world_model/checkpoints/best_world_model.pt"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--horizon-seconds", type=int, choices=(1, 5), default=5)
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=0,
        help="Bounded smoke run only; 0 is the complete test split.",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

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
    device = select_device(str(ramp_cfg.get("evaluation", {}).get("device", "auto")))
    set_seed(args.seed)
    owner = sequence_cache_owner_dir(ramp_cfg, config_dir=ramp_path.parent)
    arrays, manifest = load_sequential_dataset(owner)
    if manifest.get("bounded_development_cache", True):
        raise RuntimeError(
            "formal paired comparison requires the complete highD sequence cache"
        )
    schema_path = ramp_cfg.get("paths", {}).get("flow_schema")
    if not schema_path:
        raise ValueError("RAMP paired comparison requires paths.flow_schema")
    schema_file = Path(schema_path)
    schema = FrozenLegacyFlowSchema.load(
        schema_file
        if schema_file.is_absolute()
        else (ramp_path.parent / schema_file).resolve()
    )
    arrays.update(
        ensure_frozen_flow_behavior_anchor_cache(owner, arrays, manifest, schema)
    )
    # The Semi-Markov model uses identical cached B0 values on the same rows.
    ramp = load_ramp_checkpoint(Path(args.ramp_checkpoint).resolve(), device=device)
    semi = load_semi_markov_checkpoint(
        Path(args.semi_checkpoint).resolve(), device=device
    )
    semi.set_frozen_flow_schema(schema)
    source_dir = dataset_dir_from_config(cat_cfg, cat_path.parent)
    source_arrays, source_schema = load_world_model_dataset(source_dir)
    cat, _ = load_catk_checkpoint(str(Path(args.catk_checkpoint).resolve()), device)
    frames = int(args.horizon_seconds) * 25
    chunks = frames // int(source_schema["horizon_steps"])
    index = np.flatnonzero(np.asarray(arrays["split_index"]) == 2)
    if args.max_sequences:
        index = index[: args.max_sequences]
    legacy = _legacy_sequences(
        source_arrays,
        np.asarray(arrays["sequence_id"])[index],
        horizon_steps=int(source_schema["horizon_steps"]),
        chunks=chunks,
    )
    import torch

    rows = {"ramp": [], "semi_markov": [], "cat_topk": []}
    tails = []
    ramp_states = []
    semi_states = []
    cat_states = []
    targets = []
    valids = []
    parity = 0.0
    with torch.no_grad():
        for start in range(0, len(index), args.batch_size):
            stop = min(start + args.batch_size, len(index))
            selected = index[start:stop]
            batch = _batch(arrays, selected, device)
            ramp_roll = ramp.rollout_roll_mode(
                batch, seed=args.seed + start, deterministic=True
            )
            semi_roll = semi.rollout_roll_mode(
                batch, seed=args.seed + start, deterministic=True
            )
            target = np.asarray(
                arrays["agent_states"][selected, 25 : 25 + frames, 1:], np.float32
            )
            valid = np.asarray(
                arrays["agent_valid"][selected, 25 : 25 + frames, 1:], bool
            )
            ego = np.asarray(
                arrays["agent_states"][selected, 25 : 25 + frames, 0], np.float32
            )
            ramp_pred = ramp_roll["predicted_states"][:, :frames, 1:].cpu().numpy()
            semi_pred = semi_roll["predicted_states"][:, :frames, 1:].cpu().numpy()
            cat_pred, source_target = _catk_multichunk_rollout(
                cat,
                source_arrays,
                source_schema,
                legacy[start:stop],
                chunks=chunks,
                device=device,
                seed=args.seed + start,
            )
            source_valid = np.concatenate(
                [
                    np.asarray(
                        source_arrays["target_valid"][legacy[start:stop, chunk]], bool
                    )
                    for chunk in range(chunks)
                ],
                axis=1,
            )
            common = valid & source_valid
            if common.any():
                parity = max(
                    parity, float(np.abs(target - source_target)[common].max())
                )
            rows["ramp"].append(_background_metrics(ramp_pred, target, valid, ego))
            rows["semi_markov"].append(
                _background_metrics(semi_pred, target, valid, ego)
            )
            rows["cat_topk"].append(_background_metrics(cat_pred, target, valid, ego))
            tails.append(np.asarray(arrays["is_evt_tail"])[selected].astype(bool))
            ramp_states.append(ramp_pred)
            semi_states.append(semi_pred)
            cat_states.append(cat_pred)
            targets.append(target)
            valids.append(valid)
    metrics = {
        name: {
            metric: np.concatenate([item[metric] for item in chunks_])
            for metric in rows[name][0]
        }
        for name, chunks_ in rows.items()
    }
    tail = np.concatenate(tails)
    result = {
        "ramp_vs_semi_markov": _one_sided_report(
            metrics["ramp"],
            metrics["semi_markov"],
            tail,
            repetitions=args.bootstrap_repetitions,
            seed=args.seed,
        ),
        "ramp_vs_cat_topk": _one_sided_report(
            metrics["ramp"],
            metrics["cat_topk"],
            tail,
            repetitions=args.bootstrap_repetitions,
            seed=args.seed + 1000,
        ),
    }
    report = {
        "protocol": {
            "same_sequence": True,
            "split": "test",
            "horizon_seconds": args.horizon_seconds,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "candidate_start_information": "frozen B0 behavior anchor; removed after first second",
            "semi_markov_start_information": "frozen B0 behavior anchor",
            "cat_topk_start_information": "future action summary (archived frozen interface)",
            "cat_topk_information_symmetric": False,
        },
        "checkpoints": {
            "ramp": {
                "path": str(Path(args.ramp_checkpoint).resolve()),
                "sha256": _sha256(Path(args.ramp_checkpoint).resolve()),
            },
            "semi_markov": {
                "path": str(Path(args.semi_checkpoint).resolve()),
                "sha256": _sha256(Path(args.semi_checkpoint).resolve()),
            },
            "cat_topk": {
                "path": str(Path(args.catk_checkpoint).resolve()),
                "sha256": _sha256(Path(args.catk_checkpoint).resolve()),
            },
        },
        "num_paired_sequences": int(len(index)),
        "target_coordinate_parity_max_abs_error": parity,
        "comparisons": result,
        "relationship_distribution": (
            _relation_report(ramp_states, semi_states, cat_states, targets, valids)
            if args.horizon_seconds == 5
            else {}
        ),
    }
    if args.output:
        output = Path(args.output)
    else:
        # Configuration paths are relative to the configuration file, not the
        # caller's CWD.
        output_dir = Path(ramp_cfg["paths"]["output_dir"])
        if not output_dir.is_absolute():
            output_dir = ramp_path.parent / output_dir
        output = output_dir / f"paired_ramp_baselines_{args.horizon_seconds}s.json"
    output = output.resolve()
    ensure_dir(output.parent)
    save_json(report, output)
    print(output)


if __name__ == "__main__":
    main()

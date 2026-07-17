#!/usr/bin/env python3
"""Test BARS-M2 v5 against the frozen BARS-M1 checkpoint on held-out highD.

Both checkpoints receive the same immutable sequential-cache rows, frozen
behavior anchors, and deterministic rollout seed.  The result is intentionally
an evidence artifact: it reports each horizon separately and never treats an
unpaired summary as a model-selection result.
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

from world_model.scripts.test_bars_m2_v5_against_cat import _bootstrap, _metrics
from world_model.src.initial_behavior_anchor import FrozenLegacyFlowSchema
from world_model.src.semi_markov_train import FIELDS, _to_batch, load_semi_markov_checkpoint
from world_model.src.sequential_dataset import FLOW_ANCHOR_ARRAYS, ensure_frozen_flow_behavior_anchor_cache, load_sequential_dataset, sequence_cache_owner_dir
from world_model.src.utils import ensure_dir, load_yaml, save_json, select_device, set_seed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved(path: str, config_path: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (config_path.parent / value).resolve()


def _batch(arrays: dict[str, np.ndarray], index: np.ndarray, device):
    import torch

    names = tuple([*FIELDS, *[name for name in FLOW_ANCHOR_ARRAYS if name in arrays]])
    values = tuple(torch.from_numpy(np.asarray(arrays[name][index])) for name in names)
    return _to_batch(values, names, device)


def _collect_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    ego: np.ndarray,
    *,
    frames: int,
) -> dict[str, np.ndarray]:
    return _metrics(prediction[:, :frames], target[:, :frames], valid[:, :frames], ego[:, :frames])


def _paired_report(
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    *,
    repetitions: int,
    seed: int,
    tolerance_m: float,
) -> dict[str, Any]:
    bootstrap = {
        key: _bootstrap(candidate[key] - float(tolerance_m), baseline[key], repetitions=repetitions, seed=seed + offset)
        for offset, key in enumerate(("ADE_m", "FDE_m", "gap_mae_m"))
    }
    # Make the declared tolerance explicit rather than concealing it in the
    # bootstrap inputs above.  The raw candidate-minus-baseline numbers remain
    # the headline result.
    for key, entry in bootstrap.items():
        entry["candidate_minus_baseline"] += float(tolerance_m)
        entry["upper_95"] += float(tolerance_m)
        entry["non_inferior_within_tolerance"] = bool(
            entry["candidate_minus_baseline"] <= float(tolerance_m)
            and entry["upper_95"] <= float(tolerance_m)
        )
        entry["passes"] = entry["non_inferior_within_tolerance"]
    return {
        "candidate": {key: float(np.mean(value)) for key, value in candidate.items()},
        "bars_m1": {key: float(np.mean(value)) for key, value in baseline.items()},
        "paired_bootstrap": bootstrap,
        "all_metrics_non_inferior_within_tolerance": bool(all(item["passes"] for item in bootstrap.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-config", default=str(ROOT / "world_model/scripts/configs/highd_bars_m2_plan_carry_3s.yaml"))
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--bars-m1-config", default=str(ROOT / "world_model/scripts/configs/highd_behavior_anchored_semi_markov.yaml"))
    parser.add_argument("--bars-m1-checkpoint", default=str(ROOT / "results/highd_world_model/behavior_anchored_semi_markov_m1_start_roll_v2/checkpoints/best_semi_markov_relational.pt"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-sequences", type=int, default=0, help="Bounded smoke-run only; 0 evaluates the complete held-out split.")
    parser.add_argument("--split", choices=("val", "test"), default="test", help="Validation is calibration-only and must use --output-suffix.")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--non-inferiority-tolerance-m", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()
    if int(args.batch_size) <= 0 or int(args.max_sequences) < 0 or int(args.bootstrap_repetitions) <= 0:
        raise ValueError("batch size and bootstrap repetitions must be positive; max sequences cannot be negative")
    if float(args.non_inferiority_tolerance_m) < 0.0:
        raise ValueError("non-inferiority tolerance cannot be negative")
    if args.split != "test" and not args.output_suffix:
        raise ValueError("validation calibration requires --output-suffix so it cannot overwrite a formal report")
    candidate_config_path = Path(args.candidate_config).resolve()
    m1_config_path = Path(args.bars_m1_config).resolve()
    candidate_config = load_yaml(candidate_config_path)
    m1_config = load_yaml(m1_config_path)
    device = select_device(str(candidate_config.get("evaluation", {}).get("device", "auto")))
    set_seed(int(args.seed))
    cache_owner = sequence_cache_owner_dir(candidate_config, config_dir=candidate_config_path.parent)
    arrays, manifest = load_sequential_dataset(cache_owner)
    if bool(manifest.get("bounded_development_cache", True)):
        raise RuntimeError("BARS comparison requires the complete highD sequence cache")
    sequence_index = np.flatnonzero(np.asarray(arrays["split_index"]) == {"val": 1, "test": 2}[args.split])
    if int(args.max_sequences):
        sequence_index = sequence_index[: int(args.max_sequences)]
    if not len(sequence_index):
        raise RuntimeError("held-out highD split is empty")

    candidate_checkpoint = Path(args.candidate_checkpoint).resolve()
    m1_checkpoint = Path(args.bars_m1_checkpoint).resolve()
    candidate_model = load_semi_markov_checkpoint(candidate_checkpoint, device=device)
    m1_model = load_semi_markov_checkpoint(m1_checkpoint, device=device)
    if getattr(m1_model.cfg, "variant", "m1") != "m1":
        raise ValueError("--bars-m1-checkpoint must be a BARS-M1 checkpoint")
    schema_source = candidate_config if candidate_model.uses_behavior_anchor else m1_config
    schema_config_path = candidate_config_path if candidate_model.uses_behavior_anchor else m1_config_path
    schema_value = schema_source.get("paths", {}).get("flow_schema")
    if candidate_model.uses_behavior_anchor or m1_model.uses_behavior_anchor:
        if not schema_value:
            raise ValueError("behavior-anchored BARS comparison requires paths.flow_schema")
        schema = FrozenLegacyFlowSchema.load(_resolved(str(schema_value), schema_config_path))
        arrays.update(ensure_frozen_flow_behavior_anchor_cache(cache_owner, arrays, manifest, schema))
        candidate_model.set_frozen_flow_schema(schema)
        m1_model.set_frozen_flow_schema(schema)

    horizons = (1, 2, 3, 4, 5)
    candidate_rows = {seconds: {key: [] for key in ("ADE_m", "FDE_m", "gap_mae_m")} for seconds in horizons}
    m1_rows = {seconds: {key: [] for key in ("ADE_m", "FDE_m", "gap_mae_m")} for seconds in horizons}
    candidate_tail = {key: [] for key in ("ADE_m", "FDE_m", "gap_mae_m")}
    m1_tail = {key: [] for key in ("ADE_m", "FDE_m", "gap_mae_m")}
    tail_count = 0
    import torch

    with torch.no_grad():
        for start in range(0, len(sequence_index), int(args.batch_size)):
            index = sequence_index[start : start + int(args.batch_size)]
            batch = _batch(arrays, index, device)
            candidate_rollout = candidate_model.rollout_roll_mode(batch, seed=int(args.seed) + start, deterministic=True)
            m1_rollout = m1_model.rollout_roll_mode(batch, seed=int(args.seed) + start, deterministic=True)
            candidate_prediction = candidate_rollout["predicted_states"][:, :125, 1:].cpu().numpy()
            m1_prediction = m1_rollout["predicted_states"][:, :125, 1:].cpu().numpy()
            target = np.asarray(arrays["agent_states"][index, 25:150, 1:], np.float32)
            valid = np.asarray(arrays["agent_valid"][index, 25:150, 1:], bool)
            ego = np.asarray(arrays["agent_states"][index, 25:150, 0], np.float32)
            for seconds in horizons:
                frames = seconds * 25
                for key, value in _collect_metrics(candidate_prediction, target, valid, ego, frames=frames).items():
                    candidate_rows[seconds][key].append(value)
                for key, value in _collect_metrics(m1_prediction, target, valid, ego, frames=frames).items():
                    m1_rows[seconds][key].append(value)
            tail = np.asarray(arrays["is_evt_tail"][index], bool)
            if tail.any():
                tail_count += int(tail.sum())
                for key, value in _collect_metrics(candidate_prediction[tail], target[tail], valid[tail], ego[tail], frames=125).items():
                    candidate_tail[key].append(value)
                for key, value in _collect_metrics(m1_prediction[tail], target[tail], valid[tail], ego[tail], frames=125).items():
                    m1_tail[key].append(value)

    candidate_all = {seconds: {key: np.concatenate(values) for key, values in rows.items()} for seconds, rows in candidate_rows.items()}
    m1_all = {seconds: {key: np.concatenate(values) for key, values in rows.items()} for seconds, rows in m1_rows.items()}
    report: dict[str, Any] = {
        "protocol": {
            "same_sequence": True,
            "split": str(args.split),
            "deterministic": True,
            "seed": int(args.seed),
            "bootstrap_repetitions": int(args.bootstrap_repetitions),
            "non_inferiority_tolerance_m": float(args.non_inferiority_tolerance_m),
            "criterion": "candidate_minus_BARS-M1 one-sided 95% bootstrap upper <= declared tolerance",
        },
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_checkpoint_sha256": _sha256(candidate_checkpoint),
        "bars_m1_checkpoint": str(m1_checkpoint),
        "bars_m1_checkpoint_sha256": _sha256(m1_checkpoint),
        "num_paired_sequences": int(len(sequence_index)),
        "horizons": {
            f"{seconds}s": _paired_report(
                candidate_all[seconds], m1_all[seconds], repetitions=int(args.bootstrap_repetitions),
                seed=int(args.seed) + seconds * 10, tolerance_m=float(args.non_inferiority_tolerance_m),
            )
            for seconds in horizons
        },
        "evt_tail_5s": None,
    }
    if tail_count:
        report["evt_tail_5s"] = _paired_report(
            {key: np.concatenate(values) for key, values in candidate_tail.items()},
            {key: np.concatenate(values) for key, values in m1_tail.items()},
            repetitions=int(args.bootstrap_repetitions), seed=int(args.seed) + 99,
            tolerance_m=float(args.non_inferiority_tolerance_m),
        )
        report["evt_tail_5s"]["num_paired_sequences"] = int(tail_count)
    extra = str(args.output_suffix)
    if extra and not extra.startswith("_"):
        extra = "_" + extra
    output_value = Path(candidate_config["paths"]["output_dir"])
    output_dir = ensure_dir(output_value if output_value.is_absolute() else (candidate_config_path.parent / output_value).resolve())
    output_path = output_dir / f"paired_bars_m2_vs_m1{extra}.json"
    save_json(report, output_path)
    print(output_path)


if __name__ == "__main__":
    main()

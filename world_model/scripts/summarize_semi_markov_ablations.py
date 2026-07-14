#!/usr/bin/env python3
"""Collect comparable B0/B1/B2/Full highD ablation evidence.

The runner deliberately refuses to call a partial set a completed core
ablation.  It also records the sequence-cache identity and split counts, so a
development-cache comparison cannot accidentally be presented as a full-data
claim.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.utils import load_json, save_json


VARIANTS = {
    "b0": "B0: single-mode dynamic graph",
    "b1": "B1: joint latent state, stepwise refresh",
    "b2": "B2: learned duration, no immediate response",
    "full": "Full: duration plus intent-response decomposition",
}


def _number(data: dict[str, Any], *path: str) -> float | None:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return None if value is None else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default=str(ROOT / "results/highd_world_model/semi_markov_relational_10k"),
        help="Directory containing ablation_b0 through ablation_full.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows: dict[str, dict[str, Any]] = {}
    cache_identities: list[tuple[Any, ...]] = []
    missing: list[str] = []
    for variant, label in VARIANTS.items():
        folder = root / f"ablation_{variant}"
        training_path, evaluation_path = folder / "training_summary.json", folder / "semi_markov_evaluation_summary.json"
        if not training_path.exists() or not evaluation_path.exists():
            missing.append(variant)
            continue
        training, evaluation = load_json(training_path), load_json(evaluation_path)
        manifest = training.get("sequence_manifest", {})
        identity = (
            manifest.get("source_dataset"), manifest.get("cache_version"),
            manifest.get("num_sequences"), tuple(sorted((manifest.get("split_summary") or {}).items())),
            bool(manifest.get("bounded_development_cache", True)),
        )
        cache_identities.append(identity)
        rows[variant] = {
            "label": label,
            "checkpoint_sha256": training.get("checkpoint_sha256"),
            "validation_causal_FDE_m": training.get("best_causal_prior_rollout_FDE_m"),
            "test_sequences": evaluation.get("test_sequences"),
            "test_ADE_1s_m": _number(evaluation, "one_second_conditional_reconstruction", "ADE_1s_m"),
            "test_FDE_1s_m": _number(evaluation, "one_second_conditional_reconstruction", "FDE_1s_m"),
            "test_ADE_5s_m": _number(evaluation, "five_second_causal_prior_rollout", "ADE_5s_m"),
            "test_FDE_5s_m": _number(evaluation, "five_second_causal_prior_rollout", "FDE_5s_m"),
            "duration_calibration": evaluation.get("duration_calibration"),
        }
    complete = not missing
    comparable = bool(cache_identities) and all(item == cache_identities[0] for item in cache_identities)
    report = {
        "protocol": {
            "required_variants": VARIANTS,
            "all_variants_present": complete,
            "missing_variants": missing,
            "same_sequence_cache_and_split": comparable,
            "cache_identity": None if not cache_identities else {
                "source_dataset": cache_identities[0][0], "cache_version": cache_identities[0][1],
                "num_sequences": cache_identities[0][2], "split_summary": dict(cache_identities[0][3]),
                "bounded_development_cache": cache_identities[0][4],
            },
            "interpretation": "Development ablations are mechanism evidence only; they do not replace the full held-out highD evaluation.",
        },
        "variants": rows,
    }
    output = Path(args.output).resolve() if args.output else root / "core_ablation_summary.json"
    save_json(report, output)
    print(output)
    if not complete or not comparable:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""Compute recording-clustered paired effects between two completed runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from interactive_behavior_world_model.evaluation.statistics import clustered_paired_interval

EXCLUDED_NUMERIC = {
    "fit_seed",
    "recording_id",
    "requested_futures",
    "completed_futures",
    "nn_evaluations",
    "proxy_steps",
}


def compare(args: argparse.Namespace) -> dict:
    reference = pd.read_csv(args.reference)
    candidate = pd.read_csv(args.candidate)
    is_event = "event_id" in reference.columns
    is_probe = "probe_id" in reference.columns
    if is_probe != ("probe_id" in candidate.columns):
        raise ValueError("reference and candidate must belong to the same suite type")
    if is_probe:
        dose_column = "dose_mps2" if "dose_mps2" in reference else "controller_rate_rps"
        required = {"probe_id", "stimulus_family", dose_column}
        if not required.issubset(reference) or not required.issubset(candidate):
            raise ValueError(
                "probe comparison requires probe, stimulus family, and dose columns"
            )
        reference = reference.copy()
        candidate = candidate.copy()
        for frame in (reference, candidate):
            frame["probe_condition_id"] = (
                frame["probe_id"].astype(str)
                + ":"
                + frame["stimulus_family"].astype(str)
                + ":"
                + frame[dose_column].map(lambda value: f"{value:g}")
            )
        key = "probe_condition_id"
    else:
        key = "event_id" if is_event else "scenario_id"
    if is_event != ("event_id" in candidate.columns):
        raise ValueError("reference and candidate must belong to the same suite type")
    numeric = set(reference.select_dtypes("number")) & set(
        candidate.select_dtypes("number")
    )
    metrics = args.metrics or sorted(numeric - EXCLUDED_NUMERIC)
    effects = {}
    for metric in metrics:
        effect = clustered_paired_interval(
            reference,
            candidate,
            metric,
            key=key,
            event_macro=is_event,
            replicates=args.replicates,
            confidence=args.confidence,
            seed=args.seed,
        )
        reference_estimate = (
            float(reference.groupby("event_type")[metric].mean().mean())
            if is_event
            else float(reference[metric].mean())
        )
        effect["reference_estimate"] = reference_estimate
        effect["candidate_estimate"] = reference_estimate + effect["estimate"]
        effect["relative_change"] = (
            effect["estimate"] / reference_estimate if reference_estimate else None
        )
        effects[metric] = effect
    report = {
        "effect_definition": "candidate_minus_reference",
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "key": key,
        "event_macro": is_event,
        "probe_conditions": is_probe,
        "effects": effects,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", nargs="*")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    print(json.dumps(compare(args), indent=2))


if __name__ == "__main__":
    main()

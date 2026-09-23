"""Aggregate completed benchmark runs with recording-clustered intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from interactive_behavior_world_model.data.dataset import event_behavior_condition
from interactive_behavior_world_model.data.manifest import load_benchmark_config
from interactive_behavior_world_model.evaluation.statistics import (
    clustered_event_macro_interval,
    clustered_mean_interval,
)


def _run_id(path: Path, methods_root: Path) -> str:
    return "/".join(path.relative_to(methods_root).parts[:-1])


def _configuration_id(path: Path, methods_root: Path) -> str:
    parts = path.relative_to(methods_root).parts
    return f"{parts[0]}/{parts[3]}"


def aggregate(args: argparse.Namespace) -> dict:
    config, _ = load_benchmark_config(args.config)
    root = Path(config["paths"]["output_dir"])
    methods_root = root / "methods"
    allowed_seeds = {int(value) for value in config["sampling"]["fit_seeds"]}
    dataset_manifest = json.loads((root / "dataset_manifest.json").read_text())
    expected_scenes = int(
        dataset_manifest["split_summary"][
            "validation" if args.split == "validation" else "test"
        ]["strict_causal_rows"]
    )
    event_manifest = pd.read_csv(root / "event_manifest.csv")
    split_index = 1 if args.split == "validation" else 2
    expected_events = int(
        (
            (event_manifest.split_index == split_index)
            & (event_manifest.response_agent_index > 0)
        ).sum()
    )
    expected_t4a_events = int(
        sum(
            event_behavior_condition(event) is not None
            for _, event in event_manifest[
                event_manifest.split_index == split_index
            ].iterrows()
        )
    )
    probe_manifest = (
        pd.read_csv(root / "probe_manifest.csv")
        if (root / "probe_manifest.csv").exists()
        else pd.DataFrame()
    )
    expected_probe_conditions = (
        int(
            (
                probe_manifest.split_index == (1 if args.split == "validation" else 2)
            ).sum()
        )
        * 6
        if len(probe_manifest)
        else 0
    )
    lateral_probe_manifest = (
        pd.read_csv(root / "lateral_probe_manifest.csv")
        if (root / "lateral_probe_manifest.csv").exists()
        else pd.DataFrame()
    )
    expected_lateral_probe_conditions = (
        int(
            (
                lateral_probe_manifest.split_index
                == (1 if args.split == "validation" else 2)
            ).sum()
        )
        * 3
        if len(lateral_probe_manifest)
        else 0
    )
    rows: list[dict] = []
    partial_files = []
    files = sorted(methods_root.glob(f"*/[0-9]*/{args.split}/**/per_*_metrics.csv"))
    for path in files:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            partial_files.append(
                {
                    "path": str(path.relative_to(root)),
                    "rows": 0,
                    "expected_rows": "nonempty suite denominator",
                    "reason": "empty metrics file",
                }
            )
            continue
        if frame.empty or "fit_seed" not in frame:
            continue
        fit_seed = int(frame.fit_seed.iloc[0])
        if fit_seed != 0 and fit_seed not in allowed_seeds:
            continue
        is_probe = "probe_id" in frame
        suite = (
            str(frame.suite.iloc[0])
            if "suite" in frame
            else (
                "t2a_logged_events"
                if "event_type" in frame
                else "t2b_fixed_stimulus" if is_probe else "t1_natural"
            )
        )
        is_t4a = (
            path.name == "per_event_metrics.csv"
            and "t4a_controlled_" in path.parent.name
        )
        is_lateral_probe = (
            is_probe
            and "probe_manifest_version" in frame
            and str(frame.probe_manifest_version.iloc[0])
            == "npc_interaction_d3_lateral_v1"
        )
        expected_rows = (
            (
                expected_lateral_probe_conditions
                if is_lateral_probe
                else expected_probe_conditions
            )
            if is_probe
            else (
                expected_t4a_events
                if is_t4a
                else expected_events if "event_type" in frame else expected_scenes
            )
        )
        if len(frame) != expected_rows:
            partial_files.append(
                {
                    "path": str(path.relative_to(root)),
                    "rows": len(frame),
                    "expected_rows": expected_rows,
                }
            )
            continue
        metrics = [
            name
            for name in frame.select_dtypes("number").columns
            if name
            not in {
                "fit_seed",
                "recording_id",
                "requested_futures",
                "completed_futures",
                "requested_pairs",
                "completed_pairs",
                "nn_evaluations",
                "proxy_steps",
                "dose_mps2",
                "controller_rate_rps",
                "target_agent_index",
            }
            and frame[name].notna().any()
        ]
        for metric in metrics:
            interval = (
                clustered_event_macro_interval
                if "event_type" in frame
                else clustered_mean_interval
            )
            estimate = interval(
                frame,
                metric,
                replicates=args.replicates,
                confidence=float(config["metrics"]["confidence_level"]),
                seed=args.seed,
            )
            rows.append(
                {
                    "run_id": _run_id(path, methods_root),
                    "suite": suite,
                    "configuration_id": _configuration_id(path, methods_root),
                    "method_id": str(frame.method_id.iloc[0]),
                    "fit_seed": int(frame.fit_seed.iloc[0]),
                    "metric": metric,
                    "rows": len(frame),
                    **estimate,
                }
            )
    output = root / "aggregate"
    output.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output / f"{args.split}_suite_summary.csv", index=False)
    if len(result):
        seed_summary = result.groupby(
            ["configuration_id", "suite", "method_id", "metric"], as_index=False
        ).agg(
            fit_seeds=("fit_seed", "nunique"),
            seed_mean=("estimate", "mean"),
            seed_std=("estimate", "std"),
            mean_ci_low=("ci_low", "mean"),
            mean_ci_high=("ci_high", "mean"),
            rows_per_seed=("rows", "min"),
        )
        seed_summary.to_csv(output / f"{args.split}_seed_summary.csv", index=False)
    coverage = {
        "split": args.split,
        "result_files": len(files),
        "aggregated_runs": int(result.run_id.nunique()) if len(result) else 0,
        "rows": int(len(result)),
        "configured_fit_seeds": sorted(allowed_seeds),
        "bootstrap_replicates": args.replicates,
        "partial_or_smoke_files_excluded": partial_files,
    }
    (output / f"{args.split}_coverage_audit.json").write_text(
        json.dumps(coverage, indent=2) + "\n"
    )
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/benchmark_v1.yaml",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    print(json.dumps(aggregate(args), indent=2))


if __name__ == "__main__":
    main()

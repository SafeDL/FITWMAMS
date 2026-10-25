"""Fixed-event DQN reports by cohort, training seed and paired repair outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest

from ads.common.instances import artifact_record


def _percentile_interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, (0.025, 0.975))]


def bootstrap_summary(
    frame: pd.DataFrame,
    *,
    seed: int,
    replicates: int,
    cluster_column: str | None = None,
) -> dict[str, object]:
    if not len(frame):
        raise ValueError("evaluation frame is empty")
    rng = np.random.default_rng(seed)
    clusters = (
        frame[cluster_column].astype(str).to_numpy()
        if cluster_column is not None
        else np.arange(len(frame)).astype(str)
    )
    unique_clusters, cluster_index = np.unique(clusters, return_inverse=True)
    draws = rng.integers(
        0, len(unique_clusters), size=(replicates, len(unique_clusters))
    )

    def clustered_means(values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        sums = np.bincount(
            cluster_index[finite],
            weights=values[finite],
            minlength=len(unique_clusters),
        )
        counts = np.bincount(cluster_index[finite], minlength=len(unique_clusters))
        return sums[draws].sum(axis=1) / counts[draws].sum(axis=1).clip(min=1)

    failure = frame["failure"].astype(float).to_numpy()
    progress = frame["progress_m"].astype(float).to_numpy()
    reward = frame["reward"].astype(float).to_numpy()
    distance = frame["minimum_distance_m"].astype(float).to_numpy()
    finite_distance = distance[np.isfinite(distance)]
    boot_failure = clustered_means(failure)
    boot_progress = clustered_means(progress)
    boot_reward = clustered_means(reward)
    boot_distance = clustered_means(distance)
    return {
        "events": len(frame),
        "bootstrap_unit": cluster_column or "event_id",
        "bootstrap_cluster_count": len(unique_clusters),
        "failures": int(failure.sum()),
        "failure_rate": float(failure.mean()),
        "failure_rate_ci95": _percentile_interval(boot_failure),
        "mean_progress_m": float(progress.mean()),
        "mean_progress_m_ci95": _percentile_interval(boot_progress),
        "mean_reward": float(reward.mean()),
        "mean_reward_ci95": _percentile_interval(boot_reward),
        "mean_minimum_distance_m": float(finite_distance.mean()),
        "minimum_distance_event_count": len(finite_distance),
        "mean_minimum_distance_m_ci95": _percentile_interval(boot_distance),
        "collision_rate": float(frame["collision"].astype(float).mean()),
        "offroad_rate": float(frame["offroad"].astype(float).mean()),
        "stalled_rate": float(frame["stalled"].astype(float).mean()),
        "mean_abs_acceleration_mps2": float(frame["mean_abs_acceleration_mps2"].mean()),
        "mean_action_switches": float(frame["action_switches"].mean()),
    }


def _clopper_pearson(successes: int, total: int, alpha: float = 0.05):
    if total == 0:
        return None
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    )
    upper = (
        1.0
        if successes == total
        else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    )
    return [lower, upper]


def paired_repair_summary(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    bootstrap_seed: int = 20260926,
    bootstrap_replicates: int = 10_000,
    cluster_column: str | None = None,
) -> dict[str, Any]:
    required = {"event_id", "failure", "progress_m"}
    if required - set(before) or required - set(after):
        raise KeyError("paired results require event_id, failure and progress_m")
    merged = before.merge(
        after, on="event_id", suffixes=("_before", "_after"), validate="one_to_one"
    )
    if len(merged) != len(before) or len(merged) != len(after):
        raise ValueError("before/after event sets differ")
    old = merged["failure_before"].to_numpy(bool)
    new = merged["failure_after"].to_numpy(bool)
    n00 = int((~old & ~new).sum())
    n01 = int((~old & new).sum())
    n10 = int((old & ~new).sum())
    n11 = int((old & new).sum())
    repair_denominator = n10 + n11
    regression_denominator = n00 + n01
    progress_delta = (
        merged["progress_m_after"] - merged["progress_m_before"]
    ).to_numpy(np.float64)
    if cluster_column is None:
        if {"recording_id_before", "recording_id_after"}.issubset(merged.columns):
            cluster_column = "recording_id"
        elif {"source_event_id_before", "source_event_id_after"}.issubset(
            merged.columns
        ):
            cluster_column = "source_event_id"
    if cluster_column is None:
        clusters = merged["event_id"].astype(str).to_numpy()
        bootstrap_unit = "event_id"
    else:
        before_name = f"{cluster_column}_before"
        after_name = f"{cluster_column}_after"
        if before_name not in merged or after_name not in merged:
            raise KeyError(f"paired results require {cluster_column!r} in both inputs")
        if not merged[before_name].astype(str).equals(merged[after_name].astype(str)):
            raise ValueError(f"before/after {cluster_column} assignments differ")
        clusters = merged[before_name].astype(str).to_numpy()
        bootstrap_unit = cluster_column
    unique_clusters, cluster_index = np.unique(clusters, return_inverse=True)
    if not len(unique_clusters):
        raise ValueError("paired results are empty")
    rng = np.random.default_rng(int(bootstrap_seed))
    draws = rng.integers(
        0,
        len(unique_clusters),
        size=(int(bootstrap_replicates), len(unique_clusters)),
    )

    def grouped_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        numerator_sum = np.bincount(
            cluster_index, weights=numerator, minlength=len(unique_clusters)
        )
        denominator_sum = np.bincount(
            cluster_index, weights=denominator, minlength=len(unique_clusters)
        )
        sampled_numerator = numerator_sum[draws].sum(axis=1)
        sampled_denominator = denominator_sum[draws].sum(axis=1)
        return np.divide(
            sampled_numerator,
            sampled_denominator,
            out=np.full_like(sampled_numerator, np.nan, dtype=np.float64),
            where=sampled_denominator > 0,
        )

    def interval(values: np.ndarray) -> list[float] | None:
        finite = values[np.isfinite(values)]
        if not len(finite):
            return None
        return [float(value) for value in np.quantile(finite, (0.025, 0.975))]

    repair_bootstrap = grouped_ratio((old & ~new).astype(float), old.astype(float))
    regression_bootstrap = grouped_ratio(
        (~old & new).astype(float), (~old).astype(float)
    )
    net_bootstrap = grouped_ratio(
        old.astype(float) - new.astype(float), np.ones(len(merged), float)
    )
    progress_bootstrap = grouped_ratio(progress_delta, np.ones(len(merged), float))
    discordant = n01 + n10
    event_set_sha256 = hashlib.sha256(
        "\n".join(sorted(merged["event_id"].astype(str))).encode("utf-8")
    ).hexdigest()
    return {
        "events": len(merged),
        "event_set_sha256": event_set_sha256,
        "n00_success_to_success": n00,
        "n01_success_to_failure": n01,
        "n10_failure_to_success": n10,
        "n11_failure_to_failure": n11,
        "repair_rate": None if repair_denominator == 0 else n10 / repair_denominator,
        "repair_rate_ci95": interval(repair_bootstrap),
        "repair_rate_binomial_ci95_unclustered": _clopper_pearson(
            n10, repair_denominator
        ),
        "regression_rate": None
        if regression_denominator == 0
        else n01 / regression_denominator,
        "regression_rate_ci95": interval(regression_bootstrap),
        "regression_rate_binomial_ci95_unclustered": _clopper_pearson(
            n01, regression_denominator
        ),
        "net_gain": None if not len(merged) else (n10 - n01) / len(merged),
        "net_gain_ci95": interval(net_bootstrap),
        "mcnemar_exact_p": (
            None
            if discordant == 0
            else float(binomtest(min(n01, n10), discordant, 0.5).pvalue)
        ),
        "mean_progress_before_m": float(merged["progress_m_before"].mean()),
        "mean_progress_after_m": float(merged["progress_m_after"].mean()),
        "mean_progress_delta_m": float(progress_delta.mean()),
        "mean_progress_delta_m_ci95": interval(progress_bootstrap),
        "bootstrap_unit": bootstrap_unit,
        "bootstrap_cluster_count": len(unique_clusters),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_replicates": int(bootstrap_replicates),
    }


def report_cohorts(args) -> None:
    frames: dict[str, pd.DataFrame] = {}
    manifests = {label: path for label, path in args.manifest}
    summaries = {}
    for index, (label, path) in enumerate(args.evaluation):
        if label in frames:
            raise ValueError(f"duplicate evaluation label {label!r}")
        frame = pd.read_csv(path)
        if frame["event_id"].duplicated().any():
            raise ValueError(f"{label} contains duplicate event IDs")
        cluster_column = None
        if label in manifests:
            manifest = pd.read_csv(manifests[label])
            frame = frame.merge(
                manifest[["event_id", "recording_id", "event_risk"]],
                on="event_id",
                how="left",
                validate="one_to_one",
            )
            if frame["recording_id"].isna().any():
                raise ValueError(f"{label} manifest does not cover every event")
            cluster_column = "recording_id"
        frames[label] = frame
        summaries[label] = bootstrap_summary(
            frame,
            seed=args.seed + index,
            replicates=args.bootstrap_replicates,
            cluster_column=cluster_column,
        )
    labels = list(frames)
    overlaps = {}
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            overlaps[f"{left}:{right}"] = len(
                set(frames[left]["source_event_id"].astype(str))
                & set(frames[right]["source_event_id"].astype(str))
            )
    result = {
        "bootstrap_seed": args.seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_resampling": ("recording_cluster" if manifests else "event"),
        "summaries": summaries,
        "source_event_overlap": overlaps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def report_seeds(args) -> None:
    rows = []
    event_sets: dict[str, set[str]] = {}
    for algorithm, seed, path in args.evaluation:
        frame = pd.read_csv(path)
        required = {"event_id", "failure", "progress_m", "reward"}
        if missing := required - set(frame):
            raise KeyError(f"{path} lacks {sorted(missing)}")
        current = set(frame["event_id"].astype(str))
        event_sets.setdefault(algorithm, current)
        if event_sets[algorithm] != current:
            raise ValueError(f"{algorithm} seed evaluations use different event sets")
        rows.append(
            {
                "algorithm": algorithm,
                "seed": int(seed),
                "events": len(frame),
                "failures": int(frame["failure"].sum()),
                "failure_rate": float(frame["failure"].mean()),
                "mean_progress_m": float(frame["progress_m"].mean()),
                "mean_reward": float(frame["reward"].mean()),
                "mean_abs_acceleration_mps2": float(
                    frame["mean_abs_acceleration_mps2"].mean()
                ),
                "mean_action_switches": float(frame["action_switches"].mean()),
            }
        )
    runs = pd.DataFrame(rows).sort_values(["algorithm", "seed"])
    reference_event_set = next(iter(event_sets.values()))
    if any(values != reference_event_set for values in event_sets.values()):
        raise ValueError("algorithms use different fixed evaluation events")
    aggregate = {}
    for algorithm, group in runs.groupby("algorithm", sort=True):
        metrics = {}
        for column in (
            "failure_rate",
            "mean_progress_m",
            "mean_reward",
            "mean_abs_acceleration_mps2",
            "mean_action_switches",
        ):
            metrics[column] = {
                "mean": float(group[column].mean()),
                "std_across_seeds": (
                    float(group[column].std(ddof=1)) if len(group) > 1 else None
                ),
                "raw": [float(value) for value in group[column]],
            }
        aggregate[algorithm] = {
            "seed_count": len(group),
            "events_per_seed": int(group["events"].iloc[0]),
            "metrics": metrics,
        }
    result = {
        "algorithms": aggregate,
        "event_set_overlap": {
            f"{left}:{right}": len(event_sets[left] & event_sets[right])
            for left in event_sets
            for right in event_sets
            if left < right
        },
        "inference_unit": "training_seed; the same fixed events are paired within each seed",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "seed_runs.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


def report_paired(args) -> None:
    before = pd.read_csv(args.before)
    after = pd.read_csv(args.after)
    cluster_column = None
    if args.manifest is not None:
        manifest = pd.read_csv(args.manifest)
        columns = ["event_id"]
        if "recording_id" in manifest:
            columns.append("recording_id")
            cluster_column = "recording_id"
        elif "source_event_id" in manifest:
            columns.append("source_event_id")
            cluster_column = "source_event_id"
        else:
            raise KeyError("manifest requires recording_id or source_event_id")
        before = before.merge(manifest[columns], on="event_id", validate="one_to_one")
        after = after.merge(manifest[columns], on="event_id", validate="one_to_one")
    summary = paired_repair_summary(before, after, cluster_column=cluster_column)
    summary["input_artifacts"] = {
        "before": artifact_record(args.before),
        "after": artifact_record(args.after),
        "manifest": artifact_record(args.manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="report", required=True)
    cohorts = commands.add_parser("cohorts", help="summarise fixed-event cohorts")
    cohorts.add_argument(
        "--evaluation",
        action="append",
        nargs=2,
        metavar=("LABEL", "CSV"),
        required=True,
    )
    cohorts.add_argument(
        "--manifest", action="append", nargs=2, metavar=("LABEL", "CSV"), default=[]
    )
    cohorts.add_argument("--output", type=Path, required=True)
    cohorts.add_argument("--seed", type=int, default=20260926)
    cohorts.add_argument("--bootstrap-replicates", type=int, default=5000)
    cohorts.set_defaults(handler=report_cohorts)
    seeds = commands.add_parser(
        "seeds", help="compare training seeds on identical events"
    )
    seeds.add_argument(
        "--evaluation",
        nargs=3,
        action="append",
        metavar=("ALGORITHM", "SEED", "CSV"),
        required=True,
    )
    seeds.add_argument("--output-dir", type=Path, required=True)
    seeds.set_defaults(handler=report_seeds)
    paired = commands.add_parser("paired", help="compare repair and regression")
    paired.add_argument("--before", type=Path, required=True)
    paired.add_argument("--after", type=Path, required=True)
    paired.add_argument("--manifest", type=Path)
    paired.add_argument("--output", type=Path, required=True)
    paired.set_defaults(handler=report_paired)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

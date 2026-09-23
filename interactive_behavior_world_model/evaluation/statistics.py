"""Recording-clustered uncertainty estimates for benchmark results."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def clustered_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    cluster: str = "recording_id",
    replicates: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260919,
) -> dict[str, float]:
    """Estimate a statistic and percentile CI by resampling recordings."""
    if frame.empty or cluster not in frame:
        raise ValueError("bootstrap requires non-empty rows and a cluster column")
    clusters = np.asarray(pd.unique(frame[cluster]))
    grouped = {value: group for value, group in frame.groupby(cluster, sort=False)}
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, np.float64)
    for index in range(replicates):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([grouped[value] for value in selected], ignore_index=True)
        values[index] = statistic(sample)
    alpha = (1.0 - confidence) * 0.5
    return {
        "estimate": float(statistic(frame)),
        "ci_low": float(np.quantile(values, alpha, method="linear")),
        "ci_high": float(np.quantile(values, 1.0 - alpha, method="linear")),
        "bootstrap_replicates": int(replicates),
        "recording_clusters": int(len(clusters)),
    }


def event_macro_mean(frame: pd.DataFrame, metric: str) -> float:
    """Equal-weight event-family mean required by the T2a protocol."""
    if "event_type" not in frame or metric not in frame:
        raise ValueError("event macro mean requires event_type and metric")
    return float(frame.groupby("event_type", sort=False)[metric].mean().mean())


def clustered_mean_interval(
    frame: pd.DataFrame,
    metric: str,
    *,
    cluster: str = "recording_id",
    replicates: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260919,
) -> dict[str, float]:
    """Fast clustered interval for an ordinary row mean."""
    grouped = frame.groupby(cluster, sort=False)[metric].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    choice = rng.integers(0, len(grouped), size=(replicates, len(grouped)))
    sums = grouped["sum"].to_numpy()[choice].sum(1)
    counts = grouped["count"].to_numpy()[choice].sum(1)
    values = sums / np.maximum(counts, 1)
    alpha = (1.0 - confidence) * 0.5
    return {
        "estimate": float(frame[metric].mean()),
        "ci_low": float(np.quantile(values, alpha)),
        "ci_high": float(np.quantile(values, 1 - alpha)),
        "bootstrap_replicates": replicates,
        "recording_clusters": len(grouped),
    }


def clustered_event_macro_interval(
    frame: pd.DataFrame,
    metric: str,
    *,
    cluster: str = "recording_id",
    replicates: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260919,
) -> dict[str, float]:
    """Fast clustered interval retaining equal event-family weights."""
    clusters = pd.Index(pd.unique(frame[cluster]))
    families = pd.Index(pd.unique(frame["event_type"]))
    sums = np.zeros((len(clusters), len(families)))
    counts = np.zeros_like(sums)
    c_lookup = {value: index for index, value in enumerate(clusters)}
    f_lookup = {value: index for index, value in enumerate(families)}
    grouped = frame.groupby([cluster, "event_type"], sort=False)[metric].agg(
        ["sum", "count"]
    )
    for (c_value, f_value), row in grouped.iterrows():
        c_index, f_index = c_lookup[c_value], f_lookup[f_value]
        sums[c_index, f_index] = row["sum"]
        counts[c_index, f_index] = row["count"]
    rng = np.random.default_rng(seed)
    choice = rng.integers(0, len(clusters), size=(replicates, len(clusters)))
    selected_sums = sums[choice].sum(1)
    selected_counts = counts[choice].sum(1)
    family_means = np.divide(
        selected_sums,
        selected_counts,
        out=np.full_like(selected_sums, np.nan),
        where=selected_counts > 0,
    )
    values = np.nanmean(family_means, axis=1)
    alpha = (1.0 - confidence) * 0.5
    return {
        "estimate": event_macro_mean(frame, metric),
        "ci_low": float(np.quantile(values, alpha)),
        "ci_high": float(np.quantile(values, 1 - alpha)),
        "bootstrap_replicates": replicates,
        "recording_clusters": len(clusters),
    }


def paired_metric_frame(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    metric: str,
    *,
    key: str = "scenario_id",
    cluster: str = "recording_id",
) -> pd.DataFrame:
    """Align two complete runs and return candidate-minus-reference rows."""
    required = {key, cluster, metric}
    if not required.issubset(reference) or not required.issubset(candidate):
        raise ValueError(f"paired comparison requires columns {sorted(required)}")
    if reference[key].duplicated().any() or candidate[key].duplicated().any():
        raise ValueError("paired comparison keys must be unique within each run")
    merged = reference[[key, cluster, metric]].merge(
        candidate[[key, cluster, metric]],
        on=key,
        how="inner",
        suffixes=("_reference", "_candidate"),
        validate="one_to_one",
    )
    if len(merged) != len(reference) or len(merged) != len(candidate):
        raise ValueError("paired comparison requires identical complete key sets")
    if not np.array_equal(
        merged[f"{cluster}_reference"].to_numpy(),
        merged[f"{cluster}_candidate"].to_numpy(),
    ):
        raise ValueError("paired rows disagree on recording cluster")
    result = pd.DataFrame(
        {
            key: merged[key],
            cluster: merged[f"{cluster}_reference"],
            "paired_delta": merged[f"{metric}_candidate"]
            - merged[f"{metric}_reference"],
        }
    )
    if "event_type" in reference and "event_type" in candidate:
        event_types = reference[[key, "event_type"]].merge(
            candidate[[key, "event_type"]],
            on=key,
            suffixes=("_reference", "_candidate"),
            validate="one_to_one",
        )
        if not np.array_equal(
            event_types.event_type_reference.to_numpy(),
            event_types.event_type_candidate.to_numpy(),
        ):
            raise ValueError("paired rows disagree on event family")
        result["event_type"] = event_types.event_type_reference.to_numpy()
    return result


def clustered_paired_interval(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    metric: str,
    *,
    key: str = "scenario_id",
    cluster: str = "recording_id",
    event_macro: bool = False,
    replicates: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260919,
) -> dict[str, float]:
    """Clustered CI for candidate-minus-reference on matched scenes/events."""
    paired = paired_metric_frame(reference, candidate, metric, key=key, cluster=cluster)
    interval = (
        clustered_event_macro_interval if event_macro else clustered_mean_interval
    )
    result = interval(
        paired,
        "paired_delta",
        cluster=cluster,
        replicates=replicates,
        confidence=confidence,
        seed=seed,
    )
    result["paired_rows"] = int(len(paired))
    return result

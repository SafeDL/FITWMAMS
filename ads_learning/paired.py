"""Paired repair/regression metrics on identical events and randomness."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest


def _clopper_pearson(successes: int, total: int, alpha: float = 0.05):
    if total == 0:
        return None
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
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
    merged = before.merge(after, on="event_id", suffixes=("_before", "_after"), validate="one_to_one")
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
        if {
            "recording_id_before", "recording_id_after"
        }.issubset(merged.columns):
            cluster_column = "recording_id"
        elif {
            "source_event_id_before", "source_event_id_after"
        }.issubset(merged.columns):
            cluster_column = "source_event_id"
    if cluster_column is None:
        clusters = merged["event_id"].astype(str).to_numpy()
        bootstrap_unit = "event_id"
    else:
        before_name = f"{cluster_column}_before"
        after_name = f"{cluster_column}_after"
        if before_name not in merged or after_name not in merged:
            raise KeyError(
                f"paired results require {cluster_column!r} in both inputs"
            )
        if not merged[before_name].astype(str).equals(
            merged[after_name].astype(str)
        ):
            raise ValueError(
                f"before/after {cluster_column} assignments differ"
            )
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
    regression_bootstrap = grouped_ratio((~old & new).astype(float), (~old).astype(float))
    net_bootstrap = grouped_ratio(
        old.astype(float) - new.astype(float), np.ones(len(merged), float)
    )
    progress_bootstrap = grouped_ratio(
        progress_delta, np.ones(len(merged), float)
    )
    discordant = n01 + n10
    event_set_sha256 = hashlib.sha256(
        "\n".join(sorted(merged["event_id"].astype(str))).encode("utf-8")
    ).hexdigest()
    return {
        "events": int(len(merged)),
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
        "regression_rate": None if regression_denominator == 0 else n01 / regression_denominator,
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
        "bootstrap_cluster_count": int(len(unique_clusters)),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_replicates": int(bootstrap_replicates),
    }

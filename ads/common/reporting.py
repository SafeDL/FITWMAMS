"""Paired collision statistics; independent units follow the cohort design."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def paired_rows(before, after):
    old, new = pd.DataFrame(before), pd.DataFrame(after)
    required = {
        "event_id",
        "instance_hash",
        "assets_hash",
        "root_sample_id",
        "policy_stream_id",
        "collision",
        "offroad",
        "stalled",
        "safety_failure",
        "progress_m",
    }
    if required - set(old) or required - set(new):
        raise ValueError(
            "paired evaluation requires complete instance, asset and policy random stream hashes"
        )
    joined = old.merge(
        new, on="event_id", suffixes=("_old", "_new"), validate="one_to_one"
    )
    if len(joined) != len(old) or len(old) != len(new):
        raise ValueError("paired cohorts differ")
    for key in ("instance_hash", "assets_hash", "root_sample_id", "policy_stream_id"):
        if not joined[key + "_old"].equals(joined[key + "_new"]):
            raise ValueError(f"paired {key} differs")
    return old.set_index("event_id").loc[joined.event_id].reset_index(), new.set_index(
        "event_id"
    ).loc[joined.event_id].reset_index()


def cluster_difference_ci(values, clusters, *, replicates=10000, seed=20260926):
    codes, names = pd.factorize(clusters, sort=True)
    totals = np.bincount(codes, weights=np.asarray(values, float))
    sizes = np.bincount(codes)
    rng, boot = np.random.default_rng(seed), np.empty(replicates)
    for start in range(0, replicates, 128):
        n = min(128, replicates - start)
        sampled = rng.integers(0, len(names), (n, len(names)))
        boot[start : start + n] = totals[sampled].sum(-1) / sizes[sampled].sum(-1)
    return np.quantile(boot, (0.025, 0.975)).tolist(), len(names)


def paired_binary_difference_ci(delta):
    """Project simultaneous discordance bounds; coverage is at least 95% for IID pairs.

    The difference is P(after-only failure) - P(before-only failure).
    Each discordance count is marginally binomial. Two 97.5% Clopper-Pearson
    intervals cover jointly with probability at least 95% by Bonferroni,
    without assuming independence of the two outcomes within an instance.
    """
    delta = np.asarray(delta)
    intervals = [
        binomtest(int((delta == sign).sum()), len(delta)).proportion_ci(
            confidence_level=0.975, method="exact"
        )
        for sign in (1, -1)
    ]
    positive, negative = intervals
    return [float(positive.low - negative.high), float(positive.high - negative.low)]


def paired_summary(before, after, *, diagnostic=False, bootstrap_replicates=10000):
    old, new = paired_rows(before, after)
    source = str(old.source_kind.iloc[0]) if "source_kind" in old else "tail_generated"
    cluster = (
        "root_sample_id"
        if diagnostic
        else "recording_id"
        if source == "empirical_natural"
        else "instance_hash"
    )
    if cluster == "recording_id" and (
        old[cluster].isna().any() or not old[cluster].equals(new[cluster])
    ):
        raise ValueError("paired empirical recordings differ")
    delta = new.collision.to_numpy(float) - old.collision.to_numpy(float)
    bootstrap, n = cluster_difference_ci(
        delta, old[cluster], replicates=bootstrap_replicates
    )
    iid = cluster == "instance_hash"
    interval = paired_binary_difference_ci(delta) if iid else bootstrap
    result = {
        "cohort_kind": "search_archive_diagnostic" if diagnostic else source,
        "instances": len(old),
        "independent_units": n,
        "bootstrap_unit": cluster,
        "collision_rate_difference_after_minus_before": float(delta.mean()),
        "collision_rate_difference_ci95": interval,
        "collision_rate_difference_bootstrap_ci95": bootstrap,
        "collision_ci_method": (
            "paired_discordance_clopper_pearson_bonferroni"
            if iid
            else "cluster_percentile_bootstrap"
        ),
        "collision_ci_is_degenerate": interval[0] == interval[1],
        "collision_discordant_instances": int((delta != 0).sum()),
        "repaired": int((old.safety_failure & ~new.safety_failure).sum()),
        "regressed": int((~old.safety_failure & new.safety_failure).sum()),
        "mean_progress_before_m": float(old.progress_m.mean()),
        "mean_progress_after_m": float(new.progress_m.mean()),
    }
    for name in ("collision", "offroad", "stalled", "safety_failure"):
        result[name + "_rate_before"] = float(old[name].mean())
        result[name + "_rate_after"] = float(new[name].mean())
    # Secondary safety-envelope outcome remains separately identified.
    difference = new.safety_failure.to_numpy(float) - old.safety_failure.to_numpy(float)
    result["failure_rate_difference_after_minus_before"] = float(difference.mean())
    failure_bootstrap = cluster_difference_ci(
        difference, old[cluster], replicates=bootstrap_replicates
    )[0]
    result["failure_rate_difference_ci95"] = (
        paired_binary_difference_ci(difference) if iid else failure_bootstrap
    )
    return result


def shared_world_seed_summary(pairs, *, bootstrap_replicates=10000, seed=20260926):
    aligned = []
    expected = None
    for before, after in pairs:
        old, new = paired_rows(before, after)
        order = np.argsort(old.instance_hash.to_numpy())
        old, new = old.iloc[order], new.iloc[order]
        identities = old.instance_hash.tolist()
        if expected is not None and expected != identities:
            raise ValueError("policies must share the frozen cohort")
        expected = identities
        aligned.append(new.collision.to_numpy(float) - old.collision.to_numpy(float))
    values = np.stack(aligned)
    source = str(old.source_kind.iloc[0]) if "source_kind" in old else "tail_generated"
    cluster = "recording_id" if source == "empirical_natural" else "instance_hash"
    bootstrap, units = cluster_difference_ci(
        values.mean(0), old[cluster], replicates=bootstrap_replicates, seed=seed
    )
    iid = cluster == "instance_hash" and len(values) == 1
    interval = paired_binary_difference_ci(values[0]) if iid else bootstrap
    return {
        "training_seeds": len(values),
        "instances": values.shape[1],
        "independent_units": units,
        "mean_collision_difference": float(values.mean()),
        "per_seed_collision_differences": values.mean(1).tolist(),
        "paired_collision_difference_ci95": interval,
        "paired_collision_difference_bootstrap_ci95": bootstrap,
        "collision_ci_method": (
            "paired_discordance_clopper_pearson_bonferroni"
            if iid
            else "cluster_percentile_bootstrap"
        ),
        "collision_ci_is_degenerate": interval[0] == interval[1],
        "collision_discordant_instances": int(np.any(values != 0, axis=0).sum()),
        "bootstrap_unit": cluster,
        "pooled_as_independent_seed_times_world": False,
    }


def repair_method_comparisons(
    rows, algorithms, seeds, methods, *, bootstrap_replicates
):
    result = {"per_seed": {}, "shared_world_aggregate": {}}
    if "guided" not in methods:
        return result
    for algorithm in algorithms:
        for reference in (method for method in methods if method != "guided"):
            pairs = []
            for seed in seeds:
                before, after = (
                    rows[f"{algorithm}/{seed}/{reference}"],
                    rows[f"{algorithm}/{seed}/guided"],
                )
                key = f"{algorithm}/{seed}/guided_vs_{reference}"
                result["per_seed"][key] = paired_summary(
                    before, after, bootstrap_replicates=bootstrap_replicates
                )
                pairs.append((before, after))
            result["shared_world_aggregate"][f"{algorithm}/guided_vs_{reference}"] = (
                shared_world_seed_summary(
                    pairs, bootstrap_replicates=bootstrap_replicates
                )
            )
    return result

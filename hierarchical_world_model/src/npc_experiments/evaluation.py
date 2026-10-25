"""Frozen natural-behavior, road-execution and complete Test response reports."""

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.utils import (
    load_json,
    save_json,
)
from ..data import artifact_identity
from ..stochastic_drivers.raw_data import (
    DRIVER_DATA_DIR,
    SPLITS,
    load_lane_opportunities,
)
from ..stochastic_drivers.response_data import load_response
from .calibration import (
    SEED,
    conditional_metrics,
    conditional_predictions,
    group_histograms,
    sampled_lane_probabilities,
)
from .metrics import (
    FEATURE_EDGES,
    hellinger,
)
from .search import (
    closed_summary,
    load_cohorts,
    run_cohorts,
)

SUBGROUPS = (
    "car",
    "truck",
    "direction_1",
    "direction_2",
    "density_low",
    "density_middle",
    "density_high",
)
ROLLOUT_DIRECTORIES = {"short": "short_horizon", "long": "continuous_runs"}


def stage_models(output):
    output = Path(output)
    selected = load_json(output / "policy_calibration/selected_model.json")
    with np.load(selected["driver_parameter_source"]) as saved:
        population = dict(saved)
    return "reference", {
        "reference": (selected["selected_candidate"], population, selected["mobil"])
    }


def combine_splits(data):
    return {
        name: np.concatenate([data[split][name] for split in SPLITS])
        for name in data["train"]
    }


def convergence_subset(data, fraction, *, lane=False):
    """Select fixed, identity-ordered samples within physical recording/behavior strata."""
    if lane:
        group = data["label"]
    else:
        closing = data["speed"] - data["leader_speed"]
        group = np.where(
            data["free"], 3, np.where(closing < -1, 0, np.where(closing > 1, 2, 1))
        )
    order = np.lexsort((data["frame"], data["vehicle"], group, data["recording"]))
    keys = np.column_stack((data["recording"][order], group[order]))
    starts = np.r_[0, np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1]
    stops = np.r_[starts[1:], len(order)]
    selected = np.concatenate(
        [
            order[
                start
                + np.floor(
                    np.arange(max(1, round((stop - start) * fraction))) / fraction
                )
                .astype(int)
                .clip(max=stop - start - 1)
            ]
            for start, stop in zip(starts, stops)
        ]
    )
    return {name: value[selected] for name, value in data.items()}


def _save_statistics(path, statistics):
    temporary = path.with_name(path.stem + ".pending.npz")
    np.savez_compressed(temporary, **statistics)
    temporary.replace(path)


def _conditional_rows(split, stage, result):
    rows = []
    for value in result["acceleration"]:
        rows.append(
            {
                "split": split,
                "model": stage,
                "group": value["group"],
                "metric": "acceleration_Hellinger",
                "observations": value["observations"],
                "value": value["Hellinger"],
                "CI_low": value["CI95"][0],
                "CI_high": value["CI95"][1],
            }
        )
        rows.append(
            {
                "split": split,
                "model": stage,
                "group": value["group"],
                "metric": "mean_acceleration_RMSE_mps2",
                "observations": value["observations"],
                "value": value["mean_RMSE_mps2"],
            }
        )
    value = result["lane"]
    rows.append(
        {
            "split": split,
            "model": stage,
            "group": "lane_choice",
            "metric": "Brier",
            "observations": value["opportunities"],
            "value": value["Brier"],
            "CI_low": value["Brier_CI95"][0],
            "CI_high": value["Brier_CI95"][1],
        }
    )
    for index, name in enumerate(("rear", "front")):
        rows.append(
            {
                "split": split,
                "model": stage,
                "group": "lane_start_opportunity",
                "metric": f"accepted_{name}_gap_Hellinger",
                "observations": int(sum(value["observed_choice_frequency"][1:])),
                "value": value["acceptance_Hellinger"][index],
                "standard_error": value["acceptance_SE"][index],
                "CI_low": value["acceptance_CI95"][index][0],
                "CI_high": value["acceptance_CI95"][index][1],
            }
        )
    return rows


def _save_conditional_table(output, summary):
    tables = output / "tables"
    tables.mkdir(exist_ok=True)
    rows = [
        row
        for split, stages in summary["splits"].items()
        for stage, result in stages.items()
        for row in _conditional_rows(split, stage, result)
    ]
    for split, stages in summary["splits"].items():
        for stage, result in stages.items():
            for group, values in result.get("subgroups", {}).items():
                for metric, observations in (
                    ("acceleration_Hellinger", "following_states"),
                    ("mean_acceleration_RMSE_mps2", "following_states"),
                    ("Brier", "lane_opportunities"),
                ):
                    interval = values.get(metric + "_CI95")
                    rows.append(
                        {
                            "split": split,
                            "model": stage,
                            "group": group,
                            "metric": metric,
                            "observations": values[observations],
                            "value": values[metric],
                            "CI_low": None if interval is None else interval[0],
                            "CI_high": None if interval is None else interval[1],
                        }
                    )
    pd.DataFrame(rows).assign(
        phase="evaluation",
        fit_split="train",
    ).to_csv(tables / "conditional_metrics.csv", index=False)


def subgroup_masks(data, density_edges):
    density = np.searchsorted(density_edges, data["density"], side="right")
    return {
        "car": data["class"] == 0,
        "truck": data["class"] == 1,
        "direction_1": data["direction"] == 1,
        "direction_2": data["direction"] == 2,
        **{name: density == index for index, name in enumerate(SUBGROUPS[4:])},
    }


def subgroup_report(numeric, bootstrap):
    real, model = numeric["real"], numeric["model"]
    count, squared = real.sum(1), numeric["squared_error"]
    lane_count, lane_error = numeric["lane_count"], numeric["lane_error"]
    draws = np.random.default_rng(SEED).integers(
        len(count), size=(bootstrap, len(count))
    )
    observed, simulated = real[draws].sum(1), model[draws].sum(1)
    supported = observed.sum(1) > 0
    lane_total = lane_count[draws].sum(1)
    lane_supported = lane_total > 0
    return {
        "following_states": int(count.sum()),
        "lane_opportunities": int(lane_count.sum()),
        "acceleration_Hellinger": (
            float(hellinger(real.sum(0), model.sum(0))) if count.sum() else None
        ),
        "acceleration_Hellinger_CI95": (
            np.quantile(
                hellinger(observed[supported], simulated[supported]), [0.025, 0.975]
            ).tolist()
            if supported.any()
            else None
        ),
        "mean_acceleration_RMSE_mps2": (
            float(np.sqrt(squared.sum() / count.sum())) if count.sum() else None
        ),
        "Brier": (
            float(lane_error.sum() / lane_count.sum()) if lane_count.sum() else None
        ),
        "Brier_CI95": (
            np.quantile(
                lane_error[draws].sum(1)[lane_supported] / lane_total[lane_supported],
                [0.025, 0.975],
            ).tolist()
            if lane_supported.any()
            else None
        ),
        "observed_choice_probability": (
            (numeric["lane_observed"].sum(0) / lane_count.sum()).tolist()
            if lane_count.sum()
            else None
        ),
        "predicted_choice_probability": (
            (numeric["lane_predicted"].sum(0) / lane_count.sum()).tolist()
            if lane_count.sum()
            else None
        ),
    }


def conditional_subgroups(output, settings):
    """Report class/direction/Train-density strata without changing model selection."""
    output = Path(output)
    directory = output / "conditional_behavior"
    summary = load_json(directory / "summary.json")
    if all(
        "subgroups" in result
        for stages in summary["splits"].values()
        for result in stages.values()
    ):
        _save_conditional_table(output, summary)
        return summary
    following, lanes = load_response(), load_lane_opportunities(DRIVER_DATA_DIR)
    edges = np.quantile(following["train"]["density"], [1 / 3, 2 / 3])
    summary["subgroup_protocol"] = {
        "groups": list(SUBGROUPS),
        "density_edges": edges.tolist(),
        "density_source": "Train following-state terciles; frozen for all splits",
        "density_unit": "observed vehicles per km per lane",
        "samples": settings["conditional_samples"],
        "bootstrap_samples": settings["bootstrap_samples"],
        "empty_bootstrap_groups": "no-exposure resamples omitted; zero-data groups remain null",
    }
    following["all"], lanes["all"] = combine_splits(following), combine_splits(lanes)
    _, models = stage_models(output)
    with np.load(directory / "statistics.npz") as saved:
        statistics = {name: saved[name] for name in saved.files}
    for split in (*SPLITS, "all"):
        data, lane = following[split], lanes[split]
        recordings = np.unique(data["recording"])
        lane_index = np.searchsorted(recordings, lane["recording"])
        state_index = np.searchsorted(recordings, data["recording"])
        masks, lane_masks = subgroup_masks(data, edges), subgroup_masks(lane, edges)
        for stage, (_, population, mobil) in models.items():
            result = summary["splits"][split][stage]
            if "subgroups" in result:
                continue
            predicted = conditional_predictions(
                data, population, mobil, settings["conditional_samples"]
            )
            probabilities = sampled_lane_probabilities(
                lane, population, mobil, settings["conditional_samples"]
            )
            squared = np.square(predicted.mean(0) - data["acceleration"])
            lane_error = np.square(probabilities - np.eye(3)[lane["label"]]).sum(1)
            result["subgroups"] = {}
            for group in SUBGROUPS:
                keep, lane_keep = masks[group], lane_masks[group]
                numeric = {
                    "real": group_histograms(
                        data["acceleration"],
                        data["recording"],
                        keep,
                        FEATURE_EDGES["acceleration"],
                        recordings,
                    ),
                    "model": group_histograms(
                        predicted,
                        data["recording"],
                        keep,
                        FEATURE_EDGES["acceleration"],
                        recordings,
                    ),
                    "squared_error": np.bincount(
                        state_index[keep],
                        weights=squared[keep],
                        minlength=len(recordings),
                    ),
                    "lane_count": np.bincount(
                        lane_index[lane_keep], minlength=len(recordings)
                    ),
                    "lane_error": np.bincount(
                        lane_index[lane_keep],
                        weights=lane_error[lane_keep],
                        minlength=len(recordings),
                    ),
                    "lane_observed": np.stack(
                        [
                            np.bincount(
                                lane_index[lane_keep],
                                weights=(lane["label"][lane_keep] == choice),
                                minlength=len(recordings),
                            )
                            for choice in range(3)
                        ],
                        axis=1,
                    ),
                    "lane_predicted": np.stack(
                        [
                            np.bincount(
                                lane_index[lane_keep],
                                weights=probabilities[lane_keep, choice],
                                minlength=len(recordings),
                            )
                            for choice in range(3)
                        ],
                        axis=1,
                    ),
                }
                result["subgroups"][group] = subgroup_report(
                    numeric, settings["bootstrap_samples"]
                )
                statistics.update(
                    {
                        f"{split}__{stage}__subgroup__{group}__{name}": value
                        for name, value in numeric.items()
                    }
                )
            _save_statistics(directory / "statistics.npz", statistics)
            save_json(summary, directory / "summary.json")
            print(
                f"Conditional subgroups {split}/{stage}: class, direction and Train-density strata",
                flush=True,
            )
    _save_conditional_table(output, summary)
    return summary


def conditional_evaluation(output, settings):
    output = Path(output)
    directory = output / "conditional_behavior"
    directory.mkdir(parents=True, exist_ok=True)
    selected_stage, models = stage_models(output)
    data_manifest = load_json(DRIVER_DATA_DIR / "manifest.json")
    recordings = sorted(
        recording
        for group in data_manifest["split_recordings"].values()
        for recording in group
    )
    protocol = {
        "samples": settings["conditional_samples"],
        "convergence_samples": settings["convergence_samples"],
        "convergence_fraction": settings["convergence_fraction"],
        "bootstrap_samples": settings["bootstrap_samples"],
        "paired_acceleration_statistics": "recording squared-error sums",
        "candidates": {stage: name for stage, (name, _, _) in models.items()},
        "parameters": {
            stage: {
                "population": {
                    name: np.asarray(value).tolist()
                    for name, value in population.items()
                },
                "mobil": mobil,
            }
            for stage, (_, population, mobil) in models.items()
        },
        "selected_stage": selected_stage,
        "phase": "evaluation",
        "fit_split": "train",
        "data": data_manifest,
        "data_files": [
            artifact_identity(DRIVER_DATA_DIR / f"recordings/{recording:02d}.npz")
            for recording in recordings
        ],
    }
    protocol_path = directory / "protocol.json"
    if protocol_path.exists() and load_json(protocol_path) != protocol:
        raise ValueError(
            "Conditional data or frozen model changed; regenerate conditional results"
        )
    save_json(protocol, protocol_path)
    summary_path = directory / "summary.json"
    statistics_path = directory / "statistics.npz"
    summary = (
        load_json(summary_path)
        if summary_path.exists()
        else {"selected_stage": selected_stage, "splits": {}, "convergence": {}}
    )
    statistics = {}
    if statistics_path.exists():
        with np.load(statistics_path) as saved:
            statistics = {name: saved[name] for name in saved.files}
    if summary.get("status") == "complete":
        conditional_subgroups(output, settings)
        return conditional_process_evaluation(output, settings)
    following = load_response()
    lanes = load_lane_opportunities(DRIVER_DATA_DIR)
    following["all"], lanes["all"] = combine_splits(following), combine_splits(lanes)
    for split in (*SPLITS, "all"):
        summary["splits"].setdefault(split, {})
        for stage, (_, population, mobil) in models.items():
            if stage in summary["splits"][split]:
                continue
            result, numeric = conditional_metrics(
                following[split],
                lanes[split],
                population,
                mobil,
                samples=settings["conditional_samples"],
                bootstrap=settings["bootstrap_samples"],
            )
            probability = numeric["lane_probability"]
            lateral_protocol = load_json(DRIVER_DATA_DIR / "lateral_protocol.json")
            lane = lanes[split]
            recording_ids, recording_index = np.unique(
                lane["recording"], return_inverse=True
            )
            edges = [
                lateral_protocol["joint_edges"][name]
                for name in ("front_gap", "rear_gap")
            ]
            shape = tuple(len(value) for value in edges)
            joint_real = np.zeros((len(recording_ids), *shape))
            joint_model = joint_real.copy()
            for side in range(2):
                bins = []
                for name, boundaries in zip(("front", "rear"), edges):
                    index = (
                        np.searchsorted(
                            boundaries,
                            lane[f"target_{name}_gap"][:, side],
                            side="right",
                        )
                        - 1
                    )
                    bins.append(
                        np.where(
                            lane[f"target_{name}_vehicle"][:, side] >= 0,
                            index.clip(0, len(boundaries) - 2),
                            len(boundaries) - 1,
                        )
                    )
                flat = recording_index * np.prod(shape) + bins[0] * shape[1] + bins[1]
                joint_real += np.bincount(
                    flat, weights=(lane["label"] == side + 1), minlength=joint_real.size
                ).reshape(joint_real.shape)
                joint_model += np.bincount(
                    flat, weights=probability[:, side + 1], minlength=joint_model.size
                ).reshape(joint_model.shape)
            numeric["start_gap_joint_real_recording"] = joint_real
            numeric["start_gap_joint_model_recording"] = joint_model
            side_curves = np.zeros((2, 4, 3))
            side_intervals = np.zeros((2, 4, 2, 2))
            for side in range(2):
                for index, (lower, upper) in enumerate(
                    zip((0, 15, 30, 60), (15, 30, 60, np.inf))
                ):
                    lane = lanes[split]
                    keep = (
                        lane["available"][:, side]
                        & (lane["target_rear_gap"][:, side] >= lower)
                        & (lane["target_rear_gap"][:, side] < upper)
                    )
                    ids, cluster = np.unique(
                        lane["recording"][keep], return_inverse=True
                    )
                    counts = np.bincount(cluster, minlength=len(ids))
                    observed = np.bincount(
                        cluster,
                        weights=(lane["label"][keep] == side + 1),
                        minlength=len(ids),
                    )
                    predicted = np.bincount(
                        cluster, weights=probability[keep, side + 1], minlength=len(ids)
                    )
                    draws = np.random.default_rng(SEED).integers(
                        len(ids), size=(settings["bootstrap_samples"], len(ids))
                    )
                    side_curves[side, index] = [
                        counts.sum(),
                        observed.sum() / counts.sum(),
                        predicted.sum() / counts.sum(),
                    ]
                    side_intervals[side, index, 0] = np.quantile(
                        observed[draws].sum(1) / counts[draws].sum(1), [0.025, 0.975]
                    )
                    side_intervals[side, index, 1] = np.quantile(
                        predicted[draws].sum(1) / counts[draws].sum(1), [0.025, 0.975]
                    )
            numeric["lane_side_probability"] = side_curves
            numeric["lane_side_probability_CI95"] = side_intervals
            summary["splits"][split][stage] = result
            for name, value in numeric.items():
                if name == "lane_probability":
                    continue
                statistics[
                    (
                        "acceptance_edges"
                        if name == "acceptance_edges"
                        else f"{split}__{stage}__{name}"
                    )
                ] = value
            summary["status"] = "running"
            _save_statistics(statistics_path, statistics)
            save_json(summary, summary_path)
            print(
                f"Conditional NPC {split}/{stage}: {len(following[split]['gap']):,} longitudinal states",
                flush=True,
            )
    subset_following = convergence_subset(
        following["all"], settings["convergence_fraction"]
    )
    subset_lane = convergence_subset(
        lanes["all"], settings["convergence_fraction"], lane=True
    )
    subset_path = directory / "convergence_subset.npz"
    _save_statistics(
        subset_path,
        {
            "following_ids": np.column_stack(
                [subset_following[key] for key in ("recording", "vehicle", "frame")]
            ),
            "lane_ids": np.column_stack(
                [subset_lane[key] for key in ("recording", "vehicle", "frame")]
            ),
        },
    )
    for stage, (_, population, mobil) in models.items():
        if stage in summary["convergence"]:
            continue
        comparisons = {}
        for samples in (
            settings["conditional_samples"],
            settings["convergence_samples"],
        ):
            comparisons[str(samples)] = conditional_metrics(
                subset_following,
                subset_lane,
                population,
                mobil,
                samples=samples,
                bootstrap=settings["bootstrap_samples"],
            )[0]
        summary["convergence"][stage] = comparisons
        save_json(summary, summary_path)
    summary["status"] = "complete"
    summary["convergence_protocol"] = {
        "fraction": settings["convergence_fraction"],
        "following_states": len(subset_following["gap"]),
        "lane_opportunities": len(subset_lane["label"]),
        "identity": "recording, vehicle, frame",
        "strata": "recording and longitudinal condition or lane label",
    }
    save_json(summary, summary_path)
    conditional_subgroups(output, settings)
    return conditional_process_evaluation(output, settings)


def _closed_rows(section, stage, result, duration):
    rows = [
        {
            "experiment": section,
            "model": stage,
            "metric": f"{feature}_Hellinger",
            "value": value["Hellinger"],
            "CI_low": value["CI95"][0],
            "CI_high": value["CI95"][1],
            "horizon_s": duration,
        }
        for feature, value in result.items()
        if feature in (*FEATURE_EDGES, "joint")
    ]
    for feature, value in result.items():
        if feature in (*FEATURE_EDGES, "joint") and "cumulative_60s_Hellinger" in value:
            rows.append(
                {
                    "experiment": section,
                    "model": stage,
                    "metric": feature + "_cumulative_60s_Hellinger",
                    "value": value["cumulative_60s_Hellinger"],
                    "CI_low": value["cumulative_60s_CI95"][0],
                    "CI_high": value["cumulative_60s_CI95"][1],
                    "horizon_s": 60,
                }
            )
    for name, value in result["exposure"].items():
        if name in result.get("road_collision", {}):
            continue
        interval = result["exposure_CI95"].get(name)
        rows.append(
            {
                "experiment": section,
                "model": stage,
                "metric": name,
                "value": value,
                "horizon_s": duration,
                "CI_low": None if interval is None else interval[0],
                "CI_high": None if interval is None else interval[1],
                "first_8_repeats_value": result["first_8_repeats_exposure"][name],
            }
        )
    for name, value in result.get("road_collision", {}).items():
        if isinstance(value, dict) and "value" in value:
            interval = value["CI95"]
            rows.append(
                {
                    "experiment": section,
                    "model": stage,
                    "metric": name,
                    "value": value["value"],
                    "CI_low": interval[0],
                    "CI_high": interval[1],
                    "first_8_repeats_value": value["first_half"],
                    "horizon_s": duration,
                }
            )
    rows.extend(
        {
            "experiment": section,
            "model": stage,
            "metric": "lane_" + name,
            "value": value,
            "horizon_s": duration,
        }
        for name, value in result.get("lane_decision", {})
        .get("expected_counts", {})
        .items()
    )
    return rows


def runtime_summary(numeric, keys, records, repeats, bootstrap=2000):
    """Road crash flags and explicit decision accounting, clustered by recording."""
    recording = np.asarray([int(key.split("_")[0]) for key in keys])
    unique = np.unique(recording)
    flags = [numeric[f"{key}_road_crashed_vehicles"] for key in keys]
    collision = np.asarray([value.any(1).mean() for value in flags])
    time = np.asarray([value[:, 1:].sum(1).mean() * 0.04 for value in flags])
    half_collision = np.asarray(
        [value[: max(1, repeats // 2)].any(1).mean() for value in flags]
    )
    half_time = np.asarray(
        [value[: max(1, repeats // 2), 1:].sum(1).mean() * 0.04 for value in flags]
    )
    real_time = np.asarray(
        [numeric[f"{key}_real_exposure"].mean(0).sum(0)[0] for key in keys]
    )
    count = np.asarray([(recording == value).sum() for value in unique])
    clustered = np.stack(
        [
            np.array(
                [
                    collision[recording == value].sum(),
                    time[recording == value].sum(),
                    real_time[recording == value].sum(),
                ]
            )
            for value in unique
        ]
    )
    draws = np.random.default_rng(20261012).integers(
        len(unique), size=(bootstrap, len(unique))
    )
    boot = clustered[draws].sum(1)
    estimates = {
        "road_collision_run_fraction": (
            collision.mean(),
            half_collision.mean(),
            boot[:, 0] / count[draws].sum(1),
        ),
        "road_crashed_vehicle_s": (time.sum(), half_time.sum(), boot[:, 1]),
        "road_crashed_exposure_ratio": (
            time.sum() / real_time.sum(),
            half_time.sum() / real_time.sum(),
            boot[:, 1] / boot[:, 2],
        ),
    }
    crash = {
        name: {
            "value": float(value),
            "first_half": float(half),
            "CI95": np.quantile(samples, [0.025, 0.975]).tolist(),
        }
        for name, (value, half, samples) in estimates.items()
    }
    crash["time_scope"] = (
        "HighwayEnv crashed flag on all road actors; post-step .04s exposure, including exit footprints"
    )
    counters = {
        name[5:]: int(sum(row[name] for row in records))
        for name in records[0]
        if name.startswith("lane_") and name != "lane_starts"
    }
    decision = {
        "actual_counts": counters,
        "expected_counts": {name: value / repeats for name, value in counters.items()},
        "eligible_balance": counters["eligible_steps"]
        == sum(
            counters[name]
            for name in (
                "keep_choice_steps",
                "lower_choice_steps",
                "higher_choice_steps",
            )
        ),
        "request_balance": counters["requested_changes"]
        == sum(
            counters[name]
            for name in ("path_rejected", "reservation_rejected", "started_changes")
        ),
    }
    return crash, decision


def temporal_summary(numeric, keys, bootstrap):
    """Recording-cluster intervals from saved residual covariance and jerk counts."""
    recording = np.array([int(key.split("_")[0]) for key in keys])
    unique = np.unique(recording)
    draws = np.random.default_rng(20261012).integers(
        len(unique), size=(bootstrap, len(unique))
    )

    def correlation(values):
        return values[..., 0] / np.sqrt(
            np.maximum(values[..., 1] * values[..., 2], 1e-12)
        )

    result = {
        "lag_seconds": [0.2, 0.4, 1.0, 2.0, 4.0],
        "residual_definition": "common Train population-mean IDM; per-vehicle mean removed",
    }
    jerk = []
    for owner in ("real", "model"):
        covariance = np.stack(
            [numeric[f"{key}_{owner}_residual_covariance"].mean(0) for key in keys]
        )
        clusters = np.stack([covariance[recording == item].sum(0) for item in unique])
        boots = correlation(clusters[draws].sum(1))
        result[owner] = {
            "correlation": correlation(covariance.sum(0)).tolist(),
            "CI95": np.quantile(boots, [0.025, 0.975], axis=0).T.tolist(),
            "pairs": covariance.sum(0)[..., 3].tolist(),
        }
        jerk.append(np.stack([numeric[f"{key}_{owner}_jerk"].mean(0) for key in keys]))
    r, m = jerk
    r = np.stack([r[recording == item].sum(0) for item in unique])
    m = np.stack([m[recording == item].sum(0) for item in unique])
    boot = hellinger(r[draws].sum(1), m[draws].sum(1))
    result["jerk_Hellinger"] = {
        "value": float(hellinger(r.sum(0), m.sum(0))),
        "CI95": np.quantile(boot, [0.025, 0.975]).tolist(),
    }
    return result


def diagnostic_summary(numeric, keys, bootstrap):
    recording = np.array([int(key.split("_")[0]) for key in keys])
    unique = np.unique(recording)
    draws = np.random.default_rng(SEED).integers(
        len(unique), size=(bootstrap, len(unique))
    )
    report = {}
    for owner in ("real", "model"):
        ttc = np.stack([numeric[f"{key}_{owner}_ttc"].mean(0).sum(0) for key in keys])
        status = np.stack(
            [numeric[f"{key}_{owner}_following_status"].mean(0).sum(0) for key in keys]
        )
        pairs = {
            "TTC_below_1s_following_fraction": (ttc[:, :2].sum(1), ttc.sum(1)),
            "TTC_below_3s_following_fraction": (ttc[:, :3].sum(1), ttc.sum(1)),
            "nonclosing_following_fraction": (status[:, 2], status[:, 1:].sum(1)),
            "no_visible_valid_leader_fraction": (status[:, 0], status.sum(1)),
        }
        result = {}
        for name, (top, bottom) in pairs.items():
            counts = np.stack(
                [
                    np.array(
                        [top[recording == rec].sum(), bottom[recording == rec].sum()]
                    )
                    for rec in unique
                ]
            )
            sampled = counts[draws].sum(1)
            keep = sampled[:, 1] > 0
            result[name] = {
                "value": float(top.sum() / bottom.sum()) if bottom.sum() else None,
                "CI95": (
                    np.quantile(
                        sampled[keep, 0] / sampled[keep, 1], [0.025, 0.975]
                    ).tolist()
                    if keep.any()
                    else [None, None]
                ),
            }
        report[owner] = result
    report["residence"] = {}
    from .lateral import distance

    for feature in ("speed", "joint"):
        real = sum(numeric[f"{key}_real_age_{feature}"].mean(0) for key in keys)
        model = sum(numeric[f"{key}_model_age_{feature}"].mean(0) for key in keys)
        report["residence"][feature] = {
            "bins_s": [0, 2, 5, 10, None],
            "Hellinger": distance(real, model).tolist(),
            "real_observations": real.sum(-1).tolist(),
            "model_observations_per_repeat": model.sum(-1).tolist(),
        }
    return report


def closed_evaluation(output, settings):
    output = Path(output)
    directory = output / "autonomous_traffic"
    directory.mkdir(parents=True, exist_ok=True)
    selected_stage, models = stage_models(output)
    cohorts = load_cohorts()
    settings = {
        **settings,
        "lateral_protocol": load_json(DRIVER_DATA_DIR / "lateral_protocol.json"),
    }
    evaluation = settings["evaluation"]
    sections = {
        "short": (
            sorted(cohorts),
            evaluation["short_repeats"],
            evaluation["short_duration_s"],
            1.0,
        ),
        "long": (
            sorted(cohorts),
            evaluation["long_repeats"],
            evaluation["long_duration_s"],
            evaluation["window_s"],
        ),
    }
    report = {
        "selected_stage": selected_stage,
        "status": "running",
        "abnormal_exposure_denominator": "matched observed vehicle seconds",
        "bootstrap_samples": settings["bootstrap_samples"],
    }
    records, table = [], []
    for section, (keys, repeats, duration, window) in sections.items():
        report[section] = {
            "models": {},
            "cohorts": [section + "_" + key for key in keys],
            "duration_seconds": duration,
            "window_seconds": window,
            "repeats": repeats,
        }
        stage_settings = {**settings, "evaluation": {**evaluation, "window_s": window}}
        for stage, (_, population, mobil) in models.items():
            result, numeric, runs = run_cohorts(
                directory,
                ROLLOUT_DIRECTORIES[section],
                keys,
                cohorts,
                population,
                mobil,
                stage_settings,
                repeats,
                duration,
            )
            result["road_collision"], result["lane_decision"] = runtime_summary(
                numeric, keys, runs, repeats, settings["bootstrap_samples"]
            )
            result["temporal"] = temporal_summary(
                numeric, keys, settings["bootstrap_samples"]
            )
            result["diagnostics"] = diagnostic_summary(
                numeric, keys, settings["bootstrap_samples"]
            )
            report[section]["models"][stage] = result
            records.extend(
                {"experiment": section, "model": stage, **run} for run in runs
            )
            table.extend(_closed_rows(section, stage, result, duration))
            for name, value in result["diagnostics"]["model"].items():
                table.append(
                    {
                        "experiment": section,
                        "model": stage,
                        "metric": name,
                        "value": value["value"],
                        "CI_low": value["CI95"][0],
                        "CI_high": value["CI95"][1],
                        "real_value": result["diagnostics"]["real"][name]["value"],
                        "horizon_s": duration,
                    }
                )
            for metric, value in result["lateral"].items():
                if metric in (
                    "counts",
                    "score",
                    "start",
                    "process",
                    "model_rate",
                    "real_rate",
                ):
                    continue
                table.append(
                    {
                        "experiment": section,
                        "model": stage,
                        "metric": "lateral_" + metric,
                        "value": value["value"],
                        "CI_low": value["CI95"][0],
                        "CI_high": value["CI95"][1],
                        "horizon_s": duration,
                    }
                )
            for index, metric in enumerate(
                (
                    "motion_episodes",
                    "completed_events",
                    "observed_starts",
                    "fully_observed_events",
                    "initial_completed_continuations",
                    "exit_censored_events",
                    "end_censored_events",
                )
            ):
                table.append(
                    {
                        "experiment": section,
                        "model": stage,
                        "metric": metric,
                        "value": result["lateral"]["counts"]["model_per_repeat"][index],
                        "real_value": result["lateral"]["counts"]["real"][index],
                        "horizon_s": duration,
                    }
                )

            for index, lag in enumerate(result["temporal"]["lag_seconds"]):
                value = result["temporal"]["model"]
                table.append(
                    {
                        "experiment": section,
                        "model": stage,
                        "metric": f"acceleration_residual_correlation_{lag:g}s",
                        "value": value["correlation"][index],
                        "CI_low": value["CI95"][index][0],
                        "CI_high": value["CI95"][index][1],
                        "horizon_s": duration,
                    }
                )
            value = result["temporal"]["jerk_Hellinger"]
            table.append(
                {
                    "experiment": section,
                    "model": stage,
                    "metric": "jerk_Hellinger_diagnostic",
                    "value": value["value"],
                    "CI_low": value["CI95"][0],
                    "CI_high": value["CI95"][1],
                    "horizon_s": duration,
                }
            )
            save_json(report, directory / "report.json")
            print(
                f"HighwayEnv NPC {section}/{stage}: {len(keys) * repeats:,} runs completed",
                flush=True,
            )
    report["status"] = (
        "complete"
        if all(row["status"] == "complete" for row in records)
        else "incomplete"
    )
    pd.DataFrame(records).to_csv(directory / "runs.csv", index=False)
    report["actual_road_steps"] = int(sum(row["road_step_calls"] for row in records))
    report["completed_runs"] = len(records)
    report["measured_compute_seconds"] = float(
        sum(row["elapsed_seconds"] for row in records)
    )
    save_json(report, directory / "report.json")
    tables = output / "tables"
    tables.mkdir(exist_ok=True)
    pd.DataFrame(table).assign(
        phase="evaluation",
        fit_split="train",
    ).to_csv(tables / "closed_loop_metrics.csv", index=False)
    return report


def conditional_process_evaluation(output, settings):
    from .lateral import (
        PROCESS_FEATURES,
        conditional_lane_statistics,
        distance,
    )

    output = Path(output)
    summary_path, statistics_path = (
        output / "conditional_behavior/summary.json",
        output / "conditional_behavior/statistics.npz",
    )
    summary, statistics = load_json(summary_path), dict(np.load(statistics_path))
    _, models = stage_models(output)
    for split in SPLITS:
        for stage, (_, _, mobil) in models.items():
            result = summary["splits"][split][stage]
            if "process" in result:
                continue
            report, numeric = conditional_lane_statistics(
                DRIVER_DATA_DIR, split, mobil, samples=settings["conditional_samples"]
            )
            result["process"] = report
            statistics.update(
                {
                    f"{split}__{stage}__process__{name}": value
                    for name, value in numeric.items()
                }
            )
            _save_statistics(statistics_path, statistics)
            save_json(summary, summary_path)
            print(
                "Conditional lane execution", split, stage, report["events"], flush=True
            )
    for stage in models:
        all_numeric = {}
        for split in SPLITS:
            prefix = f"{split}__{stage}__process__"
            for name, value in statistics.items():
                if name.startswith(prefix):
                    field = name[len(prefix) :]
                    all_numeric.setdefault(field, []).append(value)
        all_numeric = {
            name: np.concatenate(values) for name, values in all_numeric.items()
        }
        report = {
            "events": sum(
                summary["splits"][split][stage]["process"]["events"] for split in SPLITS
            ),
            "samples": settings["conditional_samples"],
        }
        draws = np.random.default_rng(SEED).integers(
            len(all_numeric["recordings"]),
            size=(settings["bootstrap_samples"], len(all_numeric["recordings"])),
        )
        for name in (*PROCESS_FEATURES, "duration_speed_joint"):
            r, m = (
                all_numeric[owner + "_" + name].reshape(
                    len(all_numeric["recordings"]), -1
                )
                for owner in ("real", "model")
            )
            boot = distance(r[draws].sum(1), m[draws].sum(1))
            report[name] = {
                "Hellinger": float(distance(r.sum(0), m.sum(0))),
                "SE": float(boot.std(ddof=1)),
                "CI95": np.quantile(boot, [0.025, 0.975]).tolist(),
            }
        summary["splits"]["all"][stage]["process"] = report
        statistics.update(
            {
                f"all__{stage}__process__{name}": value
                for name, value in all_numeric.items()
            }
        )
    _save_statistics(statistics_path, statistics)
    save_json(summary, summary_path)
    _save_conditional_table(output, summary)
    table_path = output / "tables/conditional_metrics.csv"
    rows = []
    for split, stages in summary["splits"].items():
        for stage, result in stages.items():
            for name in (*PROCESS_FEATURES, "duration_speed_joint"):
                value = result["process"][name]
                rows.append(
                    {
                        "split": split,
                        "model": stage,
                        "group": "conditional_lane_execution",
                        "metric": name + "_Hellinger",
                        "observations": result["process"]["events"],
                        "value": value["Hellinger"],
                        "CI_low": value["CI95"][0],
                        "CI_high": value["CI95"][1],
                    }
                )
    calibration_rows = []
    for split, stages in summary["splits"].items():
        for stage, result in stages.items():
            prefixes = [
                f"{split}__{stage}__subgroup__{group}__" for group in ("car", "truck")
            ]
            observed = sum(statistics[prefix + "lane_observed"] for prefix in prefixes)
            predicted = sum(
                statistics[prefix + "lane_predicted"] for prefix in prefixes
            )
            counts = sum(statistics[prefix + "lane_count"] for prefix in prefixes)
            draws = np.random.default_rng(SEED).integers(
                len(counts), size=(settings["bootstrap_samples"], len(counts))
            )
            bias = (predicted.sum(0) - observed.sum(0)) / counts.sum()
            samples = (predicted[draws].sum(1) - observed[draws].sum(1)) / counts[
                draws
            ].sum(1)[:, None]
            for index, side in enumerate(("keep", "right", "left")):
                interval = np.quantile(samples[:, index], [0.025, 0.975])
                calibration_rows.append(
                    {
                        "split": split,
                        "model": stage,
                        "group": "lane_choice",
                        "metric": side + "_probability_bias",
                        "observations": int(counts.sum()),
                        "value": bias[index],
                        "real_value": observed[:, index].sum() / counts.sum(),
                        "model_value": predicted[:, index].sum() / counts.sum(),
                        "CI_low": interval[0],
                        "CI_high": interval[1],
                    }
                )
    pd.concat(
        [pd.read_csv(table_path), pd.DataFrame(rows), pd.DataFrame(calibration_rows)],
        ignore_index=True,
    ).assign(
        phase="evaluation",
        fit_split="train",
    ).to_csv(table_path, index=False)
    return summary


def verify_npc_results(output, settings):
    """Recompute current NPC statistics and check complete physical-run coverage."""
    from ..stochastic_drivers.raw_data import BEHAVIOR_PROTOCOL
    from .figures import FIGURE_FILES

    output = Path(output)
    raw = load_json(DRIVER_DATA_DIR / "manifest.json")
    fit = load_json(output / "policy_calibration/selected_model.json")
    selected_stage, models = stage_models(output)
    conditional = load_json(output / "conditional_behavior/summary.json")
    closed = load_json(output / "autonomous_traffic/report.json")
    runs = pd.read_csv(output / "autonomous_traffic/runs.csv")
    cohorts = load_cohorts()
    population = models[selected_stage][1]
    checks = {
        "sixty_recordings": sum(map(len, raw["split_recordings"].values())) == 60,
        "recording_isolation": all(
            not set(raw["split_recordings"][left]) & set(raw["split_recordings"][right])
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        ),
        "causal_behavior_protocol": raw["behavior_protocol"] == BEHAVIOR_PROTOCOL,
        "train_fit_val_selection": fit["fit_split"] == "train"
        and fit["selection_split"] == "val"
        and not fit["test_used_for_fit_or_selection"],
        "one_selected_model": set(models) == {selected_stage},
        "frozen_parameter_values": all(
            np.array_equal(population[name], fit["driver_population"][name])
            for name in ("log_mean", "log_cov")
        ),
        "all_conditional_splits": set(conditional["splits"]) == {*SPLITS, "all"},
        "conditional_process_complete": all(
            set(group) == {selected_stage} and "process" in group[selected_stage]
            for group in conditional["splits"].values()
        ),
        "no_symbolic_links": not any(path.is_symlink() for path in output.rglob("*")),
        "flat_figures": {path.name for path in (output / "figures").iterdir()}
        == set(FIGURE_FILES),
        "two_csv_tables": {path.name for path in (output / "tables").iterdir()}
        == {"conditional_metrics.csv", "closed_loop_metrics.csv"},
        "road_runs_complete": closed["status"] == "complete"
        and runs.status.eq("complete").all(),
        "actual_steps_match": int(runs.road_step_calls.sum())
        == closed["actual_road_steps"],
        "run_identity_unique": not runs.duplicated(
            ["experiment", "model", "cohort", "repeat"]
        ).any(),
    }
    for name in ("conditional_metrics", "closed_loop_metrics"):
        table = pd.read_csv(output / "tables" / (name + ".csv"))
        checks[name + "/scope"] = (
            table.phase.eq("evaluation").all()
            and table.fit_split.eq("train").all()
            and table.model.eq(selected_stage).all()
        )
    with np.load(output / "conditional_behavior/statistics.npz") as statistics:
        for split, stages in conditional["splits"].items():
            for stage, result in stages.items():
                for group in result["acceleration"]:
                    prefix = f"{split}__{stage}__{group['group']}"
                    r, m = statistics[prefix + "_real"], statistics[prefix + "_model"]
                    checks[f"{split}/{stage}/{group['group']}/counts"] = np.isclose(
                        hellinger(r.sum(0), m.sum(0)), group["Hellinger"]
                    )
                sums = statistics[f"{split}__{stage}__lane_Brier_recording_sums"]
                counts = statistics[f"{split}__{stage}__lane_Brier_recording_counts"]
                checks[f"{split}/{stage}/Brier"] = np.isclose(
                    sums.sum() / counts.sum(), result["lane"]["Brier"]
                )
    for section in ("short", "long"):
        expected_keys = sorted(cohorts)
        repeats = settings["evaluation"][section + "_repeats"]
        duration = settings["evaluation"][section + "_duration_s"]
        for stage in models:
            records = runs[(runs.experiment == section) & (runs.model == stage)]
            checks[f"{section}/{stage}/coverage"] = set(
                zip(records.cohort, records.repeat)
            ) == {(key, repeat) for key in expected_keys for repeat in range(repeats)}
            checks[f"{section}/{stage}/horizon"] = records.road_step_calls.eq(
                round(duration / 0.04)
            ).all()
            folder = output / "autonomous_traffic" / ROLLOUT_DIRECTORIES[section]
            with np.load(folder / "statistics.npz") as saved:
                numeric = dict(saved)
            execution = load_json(folder / "runs.json")
            checks[f"{section}/{stage}/saved_execution"] = set(execution) == set(
                expected_keys
            ) and all(
                {run["repeat"] for run in item["runs"]} == set(range(repeats))
                and len(item["runs"]) == repeats
                and all(
                    run["status"] == "complete"
                    and run["road_step_calls"] == round(duration / 0.04)
                    for run in item["runs"]
                )
                for item in execution.values()
            )
            recomputed = closed_summary(
                numeric, expected_keys, settings["bootstrap_samples"]
            )
            reported = closed[section]["models"][stage]
            checks[f"{section}/{stage}/distributions"] = all(
                np.isclose(recomputed[name]["Hellinger"], reported[name]["Hellinger"])
                for name in (*FEATURE_EDGES, "joint")
            )
            checks[f"{section}/{stage}/lateral"] = all(
                np.isclose(value["value"], reported["lateral"][name]["value"])
                for name, value in recomputed["lateral"].items()
                if name != "counts"
            )
            checks[f"{section}/{stage}/execution_balance"] = (
                reported["lane_decision"]["eligible_balance"]
                and reported["lane_decision"]["request_balance"]
            )
    checks = {name: bool(value) for name, value in checks.items()}
    verification = {
        "status": "complete",
        "passed": all(checks.values()),
        "checks": checks,
        "scope": "Selected NPC policy; natural conditions and autonomous highway-env traffic",
    }
    save_json(verification, output / "verification.json")
    return verification

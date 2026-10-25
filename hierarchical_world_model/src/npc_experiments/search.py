"""Paired, finite-budget driver refinement on actual HighwayEnv rollouts."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from process_highD.src.io_utils import load_config, resolve_data_path

from ..core.utils import (
    load_json,
    save_json,
)
from ..data import artifact_identity
from ..stochastic_drivers.idm_population import LOWER, UPPER
from ..stochastic_drivers.raw_data import (
    BEHAVIOR_PROTOCOL,
    DRIVER_DATA_DIR,
    load_lane_opportunities,
)
from ..stochastic_drivers.response_data import load_response
from .calibration import (
    compact_population,
    conditional_metrics,
    fit_correction,
)
from .lateral import (
    EVENT_FIELDS,
    event_histograms,
    lane_events,
    lateral_distances,
)
from .metrics import (
    FEATURE_EDGES,
    hellinger,
)
from .residual import (
    autoregression_parameters,
    temporal_statistics,
)
from .road import (
    road_cohorts,
    simulate_road,
    traffic_statistics,
)

ROOT = Path(__file__).resolve().parents[3]
SEED = 20261012


def load_cohorts():
    path = DRIVER_DATA_DIR / "road_cohorts.npz"
    if not path.exists():
        config_path = ROOT / "process_highD/config.yaml"
        config = load_config(config_path)
        raw = resolve_data_path(config["paths"]["raw_dir"], config_path)
        manifest = load_json(DRIVER_DATA_DIR / "manifest.json")
        recordings = sorted(
            rec for group in manifest["split_recordings"].values() for rec in group
        )
        cohorts, _ = road_cohorts(raw, config, recordings)
        np.savez_compressed(
            path,
            **{
                key + "__" + name: value
                for key, cohort in cohorts.items()
                for name, value in cohort.items()
            },
        )
    with np.load(path) as data:
        keys = sorted({name.split("__")[0] for name in data.files})
        return {
            key: {
                name.split("__")[1]: data[name]
                for name in data.files
                if name.startswith(key + "__")
            }
            for key in keys
        }


def cohort_selection(cohorts, recordings, count):
    """Fixed density strata, then spread across physical recordings/directions."""
    keys = [key for key in sorted(cohorts) if int(key.split("_")[0]) in recordings]
    keys.sort(
        key=lambda key: (
            cohorts[key]["valid"][0].sum() / np.diff(cohorts[key]["bounds"])[0],
            key,
        )
    )
    indices = np.linspace(0, len(keys) - 1, min(count, len(keys))).round().astype(int)
    return [keys[index] for index in indices]


def rollout_job(
    key,
    cohort,
    population,
    mobil,
    repeats,
    duration,
    window,
    lateral_protocol,
    diagnostic_population,
):
    """Save only sufficient statistics and audit metadata, not all trial traces."""
    import torch

    torch.set_num_threads(1)
    states, valid, metadata = simulate_road(
        cohort,
        population,
        mobil,
        repeats=repeats,
        duration_s=duration,
        seed=SEED,
    )
    all_events, event_numeric, masks = [], {}, []
    for label, trajectory, present in (
        ("model", states, valid),
        (
            "real",
            cohort["states"][: states.shape[1]][None],
            cohort["valid"][: states.shape[1]][None],
        ),
    ):
        histograms, curve_sums, curve_counts, phase_masks = [], [], [], []
        for repeat in range(len(trajectory)):
            inside = (trajectory[repeat, :, :, 0] >= cohort["bounds"][0]) & (
                trajectory[repeat, :, :, 0] <= cohort["bounds"][1]
            )
            events, finished, curves = lane_events(
                trajectory[repeat],
                present[repeat] & inside,
                cohort["lane_centers"],
                cohort["dimensions"],
                recording=int(cohort["recording_id"]),
                repeat=repeat,
                vehicle_ids=cohort["vehicle_ids"],
            )
            all_events.extend({**event, "source": label} for event in events)
            histograms.append(event_histograms(events, lateral_protocol))
            curve_sums.append(curves.sum(0))
            curve_counts.append(len(curves))
            phase_masks.append(finished)
        for name in histograms[0]:
            event_numeric[label + "_lane_" + name] = np.stack(
                [histogram[name] for histogram in histograms]
            )
        event_numeric[label + "_phase_sum"] = np.asarray(curve_sums)
        event_numeric[label + "_phase_count"] = np.asarray(curve_counts)
        masks.append(np.stack(phase_masks))
    complete, real_complete = masks
    histogram, exposure = traffic_statistics(
        states, valid, cohort, window_s=window, completed_changes=complete
    )
    ticks = states.shape[1]
    real, real_exposure = traffic_statistics(
        cohort["states"][:ticks],
        cohort["valid"][:ticks],
        cohort,
        window_s=window,
        completed_changes=real_complete,
    )
    recorded_runs = []
    for row in metadata["runs"]:
        record = {
            name: value
            for name, value in row.items()
            if name
            not in (
                "completed_lane_changes",
                "queue_vehicles",
                "crashed_vehicles",
                "entry_tick",
                "exit_tick",
                "entry_wait_seconds",
                "lane_decision",
            )
        }
        waits = np.asarray(row["entry_wait_seconds"])
        entered = np.asarray(row["entry_tick"]) >= 0
        record["mean_entry_wait_s"] = (
            float(waits[entered].mean()) if entered.any() else 0.0
        )
        record["max_entry_wait_s"] = (
            float(waits[entered].max()) if entered.any() else 0.0
        )
        record["final_queue"] = int(row["queue_vehicles"][-1])
        record["queued_vehicle_s"] = float(
            np.asarray(row["queue_vehicles"]).sum() * 0.04
        )
        record.update(
            {"lane_" + name: int(value) for name, value in row["lane_decision"].items()}
        )
        record["road_collision"] = bool(np.asarray(row["crashed_vehicles"]).any())
        record["road_crashed_vehicle_s"] = float(
            np.asarray(row["crashed_vehicles"])[1:].sum() * 0.04
        )
        recorded_runs.append(record)
    numeric = {
        **event_numeric,
        **{f"model_{name}": values for name, values in histogram.items()},
        **{f"real_{name}": values for name, values in real.items()},
        "model_exposure": exposure,
        "real_exposure": real_exposure,
        "entry_ticks": np.stack([row["entry_tick"] for row in metadata["runs"]]),
        "exit_ticks": np.stack([row["exit_tick"] for row in metadata["runs"]]),
        "entry_wait_s": np.stack(
            [row["entry_wait_seconds"] for row in metadata["runs"]]
        ),
        "road_crashed_vehicles": np.stack(
            [row["crashed_vehicles"] for row in metadata["runs"]]
        ),
    }
    for label, trajectory, present in (
        ("model", states, valid),
        ("real", cohort["states"][:ticks], cohort["valid"][:ticks]),
    ):
        covariance, jerk = temporal_statistics(
            trajectory, present, cohort, diagnostic_population
        )
        numeric[label + "_residual_covariance"] = covariance
        numeric[label + "_jerk"] = jerk
    return (
        key,
        numeric,
        {"backend": metadata["backend"], "runs": recorded_runs, "events": all_events},
    )


def consolidate_road_results(directory, keys):
    """Replace completed road shards with lossless statistics, runs and events."""
    directory = Path(directory)
    numeric, metadata, events = {}, {}, []
    for key in keys:
        metadata[key] = load_json(directory / f"{key}.json")
        with np.load(directory / f"{key}.npz") as saved:
            numeric.update({f"{key}_{name}": saved[name] for name in saved.files})
        events.append(
            pd.read_csv(
                directory / f"{key}_events.csv", float_precision="round_trip"
            ).assign(cohort=key)
        )
    np.savez_compressed(directory / "statistics.npz", **numeric)
    pd.concat(events, ignore_index=True).to_csv(directory / "events.csv", index=False)
    # Written last: a resumed evaluation only consumes a complete bundle.
    save_json(metadata, directory / "runs.json")
    for key in keys:
        for suffix in (".json", ".npz", "_events.csv"):
            (directory / (key + suffix)).unlink()


def run_cohorts(
    output,
    name,
    keys,
    cohorts,
    population,
    mobil,
    settings,
    repeats,
    duration,
    *,
    executor=None,
):
    directory = Path(output) / name
    directory.mkdir(parents=True, exist_ok=True)
    diagnostic_population = dict(
        np.load(Path(output).parent / "policy_calibration/reference.npz")
    )
    protocol = {
        "population": {
            key: np.asarray(value).tolist()
            for key, value in compact_population(population).items()
        },
        "mobil": mobil,
        "keys": keys,
        "repeats": repeats,
        "duration_s": duration,
        "seed": SEED,
        "backend": "highway_env.road.road.Road.step",
        "initial_maneuver": "25-frame observed entry history",
        "behavior_protocol": BEHAVIOR_PROTOCOL,
        "lateral_protocol": settings["lateral_protocol"],
        "randomness": "recording/direction/physical vehicle/repeat/time/purpose",
        "lane_decision_statistics": "25Hz eligibility, choice, clearance, reservation and completion counters",
        "diagnostic_population": {
            key: diagnostic_population[key].tolist() for key in ("log_mean", "log_cov")
        },
    }
    manifest = directory / "protocol.json"
    if manifest.exists() and load_json(manifest) != protocol:
        raise ValueError(f"Changed rollout protocol: {directory}")
    save_json(protocol, manifest)
    consolidated = directory / "runs.json"
    pending = [
        key
        for key in keys
        if not consolidated.exists() and not (directory / f"{key}.json").exists()
    ]
    with (
        ProcessPoolExecutor(
            max_workers=settings["workers"], mp_context=get_context("spawn")
        )
        if executor is None
        else nullcontext(executor)
    ) as pool:
        futures = [
            pool.submit(
                rollout_job,
                key,
                cohorts[key],
                compact_population(population),
                mobil,
                repeats,
                duration,
                (
                    settings["evaluation"]["short_window_s"]
                    if duration <= 6
                    else settings["evaluation"]["window_s"]
                ),
                settings["lateral_protocol"],
                diagnostic_population,
            )
            for key in pending
        ]
        for future in as_completed(futures):
            key, numeric, metadata = future.result()
            np.savez_compressed(directory / f"{key}.npz", **numeric)
            events = metadata.pop("events")
            pd.DataFrame(events, columns=(*EVENT_FIELDS, "source")).to_csv(
                directory / f"{key}_events.csv", index=False
            )
            save_json(metadata, directory / f"{key}.json")
    if not consolidated.exists():
        consolidate_road_results(directory, keys)
    with np.load(directory / "statistics.npz") as saved:
        numeric = dict(saved)
    metadata = load_json(consolidated)
    records = []
    for key in keys:
        observed = numeric[f"{key}_real_exposure"].mean(0).sum(0)
        for run in metadata[key]["runs"]:
            measured = numeric[f"{key}_model_exposure"][run["repeat"]].sum(0)
            records.append(
                {
                    "cohort": key,
                    **run,
                    "vehicle_s": float(measured[0]),
                    "contact_vehicle_s": float(measured[3]),
                    "boundary_vehicle_s": float(measured[4]),
                    "matched_real_vehicle_s": float(observed[0]),
                    "contact_exposure_ratio": float(measured[3] / observed[0]),
                    "boundary_exposure_ratio": float(measured[4] / observed[0]),
                }
            )
    return (
        closed_summary(numeric, keys, settings["bootstrap_samples"]),
        numeric,
        records,
    )


def exposure_summary(model, real):
    rate = np.divide(
        model[..., 2] * 1000,
        model[..., 1],
        out=np.zeros_like(model[..., 1]),
        where=model[..., 1] > 0,
    )
    real_rate = np.divide(
        real[..., 2] * 1000,
        real[..., 1],
        out=np.zeros_like(real[..., 1]),
        where=real[..., 1] > 0,
    )
    return {
        "vehicle_s": model[..., 0],
        "vehicle_km": model[..., 1],
        "real_vehicle_s": real[..., 0],
        "real_vehicle_km": real[..., 1],
        "completed_lane_changes": model[..., 2],
        "real_completed_lane_changes": real[..., 2],
        "lane_rate_per_1000_vehicle_km": rate,
        "real_lane_rate": real_rate,
        "lane_rate_absolute_error": np.abs(rate - real_rate),
        "contact_vehicle_s": model[..., 3],
        "boundary_vehicle_s": model[..., 4],
        "contact_exposure_ratio": model[..., 3] / real[..., 0],
        "boundary_exposure_ratio": model[..., 4] / real[..., 0],
        "occupancy_ratio_to_real": model[..., 0] / real[..., 0],
        "diagnostic_exposure_fraction": model[..., 5] / model[..., 0],
    }


def closed_summary(numeric, keys, bootstrap=2000):
    recordings = np.array([int(key.split("_")[0]) for key in keys])
    unique = np.unique(recordings)
    draws = np.random.default_rng(SEED).integers(
        len(unique), size=(bootstrap, len(unique))
    )
    result = {}
    for feature in (*FEATURE_EDGES, "joint"):
        real = np.stack([numeric[f"{key}_real_{feature}"].mean(0) for key in keys])
        model = np.stack([numeric[f"{key}_model_{feature}"].mean(0) for key in keys])
        rec_real = np.stack([real[recordings == rec].sum(0) for rec in unique])
        rec_model = np.stack([model[recordings == rec].sum(0) for rec in unique])
        boots = hellinger(rec_real[draws].sum((1, 2)), rec_model[draws].sum((1, 2)))
        window_boots = hellinger(rec_real[draws].sum(1), rec_model[draws].sum(1))
        first = np.stack(
            [
                numeric[f"{key}_model_{feature}"][
                    : max(1, len(numeric[f"{key}_model_{feature}"]) // 2)
                ].mean(0)
                for key in keys
            ]
        )
        result[feature] = {
            "Hellinger": float(hellinger(real.sum((0, 1)), model.sum((0, 1)))),
            "SE": float(boots.std(ddof=1)),
            "CI95": np.quantile(boots, [0.025, 0.975]).tolist(),
            "window": hellinger(real.sum(0), model.sum(0)).tolist(),
            "window_CI95": np.quantile(window_boots, [0.025, 0.975], axis=0).T.tolist(),
            "first_8_repeats_Hellinger": float(
                hellinger(real.sum((0, 1)), first.sum((0, 1)))
            ),
        }
        if real.shape[1] >= 12:
            early = hellinger(
                rec_real[draws, :6].sum((1, 2)), rec_model[draws, :6].sum((1, 2))
            )
            result[feature]["cumulative_60s_Hellinger"] = float(
                hellinger(real[:, :6].sum((0, 1)), model[:, :6].sum((0, 1)))
            )
            result[feature]["cumulative_60s_CI95"] = np.quantile(
                early, [0.025, 0.975]
            ).tolist()
    real_totals = np.stack(
        [numeric[f"{key}_real_exposure"].mean(0).sum(0) for key in keys]
    )
    model_totals = np.stack(
        [numeric[f"{key}_model_exposure"].mean(0).sum(0) for key in keys]
    )
    result["exposure"] = {
        name: float(value)
        for name, value in exposure_summary(
            model_totals.sum(0), real_totals.sum(0)
        ).items()
    }
    real_clusters = np.stack([real_totals[recordings == rec].sum(0) for rec in unique])
    model_clusters = np.stack(
        [model_totals[recordings == rec].sum(0) for rec in unique]
    )
    boot_exposure = exposure_summary(
        model_clusters[draws].sum(1), real_clusters[draws].sum(1)
    )
    result["exposure_CI95"] = {
        name: np.quantile(boot_exposure[name], [0.025, 0.975]).tolist()
        for name in (
            "vehicle_s",
            "occupancy_ratio_to_real",
            "lane_rate_per_1000_vehicle_km",
            "contact_exposure_ratio",
            "boundary_exposure_ratio",
        )
    }
    first_half = np.stack(
        [
            numeric[f"{key}_model_exposure"][
                : max(1, len(numeric[f"{key}_model_exposure"]) // 2)
            ]
            .mean(0)
            .sum(0)
            for key in keys
        ]
    )
    result["first_8_repeats_exposure"] = {
        name: float(value)
        for name, value in exposure_summary(
            first_half.sum(0), real_totals.sum(0)
        ).items()
    }
    result["search_score"] = float(
        np.mean([result[feature]["Hellinger"] for feature in (*FEATURE_EDGES, "joint")])
    )
    # All four distances are resampled on the same recording clusters.
    bootstrap_scores = []
    for feature in (*FEATURE_EDGES, "joint"):
        r = np.stack([numeric[f"{key}_real_{feature}"].mean(0).sum(0) for key in keys])
        m = np.stack([numeric[f"{key}_model_{feature}"].mean(0).sum(0) for key in keys])
        r = np.stack([r[recordings == rec].sum(0) for rec in unique])
        m = np.stack([m[recordings == rec].sum(0) for rec in unique])
        bootstrap_scores.append(hellinger(r[draws].sum(1), m[draws].sum(1)))
    result["search_score_SE"] = float(np.mean(bootstrap_scores, axis=0).std(ddof=1))
    lateral_names = (
        "direction",
        "duration",
        "peak_lateral_speed",
        "peak_lateral_acceleration",
        "start_gap_joint",
        "start_relative_speed_joint",
        "duration_speed_joint",
        "event_counts",
    )
    r = {
        name: np.stack([numeric[f"{key}_real_lane_{name}"].mean(0) for key in keys])
        for name in lateral_names
    }
    m = {
        name: np.stack([numeric[f"{key}_model_lane_{name}"].mean(0) for key in keys])
        for name in lateral_names
    }
    rc = {
        name: np.stack([value[recordings == rec].sum(0) for rec in unique])
        for name, value in r.items()
    }
    mc = {
        name: np.stack([value[recordings == rec].sum(0) for rec in unique])
        for name, value in m.items()
    }
    point = lateral_distances(
        {name: value.sum(0) for name, value in m.items()},
        {name: value.sum(0) for name, value in r.items()},
        model_totals[:, 1].sum(),
        real_totals[:, 1].sum(),
    )
    bootstrap_lateral = lateral_distances(
        {name: value[draws].sum(1) for name, value in mc.items()},
        {name: value[draws].sum(1) for name, value in rc.items()},
        model_clusters[draws].sum(1)[:, 1],
        real_clusters[draws].sum(1)[:, 1],
    )
    result["lateral"] = {
        name: {
            "value": float(value),
            "SE": float(np.std(bootstrap_lateral[name], ddof=1)),
            "CI95": np.quantile(bootstrap_lateral[name], [0.025, 0.975]).tolist(),
        }
        for name, value in point.items()
    }
    result["lateral"]["counts"] = {
        "real": r["event_counts"].sum(0).tolist(),
        "model_per_repeat": m["event_counts"].sum(0).tolist(),
    }
    longitudinal_draws, acceleration_draws = [], None
    for feature in ("speed", "gap", "joint", "acceleration"):
        rr = np.stack([numeric[f"{key}_real_{feature}"].mean(0) for key in keys])
        mm = np.stack([numeric[f"{key}_model_{feature}"].mean(0) for key in keys])
        rr = np.stack([rr[recordings == rec].sum(0) for rec in unique])
        mm = np.stack([mm[recordings == rec].sum(0) for rec in unique])
        pooled = hellinger(rr[draws].sum((1, 2)), mm[draws].sum((1, 2)))
        windows = hellinger(rr[draws].sum(1), mm[draws].sum(1)).mean(1)
        combined = 0.5 * (pooled + windows)
        if feature == "acceleration":
            acceleration_draws = combined
        else:
            longitudinal_draws.append(combined)
    longitudinal_draws = np.mean(longitudinal_draws, axis=0)
    result["components"] = {
        "longitudinal": float(
            np.mean(
                [
                    0.5 * (result[name]["Hellinger"] + np.mean(result[name]["window"]))
                    for name in ("speed", "gap", "joint")
                ]
            )
        ),
        "lateral": float(point["score"]),
        "acceleration": float(
            0.5
            * (
                result["acceleration"]["Hellinger"]
                + np.mean(result["acceleration"]["window"])
            )
        ),
    }
    result["bootstrap"] = {
        "longitudinal": longitudinal_draws.tolist(),
        "lateral": bootstrap_lateral["score"].tolist(),
        "acceleration": acceleration_draws.tolist(),
        "contact_exposure_ratio": boot_exposure["contact_exposure_ratio"].tolist(),
        "boundary_exposure_ratio": boot_exposure["boundary_exposure_ratio"].tolist(),
    }
    crash = np.array(
        [numeric[f"{key}_road_crashed_vehicles"].any(1).mean() for key in keys]
    )
    crash_clusters = np.array([crash[recordings == rec].sum() for rec in unique])
    cohort_counts = np.array([(recordings == rec).sum() for rec in unique])
    result["exposure"]["road_collision_run_fraction"] = float(crash.mean())
    result["bootstrap"]["road_collision_run_fraction"] = (
        crash_clusters[draws].sum(1) / cohort_counts[draws].sum(1)
    ).tolist()
    result["first_8_repeats_exposure"]["road_collision_run_fraction"] = float(
        np.mean(
            [
                numeric[f"{key}_road_crashed_vehicles"][
                    : max(1, len(numeric[f"{key}_road_crashed_vehicles"]) // 2)
                ]
                .any(1)
                .mean()
                for key in keys
            ]
        )
    )
    result["exposure_CI95"]["road_collision_run_fraction"] = np.quantile(
        result["bootstrap"]["road_collision_run_fraction"], [0.025, 0.975]
    ).tolist()
    for feature in ("speed", "joint"):
        result["components"][feature] = float(
            0.5 * (result[feature]["Hellinger"] + np.mean(result[feature]["window"]))
        )
        rr = np.stack([numeric[f"{key}_real_{feature}"].mean(0) for key in keys])
        mm = np.stack([numeric[f"{key}_model_{feature}"].mean(0) for key in keys])
        rr = np.stack([rr[recordings == rec].sum(0) for rec in unique])
        mm = np.stack([mm[recordings == rec].sum(0) for rec in unique])
        result["bootstrap"][feature] = (
            0.5
            * (
                hellinger(rr[draws].sum((1, 2)), mm[draws].sum((1, 2)))
                + hellinger(rr[draws].sum(1), mm[draws].sum(1)).mean(1)
            )
        ).tolist()
    result["components"].update(
        lane_rate=float(point["rate_error"]), lane_execution=float(point["process"])
    )
    result["bootstrap"].update(
        lane_rate=bootstrap_lateral["rate_error"].tolist(),
        lane_execution=bootstrap_lateral["process"].tolist(),
    )
    return result


def save_candidate(folder, name, population, mobil, kind, coordinates=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"{name}.npz", **compact_population(population))
    save_json(
        {
            "kind": kind,
            "coordinates": (
                None if coordinates is None else np.asarray(coordinates).tolist()
            ),
            "mobil": mobil,
        },
        folder / f"{name}.json",
    )


def load_candidate(folder, name):
    with np.load(Path(folder) / f"{name}.npz") as saved:
        population = dict(saved)
    return population, load_json(Path(folder) / f"{name}.json")["mobil"]


def correction_bounds(settings):
    ranges = settings["residual"]
    return np.array(
        [
            ranges["mean_strength_range"],
            ranges["noise_strength_range"],
            np.log(ranges["time_scale_range"]),
            ranges["lane_strength_range"],
            ranges["direction_bias_range"],
            ranges["log_duration_range"],
            ranges["shape_shift_range"],
            ranges["shape_shift_range"],
        ]
    )


def population_bounds(settings):
    ranges = settings["population"]
    return np.array(
        [ranges["log_mean_shift_range"]] * 5
        + [
            ranges["std_scale_range"],
            ranges["lane_mean_shift_range"],
            ranges["lane_std_scale_range"],
        ]
    )


def correction_coordinates(fit):
    correction = fit["correction"]
    longitudinal, lane, trajectory = (
        correction["longitudinal"],
        correction["lane"],
        correction["trajectory"],
    )
    return np.array(
        [
            longitudinal["mean_strength"],
            longitudinal["noise_strength"],
            np.log(longitudinal["autoregression"]["time_scale"]),
            lane["strength"],
            lane["direction_bias"],
            trajectory["log_duration_scale"],
            *trajectory["shape_shift"],
        ]
    )


def corrected_fit(local, coordinates):
    fit = deepcopy(local)
    correction = fit["correction"]
    longitudinal = correction["longitudinal"]
    longitudinal["mean_strength"], longitudinal["noise_strength"] = map(
        float, coordinates[:2]
    )
    longitudinal["autoregression"] = autoregression_parameters(
        longitudinal["autoregression"]["source_coefficients"],
        time_scale=np.exp(coordinates[2]),
    )
    correction["lane"].update(
        strength=float(coordinates[3]), direction_bias=float(coordinates[4])
    )
    correction["lane"].pop("logit_bias", None)
    correction["trajectory"] = {
        "log_duration_scale": float(coordinates[5]),
        "shape_shift": coordinates[6:].tolist(),
    }
    return fit


def shifted_population(population, coordinates):
    result = {key: np.asarray(value).copy() for key, value in population.items()}
    result["log_mean"] = population["log_mean"] + coordinates[:5]
    result["log_cov"] = population["log_cov"] * coordinates[5] ** 2
    if "driver_log_parameters" in population:
        result["driver_log_parameters"] = result["log_mean"] + coordinates[5] * (
            population["driver_log_parameters"] - population["log_mean"]
        )
        result["driver_log_parameters"] = np.log(
            np.clip(np.exp(result["driver_log_parameters"]), LOWER, UPPER)
        )
    return result


def state_subset(data, fraction, lane=False):
    group = (
        data["label"]
        if lane
        else data["free"].astype(int) + 2 * data["changing"].astype(int)
    )
    order = np.lexsort((data["frame"], data["vehicle"], group, data["recording"]))
    codes = np.column_stack((data["recording"][order], group[order]))
    starts = np.r_[0, np.flatnonzero(np.any(codes[1:] != codes[:-1], axis=1)) + 1]
    stops = np.r_[starts[1:], len(order)]
    chosen = np.concatenate(
        [
            order[start : stop : round(1 / fraction)]
            for start, stop in zip(starts, stops)
        ]
    )
    return {name: values[chosen] for name, values in data.items()}


def continuous_subset(data, fraction):
    """Sample whole vehicle histories within recording/behavior strata for temporal fitting."""
    identity, index = np.unique(
        np.column_stack((data["recording"], data["vehicle"])),
        axis=0,
        return_inverse=True,
    )
    counts = np.bincount(index)
    free = np.bincount(index, weights=data["free"]) / counts
    changing = np.bincount(index, weights=data["changing"]) / counts
    group = (free > 0.5).astype(int) + 2 * (changing > 0)
    order = np.lexsort((identity[:, 1], group, identity[:, 0]))
    codes = np.column_stack((identity[order, 0], group[order]))
    starts = np.r_[0, np.flatnonzero(np.any(codes[1:] != codes[:-1], axis=1)) + 1]
    stops = np.r_[starts[1:], len(order)]
    chosen = np.concatenate(
        [
            order[start : stop : round(1 / fraction)]
            for start, stop in zip(starts, stops)
        ]
    )
    keep = np.isin(index, chosen)
    return {name: value[keep] for name, value in data.items()}


def local_score(report):
    return float(
        np.mean(
            [
                np.mean([row["Hellinger"] for row in report["acceleration"]]),
                report["lane"]["Brier"] / 2,
                np.mean(report["lane"]["acceptance_Hellinger"]),
            ]
        )
    )


MAIN_COMPONENTS = ("speed", "joint", "lane_rate", "lane_execution")


def objective(closed, local, coordinates, origin, bounds, weights):
    change = (
        0
        if coordinates is None
        else np.square((coordinates - origin) / np.diff(bounds, axis=1)[:, 0]).mean()
    )
    return float(
        sum(
            weights[name] * closed["components"][name]
            for name in (*MAIN_COMPONENTS, "acceleration")
        )
        + weights["local"] * local_score(local)
        + weights["change"] * change
    )


def paired_closed_acceptable(candidate, baseline, runs):
    checks = {}
    for name in MAIN_COMPONENTS:
        delta = np.asarray(candidate["bootstrap"][name]) - baseline["bootstrap"][name]
        checks[name] = bool(
            candidate["components"][name]
            <= baseline["components"][name] + delta.std(ddof=1)
        )
    for name in (
        "contact_exposure_ratio",
        "boundary_exposure_ratio",
        "road_collision_run_fraction",
    ):
        delta = np.asarray(candidate["bootstrap"][name]) - baseline["bootstrap"][name]
        checks[name] = bool(
            candidate["exposure"][name]
            <= baseline["exposure"][name] + delta.std(ddof=1)
        )
    checks["runs_complete"] = all(run["status"] == "complete" for run in runs)
    return all(checks.values()), checks


def local_guard(candidate, baseline, numeric, baseline_numeric):
    draws = np.random.default_rng(SEED).integers(
        len(numeric["lane_recordings"]), size=(2000, len(numeric["lane_recordings"]))
    )
    diff = (
        numeric["lane_Brier_recording_sums"]
        - baseline_numeric["lane_Brier_recording_sums"]
    )
    counts = numeric["lane_Brier_recording_counts"]
    se = np.std(diff[draws].sum(1) / counts[draws].sum(1), ddof=1)
    checks = {
        "Brier": bool(candidate["lane"]["Brier"] <= baseline["lane"]["Brier"] + se)
    }
    for index, name in enumerate(("target_rear_gap", "target_front_gap")):
        real, first, prior = (
            numeric[name + "_real_recording"],
            numeric[name + "_model_recording"],
            baseline_numeric[name + "_model_recording"],
        )
        delta = hellinger(real[draws].sum(1), first[draws].sum(1)) - hellinger(
            real[draws].sum(1), prior[draws].sum(1)
        )
        checks[name] = bool(
            candidate["lane"]["acceptance_Hellinger"][index]
            <= baseline["lane"]["acceptance_Hellinger"][index] + delta.std(ddof=1)
        )
    return all(checks.values()), checks


def paired_objective_se(first, second, weights):
    difference = sum(
        weights[name]
        * (
            np.asarray(first["closed"]["bootstrap"][name])
            - np.asarray(second["closed"]["bootstrap"][name])
        )
        for name in (*MAIN_COMPONENTS, "acceleration")
    )
    return float(difference.std(ddof=1))


def refinement_points(center, bounds, round_index, count):
    half_width = np.diff(bounds, axis=1)[:, 0] / (4 * 2**round_index)
    lower, upper = (
        np.maximum(bounds[:, 0], center - half_width),
        np.minimum(bounds[:, 1], center + half_width),
    )
    points = qmc.Sobol(len(center), scramble=True, seed=SEED + 10 + round_index).random(
        count
    )
    points = lower + points * (upper - lower)
    points[0] = center
    if len(center) == 16:
        points[0, 8:10] = 0
    return points


def run_search(output, population, mobil, settings):
    from .calibration import concatenate_groups

    output = Path(output)
    search = folder = output / "policy_calibration"
    search.mkdir(parents=True, exist_ok=True)
    completed = search / "selection.json"
    if completed.exists():
        selection = load_json(completed)
        expected_bounds = np.vstack(
            (population_bounds(settings), correction_bounds(settings))
        ).tolist()
        saved_protocol = selection["selection_protocol"]
        if (
            saved_protocol["budget"] != settings["search"]
            or saved_protocol["bounds"] != expected_bounds
            or saved_protocol["data"]
            != artifact_identity(DRIVER_DATA_DIR / "manifest.json")
        ):
            raise ValueError(
                f"Frozen {output.name} protocol differs; archive this phase before rerunning"
            )
        return *load_candidate(folder, selection["selected"]), selection
    raw = load_json(DRIVER_DATA_DIR / "manifest.json")
    cohorts = load_cohorts()
    settings = {
        **settings,
        "lateral_protocol": load_json(DRIVER_DATA_DIR / "lateral_protocol.json"),
    }
    budget, weights = settings["search"], settings["search"]["objective_weights"]
    round_sizes = (budget["initial_candidates"],) + (
        budget["local_candidates_per_round"],
    ) * budget["refinement_rounds"]
    if sum(round_sizes) != budget["candidates_per_arm"]:
        raise ValueError("candidate round sizes must sum to candidates_per_arm")
    fit_splits = ("train",)
    recordings = sorted(
        rec for split in fit_splits for rec in raw["split_recordings"][split]
    )
    train_keys = cohort_selection(
        cohorts,
        recordings,
        budget["train_scenes"],
    )
    val_keys = cohort_selection(
        cohorts, raw["split_recordings"]["val"], budget["val_scenes"]
    )
    full_states = concatenate_groups(load_response(splits=fit_splits))
    full_lanes = concatenate_groups(
        load_lane_opportunities(DRIVER_DATA_DIR, splits=fit_splits)
    )
    train = state_subset(full_states, budget["local_fraction"])
    learning_states = continuous_subset(full_states, budget["local_fraction"])
    lanes = state_subset(full_lanes, budget["local_fraction"], lane=True)
    validation = load_response(splits=("val",))["val"]
    lane_validation = load_lane_opportunities(DRIVER_DATA_DIR, splits=("val",))["val"]
    reference = dict(np.load(output / "policy_calibration/reference.npz"))
    baseline = deepcopy(
        load_json(output / "policy_calibration/reference_mobil.json")["mobil"]
    )
    baseline["lane_model"] = load_json(output / "policy_calibration/lane_model.json")
    p_bounds, c_bounds = population_bounds(settings), correction_bounds(settings)
    bounds = np.vstack((p_bounds, c_bounds))
    origin = np.r_[np.zeros(5), 1, 0, 1, correction_coordinates(mobil)]
    records, arms = {}, {}
    protocol = {
        "phase": "calibration",
        "fit_recordings": recordings,
        "train_cohorts": train_keys,
        "val_cohorts": val_keys,
        "seed": SEED,
        "budget": budget,
        "bounds": bounds.tolist(),
        "origin": origin.tolist(),
        "weights": weights,
        "data": artifact_identity(DRIVER_DATA_DIR / "manifest.json"),
        "learning_state_sampling": "whole physical vehicle histories within recording/behavior strata",
        "learning_state_rows": len(learning_states["frame"]),
        "conditional_search_rows": len(train["frame"]),
        "correction_protocol": "leader_history_only_single_residual_process",
        "statistical_role": "Train_fit_Val_selection",
    }
    save_json(protocol, search / "protocol.json")
    ridges = mobil["correction"]["ridge"]

    def build(vector, kind, full=False):
        candidate_population = shifted_population(population, vector[:8])
        fit = deepcopy(mobil)
        fit["lane_style"]["mean"] += float(vector[6])
        fit["lane_style"]["std"] *= float(vector[7])
        if kind == "joint":
            source, source_lanes = (
                (full_states, full_lanes) if full else (learning_states, lanes)
            )
            corrected = fit_correction(
                source, source_lanes, candidate_population, fit, ridges["longitudinal"]
            )
            if ridges["lane"] != ridges["longitudinal"]:
                corrected["lane"] = fit_correction(
                    source, source_lanes, candidate_population, fit, ridges["lane"]
                )["lane"]
            corrected["ridge"] = ridges
            fit["correction"] = corrected
        fit = corrected_fit(fit, vector[8:])
        return candidate_population, fit

    with ProcessPoolExecutor(
        max_workers=settings["workers"], mp_context=get_context("spawn")
    ) as executor:

        def evaluate(name, kind, vector=None, source=None, long_validation=False):
            destination = (
                search / f"{name}{'_validation' if long_validation else '_result'}.json"
            )
            numeric_path = destination.with_suffix(".npz")
            if destination.exists():
                record = load_json(destination)
                return record, dict(np.load(numeric_path))
            if source is not None:
                candidate_population, fit = source
            elif long_validation:
                candidate_population, fit = build(vector, kind, full=True)
                save_candidate(folder, name, candidate_population, fit, kind, vector)
            else:
                candidate_population, fit = build(vector, kind)
                save_candidate(folder, name, candidate_population, fit, kind, vector)
            state, lane = (
                (validation, lane_validation) if long_validation else (train, lanes)
            )
            natural, numeric = conditional_metrics(
                state,
                lane,
                candidate_population,
                fit,
                samples=settings["conditional_samples"],
                bootstrap=settings["bootstrap_samples"],
            )
            closed, _, runs = run_cohorts(
                search,
                name + ("_val" if long_validation else "_train"),
                val_keys if long_validation else train_keys,
                cohorts,
                candidate_population,
                fit,
                settings,
                budget["val_repeats"] if long_validation else budget["train_repeats"],
                (
                    budget["val_duration_s"]
                    if long_validation
                    else budget["train_duration_s"]
                ),
                executor=executor,
            )
            record = {
                "candidate": name,
                "kind": kind,
                "coordinates": None if vector is None else vector.tolist(),
                "local": natural,
                "closed": closed,
                "runs": runs,
                "score": objective(closed, natural, vector, origin, bounds, weights),
            }
            np.savez_compressed(
                numeric_path,
                **{
                    key: value
                    for key, value in numeric.items()
                    if key != "lane_probability"
                },
            )
            save_json(record, destination)
            print(
                output.name,
                name,
                "score",
                record["score"],
                "components",
                closed["components"],
                flush=True,
            )
            return record, numeric

        for name, source in (
            ("reference", (reference, baseline)),
            ("local", (population, mobil)),
        ):
            save_candidate(
                folder, name, *source, name, None if name == "reference" else origin
            )
            records[name], _ = evaluate(
                name, name, None if name == "reference" else origin, source
            )
        modes = ("population", "joint")
        for kind in modes:
            dimension = 8 if kind == "population" else 16
            center = origin.copy()
            if kind == "joint" and "population" in arms:
                best = min(
                    arms["population"]["candidates"],
                    key=lambda name: records[name]["score"],
                )
                center = np.asarray(records[best]["coordinates"])
            points = qmc.Sobol(dimension, scramble=True, seed=SEED).random(
                round_sizes[0]
            )
            points = (
                bounds[:dimension, 0]
                + points * np.diff(bounds[:dimension], axis=1).ravel()
            )
            points[0] = center[:dimension]
            names = []
            for round_index, count in enumerate(round_sizes):
                if round_index:
                    best = min(names, key=lambda name: records[name]["score"])
                    center = np.asarray(records[best]["coordinates"])
                    points = refinement_points(
                        center[:dimension], bounds[:dimension], round_index - 1, count
                    )
                for index, point in enumerate(points):
                    vector = origin.copy()
                    vector[:dimension] = point
                    if dimension == 16 and index < 2:
                        vector[8:10] = 0
                    name = f"{kind}_{round_index}_{index:02d}"
                    records[name], _ = evaluate(name, kind, vector)
                    names.append(name)
            ranked = sorted(names, key=lambda name: records[name]["score"])
            arms[kind] = {
                "candidates": names,
                "evaluations": len(names),
                "shortlist": ranked[:2],
            }
        shortlist = [
            "local",
            *[name for group in arms.values() for name in group["shortlist"]],
        ][: budget["finalists"]]
        validations, arrays = {}, {}
        for name in ["reference", *shortlist]:
            record = records[name]
            vector = (
                None
                if record["coordinates"] is None
                else np.asarray(record["coordinates"])
            )
            source = (
                (reference, baseline)
                if name == "reference"
                else ((population, mobil) if name == "local" else None)
            )
            validations[name], arrays[name] = evaluate(
                name, record["kind"], vector, source, long_validation=True
            )
        baseline_record = validations["reference"]
        for name, record in validations.items():
            closed_ok, checks = paired_closed_acceptable(
                record["closed"], baseline_record["closed"], record["runs"]
            )
            local_ok, local_checks = local_guard(
                record["local"],
                baseline_record["local"],
                arrays[name],
                arrays["reference"],
            )
            record.update(
                accepted=bool(closed_ok and local_ok), guards={**checks, **local_checks}
            )
            save_json(record, search / f"{name}_validation.json")
        available = [item for item in validations.values() if item["accepted"]]
        best = min(available, key=lambda item: item["score"])
        tied = [
            item
            for item in available
            if item["score"] <= best["score"] + paired_objective_se(item, best, weights)
        ]
        complexity = {"reference": 0, "local": 1, "population": 2, "joint": 3}
        selected = min(
            tied, key=lambda item: (complexity[item["kind"]], item["score"])
        )["candidate"]
        selection = {
            "status": "complete",
            "publishable": selected != "reference",
            "selected": selected,
            "fit_split": "train",
            "selection_split": "val",
            "Test_used_for_selection": False,
            "correction_protocol": protocol["correction_protocol"],
            "phase": protocol["phase"],
            "arm_evaluations": {
                kind: item["evaluations"] for kind, item in arms.items()
            },
            "candidates_evaluated": sum(
                len(group["candidates"]) for group in arms.values()
            ),
            "long_validation_candidates": len(shortlist),
            "candidate_results": list(records.values()),
            "finalists": list(validations.values()),
            "selection_protocol": protocol,
        }
        save_json(selection, completed)
    return *load_candidate(folder, selected), selection

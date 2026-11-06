"""Paired road rollouts and physical traffic statistics for frozen NPC policies."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from process_highD.src.io_utils import load_config, resolve_data_path

from ..core.utils import load_json, save_json
from ..stochastic_drivers.raw_data import (
    BEHAVIOR_PROTOCOL,
    DRIVER_DATA_DIR,
)
from .calibration import compact_population
from .lateral import (
    EVENT_FIELDS,
    event_histograms,
    lane_events,
    lateral_distances,
)
from .metrics import FEATURE_EDGES, hellinger
from .residual import temporal_statistics
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

"""Shared observed lane events and frozen physical distribution statistics."""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

from .metrics import DT

EVENT_FIELDS = (
    "recording",
    "vehicle",
    "repeat",
    "start_frame",
    "cross_frame",
    "end_frame",
    "direction",
    "status",
    "completed",
    "observed_start",
    "observed_complete",
    "duration",
    "peak_lateral_speed",
    "peak_lateral_acceleration",
    "front_gap",
    "rear_gap",
    "front_relative_speed",
    "rear_relative_speed",
    "front_present",
    "rear_present",
)
PROCESS_FEATURES = ("duration", "peak_lateral_speed", "peak_lateral_acceleration")


def distance(left, right):
    """Hellinger with explicit empty-event semantics."""
    left, right = np.asarray(left, float), np.asarray(right, float)
    lp, rp = left.sum(-1, keepdims=True), right.sum(-1, keepdims=True)
    p = np.divide(left, lp, out=np.zeros_like(left), where=lp > 0)
    q = np.divide(right, rp, out=np.zeros_like(right), where=rp > 0)
    value = np.sqrt(0.5 * np.square(np.sqrt(p) - np.sqrt(q)).sum(-1))
    return np.where((lp[..., 0] == 0) ^ (rp[..., 0] == 0), 1.0, value)


def motion_from_positions(states):
    values = np.asarray(states, float).copy()
    if len(values) >= 25:
        for derivative, columns in ((1, slice(2, 4)), (2, slice(4, 6))):
            values[..., columns] = savgol_filter(
                values[..., :2],
                25,
                3,
                deriv=derivative,
                delta=DT,
                axis=0,
                mode="interp",
            )
    return values


def start_neighbors(states, valid, dimensions, centers, actor, target):
    lane = np.abs(states[:, 1, None] - centers).argmin(-1)
    dx = states[:, 0] - states[actor, 0]
    eligible = valid & (lane == target)
    eligible[actor] = False
    result = {}
    for name, mask, sign in (
        ("front", eligible & (dx > 0), 1),
        ("rear", eligible & (dx <= 0), -1),
    ):
        candidates = np.flatnonzero(mask)
        present = bool(len(candidates))
        other = candidates[np.argmin(np.abs(dx[candidates]))] if present else actor
        gap = sign * dx[other] - 0.5 * (dimensions[actor, 0] + dimensions[other, 0])
        result[name + "_present"] = present
        result[name + "_gap"] = float(gap) if present else np.nan
        result[name + "_relative_speed"] = (
            float(states[other, 2] - states[actor, 2]) if present else np.nan
        )
    return result


def lane_events(
    states,
    valid,
    centers,
    dimensions,
    *,
    recording=0,
    repeat=0,
    vehicle_ids=None,
    frame_origin=0,
):
    """Extract actual motion, including observations censored by entry or exit.

    Future positions define evaluation labels only. No event label is a driver input.
    """
    states, valid = np.asarray(states), np.asarray(valid, bool)
    centers = np.asarray(centers)
    ids = np.arange(states.shape[1]) if vehicle_ids is None else vehicle_ids
    completed = np.zeros(valid.shape, bool)
    events, curves = [], []
    for actor in range(states.shape[1]):
        present = np.flatnonzero(valid[:, actor])
        for part in np.split(present, np.flatnonzero(np.diff(present) != 1) + 1):
            if len(part) < 25:
                continue
            motion = motion_from_positions(states[part, actor])
            y, vy, ay = motion[:, 1], motion[:, 3], motion[:, 5]
            active = uniform_filter1d(np.abs(vy), 5, mode="nearest") > 0.15
            edges = np.diff(np.r_[False, active, False].astype(np.int8))
            starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
            lane = np.abs(y[:, None] - centers).argmin(-1)
            crosses = np.flatnonzero(np.diff(lane) != 0) + 1
            for start, stop in zip(starts, stops):
                nearby = crosses[(crosses >= start - 10) & (crosses <= stop + 10)]
                # Multiple crossings remain abnormal events, never normal completed changes.
                crossing = int(nearby[0]) if len(nearby) else -1
                direction = int(np.sign(y[stop] - y[start]))
                if direction == 0:
                    direction = int(np.sign(vy[start])) or 1
                observed_start = bool(
                    start >= 12 and abs(y[start] - centers[lane[start]]) < 0.7
                )
                source = int(lane[max(0, start - 1)])
                if not observed_start and start < 12:
                    width = np.median(np.diff(centers)) if len(centers) > 1 else 4.0
                    source = int(
                        np.abs(centers - (y[start] - direction * width / 2)).argmin()
                    )
                target = source + direction
                adjacent = (
                    0 <= target < len(centers)
                    and len(nearby) == 1
                    and abs(int(lane[crossing]) - int(lane[crossing - 1])) == 1
                )
                settles = bool(
                    adjacent
                    and stop + 5 < len(part)
                    and np.all(np.abs(y[stop : stop + 6] - centers[target]) < 0.7)
                )
                observed_complete = bool(
                    observed_start and settles and stop + 12 < len(part)
                )
                if settles:
                    completed[part[stop], actor] = True
                    status = "complete" if observed_start else "initial_continuation"
                elif stop >= len(part) - 12:
                    status = (
                        "end_censored"
                        if part[-1] == len(states) - 1
                        else "exit_censored"
                    )
                elif crossing < 0:
                    status = "cancelled"
                else:
                    status = "incomplete"
                tick = int(part[start])
                row = {
                    "recording": int(recording),
                    "vehicle": int(ids[actor]),
                    "repeat": int(repeat),
                    "start_frame": tick + frame_origin,
                    "cross_frame": (
                        int(part[crossing]) + frame_origin if crossing >= 0 else -1
                    ),
                    "end_frame": int(part[stop]) + frame_origin,
                    "direction": direction,
                    "status": status,
                    "completed": settles,
                    "observed_start": observed_start,
                    "observed_complete": observed_complete,
                    "duration": float((stop - start + 1) * DT),
                    "peak_lateral_speed": float(np.abs(vy[start : stop + 1]).max()),
                    "peak_lateral_acceleration": float(
                        np.abs(ay[start : stop + 1]).max()
                    ),
                }
                row.update(
                    start_neighbors(
                        states[tick],
                        valid[tick],
                        dimensions,
                        centers,
                        actor,
                        int(np.clip(target, 0, len(centers) - 1)),
                    )
                )
                events.append(row)
                if observed_complete and stop > start:
                    phase = np.linspace(0, 1, stop - start + 1)
                    width = abs(centers[target] - centers[source])
                    curves.append(
                        np.stack(
                            [
                                np.interp(
                                    np.linspace(0, 1, 101),
                                    phase,
                                    direction
                                    * (y[start : stop + 1] - centers[source])
                                    / width,
                                ),
                                np.interp(
                                    np.linspace(0, 1, 101),
                                    phase,
                                    direction * vy[start : stop + 1],
                                ),
                            ]
                        )
                    )
    return events, completed, np.asarray(curves).reshape(-1, 2, 101)


def quantile_edges(values, bins):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("Train has no observed events for lateral distribution bins")
    interior = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    return np.r_[-np.inf, interior[1:-1], np.inf].tolist()


def lateral_protocol(events):
    complete = events[events.observed_complete.astype(bool)]
    starts = events[events.observed_start.astype(bool) & (events.cross_frame >= 0)]
    return {
        "fit_split": "train",
        "derivative_window": 25,
        "motion_threshold_mps": 0.15,
        "state_features": ["changing", "progress", "lateral_speed"],
        "edges": {
            name: quantile_edges(complete[name], 20) for name in PROCESS_FEATURES
        },
        "joint_edges": {
            name: quantile_edges(starts[name], 8)
            for name in (
                "front_gap",
                "rear_gap",
                "front_relative_speed",
                "rear_relative_speed",
            )
        }
        | {
            name: quantile_edges(complete[name], 8)
            for name in ("duration", "peak_lateral_speed")
        },
        "no_neighbor": "extra category after finite gap/relative-speed bins",
    }


def event_histograms(events, protocol):
    result = {
        name: np.zeros(len(edges) - 1) for name, edges in protocol["edges"].items()
    }
    pairs = {
        "start_gap_joint": ("front_gap", "rear_gap"),
        "start_relative_speed_joint": ("front_relative_speed", "rear_relative_speed"),
        "duration_speed_joint": ("duration", "peak_lateral_speed"),
    }
    for name, fields in pairs.items():
        extra = int(name.startswith("start"))
        result[name] = np.zeros(
            tuple(len(protocol["joint_edges"][field]) - 1 + extra for field in fields)
        )
    result["direction"] = np.zeros(2)
    result["event_counts"] = np.zeros(7)
    for row in events:
        result["event_counts"] += [
            1,
            row["completed"],
            row["observed_start"],
            row["observed_complete"],
            row["status"] == "initial_continuation",
            row["status"] == "exit_censored",
            row["status"] == "end_censored",
        ]
        if row["completed"]:
            result["direction"][int(row["direction"] > 0)] += 1
        if row["observed_complete"]:
            for name, edges in protocol["edges"].items():
                result[name] += np.histogram([row[name]], edges)[0]
        for name, fields in pairs.items():
            eligible = (
                row["observed_start"] and row["cross_frame"] >= 0
                if name.startswith("start")
                else row["observed_complete"]
            )
            if not eligible:
                continue
            indices = []
            for field in fields:
                edges = protocol["joint_edges"][field]
                value = row[field]
                index = (
                    len(edges) - 1
                    if not np.isfinite(value)
                    else int(
                        np.clip(
                            np.searchsorted(edges, value, side="right") - 1,
                            0,
                            len(edges) - 2,
                        )
                    )
                )
                indices.append(index)
            result[name][tuple(indices)] += 1
    return result


def lateral_distances(model, real, model_km, real_km):
    mrate = 1000 * model["direction"].sum(-1) / np.maximum(model_km, 1e-12)
    rrate = 1000 * real["direction"].sum(-1) / np.maximum(real_km, 1e-12)
    rate = np.divide(
        abs(mrate - rrate),
        mrate + rrate,
        out=np.zeros_like(np.asarray(mrate)),
        where=mrate + rrate > 0,
    )
    result = {"rate_error": rate, "model_rate": mrate, "real_rate": rrate}
    for name in (
        "direction",
        *PROCESS_FEATURES,
        "start_gap_joint",
        "start_relative_speed_joint",
        "duration_speed_joint",
    ):
        shape = model[name].shape
        trailing = 2 if name.endswith("joint") else 1
        result[name] = distance(
            np.asarray(model[name]).reshape(*shape[:-trailing], -1),
            np.asarray(real[name]).reshape(*shape[:-trailing], -1),
        )
    result["start"] = (
        result["start_gap_joint"] + result["start_relative_speed_joint"]
    ) / 2
    result["process"] = np.mean(
        [result[name] for name in (*PROCESS_FEATURES, "duration_speed_joint")], axis=0
    )
    result["score"] = (
        rate + result["direction"] + result["start"] + result["process"]
    ) / 4
    return result


def conditional_lane_statistics(data_dir, split, mobil, samples=16, fraction=1.0):
    """Initial-state-conditioned lane execution; no target future is a control input."""
    from pathlib import Path

    import torch

    from ..core.dynamics import (
        DynamicsConfig,
        KinematicTrafficDynamics,
    )
    from ..core.utils import load_json
    from ..stochastic_drivers.lane_trajectory import (
        LearnedLaneTrajectory,
        limit_realized_lateral_acceleration,
    )
    from ..stochastic_drivers.raw_data import load_lane_events

    data_dir = Path(data_dir)
    events, observations = load_lane_events(data_dir, splits=(split,))[split]
    if fraction < 1:
        chosen = np.concatenate(
            [
                group.index.to_numpy()[:: round(1 / fraction)]
                for _, group in events.groupby("recording_id")
            ]
        )
        events = events.loc[chosen]
    protocol = load_json(data_dir / "lateral_protocol.json")
    recordings = np.asarray(sorted(events.recording_id.unique()))
    model = LearnedLaneTrajectory(
        correction=mobil.get("correction", {}).get("trajectory"),
        model=mobil.get("lane_model"),
    )
    dynamics = KinematicTrafficDynamics(DynamicsConfig(speed_max_mps=100.0))
    real_hist, model_hist = {}, {}
    rows, real_rows = [], []
    for recording in recordings:
        rec_events = events[events.recording_id == recording]
        trajectories, initial, dimensions, targets, lengths, ids = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for event in rec_events.itertuples():
            data = observations["states"][
                observations["offsets"][event.trajectory_index] : observations[
                    "offsets"
                ][event.trajectory_index + 1]
            ]
            physical = np.zeros((len(data), 6), np.float32)
            physical[:, 0] = np.r_[0, np.cumsum(data[:-1, 1]) * DT]
            physical[:, 1] = data[:, 0]
            physical[:, 2:4] = data[:, 1:3]
            physical[:, 4] = np.gradient(data[:, 1], DT)
            physical[:, 5] = data[:, 3]
            trajectories.append(physical)
            initial.append(physical[25])
            initial[-1][4] = (data[25, 1] - data[24, 1]) / DT
            dimensions.append([event.length_m, event.width_m])
            targets.append(event.target_y_m - event.source_y_m)
            lengths.append(len(data))
            ids.append(event.vehicle_id)
            centers = np.array(sorted([-abs(targets[-1]), 0, abs(targets[-1])]))
            extracted, _, _ = lane_events(
                physical[:, None],
                np.ones((len(data), 1), bool),
                centers,
                np.array([dimensions[-1]]),
                recording=int(recording),
                vehicle_ids=[event.vehicle_id],
            )
            real_rows.extend(extracted)
        real_hist[int(recording)] = event_histograms(
            [row for row in real_rows if row["recording"] == recording], protocol
        )
        generated_rows = []
        for repeat in range(samples):
            state = torch.tensor(np.array(initial), dtype=torch.float32)
            sizes = torch.tensor(dimensions)
            target = torch.tensor(targets)
            noise = torch.tensor(
                np.stack(
                    [
                        np.random.default_rng(
                            [
                                20261012,
                                int(recording),
                                int(event.vehicle_id),
                                int(event.cross_frame),
                                repeat,
                                4,
                            ]
                        ).standard_normal(2)
                        for event in rec_events.itertuples()
                    ]
                ),
                dtype=torch.float32,
            )
            coefficients, origin, distance_x = model.plan(state, sizes, target, noise)
            trace = np.zeros((max(lengths), len(state), 6), np.float32)
            present = np.arange(max(lengths))[:, None] < np.array(lengths)[None]
            for actor, trajectory in enumerate(trajectories):
                trace[:25, actor] = trajectory[:25]
            trace[25] = state.numpy()
            for tick in range(26, max(lengths)):
                acceleration = torch.zeros(len(state))
                yaw = model.track(state, coefficients, origin, distance_x)
                yaw = limit_realized_lateral_acceleration(
                    state,
                    yaw,
                    acceleration,
                    acceleration_limit=model.acceleration_limit,
                    jerk_limit=model.jerk_limit,
                )
                state = dynamics.step(
                    state,
                    torch.stack((acceleration, yaw), -1),
                    torch.tensor(present[tick]),
                    DT,
                )
                trace[tick] = state.numpy()
            for actor, length in enumerate(lengths):
                centers = np.array(
                    sorted([-abs(targets[actor]), 0, abs(targets[actor])])
                )
                extracted, _, _ = lane_events(
                    trace[:length, actor : actor + 1],
                    present[:length, actor : actor + 1],
                    centers,
                    np.array([dimensions[actor]]),
                    recording=int(recording),
                    repeat=repeat,
                    vehicle_ids=[ids[actor]],
                )
                generated_rows.extend(extracted)
        rows.extend(generated_rows)
        histogram = event_histograms(generated_rows, protocol)
        model_hist[int(recording)] = {
            key: value / samples for key, value in histogram.items()
        }
    numeric = {
        f"real_{name}": np.stack([real_hist[int(rec)][name] for rec in recordings])
        for name in real_hist[int(recordings[0])]
    }
    numeric.update(
        {
            f"model_{name}": np.stack(
                [model_hist[int(rec)][name] for rec in recordings]
            )
            for name in model_hist[int(recordings[0])]
        }
    )
    numeric["recordings"] = recordings
    report = {}
    draws = np.random.default_rng(20261012).integers(
        len(recordings), size=(2000, len(recordings))
    )
    for name in (*PROCESS_FEATURES, "duration_speed_joint"):
        real, simulated = numeric["real_" + name], numeric["model_" + name]
        flat_r, flat_m = (
            real.reshape(len(real), -1),
            simulated.reshape(len(simulated), -1),
        )
        boot = distance(flat_r[draws].sum(1), flat_m[draws].sum(1))
        report[name] = {
            "Hellinger": float(distance(flat_r.sum(0), flat_m.sum(0))),
            "SE": float(boot.std(ddof=1)),
            "CI95": np.quantile(boot, [0.025, 0.975]).tolist(),
        }
    report.update(
        {
            "events": len(events),
            "samples": samples,
            "real_observed_complete": int(
                sum(row["observed_complete"] for row in real_rows)
            ),
            "model_observed_complete_per_repeat": sum(
                row["observed_complete"] for row in rows
            )
            / samples,
            "future_target_trajectory_used_for_control": False,
            "protocol": "initial observed history and selected lane; autonomous tracking with zero longitudinal acceleration",
        }
    )
    return report, numeric

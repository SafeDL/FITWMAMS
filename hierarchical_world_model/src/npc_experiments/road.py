"""Autonomous NPC experiments on the actual HighwayEnv road backend."""

from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.ndimage import minimum_filter1d
from scipy.signal import savgol_filter

from process_highD.src.natural_segments import _build_vehicle_cache
from process_highD.src.preprocess import prepare_recording

from ..highway import (
    UnicycleDynamicsVehicleMixin,
    _highway_classes,
)
from ..influence_graph import build_dynamic_interaction_relations
from ..stochastic_drivers.driver_policy import (
    anticipated_entry_features,
    causal_maneuver_features,
    corrected_acceleration,
    exclusive_lane_probabilities,
    physical_leader_observation,
    response_history_features,
    sample_lane_choice,
)
from ..stochastic_drivers.idm_population import sample_theta
from ..stochastic_drivers.lane_choice import lane_choice_features
from ..stochastic_drivers.lane_trajectory import (
    LearnedLaneTrajectory,
    limit_realized_lateral_acceleration,
)
from ..stochastic_drivers.mobil_response import evaluate_mobil_candidates
from .metrics import (
    DT,
    FEATURE_EDGES,
    JOINT_EDGES,
)

DURATION_S = 120
START_S = 60
WINDOW_S = 10
SEED = 20261011


def road_cohorts(
    raw_dir, config, recordings, *, start_s=START_S, duration_s=DURATION_S
):
    """Retain both directions, true maps/classes and measured boundary demand."""
    cohorts, free = {}, []
    first_frame = int(start_s / DT) + 1
    ticks = int(duration_s / DT) + 1
    for recording_id in recordings:
        recording = prepare_recording(Path(raw_dir), int(recording_id), config)
        vehicles = _build_vehicle_cache(recording)
        for direction in (1, 2):
            sign = 1 if direction == 1 else -1
            markings = recording.recording_meta[
                "upperLaneMarkings" if direction == 1 else "lowerLaneMarkings"
            ]
            centers = np.sort(sign * (markings[:-1] + markings[1:]) / 2)
            widths = np.diff(markings)
            if sign == -1:
                widths = widths[::-1]
            ids = [
                key
                for key, value in vehicles.items()
                if value["direction"] == direction
            ]
            tracks = recording.tracks.loc[ids]
            # Sight distances identify the camera geometry, not target traffic statistics.
            low = float(np.median(tracks.x - tracks.backSightDistance)) + 40
            high = float(np.median(tracks.x + tracks.frontSightDistance)) - 40
            blocks, active, dimensions, included = [], [], [], []
            entry_ticks, entry_states, entry_histories, entry_history_valid = (
                [],
                [],
                [],
                [],
            )
            for vehicle_id in ids:
                vehicle = vehicles[vehicle_id]
                indices = vehicle["frames"] - first_frame
                visible = (indices >= 0) & (indices < ticks)
                inside = (vehicle["x"] >= low) & (vehicle["x"] <= high)
                usable = visible & inside & ~vehicle["abnormal"]
                if not usable.any():
                    continue
                observed = np.column_stack(
                    [
                        vehicle[name]
                        for name in ("x", "y_left", "vx", "vy_left", "ax", "ay_left")
                    ]
                )
                physical_entry = np.flatnonzero(visible & inside)
                entry = int(physical_entry[0])
                # An abnormal entry cannot supply a reliable initialization.
                if vehicle["abnormal"][entry]:
                    continue
                states = np.zeros((ticks, 6), np.float32)
                valid = np.zeros(ticks, bool)
                states[indices[usable]] = observed[usable]
                valid[indices[usable]] = True
                blocks.append(states)
                active.append(valid)
                included.append(vehicle_id)
                dimensions.append((vehicle["length"], vehicle["lat_width"]))
                entry_ticks.append(int(indices[entry]))
                entry_states.append(observed[entry])
                past_frames = vehicle["frames"][entry] - np.arange(25, 0, -1)
                past_index = np.searchsorted(vehicle["frames"], past_frames)
                past_index = np.minimum(past_index, len(vehicle["frames"]) - 1)
                past_valid = (vehicle["frames"][past_index] == past_frames) & ~vehicle[
                    "abnormal"
                ][past_index]
                past_valid &= np.isfinite(observed[past_index]).all(-1)
                history = np.zeros((25, 6), np.float32)
                history[past_valid] = observed[past_index[past_valid]]
                entry_histories.append(history)
                entry_history_valid.append(past_valid)
                # No observed leader is called free driving only with adequate sight.
                track = recording.get_vehicle_track(vehicle_id)
                mask = (
                    usable
                    & (vehicle["frames"] % 5 == 0)
                    & track.precedingId.lt(0).to_numpy()
                    & track.frontSightDistance.ge(80).to_numpy()
                    & (vehicle["vx"] > 5)
                    & (vehicle["vx"] < 60)
                    & (np.abs(vehicle["vy_left"]) < 0.3)
                )
                for point in np.flatnonzero(mask):
                    free.append(
                        [
                            recording_id,
                            vehicle_id,
                            vehicle["frames"][point],
                            vehicle["vx"][point],
                            vehicle["ax"][point],
                        ]
                    )
            key = f"{recording_id:02d}_{direction}"
            cohorts[key] = {
                "recording_id": np.asarray(recording_id),
                "direction": np.asarray(direction),
                "states": np.stack(blocks, axis=1),
                "valid": np.stack(active, axis=1),
                "dimensions": np.asarray(dimensions, np.float32),
                "vehicle_ids": np.asarray(included, int),
                "entry_ticks": np.asarray(entry_ticks, int),
                "entry_states": np.asarray(entry_states, np.float32),
                "entry_history": np.asarray(entry_histories, np.float32),
                "entry_history_valid": np.asarray(entry_history_valid, bool),
                "lane_centers": centers.astype(np.float32),
                "lane_widths": widths.astype(np.float32),
                "bounds": np.asarray([low, high], np.float32),
            }
            print(
                f"Road {key}: {len(included)} initialized/entering vehicles, {high - low:.1f} m",
                flush=True,
            )
    return cohorts, np.asarray(free, np.float32)


def neighbors(states, valid, dimensions, centers):
    """Full-road nearest physical actors, not six cached neighbors."""
    lane = np.abs(states[..., 1, None] - centers).argmin(-1)
    dx = states[..., None, :, 0] - states[..., :, None, 0]
    eligible = valid[..., :, None] & valid[..., None, :]
    leader = np.full(valid.shape, -1, int)
    adjacent = []
    for offset in (0, -1, 1):
        target = lane[..., :, None] + offset
        same = (lane[..., None, :] == target) & eligible
        front = np.where(same & (dx > 0), dx, np.inf)
        rear = np.where(same & (dx < 0), -dx, np.inf)
        front_index = np.where(np.isfinite(front.min(-1)), front.argmin(-1), -1)
        rear_index = np.where(np.isfinite(rear.min(-1)), rear.argmin(-1), -1)
        if offset == 0:
            leader = front_index
            current_rear = rear_index
        else:
            adjacent.extend((front_index, rear_index))
    index = leader.clip(0)
    front_x = np.take_along_axis(states[..., 0], index, -1)
    front_speed = np.take_along_axis(states[..., 2], index, -1)
    length = np.broadcast_to(dimensions[:, 0], valid.shape)
    front_length = np.take_along_axis(length, index, -1)
    gap = front_x - states[..., 0] - 0.5 * (length + front_length)
    gap = np.where(leader >= 0, gap, 1e6)
    speed = np.linalg.norm(states[..., 2:4], axis=-1)
    front_speed = np.where(leader >= 0, front_speed, speed)
    return lane, leader, gap, front_speed, [leader, current_rear, *adjacent]


def packed_candidates(states, valid, dimensions, centers, widths, theta):
    """Reuse the unchanged seven-actor MOBIL equation once per focal actor."""
    _, _, _, _, partners = neighbors(states, valid, dimensions, centers)
    actor = np.broadcast_to(np.arange(states.shape[1]), valid.shape)
    # Focal NPC is index 1; index 0 is its current leader, never a follower.
    index = np.stack([partners[0], actor, *partners[1:]], -1)
    repeat = np.arange(len(states))[:, None, None]
    packed = states[repeat, index.clip(0)].reshape(-1, 7, 6)
    present = (index >= 0) & valid[repeat, index.clip(0)]
    shape = index.shape[:2]
    sizes = np.broadcast_to(dimensions, (*shape, 2))
    sizes = sizes[repeat, index.clip(0)].reshape(-1, 7, 2)
    drivers = theta[repeat, index.clip(0)][..., 1:, :].reshape(-1, 6, 5)
    maps = np.zeros((len(packed), len(centers), 2, 6), np.float32)
    maps[..., 0] = np.asarray([-1000, 1000])
    maps[..., 1] = centers[None, :, None]
    maps[..., 2] = 1
    maps[..., 4] = widths[None, :, None]
    current = torch.from_numpy(packed.copy())
    present = torch.from_numpy(present.reshape(-1, 7).copy())
    dimensions = torch.from_numpy(sizes.copy())
    relation = build_dynamic_interaction_relations(
        current,
        present,
        torch.from_numpy(maps),
        torch.ones(maps.shape[:-1], dtype=torch.bool),
        vehicle_lengths_m=dimensions[..., 0],
    )
    return current, present, dimensions, torch.from_numpy(drivers.copy()), relation


def rectangular_contacts(states, valid, dimensions):
    """Oriented rectangle intersections; diagnostic only, no impact model."""
    heading = np.arctan2(states[..., 3], states[..., 2])
    c, s = np.cos(heading), np.sin(heading)
    dx = states[..., None, :, 0] - states[..., :, None, 0]
    dy = states[..., None, :, 1] - states[..., :, None, 1]
    cosine = np.abs(
        c[..., :, None] * c[..., None, :] + s[..., :, None] * s[..., None, :]
    )
    sine = np.abs(c[..., :, None] * s[..., None, :] - s[..., :, None] * c[..., None, :])
    length, width = dimensions.T / 2
    forward = np.abs(dx * c[..., :, None] + dy * s[..., :, None])
    lateral = np.abs(-dx * s[..., :, None] + dy * c[..., :, None])
    overlap = (
        forward < length[:, None] + length[None, :] * cosine + width[None, :] * sine
    ) & (lateral < width[:, None] + length[None, :] * sine + width[None, :] * cosine)
    overlap &= overlap.swapaxes(-1, -2)
    return (
        overlap
        & valid[..., :, None]
        & valid[..., None, :]
        & ~np.eye(len(dimensions), dtype=bool)
    )


def physical_leaders(states, dimensions):
    """Resolve current leaders from actual vehicle footprints."""
    result = physical_leader_observation(
        torch.from_numpy(states),
        torch.ones(len(states), dtype=torch.bool),
        torch.from_numpy(dimensions),
    )
    return tuple(value.numpy() for value in result)


def _entry_clear(entry, dimensions, states, actor_dimensions):
    if not len(states):
        return True
    dx = states[:, 0] - entry[0]
    dy = states[:, 1] - entry[1]
    widths = 0.5 * (actor_dimensions[:, 1] + dimensions[1]) + 0.5
    lengths = 0.5 * (actor_dimensions[:, 0] + dimensions[0])
    prospective = dx + (states[:, 2] - entry[2]) * 0.5
    overlap = (np.abs(dy) < widths) & (
        np.minimum(np.abs(dx), np.abs(prospective)) < lengths + 2
    )
    speed = np.linalg.norm(states[:, 2:4], axis=-1)
    entry_speed = float(np.linalg.norm(entry[2:4]))
    stopping_margin = (
        2 + max(0.0, entry_speed * 0.5) + np.maximum(entry_speed**2 - speed**2, 0) / 16
    )
    unsafe_front = (np.abs(dy) < widths) & (dx > 0) & (dx - lengths < stopping_margin)
    return not (overlap | unsafe_front).any()


def _vehicle_state(vehicle):
    control = getattr(vehicle, "_external_control", np.zeros(2))
    speed, heading = vehicle.speed, vehicle.heading
    acceleration, yaw_rate = control
    return np.asarray(
        [
            *vehicle.position,
            speed * np.cos(heading),
            speed * np.sin(heading),
            acceleration * np.cos(heading) - speed * yaw_rate * np.sin(heading),
            acceleration * np.sin(heading) + speed * yaw_rate * np.cos(heading),
        ],
        np.float32,
    )


def entry_lane_intent(state, history, history_valid, centers):
    """Continue a maneuver evidenced by past lateral motion at the road boundary."""
    nearest = int(np.abs(state[1] - centers).argmin())
    target = nearest
    moving = False
    if history_valid[-5:].all():
        last = history[-5:, 3]
        direction = 1 if last.mean() > 0 else -1
        first = history[np.flatnonzero(history_valid)[0], 1]
        moving = bool(
            (direction * last > 0.15).all() and direction * (state[1] - first) >= 0.1
        )
        if moving:
            arriving = direction * (first - centers[nearest]) < -0.1
            behind = direction * (centers[nearest] - state[1]) < 0
            if not (arriving and behind):
                ahead = np.flatnonzero(direction * (centers - state[1]) >= 0)
                if len(ahead):
                    target = int(ahead[np.abs(centers[ahead] - state[1]).argmin()])
                elif 0 <= nearest + direction < len(centers):
                    target = nearest + direction
    return float(centers[target]), moving, target != nearest


@torch.no_grad()
def simulate_road(
    cohort,
    population,
    mobil,
    *,
    repeats=16,
    duration_s=None,
    correction=None,
    seed=SEED,
):
    """Advance real Road.step calls with causal NPC actions and queued demand."""
    StraightLane, Road, RoadNetwork, _, Vehicle = _highway_classes()
    vehicle_type = type("DriverVehicle", (UnicycleDynamicsVehicleMixin, Vehicle), {})
    available_ticks, actors = cohort["valid"].shape
    ticks = (
        available_ticks
        if duration_s is None
        else min(available_ticks, round(duration_s / DT) + 1)
    )
    dimensions = cohort["dimensions"]
    centers, widths = cohort["lane_centers"], cohort["lane_widths"]
    lower, upper = cohort["bounds"]
    correction = mobil.get("correction") if correction is None else correction
    namespace = [seed, int(cohort["recording_id"]), int(cohort["direction"])]
    draws = np.stack(
        [
            np.random.default_rng([*namespace, int(actor), 0]).standard_normal(
                (repeats, 7)
            )
            for actor in cohort["vehicle_ids"]
        ],
        axis=1,
    )
    theta = sample_theta(population, draws[..., :5])
    path = LearnedLaneTrajectory(
        correction=None if correction is None else correction["trajectory"],
        model=mobil.get("lane_model"),
    )
    lane_style = mobil.get("lane_style", {"mean": 0.0, "std": 0.0})
    trace = np.zeros((repeats, ticks, actors, 6), np.float32)
    trace_valid = np.zeros((repeats, ticks, actors), bool)
    metadata = []
    for repeat in range(repeats):
        started = perf_counter()
        generator = np.random.default_rng([*namespace, repeat, 1])
        style_offset = np.asarray(
            [
                lane_style["mean"]
                + lane_style["std"]
                * np.random.default_rng(
                    [*namespace, int(actor), repeat, 4]
                ).standard_normal()
                for actor in cohort["vehicle_ids"]
            ],
            np.float32,
        )
        innovations = np.stack(
            [
                np.random.default_rng(
                    [*namespace, int(actor), repeat, 1]
                ).standard_normal((ticks, 2))
                for actor in cohort["vehicle_ids"]
            ],
            axis=1,
        ).astype(np.float32)
        initial_innovations = np.stack(
            [
                np.random.default_rng(
                    [*namespace, int(actor), repeat, 2]
                ).standard_normal(2)
                for actor in cohort["vehicle_ids"]
            ]
        ).astype(np.float32)
        shape_innovations = np.stack(
            [
                np.random.default_rng(
                    [*namespace, int(actor), repeat, 3]
                ).standard_normal((ticks, 2))
                for actor in cohort["vehicle_ids"]
            ],
            axis=1,
        ).astype(np.float32)
        network = RoadNetwork()
        for center, width in zip(centers, widths):
            network.add_lane(
                "start",
                "end",
                StraightLane(
                    [lower - 100, center], [upper + 100, center], width=float(width)
                ),
            )
        road = Road(network, np_random=generator, record_history=False)
        vehicles, queues = {}, [deque() for _ in centers]
        lane_probabilities = {}
        target = np.zeros(actors, np.float32)
        coefficients = torch.zeros(actors, 8)
        origin = torch.zeros(actors)
        distance = torch.ones(actors)
        residual = torch.zeros(actors, 4)
        changing = np.zeros(actors, bool)
        source_y = torch.zeros(actors)
        count_completion = np.zeros(actors, bool)
        cooldown = np.zeros(actors, int)
        stable = np.zeros(actors, int)
        arrival_lane = np.abs(cohort["entry_states"][:, 1, None] - centers).argmin(-1)
        entered = np.full(actors, -1, int)
        exited = np.full(actors, -1, int)
        finished = np.zeros((ticks, actors), bool)
        queue_count = np.zeros(ticks, int)
        crash_count = np.zeros(ticks, int)
        steps = 0
        failure = None
        lane_starts = []
        lane_decision = {
            name: 0
            for name in (
                "eligible_steps",
                "keep_choice_steps",
                "lower_choice_steps",
                "higher_choice_steps",
                "no_available_target_steps",
                "requested_changes",
                "path_rejected",
                "reservation_rejected",
                "started_changes",
                "initial_continuations",
                "unfinished_at_end",
                "unfinished_on_exit",
            )
        }
        for tick in range(ticks):
            for actor in np.flatnonzero(cohort["entry_ticks"] == tick):
                queues[arrival_lane[actor]].append(int(actor))
            for queue in queues:
                while queue:
                    actor = queue[0]
                    entry = cohort["entry_states"][actor]
                    active = list(vehicles)
                    current = np.asarray(
                        [_vehicle_state(vehicles[index]) for index in active]
                    ).reshape(-1, 6)
                    if tick and not _entry_clear(
                        entry, dimensions[actor], current, dimensions[active]
                    ):
                        break
                    queue.popleft()
                    vehicle = vehicle_type(
                        road,
                        entry[:2].astype(float).copy(),
                        heading=float(np.arctan2(entry[3], entry[2])),
                        speed=float(np.linalg.norm(entry[2:4])),
                    )
                    vehicle.LENGTH, vehicle.WIDTH = map(float, dimensions[actor])
                    vehicle.diagonal = float(np.linalg.norm(dimensions[actor]))
                    vehicle.MAX_SPEED = 100.0
                    vehicle._external_control = np.zeros(2, np.float32)
                    vehicle.actor_index = actor
                    vehicles[actor] = vehicle
                    road.vehicles.append(vehicle)
                    entered[actor] = tick
                    source_y[actor] = float(centers[arrival_lane[actor]])
                    for past_state, present in zip(
                        cohort["entry_history"][actor],
                        cohort["entry_history_valid"][actor],
                    ):
                        if present:
                            nearest = int(np.abs(centers - past_state[1]).argmin())
                            _, source_y[actor] = causal_maneuver_features(
                                torch.from_numpy(past_state),
                                source_y[actor],
                                torch.tensor(centers[nearest]),
                                torch.tensor(widths[nearest]),
                            )
                    target[actor], changing[actor], count_completion[actor] = (
                        entry_lane_intent(
                            entry,
                            cohort["entry_history"][actor],
                            cohort["entry_history_valid"][actor],
                            centers,
                        )
                    )
                    lane_decision["initial_continuations"] += int(changing[actor])
                    planned = path.plan(
                        torch.from_numpy(entry[None]),
                        torch.from_numpy(dimensions[actor : actor + 1]),
                        torch.tensor(target[actor : actor + 1]),
                        torch.from_numpy(
                            draws[repeat, actor : actor + 1, 5:].astype(np.float32)
                        ),
                    )
                    coefficients[actor], origin[actor], distance[actor] = [
                        part[0] for part in planned
                    ]
            for actor, vehicle in list(vehicles.items()):
                if vehicle.position[0] - vehicle.LENGTH / 2 > upper:
                    lane_decision["unfinished_on_exit"] += int(changing[actor])
                    road.vehicles.remove(vehicle)
                    del vehicles[actor]
                    exited[actor] = tick
            queue_count[tick] = sum(map(len, queues))
            active = np.asarray(list(vehicles), int)
            current = np.asarray(
                [_vehicle_state(vehicles[index]) for index in active], np.float32
            ).reshape(-1, 6)
            if not np.isfinite(current).all():
                failure = {"tick": tick, "reason": "nonfinite_vehicle_state"}
                break
            trace[repeat, tick, active] = current
            trace_valid[repeat, tick, active] = True
            crash_count[tick] = sum(vehicle.crashed for vehicle in vehicles.values())
            if tick == ticks - 1:
                break
            if len(active):
                sizes = dimensions[active]
                gap, speed, leader_speed, leader = physical_leaders(current, sizes)
                history_features = None
                lane = np.abs(current[:, 1, None] - centers).argmin(-1)
                maneuver, source_y[active] = causal_maneuver_features(
                    torch.from_numpy(current),
                    source_y[active],
                    torch.from_numpy(centers[lane]),
                    torch.from_numpy(widths[lane]),
                )
                if correction is not None:
                    past = np.zeros_like(current)
                    past_valid = np.zeros(len(active), bool)
                    if tick >= 5:
                        past[:] = trace[repeat, tick - 5, active]
                        past_valid[:] = trace_valid[repeat, tick - 5, active]
                    ages = tick - entered[active]
                    recent = ages < 5
                    past[recent] = cohort["entry_history"][
                        active[recent], 19 + ages[recent]
                    ]
                    past_valid[recent] = cohort["entry_history_valid"][
                        active[recent], 19 + ages[recent]
                    ]
                    packed = torch.cat(
                        (torch.zeros(1, 1, 6), torch.from_numpy(current)[None]), 1
                    )
                    past_packed = torch.cat(
                        (torch.zeros(1, 1, 6), torch.from_numpy(past)[None]), 1
                    )
                    history = torch.zeros(1, 6, len(active) + 1, 6)
                    history[:, 0] = past_packed
                    present = torch.zeros(1, 6, len(active) + 1, dtype=torch.bool)
                    present[:, 0, 1:] = torch.from_numpy(past_valid)
                    sizes_packed = torch.cat(
                        (torch.ones(1, 1, 2), torch.from_numpy(sizes)[None]), 1
                    )
                    history_features = response_history_features(
                        packed,
                        history,
                        present,
                        torch.from_numpy(np.where(leader >= 0, leader + 1, -1))[None],
                        sizes_packed,
                    )[0]
                observed_history = np.zeros((6, len(active), 6), np.float32)
                observed_valid = np.zeros((6, len(active)), bool)
                for k, lag in enumerate(range(5, -1, -1)):
                    use_trace = tick - lag >= entered[active]
                    if tick >= lag:
                        observed_history[k, use_trace] = trace[
                            repeat, tick - lag, active[use_trace]
                        ]
                        observed_valid[k, use_trace] = trace_valid[
                            repeat, tick - lag, active[use_trace]
                        ]
                    before_entry = ~use_trace
                    historical_index = 24 + tick - entered[active[before_entry]] - lag
                    observed_history[k, before_entry] = cohort["entry_history"][
                        active[before_entry], historical_index
                    ]
                    observed_valid[k, before_entry] = cohort["entry_history_valid"][
                        active[before_entry], historical_index
                    ]
                acceleration, next_residual = corrected_acceleration(
                    torch.from_numpy(gap.astype(np.float32)),
                    torch.from_numpy(speed.astype(np.float32)),
                    torch.from_numpy(leader_speed.astype(np.float32)),
                    torch.from_numpy(theta[repeat, active].astype(np.float32)),
                    correction=correction,
                    residual_state=residual[active],
                    innovation=torch.from_numpy(innovations[tick, active, 1]),
                    initial_innovation=torch.from_numpy(initial_innovations[active]),
                    history_features=history_features,
                    maneuver_features=maneuver,
                    entry_features=anticipated_entry_features(
                        torch.from_numpy(current)[None],
                        torch.ones(1, len(active), dtype=torch.bool),
                        torch.from_numpy(sizes)[None],
                        torch.from_numpy(trace[repeat, max(0, tick - 1), active])[None],
                        torch.from_numpy(trace_valid[repeat, max(0, tick - 1), active])[
                            None
                        ],
                        history=torch.from_numpy(observed_history)[None],
                        history_valid=torch.from_numpy(observed_valid)[None],
                    )[0],
                    initialize=torch.from_numpy(entered[active] == tick),
                    dt=DT,
                )
                if next_residual is not None:
                    residual[active] = next_residual
                lane = np.abs(current[:, 1, None] - centers).argmin(-1)
                settled = (
                    np.abs(current[:, 1] - target[active]) < 0.15 * widths[lane]
                ) & (np.abs(current[:, 3]) < 0.2)
                stable[active] = np.where(
                    changing[active] & settled, stable[active] + 1, 0
                )
                completed = changing[active] & (stable[active] >= 5)
                finished[tick, active] = completed & count_completion[active]
                changing[active[completed]] = False
                cooldown[active[completed]] = tick + 25
                if tick % 5 == 0:
                    packed, present, packed_sizes, drivers, relation = (
                        packed_candidates(
                            current[None],
                            np.ones((1, len(active)), bool),
                            sizes,
                            centers,
                            widths,
                            theta[repeat : repeat + 1, active],
                        )
                    )
                    decision = evaluate_mobil_candidates(
                        packed,
                        present,
                        drivers,
                        relation,
                        vehicle_lengths_m=packed_sizes[..., 0],
                        politeness=mobil["politeness"],
                        minimum_incentive_mps2=0.0,
                        safe_braking_mps2=4.0,
                        plan_keep_bias_mps2=0.0,
                    )
                    features = lane_choice_features(
                        packed, packed_sizes[..., 0], relation
                    )[:, 0]
                    probabilities = exclusive_lane_probabilities(
                        decision.incentive_mps2[:, 0],
                        decision.safe[:, 0],
                        mobil,
                        features=features,
                        correction=correction,
                        style_offset=(
                            torch.from_numpy(style_offset[active])
                            if "lane_style" in mobil
                            else None
                        ),
                    )
                    lane_probabilities = {
                        int(actor): probabilities[index]
                        for index, actor in enumerate(active)
                    }
                probabilities = torch.stack(
                    [
                        lane_probabilities.get(
                            int(actor), torch.tensor([1.0, 0.0, 0.0])
                        )
                        for actor in active
                    ]
                )
                choice = sample_lane_choice(
                    probabilities,
                    torch.special.ndtr(torch.from_numpy(innovations[tick, active, 0])),
                    DT,
                ).numpy()
                eligible = (
                    ~changing[active]
                    & (tick >= cooldown[active])
                    & (np.abs(current[:, 1] - centers[lane]) < 0.2 * widths[lane])
                    & (np.abs(current[:, 3]) < 0.3)
                )
                available_target = (lane > 0) | (lane < len(centers) - 1)
                lane_decision["no_available_target_steps"] += int(
                    (eligible & ~available_target).sum()
                )
                eligible &= available_target
                lane_decision["eligible_steps"] += int(eligible.sum())
                for value, name in (
                    (0, "keep_choice_steps"),
                    (-1, "lower_choice_steps"),
                    (1, "higher_choice_steps"),
                ):
                    lane_decision[name] += int((eligible & (choice == value)).sum())
                selected = (
                    (choice != 0) & ~changing[active] & (tick >= cooldown[active])
                )
                selected &= (
                    np.abs(current[:, 1] - centers[lane]) < 0.2 * widths[lane]
                ) & (np.abs(current[:, 3]) < 0.3)
                if selected.any():
                    destination = lane + choice
                    selected &= (destination >= 0) & (destination < len(centers))
                    lane_decision["requested_changes"] += int(selected.sum())
                    candidate_y = torch.from_numpy(
                        centers[destination.clip(0, len(centers) - 1)]
                    )
                    global_states = torch.cat(
                        (torch.zeros(1, 1, 6), torch.from_numpy(current)[None]), 1
                    )
                    global_sizes = torch.cat(
                        (torch.ones(1, 1, 2), torch.from_numpy(sizes)[None]), 1
                    )
                    global_valid = torch.ones(1, len(active) + 1, dtype=torch.bool)
                    global_valid[:, 0] = False
                    planned, clear = path.safe_plan(
                        global_states,
                        global_valid,
                        global_sizes,
                        candidate_y[None],
                        torch.from_numpy(shape_innovations[tick, active])[None],
                        acceleration[None],
                        (
                            torch.tensor([centers[0] - widths[0] / 2]),
                            torch.tensor([centers[-1] + widths[-1] / 2]),
                        ),
                    )
                    lane_decision["path_rejected"] += int(
                        (selected & ~clear[0].numpy()).sum()
                    )
                    selected &= clear[0].numpy()
                    claimed = []
                    for local in np.flatnonzero(selected):
                        actor = active[local]
                        if any(
                            destination[local] == target_lane
                            and abs(current[local, 0] - other_x)
                            < 2 * max(sizes[local, 0], other_length)
                            for target_lane, other_x, other_length in claimed
                        ):
                            lane_decision["reservation_rejected"] += 1
                            continue
                        lane_decision["started_changes"] += 1
                        lane_starts.append(
                            {
                                "tick": tick,
                                "vehicle": int(cohort["vehicle_ids"][actor]),
                                "source_lane": int(lane[local]),
                                "target_lane": int(destination[local]),
                            }
                        )
                        target[actor] = candidate_y[local]
                        coefficients[actor], origin[actor], distance[actor] = [
                            part[0, local] for part in planned
                        ]
                        changing[actor] = True
                        count_completion[actor] = True
                        stable[actor] = 0
                        claimed.append(
                            (destination[local], current[local, 0], sizes[local, 0])
                        )
                yaw = path.track(
                    torch.from_numpy(current),
                    coefficients[active],
                    origin[active],
                    distance[active],
                )
                yaw = limit_realized_lateral_acceleration(
                    torch.from_numpy(current),
                    yaw,
                    acceleration,
                    acceleration_limit=path.acceleration_limit,
                    jerk_limit=path.jerk_limit,
                    dt=DT,
                ).clamp(-0.6, 0.6)
                for local, actor in enumerate(active):
                    vehicles[actor]._external_control = np.asarray(
                        [float(acceleration[local]), float(yaw[local])], np.float32
                    )
            road.step(DT)
            steps += 1
        wait = (
            np.where(
                entered >= 0,
                entered - cohort["entry_ticks"],
                ticks - 1 - cohort["entry_ticks"],
            ).clip(0)
            * DT
        )
        lane_decision["unfinished_at_end"] = int(
            sum(changing[actor] for actor in vehicles)
        )
        metadata.append(
            {
                "repeat": repeat,
                "status": "complete" if failure is None else "failed",
                "failure": failure,
                "road_step_calls": steps,
                "simulated_seconds": steps * DT,
                "elapsed_seconds": perf_counter() - started,
                "entered": int((entered >= 0).sum()),
                "requested": int((cohort["entry_ticks"] < ticks).sum()),
                "pending": int(sum(map(len, queues))),
                "exited": int((exited >= 0).sum()),
                "entry_tick": entered,
                "exit_tick": exited,
                "entry_wait_seconds": wait,
                "queue_vehicles": queue_count,
                "crashed_vehicles": crash_count,
                "completed_lane_changes": finished,
                "lane_decision": lane_decision,
                "lane_starts": lane_starts,
            }
        )
    return (
        trace,
        trace_valid,
        {
            "backend": "highway_env.road.road.Road.step",
            "dynamics": "kinematic_unicycle",
            "dt": DT,
            "runs": metadata,
        },
    )


def completed_lane_events(states, valid, centers, widths):
    """Count a lane change after settling at its new centre for five frames."""
    repeats, ticks, actors = valid.shape
    anchor = np.full((repeats, actors), -1, int)
    candidate = np.full_like(anchor, -1)
    stable = np.zeros_like(anchor)
    events = np.zeros_like(valid)
    for tick in range(ticks):
        current = states[:, tick]
        present = valid[:, tick]
        lane = np.abs(current[..., 1, None] - centers).argmin(-1)
        fresh = present & (anchor < 0)
        anchor[fresh] = lane[fresh]
        settled = (
            present
            & (np.abs(current[..., 1] - centers[lane]) < 0.15 * widths[lane])
            & (np.abs(current[..., 3]) < 0.2)
        )
        proposed = settled & (lane != anchor)
        stable = np.where(
            proposed & (lane == candidate), stable + 1, np.where(proposed, 1, 0)
        )
        candidate = np.where(proposed, lane, -1)
        complete = stable >= 5
        events[:, tick] = complete
        anchor[complete] = lane[complete]
        stable[complete] = 0
        anchor[~present] = -1
    return events


def traffic_statistics(
    states,
    valid,
    cohort,
    *,
    motion_window=25,
    window_s=WINDOW_S,
    completed_changes=None,
):
    """Common 5 Hz exposure and position derivatives, retaining overflow mass."""
    if states.ndim == 3:
        states, valid = states[None], valid[None]
    inside = (states[..., 0] >= cohort["bounds"][0]) & (
        states[..., 0] <= cohort["bounds"][1]
    )
    valid = valid & inside
    repetitions, ticks, _actors = valid.shape
    windows = max(1, int(np.ceil((ticks - 1) * DT / window_s)))
    derived = states.copy()
    for derivative, columns in ((1, slice(2, 4)), (2, slice(4, 6))):
        derived[..., columns] = savgol_filter(
            states[..., :2], motion_window, 3, deriv=derivative, delta=DT, axis=1
        )
    continuous = minimum_filter1d(
        valid.astype(np.uint8), motion_window, axis=1, mode="constant"
    ).astype(bool)
    histogram = {
        name: np.zeros((repetitions, windows, len(edges) - 1))
        for name, edges in FEATURE_EDGES.items()
    }
    histogram["joint"] = np.zeros(
        (
            repetitions,
            windows,
            (len(JOINT_EDGES[0]) - 1) * (len(JOINT_EDGES[1]) - 1),
        )
    )
    histogram["age_speed"] = np.zeros((repetitions, 4, len(FEATURE_EDGES["speed"]) - 1))
    histogram["age_joint"] = np.zeros(
        (repetitions, 4, (len(JOINT_EDGES[0]) - 1) * (len(JOINT_EDGES[1]) - 1))
    )
    histogram["following_status"] = np.zeros((repetitions, windows, 3))
    first_visible = valid.argmax(axis=1)
    exposure = np.zeros((repetitions, windows, 6))
    # Columns: vehicle time, km, completed changes, contact vehicle time, offroad time, derivative time.
    if completed_changes is None:
        completed_changes = completed_lane_events(
            states, valid, cohort["lane_centers"], cohort["lane_widths"]
        )
    completed_changes = completed_changes & valid
    boundaries = np.asarray(
        [
            cohort["lane_centers"][0] - cohort["lane_widths"][0] / 2,
            cohort["lane_centers"][-1] + cohort["lane_widths"][-1] / 2,
        ]
    )
    for tick in range(1, ticks, 5):
        window = min(int(tick * DT // window_s), windows - 1)
        current, present = states[:, tick], valid[:, tick]
        _lane, leader, gap, _front_speed, _ = neighbors(
            current, present, cohort["dimensions"], cohort["lane_centers"]
        )
        derivative = derived[:, tick]
        velocity = np.linalg.norm(derivative[..., 2:4], axis=-1)
        acceleration = (derivative[..., 2:4] * derivative[..., 4:6]).sum(
            -1
        ) / velocity.clip(0.5)
        leader_v = np.take_along_axis(velocity, leader.clip(0), -1)
        contacts = rectangular_contacts(current, present, cohort["dimensions"]).any(-1)
        past_tick = max(tick - 5, 0)
        crossing = completed_changes[:, past_tick + 1 : tick + 1].sum(1)
        lateral_extent = 0.5 * cohort["dimensions"][:, 1]
        offroad = (current[..., 1] - lateral_extent < boundaries[0]) | (
            current[..., 1] + lateral_extent > boundaries[1]
        )
        for repeat in range(repetitions):
            mask = continuous[repeat, tick]
            follow = (
                mask
                & (leader[repeat] >= 0)
                & np.take_along_axis(
                    continuous[repeat, tick], leader[repeat].clip(0), -1
                )
            )
            for name, values, keep in (
                ("speed", velocity, mask),
                ("acceleration", acceleration, mask),
                ("gap", gap, follow),
                ("headway", gap / np.maximum(velocity, 0.5), follow),
                (
                    "ttc",
                    np.where(
                        velocity > leader_v,
                        gap / np.maximum(velocity - leader_v, 1e-6),
                        np.inf,
                    ),
                    follow,
                ),
            ):
                histogram[name][repeat, window] += np.histogram(
                    values[repeat, keep], FEATURE_EDGES[name]
                )[0]
            histogram["joint"][repeat, window] += np.histogram2d(
                gap[repeat, follow], (leader_v - velocity)[repeat, follow], JOINT_EDGES
            )[0].ravel()
            age = (tick - first_visible[repeat]) * DT
            histogram["following_status"][repeat, window] += [
                int((mask & ~follow).sum()),
                int((follow & (velocity[repeat] > leader_v[repeat])).sum()),
                int((follow & (velocity[repeat] <= leader_v[repeat])).sum()),
            ]
            age_group = np.searchsorted([2, 5, 10], age, side="right")
            for group in range(4):
                selected = age_group == group
                histogram["age_speed"][repeat, group] += np.histogram(
                    velocity[repeat, mask & selected], FEATURE_EDGES["speed"]
                )[0]
                histogram["age_joint"][repeat, group] += np.histogram2d(
                    gap[repeat, follow & selected],
                    (leader_v - velocity)[repeat, follow & selected],
                    JOINT_EDGES,
                )[0].ravel()
            exposure[repeat, window] += [
                present[repeat].sum() * 0.2,
                np.linalg.norm(current[repeat, present[repeat], 2:4], axis=-1).sum()
                * 0.2
                / 1000,
                crossing[repeat].sum(),
                (contacts[repeat] & present[repeat]).sum() * 0.2,
                (offroad[repeat] & present[repeat]).sum() * 0.2,
                mask.sum() * 0.2,
            ]
    # Count all observed completions, including the final partial 5-Hz interval.
    exposure[..., 2] = 0
    repetitions_index, frame_index, _ = np.nonzero(completed_changes[:, 1:])
    event_windows = np.minimum(
        ((frame_index + 1) * DT / window_s).astype(int), windows - 1
    )
    np.add.at(exposure[..., 2], (repetitions_index, event_windows), 1)
    return histogram, exposure

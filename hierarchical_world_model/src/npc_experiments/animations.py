"""Autonomous NPC movies from the selected-policy road experiment."""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgba
from PIL import Image

from process_highD.src.natural_event_playback import (
    NaturalPlaybackOptions,
    ROAD_COLOR,
    SLOT_COLOR,
    TARGET_COLOR,
)
from tools.plot_style import get_pyplot
from ..core.utils import load_json, save_json
from ..stochastic_drivers.raw_data import DRIVER_DATA_DIR
from .evaluation import stage_models
from .lateral import lane_events
from .road import DT, simulate_road, traffic_statistics
from .search import SEED

MOVIE_NAMES = ("light_traffic", "moderate_traffic", "higher_density_traffic")
MOVIE_TITLES = (
    "Lower-density traffic",
    "Intermediate-density traffic",
    "Higher-density traffic",
)


def select_examples(cohorts):
    """Use the median initial density within each of three ordered Test strata."""

    def density(key):
        cohort = cohorts[key]
        return (
            cohort["valid"][0].sum()
            * 1000
            / (np.diff(cohort["bounds"])[0] * len(cohort["lane_centers"]))
        )

    ordered = sorted(cohorts, key=lambda key: (density(key), key))
    ranks = ((np.arange(3) + 0.5) * len(ordered) / 3).astype(int)
    return [(ordered[index], float(density(ordered[index]))) for index in ranks]


def vehicle_polygons(states, dimensions):
    """Keep actual dimensions, orientation and equal physical axis scales."""
    corners = np.asarray([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    local = dimensions[:, None] * corners[None] / 2
    heading = np.arctan2(states[:, 3], states[:, 2])
    cosine, sine = np.cos(heading)[:, None], np.sin(heading)[:, None]
    x = local[..., 0] * cosine - local[..., 1] * sine
    y = local[..., 0] * sine + local[..., 1] * cosine
    return np.stack((x, y), -1) + states[:, None, :2]


def render_movie(cohort, states, valid, run, destination, title, *, event=None):
    """Match EVT playback styling and display every executed 25-Hz frame at 1×."""
    options = NaturalPlaybackOptions()
    plt = get_pyplot()
    figure, axis = plt.subplots(figsize=options.figsize, dpi=options.dpi)
    figure.subplots_adjust(left=0.06, right=0.98, bottom=0.24, top=0.76)
    centers, widths = cohort["lane_centers"], cohort["lane_widths"]
    lower, upper = cohort["bounds"]
    origin_x = float((lower + upper) / 2)
    actor = None
    begin, end = 0, len(states) - 1
    if event is not None:
        actor = int(np.flatnonzero(cohort["vehicle_ids"] == event["vehicle"])[0])
        visible = np.flatnonzero(valid[:, actor])
        begin = max(int(visible[0]), event["start_frame"] - 50)
        end = min(int(visible[-1]), event["end_frame"] + 50)
        origin_x = float(states[begin, actor, 0])
    origin_y = float(np.mean(centers))
    motion = states.copy()
    motion[..., :2] -= np.asarray([origin_x, origin_y])
    lane_edges = np.r_[centers - widths / 2, centers[-1] + widths[-1] / 2] - origin_y
    axis.set_facecolor(ROAD_COLOR)
    axis.set_ylim(lane_edges[0] - 2.5, lane_edges[-1] + 2.5)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [m]", fontsize=10)
    axis.set_ylabel("y [m]", fontsize=10)
    axis.tick_params(labelsize=8)
    for boundary, edge in enumerate(lane_edges):
        outer = boundary in (0, len(lane_edges) - 1)
        axis.axhline(
            edge,
            color="white",
            linewidth=0.8,
            linestyle="-" if outer else "--",
            alpha=0.45 if outer else 0.28,
        )
    body = PolyCollection([], edgecolors="black", linewidths=0.8, zorder=4)
    trail = LineCollection([], linewidths=1.6, alpha=0.78, zorder=2)
    axis.add_collection(trail)
    axis.add_collection(body)
    labels = [
        axis.text(
            0,
            0,
            str(identifier),
            ha="center",
            va="bottom",
            fontsize=7,
            color="black",
            zorder=5,
            visible=False,
            clip_on=True,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.4},
        )
        for identifier in cohort["vehicle_ids"]
    ]
    heading = axis.set_title("", fontsize=9)
    frames = np.arange(begin, end + 1)
    images = []
    for tick in frames:
        center = float(motion[tick, actor, 0]) if actor is not None else 0.0
        axis.set_xlim(
            center - options.view_width_m / 2, center + options.view_width_m / 2
        )
        in_view = valid[tick] & (
            np.abs(motion[tick, :, 0] - center) <= options.view_width_m / 2
        )
        active = np.flatnonzero(in_view)
        vertices = vehicle_polygons(motion[tick, active], cohort["dimensions"][active])
        colours = np.tile(to_rgba(SLOT_COLOR, 0.92), (len(active), 1))
        if actor is not None:
            colours[active == actor] = to_rgba(TARGET_COLOR, 0.95)
        body.set_verts(vertices)
        body.set_facecolors(colours)
        segments, trail_colours = [], []
        for index, vehicle in enumerate(active):
            past = np.arange(max(0, tick - options.trail_frames), tick + 1)
            past = past[valid[past, vehicle]]
            if len(past) > 1:
                segments.append(motion[past, vehicle, :2])
                trail_colours.append(colours[index])
        trail.set_segments(segments)
        trail.set_colors(trail_colours)
        for vehicle, label in enumerate(labels):
            label.set_visible(bool(in_view[vehicle]))
            if in_view[vehicle]:
                position = motion[tick, vehicle, :2].copy()
                position[1] += 0.8 * cohort["dimensions"][vehicle, 1]
                label.set_position(position)
                label.set_text(
                    f"focus NPC #{cohort['vehicle_ids'][vehicle]}"
                    if vehicle == actor
                    else str(cohort["vehicle_ids"][vehicle])
                )
        heading.set_text(
            f"{title} | rec {int(cohort['recording_id']):02d}, direction {int(cohort['direction'])}"
            f" | t={tick * DT:.2f}s | NPCs={valid[tick].sum()} | queue={run['queue_vehicles'][tick]}"
            " | 25 fps, normal speed"
        )
        figure.canvas.draw()
        rgb = Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[:, :, :3])
        images.append(
            rgb.quantize(colors=192)
            if not images
            else rgb.quantize(palette=images[0], dither=Image.Dither.NONE)
        )
    images[0].save(
        destination,
        save_all=True,
        append_images=images[1:],
        duration=int(round(1000 * DT)),
        loop=0,
        disposal=1,
        optimize=True,
    )
    plt.close(figure)
    return {
        "gif": destination.name,
        "frame_count": len(frames),
        "first_frame": int(begin),
        "last_frame": int(end),
        "simulation_seconds": float((end - begin) * DT),
        "playback_speed": 1,
        "gif_duration_seconds": float(len(frames) * DT),
        "fps": 25,
        "resolution": [1440, 360],
        "view_width_m": options.view_width_m,
        "camera": "focus NPC" if actor is not None else "fixed road midpoint",
        "format_reference": "results/evt/playbacks/",
    }


def select_lane_example(directory):
    """Choose the median autonomous complete maneuver across all formal repeat-0 runs."""
    candidates = []
    directory = Path(directory)
    execution = load_json(directory / "runs.json")
    for key, events in pd.read_csv(directory / "events.csv").groupby("cohort"):
        run = next(run for run in execution[key]["runs"] if run["repeat"] == 0)
        events = events[
            (events.source == "model") & (events.repeat == 0) & events.observed_complete
        ]
        for event in events.to_dict("records"):
            matches = [
                start
                for start in run["lane_starts"]
                if start["vehicle"] == event["vehicle"]
                and abs(start["tick"] - event["start_frame"]) <= 25
            ]
            if matches:
                candidates.append((key, {**event, "autonomous_start": matches[0]}))
    candidates.sort(
        key=lambda item: (
            item[1]["duration"],
            item[0],
            item[1]["vehicle"],
            item[1]["start_frame"],
        )
    )
    return candidates[len(candidates) // 2], len(candidates)


def render_autonomous_examples(output):
    output = Path(output)
    destination = output / "animations"
    destination.mkdir(exist_ok=True)
    selected_stage, models = stage_models(output)
    candidate, population, mobil = models[selected_stage]
    report = load_json(output / "autonomous_traffic/report.json")
    keys = [key.removeprefix("long_") for key in report["long"]["cohorts"]]
    parameters = {
        "population": {
            name: population[name].tolist() for name in ("log_mean", "log_cov")
        },
        "mobil": mobil,
        "cohorts": keys,
        "seed": SEED,
        "duration_s": report["long"]["duration_seconds"],
    }
    manifest_path = destination / "manifest.json"
    if (
        manifest_path.exists()
        and load_json(manifest_path).get("parameters") == parameters
    ):
        return load_json(manifest_path)
    with np.load(DRIVER_DATA_DIR / "road_cohorts.npz") as saved:
        cohorts = {
            key: {
                name.split("__")[1]: saved[name]
                for name in saved.files
                if name.startswith(key + "__")
            }
            for key in keys
        }
    movies, traces = [], {}
    rollouts = output / "autonomous_traffic/continuous_runs"

    def rerun(key):
        cohort = cohorts[key]
        states, valid, metadata = simulate_road(
            cohort,
            population,
            mobil,
            repeats=1,
            duration_s=report["long"]["duration_seconds"],
            seed=SEED,
        )
        run = metadata["runs"][0]
        if run["status"] != "complete":
            raise RuntimeError(f"Autonomous movie road {key}: {run['failure']}")
        inside = (states[0, :, :, 0] >= cohort["bounds"][0]) & (
            states[0, :, :, 0] <= cohort["bounds"][1]
        )
        _, complete, _ = lane_events(
            states[0],
            valid[0] & inside,
            cohort["lane_centers"],
            cohort["dimensions"],
            recording=int(cohort["recording_id"]),
            vehicle_ids=cohort["vehicle_ids"],
        )
        histogram, exposure = traffic_statistics(
            states,
            valid,
            cohort,
            window_s=report["long"]["window_seconds"],
            completed_changes=complete[None],
        )
        source = rollouts / "statistics.npz"
        with np.load(source) as statistics:
            for feature, counts in histogram.items():
                np.testing.assert_array_equal(
                    counts[0], statistics[f"{key}_model_{feature}"][0]
                )
            np.testing.assert_array_equal(
                exposure[0], statistics[f"{key}_model_exposure"][0]
            )
            np.testing.assert_array_equal(
                run["entry_tick"], statistics[f"{key}_entry_ticks"][0]
            )
            np.testing.assert_array_equal(
                run["exit_tick"], statistics[f"{key}_exit_ticks"][0]
            )
        traces[key] = (states[0], valid[0], run)
        return source

    for name, title, (key, density) in zip(
        MOVIE_NAMES, MOVIE_TITLES, select_examples(cohorts)
    ):
        print(
            f"Autonomous movie {name}: road {key}, density {density:.2f}",
            flush=True,
        )
        source = rerun(key)
        states, valid, run = traces[key]
        movie = render_movie(
            cohorts[key], states, valid, run, destination / f"{name}.gif", title
        )
        movie.update(
            cohort=key,
            initial_density_vehicles_per_km_per_lane=density,
            repeat=0,
            status=run["status"],
            road_step_calls=run["road_step_calls"],
            entered=run["entered"],
            exited=run["exited"],
            pending=run["pending"],
            lane_decision=run["lane_decision"],
            lane_starts=run["lane_starts"],
            statistics_match_frozen_run=True,
            source_statistics=str(source.relative_to(output)),
        )
        movies.append(movie)
        print(
            f"Saved {movie['gif']}: {movie['frame_count']} frames, {run['road_step_calls']} actual road steps",
            flush=True,
        )
    (key, event), event_count = select_lane_example(rollouts)
    if key not in traces:
        rerun(key)
    event = {
        **event,
        **{
            side + "_" + field: None
            for side in ("front", "rear")
            if not event[side + "_present"]
            for field in ("gap", "relative_speed")
        },
    }
    states, valid, run = traces[key]
    detail = render_movie(
        cohorts[key],
        states,
        valid,
        run,
        destination / "autonomous_lane_change.gif",
        "Autonomously initiated lane change",
        event=event,
    )
    detail.update(
        cohort=key,
        event=event,
        repeat=0,
        statistics_match_frozen_run=True,
        source_statistics=str((rollouts / "statistics.npz").relative_to(output)),
    )
    manifest = {
        "status": "complete",
        "model_stage": selected_stage,
        "candidate": candidate,
        "parameters": parameters,
        "seed": SEED,
        "backend": "highway_env.road.road.Road.step",
        "result_root": str(output.resolve()),
        "actual_additional_road_steps": sum(
            run["road_step_calls"] for _, _, run in traces.values()
        ),
        "selection": f"Median initial density in each ordered third of all {len(keys)} evaluated road-direction cohorts; repeat 0",
        "lane_case_selection": "Median duration of fully observed autonomous maneuvers across every evaluated road-direction cohort; repeat 0",
        "eligible_lane_events": event_count,
        "inputs": "Real road geometry, vehicle dimensions, initial/entry states, observed past and boundary arrivals",
        "controller": "Published persistent-driver IDM, conditional MOBIL and learned lane trajectory",
        "flow_used": False,
        "diffusion_used": False,
        "ADS_intervention_used": False,
        "future_log_state_control_used": False,
        "movies": movies,
        "lane_change": detail,
    }
    save_json(manifest, destination / "manifest.json")
    return manifest

#!/usr/bin/env python3
"""EVT-style paired GIFs of the intervention cases."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image
from matplotlib.collections import PolyCollection, LineCollection

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.core.utils import load_json, save_json
from hierarchical_world_model.src.npc_experiments.animations import vehicle_polygons
from process_highD.src.natural_event_playback import (
    ROAD_COLOR,
    EGO_COLOR,
    TARGET_COLOR,
    SLOT_COLOR,
)
from tools.plot_style import get_pyplot

SOURCE = ROOT / "results/hierarchical_world_model/bridge"
OUTPUT = SOURCE / "playbacks"
TITLES = {
    "null": "No intervention",
    "free_flow": "Fast front vehicle: remain on plan",
    "braking": "Scripted ego braking",
    "acceleration": "Scripted ego acceleration",
    "cut_in": "Scripted ego cut-in",
    "corridor_release": "Leader leaves the lane",
    "lateral_conflict": "Lane change: response in the destination lane",
}
REASONS = {
    1: "Initial conflict",
    2: "Predicted conflict",
    3: "Plan ended",
    4: "Following change",
    5: "Corridor released",
}


def read_case(spec, seed, records):
    with np.load(SOURCE / "intervention_traces.npz") as archive:
        base_prefix = f"{spec['case']}_tracker_20261015/"
        new_prefix = f"{spec['case']}_takeover_{seed}/"
        base = {
            key.removeprefix(base_prefix): archive[key]
            for key in archive.files
            if key.startswith(base_prefix)
        }
        new = {
            key.removeprefix(new_prefix): archive[key]
            for key in archive.files
            if key.startswith(new_prefix)
        }
    result = next(
        r
        for r in records
        if r["case"] == spec["case"] and r["seed"] == seed and r["arm"] == "takeover"
    )
    maps, valid = base["maps"], base["map_valid"]
    present = valid.sum(-1) >= 2
    centers = (maps[..., 1] * valid).sum(-1) / valid.sum(-1).clip(1)
    widths = (maps[..., 4] * valid).sum(-1) / valid.sum(-1).clip(1)
    edges = np.unique(
        np.round(
            np.r_[
                centers[present] - widths[present] / 2,
                centers[present] + widths[present] / 2,
            ],
            4,
        )
    )
    states = [
        np.concatenate((t["initial_states"][None], t["states"])) for t in (base, new)
    ]
    lateral = np.concatenate([s[:, base["valid"], 1].ravel() for s in states])
    return dict(
        spec=spec,
        seed=seed,
        baseline=base,
        bridge=new,
        states=states,
        edges=edges,
        lane_bands=np.stack(
            (
                centers[present] - widths[present] / 2,
                centers[present] + widths[present] / 2,
            ),
            axis=-1,
        ),
        lateral_bounds=[float(lateral.min() - 2.7), float(lateral.max() + 2.7)],
        result=result,
    )


def render_gif(case, destination):
    spec, result = case["spec"], case["result"]
    target = spec["actor"] + 1
    base, new = case["baseline"], case["bridge"]
    states, edges = case["states"], case["edges"]
    entered = [e for e in result["events"] if e["actor"] == spec["actor"]]
    event = entered[0] if entered else None
    first = None if event is None else event["frame"]
    plt = get_pyplot()
    fig = plt.figure(figsize=(12, 5.8), dpi=120)
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2], hspace=0.6, wspace=0.4)
    roads = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, :])]
    metrics = [fig.add_subplot(grid[2, k]) for k in range(3)]
    fig.subplots_adjust(top=0.83, bottom=0.09, left=0.065, right=0.98)
    title = fig.suptitle("", fontsize=12, y=0.99)
    fig.text(
        0.065,
        0.89,
        "RED: scripted ego    BLUE: tracked focus NPC    ORANGE: autonomous NPC    DOTTED: stored plan    25 fps / 1x",
        fontsize=9,
    )
    bodies, trails, plans, labels = [], [], [], []
    for ax in roads:
        ax.set_facecolor("#e9edf1")
        for low, high in case["lane_bands"]:
            ax.axhspan(low, high, color=ROAD_COLOR)
        ax.set_ylim(*case["lateral_bounds"])
        ax.set_aspect("equal", adjustable="box")
        for boundary, y in enumerate(edges):
            ax.axhline(
                y,
                color="white",
                ls="-" if boundary in (0, len(edges) - 1) else "--",
                lw=0.7,
                alpha=0.5,
            )
        ax.set_ylabel("y [m]", fontsize=9)
        ax.tick_params(labelsize=8)
        body = PolyCollection([], edgecolors="black", linewidths=0.8, zorder=5)
        trail = LineCollection([], linewidths=1.4, alpha=0.7, zorder=2)
        ax.add_collection(body)
        ax.add_collection(trail)
        bodies.append(body)
        trails.append(trail)
        plans.append(ax.plot([], [], ":", color="#b5e8f6", lw=1.3, zorder=3)[0])
        labels.append(
            [
                ax.text(
                    0,
                    0,
                    "",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    clip_on=True,
                    zorder=7,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.2),
                )
                for _ in range(7)
            ]
        )
    time = np.arange(150) * 0.04
    speed = [np.linalg.norm(s[:, target, 2:4], axis=-1) for s in states]
    for k, values in enumerate(speed):
        metrics[0].plot(
            time,
            values,
            color=TARGET_COLOR if k == 0 else SLOT_COLOR,
            ls="--" if k == 0 else "-",
            lw=1.4,
            label="Tracking" if k == 0 else "Bridge",
        )
        metrics[1].plot(
            time[:-1],
            (base if k == 0 else new)["actions"][:, spec["actor"], 0],
            color=TARGET_COLOR if k == 0 else SLOT_COLOR,
            ls="--" if k == 0 else "-",
            lw=1.4,
        )
    config = load_json(SOURCE / "interventions.json")["parameters"]
    score = np.maximum(
        new["takeover_score"][:, spec["actor"]] / config["support_threshold"],
        new["takeover_interaction_score"][:, spec["actor"]]
        / config["interaction_threshold"],
    )
    if first is not None:
        score[first + 1 :] = np.nan
    metrics[2].plot(time[:-1], score, color="#5446a5", lw=1.4)
    metrics[2].axhline(1, color="black", ls="--", lw=0.8)
    metrics[0].legend(fontsize=8, loc="best")
    cursors = []
    for ax, label in zip(
        metrics,
        (
            "Focus NPC speed [m/s]",
            "Focus NPC acceleration [m/s2]",
            "Trigger score / threshold",
        ),
    ):
        ax.set_title(label, fontsize=10)
        ax.set_xlim(0, 5.96)
        ax.set_xlabel("time [s]", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2)
        if spec["kind"] not in ("null", "free_flow"):
            ax.axvline(spec["onset_s"], color=EGO_COLOR, alpha=0.7, lw=0.9)
        if first is not None:
            ax.axvline(first * 0.04, color="#267344", ls="--", lw=1)
        cursors.append(ax.axvline(0, color="black", alpha=0.45, lw=0.8))
    images = []
    for frame in range(150):
        center = (states[0][frame, 0, 0] + states[0][frame, target, 0]) / 2
        status = (
            "no handover"
            if event is None
            else f"handover {first*.04:.2f}s: {REASONS[event['reason']]}"
        )
        title.set_text(
            f"{TITLES[spec['kind']]} | Test rec {spec['recording']:02d}, row {spec['row']} | t={frame*.04:.2f}s\n{status}"
        )
        for arm, ax in enumerate(roads):
            s = states[arm][frame]
            owned = (
                np.r_[False, new["takeover_owned"][frame - 1]]
                if arm and frame
                else np.zeros(7, bool)
            )
            colors = [
                (
                    EGO_COLOR
                    if j == 0
                    else (
                        SLOT_COLOR
                        if owned[j]
                        else TARGET_COLOR if j == target else "#bdbdbd"
                    )
                )
                for j in range(7)
            ]
            visible = base["valid"] & (abs(s[:, 0] - center) <= 80)
            bodies[arm].set_verts(vehicle_polygons(s, base["dimensions"])[visible])
            bodies[arm].set_facecolors(np.array(colors)[visible])
            trail = [
                states[arm][max(0, frame - 50) : frame + 1, j, :2]
                for j in np.flatnonzero(visible)
            ]
            trails[arm].set_segments(trail)
            trails[arm].set_colors(np.array(colors)[visible])
            future = base["plan"][frame : min(frame + 38, 149), spec["actor"]]
            plans[arm].set_data(future[:, 0], future[:, 1])
            crash = (
                np.zeros(7, bool)
                if frame == 0
                else (base if arm == 0 else new)["crashed"][frame - 1]
            )
            for j, text in enumerate(labels[arm]):
                text.set_visible(bool(visible[j]))
                text.set_position((s[j, 0], s[j, 1] + base["dimensions"][j, 1] * 0.7))
                text.set_text(
                    ("ego" if j == 0 else f"NPC {j}" + (" *" if j == target else ""))
                    + (" COLLISION" if crash[j] else "")
                )
            ax.set_xlim(center - 80, center + 80)
            ax.set_title(
                ("Pure tracking" if arm == 0 else "Bridge enabled")
                + (" | collision recorded" if crash.any() else ""),
                fontsize=10,
                loc="left",
            )
        for cursor in cursors:
            cursor.set_xdata([frame * 0.04, frame * 0.04])
        fig.canvas.draw()
        image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
        images.append(image.quantize(colors=192))
    images[0].save(
        destination,
        save_all=True,
        append_images=images[1:],
        duration=40,
        loop=0,
        disposal=2,
        optimize=False,
    )
    plt.close(fig)
    return dict(
        file=destination.name,
        case=spec["case"],
        row=spec["row"],
        seed=case["seed"],
        kind=spec["kind"],
        focus_actor=spec["actor"],
        first_target_handover_s=None if first is None else first * 0.04,
        collision=bool(new["crashed"].any()),
        frames=150,
        fps=25,
        playback_speed=1,
    )


def main():
    OUTPUT.mkdir(exist_ok=True)
    specs = load_json(SOURCE / "intervention_cases.json")["cases"]
    report = load_json(SOURCE / "interventions.json")
    seed = report["seeds"][0]
    movies = []
    for kind in (
        "braking",
        "acceleration",
        "cut_in",
        "corridor_release",
        "free_flow",
        "null",
    ):
        group = [s for s in specs if s["kind"] == kind]
        # Demonstration selection is explicit; the experiment report retains all cases.
        responsive = [
            s
            for s in group
            if any(
                r["case"] == s["case"]
                and r["seed"] == seed
                and r["arm"] == "takeover"
                and any(e["actor"] == s["actor"] for e in r["events"])
                for r in report["records"]
            )
        ]
        selected = (responsive or group)[0]
        case = read_case(selected, seed, report["records"])
        movies.append(render_gif(case, OUTPUT / f"{kind}.gif"))
        print("Rendered", kind, selected["row"], flush=True)
    # The same lane departure also affects another lane: expose its first
    # conflict admission without adding an experiment or suppressing collisions.
    lateral = next(
        r
        for r in report["records"]
        if r["arm"] == "takeover"
        and r["seed"] == seed
        and r["stimulus"] == "corridor_release"
        and any(
            e["reason"] == 2 and e["actor"] != r["target_actor"] for e in r["events"]
        )
    )
    focus = next(
        e["actor"]
        for e in lateral["events"]
        if e["reason"] == 2 and e["actor"] != lateral["target_actor"]
    )
    spec = dict(
        next(s for s in specs if s["case"] == lateral["case"]),
        actor=focus,
        kind="lateral_conflict",
    )
    movies.append(
        render_gif(
            read_case(spec, seed, report["records"]), OUTPUT / "lateral_conflict.gif"
        )
    )
    save_json(
        dict(
            movies=movies,
            format_reference="results/evt/playbacks",
            scope="Scripted Test interventions, not ADS algorithm performance",
            selection="First case with a target handover in each kind, otherwise first assigned case. Lateral conflict shows the first other-lane conflict admission in a corridor-release case. All cases and three seeds remain in the experiment report.",
        ),
        OUTPUT / "manifest.json",
    )


if __name__ == "__main__":
    main()

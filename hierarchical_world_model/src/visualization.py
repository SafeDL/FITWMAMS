"""Paper figures and reconstruction playbacks for the final world model."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from normalizing_flow.src.constraints import derived_modes

from tools.plot_style import (
    GENERATED_COLOR,
    PAPER_SIX_PANEL_FIGSIZE,
    REAL_COLOR,
    get_pyplot,
    style_axes,
)
from world_model.src.core.utils import ensure_dir, load_json, save_json, select_device

from .data import ANCHOR_INDEX, prepare_experiment_data
from .evaluation import rollout
from .planner import frozen_diffusion_plans
from .train import load_checkpoint

EGO_COLOR = "#D62728"
DIFFUSION_COLOR = "#1f78b4"
LOGGED_REFERENCE_COLOR = "#f7f7f7"
ROAD_COLOR = "#6f7378"
LANE_COLOR = "#ffffff"
DT_S = 0.04


def _summary_figures(output: Path) -> dict[str, str]:
    plt = get_pyplot()
    evaluation = load_json(output / "evaluation.json")
    if evaluation.get("evaluation_schema_version") != 2:
        raise RuntimeError(
            "evaluation.json predates the complete three-objective figure schema; "
            "run evaluate.py before visualizing results"
        )
    randomness = load_json(output / "randomness_ablation.json")
    calibration = load_json(output / "natural_response_calibration.json")
    history = load_json(output / "training_history.json")["epochs"]
    figures = ensure_dir(output / "figures")

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
    epoch = [row["epoch"] for row in history]
    axes[0].plot(epoch, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epoch, [row["validation_loss"] for row in history], label="validation")
    axes[0].set(title="Training objective", xlabel="Epoch", ylabel="Loss")
    axes[0].legend(frameon=False)
    axes[1].plot(
        epoch,
        [row["validation_closed_loop_ADE_m"] for row in history],
        marker="o",
        label="ADE",
    )
    axes[1].plot(
        epoch,
        [row["validation_closed_loop_FDE_m"] for row in history],
        marker="s",
        label="FDE",
    )
    axes[1].set(title="Closed-loop validation", xlabel="Epoch", ylabel="Error (m)")
    axes[1].legend(frameon=False, ncol=2)
    for axis in axes:
        style_axes(axis)
    training_path = figures / "training_diagnostics.png"
    figure.savefig(training_path, dpi=300)
    plt.close(figure)

    factual_fidelity = evaluation["factual_fidelity"]
    names = ("Open-loop", "No long-horizon", "5-frame history", "25 Hz HiQR")
    factual = [
        factual_fidelity["open_loop_diffusion"],
        factual_fidelity["without_long_horizon_constraint"],
        factual_fidelity["history_ablation"]["5"],
        factual_fidelity["diffusion_guided_hiqr"],
    ]
    figure, axes = plt.subplots(
        2, 3, figsize=PAPER_SIX_PANEL_FIGSIZE, constrained_layout=True
    )
    x = np.arange(len(names))
    axes[0, 0].bar(x - 0.18, [item["ADE_m"] for item in factual], 0.36, label="ADE")
    axes[0, 0].bar(x + 0.18, [item["FDE_m"] for item in factual], 0.36, label="FDE")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticks(x, names, rotation=25, ha="right")
    axes[0, 0].set(title="Factual fidelity", ylabel="Error (m, log scale)")
    axes[0, 0].legend(frameon=False)

    history = factual_fidelity["history_ablation"]
    history_frames = (5, 10, 15, 25)
    axes[0, 1].plot(
        history_frames,
        [history[str(value)]["ADE_m"] for value in history_frames],
        marker="o",
        label="ADE",
    )
    axes[0, 1].plot(
        history_frames,
        [history[str(value)]["FDE_m"] for value in history_frames],
        marker="s",
        label="FDE",
    )
    axes[0, 1].set(
        title="History robustness",
        xlabel="Observed history (frames)",
        ylabel="Error (m)",
        xticks=history_frames,
    )
    axes[0, 1].legend(frameon=False)

    compared = ("Deterministic", "Diffusion", "Response", "Full")
    quadrants = randomness["quadrants"]
    distributions = tuple(
        quadrants[name]
        for name in (
            "fully_deterministic",
            "diffusion_only_random",
            "response_only_random",
            "full_hierarchical_random",
        )
    )
    x = np.arange(len(compared))
    axes[0, 2].bar(
        x - 0.18,
        [item["energy_score_m"] for item in distributions],
        0.36,
        label="energy score",
        color=REAL_COLOR,
    )
    axes[0, 2].bar(
        x + 0.18,
        [item["terminal_pairwise_distance_m"] for item in distributions],
        0.36,
        label="terminal diversity",
        color=GENERATED_COLOR,
    )
    axes[0, 2].set_xticks(x, compared, rotation=20, ha="right")
    axes[0, 2].set(title="Stochasticity", ylabel="Distance (m)")
    axes[0, 2].legend(frameon=False)

    distribution = evaluation["distribution_stochasticity"]
    distribution_names = ("speed", "ax", "jx", "jy", "gap", "TTC")
    distribution_ks = (
        distribution["motion_distribution"]["speed"]["KS"],
        distribution["motion_distribution"]["ax"]["KS"],
        distribution["jerk_resolution_diagnostic"]["windowed_0p2s"]["jx"]["KS"],
        distribution["jerk_resolution_diagnostic"]["windowed_0p2s"]["jy"]["KS"],
        distribution["risk_distribution"]["gap_m"]["KS"],
        distribution["risk_distribution"]["TTC_s"]["KS"],
    )
    axes[1, 0].bar(distribution_names, distribution_ks, color=GENERATED_COLOR)
    axes[1, 0].axhline(0.10, color="#666666", linestyle="--", linewidth=1.0)
    axes[1, 0].set(
        title="Natural-driving distribution error",
        ylabel="KS distance",
        ylim=(0.0, max(0.12, 1.12 * max(distribution_ks))),
    )

    interventions = evaluation["intervention_effectiveness"]
    kinds = ("brake", "accelerate", "left")
    longitudinal = ("brake", "accelerate")
    quality = np.asarray(
        [
            [interventions[name]["direction_success_rate"] for name in longitudinal],
            [interventions[name]["dose_monotonicity_rate"] for name in longitudinal],
            [
                interventions[name]["response_within_natural_p10_p90_rate"]
                for name in longitudinal
            ],
        ]
    )
    x = np.arange(len(longitudinal))
    width = 0.24
    for index, label in enumerate(("direction", "monotonicity", "natural P10-P90")):
        axes[1, 1].bar(x + (index - 1) * width, quality[index], width, label=label)
    axes[1, 1].set(
        title="Intervention validity",
        ylabel="Rate",
        ylim=(0.0, 1.05),
        xticks=x,
        xticklabels=longitudinal,
    )
    axes[1, 1].legend(frameon=False, fontsize=8)

    axes[1, 2].bar(
        np.arange(3) - 0.18,
        [interventions[name]["near_response_magnitude"] for name in kinds],
        0.36,
        label="near",
        color=GENERATED_COLOR,
    )
    axes[1, 2].bar(
        np.arange(3) + 0.18,
        [interventions[name]["far_response_magnitude"] for name in kinds],
        0.36,
        label="far",
        color="#BDBDBD",
    )
    axes[1, 2].set_xticks(np.arange(3), kinds)
    axes[1, 2].set(title="Intervention locality", ylabel="Mean action change")
    axes[1, 2].legend(frameon=False)
    for axis in axes.flat:
        style_axes(axis)
    summary_path = figures / "three_objective_evaluation.png"
    figure.savefig(summary_path, dpi=300)
    plt.close(figure)

    strata = factual_fidelity["event_strata"]
    names = tuple(strata)
    figure, axis = plt.subplots(figsize=(7.0, 3.6), constrained_layout=True)
    x = np.arange(len(names))
    axis.bar(
        x - 0.18,
        [strata[name]["ADE_m"] for name in names],
        0.36,
        label="ADE",
        color=REAL_COLOR,
    )
    axis.bar(
        x + 0.18,
        [strata[name]["FDE_m"] for name in names],
        0.36,
        label="FDE",
        color=GENERATED_COLOR,
    )
    axis.set(
        title="Conditional reconstruction across event strata",
        ylabel="Error (m)",
        xticks=x,
        xticklabels=("all natural", "EVT-labelled", "semantic cut-in"),
    )
    axis.legend(frameon=False)
    style_axes(axis)
    event_path = figures / "event_fidelity.png"
    figure.savefig(event_path, dpi=300)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True)
    horizons = np.asarray(calibration["horizons_s"], np.float32)
    diagnostics = calibration["horizon_diagnostics"]
    for axis, name in zip(axes, ("brake", "accelerate", "lane_change")):
        quantiles = np.asarray(
            [
                diagnostics[f"{horizon:.1f}s"][name]["effect_p10_p50_p90_mps2"]
                for horizon in horizons
            ]
        )
        axis.fill_between(
            horizons,
            quantiles[:, 0],
            quantiles[:, 2],
            color=REAL_COLOR,
            alpha=0.25,
            label="P10-P90",
        )
        axis.plot(horizons, quantiles[:, 1], marker="o", color=REAL_COLOR, label="P50")
        axis.set(
            title=name.replace("_", " "),
            xlabel="Response horizon (s)",
            ylabel="Matched effect (m/s²)",
            xticks=horizons,
        )
        axis.legend(frameon=False)
        style_axes(axis)
    calibration_path = figures / "natural_response_calibration.png"
    figure.savefig(calibration_path, dpi=300)
    plt.close(figure)

    temporal = factual_fidelity["temporal_error"]
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 3.5), constrained_layout=True)
    for axis, key, label in zip(
        axes,
        ("ADE_m", "P95_displacement_error_m", "speed_MAE_mps"),
        (
            "Mean displacement error (m)",
            "P95 displacement error (m)",
            "Speed MAE (m/s)",
        ),
    ):
        for name, color in (
            ("open_loop_diffusion", DIFFUSION_COLOR),
            ("diffusion_guided_hiqr", GENERATED_COLOR),
        ):
            values = temporal[name]
            axis.plot(
                values["time_s"], values[key], color=color, label=name.replace("_", " ")
            )
        axis.set(xlabel="Prediction horizon (s)", ylabel=label)
        style_axes(axis)
    axes[0].set_title("Closed-loop drift")
    axes[1].set_title("Tail reconstruction error")
    axes[2].set_title("Velocity recovery")
    axes[0].legend(frameon=False, fontsize=8)
    factual_temporal_path = figures / "factual_temporal_error.png"
    figure.savefig(factual_temporal_path, dpi=300)
    plt.close(figure)

    realism = distribution["highd_adapted_realism"]
    labels = {
        "speed_mps": "speed (m/s)",
        "acceleration_magnitude_mps2": "acceleration magnitude (m/s²)",
        "yaw_rate_rps": "yaw rate (rad/s)",
        "yaw_acceleration_rps2": "yaw acceleration (rad/s²)",
        "nearest_object_distance_m": "nearest-object distance (m)",
        "gap_m": "same-lane front gap (m)",
        "TTC_s": "TTC (s)",
        "collision_incidence": "collision incidence",
    }
    figure, axes = plt.subplots(2, 4, figsize=(14.4, 6.8), constrained_layout=True)
    for axis, (name, histogram) in zip(axes.flat, realism["components"].items()):
        edges = np.asarray(histogram["bin_edges"], np.float32)
        centers = 0.5 * (edges[:-1] + edges[1:])
        axis.step(
            centers,
            histogram["real_probability"],
            where="mid",
            color=REAL_COLOR,
            label="highD",
        )
        axis.step(
            centers,
            histogram["generated_probability"],
            where="mid",
            color=GENERATED_COLOR,
            label="model",
        )
        axis.set(title=labels[name], xlabel="value", ylabel="probability")
        axis.text(
            0.98,
            0.93,
            f"TV={histogram['total_variation']:.3f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
        style_axes(axis)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("highD-adapted distribution realism (not an official WOSAC score)")
    realism_path = figures / "highd_adapted_distribution_realism.png"
    figure.savefig(realism_path, dpi=300)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    for name, values in quadrants.items():
        axis.scatter(
            values["terminal_pairwise_distance_m"],
            values["energy_score_m"],
            s=56,
            label=name.replace("_", " "),
        )
    axis.set(
        title="Quality–diversity trade-off",
        xlabel="Terminal pairwise distance (m, higher is diverse)",
        ylabel="Energy score (m, lower is better)",
    )
    axis.legend(frameon=False, fontsize=8)
    style_axes(axis)
    diversity_path = figures / "quality_diversity_tradeoff.png"
    figure.savefig(diversity_path, dpi=300)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(13.2, 6.6), constrained_layout=True)
    for column, name in enumerate(("brake", "accelerate", "left")):
        curve = interventions[name]["dose_response"]
        top, bottom = axes[:, column]
        horizons = np.asarray(curve["horizons_s"], np.float32)
        natural = curve["natural_p10_p50_p90"]
        if natural is not None:
            quantiles = np.asarray(
                [natural[f"{horizon:.1f}s"] for horizon in horizons], np.float32
            )
            top.fill_between(
                horizons,
                quantiles[:, 0],
                quantiles[:, 2],
                color=REAL_COLOR,
                alpha=0.20,
                label="highD P10–P90",
            )
            top.plot(
                horizons,
                quantiles[:, 1],
                color=REAL_COLOR,
                linestyle=":",
                label="highD P50",
            )
        for dose, values in curve["doses"].items():
            top.plot(
                horizons,
                [values[f"{horizon:.1f}s"]["mean"] for horizon in horizons],
                marker="o",
                label=f"dose={dose}",
            )
        profile = interventions[name]["response_magnitude_profile"]
        bottom.plot(
            profile["time_s"], profile["near"], color=GENERATED_COLOR, label="near"
        )
        bottom.plot(profile["time_s"], profile["far"], color="#777777", label="far")
        top.set(
            title=name.replace("_", " "),
            xlabel="response horizon (s)",
            ylabel=curve["metric"],
        )
        bottom.set(
            xlabel="time after intervention (s)", ylabel="action-change magnitude"
        )
        for axis in (top, bottom):
            style_axes(axis)
        top.legend(frameon=False, fontsize=7)
        bottom.legend(frameon=False, fontsize=8)
    intervention_path = figures / "intervention_dose_and_locality.png"
    figure.savefig(intervention_path, dpi=300)
    plt.close(figure)
    return {
        "training_diagnostics": str(training_path.relative_to(output)),
        "three_objective_evaluation": str(summary_path.relative_to(output)),
        "event_fidelity": str(event_path.relative_to(output)),
        "natural_response_calibration": str(calibration_path.relative_to(output)),
        "factual_temporal_error": str(factual_temporal_path.relative_to(output)),
        "highd_adapted_distribution_realism": str(realism_path.relative_to(output)),
        "quality_diversity_tradeoff": str(diversity_path.relative_to(output)),
        "intervention_dose_and_locality": str(intervention_path.relative_to(output)),
    }


def _select_examples(experiment: Any) -> np.ndarray:
    rows = experiment.test_rows
    flow_rows = experiment.bundle.flow_row_for_sequence[rows]
    flow = experiment.bundle.flow_arrays
    modes = derived_modes(
        np.asarray(flow["trajectory_constraint"])[flow_rows],
        np.asarray(flow["slot_mask"])[flow_rows],
    )
    lane_change = (modes[..., 1] != 0).any(axis=1)
    evt = np.asarray(experiment.bundle.arrays["is_evt_tail"])[rows].astype(bool)
    slot_count = np.asarray(flow["slot_mask"])[flow_rows].sum(axis=1)
    selected: list[int] = []
    for candidates in (
        np.flatnonzero(lane_change),
        np.flatnonzero(evt),
        np.argsort(-slot_count),
    ):
        selected.extend(
            int(index) for index in candidates if int(index) not in selected
        )
        if len(selected) >= 3:
            break
    if len(selected) < 3:
        raise RuntimeError("the test split does not contain three distinct examples")
    return rows[np.asarray(selected[:3], np.int64)]


def _draw_vehicle(
    axis: Any,
    state: np.ndarray,
    *,
    color: str,
    label: str | None = None,
    filled: bool,
    alpha: float,
) -> None:
    x, y, vx, vy = (float(value) for value in state[:4])
    heading = float(np.arctan2(vy, vx)) if np.hypot(vx, vy) > 1.0e-6 else 0.0
    patch = Rectangle(
        (-2.4, -0.9),
        4.5,
        1.8,
        facecolor=color if filled else "none",
        edgecolor="black",
        linewidth=0.8,
        alpha=alpha,
        zorder=6 if filled else 5,
    )
    patch.set_transform(Affine2D().rotate(heading).translate(x, y) + axis.transData)
    axis.add_patch(patch)
    if label is not None:
        axis.text(
            x,
            y + 1.9,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            color="black",
            zorder=7,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.2},
        )


def _draw_lane_markings(axis: Any) -> None:
    for lane in (-7.5, -5.625, -1.875, 1.875, 5.625, 7.5):
        outer = abs(lane) > 7.0
        axis.axhline(
            lane,
            color=LANE_COLOR,
            linewidth=0.8,
            linestyle="-" if outer else "--",
            alpha=0.45 if outer else 0.28,
            zorder=0,
        )


def _reconstruction_outputs(
    config: dict[str, Any], config_dir: Path, output: Path
) -> dict[str, Any]:
    plt = get_pyplot()
    device = select_device(config["training"].get("device", "auto"))
    experiment = prepare_experiment_data(config, config_dir)
    rows = _select_examples(experiment)

    with tempfile.TemporaryDirectory(prefix="hierarchical_wm_visual_") as cache:
        plans = frozen_diffusion_plans(
            experiment.bundle,
            rows,
            checkpoint=config["paths"]["diffusion_checkpoint"],
            output_dir=cache,
            device=device,
            batch_size=32,
            ddim_steps=20,
            experiment_scope="visualization",
        )

    states = np.asarray(experiment.bundle.arrays["agent_states"][rows], np.float32)
    valid = np.asarray(experiment.bundle.arrays["agent_valid"][rows], bool)
    sequence_id = np.asarray(experiment.bundle.arrays["sequence_id"])[rows]
    model, _ = load_checkpoint(
        config["paths"]["evaluation_checkpoint"], device=device
    )
    generated = rollout(
        model,
        states,
        valid,
        plans,
        np.asarray(experiment.bundle.arrays["map_polylines"])[rows],
        np.asarray(experiment.bundle.arrays["map_polyline_valid"])[rows],
        device=device,
        history_frames=25,
        motion_seed=None,
    ).states

    logged = states[:, ANCHOR_INDEX:174]
    target = logged[:, 1:]
    generated_full = np.concatenate((logged[:, :1], generated), axis=1)
    figures = ensure_dir(output / "figures")
    playbacks = ensure_dir(output / "playbacks")

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    for example, axis in enumerate(axes):
        active = valid[example, ANCHOR_INDEX, 1:]
        slots = np.flatnonzero(active)
        for agent in slots:
            axis.plot(
                logged[example, :, agent + 1, 0],
                logged[example, :, agent + 1, 1],
                color=REAL_COLOR,
                alpha=0.75,
                label="highD" if agent == slots[0] else None,
            )
            axis.plot(
                generated_full[example, :, agent + 1, 0],
                generated_full[example, :, agent + 1, 1],
                color=GENERATED_COLOR,
                linestyle="--",
                label="model" if agent == slots[0] else None,
            )
        axis.set(
            title=f"Test sequence {int(rows[example])}",
            xlabel="Longitudinal position (m)",
            ylabel="Lateral position (m)",
        )
        axis.legend(frameon=False)
        style_axes(axis)
    trajectory_path = figures / "trajectory_reconstruction.png"
    figure.savefig(trajectory_path, dpi=300)
    plt.close(figure)

    frame_stride = 1
    fps = 25
    duration_ms = max(int(round(1000.0 * frame_stride * DT_S)), 1)
    playback_manifest = []
    playback_paths = []
    for example, row in enumerate(rows):
        path = playbacks / f"reconstruction_{int(row)}.gif"
        slots = np.flatnonzero(valid[example, ANCHOR_INDEX, 1:]).astype(int)
        if len(slots) == 0:
            raise RuntimeError(
                f"selected sequence has no active background slots: row={int(row)}"
            )
        slot_dist = np.linalg.norm(
            target[example, :, 1:, :] - generated[example, :, 1:, :], axis=-1
        )
        cumulative_ade = float(slot_dist[:, slots].mean())
        cumulative_fde = float(slot_dist[-1, slots].mean())
        focus_slot = int(slots[np.argmax(slot_dist[-1, slots])])
        with imageio.get_writer(path, mode="I", duration=duration_ms, loop=0) as writer:
            figure, axis = plt.subplots(figsize=(12.0, 4.8), dpi=100)
            figure.subplots_adjust(left=0.065, right=0.965, bottom=0.18, top=0.83)
            frames = np.arange(0, logged.shape[1], frame_stride)
            if frames[-1] != logged.shape[1] - 1:
                frames = np.append(frames, logged.shape[1] - 1)
            for frame in frames:
                axis.clear()
                axis.set_facecolor(ROAD_COLOR)
                center_x = float(
                    0.5
                    * (
                        logged[example, frame, 0, 0]
                        + generated_full[example, frame, focus_slot + 1, 0]
                    )
                )
                _draw_lane_markings(axis)
                axis.set(
                    xlim=(center_x - 80.0, center_x + 80.0),
                    ylim=(-8.2, 8.2),
                    xlabel="x [m]",
                    ylabel="y [m]",
                    aspect="equal",
                    title=(
                        f"{sequence_id[example]} | t={frame * DT_S:.2f}s | "
                        f"ADE/FDE={cumulative_ade:.2f}/{cumulative_fde:.2f} m"
                    ),
                )
                ego_start = max(0, frame - 45)
                axis.plot(
                    logged[example, ego_start : frame + 1, 0, 0],
                    logged[example, ego_start : frame + 1, 0, 1],
                    color=EGO_COLOR,
                    linewidth=1.8,
                    alpha=0.78,
                    label="ego (logged replay)",
                )
                for slot in slots:
                    axis.plot(
                        logged[example, ego_start : frame + 1, slot + 1, 0],
                        logged[example, ego_start : frame + 1, slot + 1, 1],
                        color="#d9d9d9",
                        linestyle=":",
                        linewidth=1.2,
                        alpha=0.9,
                    )
                    axis.plot(
                        generated_full[example, ego_start : frame + 1, slot + 1, 0],
                        generated_full[example, ego_start : frame + 1, slot + 1, 1],
                        color=DIFFUSION_COLOR,
                        linewidth=1.5,
                        alpha=0.86,
                    )
                    _draw_vehicle(
                        axis,
                        logged[example, frame, slot + 1],
                        color=LOGGED_REFERENCE_COLOR,
                        label=(f"b{int(slot)+1}" if slot == focus_slot else None),
                        filled=False,
                        alpha=0.9,
                    )
                    _draw_vehicle(
                        axis,
                        generated_full[example, frame, slot + 1],
                        color=DIFFUSION_COLOR,
                        label=(
                            f"diffusion-guided HiQR b{int(slot) + 1}"
                            if slot == focus_slot
                            else None
                        ),
                        filled=True,
                        alpha=0.52,
                    )
                _draw_vehicle(
                    axis,
                    logged[example, frame, 0],
                    color=EGO_COLOR,
                    label="ego (logged)",
                    filled=True,
                    alpha=0.9,
                )
                axis.text(
                    0.01,
                    0.02,
                    "red: logged ego replay | blue: diffusion-guided HiQR background | "
                    "white outline/dotted: highD reference",
                    transform=axis.transAxes,
                    fontsize=7.5,
                    va="bottom",
                    ha="left",
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.78,
                        "edgecolor": "none",
                        "pad": 1.2,
                    },
                )
                axis.text(
                    0.01,
                    0.94,
                    "oracle state-knot-conditioned reconstruction",
                    transform=axis.transAxes,
                    fontsize=7.5,
                    va="top",
                    ha="left",
                    color="#1d4ed8",
                    fontweight="bold",
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.78,
                        "edgecolor": "none",
                        "pad": 1.2,
                    },
                )
                axis.tick_params(labelsize=8)

                figure.canvas.draw()
                rgba = np.asarray(figure.canvas.buffer_rgba())
                writer.append_data(np.asarray(rgba[:, :, :3], dtype=np.uint8).copy())
            plt.close(figure)
        playback_paths.append(str(path.relative_to(output)))
        playback_manifest.append(
            {
                "row": int(row),
                "focus_slot": int(focus_slot) + 1,
                "playback_frames": int(len(frames)),
                "frame_stride": int(frame_stride),
                "fps": int(fps),
                "duration_ms": int(duration_ms),
                "gif": str(path.relative_to(output)),
            }
        )

    save_json(
        {
            "role": "hierarchical reconstruction playbacks",
            "checkpoint": "checkpoints/final_world_model.pt",
            "frame_stride": int(frame_stride),
            "fps": int(fps),
            "episodes": playback_manifest,
        },
        playbacks / "playback_manifest.json",
    )

    return {
        "trajectory_reconstruction": str(trajectory_path.relative_to(output)),
        "playbacks": playback_paths,
        "rows": [int(row) for row in rows],
    }


def build_visualizations(config: dict[str, Any], *, config_dir: Path) -> dict[str, Any]:
    output = Path(config["paths"]["output_dir"])
    return {
        **_summary_figures(output),
        **_reconstruction_outputs(config, config_dir, output),
    }

#!/usr/bin/env python3
"""Render held-out constrained-reconstruction summaries and GIF playbacks.

The ego trajectory is always the logged highD trajectory.  The diffusion model
reconstructs the six stable background slots from C0 and three declared
background state knots; logged background boxes are drawn as a visual reference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import (  # noqa: E402
    ANCHOR_INDEX,
    BackgroundTrajectoryDataset,
    HORIZON_STEPS,
    load_data_bundle,
    semantic_cutin_agents,
)
from diffusion.src.sampling import decode_background_latents  # noqa: E402
from diffusion.src.train import load_checkpoint  # noqa: E402
from world_model.src.core.utils import (  # noqa: E402
    ensure_dir,
    load_json,
    load_yaml,
    save_json,
    select_device,
)

DEFAULT_CONFIG = ROOT / "diffusion/configs/highd_background_diffusion.yaml"
DT_S = 0.04
EGO_COLOR = "#e31a1c"
DIFFUSION_COLOR = "#1f78b4"
LOGGED_REFERENCE_COLOR = "#f7f7f7"
REAL_COLOR = "#4C78A8"
GENERATED_COLOR = "#F58518"
ROAD_COLOR = "#6f7378"
LANE_COLOR = "#ffffff"


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    return values, np.arange(1, len(values) + 1, dtype=np.float64) / len(values)


def _nearest_quantile(rows: np.ndarray, errors: np.ndarray, quantile: float) -> int:
    ordered = rows[np.argsort(errors[rows], kind="mergesort")]
    return int(
        ordered[min(int(round(quantile * (len(ordered) - 1))), len(ordered) - 1)]
    )


def _select_playbacks(
    evaluation: dict[str, np.ndarray], bundle: Any
) -> list[tuple[str, int, str]]:
    """Select transparent examples: typical, difficult, and natural cut-in."""
    rows = np.asarray(evaluation["row_index"], dtype=np.int64)
    sample_fde = np.asarray(evaluation["sample_fde_m"], dtype=np.float64).mean(axis=1)
    is_tail = np.asarray(evaluation["is_evt_tail"], dtype=bool)
    states = np.asarray(bundle.arrays["agent_states"][rows, ANCHOR_INDEX:174])
    valid = np.asarray(bundle.arrays["agent_valid"][rows, ANCHOR_INDEX:174], dtype=bool)
    strict_cutin = semantic_cutin_agents(states, valid).any(axis=1)
    ordinary = np.flatnonzero(~is_tail & ~strict_cutin)
    cutin = np.flatnonzero(strict_cutin)
    if len(ordinary) < 3:
        ordinary = np.arange(len(rows))
    selected = [
        (
            "typical_natural_q50",
            _nearest_quantile(ordinary, sample_fde, 0.50),
            "median-FDE ordinary natural test episode",
        ),
        (
            "challenging_natural_q90",
            _nearest_quantile(ordinary, sample_fde, 0.90),
            "90th-percentile-FDE ordinary natural test episode",
        ),
    ]
    if len(cutin):
        selected.append(
            (
                "semantic_cutin_q50",
                _nearest_quantile(cutin, sample_fde, 0.50),
                "median-FDE strict semantic cut-in within the natural test set",
            )
        )
    return [
        (name, int(rows[local]), description) for name, local, description in selected
    ]


def _decode_one(
    model: Any,
    state: dict[str, Any],
    bundle: Any,
    row: int,
    *,
    device: torch.device,
    ddim_steps: int,
    guidance_scale: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    dataset = BackgroundTrajectoryDataset(
        bundle, np.asarray([row]), state["dataset_contract"]
    )
    item = dataset[0]
    generator = torch.Generator(device=device).manual_seed(int(seed))
    latents = torch.randn(
        (1, model.config.horizon_steps, model.config.target_dim),
        generator=generator,
        device=device,
    )
    decoded = decode_background_latents(
        model,
        item["condition"][None].to(device),
        item["target_mask"][None].to(device),
        item["c0_states"].numpy()[1:],
        state["dataset_contract"],
        latents,
        trajectory_reference=item["trajectory_reference"].numpy(),
        inference_steps=ddim_steps,
        guidance_scale=guidance_scale,
        x0_clip_abs=model.config.x0_clip_abs,
    )
    target = item["future_states"].numpy()[:, 1:, :2]
    active = decoded["active_background"]
    distance = np.linalg.norm(decoded["background_positions_xy"][0] - target, axis=-1)
    active_slots = np.flatnonzero(active)
    focus_slot = int(active_slots[np.argmax(distance[-1, active_slots])])
    metrics = {
        "ADE_m": float(distance[:, active].mean()),
        "FDE_m": float(distance[-1, active].mean()),
        "active_background_slots": [int(slot) + 1 for slot in np.flatnonzero(active)],
        "focus_background_slot": focus_slot + 1,
        "row_index": int(row),
        "sequence_id": str(bundle.arrays["sequence_id"][row]),
    }
    return decoded, {"item": item, "metrics": metrics}


def _draw_vehicle(
    axis: Any,
    state: np.ndarray,
    *,
    color: str,
    filled: bool,
    label: str | None = None,
) -> None:
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import Affine2D

    values = np.asarray(state, dtype=np.float32)
    heading = (
        float(np.arctan2(values[3], values[2]))
        if np.hypot(values[2], values[3]) > 1.0e-6
        else 0.0
    )
    rectangle = Rectangle(
        (-2.25, -0.9),
        4.5,
        1.8,
        linewidth=0.8,
        edgecolor="black",
        facecolor=color if filled else "none",
        alpha=0.94 if filled else 0.9,
        label=label,
        zorder=6 if filled else 5,
    )
    rectangle.set_transform(
        Affine2D().rotate(heading).translate(float(values[0]), float(values[1]))
        + axis.transData
    )
    axis.add_patch(rectangle)
    if label:
        axis.text(
            float(values[0]),
            float(values[1]) + 1.65,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            color="black",
            zorder=7,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.2},
        )


def _style_axes(axis: Any) -> None:
    axis.grid(True, color="#D9D9D9", linewidth=0.45, alpha=0.65)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(direction="out", length=3.0, width=0.7)


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


def _render_playback(
    path: Path,
    decoded: dict[str, np.ndarray],
    record: dict[str, Any],
    *,
    title: str,
    frame_stride: int,
    fps: int,
) -> None:
    """Write a road-view GIF in the canonical natural-driving playback style."""
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    item = record["item"]
    c0 = item["c0_states"].numpy()
    future = item["future_states"].numpy()
    ego = np.concatenate((c0[None, 0], future[:, 0]), axis=0)
    logged = np.concatenate((c0[None, 1:], future[:, 1:]), axis=0)
    generated = np.concatenate((c0[None, 1:], decoded["background_states"][0]), axis=0)
    active = decoded["active_background"]
    slots = np.flatnonzero(active)
    focus_slot = int(record["metrics"]["focus_background_slot"]) - 1
    frames = np.arange(0, HORIZON_STEPS + 1, max(int(frame_stride), 1))
    if frames[-1] != HORIZON_STEPS:
        frames = np.append(frames, HORIZON_STEPS)
    figure, axis = plt.subplots(figsize=(12.0, 4.8), dpi=100)
    figure.subplots_adjust(left=0.065, right=0.965, bottom=0.18, top=0.83)
    duration_ms = max(
        int(round(1000.0 * max(int(frame_stride), 1) / max(float(fps), 1.0e-6))),
        1,
    )

    def draw_frame(frame: int) -> None:
        axis.clear()
        axis.set_facecolor(ROAD_COLOR)
        center_x = 0.5 * (ego[frame, 0] + generated[frame, focus_slot, 0])
        _draw_lane_markings(axis)
        axis.set(
            xlim=(center_x - 80.0, center_x + 80.0),
            ylim=(-8.2, 8.2),
            xlabel="x [m]",
            ylabel="y [m]",
            title=(
                f"{record['metrics']['sequence_id']} | {title} | t={frame * DT_S:.2f}s | "
                f"ADE/FDE={record['metrics']['ADE_m']:.2f}/{record['metrics']['FDE_m']:.2f} m"
            ),
            aspect="equal",
        )
        start = max(0, frame - 50)
        axis.plot(
            ego[start : frame + 1, 0],
            ego[start : frame + 1, 1],
            color=EGO_COLOR,
            linewidth=1.6,
            alpha=0.78,
        )
        for slot in slots:
            axis.plot(
                logged[start : frame + 1, slot, 0],
                logged[start : frame + 1, slot, 1],
                color="#d9d9d9",
                linestyle=":",
                linewidth=1.25,
                alpha=0.9,
                zorder=2,
            )
            axis.plot(
                generated[start : frame + 1, slot, 0],
                generated[start : frame + 1, slot, 1],
                color=DIFFUSION_COLOR,
                linewidth=1.55,
                alpha=0.86,
                zorder=3,
            )
            _draw_vehicle(
                axis,
                logged[frame, slot],
                color=LOGGED_REFERENCE_COLOR,
                filled=False,
            )
            _draw_vehicle(
                axis,
                generated[frame, slot],
                color=DIFFUSION_COLOR,
                filled=True,
                label=(f"diffusion b{slot + 1}" if slot == focus_slot else None),
            )
        _draw_vehicle(
            axis, ego[frame], color=EGO_COLOR, filled=True, label="ego (logged)"
        )
        axis.text(
            0.01,
            0.02,
            "red: logged ego replay | blue: diffusion background | "
            "white outline/dotted: highD reference",
            transform=axis.transAxes,
            fontsize=7.5,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.4},
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
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.4},
        )
        axis.tick_params(labelsize=8)

    with imageio.get_writer(path, mode="I", duration=duration_ms) as writer:
        for frame in frames:
            draw_frame(int(frame))
            figure.canvas.draw()
            rgba = np.asarray(figure.canvas.buffer_rgba())
            writer.append_data(np.asarray(rgba[:, :, :3], dtype=np.uint8))
    plt.close(figure)


def _write_testset_overview(
    path: Path,
    summary: dict[str, Any],
    evaluation: dict[str, np.ndarray],
    selected: list[tuple[str, int, str]],
    decoded_records: dict[int, tuple[dict[str, np.ndarray], dict[str, Any]]],
) -> None:
    """Use all held-out rows for aggregate panels, plus named examples."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    metrics = summary["metrics"]["all"]
    fde = np.asarray(evaluation["sample_fde_m"], dtype=np.float64).mean(axis=1)
    ade = np.asarray(evaluation["sample_ade_m"], dtype=np.float64).mean(axis=1)
    generated_endpoint = np.asarray(evaluation["generated_endpoint"], dtype=np.float64)
    target_endpoint = np.asarray(evaluation["target_endpoint"], dtype=np.float64)
    active = np.asarray(evaluation["active_background"], dtype=bool)
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), squeeze=False)
    seconds = np.asarray((1.0, 2.0, 3.0, 4.0, 5.0, 5.96))
    horizon_names = ("1.00s", "2.00s", "3.00s", "4.00s", "5.00s", "5.96s")
    horizons = metrics["horizons"]
    axes[0, 0].plot(
        seconds,
        [horizons[name]["zero_latent_FDE_m"] for name in horizon_names],
        marker="o",
        color="#333333",
        label="zero latent",
    )
    axes[0, 0].plot(
        seconds,
        [horizons[name]["sample_mean_FDE_m"] for name in horizon_names],
        marker="s",
        color=GENERATED_COLOR,
        label=f"N(0,I), K={summary['samples_per_condition']}",
    )
    axes[0, 0].set(
        title="All held-out episodes: horizon error",
        xlabel="horizon (s)",
        ylabel="FDE (m)",
    )
    axes[0, 0].legend()
    for values, label, color in (
        (ade, "per-episode sample-mean ADE", REAL_COLOR),
        (fde, "per-episode sample-mean FDE", GENERATED_COLOR),
    ):
        x, y = _ecdf(values)
        axes[0, 1].plot(x, y, label=label, color=color)
        for quantile in (0.5, 0.9, 0.95):
            axes[0, 1].axvline(
                np.quantile(values, quantile), color=color, alpha=0.15, linewidth=0.8
            )
    axes[0, 1].set(
        title="All held-out episode error distribution",
        xlabel="error (m)",
        ylabel="empirical CDF",
        xlim=(0.0, np.quantile(np.concatenate((ade, fde)), 0.99) * 1.05),
    )
    axes[0, 1].legend(fontsize=8)
    positions = active
    target_xy = target_endpoint[positions]
    generated_active = np.broadcast_to(active[:, None], generated_endpoint.shape[:-1])
    generated_xy = generated_endpoint[generated_active]
    for component, label, color in (
        (0, "longitudinal x", REAL_COLOR),
        (1, "lateral y", "#54A24B"),
    ):
        center = target_xy[:, component].mean()
        scale = max(target_xy[:, component].std(), 1.0e-6)
        target_standardized = (target_xy[:, component] - center) / scale
        generated_standardized = (generated_xy[:, component] - center) / scale
        low, high = np.quantile(target_standardized, (0.005, 0.995))
        bins = np.linspace(low, high, 70)
        axes[0, 2].hist(
            target_standardized,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            color=color,
            label=f"logged {label}",
        )
        axes[0, 2].hist(
            generated_standardized,
            bins=bins,
            density=True,
            histtype="step",
            linestyle="--",
            linewidth=1.3,
            color=color,
            alpha=0.75,
            label=f"generated {label}",
        )
    axes[0, 2].set(
        title="Final-position marginals, all active slots",
        xlabel="z-score against logged component",
        ylabel="density",
    )
    axes[0, 2].legend(fontsize=7)
    rows = np.asarray(evaluation["row_index"], dtype=np.int64)
    for axis, (name, row, description) in zip(axes[1], selected):
        decoded, record = decoded_records[row]
        item = record["item"]
        ego = item["future_states"].numpy()[:, 0, :2]
        target = item["future_states"].numpy()[:, 1:, :2]
        predicted = decoded["background_positions_xy"][0]
        for slot in np.flatnonzero(decoded["active_background"]):
            axis.plot(*(target[:, slot] - ego).T, color=REAL_COLOR, linewidth=1.3)
            axis.plot(
                *(predicted[:, slot] - ego).T,
                color=GENERATED_COLOR,
                linewidth=1.15,
            )
        axis.set(
            title=f"{name}\n{record['metrics']['sequence_id']}",
            xlabel="ego-relative x (m)",
            ylabel="ego-relative y (m)",
        )
        axis.legend(
            handles=[
                Line2D([0], [0], color=REAL_COLOR, linewidth=1.5, label="highD"),
                Line2D(
                    [0],
                    [0],
                    color=GENERATED_COLOR,
                    linewidth=1.5,
                    label="Diffusion",
                ),
            ],
            fontsize=7,
            loc="upper right",
        )
        local = int(np.flatnonzero(rows == row)[0])
        axis.text(
            0.02,
            0.02,
            f"{description}\naggregate-draw FDE={fde[local]:.2f} m\n"
            f"fixed-seed decode FDE={record['metrics']['FDE_m']:.2f} m",
            transform=axis.transAxes,
            fontsize=7.2,
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    for axis in axes.flat:
        _style_axes(axis)
    figure.suptitle(
        "Conditional diffusion: held-out state-knot-conditioned reconstruction"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_selected_rollout_profiles(
    path: Path,
    selected: list[tuple[str, int, str]],
    decoded_records: dict[int, tuple[dict[str, np.ndarray], dict[str, Any]]],
) -> None:
    """Write TREAD-style highD/diffusion time-series comparisons.

    Each row uses the active background slot with the largest generated final
    displacement error, so the panel is a transparent diagnostic rather than a
    hand-picked best trajectory.
    """
    import matplotlib.pyplot as plt

    columns = (
        (
            "relative longitudinal position",
            r"$x_b-x_{ego}$ (m)",
            lambda ego, state: state[:, 0] - ego[:, 0],
        ),
        (
            "relative lateral position",
            r"$y_b-y_{ego}$ (m)",
            lambda ego, state: state[:, 1] - ego[:, 1],
        ),
        (
            "longitudinal acceleration",
            r"$a_x$ (m/s$^2$)",
            lambda _ego, state: state[:, 4],
        ),
        ("lateral acceleration", r"$a_y$ (m/s$^2$)", lambda _ego, state: state[:, 5]),
    )
    figure, axes = plt.subplots(
        len(selected),
        len(columns),
        figsize=(15.2, max(2.7 * len(selected), 4.0)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, (name, row, _description) in enumerate(selected):
        decoded, record = decoded_records[row]
        item = record["item"]
        ego = item["future_states"].numpy()[:, 0]
        slot = int(record["metrics"]["focus_background_slot"]) - 1
        logged = item["future_states"].numpy()[:, slot + 1]
        generated = decoded["background_states"][0, :, slot]
        time_s = (np.arange(HORIZON_STEPS, dtype=np.float32) + 1.0) * DT_S
        for column_index, (title, ylabel, transform) in enumerate(columns):
            axis = axes[row_index, column_index]
            axis.plot(
                time_s,
                transform(ego, logged),
                color=REAL_COLOR,
                label="highD",
            )
            axis.plot(
                time_s,
                transform(ego, generated),
                color=GENERATED_COLOR,
                label="Diffusion",
            )
            if row_index == 0:
                axis.set_title(title)
            if column_index == 0:
                axis.set_ylabel(f"{name}\nb{slot + 1}: {ylabel}")
            else:
                axis.set_ylabel(ylabel)
            if row_index == len(selected) - 1:
                axis.set_xlabel(r"$t$ (s)")
            _style_axes(axis)
        axes[row_index, 0].legend(frameon=False, loc="upper right")
    figure.suptitle(
        "Selected held-out rollouts: logged highD versus one fixed-seed diffusion decode"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--ddim-steps",
        type=int,
        help="Override the evaluation config.",
    )
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(config["paths"]["output_dir"]).resolve()
    )
    config["paths"]["output_dir"] = str(output)
    ddim_steps = (
        int(args.ddim_steps)
        if args.ddim_steps is not None
        else int(config["evaluation"]["ddim_steps"])
    )
    guidance_scale = (
        float(args.guidance_scale)
        if args.guidance_scale is not None
        else float(config["evaluation"].get("guidance_scale", 1.0))
    )
    evaluation_path = output / "evaluation_per_sequence.npz"
    summary_path = output / "evaluation_summary.json"
    if not evaluation_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            "Run evaluate_background_diffusion.py before rendering test-set visualizations."
        )
    evaluation = dict(np.load(evaluation_path))
    summary = load_json(summary_path)
    device = select_device(config["evaluation"].get("device", "auto"))
    checkpoint = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else output / "checkpoints/best_background_diffusion.pt"
    )
    model, state = load_checkpoint(checkpoint, device=device)
    model.eval()
    bundle = load_data_bundle(config, config_path.parent)
    selected = _select_playbacks(evaluation, bundle)
    playback_dir = ensure_dir(output / "playbacks")
    decoded_records: dict[int, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}
    entries = []
    for index, (name, row, description) in enumerate(selected):
        decoded, record = _decode_one(
            model,
            state,
            bundle,
            row,
            device=device,
            ddim_steps=ddim_steps,
            guidance_scale=guidance_scale,
            seed=args.seed + index,
        )
        decoded_records[row] = (decoded, record)
        gif = playback_dir / f"{name}.gif"
        _render_playback(
            gif,
            decoded,
            record,
            title=name.replace("_", " "),
            frame_stride=args.frame_stride,
            fps=args.fps,
        )
        entries.append(
            {
                "name": name,
                "selection_rule": description,
                "gif": str(gif.relative_to(output)),
                "latent_seed": int(args.seed + index),
                "metrics": record["metrics"],
            }
        )
    overview = output / "testset_constrained_reconstruction.png"
    _write_testset_overview(overview, summary, evaluation, selected, decoded_records)
    profiles = output / "testset_selected_rollout_profiles.png"
    _write_selected_rollout_profiles(profiles, selected, decoded_records)
    manifest = {
        "role": "all-natural-driving constrained-reconstruction visualization",
        "checkpoint": str(checkpoint.relative_to(output)),
        "checkpoint_epoch": int(state["epoch"]),
        "condition": (
            "logged 40-D C0, six-slot mask and background-only state knots "
            "at 2.00, 4.00 and 5.96 seconds"
        ),
        "ego_future_in_model_condition": False,
        "ego_policy": "logged replay for visualization only; replaceable by ADS output",
        "background_policy": "one fixed N(0,I) diffusion decode per selected condition",
        "logged_background": "reference overlay only; never fed back into the generated rollout",
        "ddim_steps": ddim_steps,
        "guidance_scale": guidance_scale,
        "frame_stride": int(args.frame_stride),
        "fps": int(args.fps),
        "aggregate_overview": str(overview.relative_to(output)),
        "selected_rollout_profiles": str(profiles.relative_to(output)),
        "playbacks": entries,
        "interpretation": (
            "background-state-knot-conditioned reconstruction without future "
            "ego information; this is not C0-only motion prediction"
        ),
    }
    manifest_path = playback_dir / "playback_manifest.json"
    save_json(manifest, manifest_path)
    print(manifest_path)


if __name__ == "__main__":
    main()

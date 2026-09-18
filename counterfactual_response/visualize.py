"""Reference-style GIF and static response plots."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v2 as imageio
from tools.plot_style import get_pyplot
from matplotlib.patches import Rectangle
import numpy as np

from .experiment import MODEL_NAMES


plt = get_pyplot()


COLORS = {
    "factual": "#e69500", "counterfactual": "#7c2ae8", "leader": "#d62728",
    "road": "#777b80", "lane": "#e6e6e6", "band": "#f3b5b5",
    "b_idm": "#6b7280", "ma_idm": "#14866d", "dynamic_ar5": "#2563eb", "multi_regime": "#8b5cf6",
}
LABELS = {"b_idm": "B-IDM", "ma_idm": "MA-IDM", "dynamic_ar5": "Dynamic-AR(5)", "multi_regime": "Multi-regime B-IDM"}


def _key(dose: float) -> str:
    return str(dose).replace(".", "p")


def _draw_vehicle(axis: Any, x: float, y: float, color: str, label: str) -> None:
    axis.add_patch(Rectangle((x - 2.4, y - 0.9), 4.8, 1.8, facecolor=color, edgecolor="#222", linewidth=1.0, zorder=4))
    axis.text(x, y + 1.25, label, fontsize=7.5, ha="center", va="bottom",
              bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.0})


def _road(axis: Any, leader_x: float, follower_x: float, trail_x: np.ndarray, *, reactive: bool) -> None:
    center = 0.5 * (leader_x + follower_x)
    color = COLORS["counterfactual"] if reactive else COLORS["factual"]
    axis.set_facecolor(COLORS["road"])
    for lane in (-3.75, 0.0, 3.75):
        axis.axhline(lane, color=COLORS["lane"], linestyle="-" if lane else "--", linewidth=0.8, alpha=0.65)
    axis.plot(trail_x, np.full_like(trail_x, -1.8), color=color, linewidth=1.5)
    _draw_vehicle(axis, follower_x, -1.8, color, "affected follower")
    _draw_vehicle(axis, leader_x, -1.8, COLORS["leader"], "ADS: added braking")
    axis.set(
        xlim=(center - 65.0, center + 65.0), ylim=(-8.2, 8.2), yticks=[],
        xlabel="longitudinal position [m]", aspect="equal",
    )
    axis.tick_params(labelsize=7)


def _representative(
    natural_gap: np.ndarray, logged_gap: np.ndarray, frozen_a: np.ndarray, reactive_a: np.ndarray,
    frozen_gap: np.ndarray, reactive_gap: np.ndarray, hold: int,
) -> int:
    """Select the ensemble medoid without rewarding an unusually tidy baseline."""
    peak = np.max(np.maximum(0.0, -(reactive_a[:, ::hold] - frozen_a[:, ::hold])), axis=1)
    baseline_rmse = np.sqrt(np.mean(np.square(natural_gap - logged_gap[None, :]), axis=1))
    terminal_benefit = reactive_gap[:, -1] - frozen_gap[:, -1]
    features = np.stack((peak, baseline_rmse, terminal_benefit), axis=1)
    median = np.median(features, axis=0)
    scale = np.maximum(np.median(np.abs(features - median), axis=0), 1.0e-8)
    return int(np.argmin(np.sum(np.square((features - median) / scale), axis=1)))


def render_gif(model: str, report: dict[str, Any], ensemble: dict[str, np.ndarray], output: Path,
               *, dose: float = 3.0) -> Path:
    scenario = report["scenario"]
    dt, frames = float(scenario["dt_s"]), int(round(scenario["horizon_s"] / scenario["dt_s"]))
    hold = int(round(scenario["decision_dt_s"] / dt))
    prefix = f"{model}__dose_{_key(dose)}"
    natural_gap = ensemble[f"{model}__natural__gap"]
    frozen_a = ensemble[f"{prefix}__frozen__acceleration"]
    frozen_gap = ensemble[f"{prefix}__frozen__gap"]
    frozen_v = ensemble[f"{prefix}__frozen__speed"]
    frozen_x = ensemble[f"{prefix}__frozen__position"]
    reactive_a = ensemble[f"{prefix}__reactive__acceleration"]
    reactive_gap = ensemble[f"{prefix}__reactive__gap"]
    reactive_v = ensemble[f"{prefix}__reactive__speed"]
    reactive_x = ensemble[f"{prefix}__reactive__position"]
    logged_gap = ensemble["highd_logged_gap"]
    index = _representative(natural_gap, logged_gap, frozen_a, reactive_a, frozen_gap, reactive_gap, hold)
    leader_x = ensemble[f"leader_dose_{_key(dose)}_x"]
    leader_v = ensemble[f"leader_dose_{_key(dose)}_v"]
    times = np.arange(frames) * dt
    path = output / f"{model}_counterfactual_brake.gif"
    figure, grid_axes = plt.subplots(2, 2, figsize=(15, 8), dpi=100)
    figure.subplots_adjust(left=.045, right=.99, bottom=.08, top=.87, hspace=.36, wspace=.08)
    axes = list(grid_axes.ravel())
    frames_to_render = np.arange(0, frames, 2)
    with imageio.get_writer(path, mode="I", duration=80, loop=0) as writer:
        for frame in frames_to_render:
            for axis in axes:
                axis.clear()
            figure.suptitle(
                f"Stochastic-driver matched braking probe | {LABELS[model]} | validation row {report['highd_anchor']['row']} | t={times[frame]:.2f}s\n"
                f"Logged leader control + {dose:g} m/s² braking during "
                f"{scenario['brake_start_s']:.2f}–{scenario['brake_start_s'] + scenario['brake_duration_s']:.2f}s; matched intervention and initial state.",
                fontsize=10, fontweight="bold",
            )
            start = max(0, frame - 45)
            _road(axes[0], leader_x[frame], frozen_x[index, frame], frozen_x[index, start:frame + 1], reactive=False)
            _road(axes[1], leader_x[frame], reactive_x[index, frame], reactive_x[index, start:frame + 1], reactive=True)
            axes[0].set_title("Frozen no-intervention action plan | same ADS brake, no action update", fontsize=9)
            axes[1].set_title(f"Closed-loop {LABELS[model]} response | same ADS brake", fontsize=9)
            for axis, follower_speed in (
                (axes[0], frozen_v[index, frame]),
                (axes[1], reactive_v[index, frame]),
            ):
                closing = float(follower_speed - leader_v[frame])
                axis.text(
                    .01, .02,
                    f"v_leader={leader_v[frame]:.2f} m/s | v_follower={follower_speed:.2f} m/s | "
                    f"closing={closing:+.2f} m/s (+ means gap shrinking)",
                    transform=axis.transAxes, fontsize=7, va="bottom", ha="left",
                    bbox={"facecolor": "white", "alpha": .78, "edgecolor": "none", "pad": 1.2},
                )
            for axis, title, factual, counter in (
                (axes[2], "Affected follower longitudinal action", frozen_a[index], reactive_a[index]),
                (axes[3], "Leader → follower bumper gap", frozen_gap[index], reactive_gap[index]),
            ):
                axis.axvspan(scenario["brake_start_s"], scenario["brake_start_s"] + scenario["brake_duration_s"],
                             color=COLORS["band"], alpha=0.42, label="ADS brake window")
                axis.plot(times[:frame + 1], factual[:frame + 1], color=COLORS["factual"], linewidth=1.8, label="frozen actions")
                axis.plot(times[:frame + 1], counter[:frame + 1], color=COLORS["counterfactual"], linewidth=1.8, label="closed-loop response")
                axis.set_xlim(0.0, scenario["horizon_s"])
                axis.grid(alpha=0.22)
                axis.set_xlabel("time [s]")
                axis.set_title(title)
            lo_a = min(-2.0, float(np.min((frozen_a[index], reactive_a[index]))) - 0.3)
            hi_a = max(1.0, float(np.max((frozen_a[index], reactive_a[index]))) + 0.3)
            axes[2].set(ylim=(lo_a, hi_a), ylabel="acceleration [m/s²]")
            initial_gap = float(report["highd_anchor"]["initial_gap_m"])
            lo_gap = min(float(np.min((frozen_gap[index], reactive_gap[index]))) - 1.0, initial_gap - 2.0)
            hi_gap = max(float(np.max((frozen_gap[index], reactive_gap[index]))) + 1.0, initial_gap + 2.0)
            axes[3].set(ylim=(lo_gap, hi_gap), ylabel="gap [m]")
            for axis in axes[2:4]:
                axis.legend(frameon=True, fontsize=7, loc="best")
            figure.canvas.draw()
            rgba = np.asarray(figure.canvas.buffer_rgba())
            writer.append_data(np.asarray(rgba[:, :, :3], dtype=np.uint8).copy())
    plt.close(figure)
    return path


def render_summary(report: dict[str, Any], output: Path) -> Path:
    doses = np.asarray(report["scenario"]["brake_doses_mps2"], float)
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)
    specs = (
        ("response_probability", None, "response probability", "probability"),
        ("response_latency_s", "median", "median response latency", "seconds"),
        ("peak_additional_braking_mps2", "median", "median peak additional braking", "m/s²"),
        ("terminal_gap_benefit_m", "median", "median terminal gap benefit vs frozen actions", "metres"),
    )
    for model in MODEL_NAMES:
        rows = report["models"][model]["dose_response"]
        for axis, (metric, field, title, ylabel) in zip(axes.ravel(), specs):
            values = []
            for dose in doses:
                value = rows[str(float(dose))][metric]
                values.append(value if field is None else value[field])
            axis.plot(doses, values, marker="o", linewidth=2, color=COLORS[model], label=LABELS[model])
            axis.set(title=title, xlabel="leader braking dose [m/s²]", ylabel=ylabel)
            axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "highD-anchored matched braking response surfaces\n"
        "same intervened leader in frozen-action and closed-loop branches",
        fontweight="bold",
    )
    path = output / "counterfactual_braking_response_summary.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def render_outputs(report: dict[str, Any], ensemble: dict[str, np.ndarray], output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    paths = [render_summary(report, output)]
    paths.extend(render_gif(model, report, ensemble, output) for model in MODEL_NAMES)
    return paths

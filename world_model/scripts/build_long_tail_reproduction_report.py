#!/usr/bin/env python3
"""Build publication-quality, model-organized reports for long-tail replay."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODELS = (
    ("ramp", "ramp_world_model", "RAMP-WM", "#377eb8"),
    ("semi_markov", "semi_markov_world_model", "Semi-Markov WM", "#ff7f00"),
    ("cat_topk", "cat_topk_world_model", "CAT-TopK", "#4daf4a"),
)
REAL_COLOR = "#1b1b1b"
TAIL_COLORS = ("#2166ac", "#f4a582", "#b2182b")
EVENT_LABELS = {
    "high_risk_following": "High-risk\nfollowing",
    "hard_braking": "Hard\nbraking",
    "high_speed_approach": "High-speed\napproach",
    "close_interaction": "Close\ninteraction",
    "strong_relative_speed_change": "Strong Δv",
}
MOTION_VARIABLES = (
    "speed_mps",
    "delta_speed_mps",
    "acceleration_mps2",
    "jerk_mps3",
    "relative_speed_mps",
    "closing_speed_mps",
)
SAFETY_VARIABLES = (
    "gap_m",
    "ttc_s",
    "drac_mps2",
    "min_acceleration_mps2",
    "min_gap_m",
    "min_ttc_s",
    "max_drac_mps2",
)
VARIABLE_LABELS = {
    "speed_mps": "speed",
    "delta_speed_mps": "Δ speed",
    "acceleration_mps2": "long. accel.",
    "jerk_mps3": "jerk",
    "relative_speed_mps": "relative speed",
    "closing_speed_mps": "closing speed",
    "gap_m": "gap",
    "ttc_s": "TTC",
    "drac_mps2": "DRAC",
    "min_acceleration_mps2": "min accel.",
    "min_gap_m": "min gap",
    "min_ttc_s": "min TTC",
    "max_drac_mps2": "max DRAC",
}


def _pyplot():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": True,
            "legend.framealpha": 0.94,
            "legend.fontsize": 8.5,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )
    return plt


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    _pyplot().close(fig)


def _panel(ax: Any, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
    )


def _bar_labels(ax: Any, values: list[float], *, fmt: str = ".3f") -> None:
    for index, value in enumerate(values):
        if np.isfinite(value):
            ax.text(index, value, format(value, fmt), ha="center", va="bottom", fontsize=7)


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _model_meta(name: str) -> tuple[str, str, str]:
    for key, directory, display, color in MODELS:
        if key == name:
            return directory, display, color
    raise KeyError(name)


def _event_values(report: dict[str, Any], metric: str) -> list[float]:
    result = []
    for event in EVENT_LABELS:
        event_report = report["events"][event]
        result.append(_float(event_report.get("trajectory_reproduction", {}).get(metric)))
    return result


def _event_risk_values(report: dict[str, Any], metric: str) -> list[float]:
    return [
        _float(report["events"][event].get("risk_tail", {}).get(metric))
        for event in EVENT_LABELS
    ]


def _plot_comparison(summary: dict[str, Any], output: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.1), constrained_layout=True)
    labels = [display for _key, _directory, display, _color in MODELS]
    colors = [color for _key, _directory, _display, color in MODELS]
    reports = [summary["models"][key] for key, _directory, _display, _color in MODELS]
    x = np.arange(len(labels))
    width = 0.36

    ax = axes[0, 0]
    ade = [_float(item["trajectory_reproduction"]["ADE_m"]) for item in reports]
    fde = [_float(item["trajectory_reproduction"]["FDE_m"]) for item in reports]
    ax.bar(x - width / 2, ade, width, color=colors, label="ADE")
    ax.bar(x + width / 2, fde, width, color=colors, alpha=0.42, label="FDE")
    ax.set_xticks(x, labels, rotation=8)
    ax.set_ylabel("metres (↓)")
    ax.set_title("Deterministic long-tail reconstruction")
    ax.legend(ncol=2, loc="upper left")
    _panel(ax, "a")

    ax = axes[0, 1]
    min_ade = [_float(item["trajectory_reproduction"]["minADE_at_K_m"]) for item in reports]
    min_fde = [_float(item["trajectory_reproduction"]["minFDE_at_K_m"]) for item in reports]
    ax.bar(x - width / 2, min_ade, width, color=colors, label=r"minADE@32")
    ax.bar(x + width / 2, min_fde, width, color=colors, alpha=0.42, label=r"minFDE@32")
    ax.set_xticks(x, labels, rotation=8)
    ax.set_ylabel("metres (↓)")
    ax.set_title("Coverage of logged future")
    ax.legend(ncol=2, loc="upper left")
    _panel(ax, "b")

    ax = axes[1, 0]
    risk_w1 = [_float(item["risk_tail"]["wasserstein_1"]) for item in reports]
    risk_ks = [_float(item["risk_tail"]["ks"]) for item in reports]
    ax.bar(x - width / 2, risk_w1, width, color=colors, label=r"Risk $W_1$")
    right = ax.twinx()
    right.bar(x + width / 2, risk_ks, width, color=colors, alpha=0.42, label=r"Risk $D_{KS}$")
    ax.set_xticks(x, labels, rotation=8)
    ax.set_ylabel(r"$W_1$ (↓)")
    right.set_ylabel(r"$D_{KS}$ (↓)")
    ax.set_title("Risk-tail distribution fidelity")
    handles, texts = ax.get_legend_handles_labels(); handles2, texts2 = right.get_legend_handles_labels()
    ax.legend(handles + handles2, texts + texts2, loc="upper left")
    _panel(ax, "c")

    ax = axes[1, 1]
    diversity = [_float(item["diversity"]["average_pairwise_FDE_m"]) for item in reports]
    coverage = [_float(item["diversity"]["coverage"]["minFDE_le_1m"]) for item in reports]
    ax.bar(x - width / 2, diversity, width, color=colors, label="Pairwise FDE")
    right = ax.twinx()
    right.bar(x + width / 2, coverage, width, color=colors, alpha=0.42, label="Coverage ≤1 m")
    ax.set_xticks(x, labels, rotation=8)
    ax.set_ylabel("metres (↑ diversity)")
    right.set_ylabel("episode rate (↑)")
    right.set_ylim(0.0, 1.08)
    ax.set_title("Stochastic diversity and coverage")
    handles, texts = ax.get_legend_handles_labels(); handles2, texts2 = right.get_legend_handles_labels()
    ax.legend(handles + handles2, texts + texts2, loc="upper left")
    _panel(ax, "d")
    _save(fig, output / "01_model_comparison_overview.png")


def _plot_reproduction(report: dict[str, Any], display: str, color: str, output: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.1), constrained_layout=True)
    labels = list(EVENT_LABELS.values())
    x = np.arange(len(labels))
    width = 0.36

    ax = axes[0, 0]
    fde = _event_values(report, "FDE_m")
    min_fde = _event_values(report, "minFDE_at_K_m")
    ax.bar(x - width / 2, fde, width, color=color, label="FDE")
    ax.bar(x + width / 2, min_fde, width, color=color, alpha=0.42, label=r"minFDE@32")
    ax.set_xticks(x, labels)
    ax.set_ylabel("metres (↓)")
    ax.set_title(f"{display}: event-wise endpoint error")
    ax.legend(ncol=2)
    _panel(ax, "a")

    ax = axes[0, 1]
    ade = _event_values(report, "ADE_m")
    min_ade = _event_values(report, "minADE_at_K_m")
    ax.bar(x - width / 2, ade, width, color=color, label="ADE")
    ax.bar(x + width / 2, min_ade, width, color=color, alpha=0.42, label=r"minADE@32")
    ax.set_xticks(x, labels)
    ax.set_ylabel("metres (↓)")
    ax.set_title(f"{display}: event-wise average error")
    ax.legend(ncol=2)
    _panel(ax, "b")

    ax = axes[1, 0]
    counts = [int(report["events"][event]["num_sequences"]) for event in EVENT_LABELS]
    bars = ax.bar(x, counts, color=TAIL_COLORS[0], alpha=0.8)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count, str(count), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("held-out scenarios")
    ax.set_title("Physical long-tail event support")
    _panel(ax, "c")

    ax = axes[1, 1]
    risk = _event_risk_values(report, "wasserstein_1")
    ks = _event_risk_values(report, "ks")
    ax.bar(x - width / 2, risk, width, color=color, label=r"risk $W_1$")
    right = ax.twinx()
    right.bar(x + width / 2, ks, width, color=color, alpha=0.42, label=r"risk $D_{KS}$")
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"$W_1$ (↓)")
    right.set_ylabel(r"$D_{KS}$ (↓)")
    ax.set_title("Risk-tail fidelity by event")
    handles, texts = ax.get_legend_handles_labels(); handles2, texts2 = right.get_legend_handles_labels()
    ax.legend(handles + handles2, texts + texts2, loc="upper left")
    _panel(ax, "d")
    _save(fig, output / "01_event_reproduction.png")


def _plot_risk(report: dict[str, Any], display: str, color: str, output: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.1), constrained_layout=True)

    for ax, keys, title, label in (
        (axes[0, 0], MOTION_VARIABLES, "Motion-variable distribution fidelity", r"$W_1$ (↓)"),
        (axes[0, 1], SAFETY_VARIABLES, "Safety-variable distribution fidelity", r"$W_1$ (↓)"),
    ):
        values = [_float(report["risk_variable_distribution"][key]["wasserstein_1"]) for key in keys]
        y = np.arange(len(keys))
        ax.barh(y, values, color=color, alpha=0.85)
        ax.set_yticks(y, [VARIABLE_LABELS[key] for key in keys])
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.set_title(title)
        for index, value in enumerate(values):
            ax.text(value, index, f" {value:.3g}", va="center", fontsize=7)
    _panel(axes[0, 0], "a")
    _panel(axes[0, 1], "b")

    ax = axes[1, 0]
    tail = report["risk_tail"]["exceedance_at_real_quantiles"]
    quantile_labels = ["q90", "q95", "q99"]
    x = np.arange(3)
    width = 0.34
    real = [_float(tail[key]["real"]) for key in quantile_labels]
    generated = [_float(tail[key]["generated"]) for key in quantile_labels]
    ax.bar(x - width / 2, real, width, color=REAL_COLOR, label="highD real")
    ax.bar(x + width / 2, generated, width, color=color, label=display)
    ax.set_xticks(x, [r"$P(R>q_{90})$", r"$P(R>q_{95})$", r"$P(R>q_{99})$"])
    ax.set_ylabel("exceedance probability")
    ax.set_ylim(0.0, max([*real, *generated]) * 1.35)
    ax.set_title("Risk-tail calibration at real thresholds")
    ax.legend()
    _panel(ax, "c")

    ax = axes[1, 1]
    risk = report["risk_tail"]
    qs = risk["quantiles"]
    observed = [_float(qs[key]["generated"]) for key in quantile_labels]
    target = [_float(qs[key]["real"]) for key in quantile_labels]
    ax.plot(target, target, "--", color="0.45", label="perfect calibration")
    ax.scatter(target, observed, s=62, color=color, zorder=3, label=display)
    for label, x_value, y_value in zip(quantile_labels, target, observed):
        ax.annotate(label, (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)
    extent = [min([*target, *observed]), max([*target, *observed])]
    margin = max((extent[1] - extent[0]) * 0.16, 0.08)
    ax.set_xlim(extent[0] - margin, extent[1] + margin)
    ax.set_ylim(extent[0] - margin, extent[1] + margin)
    ax.set_xlabel("highD risk quantile")
    ax.set_ylabel("generated risk quantile")
    ax.set_title(r"Tail quantiles: $q_{90}$, $q_{95}$, $q_{99}$")
    ax.legend(loc="upper left")
    _panel(ax, "d")
    _save(fig, output / "02_risk_distribution_fidelity.png")


def _heatmap(ax: Any, values: list[list[float]], title: str, *, vmax: float = 1.0) -> Any:
    image = ax.imshow(np.asarray(values), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    labels = ("speed", "accel.", "gap")
    ax.set_xticks(range(3), labels, rotation=26, ha="right")
    ax.set_yticks(range(3), labels)
    ax.set_title(title)
    for row in range(3):
        for col in range(3):
            ax.text(col, row, f"{values[row][col]:.2f}", ha="center", va="center", fontsize=7)
    return image


def _plot_interaction_temporal(report: dict[str, Any], display: str, color: str, output: Path) -> None:
    plt = _pyplot()
    fig = plt.figure(figsize=(13.2, 8.3))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.985,
        bottom=0.095,
        top=0.90,
        wspace=0.14,
        hspace=0.25,
    )
    correlation_grid = grid[0, 0].subgridspec(1, 3, wspace=0.46)
    interaction = report["multi_vehicle_interaction"]
    temporal = report["temporal_dynamics"]
    ax_real = fig.add_subplot(correlation_grid[0, 0])
    ax_generated = fig.add_subplot(correlation_grid[0, 1])
    ax_diff = fig.add_subplot(correlation_grid[0, 2])
    _heatmap(ax_real, interaction["correlation_real"], "highD")
    _heatmap(ax_generated, interaction["correlation_generated"], "generated")
    difference = (np.asarray(interaction["correlation_generated"]) - np.asarray(interaction["correlation_real"])).tolist()
    _heatmap(ax_diff, difference, "difference", vmax=max(float(np.abs(np.asarray(difference)).max()), 0.01))
    fig.text(0.275, 0.94, "Speed–acceleration–gap correlation", ha="center", fontsize=11, fontweight="semibold")
    _panel(ax_real, "a")

    ax = fig.add_subplot(grid[0, 1])
    lag = np.arange(1, len(temporal["acceleration_acf_real"]) + 1) * 0.04
    ax.plot(lag, temporal["acceleration_acf_real"], color=REAL_COLOR, linewidth=2.0, label="highD real")
    ax.plot(lag, temporal["acceleration_acf_generated"], color=color, linewidth=2.0, label=display)
    ax.set_xlabel(r"lag $\Delta t$ (s)")
    ax.set_ylabel("acceleration ACF")
    ax.set_title(f"Closed-loop acceleration persistence  |  MAE={_float(temporal['acceleration_acf_mean_absolute_error']):.3f}")
    ax.legend()
    _panel(ax, "b")

    ax = fig.add_subplot(grid[1, 0])
    real_brake = interaction["brake_response_real"]
    generated_brake = interaction["brake_response_generated"]
    edges = np.asarray(real_brake["front_acceleration_bin_edges_mps2"])
    centers = (edges[:-1] + edges[1:]) / 2.0
    ax.plot(centers, real_brake["mean_rear_acceleration_mps2"], "o-", color=REAL_COLOR, label=f"highD (r={_float(real_brake['pearson_correlation']):.2f})")
    ax.plot(centers, generated_brake["mean_rear_acceleration_mps2"], "o-", color=color, label=f"{display} (r={_float(generated_brake['pearson_correlation']):.2f})")
    ax.axhline(0.0, color="0.55", linewidth=0.7)
    ax.set_xlabel(r"front acceleration $a_f$ bin centre (m/s$^2$)")
    ax.set_ylabel(r"mean rear acceleration $E[a_r|a_f]$ (m/s$^2$)")
    ax.set_title("Car-following braking response")
    ax.legend()
    _panel(ax, "c")

    ax = fig.add_subplot(grid[1, 1])
    duration_keys = ("braking_duration_s", "acceleration_duration_s", "lateral_motion_duration_s")
    duration_labels = ("braking", "accelerating", "lateral")
    w1 = [_float(temporal[key]["wasserstein_1"]) for key in duration_keys]
    ks = [_float(temporal[key]["ks"]) for key in duration_keys]
    x = np.arange(3)
    width = 0.34
    ax.bar(x - width / 2, w1, width, color=color, label=r"$W_1$")
    right = ax.twinx()
    right.bar(x + width / 2, ks, width, color=color, alpha=0.42, label=r"$D_{KS}$")
    ax.set_xticks(x, duration_labels)
    ax.set_ylabel(r"duration $W_1$ (s, ↓)")
    right.set_ylabel(r"duration $D_{KS}$ (↓)")
    ax.set_title("Physical behaviour duration fidelity")
    handles, texts = ax.get_legend_handles_labels(); handles2, texts2 = right.get_legend_handles_labels()
    ax.legend(handles + handles2, texts + texts2, loc="upper left")
    _panel(ax, "d")
    _save(fig, output / "03_interaction_temporal_dynamics.png")


def _plot_diversity_physics(report: dict[str, Any], display: str, color: str, output: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), constrained_layout=True)
    diversity = report["diversity"]
    ax = axes[0]
    coverage = diversity["coverage"]
    names = (r"≤1 m", r"≤2 m", r"≤5 m")
    values = [_float(coverage["minFDE_le_1m"]), _float(coverage["minFDE_le_2m"]), _float(coverage["minFDE_le_5m"])]
    bars = ax.bar(names, values, color=color, alpha=0.85)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel("episode coverage")
    ax.set_title(rf"{display}: logged-future coverage by minFDE@32")
    ax.text(0.02, 0.94, f"AP FDE distance = {_float(diversity['average_pairwise_FDE_m']):.3f} m", transform=ax.transAxes, va="top", fontsize=9, bbox={"boxstyle": "round", "fc": "white", "ec": "0.75"})
    _panel(ax, "a")

    ax = axes[1]
    physics = report["physical_validity"]
    labels = ("invalid", "speed", "accel.", "jerk", "collision")
    values = [
        _float(physics["invalid_trajectory_rate"]),
        _float(physics["speed_out_of_range_rate"]),
        _float(physics["acceleration_out_of_range_rate"]),
        _float(physics["jerk_out_of_range_rate"]),
        _float(physics["collision_overlap_rate"]),
    ]
    display_values = np.maximum(values, 1.0e-7)
    ax.bar(labels, display_values, color=color, alpha=0.85)
    ax.set_yscale("log")
    ax.set_ylim(1.0e-7, max(max(display_values) * 5.0, 1.0e-5))
    ax.set_ylabel("rate (log scale, ↓)")
    ax.set_title("Physical validity of 32 generated futures")
    for index, value in enumerate(values):
        text = "0" if value == 0 else f"{value:.2g}"
        ax.text(index, display_values[index], text, ha="center", va="bottom", fontsize=8)
    _panel(ax, "b")
    _save(fig, output / "04_diversity_and_physics.png")


def _plot_flow_ramp_composition(report: dict[str, Any], output: Path) -> None:
    """Render the composed Flow×RAMP test without implying paired ADE/FDE."""
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.1), constrained_layout=True)
    flow = report["flow_input_fidelity_on_held_out_structure"]
    closed = report["closed_loop_distribution"]

    ax = axes[0, 0]
    keys = ("c0", "b0")
    labels = ("C0", "B0")
    w1 = [_float(flow[key]["mean_wasserstein_1"]) for key in keys]
    ks = [_float(flow[key]["mean_ks"]) for key in keys]
    x = np.arange(2)
    ax.bar(x - 0.17, w1, 0.34, color="#756bb1", label=r"mean $W_1$")
    right = ax.twinx()
    right.bar(x + 0.17, ks, 0.34, color="#bcbddc", label=r"mean $D_{KS}$")
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"input $W_1$ (↓)")
    right.set_ylabel(r"input $D_{KS}$ (↓)")
    ax.set_title("Flow outer-sample fidelity")
    handles, texts = ax.get_legend_handles_labels(); handles2, texts2 = right.get_legend_handles_labels()
    ax.legend(handles + handles2, texts + texts2, loc="upper left")
    _panel(ax, "a")

    ax = axes[0, 1]
    b0 = report["b0_execution_fidelity"]
    labels = (r"$\Delta v_x$", r"$\Delta v_y$", "mean a", "min a", "final a", "mean ay")
    values = [_float(value) for value in b0["per_feature_mean_absolute_error"]]
    ax.bar(labels, values, color="#377eb8")
    ax.set_ylabel("absolute error")
    ax.set_title(f"B0 execution fidelity  |  mean MAE={_float(b0['mean_absolute_error']):.3f}")
    _panel(ax, "b")

    ax = axes[1, 0]
    risk = closed["risk_tail"]
    tail = risk["exceedance_at_real_quantiles"]
    labels = ("q90", "q95", "q99")
    x = np.arange(3)
    ax.bar(x - 0.17, [_float(tail[key]["real"]) for key in labels], 0.34, color=REAL_COLOR, label="highD replay")
    ax.bar(x + 0.17, [_float(tail[key]["generated"]) for key in labels], 0.34, color="#377eb8", label="Flow×RAMP")
    ax.set_xticks(x, [r"$P(R>q_{90})$", r"$P(R>q_{95})$", r"$P(R>q_{99})$"])
    ax.set_ylabel("exceedance probability")
    ax.set_title(rf"Closed-loop risk tail  |  $W_1$={_float(risk['wasserstein_1']):.3f}, $D_{{KS}}$={_float(risk['ks']):.3f}")
    ax.legend()
    _panel(ax, "c")

    ax = axes[1, 1]
    validity = closed["physical_validity"]
    diversity = closed["within_flow_start_diversity"]
    labels = ("invalid", "speed", "accel.", "jerk", "overlap", "inner FDE", "risk std")
    values = [
        _float(validity["invalid_trajectory_rate"]),
        _float(validity["speed_out_of_range_rate"]),
        _float(validity["acceleration_out_of_range_rate"]),
        _float(validity["jerk_out_of_range_rate"]),
        _float(validity["collision_overlap_rate"]),
        _float(diversity["mean_pairwise_endpoint_distance_m"]),
        _float(diversity["mean_risk_score_standard_deviation"]),
    ]
    ax.bar(labels, np.maximum(values, 1.0e-7), color=["#377eb8"] * 5 + ["#ff7f00", "#ff7f00"])
    ax.set_yscale("log")
    ax.set_ylabel("rate or scale (log)")
    ax.set_title("Physical validity and inner stochastic spread")
    _panel(ax, "d")
    _save(fig, output / "05_flow_ramp_composition.png")


def _flow_ramp_composition_analysis(report: dict[str, Any]) -> str:
    protocol = report["protocol"]
    closed = report["closed_loop_distribution"]
    validity = closed["physical_validity"]
    return f"""

## Flow×RAMP composed test distribution

This separate report samples **{protocol['flow_outer_samples_per_replay']}** Flow C0/B0 starts and **{protocol['ramp_inner_samples_per_flow_start']}** RAMP candidate futures for each of **{protocol['held_out_replay_conditions']}** held-out replay conditions. It therefore contains **{protocol['generated_closed_loop_trajectories']:,}** five-second futures. Two held-out replays are unsupported because their discrete event structures have no Flow-training support. It is a distribution-level replay-controlled test, so per-donor ADE/FDE and minFDE are intentionally not reported.

- Flow input mean C0/B0 W1: **{_float(report['flow_input_fidelity_on_held_out_structure']['c0']['mean_wasserstein_1']):.3f} / {_float(report['flow_input_fidelity_on_held_out_structure']['b0']['mean_wasserstein_1']):.3f}**.
- B0 execution MAE: **{_float(report['b0_execution_fidelity']['mean_absolute_error']):.3f}**.
- Closed-loop risk W1/KS: **{_float(closed['risk_tail']['wasserstein_1']):.3f} / {_float(closed['risk_tail']['ks']):.3f}**.
- Invalid trajectory / overlap rate: **{_float(validity['invalid_trajectory_rate']):.2%} / {_float(validity['collision_overlap_rate']):.2%}**.
- Same-C0/B0 inner branch endpoint spread: **{_float(closed['within_flow_start_diversity']['mean_pairwise_endpoint_distance_m']):.3f} m**.

The frozen Flow does not model map geometry or the external ego policy. Both are supplied by a held-out replay matched by the cache-derived START event structure (same-front when present, otherwise the first active fixed slot); the result is consequently evidence for the composed test environment under that explicit policy, not a claim of unconditional five-second natural-traffic generation. **The current composition does not pass as a usable long-tail generator:** its 32.28% invalid-trajectory rate and 0.011 m same-start branch spread show that physical safety and stochastic diversity must be repaired before it is used for ADS testing.
"""


def _model_analysis(name: str, display: str, report: dict[str, Any]) -> str:
    trajectory = report["trajectory_reproduction"]
    tail = report["risk_tail"]
    interaction = report["multi_vehicle_interaction"]
    temporal = report["temporal_dynamics"]
    diversity = report["diversity"]
    hard_braking = report["events"]["hard_braking"]
    hard_tail = hard_braking["risk_tail"]
    return f"""# {display}: long-tail reproduction analysis

## Reproduction and coverage

- Deterministic ADE/FDE: **{_float(trajectory['ADE_m']):.3f} / {_float(trajectory['FDE_m']):.3f} m**.
- minADE@32/minFDE@32: **{_float(trajectory['minADE_at_K_m']):.3f} / {_float(trajectory['minFDE_at_K_m']):.3f} m**.
- Pairwise branch FDE distance: **{_float(diversity['average_pairwise_FDE_m']):.3f} m**; coverage at 1/2/5 m: **{_float(diversity['coverage']['minFDE_le_1m']):.1%} / {_float(diversity['coverage']['minFDE_le_2m']):.1%} / {_float(diversity['coverage']['minFDE_le_5m']):.1%}**.

## Risk-tail fidelity

The trajectory risk score uses TTC, DRAC, gap and longitudinal acceleration. Its all-tail empirical discrepancy is **W1={_float(tail['wasserstein_1']):.3f}**, **KS={_float(tail['ks']):.3f}**. At the real q90/q95/q99 thresholds, generated exceedance probabilities are {" / ".join(f"{_float(tail['exceedance_at_real_quantiles'][key]['generated']):.1%}" for key in ('q90', 'q95', 'q99'))}.

The hard-braking subgroup has {int(hard_braking['num_sequences'])} scenarios. Its risk W1 is {_float(hard_tail['wasserstein_1']):.3f}; the q99 generated/real exceedance is {_float(hard_tail['exceedance_at_real_quantiles']['q99']['generated']):.1%} / {_float(hard_tail['exceedance_at_real_quantiles']['q99']['real']):.1%}. This small subgroup must be interpreted as a stress test rather than high-power population evidence.

## Interaction and dynamics

- Speed/acceleration/gap correlation-matrix MAE: **{_float(interaction['correlation_mean_absolute_error']):.4f}**.
- Rear-vs-front braking correlation: generated **{_float(interaction['brake_response_generated']['pearson_correlation']):.3f}**, highD **{_float(interaction['brake_response_real']['pearson_correlation']):.3f}**.
- Acceleration ACF MAE: **{_float(temporal['acceleration_acf_mean_absolute_error']):.3f}**.
- Braking/acceleration/lateral duration W1: **{_float(temporal['braking_duration_s']['wasserstein_1']):.3f} / {_float(temporal['acceleration_duration_s']['wasserstein_1']):.3f} / {_float(temporal['lateral_motion_duration_s']['wasserstein_1']):.3f} s**.

## Validity

Generated trajectories have invalid-rate **{_float(report['physical_validity']['invalid_trajectory_rate']):.2%}**, collision-overlap rate **{_float(report['physical_validity']['collision_overlap_rate']):.2%}**, and jerk-out-of-range rate **{_float(report['physical_validity']['jerk_out_of_range_rate']):.2%}**.

See the four numbered PNG panels in this directory for the visual evidence. Values are evaluated on the fixed full held-out long-tail protocol, not per-event retraining.
"""


def _root_analysis(summary: dict[str, Any]) -> str:
    protocol = summary["protocol"]
    counts = summary["tail_event_selection"]
    reports = summary["models"]
    rows = []
    for key, _directory, display, _color in MODELS:
        report = reports[key]
        trajectory = report["trajectory_reproduction"]
        diversity = report["diversity"]
        interaction = report["multi_vehicle_interaction"]
        rows.append(
            f"| {display} | {_float(trajectory['FDE_m']):.3f} | {_float(trajectory['minFDE_at_K_m']):.3f} | "
            f"{_float(report['risk_tail']['wasserstein_1']):.3f} / {_float(report['risk_tail']['ks']):.3f} | "
            f"{_float(diversity['average_pairwise_FDE_m']):.3f} | {_float(diversity['coverage']['minFDE_le_1m']):.1%} | "
            f"{_float(interaction['correlation_mean_absolute_error']):.4f} | {_float(report['physical_validity']['invalid_trajectory_rate']):.2%} |"
        )
    return f"""# highD world-model long-tail reproduction report

## Protocol

This report evaluates **{protocol['num_sequences']}** held-out highD EVT-tail scenes for **{protocol['horizon_seconds']:.0f} s**. Every model receives the same initial traffic state, B0, road graph, ego history and observed ego replay; **{protocol['num_stochastic_futures']}** futures are generated per scene. Empirical CDF metrics use a deterministic cap of {protocol['distribution_empirical_max_points']:,} points while trajectory, tail probability and diversity metrics use all scenes and branches.

Physical event subsets are overlapping: high-risk following **{counts['high_risk_following']}**, hard braking **{counts['hard_braking']}**, high-speed approach **{counts['high_speed_approach']}**, close interaction **{counts['close_interaction']}**, and strong within-slot relative-speed change **{counts['strong_relative_speed_change']}**.

## How to read this directory

- `comparison/`: cross-model overview, the full risk CCDF, and the canonical full summary.
- `highd_real_tail/`: fixed protocol and event-selection definition.
- `ramp_world_model/`, `semi_markov_world_model/`, `cat_topk_world_model/`: model-only JSON, four publication-ready panels, and a compact interpretation.

## Quantitative comparison

| Model | FDE (m) ↓ | minFDE@32 (m) ↓ | Risk W1 / KS ↓ | Branch FDE (m) ↑ | Coverage ≤1 m ↑ | Interaction MAE ↓ | Invalid ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

The numerical source of this table is `comparison/long_tail_reproduction_summary.json`; values in the per-model `metrics.json` files are identical subsets, included for independent use.

## Evidence-supported conclusions

1. **Logged-future reconstruction.** RAMP-WM is best on deterministic ADE/FDE (0.190/0.752 m), but its 32 branches are nearly identical (pairwise FDE 0.004 m). It reconstructs the conditional mean trajectory; it does not yet create a useful multi-modal test distribution.
2. **Test-distribution coverage.** CAT-TopK has the lowest minFDE@32 (0.573 m) and highest 1 m coverage (85.7%), with the only materially separated branches (0.273 m). Its archived START interface, however, receives a future-action summary; it remains a reproducibility reference rather than a same-information superiority claim.
3. **Risk and interaction fidelity.** Semi-Markov is closest on aggregate tail risk (W1/KS 0.023/0.015) and speed/acceleration/gap interaction (MAE 0.0006). Its generated braking-response correlation is 0.327 versus 0.302 in highD, but its deterministic FDE and 1 m coverage are weaker than the other two models.
4. **Rare physical dynamics remain the limiting evidence.** In the 17 hard-braking scenes, the q99 risk exceedance is under-reproduced by every model; several braking/acceleration-duration W1 values are near or above one second. These observations identify precisely which tail mechanisms require more data or targeted training.

## Scope and decision

The fixed-condition results establish conditional reconstruction and diagnostic capability under a logged C0/B0 and ego replay. They do **not** by themselves establish that the model can construct a long-tail test distribution: that claim requires the separate Flow×world-model composition test below. The hard-braking sample is small, event groups overlap, and CAT-TopK is information-asymmetric. Retain these boundaries in any performance claim.
"""


def _selection_readme(summary: dict[str, Any]) -> str:
    counts = summary["tail_event_selection"]
    return f"""# highD real long-tail reference

The reference set is the held-out `is_evt_tail` split. It contains {summary['protocol']['num_sequences']} scenes and uses the logged future solely as the evaluation target and ego replay.

| Event | Physical criterion | Scenes |
| --- | --- | ---: |
| High-risk following | minimum TTC < 3 s | {counts['high_risk_following']} |
| Hard braking | minimum longitudinal acceleration < -1.5 m/s² | {counts['hard_braking']} |
| High-speed approach | maximum closing speed > 5 m/s | {counts['high_speed_approach']} |
| Close interaction | minimum body-clearance-adjusted gap < 8 m | {counts['close_interaction']} |
| Strong relative-speed change | one fixed slot changes relative speed by > 3 m/s over 5 s | {counts['strong_relative_speed_change']} |

Events overlap by design. The fixed-slot requirement in the final row avoids falsely treating two different background vehicles as one changing interaction.
"""


def _resolve_source(root: Path, supplied: str | None) -> Path:
    if supplied:
        return Path(supplied).resolve()
    for candidate in (
        root / "long_tail_reproduction_summary.json",
        root / "comparison" / "long_tail_reproduction_summary.json",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("missing long_tail_reproduction_summary.json")


def _move_shared_artifacts(root: Path, source: Path, comparison: Path) -> Path:
    comparison.mkdir(parents=True, exist_ok=True)
    canonical_summary = comparison / "long_tail_reproduction_summary.json"
    if source.resolve() != canonical_summary.resolve():
        shutil.move(str(source), canonical_summary)
    source_ccdf = root / "risk_ccdf.png"
    canonical_ccdf = comparison / "00_risk_ccdf.png"
    if source_ccdf.exists() and source_ccdf.resolve() != canonical_ccdf.resolve():
        shutil.move(str(source_ccdf), canonical_ccdf)
    return canonical_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results/highd_world_model/long_tail_reproduction"),
    )
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    root = Path(args.output_dir).resolve()
    source = _resolve_source(root, args.summary)
    comparison = root / "comparison"
    source = _move_shared_artifacts(root, source, comparison)
    summary = _json(source)

    _write_json(comparison / "study_manifest.json", {"protocol": summary["protocol"], "checkpoints": summary["checkpoints"]})
    _plot_comparison(summary, comparison)
    composition_path = root / "ramp_world_model/flow_ramp_composition.json"
    composition = _json(composition_path) if composition_path.exists() else None
    root_text = _root_analysis(summary)
    if composition is not None:
        root_text += _flow_ramp_composition_analysis(composition)
    (root / "README.md").write_text(root_text, encoding="utf-8")

    real = root / "highd_real_tail"
    real.mkdir(exist_ok=True)
    _write_json(real / "event_selection.json", {"protocol": summary["protocol"], "tail_event_selection": summary["tail_event_selection"]})
    (real / "README.md").write_text(_selection_readme(summary), encoding="utf-8")

    for name, directory, display, color in MODELS:
        report = summary["models"][name]
        output = root / directory
        output.mkdir(exist_ok=True)
        _write_json(output / "metrics.json", report)
        _plot_reproduction(report, display, color, output)
        _plot_risk(report, display, color, output)
        _plot_interaction_temporal(report, display, color, output)
        _plot_diversity_physics(report, display, color, output)
        model_text = _model_analysis(name, display, report)
        if name == "ramp" and composition is not None:
            _plot_flow_ramp_composition(composition, output)
            model_text += _flow_ramp_composition_analysis(composition)
        (output / "README.md").write_text(model_text, encoding="utf-8")

    legacy_conclusion = root / "conclusions.md"
    if legacy_conclusion.exists():
        legacy_conclusion.unlink()
    print(root)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build paper figures from completed FIRM-WM evaluation artifacts only."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.core.utils import ensure_dir, load_json, save_json


PAPER_COLORS = {
    "firm": "#1f6aa5",
    "sample": "#58b9b1",
    "truth": "#555555",
    "brake": "#cf4b5f",
    "ramp": "#505050",
    "semi": "#d97a1d",
    "cat": "#7a70b3",
    "warning": "#bc3f4f",
}


def _style_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="0.90", lw=0.65, zorder=0)
    axis.set_axisbelow(True)


def _panel(axis, label: str) -> None:
    axis.text(
        -0.11,
        1.16,
        label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _risk(states: np.ndarray, ego: np.ndarray, valid: np.ndarray) -> np.ndarray:
    gap = np.abs(states[..., 0] - ego[:, :, None, 0]).clip(0.1)
    closing = np.maximum(-(states[..., 2] - ego[:, :, None, 2]), 0.0)
    score = np.where(valid, closing**2 / (2.0 * gap), 0.0)
    return score.max(axis=(1, 2))


def _ccdf(axis, values: np.ndarray, label: str, color: str) -> None:
    data = np.sort(np.asarray(values, np.float64))
    if len(data):
        axis.step(data, 1.0 - np.arange(len(data)) / len(data), where="post", label=label, color=color)


def _nested(payload: dict[str, Any], *keys: str) -> float | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return float(value) if isinstance(value, (int, float)) else None


def _baseline_rows(
    firm: dict[str, Any], comparison: dict[str, Any] | None
) -> list[tuple[str, float | None, float | None, str]]:
    rows = [
        (
            "FIRM-WM",
            _nested(firm, "five_second_roll_mode", "ADE_m"),
            _nested(firm, "five_second_roll_mode", "FDE_m"),
            "#20639b",
        )
    ]
    for key, name, color in (
        ("ramp_world_model", "RAMP-WM", "#4d4d4d"),
        ("semi_markov_world_model", "Semi-Markov", "#d95f02"),
    ):
        metrics = (
            comparison.get("models", {}).get(key, {}).get("matched_background_replay", {}).get("five_second")
            if comparison
            else None
        )
        rows.append(
            (
                name,
                metrics.get("ADE_m") if metrics else None,
                metrics.get("FDE_m") if metrics else None,
                color,
            )
        )
    cat_path = ROOT / "results/highd_world_model/cat_topk_world_model/evaluation_summary.json"
    cat = load_json(cat_path) if cat_path.exists() else {}
    # This archived result is only a disclosed reference: CAT's START sees a
    # future-action summary and is not a same-information comparison.
    rows.append(
        (
            "CAT-TopK†",
            _nested(cat, "model_state_reconstruction", "test", "5_chunks", "ADE_m"),
            _nested(cat, "model_state_reconstruction", "test", "5_chunks", "FDE_m"),
            "#7570b3",
        )
    )
    return rows


def _heldout_comparison_status(
    firm: dict[str, Any], comparison: dict[str, Any] | None
) -> dict[str, str]:
    """Describe the saved comparison without inheriting legacy-result claims."""
    if comparison is None:
        return {
            "ramp": "Matched frozen RAMP-WM replay is unavailable.",
            "semi": "Matched frozen Semi-Markov replay is unavailable.",
            "manifest": "matched frozen baselines unavailable",
        }
    firm_fde = _nested(firm, "five_second_roll_mode", "FDE_m")
    models = comparison.get("models", {})
    ramp = _nested(models, "ramp_world_model", "matched_background_replay", "five_second", "FDE_m")
    semi = _nested(models, "semi_markov_world_model", "matched_background_replay", "five_second", "FDE_m")

    def verdict(name: str, reference: float | None) -> str:
        if firm_fde is None or reference is None:
            return f"Matched frozen {name} FDE is unavailable."
        relation = "matches or improves" if firm_fde <= reference else "does not yet match"
        return f"FIRM-WM {relation} matched frozen {name} FDE ({firm_fde:.3f} vs {reference:.3f} m)."

    return {
        "ramp": verdict("RAMP-WM", ramp),
        "semi": verdict("Semi-Markov", semi),
        "manifest": (
            "FIRM-WM matches or improves matched frozen RAMP-WM FDE."
            if firm_fde is not None and ramp is not None and firm_fde <= ramp
            else "FIRM-WM does not yet match matched frozen RAMP-WM FDE."
        ),
    }


def _save(figure, path: Path, *, rect: tuple[float, float, float, float] | None = None) -> None:
    figure.tight_layout(rect=rect)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _reconstruction_panel(data, path: Path) -> None:
    samples = data["example_sampled_background_states"]
    predicted = data["example_predicted_states"]
    target = data["example_target_states"]
    valid = data["example_valid"]
    figure, axes = plt.subplots(2, 3, figsize=(12.4, 6.7))
    # Evaluation saves the first examples encountered in each event stratum.
    # Preserve that fixed order here: use one nominal and two EVT-tail cases
    # when both are available, never trajectory error or visual quality.
    tail = np.asarray(data["example_is_evt_tail"], dtype=bool)
    nominal = np.flatnonzero(~tail)
    evt_tail = np.flatnonzero(tail)
    selected = list(nominal[:1]) + list(evt_tail[:2])
    if len(selected) < min(3, len(samples)):
        selected.extend(index for index in range(len(samples)) if index not in selected)
    selected = selected[: min(3, len(samples))]
    for column, index in enumerate(selected):
        bird, time = axes[0, column], axes[1, column]
        _panel(bird, chr(ord("a") + column))
        _panel(time, chr(ord("d") + column))
        ego = target[index, :, 0]
        bird.plot(ego[:, 0], ego[:, 1], color="black", lw=1.5, label="ego replay", zorder=4)
        active_agents = np.flatnonzero(valid[index, :, 1:].any(axis=0))
        for agent in range(6):
            keep = valid[index, :, agent + 1]
            if not keep.any():
                continue
            bird.plot(target[index, keep, agent + 1, 0], target[index, keep, agent + 1, 1], color="0.55", lw=1.05, label="highD" if agent == active_agents[0] else None)
            bird.plot(predicted[index, keep, agent + 1, 0], predicted[index, keep, agent + 1, 1], color=PAPER_COLORS["firm"], lw=0.9, label="FIRM mean" if agent == active_agents[0] else None)
            for draw_index, draw in enumerate(samples[index, :, :, agent]):
                bird.plot(
                    draw[keep, 0],
                    draw[keep, 1],
                    color=PAPER_COLORS["sample"],
                    alpha=0.15,
                    lw=0.55,
                    label="FIRM sampled worlds" if agent == active_agents[0] and draw_index == 0 else None,
                )
        stratum = "EVT-tail" if tail[index] else "nominal"
        bird.set_title("Closed-loop trajectories", fontsize=11, pad=8)
        bird.text(
            0.98,
            0.96,
            f"{stratum}; fixed scan index {index}",
            transform=bird.transAxes,
            ha="right",
            va="top",
            fontsize=7.3,
            color="0.35",
        )
        bird.set_xlabel("x (m)")
        bird.set_ylabel("y (m)")
        primary = int(active_agents[0])
        primary_keep = valid[index, :, primary + 1]
        speed = np.linalg.norm(predicted[index, :, primary + 1, 2:4], axis=-1)
        true_speed = np.linalg.norm(target[index, :, primary + 1, 2:4], axis=-1)
        time_seconds = np.arange(len(speed))[primary_keep] / 25.0
        time.plot(time_seconds, true_speed[primary_keep], color=PAPER_COLORS["truth"], label="highD", zorder=3)
        time.plot(time_seconds, speed[primary_keep], color=PAPER_COLORS["firm"], label="FIRM mean", zorder=4)
        for draw in samples[index, :, :, primary]:
            time.plot(
                time_seconds,
                np.linalg.norm(draw[primary_keep, 2:4], axis=-1),
                color=PAPER_COLORS["sample"],
                alpha=0.18,
                lw=0.7,
            )
        time.set_title("Selected background speed", fontsize=11, pad=8)
        time.set_xlabel("time (s)")
        time.set_ylabel("selected background speed (m/s)")
        _style_axis(bird)
        _style_axis(time)
    axes[0, 0].legend(loc="best", fontsize=7)
    axes[1, 0].legend(loc="best", fontsize=7)
    _save(figure, path)


def _calibration_panel(calibration: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9.4, 6.2))
    rank = calibration["rank_histogram"]
    counts = np.asarray(rank["counts"], np.float64)
    frequency = counts / counts.sum()
    axes[0, 0].bar(np.arange(len(frequency)), frequency, color=PAPER_COLORS["firm"])
    axes[0, 0].axhline(1.0 / len(frequency), color="0.35", ls="--", lw=1.0, label="uniform")
    axes[0, 0].set_title("Position rank histogram")
    axes[0, 0].set_xlabel("rank among random worlds")
    axes[0, 0].set_ylabel("relative frequency")
    axes[0, 0].legend(fontsize=8)
    coverage = calibration["conditional_coverage"]
    levels = np.asarray([float(value) for value in coverage])
    values = np.asarray([coverage[str(level)] for level in levels])
    axes[0, 1].plot(levels, levels, "--", color="0.4", label="ideal")
    axes[0, 1].plot(levels, values, "o-", color=PAPER_COLORS["firm"], label="FIRM")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].set_title("Conditional coverage")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].axis("off")
    axes[1, 0].text(0.04, 0.84, "Executed-prefix probability scores", fontsize=11, weight="bold")
    for row, (label, value) in enumerate((
        ("prefix NLL", calibration["prefix_nll"]),
        ("Energy score", calibration["energy_score"]),
        ("CRPS", calibration["crps"]),
    )):
        axes[1, 0].text(0.06, 0.62 - 0.18 * row, label, color="0.35", fontsize=10)
        axes[1, 0].text(0.94, 0.62 - 0.18 * row, f"{value:.3f}", color=PAPER_COLORS["firm"], fontsize=11, weight="bold", ha="right")
    axes[1, 0].text(0.06, 0.03, "Different proper scores have different units; they are reported, not summed.", fontsize=8, color="0.35")
    axes[1, 1].axis("off")
    axes[1, 1].text(0.02, 0.72, "FIRM probability contract", fontsize=12, weight="bold")
    axes[1, 1].text(0.02, 0.44, "The scored variable is the same\n0.2 s joint control prefix that\nis written to the simulator.", fontsize=10)
    for index, axis in enumerate(axes.flat):
        _panel(axis, chr(ord("a") + index))
        if axis.axison:
            _style_axis(axis)
    _save(figure, path)


def _interaction_panel(counterfactual, path: Path) -> None:
    maintain = counterfactual["maintain_background"]
    brake = counterfactual["brake_background"]
    ego_maintain = counterfactual["maintain_ego"]
    ego_brake = counterfactual["brake_ego"]
    valid = counterfactual["valid"]
    time = np.arange(maintain.shape[1]) / 25.0
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 6.2))

    def mean_band(value):
        values = np.where(valid, value, np.nan).transpose(1, 0, 2).reshape(len(time), -1)
        return np.nanmean(values, axis=1), np.nanquantile(values, 0.25, axis=1), np.nanquantile(values, 0.75, axis=1)

    for state, ego, label, color in (
        (maintain, ego_maintain, "maintain", PAPER_COLORS["firm"]),
        (brake, ego_brake, "brake", PAPER_COLORS["brake"]),
    ):
        speed = np.linalg.norm(state[..., 2:4], axis=-1)
        gap = np.abs(state[..., 0] - ego[:, :, None, 0])
        acceleration = state[..., 4]
        closing = np.maximum(-(state[..., 2] - ego[:, :, None, 2]), 0.0)
        ttc = np.minimum(gap / np.maximum(closing, 1.0e-3), 10.0)
        mask = valid
        for axis, value, title in (
            (axes[0, 0], speed, "background speed"),
            (axes[0, 1], gap, "ego-background gap"),
            (axes[1, 0], acceleration, "background longitudinal acceleration"),
            (axes[1, 1], ttc, "TTC"),
        ):
            centre, lower, upper = mean_band(value)
            axis.plot(time, centre, color=color, label=label)
            axis.fill_between(time, lower, upper, color=color, alpha=0.16, lw=0)
            axis.set_title(title)
            axis.set_xlabel("time (s)")
            _style_axis(axis)
    axes[0, 0].legend(fontsize=8)
    for index, axis in enumerate(axes.flat):
        _panel(axis, chr(ord("a") + index))
    _save(figure, path)


def _tail_panel(
    data, path: Path, flow_scores=None, flow_report: dict[str, Any] | None = None
) -> None:
    predicted, target, valid, tail = (
        data["predicted_states"],
        data["target_states"],
        data["valid"],
        data["is_evt_tail"],
    )
    if flow_scores is None:
        choose = tail if tail.any() else np.ones(len(target), dtype=bool)
        generated_risk = _risk(predicted[choose, :, 1:], predicted[choose, :, 0], valid[choose, :, 1:])
        observed_risk = _risk(target[choose, :, 1:], target[choose, :, 0], valid[choose, :, 1:])
        title = "Risk CCDF on held-out tail stratum"
    else:
        observed_risk = flow_scores["highd_risk_scores"]
        generated_risk = flow_scores["firm_risk_scores"]
        title = "Risk CCDF: frozen Flow × FIRM-WM"
    figure, axes = plt.subplots(1, 3, figsize=(13.6, 3.9))
    _ccdf(axes[0], observed_risk, "highD", PAPER_COLORS["truth"])
    _ccdf(axes[0], generated_risk, "FIRM-WM", PAPER_COLORS["firm"])
    axes[0].set_yscale("log")
    axes[0].set_ylim(1.0e-3, 1.05)
    axes[0].set_title(title)
    axes[0].set_xlabel("episode risk score")
    axes[0].set_ylabel("P(R ≥ r)")
    axes[0].legend(fontsize=8)
    quantiles = [0.9, 0.95, 0.99]
    axes[1].bar(np.arange(3) - 0.17, [np.quantile(observed_risk, q) for q in quantiles], width=0.34, color=PAPER_COLORS["truth"], label="highD")
    axes[1].bar(np.arange(3) + 0.17, [np.quantile(generated_risk, q) for q in quantiles], width=0.34, color=PAPER_COLORS["firm"], label="FIRM")
    axes[1].set_xticks(np.arange(3), ["q90", "q95", "q99"])
    axes[1].set_title("Risk quantiles")
    axes[1].legend(fontsize=8)
    exceedance = (
        flow_report.get("closed_loop_distribution", {})
        .get("risk_tail_all_inner_samples", {})
        .get("exceedance_at_real_quantiles", {})
        if flow_report
        else {}
    )
    if exceedance:
        levels = ("q90", "q95", "q99")
        position = np.arange(len(levels))
        highd = np.asarray([exceedance[level]["highd"] for level in levels])
        generated = np.asarray([exceedance[level]["flow_firm"] for level in levels])
        lower = np.asarray([exceedance[level]["highd_bootstrap_95"][0] for level in levels])
        upper = np.asarray([exceedance[level]["highd_bootstrap_95"][1] for level in levels])
        axes[2].bar(position - 0.17, highd, width=0.34, color=PAPER_COLORS["truth"], label="highD")
        axes[2].errorbar(
            position - 0.17,
            highd,
            yerr=np.vstack((highd - lower, upper - highd)),
            fmt="none",
            color="0.20",
            capsize=2.5,
            lw=0.85,
        )
        axes[2].bar(position + 0.17, generated, width=0.34, color=PAPER_COLORS["firm"], label="Flow × FIRM")
        axes[2].set_xticks(position, levels)
        axes[2].set_ylabel("P(R > highD threshold)")
        axes[2].set_title("Tail-mass calibration")
        axes[2].legend(fontsize=7.5)
    else:
        axes[2].axis("off")
        axes[2].text(0.02, 0.64, "Tail-mass calibration", fontsize=11, weight="bold")
        axes[2].text(
            0.02,
            0.38,
            "Unavailable in this legacy Flow × FIRM report.\n"
            "New composition reports highD q90/q95/q99\n"
            "exceedance with a fixed-threshold bootstrap interval.",
            fontsize=8.8,
            color="0.35",
        )
    for index, axis in enumerate(axes):
        _panel(axis, chr(ord("a") + index))
        if axis.axison:
            _style_axis(axis)
    _save(figure, path)


def _baseline_panel(
    firm: dict[str, Any],
    comparison: dict[str, Any] | None,
    ablation: dict[str, Any] | None,
    path: Path,
) -> None:
    rows = _baseline_rows(firm, comparison)
    if ablation:
        for name, payload in ablation.get("variants", {}).items():
            rows.append((name, payload.get("ADE_5s_m"), payload.get("FDE_5s_m"), "#66a61e"))
    names = [row[0] for row in rows]
    ade = [np.nan if row[1] is None else row[1] for row in rows]
    fde = [np.nan if row[2] is None else row[2] for row in rows]
    colors = [row[3] for row in rows]
    position = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(8.8, 4.8))
    axis.bar(position - 0.18, ade, width=0.36, color=colors, label="ADE (m)")
    axis.bar(position + 0.18, fde, width=0.36, color=colors, alpha=0.45, label="FDE (m)")
    axis.set_xticks(position, names, rotation=15, ha="right")
    axis.set_title("5 s held-out background reconstruction")
    axis.legend(fontsize=8)
    finite = np.asarray(ade + fde, dtype=np.float64)
    axis.set_ylim(0.0, max(0.10, float(np.nanmax(finite)) * 1.22))
    for offset, value in ((-0.18, ade), (0.18, fde)):
        for index, metric in enumerate(value):
            if np.isfinite(metric):
                axis.text(index + offset, metric + 0.015, f"{metric:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
    _panel(axis, "a")
    _style_axis(axis)
    figure.text(0.12, 0.095, "† CAT-TopK uses an information-asymmetric future summary; it is a disclosed reference, not a same-information win/loss claim.", fontsize=7.3)
    if not ablation:
        figure.text(0.12, 0.050, "FIRM architectural ablations are not yet available and are explicitly recorded as skipped in the manifest.", fontsize=7.3, color=PAPER_COLORS["warning"])
    _save(figure, path, rect=(0.0, 0.17, 1.0, 1.0))


def _physical_panel(
    summary: dict[str, Any],
    composition: dict[str, Any] | None,
    path: Path,
    deterministic_composition: dict[str, Any] | None = None,
) -> None:
    physical = summary["physical_diagnostics"]
    keys = ("invalid_rate", "overlap_rate", "speed_out_of_range_rate", "acceleration_out_of_range_rate", "jerk_out_of_range_rate")
    labels = ("invalid", "overlap", "speed", "acceleration", "jerk")
    values = [physical.get(key, np.nan) for key in keys]
    displayed = np.maximum(np.nan_to_num(values, nan=1.0e-8), 1.0e-8)
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 3.9), gridspec_kw={"width_ratios": (1.0, 1.0, 1.15)})
    axes[0].bar(labels, displayed, color="#d1495b")
    axes[0].set_yscale("log")
    axes[0].set_ylim(1.0e-8, max(1.0e-3, max(displayed) * 10.0))
    axes[0].set_title("Held-out replay physical audit")
    axes[0].set_ylabel("rate")
    axes[1].axis("off")
    axes[1].text(0.02, 0.80, "Fixed-seed replay", fontsize=12, weight="bold")
    axes[1].text(0.02, 0.64, f"max frame error: {summary.get('fixed_seed_replay_max_abs_error', float('nan')):.3e}", fontsize=10)
    if composition:
        physical_flow = composition["closed_loop_distribution"]["physical_validity"]
        timing = composition["closed_loop_distribution"].get("overlap_timing", {})
        if not timing and deterministic_composition:
            timing = deterministic_composition.get("closed_loop_distribution", {}).get(
                "overlap_timing", {}
            )
        axes[1].text(0.02, 0.42, "Flow × FIRM safety gate", fontsize=11, weight="bold", color=PAPER_COLORS["warning"])
        axes[1].text(0.02, 0.31, f"invalid trajectories: {100.0 * physical_flow['invalid_trajectory_rate']:.2f}%  (target < 1%)", fontsize=9.1, color=PAPER_COLORS["warning"])
        axes[1].text(0.02, 0.21, f"overlap points: {100.0 * physical_flow['collision_overlap_rate']:.2f}%", fontsize=9.1, color=PAPER_COLORS["warning"])
        ahead = timing.get("first_overlap_background_ahead_of_ego")
        behind = timing.get("first_overlap_background_behind_ego")
        if ahead is not None and behind is not None:
            axes[1].text(0.02, 0.11, f"first overlap: background ahead / behind ego = {ahead:,} / {behind:,}", fontsize=8.3, color=PAPER_COLORS["warning"])
        axes[1].text(0.02, 0.02, "Result: not eligible for ADS testing; retained as a failure audit.", fontsize=7.9, color=PAPER_COLORS["warning"])
    else:
        axes[1].text(0.02, 0.25, "Flow × FIRM composition metrics are unavailable.", fontsize=9)
    jerk = (
        composition.get("closed_loop_distribution", {}).get("executed_jerk_distribution")
        if composition
        else None
    )
    if jerk:
        longitudinal = jerk["longitudinal_mps3"]
        levels = ("q90", "q99")
        position = np.arange(len(levels))
        axes[2].bar(
            position - 0.17,
            [longitudinal["highd_reference"][level] for level in levels],
            width=0.34,
            color=PAPER_COLORS["truth"],
            label="highD reference",
        )
        axes[2].bar(
            position + 0.17,
            [longitudinal["flow_firm"][level] for level in levels],
            width=0.34,
            color=PAPER_COLORS["warning"],
            label="Flow × FIRM",
        )
        axes[2].set_xticks(position, levels)
        axes[2].set_ylabel("absolute longitudinal jerk (m/s³)")
        axes[2].set_title("Executed random jerk tail")
        axes[2].legend(fontsize=7.5)
        _style_axis(axes[2])
    else:
        axes[2].axis("off")
        axes[2].text(0.02, 0.64, "Executed random jerk tail", fontsize=11, weight="bold")
        axes[2].text(
            0.02,
            0.38,
            "Unavailable in this legacy Flow × FIRM report.\n"
            "New formal composition reports q90/q95/q99\n"
            "for the jerk actually written to the simulator.",
            fontsize=8.8,
            color="0.35",
        )
    _panel(axes[0], "a")
    _panel(axes[1], "b")
    _panel(axes[2], "c")
    _style_axis(axes[0])
    _save(figure, path)


def _readme(
    path: Path,
    generated: list[str],
    skipped: list[str],
    summary: dict[str, Any],
    comparison: dict[str, Any] | None,
    flow_report: dict[str, Any] | None,
) -> None:
    heldout = summary["five_second_roll_mode"]
    result_status = [
        f"- FIRM-WM 5 s background-only held-out ADE/FDE: `{heldout['ADE_m']:.3f}` / `{heldout['FDE_m']:.3f}` m.",
    ]
    if comparison:
        status = _heldout_comparison_status(summary, comparison)
        result_status.extend((
            f"- {status['ramp']}",
            f"- {status['semi']}",
        ))
        promotion = comparison.get("promotion_gate")
        if isinstance(promotion, dict):
            result_status.append(
                "- Goal-document promotion gate: `"
                + str(promotion.get("decision", "not_evaluated"))
                + "` (every required held-out and Flow gate must pass)."
            )
    if flow_report:
        flow_physical = flow_report["closed_loop_distribution"]["physical_validity"]
        risk = flow_report["closed_loop_distribution"]["risk_tail_all_inner_samples"]
        result_status.extend((
            f"- Flow × FIRM-WM uses `{flow_report['protocol']['generated_closed_loop_trajectories']}` generated 5 s futures; risk q90 absolute error is `{risk['quantiles']['q90']['absolute_error']:.3f}`.",
            f"- **Not eligible for ADS testing:** Flow × FIRM-WM invalid-trajectory rate is `{100.0 * flow_physical['invalid_trajectory_rate']:.2f}%` (required < 1%) and overlap-point rate is `{100.0 * flow_physical['collision_overlap_rate']:.2f}%`.",
        ))
    path.write_text(
        "# FIRM-WM paper experiments\n\n"
        "## Inputs\n\n"
        "- `evaluation/evaluation_summary.json`\n"
        "- `evaluation/calibration_metrics.json`\n"
        "- `evaluation/heldout_rollouts.npz`\n"
        "- `evaluation/counterfactual_rollouts.npz`\n"
        "- optional: `evaluation/flow_firm_composition.json`, `evaluation/flow_firm_tail_scores.npz`, and `evaluation/baseline_comparison.json`\n\n"
        "## Generated Artifacts\n\n"
        + "\n".join(f"- `{name}`" for name in generated)
        + "\n\n## Reused Existing Artifacts\n\n"
        "- Frozen RAMP-WM, Semi-Markov, and CAT-TopK evaluation summaries when available.\n\n"
        "## Skipped Artifacts\n\n"
        + "\n".join(f"- {item}" for item in skipped)
        + "\n\n## Interpretation Notes\n\n"
        "Figures are a deterministic post-process of saved evaluation arrays. The reconstruction panel uses the first saved nominal and EVT-tail examples in scan order; it never selects samples by error or visual quality. No figure script trains, resamples, fits EVT, or changes a model result. CAT-TopK remains information-asymmetric.\n\n"
        "## Result Status\n\n"
        + "\n".join(result_status)
        + "\n",
        encoding="utf-8",
    )


def build(output_dir: Path) -> dict[str, Any]:
    evaluation = output_dir / "evaluation"
    summary_path = evaluation / "evaluation_summary.json"
    calibration_path = evaluation / "calibration_metrics.json"
    rollout_path = evaluation / "heldout_rollouts.npz"
    counterfactual_path = evaluation / "counterfactual_rollouts.npz"
    required = (summary_path, calibration_path, rollout_path, counterfactual_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("FIRM paper figures need completed evaluation artifacts: " + ", ".join(missing))
    # Keep candidate diagnostics separate from the formal FIRM-WM figures.
    # A caller may inspect a candidate by passing its result directory, but it
    # cannot overwrite the official paper artifacts before promotion.
    paper = ensure_dir(ROOT / "results/highd_world_model/paper_experiments" / output_dir.name)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 9})
    summary, calibration = load_json(summary_path), load_json(calibration_path)
    data = np.load(rollout_path, allow_pickle=False)
    counterfactual = np.load(counterfactual_path, allow_pickle=False)
    ablation_path = evaluation / "ablation_summary.json"
    ablation = load_json(ablation_path) if ablation_path.exists() else None
    comparison_path = evaluation / "baseline_comparison.json"
    comparison = load_json(comparison_path) if comparison_path.exists() else None
    promotion = comparison.get("promotion_gate") if isinstance(comparison, dict) else None
    flow_scores_path = evaluation / "flow_firm_tail_scores.npz"
    flow_scores = np.load(flow_scores_path, allow_pickle=False) if flow_scores_path.exists() else None
    flow_report_path = evaluation / "flow_firm_composition.json"
    flow_report = load_json(flow_report_path) if flow_report_path.exists() else None
    deterministic_flow_report_path = evaluation / "flow_firm_deterministic_composition.json"
    deterministic_flow_report = (
        load_json(deterministic_flow_report_path)
        if deterministic_flow_report_path.exists()
        else None
    )
    outputs = {
        "firm_world_model_closed_loop_reconstruction_panel.png": lambda path: _reconstruction_panel(data, path),
        "firm_world_model_probabilistic_calibration_panel.png": lambda path: _calibration_panel(calibration, path),
        "firm_world_model_interaction_response_panel.png": lambda path: _interaction_panel(counterfactual, path),
        "firm_world_model_tail_distribution_panel.png": lambda path: _tail_panel(data, path, flow_scores, flow_report),
        "firm_world_model_ablation_baseline_panel.png": lambda path: _baseline_panel(summary, comparison, ablation, path),
        "firm_world_model_physical_replay_audit_panel.png": lambda path: _physical_panel(
            summary, flow_report, path, deterministic_flow_report
        ),
    }
    for name, draw in outputs.items():
        draw(paper / name)
    skipped = [] if ablation else ["FIRM architecture ablations: evaluation/ablation_summary.json is absent."]
    if comparison is None:
        skipped.append("Matched RAMP-WM / Semi-Markov background-only replay: evaluation/baseline_comparison.json is absent.")
    if flow_scores is None:
        skipped.append("Flow × FIRM tail-risk scores: evaluation/flow_firm_tail_scores.npz is absent.")
    if flow_report is None:
        skipped.append("Flow × FIRM physical-validity audit: evaluation/flow_firm_composition.json is absent.")
    _readme(
        paper / "FIRM_WM_EXPERIMENT_README.md",
        list(outputs),
        skipped,
        summary,
        comparison,
        flow_report,
    )
    inputs = {str(path): _sha256(path) for path in required}
    if flow_scores_path.exists():
        inputs[str(flow_scores_path)] = _sha256(flow_scores_path)
    if flow_report_path.exists():
        inputs[str(flow_report_path)] = _sha256(flow_report_path)
    if deterministic_flow_report_path.exists():
        inputs[str(deterministic_flow_report_path)] = _sha256(deterministic_flow_report_path)
    if comparison_path.exists():
        inputs[str(comparison_path)] = _sha256(comparison_path)
    artifact_inputs = {
        "firm_world_model_closed_loop_reconstruction_panel.png": [str(rollout_path)],
        "firm_world_model_probabilistic_calibration_panel.png": [str(calibration_path)],
        "firm_world_model_interaction_response_panel.png": [str(counterfactual_path)],
        "firm_world_model_tail_distribution_panel.png": [str(rollout_path)] + ([str(flow_scores_path)] if flow_scores_path.exists() else []) + ([str(flow_report_path)] if flow_report_path.exists() else []),
        "firm_world_model_ablation_baseline_panel.png": [str(summary_path)] + ([str(comparison_path)] if comparison_path.exists() else []) + ([str(ablation_path)] if ablation_path.exists() else []),
        "firm_world_model_physical_replay_audit_panel.png": [str(summary_path)] + ([str(flow_report_path)] if flow_report_path.exists() else []) + ([str(deterministic_flow_report_path)] if deterministic_flow_report_path.exists() else []),
    }
    manifest = {
        "model": "FIRM-WM",
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "inputs": inputs,
        "protocol": "5 s held-out highD evaluation; fixed scan-order event strata for examples; 300 dpi deterministic post-process",
        "generated_artifacts": list(outputs),
        "artifacts": {
            name: {
                "status": "generated",
                "inputs": artifact_inputs[name],
                "selection_rule": (
                    "first saved nominal case and first two saved EVT-tail cases in deterministic evaluation scan order"
                    if name == "firm_world_model_closed_loop_reconstruction_panel.png"
                    else "all completed values from the listed saved evaluation artifacts"
                ),
            }
            for name in outputs
        },
        "skipped_artifacts": skipped,
        "counterfactual_seed_contract": "maintain and brake share the same model random seed",
        "promotion_gate": promotion,
        "result_status": {
            "heldout_reconstruction": _heldout_comparison_status(summary, comparison)["manifest"],
            "promotion_decision": (
                promotion.get("decision", "not_evaluated")
                if isinstance(promotion, dict)
                else "not_evaluated"
            ),
            "flow_firm_ads_gate": (
                "failed: invalid trajectory rate exceeds the <1% requirement"
                if flow_report
                and flow_report["closed_loop_distribution"]["physical_validity"]["invalid_trajectory_rate"] >= 0.01
                else (
                    "passed: invalid trajectory rate is below the <1% requirement"
                    if flow_report
                    else "not evaluated"
                )
            ),
        },
    }
    save_json(manifest, paper / "firm_world_model_experiment_manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT / "results/highd_world_model/firm_world_model"))
    args = parser.parse_args()
    build(Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()

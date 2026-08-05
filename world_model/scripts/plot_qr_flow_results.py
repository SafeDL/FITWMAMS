#!/usr/bin/env python3
"""Render Flow×QR distribution and runtime figures from current result files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.core.utils import ensure_dir, load_json, save_json


RISK_STYLE = {
    "ttc_s": ("TTC", "s"),
    "drac_mps2": ("DRAC", "m/s²"),
    "gap_m": ("Following gap", "m"),
    "relative_speed_mps": ("Closing speed", "m/s"),
}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.22,
    })
    return plt


def _save(figure: Any, path: Path) -> Path:
    figure.tight_layout(rect=(0, 0.035, 1, 0.95))
    figure.savefig(path, dpi=300, bbox_inches="tight")
    _plt().close(figure)
    return path


def _bar_labels(axis: Any, values: list[float], *, fmt: str = ".3f") -> None:
    maximum = max(values, default=0.0)
    for index, value in enumerate(values):
        axis.text(index, value + max(maximum * 0.025, 1e-4), format(value, fmt), ha="center", va="bottom", fontsize=7.5)


def _risk_quantile_panel(axis: Any, name: str, report: dict[str, Any]) -> None:
    label, unit = RISK_STYLE[name]
    values = report["closed_loop_distribution"]["risk_variable_distribution"][name]
    levels = ("q90", "q95", "q99")
    x, width = np.arange(len(levels)), 0.36
    real = [float(values["quantiles"][level]["real"]) for level in levels]
    generated = [float(values["quantiles"][level]["generated"]) for level in levels]
    axis.bar(x - width / 2, real, width, color="#333333", label="EVT-tail highD")
    axis.bar(x + width / 2, generated, width, color="#4e79a7", label="Flow × QR-WM")
    axis.set_xticks(x, ("P90", "P95", "P99"))
    axis.set_title(f"{label} upper quantiles")
    axis.set_ylabel(unit)
    if name == "ttc_s":
        axis.text(0.5, 0.08, "Capped at 10 s", transform=axis.transAxes, ha="center", fontsize=8)


def plot_tail_interaction_distribution(output_dir: Path) -> Path:
    """Plot real-vs-synthetic tail interaction distributions."""
    output_dir = Path(output_dir).resolve()
    report = load_json(output_dir / "flow_composition_evaluation.json")
    risk = report["closed_loop_distribution"]["risk_variable_distribution"]
    plt = _plt()
    figure, axes = plt.subplots(2, 3, figsize=(15.2, 8.2))
    for axis, name in zip(axes.flat[:4], RISK_STYLE):
        _risk_quantile_panel(axis, name, report)
    axes[0, 0].legend(fontsize=8, loc="best")
    labels = [RISK_STYLE[name][0] for name in RISK_STYLE]
    ks = [float(risk[name]["ks"]) for name in RISK_STYLE]
    w1 = [float(risk[name]["wasserstein_1"]) for name in RISK_STYLE]
    axes[1, 1].bar(np.arange(len(labels)), ks, color="#f28e2b")
    axes[1, 1].set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right")
    axes[1, 1].set_title("Kolmogorov–Smirnov distance")
    axes[1, 1].set_ylabel("lower is better")
    _bar_labels(axes[1, 1], ks)
    axes[1, 2].bar(np.arange(len(labels)), w1, color="#e15759")
    axes[1, 2].set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right")
    axes[1, 2].set_title("Wasserstein-1 distance")
    axes[1, 2].set_ylabel("native feature unit; lower is better")
    _bar_labels(axes[1, 2], w1)
    metrics = report["closed_loop_distribution"]
    protocol = report["protocol"]
    figure.suptitle(
        "highD EVT-tail: Flow × QR-WM interaction-feature distribution agreement\n"
        f"{protocol['horizon_seconds']:.2f} s; traffic-feature Fréchet = {metrics['traffic_feature_frechet_distance']:.4f}; "
        f"RBF-MMD = {metrics['mmd_rbf']:.5f}",
        fontsize=13, y=0.99,
    )
    figure.text(0.5, 0.012, "Pooled real EVT-tail states versus pooled Flow × QR-WM futures; not paired trajectory reconstruction.", ha="center", fontsize=8.3)
    return _save(figure, ensure_dir(output_dir / "figures") / "01_tail_interaction_distribution.png")


def plot_tail_sampling_and_runtime(output_dir: Path) -> Path:
    """Plot Flow-start matching, physical validity, and runtime."""
    output_dir = Path(output_dir).resolve()
    report = load_json(output_dir / "flow_composition_evaluation.json")
    audit = np.load(output_dir / "flow_start_audit.npz", allow_pickle=False)
    plt = _plt()
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.75))
    speed_error = np.asarray(audit["matched_replay_ego_vx_abs_error_mps"], dtype=float)
    cutoff = float(np.quantile(speed_error, 0.995))
    axes[0].hist(speed_error, bins=48, range=(0.0, max(cutoff, 0.1)), density=True, color="#59a14f", alpha=0.8)
    axes[0].axvline(np.quantile(speed_error, 0.5), color="#222222", ls="--", label=f"P50={np.quantile(speed_error, 0.5):.3f} m/s")
    axes[0].axvline(np.quantile(speed_error, 0.95), color="#e15759", ls="--", label=f"P95={np.quantile(speed_error, 0.95):.3f} m/s")
    axes[0].set(title="Matched ego-replay speed error", xlabel="absolute error (m/s)", ylabel="density")
    axes[0].legend(fontsize=8)
    slot_names, slot_counts = np.unique(np.asarray(audit["primary_slot_name"]), return_counts=True)
    axes[1].bar(np.arange(len(slot_names)), slot_counts / slot_counts.sum(), color="#4e79a7")
    axes[1].set_xticks(np.arange(len(slot_names)), slot_names, rotation=22, ha="right")
    axes[1].set(title="Sampled primary-risk slot", ylabel="share of QR futures")
    physical, performance = report["closed_loop_distribution"]["physical_validity"], report["performance"]
    axes[2].bar((0, 1), (physical["collision_episode_rate"], physical["collision_pair_point_rate"]), color=("#e15759", "#f28e2b"))
    axes[2].set_xticks((0, 1), ("episode", "pair-point"))
    axes[2].set(title="Generated physical-validity diagnostics", ylabel="collision rate", ylim=(0.0, max(0.13, physical["collision_episode_rate"] * 1.22)))
    axes[2].text(0.03, 0.93, f"Flow matching: {performance['flow_sampling_and_replay_matching_seconds']:.1f} s\nQR evolution: {performance['batched_qr_evolution_seconds']:.1f} s\nThroughput: {performance['evolution_world_futures_per_second']:.1f} futures/s\nBatch: {performance['independent_worlds_per_qr_batch']} worlds", transform=axes[2].transAxes, va="top", fontsize=8.4)
    protocol = report["protocol"]
    figure.suptitle(f"Flow × QR-WM composition: {protocol['flow_initial_conditions']:,} Flow starts → {protocol['generated_world_futures']:,} independently seeded {protocol['horizon_seconds']:.2f} s worlds", fontsize=13, y=0.99)
    return _save(figure, ensure_dir(output_dir / "figures") / "02_flow_sampling_and_runtime.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/highd_world_model/long_tail_reproduction")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    figures = (plot_tail_interaction_distribution(output_dir), plot_tail_sampling_and_runtime(output_dir))
    manifest_path = output_dir / "study_manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        manifest.setdefault("artifacts", {})["figures"] = [str(path.relative_to(output_dir)) for path in figures]
        save_json(manifest, manifest_path)
    for figure in figures:
        print(figure)


if __name__ == "__main__":
    main()

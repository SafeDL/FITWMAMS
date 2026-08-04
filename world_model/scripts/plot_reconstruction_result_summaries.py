#!/usr/bin/env python3
"""Render reproducible figures from the formal reconstruction result artifacts.

The full-test collector intentionally stores compact, hash-verified native
metrics instead of all 32 stochastic trajectories.  The Flow × QR study stores
its distribution summary plus start-sampling audit.  This script visualizes
those formal artifacts without re-running a world model or inventing samples.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.core.utils import ensure_dir, load_json


MODEL_STYLE = {
    "ramp_world_model": ("RAMP-WM", "#59a14f"),
    "firm_world_model": ("FIRM-WM", "#e15759"),
    "semi_markov_world_model": ("Semi-Markov WM", "#b07aa1"),
    "cat_topk_world_model": ("CAT-TopK†", "#9c9c9c"),
    "qr_world_model": ("QR-WM", "#4e79a7"),
}
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
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "figure.dpi": 140,
        "savefig.dpi": 300,
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


def plot_test_conditional_reconstruction(output_dir: Path) -> Path:
    """Plot native five-second metrics from the full held-out highD test set."""
    output_dir = Path(output_dir).resolve()
    summary = load_json(output_dir / "overview" / "test_conditional_reconstruction_summary.json")
    rows: dict[str, dict[str, Any]] = summary["model_native_five_second_metrics"]
    names = [name for name in MODEL_STYLE if name in rows]
    labels = [MODEL_STYLE[name][0] for name in names]
    colors = [MODEL_STYLE[name][1] for name in names]
    values = [rows[name] for name in names]
    metrics = (
        ("five_second_FDE_m", "5 s FDE", "m"),
        ("five_second_ADE_m", "5 s ADE", "m"),
        ("gap_mae_m", "Following-gap MAE", "m"),
        ("ttc_error_s", "TTC error", "s"),
        ("drac_error_mps2", "DRAC error", "m/s²"),
    )

    plt = _plt()
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 7.6))
    for axis, (key, title, unit) in zip(axes.flat[:5], metrics):
        metric_values = [float(row[key]) for row in values]
        bars = axis.bar(np.arange(len(names)), metric_values, color=colors, edgecolor="#555555", linewidth=0.35)
        for bar, name in zip(bars, names):
            if not rows[name]["directly_comparable"]:
                bar.set_hatch("///")
        axis.set_xticks(np.arange(len(names)), labels, rotation=18, ha="right")
        axis.set_title(title)
        axis.set_ylabel(unit)
        _bar_labels(axis, metric_values)
    note = axes.flat[5]
    note.axis("off")
    protocol = summary["protocol"]
    note.text(
        0.04, 0.94,
        "Evaluation scope\n"
        f"• {protocol['num_sequences']:,} held-out highD sequences\n"
        "• 5 s closed-loop background reconstruction\n"
        "• real C0+B0, map and logged ego replay\n"
        "• lower is better for all plotted metrics\n\n"
        "† CAT-TopK is hatched because its archived\n"
        "future-action summary makes it information-\n"
        "asymmetric; it is a reference, not a strict rank.",
        va="top", fontsize=10.2,
    )
    figure.suptitle("Full highD held-out test: native conditional reconstruction", fontsize=14, y=0.99)
    figure.text(0.5, 0.012, "Values are extracted from hash-verified full-test source reports.", ha="center", fontsize=8.7)
    return _save(figure, ensure_dir(output_dir / "overview") / "01_model_native_comparison.png")


def _risk_quantile_panel(axis: Any, name: str, report: dict[str, Any]) -> None:
    label, unit = RISK_STYLE[name]
    values = report["closed_loop_distribution"]["risk_variable_distribution"][name]
    levels = ("q90", "q95", "q99")
    x = np.arange(len(levels))
    width = 0.36
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
    """Plot the reported real-vs-synthetic tail interaction distribution summary."""
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
    figure.suptitle(
        "All highD EVT-tail: Flow × QR-WM interaction-feature distribution agreement\n"
        f"Traffic-feature Fréchet = {metrics['traffic_feature_frechet_distance']:.4f}; "
        f"RBF-MMD = {metrics['mmd_rbf']:.5f}",
        fontsize=13, y=0.99,
    )
    figure.text(0.5, 0.012, "Upper-quantile values compare pooled real EVT-tail states with pooled Flow × QR-WM futures; this is not paired trajectory reconstruction.", ha="center", fontsize=8.3)
    return _save(figure, ensure_dir(output_dir / "figures") / "01_tail_interaction_distribution.png")


def plot_tail_sampling_and_runtime(output_dir: Path) -> Path:
    """Plot Flow-start audit, physical validity, and throughput from formal artifacts."""
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
    log_prob = np.asarray(audit["conditional_log_prob"], dtype=float)
    inset = axes[1].inset_axes((0.56, 0.50, 0.40, 0.42))
    bounds = np.quantile(log_prob, (0.01, 0.99))
    inset.hist(log_prob, bins=36, range=tuple(bounds), color="#bab0ab")
    inset.set_title("Flow log density", fontsize=7)
    inset.tick_params(labelsize=6)

    physical = report["closed_loop_distribution"]["physical_validity"]
    performance = report["performance"]
    axes[2].bar((0, 1), (physical["collision_episode_rate"], physical["collision_pair_point_rate"]), color=("#e15759", "#f28e2b"))
    axes[2].set_xticks((0, 1), ("episode", "pair-point"))
    axes[2].set(title="Generated physical-validity diagnostics", ylabel="collision rate", ylim=(0.0, max(0.13, physical["collision_episode_rate"] * 1.22)))
    axes[2].text(
        0.03, 0.93,
        f"Flow matching: {performance['flow_sampling_and_replay_matching_seconds']:.1f} s\n"
        f"QR evolution: {performance['batched_qr_evolution_seconds']:.1f} s\n"
        f"Throughput: {performance['evolution_world_futures_per_second']:.1f} futures/s\n"
        f"Batch: {performance['independent_worlds_per_qr_batch']} independent worlds\n"
        f"Audited seeds: {len(np.unique(audit['world_seed'])):,} / {len(audit['world_seed']):,}",
        transform=axes[2].transAxes, va="top", fontsize=8.4,
    )
    protocol = report["protocol"]
    figure.suptitle(
        f"Flow × QR-WM composition audit: {protocol['flow_initial_conditions']:,} Flow starts → "
        f"{protocol['generated_world_futures']:,} independently seeded 5 s worlds",
        fontsize=13, y=0.99,
    )
    figure.text(0.5, 0.012, "Every world_seed controls only the QR START behavior latent; batched execution shares no world state or ego replay.", ha="center", fontsize=8.5)
    return _save(figure, ensure_dir(output_dir / "figures") / "02_flow_sampling_and_runtime.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("all", "test", "tail"), default="all")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "results/highd_world_model/test_conditional_reconstruction")
    parser.add_argument("--tail-dir", type=Path, default=ROOT / "results/highd_world_model/long_tail_reproduction")
    args = parser.parse_args()
    outputs: list[Path] = []
    if args.only in ("all", "test"):
        outputs.append(plot_test_conditional_reconstruction(args.test_dir))
    if args.only in ("all", "tail"):
        outputs.extend((plot_tail_interaction_distribution(args.tail_dir), plot_tail_sampling_and_runtime(args.tail_dir)))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the complete Flow×QR-WM long-tail study without conflating protocols.

This is the sole formal entry point.  It first runs the end-to-end Flow study,
which draws new ``C0+B0`` samples and is evaluated as a distribution, then
runs the START/ROLL study on held-out logged EVT-tail conditions and ego
replay, where per-trajectory errors are meaningful.

The resulting JSON records the raw highD limit precisely: ``START [0, 1 s]``
plus ``ROLL (1, 5.96 s]``.  That is 1.00 s of B0-conditioned reconstruction
and 4.96 s of B0-free continuation, with no invented terminal state.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from normalizing_flow.src.features import (  # noqa: E402
    EGO_FEATURES,
    SLOT_NAMES,
    TRAJECTORY_FEATURES,
    slot_feature_index,
    trajectory_feature_index,
)
from normalizing_flow.src.metrics import (  # noqa: E402
    distribution_match_metrics,
    occupancy_metrics,
    physical_validity_metrics,
)
from world_model.src.core.batching import sequence_field_names, to_device_batch  # noqa: E402
from world_model.src.core.data import SPLIT_TO_INDEX  # noqa: E402
from world_model.src.core.flow_composition import (  # noqa: E402
    FLOW_COMPOSITION_SEED,
    load_flow_tail_starts,
)
from world_model.src.core.initial_behavior_anchor import (  # noqa: E402
    FrozenLegacyFlowSchema,
    summarize_first_second_states,
)
from world_model.src.core.long_tail_metrics import (  # noqa: E402
    DT_S,
    collision_metrics,
    distribution_values,
    empirical_distance,
    feature_distribution_distance,
    following_error_metrics,
    histogram_kl_divergence,
    kinematic_reconstruction_metrics,
    masked_mean,
    social_response_metrics,
    traffic_fields,
    trajectory_metrics,
)
from world_model.src.core.sequential_dataset import (  # noqa: E402
    ensure_frozen_flow_behavior_anchor_cache,
    is_canonical_qr_manifest,
    load_sequential_dataset,
)
from world_model.src.core.utils import (  # noqa: E402
    ensure_dir,
    file_sha256,
    load_json,
    save_json,
    select_device,
    set_seed,
    setup_logging,
)
from world_model.src.qr.environment import BatchedQRWorldModelEnvironment  # noqa: E402
from world_model.src.qr.flow_evaluation import evaluate_flow_composition, replay_states_to_ego_controls  # noqa: E402
from world_model.src.qr.train import load_qr_checkpoint, require_canonical_qr_checkpoint  # noqa: E402


RISK_STYLE = {
    "ttc_s": ("TTC", "s"),
    "drac_mps2": ("DRAC", "m/s²"),
    "gap_m": ("Following gap", "m"),
    "relative_speed_mps": ("Closing speed", "m/s"),
}


def _plt():
    """Load Matplotlib lazily so non-plotting imports remain lightweight."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.22,
    })
    return plt


def _save_figure(figure: Any, path: Path) -> Path:
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


def _plot_tail_interaction_distribution(output_dir: Path) -> Path:
    """Plot real-versus-synthetic tail interaction distributions."""
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
    wasserstein = [float(risk[name]["wasserstein_1"]) for name in RISK_STYLE]
    axes[1, 1].bar(np.arange(len(labels)), ks, color="#f28e2b")
    axes[1, 1].set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right")
    axes[1, 1].set_title("Kolmogorov–Smirnov distance")
    axes[1, 1].set_ylabel("lower is better")
    _bar_labels(axes[1, 1], ks)
    axes[1, 2].bar(np.arange(len(labels)), wasserstein, color="#e15759")
    axes[1, 2].set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right")
    axes[1, 2].set_title("Wasserstein-1 distance")
    axes[1, 2].set_ylabel("native feature unit; lower is better")
    _bar_labels(axes[1, 2], wasserstein)
    metrics, protocol = report["closed_loop_distribution"], report["protocol"]
    figure.suptitle(
        "highD EVT-tail: Flow × QR-WM interaction-feature distribution agreement\n"
        f"{protocol['horizon_seconds']:.2f} s; traffic-feature Fréchet = {metrics['traffic_feature_frechet_distance']:.4f}; "
        f"RBF-MMD = {metrics['mmd_rbf']:.5f}",
        fontsize=13, y=0.99,
    )
    figure.text(0.5, 0.012, "Pooled real EVT-tail states versus pooled Flow × QR-WM futures; not paired trajectory reconstruction.", ha="center", fontsize=8.3)
    return _save_figure(figure, ensure_dir(output_dir / "figures") / "01_tail_interaction_distribution.png")


def _plot_tail_sampling_and_runtime(output_dir: Path) -> Path:
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
    return _save_figure(figure, ensure_dir(output_dir / "figures") / "02_flow_sampling_and_runtime.png")


def _json_float(value: Any) -> float:
    return float(np.asarray(value).item())


def _quantiles(values: np.ndarray) -> dict[str, float | int]:
    value = np.asarray(values, np.float64).reshape(-1)
    value = value[np.isfinite(value)]
    if not len(value):
        return {"count": 0, "p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "p99": float("nan")}
    return {
        "count": int(len(value)),
        "p50": float(np.quantile(value, 0.50)),
        "p90": float(np.quantile(value, 0.90)),
        "p95": float(np.quantile(value, 0.95)),
        "p99": float(np.quantile(value, 0.99)),
    }


def _error_summary(error: np.ndarray, target: np.ndarray | None = None) -> dict[str, Any]:
    value = np.asarray(error, np.float64).reshape(-1)
    value = value[np.isfinite(value)]
    if not len(value):
        return {"count": 0, "mae": float("nan"), "rmse": float("nan"), **_quantiles(value)}
    out: dict[str, Any] = {
        "count": int(len(value)),
        "mae": float(np.mean(np.abs(value))),
        "rmse": float(np.sqrt(np.mean(np.square(value)))),
        **_quantiles(np.abs(value)),
    }
    if target is not None:
        denominator = np.maximum(np.abs(np.asarray(target, np.float64).reshape(-1)), 0.1)
        denominator = denominator[np.isfinite(np.asarray(target, np.float64).reshape(-1))]
        if len(denominator) == len(value):
            out["mean_relative_error_floor_0p1"] = float(np.mean(np.abs(value) / denominator))
    return out


def _distance_with_moments(real: np.ndarray, generated: np.ndarray) -> dict[str, Any]:
    left = np.asarray(real, np.float64).reshape(-1)
    right = np.asarray(generated, np.float64).reshape(-1)
    left, right = left[np.isfinite(left)], right[np.isfinite(right)]
    report = empirical_distance(left, right)
    if not report.get("available", False):
        return report
    report.update({
        "real_mean": float(left.mean()),
        "generated_mean": float(right.mean()),
        "mean_absolute_error": float(abs(left.mean() - right.mean())),
        "real_std": float(left.std()),
        "generated_std": float(right.std()),
        "std_absolute_error": float(abs(left.std() - right.std())),
    })
    return report


def _probability_error(
    real: np.ndarray, generated: np.ndarray, *, threshold: float, less_than: bool,
) -> dict[str, float]:
    left = np.asarray(real, np.float64).reshape(-1)
    right = np.asarray(generated, np.float64).reshape(-1)
    left, right = left[np.isfinite(left)], right[np.isfinite(right)]
    if not len(left) or not len(right):
        return {"real_probability": float("nan"), "generated_probability": float("nan"), "absolute_error": float("nan"), "relative_error": float("nan")}
    if less_than:
        expected, observed = float(np.mean(left < threshold)), float(np.mean(right < threshold))
    else:
        expected, observed = float(np.mean(left > threshold)), float(np.mean(right > threshold))
    error = abs(observed - expected)
    return {
        "real_probability": expected,
        "generated_probability": observed,
        "absolute_error": float(error),
        "relative_error": float(error / max(expected, 1.0e-6)),
    }


def _event_structure_report(
    real_mask: np.ndarray,
    real_primary: np.ndarray,
    generated_mask: np.ndarray,
    generated_primary: np.ndarray,
) -> dict[str, Any]:
    def keys(mask: np.ndarray, primary: np.ndarray) -> list[tuple[int, int]]:
        pattern = np.sum(np.asarray(mask, bool) * (1 << np.arange(6)), axis=1)
        return [(int(item), int(slot)) for item, slot in zip(pattern, primary)]

    real_counts, generated_counts = Counter(keys(real_mask, real_primary)), Counter(keys(generated_mask, generated_primary))
    all_keys = sorted(set(real_counts) | set(generated_counts))
    real_total, generated_total = max(sum(real_counts.values()), 1), max(sum(generated_counts.values()), 1)
    p = np.asarray([real_counts[key] / real_total for key in all_keys], np.float64)
    q = np.asarray([generated_counts[key] / generated_total for key in all_keys], np.float64)
    mixture = 0.5 * (p + q)
    js = 0.5 * np.sum(np.where(p > 0.0, p * np.log(p / np.maximum(mixture, 1.0e-12)), 0.0))
    js += 0.5 * np.sum(np.where(q > 0.0, q * np.log(q / np.maximum(mixture, 1.0e-12)), 0.0))
    low_frequency = [key for key, probability in zip(all_keys, p) if 0.0 < probability <= 0.05]
    return {
        "total_variation_distance": float(0.5 * np.abs(p - q).sum()),
        "jensen_shannon_divergence_nats": float(js),
        "mean_absolute_frequency_error": float(np.abs(p - q).mean()) if len(p) else float("nan"),
        "low_frequency_structure_recall": float(np.mean([generated_counts[key] > 0 for key in low_frequency])) if low_frequency else 1.0,
        "unseen_or_illegal_generated_structure_rate": float(
            sum(value for key, value in generated_counts.items() if key not in real_counts) / generated_total
        ),
        "frequencies": {
            f"mask_{pattern}_primary_{primary}": {
                "real": float(real_counts[(pattern, primary)] / real_total),
                "generated": float(generated_counts[(pattern, primary)] / generated_total),
                "absolute_error": float(abs(real_counts[(pattern, primary)] / real_total - generated_counts[(pattern, primary)] / generated_total)),
            }
            for pattern, primary in all_keys
        },
    }


def _subsample(value: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(value) <= count:
        return np.asarray(value, np.float64)
    return np.asarray(value[rng.choice(len(value), size=count, replace=False)], np.float64)


def _mean_pairwise_l2(left: np.ndarray, right: np.ndarray) -> float:
    total = 0.0
    count = 0
    for start in range(0, len(left), 128):
        distance = np.linalg.norm(left[start : start + 128, None] - right[None], axis=-1)
        total += float(distance.sum())
        count += int(distance.size)
    return total / max(count, 1)


def _joint_distribution_report(real: np.ndarray, generated: np.ndarray, *, seed: int) -> dict[str, float | int]:
    """RBF-MMD, energy distance, and sliced W1 in a shared standardized space."""
    rng = np.random.default_rng(int(seed))
    left, right = _subsample(real, 1024, rng), _subsample(generated, 1024, rng)
    combined = np.concatenate((left, right), axis=0)
    scale = np.maximum(combined.std(axis=0), 1.0e-4)
    mean = combined.mean(axis=0)
    left, right = (left - mean) / scale, (right - mean) / scale
    probe = np.linalg.norm(left[: min(len(left), 256), None] - right[None, : min(len(right), 256)], axis=-1)
    bandwidth = float(np.median(probe).clip(min=1.0e-4))

    def kernel(first: np.ndarray, second: np.ndarray) -> float:
        total = 0.0
        count = 0
        for start in range(0, len(first), 128):
            squared = np.square(first[start : start + 128, None] - second[None]).sum(axis=-1)
            total += float(np.exp(-squared / (2.0 * bandwidth**2)).sum())
            count += int(squared.size)
        return total / max(count, 1)

    directions = rng.normal(size=(64, left.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1.0e-8)
    projected = []
    for direction in directions:
        projected.append(float(np.mean(np.abs(np.sort(left @ direction) - np.sort(right @ direction)))))
    return {
        "samples_per_distribution": int(min(len(left), len(right))),
        "rbf_mmd": float(kernel(left, left) + kernel(right, right) - 2.0 * kernel(left, right)),
        "energy_distance": float(2.0 * _mean_pairwise_l2(left, right) - _mean_pairwise_l2(left, left) - _mean_pairwise_l2(right, right)),
        "sliced_wasserstein_1": float(np.mean(projected)),
    }


def _flow_states(features: np.ndarray, slot_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized raw Flow C0 decoder used only for initial-state diagnostics."""
    flow = np.asarray(features, np.float32)
    slots = np.asarray(slot_mask, bool)
    states = np.zeros((len(flow), 7, 6), np.float32)
    valid = np.zeros((len(flow), 7), bool)
    states[:, 0, 2:6] = flow[:, :4]
    valid[:, 0] = True
    for index, slot in enumerate(SLOT_NAMES):
        offset = 4 + index * 6
        states[:, index + 1, :2] = flow[:, offset : offset + 2]
        states[:, index + 1, 2:4] = flow[:, :2] + flow[:, offset + 2 : offset + 4]
        states[:, index + 1, 4:6] = flow[:, offset + 4 : offset + 6]
    valid[:, 1:] = slots
    return states, valid


def _initial_flow_report(
    starts: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    donors: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Compare Flow samples against the exact supported EVT-tail cohort."""
    flow_checkpoint = ROOT / "results/highd_tail_flow/checkpoints/best_tail_conditional_maf.pt"
    # The Flow sampler has already been loaded by ``load_flow_tail_starts``.
    # This audit only needs its frozen data coordinates, so do not instantiate
    # a second GPU Flow model merely to read them.
    with np.load(ROOT / "results/highd_tail_flow/dataset.npz", allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    schema = load_json(ROOT / "results/highd_tail_flow/dataset_schema.json")
    donor_ids = set(np.asarray(cache["sequence_id"])[np.unique(donors)].astype(str).tolist())
    real_rows = np.asarray([
        index for index, identifier in enumerate(np.asarray(arrays["segment_id"]).astype(str)) if identifier in donor_ids
    ], np.int64)
    if not len(real_rows):
        raise RuntimeError("could not identify the Flow-supported EVT-tail reference cohort")
    real_features = np.asarray(arrays["features"])[real_rows]
    real_valid = np.asarray(arrays["feature_valid"])[real_rows]
    real_slots = np.asarray(arrays["slot_mask"])[real_rows]
    real_primary = np.asarray(arrays["primary_slot_index"])[real_rows]
    generated_features = np.asarray(starts["features"], np.float32)
    generated_valid = np.asarray(starts["feature_valid"], bool)
    generated_slots = np.asarray(starts["slot_mask"], bool)
    generated_primary = np.asarray(starts["primary_slot_index"], np.int64)
    real_states, real_present = _flow_states(real_features, real_slots)
    generated_states, generated_present = _flow_states(generated_features, generated_slots)
    real_fields = traffic_fields(real_states[:, None, 1:], real_states[:, None, 0], real_present[:, None, 1:])
    generated_fields = traffic_fields(generated_states[:, None, 1:], generated_states[:, None, 0], generated_present[:, None, 1:])
    real_values, generated_values = distribution_values(real_fields), distribution_values(generated_fields)
    b0_report: dict[str, Any] = {}
    for local, name in enumerate(TRAJECTORY_FEATURES):
        real_columns, generated_columns = [], []
        for slot_index, slot_name in enumerate(SLOT_NAMES):
            feature = trajectory_feature_index(slot_name, name)
            real_columns.append(real_features[real_slots[:, slot_index], feature])
            generated_columns.append(generated_features[generated_slots[:, slot_index], feature])
        b0_report[name] = _distance_with_moments(np.concatenate(real_columns), np.concatenate(generated_columns))
    selected_initial = {
        "speed_mps": _distance_with_moments(real_values["speed_mps"], generated_values["speed_mps"]),
        "acceleration_mps2": _distance_with_moments(real_values["acceleration_mps2"], generated_values["acceleration_mps2"]),
        "gap_m": _distance_with_moments(real_values["gap_m"], generated_values["gap_m"]),
        "ttc_s": _distance_with_moments(real_values["ttc_s"], generated_values["ttc_s"]),
        "drac_mps2": _distance_with_moments(real_values["drac_mps2"], generated_values["drac_mps2"]),
        "closing_rate_mps": _distance_with_moments(real_values["relative_speed_mps"], generated_values["relative_speed_mps"]),
    }
    return {
        "reference": {
            "scope": "Flow-supported all-highD EVT-tail cohort",
            "supported_real_sequences": int(len(real_rows)),
            "generated_flow_samples": int(len(generated_features)),
            "flow_checkpoint": {"path": str(flow_checkpoint), "sha256": file_sha256(flow_checkpoint)},
        },
        "event_structure": _event_structure_report(real_slots, real_primary, generated_slots, generated_primary),
        "occupancy": occupancy_metrics(real_slots, generated_slots, real_primary, generated_primary),
        "physical_validity": physical_validity_metrics(generated_features, generated_slots),
        "c0_initial_state_distribution": selected_initial,
        "c0_tail_probability_error": {
            "ttc_lt_1s": _probability_error(real_values["ttc_s"], generated_values["ttc_s"], threshold=1.0, less_than=True),
            "drac_gt_3mps2": _probability_error(real_values["drac_mps2"], generated_values["drac_mps2"], threshold=3.0, less_than=False),
            "gap_lt_2m": _probability_error(real_values["gap_m"], generated_values["gap_m"], threshold=2.0, less_than=True),
        },
        "b0_summary_distribution": b0_report,
        "all_76_feature_distribution": distribution_match_metrics(
            real_features, generated_features, real_valid, generated_valid, list(schema["feature_names"]),
        ),
        "conditional_joint_c0_b0_distribution": _joint_distribution_report(
            np.asarray(arrays["features_normalized"])[real_rows],
            np.asarray(starts["features_normalized"]), seed=FLOW_COMPOSITION_SEED,
        ),
    }


def _risk_by_episode(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    following = np.asarray(fields["following_valid"], bool)
    gap, ttc = np.asarray(fields["gap_m"], np.float64), np.asarray(fields["ttc_s"], np.float64)
    drac, closing = np.asarray(fields["drac_mps2"], np.float64), np.asarray(fields["relative_speed_mps"], np.float64)
    acceleration = np.asarray(fields["states"], np.float64)[..., 1:, 4]
    count = len(following)
    min_gap = np.full(count, np.nan)
    min_ttc = np.full(count, np.nan)
    max_drac = np.full(count, np.nan)
    max_closing = np.full(count, np.nan)
    max_braking = np.full(count, np.nan)
    peak_time = np.full(count, np.nan)
    primary = np.full(count, -1, np.int64)
    for index in range(count):
        mask = following[index]
        if mask.any():
            min_gap[index] = float(gap[index][mask].min())
            min_ttc[index] = float(ttc[index][mask].min())
            max_drac[index] = float(drac[index][mask].max())
            max_closing[index] = float(closing[index][mask].max())
            ranked = np.where(mask, drac[index], -np.inf)
            time, rear, front = np.unravel_index(int(np.argmax(ranked)), ranked.shape)
            peak_time[index] = float(time * DT_S)
            primary[index] = int(rear if rear > 0 else front if front > 0 else -1)
        finite_acceleration = acceleration[index][np.isfinite(acceleration[index])]
        if len(finite_acceleration):
            max_braking[index] = float(-finite_acceleration.min())
    collision = np.asarray(fields["collision"], bool).any(axis=(1, 2, 3))
    near_collision = np.isfinite(min_gap) & (min_gap < 2.0)
    return {
        "minimum_gap_m": min_gap,
        "minimum_ttc_s": min_ttc,
        "maximum_drac_mps2": max_drac,
        "maximum_closing_rate_mps": max_closing,
        "maximum_braking_mps2": max_braking,
        "risk_peak_time_s": peak_time,
        "primary_risk_agent_index": primary,
        "collision": collision,
        "near_collision_gap_lt_2m": near_collision,
    }


def _event_series(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    following = np.asarray(fields["following_valid"], bool)
    states = np.asarray(fields["states"], np.float64)
    return {
        "high_risk_following_ttc_lt_3s": np.any((np.asarray(fields["ttc_s"]) < 3.0) & following, axis=(2, 3)),
        "close_interaction_gap_lt_8m": np.any((np.asarray(fields["gap_m"]) < 8.0) & following, axis=(2, 3)),
        "near_collision_gap_lt_2m": np.any((np.asarray(fields["gap_m"]) < 2.0) & following, axis=(2, 3)),
        "hard_braking_ax_lt_minus_1p5": np.any(states[:, :, 1:, 4] < -1.5, axis=2),
        "high_speed_approach_closing_gt_5": np.any((np.asarray(fields["relative_speed_mps"]) > 5.0) & following, axis=(2, 3)),
        "collision": np.asarray(fields["collision"], bool).any(axis=(2, 3)),
    }


def _classification_report(real: np.ndarray, generated: np.ndarray) -> dict[str, Any]:
    expected = np.asarray(real, bool).any(axis=1)
    observed = np.asarray(generated, bool).any(axis=1)
    tp = int(np.sum(expected & observed))
    fp = int(np.sum(~expected & observed))
    fn = int(np.sum(expected & ~observed))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    both = expected & observed
    starts, durations = [], []
    for truth, prediction in zip(np.asarray(real)[both], np.asarray(generated)[both]):
        starts.append(abs(int(np.argmax(truth)) - int(np.argmax(prediction))) * DT_S)
        durations.append(abs(int(truth.sum()) - int(prediction.sum())) * DT_S)
    return {
        "real_episode_rate": float(expected.mean()),
        "generated_episode_rate": float(observed.mean()),
        "absolute_rate_error": float(abs(expected.mean() - observed.mean())),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2.0 * precision * recall / max(precision + recall, 1.0e-12)),
        "both_present_sequences": int(both.sum()),
        "start_time_mae_s_when_both_present": float(np.mean(starts)) if starts else float("nan"),
        "duration_mae_s_when_both_present": float(np.mean(durations)) if durations else float("nan"),
    }


def _event_report(real_fields: dict[str, np.ndarray], generated_fields: dict[str, np.ndarray]) -> dict[str, Any]:
    expected, observed = _event_series(real_fields), _event_series(generated_fields)
    return {name: _classification_report(expected[name], observed[name]) for name in expected}


def _pairwise_error_report(
    predicted: dict[str, np.ndarray], target: dict[str, np.ndarray], *, pair: str,
) -> dict[str, float]:
    mask = np.asarray(target["following_valid"], bool)
    participants = mask.shape[2]
    row, col = np.indices((participants, participants))
    if pair == "ego_background":
        selection = (row == 0) ^ (col == 0)
    elif pair == "background_background":
        selection = (row > 0) & (col > 0)
    else:
        raise ValueError(f"unknown pair scope: {pair}")
    take = mask & selection[None, None]
    return {
        "pair_points": int(take.sum()),
        "gap_mae_m": masked_mean(np.abs(predicted["gap_m"] - target["gap_m"]), take),
        "ttc_mae_s": masked_mean(np.abs(predicted["ttc_s"] - target["ttc_s"]), take),
        "drac_mae_mps2": masked_mean(np.abs(predicted["drac_mps2"] - target["drac_mps2"]), take),
        "closing_rate_mae_mps": masked_mean(np.abs(predicted["relative_speed_mps"] - target["relative_speed_mps"]), take),
    }


def _high_risk_graph_report(real_fields: dict[str, np.ndarray], generated_fields: dict[str, np.ndarray]) -> dict[str, Any]:
    def degree(fields: dict[str, np.ndarray]) -> np.ndarray:
        edge = np.asarray(fields["following_valid"], bool) & (np.asarray(fields["ttc_s"]) < 3.0)
        undirected = edge | np.swapaxes(edge, -1, -2)
        return undirected.sum(axis=-1).astype(np.float64)

    return _distance_with_moments(degree(real_fields), degree(generated_fields))


def _action_correlation_error(predicted_actions: np.ndarray, target_actions: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    reference, generated = [], []
    present = np.asarray(valid, bool)
    for row in range(len(present)):
        for left in range(present.shape[-1]):
            for right in range(left + 1, present.shape[-1]):
                take = present[row, :, left] & present[row, :, right]
                if int(take.sum()) < 5:
                    continue
                for component in range(2):
                    first = target_actions[row, take, left, component]
                    second = target_actions[row, take, right, component]
                    third = predicted_actions[row, take, left, component]
                    fourth = predicted_actions[row, take, right, component]
                    if np.std(first) > 1.0e-6 and np.std(second) > 1.0e-6 and np.std(third) > 1.0e-6 and np.std(fourth) > 1.0e-6:
                        reference.append(float(np.corrcoef(first, second)[0, 1]))
                        generated.append(float(np.corrcoef(third, fourth)[0, 1]))
    if not reference:
        return {"available": False}
    return {
        "available": True,
        "pairs": int(len(reference)),
        "mean_absolute_correlation_error": float(np.mean(np.abs(np.asarray(reference) - np.asarray(generated)))),
        "real_mean_correlation": float(np.mean(reference)),
        "generated_mean_correlation": float(np.mean(generated)),
    }


def _boundary_report(predicted: np.ndarray, target: np.ndarray, valid: np.ndarray, plans: np.ndarray) -> dict[str, Any]:
    """Measure the 1 s boundary as a physical increment, not an impossible state jump."""
    if predicted.shape[1] <= 25 or plans.shape[1] <= 5:
        return {"available": False}
    present = np.asarray(valid, bool)
    pre_valid = present[:, 23] & present[:, 24]
    post_valid = present[:, 24] & present[:, 25]
    boundary_valid = pre_valid & post_valid
    pred_pre = predicted[:, 24] - predicted[:, 23]
    pred_post = predicted[:, 25] - predicted[:, 24]
    target_pre = target[:, 24] - target[:, 23]
    target_post = target[:, 25] - target[:, 24]
    executed = np.concatenate([plans[:, response, :5] for response in range(plans.shape[1])], axis=1)
    executed = executed[:, :predicted.shape[1]]
    control_delta = np.linalg.norm(executed[:, 25] - executed[:, 24], axis=-1)
    state_increment_change = np.linalg.norm(pred_post - pred_pre, axis=-1)
    target_increment_change = np.linalg.norm(target_post - target_pre, axis=-1)
    jerk = np.linalg.norm(predicted[:, 25, :, 4:6] - predicted[:, 24, :, 4:6], axis=-1) / DT_S
    reference_jerk = np.linalg.norm(target[:, 25, :, 4:6] - target[:, 24, :, 4:6], axis=-1) / DT_S
    return {
        "available": True,
        "definition": "boundary compares the t=1.0s physical increment with its adjacent 25Hz increments; positions are not reset",
        "control_delta_norm": _quantiles(control_delta[boundary_valid]),
        "state_increment_change_norm": _quantiles(state_increment_change[boundary_valid]),
        "state_increment_change_error_norm": _error_summary(
            (state_increment_change - target_increment_change)[boundary_valid]
        ),
        "boundary_jerk_mps3": _quantiles(jerk[boundary_valid]),
        "boundary_jerk_error_mps3": _error_summary((jerk - reference_jerk)[boundary_valid]),
    }


def _b0_reconstruction_report(
    predicted: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    target_anchor: np.ndarray,
    target_anchor_valid: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    states = np.concatenate((target[:, :1], predicted[:, :25]), axis=1)
    states_valid = np.concatenate((valid[:, :1], valid[:, :25]), axis=1)
    with torch.no_grad():
        summary, summary_valid = summarize_first_second_states(
            torch.from_numpy(states), torch.from_numpy(states_valid)
        )
    # This helper receives background-only tensors, so its six returned rows
    # already align with the six Flow slots (there is no ego row to discard).
    observed, present = summary.numpy(), summary_valid.numpy()
    take = present & np.asarray(target_anchor_valid, bool)
    error = observed - np.asarray(target_anchor, np.float32)
    dimensions: dict[str, Any] = {}
    for index, name in enumerate(TRAJECTORY_FEATURES):
        dimensions[name] = _error_summary(error[..., index][take], np.asarray(target_anchor)[..., index][take])
    return {
        "definition": "B0_hat is re-extracted from generated C0 plus the first 25 generated 25Hz states using the frozen 26-state Flow summary",
        "valid_slot_windows": int(take.sum()),
        "per_dimension": dimensions,
        "mean_absolute_error_all_dimensions": float(np.mean(np.abs(error[take]))) if take.any() else float("nan"),
    }, error, take


def _multimodal_report(
    samples: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    ego: np.ndarray,
) -> dict[str, Any]:
    draws = np.asarray(samples, np.float64)
    reference, present = np.asarray(target, np.float64), np.asarray(valid, bool)
    distance = np.linalg.norm(draws[..., :2] - reference[None, ..., :2], axis=-1)
    weights = present[None].astype(np.float64)
    per_episode_ade = (distance * weights).sum(axis=(2, 3)) / weights.sum(axis=(2, 3)).clip(min=1.0)
    final = present[:, -1]
    per_episode_fde = (distance[:, :, -1] * final[None]).sum(axis=2) / final.sum(axis=1).clip(min=1)[None]
    pair_distance: list[np.ndarray] = []
    for first in range(len(draws)):
        for second in range(first + 1, len(draws)):
            branch = np.linalg.norm(draws[first, ..., :2] - draws[second, ..., :2], axis=-1)
            pair_distance.append((branch * present).sum(axis=(1, 2)) / present.sum(axis=(1, 2)).clip(min=1))
    pair = np.stack(pair_distance) if pair_distance else np.zeros((1, draws.shape[1]))
    lower, upper = np.quantile(draws[..., :2], (0.05, 0.95), axis=0)
    coverage = ((reference[..., :2] >= lower) & (reference[..., :2] <= upper)).all(axis=-1)
    real_events = _event_series(traffic_fields(reference, ego, present))
    generated_events: dict[str, list[np.ndarray]] = {key: [] for key in real_events}
    for branch in draws:
        fields = traffic_fields(branch, ego, present)
        for key, value in _event_series(fields).items():
            generated_events[key].append(value)
    event_support = {
        name: _classification_report(real_events[name], np.any(value, axis=0))
        for name, value in generated_events.items()
    }
    finite_branch = np.isfinite(draws).all(axis=(2, 3, 4))
    return {
        "samples_per_condition": int(len(draws)),
        "minADE_at_K_m": float(np.mean(np.min(per_episode_ade, axis=0))),
        "minFDE_at_K_m": float(np.mean(np.min(per_episode_fde, axis=0))),
        "sample_mean_ADE_m": float(np.mean(per_episode_ade)),
        "sample_mean_FDE_m": float(np.mean(per_episode_fde)),
        "pairwise_trajectory_diversity_ADE_m": float(np.mean(pair)),
        "trajectory_energy_score_m": float(np.mean(per_episode_ade.mean(axis=0) - 0.5 * pair.mean(axis=0))),
        "empirical_90pct_position_coverage": float(np.mean(coverage[present])),
        "finite_branch_rate": float(np.mean(finite_branch)),
        "event_mode_support": event_support,
    }


def _paired_reconstruction_report(
    model,
    arrays: dict[str, np.ndarray],
    rows: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    multimodal_samples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    fields = sequence_field_names(arrays)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    valids: list[np.ndarray] = []
    egos: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    anchor_valids: list[np.ndarray] = []
    plans: list[np.ndarray] = []
    target_actions: list[np.ndarray] = []
    branches: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        index = rows[start : start + batch_size]
        values = tuple(torch.from_numpy(np.asarray(arrays[name][index]).copy()) for name in fields)
        batch = to_device_batch(values, fields, device)
        with torch.no_grad():
            deterministic = model.rollout_reconstruction(batch, deterministic=True, start_mode=True)
        prediction = deterministic["predicted_states"].cpu().numpy()
        target = deterministic["target_states"].cpu().numpy()
        target_valid = deterministic["target_valid"].cpu().numpy()
        predictions.append(prediction[:, :, 1:])
        targets.append(target[:, :, 1:])
        valids.append(target_valid[:, :, 1:])
        egos.append(target[:, :, 0])
        anchors.append(batch["behavior_anchor_raw"].cpu().numpy())
        anchor_valids.append(batch["behavior_anchor_valid"].cpu().numpy())
        plans.append(deterministic["background_future_actions"].cpu().numpy())
        target_actions.append(batch["actions_highd"].cpu().numpy())
        sample_rows = [prediction[:, :, 1:]]
        for branch in range(1, multimodal_samples):
            torch.manual_seed(int(seed) + start * 1000 + branch)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed) + start * 1000 + branch)
            with torch.no_grad():
                stochastic = model.rollout_reconstruction(batch, deterministic=False, start_mode=True)
            sample_rows.append(stochastic["predicted_states"][:, :, 1:].cpu().numpy())
        branches.append(np.stack(sample_rows, axis=0))
    predicted, target, valid, ego = map(np.concatenate, (predictions, targets, valids, egos))
    target_anchor, target_anchor_valid = map(np.concatenate, (anchors, anchor_valids))
    plan, actions = map(np.concatenate, (plans, target_actions))
    samples = np.concatenate(branches, axis=1)
    generated_fields = traffic_fields(predicted, ego, valid)
    target_fields = traffic_fields(target, ego, valid)
    generated_start_fields = traffic_fields(predicted[:, :25], ego[:, :25], valid[:, :25])
    target_start_fields = traffic_fields(target[:, :25], ego[:, :25], valid[:, :25])
    b0, b0_error, b0_take = _b0_reconstruction_report(
        predicted, target, valid, target_anchor, target_anchor_valid,
    )
    risk_real, risk_generated = _risk_by_episode(target_start_fields), _risk_by_episode(generated_start_fields)
    risk_error = {
        name: _error_summary(risk_generated[name] - risk_real[name], risk_real[name])
        for name in ("minimum_gap_m", "minimum_ttc_s", "maximum_drac_mps2", "maximum_closing_rate_mps", "maximum_braking_mps2")
    }
    both_peak = np.isfinite(risk_real["risk_peak_time_s"]) & np.isfinite(risk_generated["risk_peak_time_s"])
    identity = (risk_real["primary_risk_agent_index"] >= 1) & (risk_generated["primary_risk_agent_index"] >= 1)
    risk_error.update({
        "risk_peak_time_mae_s": float(np.mean(np.abs(risk_generated["risk_peak_time_s"][both_peak] - risk_real["risk_peak_time_s"][both_peak]))) if both_peak.any() else float("nan"),
        "primary_risk_vehicle_identity_accuracy": float(np.mean(risk_generated["primary_risk_agent_index"][identity] == risk_real["primary_risk_agent_index"][identity])) if identity.any() else float("nan"),
        "primary_risk_vehicle_identity_comparable_sequences": int(identity.sum()),
    })
    target_values, generated_values = distribution_values(target_fields), distribution_values(generated_fields)
    executed_actions = np.concatenate([plan[:, response, :5] for response in range(plan.shape[1])], axis=1)
    executed_actions = executed_actions[:, :target.shape[1]]
    start_samples = samples[:, :, :25]
    start_target, start_valid = target[:, :25], valid[:, :25]
    roll_samples = samples[:, :, 25:]
    roll_target, roll_valid = target[:, 25:], valid[:, 25:]
    report = {
        "protocol": {
            "name": "held-out highD EVT-tail paired C0+B0 START and logged-ego ROLL reconstruction",
            "sequences": int(len(rows)),
            "generated_horizon_seconds": float(target.shape[1] * DT_S),
            "timeline": "START [0,1s] reconstructed from true C0+B0; ROLL (1s,5.96s] is 4.96 seconds under logged ego replay",
            "start_interface": "QR START encoder plus raw B0 consumed at initialization only; later plans receive only realised joint history and B0-derived state",
            "start_semantics": "segment-start behavior reconstruction; the highD natural-window anchor is not asserted to be the risk-event onset",
            "not_an_ads_free_roll_benchmark": True,
        },
        "b0_start_summary_reconstruction": b0,
        "start_first_second": {
            "trajectory": {key: value for key, value in trajectory_metrics(start_samples, start_target, start_valid).items() if key != "per_episode_min_fde_m"},
            "kinematic": kinematic_reconstruction_metrics(predicted[:, :25], target[:, :25], valid[:, :25]),
            "risk_episode_error": risk_error,
        },
        "start_roll_boundary_at_1s": _boundary_report(predicted, target, valid, plan),
        "roll_1s_to_5p96s": {
            "duration_seconds": float(roll_target.shape[1] * DT_S),
            "trajectory": {key: value for key, value in trajectory_metrics(roll_samples, roll_target, roll_valid).items() if key != "per_episode_min_fde_m"},
            "kinematic": kinematic_reconstruction_metrics(predicted[:, 25:], target[:, 25:], valid[:, 25:]),
        },
        "full_5p96_second_trajectory": {
            "trajectory": {key: value for key, value in trajectory_metrics(samples, target, valid).items() if key != "per_episode_min_fde_m"},
            "kinematic": kinematic_reconstruction_metrics(predicted, target, valid),
        },
        "motion_distribution": {
            key: {
                **_distance_with_moments(target_values[key], generated_values[key]),
                "kl_real_to_generated": histogram_kl_divergence(target_values[key], generated_values[key]),
            }
            for key in ("speed_mps", "acceleration_mps2", "jerk_mps3", "curvature_m_inv")
        },
        "risk_distribution": {
            key: _distance_with_moments(target_values[key], generated_values[key])
            for key in ("gap_m", "ttc_s", "drac_mps2", "relative_speed_mps")
        },
        "risk_tail_probability_error": {
            "ttc_lt_1s": _probability_error(target_values["ttc_s"], generated_values["ttc_s"], threshold=1.0, less_than=True),
            "drac_gt_3mps2": _probability_error(target_values["drac_mps2"], generated_values["drac_mps2"], threshold=3.0, less_than=False),
            "gap_lt_2m": _probability_error(target_values["gap_m"], generated_values["gap_m"], threshold=2.0, less_than=True),
            "collision_episode_rate": {
                "real": collision_metrics(target_fields)["collision_episode_rate"],
                "generated": collision_metrics(generated_fields)["collision_episode_rate"],
            },
        },
        "semantic_event_fidelity": _event_report(target_fields, generated_fields),
        "multi_vehicle_interaction": {
            "all_following_pairs": following_error_metrics(generated_fields, target_fields),
            "ego_background_pairs": _pairwise_error_report(generated_fields, target_fields, pair="ego_background"),
            "background_background_pairs": _pairwise_error_report(generated_fields, target_fields, pair="background_background"),
            "brake_response": social_response_metrics(generated_fields, target_fields),
            "background_action_correlation": _action_correlation_error(executed_actions, actions, valid),
            "high_risk_interaction_graph_degree": _high_risk_graph_report(target_fields, generated_fields),
            "collision": {
                "real": collision_metrics(target_fields),
                "generated": collision_metrics(generated_fields),
            },
        },
        "multimodality_and_calibration": _multimodal_report(samples, target, valid, ego),
    }
    details = {
        "b0_error": b0_error.astype(np.float32),
        "b0_valid": b0_take.astype(bool),
        "real_minimum_gap_m": risk_real["minimum_gap_m"].astype(np.float32),
        "generated_minimum_gap_m": risk_generated["minimum_gap_m"].astype(np.float32),
        "real_minimum_ttc_s": risk_real["minimum_ttc_s"].astype(np.float32),
        "generated_minimum_ttc_s": risk_generated["minimum_ttc_s"].astype(np.float32),
        "real_maximum_drac_mps2": risk_real["maximum_drac_mps2"].astype(np.float32),
        "generated_maximum_drac_mps2": risk_generated["maximum_drac_mps2"].astype(np.float32),
    }
    return report, details


def _probe_worlds(
    model,
    starts: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    donors: np.ndarray,
    *,
    device: torch.device,
    count: int,
) -> dict[str, Any]:
    """Run reproducibility, batch-independence and ADS causal-response probes."""
    count = min(int(count), len(donors))
    # ``load_flow_tail_starts`` is grouped by event structure.  Taking its
    # prefix can therefore select only same-front scenes and make the rear
    # braking intervention statistic undefined.  Stratify this small causal
    # probe across structure groups, while retaining the full formal Flow
    # sample for distribution measurements above.
    patterns = np.asarray(starts["mask_pattern"], np.int64)
    groups = [np.flatnonzero(patterns == pattern) for pattern in np.unique(patterns)]
    selected: list[int] = []
    round_index = 0
    while len(selected) < count:
        added = False
        for group in groups:
            if round_index < len(group) and len(selected) < count:
                selected.append(int(group[round_index]))
                added = True
        if not added:
            break
        round_index += 1
    chosen = np.asarray(selected, np.int64)
    raw_features = np.asarray(starts["features"], np.float32)
    has_rear = np.zeros(len(raw_features), bool)
    for slot_name in SLOT_NAMES:
        has_rear |= raw_features[:, slot_feature_index(slot_name, "rel_x_m")] < 0.0
    if len(chosen) and not has_rear[chosen].any() and has_rear.any():
        chosen[-1] = int(np.flatnonzero(has_rear)[0])
    features = raw_features[chosen]
    slots = np.asarray(starts["slot_mask"], bool)[chosen]
    donor_rows = np.asarray(donors, np.int64)[chosen]
    maps = np.asarray(cache["map_polylines"])[donor_rows]
    map_valid = np.asarray(cache["map_polyline_valid"])[donor_rows]
    edges = np.asarray(cache["lane_graph_edges"])[donor_rows]
    seeds = np.arange(91_000, 91_000 + count, dtype=np.int64)

    def run(
        feature: np.ndarray, slot: np.ndarray, polyline: np.ndarray, polyline_valid: np.ndarray,
        lane_edges: np.ndarray, world_seeds: np.ndarray, controls: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        environment = BatchedQRWorldModelEnvironment(model, device=device)
        with torch.no_grad():
            environment.reset_from_flow_batch(
                feature, slot, polyline, polyline_valid, lane_edges,
                deterministic=False, world_randomness=[int(value) for value in world_seeds],
            )
        states: list[np.ndarray] = []
        initial_plan = next_plan = None
        for tick in range(controls.shape[1]):
            current = environment.step(controls[:, tick])
            states.append(current.detach().cpu().numpy())
            if tick == 0:
                initial_plan = environment._active_plan.detach().cpu().numpy().copy()
            if tick == 5:
                next_plan = environment._active_plan.detach().cpu().numpy().copy()
        if initial_plan is None or next_plan is None:
            raise RuntimeError("ADS probe must execute six 25Hz ticks")
        return np.stack(states, axis=1), initial_plan, next_plan

    baseline_controls = np.zeros((count, 6, 2), np.float32)
    braking_controls = baseline_controls.copy()
    braking_controls[..., 0] = -4.0
    baseline_state, baseline_plan, baseline_replan = run(features, slots, maps, map_valid, edges, seeds, baseline_controls)
    replay_state, replay_plan, replay_replan = run(features, slots, maps, map_valid, edges, seeds, baseline_controls)
    braking_state, braking_plan, braking_replan = run(features, slots, maps, map_valid, edges, seeds, braking_controls)
    single_error = []
    for row in range(min(count, 8)):
        single_state, single_plan, single_replan = run(
            features[row : row + 1], slots[row : row + 1], maps[row : row + 1], map_valid[row : row + 1],
            edges[row : row + 1], seeds[row : row + 1], baseline_controls[row : row + 1],
        )
        single_error.extend([
            float(np.max(np.abs(single_state[0] - baseline_state[row]))),
            float(np.max(np.abs(single_plan[0] - baseline_plan[row]))),
            float(np.max(np.abs(single_replan[0] - baseline_replan[row]))),
        ])
    rear = np.zeros((count, 6), bool)
    for slot_index, slot_name in enumerate(SLOT_NAMES):
        rear[:, slot_index] = features[:, slot_feature_index(slot_name, "rel_x_m")] < 0.0
    acceleration_delta = braking_replan[:, :5, :, 0] - baseline_replan[:, :5, :, 0]
    rear_delta = acceleration_delta[rear[:, None, :].repeat(5, axis=1)]
    ego_speed_delta = (
        np.linalg.norm(braking_state[:, 4, 0, 2:4], axis=-1)
        - np.linalg.norm(baseline_state[:, 4, 0, 2:4], axis=-1)
    )
    return {
        "protocol": {
            "worlds": int(count),
            "controls": "baseline=[0,0], intervention=[-4 m/s^2,0 rad/s] for six 25Hz ticks; identical C0+B0, maps and explicit world_seed",
            "first_replan_after_intervention_s": 0.2,
        },
        "exact_replay": {
            "max_abs_state_error": float(np.max(np.abs(replay_state - baseline_state))),
            "max_abs_initial_plan_error": float(np.max(np.abs(replay_plan - baseline_plan))),
            "max_abs_replan_error": float(np.max(np.abs(replay_replan - baseline_replan))),
        },
        "batch_world_independence": {
            "worlds_compared_individually": int(min(count, 8)),
            "max_abs_error_batched_vs_single": float(max(single_error, default=0.0)),
        },
        "ads_nonanticipation": {
            "definition": "at the first tick QR-WM plans before either ADS action is applied; only the unexecuted action differs",
            "max_abs_background_plan_difference": float(np.max(np.abs(braking_plan - baseline_plan))),
        },
        "ads_state_response_at_next_5hz_boundary": {
            "ego_speed_change_after_0p2s_mps": _quantiles(ego_speed_delta),
            "mean_background_plan_action_change_l2": float(np.mean(np.linalg.norm(braking_replan[:, :5] - baseline_replan[:, :5], axis=-1))),
            "max_background_plan_action_change_l2": float(np.max(np.linalg.norm(braking_replan[:, :5] - baseline_replan[:, :5], axis=-1))),
            "rear_slot_mean_longitudinal_action_change_mps2": float(np.mean(rear_delta)) if len(rear_delta) else float("nan"),
            "rear_slot_fraction_more_braking": float(np.mean(rear_delta < 0.0)) if len(rear_delta) else float("nan"),
        },
    }


def _markdown_summary(report: dict[str, Any]) -> str:
    flow = report["flow_initial_distribution"]
    paired = report["paired_start_roll_reconstruction"]
    end_to_end = report["end_to_end_flow_qr_distribution"]
    start = paired["start_first_second"]["trajectory"]
    roll = paired["roll_1s_to_5p96s"]["trajectory"]
    return "\n".join([
        "# Flow×QR-WM 长尾重建审计结果",
        "",
        "本文件同时保存非配对的 Flow×QR 生成分布测试，以及使用真实 `C0+B0`、真实 ego 回放的成对 QR START/ROLL 重建测试。二者不能互相替代。",
        "",
        "## 当前 5.96 秒协议",
        "",
        "- `START [0, 1s]`：受 `B0` 条件约束的首秒重建。",
        "- `ROLL (1s, 5.96s]`：4.96 秒的 B0-free 闭环演化；150 个记录状态没有伪造的 S150。",
        "",
        "## Flow 初始分布",
        "",
        f"- 支持的真实 EVT-tail 起点：{flow['reference']['supported_real_sequences']}；Flow 样本：{flow['reference']['generated_flow_samples']}。",
        f"- 事件结构 TVD：{flow['event_structure']['total_variation_distance']:.6f}；JS：{flow['event_structure']['jensen_shannon_divergence_nats']:.6f}。",
        f"- `C0+B0` 联合 RBF-MMD：{flow['conditional_joint_c0_b0_distribution']['rbf_mmd']:.6f}。",
        "",
        "## 成对 QR 重建",
        "",
        f"- START ADE/FDE：{start['ADE_m']:.4f} m / {start['FDE_m']:.4f} m。",
        f"- ROLL（4.96 秒）ADE/FDE：{roll['ADE_m']:.4f} m / {roll['FDE_m']:.4f} m。",
        f"- 首秒 B0 六维平均 MAE：{paired['b0_start_summary_reconstruction']['mean_absolute_error_all_dimensions']:.4f}。",
        f"- 多模态 minADE@K：{paired['multimodality_and_calibration']['minADE_at_K_m']:.4f} m。",
        "",
        "## 端到端 Flow×QR 风险分布",
        "",
        f"- 交通特征 Fréchet：{end_to_end['traffic_feature_frechet_distance']:.4f}；RBF-MMD：{end_to_end['mmd_rbf']:.6f}。",
        f"- 合成 collision episode rate：{end_to_end['collision_episode_rate']:.2%}。",
        "",
        "完整逐维、风险尾部、事件、多车交互、多模态、可重放性和 ADS 因果响应数值见 `reconstruction_validation_audit.json`；逐序列风险和 B0 误差见 `reconstruction_validation_details.npz`。",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT / "results/highd_world_model/qr_world_model/checkpoints/best_qr_world_model.pt"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/highd_world_model/long_tail_reproduction"))
    parser.add_argument(
        "--sequence-cache-dir",
        default=str(ROOT / "results/highd_world_model/training_data/qr_sequence_cache"),
        help="Canonical raw-150-state QR cache.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--flow-start-batch-size", type=int, default=96,
        help="Independent Flow starts evaluated together in the end-to-end distribution study.",
    )
    parser.add_argument("--multimodal-samples", type=int, default=8)
    parser.add_argument("--ads-probe-worlds", type=int, default=32)
    parser.add_argument("--max-paired-sequences", type=int, default=0, help="development-only deterministic cap; 0 evaluates all held-out EVT-tail sequences")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if args.multimodal_samples < 2:
        raise ValueError("--multimodal-samples must be at least two")
    setup_logging(args.log_level)
    set_seed(args.seed)
    device = select_device("auto")
    checkpoint, output_dir = Path(args.checkpoint).resolve(), ensure_dir(Path(args.output_dir).resolve())
    cache_owner = Path(args.sequence_cache_dir).resolve()
    arrays, manifest = load_sequential_dataset(cache_owner)
    if not is_canonical_qr_manifest(manifest):
        raise RuntimeError("The reconstruction audit requires the QR raw-150-state START+ROLL cache.")
    # Match the proven composition order: Flow sampling first, then release
    # its temporary accelerator allocations before QR-WM is loaded.
    starts, cache, donors = load_flow_tail_starts(
        ROOT, device=device, replay_scope="all_evt_tail", sequence_cache_owner=cache_owner,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = load_qr_checkpoint(checkpoint, device=device)
    require_canonical_qr_checkpoint(model)
    # The complete study uses the exact same sampled Flow starts and matched
    # replay donors for its unpaired distribution result and paired audit.
    end_to_end_source = evaluate_flow_composition(
        checkpoint=checkpoint,
        output_dir=output_dir,
        flow_start_batch_size=args.flow_start_batch_size,
        sequence_cache_owner=cache_owner,
        model=model,
        prepared_starts=starts,
        prepared_cache=cache,
        prepared_donors=donors,
        device=device,
    )
    figures = [
        _plot_tail_interaction_distribution(output_dir),
        _plot_tail_sampling_and_runtime(output_dir),
    ]
    flow_schema = FrozenLegacyFlowSchema.load(ROOT / "results/highd_tail_flow/dataset_schema.json")
    arrays.update(ensure_frozen_flow_behavior_anchor_cache(
        cache_owner, arrays, manifest, flow_schema,
    ))
    rows = np.flatnonzero(
        (np.asarray(arrays["split_index"]) == SPLIT_TO_INDEX["test"]) & np.asarray(arrays["is_evt_tail"], bool)
    )
    if args.max_paired_sequences:
        rows = np.random.default_rng(args.seed).choice(rows, size=min(len(rows), int(args.max_paired_sequences)), replace=False)
    flow_report = _initial_flow_report(starts, cache, donors, device=device)
    paired, details = _paired_reconstruction_report(
        model, arrays, rows, device=device, batch_size=args.batch_size,
        multimodal_samples=args.multimodal_samples, seed=args.seed,
    )
    probes = _probe_worlds(
        model, starts, cache, donors, device=device, count=args.ads_probe_worlds,
    )
    end_to_end_path = output_dir / "flow_composition_evaluation.json"
    end_to_end = {
        "source": {"path": str(end_to_end_path), "sha256": file_sha256(end_to_end_path)},
        "traffic_feature_frechet_distance": end_to_end_source["closed_loop_distribution"]["traffic_feature_frechet_distance"],
        "mmd_rbf": end_to_end_source["closed_loop_distribution"]["mmd_rbf"],
        "collision_episode_rate": end_to_end_source["closed_loop_distribution"]["physical_validity"]["collision_episode_rate"],
        "report_includes_extended_motion_risk_and_event_metrics": "motion_variable_distribution" in end_to_end_source["closed_loop_distribution"],
    }
    report = {
        "study": "Flow initial distribution plus QR START/ROLL held-out reconstruction audit",
        "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
        "sequence_cache": manifest,
        "protocol": {
            "generated_horizon_seconds": 5.96,
            "timeline": "START [0,1s] + ROLL (1s,5.96s] = 1 second conditional reconstruction plus 4.96 seconds subsequent roll",
            "raw_window_limit": "150 observed 25 Hz state points provide 149 transitions; no S150 is fabricated.",
            "paired_vs_unpaired": "START/ROLL ADE/FDE use paired logged C0+B0 and ego replay. Flow×QR synthetic worlds use distribution metrics only because Flow samples have no paired target future.",
            "start_semantics": "segment-start behavior reconstruction, not a claim that the anchor is a risk-event onset.",
        },
        "flow_initial_distribution": flow_report,
        "paired_start_roll_reconstruction": paired,
        "reproducibility_and_ads_causal_response": probes,
        "end_to_end_flow_qr_distribution": end_to_end,
        "coverage": {
            "flow_initial_event_and_c0_b0_distribution": "executed",
            "paired_start_b0_trajectory_boundary_and_roll": "executed",
            "risk_tail_semantic_events_and_multi_vehicle_interaction": "executed on paired held-out EVT-tail reconstruction; end-to-end distribution values are in flow_composition_evaluation.json",
            "multimodality_replay_batch_independence_ads_response": "executed",
            "not_claimed": "A full five-second free ADS ROLL is not claimed: it needs 151 observed state points (or a seven-second raw window).",
        },
    }
    details_path = output_dir / "reconstruction_validation_details.npz"
    np.savez_compressed(details_path, **details)
    report["details"] = {"path": str(details_path), "sha256": file_sha256(details_path), "fields": sorted(details)}
    report_path = output_dir / "reconstruction_validation_audit.json"
    save_json(report, report_path)
    summary_path = output_dir / "reconstruction_validation_summary.md"
    summary_path.write_text(_markdown_summary(report), encoding="utf-8")
    manifest_path = output_dir / "study_manifest.json"
    study_manifest = {"study": report["study"], "artifacts": {}}
    study_manifest.setdefault("artifacts", {}).update({
        "flow_composition_evaluation": end_to_end_path.name,
        "flow_start_audit": "flow_start_audit.npz",
        "figures": [str(path.relative_to(output_dir)) for path in figures],
        "reconstruction_validation_audit": report_path.name,
        "reconstruction_validation_details": details_path.name,
        "reconstruction_validation_summary": summary_path.name,
    })
    save_json(study_manifest, manifest_path)
    print(report_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Formal 32-branch reconstruction study on held-out highD EVT-tail events.

Every world model receives the same logged history, map, B0 condition, and
external ego replay.  The study measures conditional reconstruction, not
unconditional scenario generation.  CAT-TopK remains an explicitly
information-asymmetric archived reference.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.cat_topk.compat_rollout import (
    legacy_sequence_rows,
    rollout_legacy_chunks,
)
from world_model.src.cat_topk.model import load_checkpoint as load_cat_topk_checkpoint
from world_model.src.core.data import dataset_dir_from_config, load_world_model_dataset
from world_model.src.core.initial_behavior_anchor import FrozenLegacyFlowSchema
from world_model.src.core.long_tail_metrics import (
    collision_metrics,
    distribution_values,
    empirical_distance,
    event_masks,
    feature_distribution_distance,
    following_error_metrics,
    kinematic_reconstruction_metrics,
    social_response_metrics,
    speed_kl_divergence,
    traffic_fields,
    trajectory_metrics,
)
from world_model.src.core.sequential_dataset import (
    FLOW_ANCHOR_ARRAYS,
    ensure_frozen_flow_behavior_anchor_cache,
    load_sequential_dataset,
    sequence_cache_owner_dir,
)
from world_model.src.core.utils import ensure_dir, file_sha256, load_yaml, save_json, select_device, set_seed
from world_model.src.firm.train import load_firm_checkpoint
from world_model.src.qr.train import load_qr_checkpoint
from world_model.src.ramp.train import load_ramp_checkpoint
from world_model.src.semi_markov.train import (
    FIELDS,
    OPTIONAL_FIELDS,
    _to_batch,
    load_semi_markov_checkpoint,
)


NUM_SAMPLES = 32
DT_S = 0.04
MODEL_SPECS = (
    ("ramp_world_model", "RAMP-WM", "#377eb8"),
    ("firm_world_model", "FIRM-WM", "#984ea3"),
    ("semi_markov_world_model", "Semi-Markov WM", "#ff7f00"),
    ("cat_topk_world_model", "CAT-TopK", "#4daf4a"),
    ("qr_world_model", "QR-WM", "#e41a1c"),
)
PLAYBACK_EVENTS = ("high_risk_following", "hard_braking", "close_interaction")


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _batch(arrays: dict[str, np.ndarray], rows: np.ndarray, device):
    """Build the common sequential-cache tensor contract without a DataLoader."""
    import torch

    names = tuple([
        *FIELDS,
        *[name for name in OPTIONAL_FIELDS if name in arrays],
        *[name for name in FLOW_ANCHOR_ARRAYS if name in arrays],
    ])
    values = tuple(torch.from_numpy(np.asarray(arrays[name][rows]).copy()) for name in names)
    return _to_batch(values, names, device)


def _repeat_batch(batch: dict[str, Any], copies: int) -> dict[str, Any]:
    return {name: value.repeat_interleave(int(copies), dim=0) for name, value in batch.items()}


def _sample_model(model, batch: dict[str, Any], *, seed: int, branch_batch_size: int) -> np.ndarray:
    """Return [K, B, 125, 6, 6], with branch zero fixed to deterministic ROLL."""
    deterministic = model.rollout_roll_mode(batch, seed=seed, deterministic=True)[
        "predicted_states"
    ][:, :, 1:].cpu().numpy()
    branches: list[np.ndarray] = []
    for start in range(0, NUM_SAMPLES, branch_batch_size):
        count = min(branch_batch_size, NUM_SAMPLES - start)
        sampled = model.rollout_roll_mode(
            _repeat_batch(batch, count), seed=seed + 10_000 + start, deterministic=False
        )["predicted_states"][:, :, 1:].cpu().numpy()
        branches.append(sampled.reshape(len(deterministic), count, *sampled.shape[1:]).transpose(1, 0, 2, 3, 4))
    result = np.concatenate(branches, axis=0)
    result[0] = deterministic
    return result


def _sample_qr(model, batch: dict[str, Any], *, seed: int, branch_batch_size: int) -> np.ndarray:
    """Sample QR-WM with the same one deterministic plus 31 stochastic branches."""
    import torch

    deterministic = model.rollout_reconstruction(batch, deterministic=True)["predicted_states"][:, :, 1:].cpu().numpy()
    branches: list[np.ndarray] = []
    for start in range(0, NUM_SAMPLES, branch_batch_size):
        count = min(branch_batch_size, NUM_SAMPLES - start)
        torch.manual_seed(seed + 10_000 + start)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 10_000 + start)
        sampled = model.rollout_reconstruction(_repeat_batch(batch, count), deterministic=False)["predicted_states"][:, :, 1:].cpu().numpy()
        branches.append(sampled.reshape(len(deterministic), count, *sampled.shape[1:]).transpose(1, 0, 2, 3, 4))
    result = np.concatenate(branches, axis=0)
    result[0] = deterministic
    return result


def _sample_cat_topk(
    model,
    arrays: dict[str, np.ndarray],
    schema: dict[str, Any],
    sequence_rows: np.ndarray,
    *,
    device,
    seed: int,
    branch_batch_size: int,
) -> np.ndarray:
    deterministic, _ = rollout_legacy_chunks(
        model, arrays, schema, sequence_rows, chunks=5, device=device, seed=seed, deterministic=True
    )
    branches: list[np.ndarray] = []
    for start in range(0, NUM_SAMPLES, branch_batch_size):
        count = min(branch_batch_size, NUM_SAMPLES - start)
        sampled, _ = rollout_legacy_chunks(
            model, arrays, schema, np.repeat(sequence_rows, count, axis=0), chunks=5,
            device=device, seed=seed + 10_000 + start, deterministic=False,
        )
        branches.append(sampled.reshape(len(deterministic), count, *sampled.shape[1:]).transpose(1, 0, 2, 3, 4))
    result = np.concatenate(branches, axis=0)
    result[0] = deterministic
    return result


def _pooled(samples: np.ndarray, ego: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count, episodes = samples.shape[:2]
    return (
        samples.reshape(count * episodes, *samples.shape[2:]),
        np.tile(ego, (count, 1, 1)),
        np.tile(valid, (count, 1, 1)),
    )


def _distribution_report(real: dict[str, np.ndarray], generated: dict[str, np.ndarray]) -> dict[str, Any]:
    keys = ("ttc_s", "drac_mps2", "gap_m", "relative_speed_mps")
    return {key: empirical_distance(real[key], generated[key]) for key in keys}


def _event_report(samples: np.ndarray, target: np.ndarray, ego: np.ndarray, valid: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, mask in masks.items():
        if not np.any(mask):
            result[name] = {"available": False, "num_sequences": 0}
            continue
        trajectory = trajectory_metrics(samples[:, mask], target[mask], valid[mask])
        deterministic = traffic_fields(samples[0, mask], ego[mask], valid[mask])
        result[name] = {
            "available": True,
            "num_sequences": int(mask.sum()),
            "trajectory": {key: value for key, value in trajectory.items() if key != "per_episode_min_fde_m"},
            "collision": collision_metrics(deterministic),
        }
    return result


def _model_report(samples: np.ndarray, target: np.ndarray, ego: np.ndarray, valid: np.ndarray, masks: dict[str, np.ndarray], *, seed: int) -> dict[str, Any]:
    trajectory = trajectory_metrics(samples, target, valid)
    deterministic = traffic_fields(samples[0], ego, valid)
    reference = traffic_fields(target, ego, valid)
    pooled_states, pooled_ego, pooled_valid = _pooled(samples, ego, valid)
    pooled = traffic_fields(pooled_states, pooled_ego, pooled_valid)
    real_values, generated_values = distribution_values(reference), distribution_values(pooled)
    return {
        "trajectory_reconstruction": {key: value for key, value in trajectory.items() if key != "per_episode_min_fde_m"},
        "kinematic_realism": {
            **kinematic_reconstruction_metrics(samples[0], target, valid),
            "speed_distribution_kl_real_to_generated": speed_kl_divergence(real_values["speed_mps"], generated_values["speed_mps"]),
        },
        "interaction_realism": {
            "deterministic_following_error": following_error_metrics(deterministic, reference),
            "real_collision": collision_metrics(reference),
            "generated_collision": collision_metrics(pooled),
            "social_response": social_response_metrics(deterministic, reference),
        },
        "distribution_realism": {
            "risk_variables": _distribution_report(real_values, generated_values),
            **feature_distribution_distance(
                pooled_states, pooled_ego, pooled_valid, target, ego, valid, seed=seed
            ),
        },
        "events": _event_report(samples, target, ego, valid, masks),
    }


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.22, "figure.dpi": 140, "savefig.dpi": 300,
    })
    return plt


def _save(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    _style().close(figure)


def _valid_series(values: np.ndarray, valid: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(values[:, frame])[np.asarray(valid[:, frame], bool)] for frame in range(values.shape[1])]


def _quantile_band(axis, values: np.ndarray, valid: np.ndarray, time: np.ndarray, *, color: str, label: str, alpha: float = 0.18) -> None:
    rows = _valid_series(values, valid)
    median = np.asarray([np.median(row) if len(row) else np.nan for row in rows])
    low, high = (
        np.asarray([np.quantile(row, level) if len(row) else np.nan for row in rows])
        for level in (0.05, 0.95)
    )
    middle_low, middle_high = (
        np.asarray([np.quantile(row, level) if len(row) else np.nan for row in rows])
        for level in (0.25, 0.75)
    )
    axis.fill_between(time, low, high, color=color, alpha=alpha * 0.55)
    axis.fill_between(time, middle_low, middle_high, color=color, alpha=alpha)
    axis.plot(time, median, color=color, lw=2.0, label=label)


def _plot_examples(path: Path, samples: np.ndarray, target: np.ndarray, ego: np.ndarray, valid: np.ndarray, selected: dict[str, int], *, color: str, label: str) -> None:
    plt = _style()
    figure, axes = plt.subplots(1, len(PLAYBACK_EVENTS), figsize=(15.0, 4.4), sharey=True)
    for axis, event in zip(axes, PLAYBACK_EVENTS):
        row = selected[event]
        for branch in samples[:, row]:
            for agent in range(branch.shape[1]):
                take = valid[row, :, agent]
                axis.plot(branch[take, agent, 0], branch[take, agent, 1], color=color, alpha=0.055, lw=0.7)
        for agent in range(target.shape[2]):
            take = valid[row, :, agent]
            axis.plot(target[row, take, agent, 0], target[row, take, agent, 1], color="#222222", lw=1.4)
            axis.plot(samples[0, row, take, agent, 0], samples[0, row, take, agent, 1], color=color, lw=1.4)
        axis.plot(ego[row, :, 0], ego[row, :, 1], "--", color="#777777", lw=1.2, label="ego replay")
        axis.set_title(event.replace("_", " "))
        axis.set_xlabel("longitudinal position (m)")
    axes[0].set_ylabel("lateral position (m)")
    axes[0].plot([], [], color="#222222", label="highD truth")
    axes[0].plot([], [], color=color, label=f"{label} deterministic")
    axes[0].plot([], [], color=color, alpha=0.28, lw=1.2, label="32 stochastic branches")
    axes[0].legend(fontsize=8, loc="best")
    _save(figure, path)


def _plot_trajectory_family(path: Path, samples: np.ndarray, target: np.ndarray, valid: np.ndarray, *, color: str, label: str) -> None:
    plt = _style()
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    time = np.arange(target.shape[1]) * DT_S
    for axis, coordinate, title in zip(axes, (0, 1), ("Longitudinal displacement", "Lateral displacement")):
        real = target[..., coordinate] - target[:, :1, :, coordinate]
        generated = samples[..., coordinate] - samples[:, :, :1, :, coordinate]
        _quantile_band(axis, real, valid, time, color="#222222", label="highD EVT tail", alpha=0.14)
        flat_values = generated.reshape(-1, generated.shape[2], generated.shape[3])
        flat_valid = np.tile(valid, (samples.shape[0], 1, 1))
        _quantile_band(axis, flat_values, flat_valid, time, color=color, label=label)
        axis.set_title(title)
        axis.set_xlabel("t from anchor (s)")
        axis.set_ylabel("displacement (m)")
        axis.legend(fontsize=8)
    _save(figure, path)


def _hist(axis, real: np.ndarray, generated: np.ndarray, *, title: str, color: str, bins: int = 45) -> None:
    real, generated = np.asarray(real), np.asarray(generated)
    combined = np.concatenate((real[np.isfinite(real)], generated[np.isfinite(generated)]))
    if len(combined):
        edges = np.linspace(np.quantile(combined, 0.005), np.quantile(combined, 0.995), bins)
        axis.hist(real, bins=edges, density=True, color="#333333", alpha=0.36, label="highD EVT tail")
        axis.hist(generated, bins=edges, density=True, color=color, alpha=0.46, label="generated")
    axis.set_title(title)
    axis.set_ylabel("density")


def _plot_kinematic(path: Path, samples: np.ndarray, target: np.ndarray, ego: np.ndarray, valid: np.ndarray, report: dict[str, Any], *, color: str, label: str) -> None:
    plt = _style()
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.3))
    real = distribution_values(traffic_fields(target, ego, valid))
    generated_states, generated_ego, generated_valid = _pooled(samples, ego, valid)
    generated = distribution_values(traffic_fields(generated_states, generated_ego, generated_valid))
    for axis, key, title in zip(axes.flat, ("speed_mps", "acceleration_mps2", "jerk_mps3", "curvature_m_inv"), ("Speed", "Acceleration magnitude", "Jerk magnitude", "Curvature")):
        _hist(axis, real[key], generated[key], title=title, color=color)
        axis.set_xlabel({"speed_mps": "m/s", "acceleration_mps2": "m/s²", "jerk_mps3": "m/s³", "curvature_m_inv": "m⁻¹"}[key])
    metric = report["kinematic_realism"]
    figure.suptitle(
        f"{label}: acceleration MAE={metric['acceleration_vector_mae_mps2']:.3f}, "
        f"jerk MAE={metric['jerk_vector_mae_mps3']:.3f}, curvature MAE={metric['curvature_mae_m_inv']:.4f}, "
        f"speed KL={metric['speed_distribution_kl_real_to_generated']:.3f}", fontsize=10
    )
    axes[0, 0].legend(fontsize=8)
    _save(figure, path)


def _plot_interaction(path: Path, samples: np.ndarray, target: np.ndarray, ego: np.ndarray, valid: np.ndarray, report: dict[str, Any], *, color: str, label: str) -> None:
    plt = _style()
    figure, axes = plt.subplots(2, 3, figsize=(14.0, 7.2))
    reference = traffic_fields(target, ego, valid)
    generated_states, generated_ego, generated_valid = _pooled(samples, ego, valid)
    generated = traffic_fields(generated_states, generated_ego, generated_valid)
    real_values, generated_values = distribution_values(reference), distribution_values(generated)
    for axis, key, title in zip(axes.flat[:4], ("ttc_s", "gap_m", "drac_mps2", "relative_speed_mps"), ("TTC", "Following gap", "DRAC", "Relative closing speed")):
        _hist(axis, real_values[key], generated_values[key], title=title, color=color)
    social = report["interaction_realism"]["social_response"]
    bins = np.asarray(social["brake_response_real"]["bin_edges_mps2"])
    centres = (bins[:-1] + bins[1:]) / 2.0
    axes[1, 1].plot(centres, social["brake_response_real"]["mean_follower_acceleration_mps2"], "o-", color="#222222", label="highD")
    axes[1, 1].plot(centres, social["brake_response_generated"]["mean_follower_acceleration_mps2"], "o-", color=color, label=label)
    axes[1, 1].set_title("Leader brake → follower response")
    axes[1, 1].set_xlabel("leader acceleration (m/s²)")
    axes[1, 1].set_ylabel("follower acceleration (m/s²)")
    axes[1, 1].legend(fontsize=8)
    real_collision = report["interaction_realism"]["real_collision"]
    generated_collision = report["interaction_realism"]["generated_collision"]
    axes[1, 2].bar((0, 1), [real_collision["collision_episode_rate"], generated_collision["collision_episode_rate"]], color=("#333333", color))
    axes[1, 2].set_xticks((0, 1), ("highD", label), rotation=12)
    axes[1, 2].set_title("Collision episode rate")
    axes[1, 2].set_ylabel("rate")
    axes[0, 0].legend(fontsize=8)
    _save(figure, path)


def _plot_distribution(path: Path, report: dict[str, Any], *, color: str, label: str) -> None:
    plt = _style()
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.4))
    risk = report["distribution_realism"]["risk_variables"]
    labels = ("TTC", "DRAC", "gap", "rel. speed")
    axes[0].bar(np.arange(4), [risk[key]["wasserstein_1"] for key in ("ttc_s", "drac_mps2", "gap_m", "relative_speed_mps")], color=color)
    axes[0].set_xticks(np.arange(4), labels, rotation=18)
    axes[0].set_title("Wasserstein-1")
    axes[1].bar(np.arange(4), [risk[key]["ks"] for key in ("ttc_s", "drac_mps2", "gap_m", "relative_speed_mps")], color=color)
    axes[1].set_xticks(np.arange(4), labels, rotation=18)
    axes[1].set_title("KS distance")
    summary = report["distribution_realism"]
    axes[2].bar((0, 1, 2), [summary["traffic_feature_frechet_distance"], summary["mmd_rbf"], report["kinematic_realism"]["speed_distribution_kl_real_to_generated"]], color=color)
    axes[2].set_xticks((0, 1, 2), ("Feature\nFréchet", "RBF-MMD", "Speed KL"))
    axes[2].set_title(f"{label} multivariate realism")
    _save(figure, path)


def _plot_events(path: Path, report: dict[str, Any], *, color: str, label: str) -> None:
    plt = _style()
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 7.2))
    events = [name for name, value in report["events"].items() if value.get("available")]
    short = [name.replace("_", "\n") for name in events]
    definitions = (
        ("ADE_m", "ADE (m)"), ("FDE_m", "FDE (m)"),
        ("minADE_at_32_m", "minADE@32 (m)"), ("minFDE_at_32_m", "minFDE@32 (m)"),
    )
    for axis, (key, title) in zip(axes.flat[:4], definitions):
        axis.bar(np.arange(len(events)), [report["events"][event]["trajectory"][key] for event in events], color=color)
        axis.set_xticks(np.arange(len(events)), short, fontsize=7)
        axis.set_title(title)
    axes[1, 1].bar(np.arange(len(events)), [report["events"][event]["collision"]["collision_episode_rate"] for event in events], color=color)
    axes[1, 1].set_xticks(np.arange(len(events)), short, fontsize=7)
    axes[1, 1].set_title("Collision episode rate")
    axes[1, 2].axis("off")
    axes[1, 2].text(0.05, 0.85, f"{label}\n32 branches per event\nCAT-TopK is information-asymmetric", va="top", fontsize=10)
    _save(figure, path)


def _gif(path: Path, samples: np.ndarray, target: np.ndarray, ego: np.ndarray, valid: np.ndarray, row: int, *, color: str, label: str) -> None:
    """Compact replay: accumulated highD/deterministic paths plus faint branches."""
    from PIL import Image

    plt = _style()
    frames: list[Image.Image] = []
    x_values = np.concatenate((target[row, ..., 0].reshape(-1), ego[row, :, 0]))
    y_values = np.concatenate((target[row, ..., 1].reshape(-1), ego[row, :, 1]))
    x_min, x_max = np.nanpercentile(x_values, (1, 99))
    y_min, y_max = np.nanpercentile(y_values, (1, 99))
    for frame in range(0, target.shape[1], 5):
        figure, axis = plt.subplots(figsize=(7.2, 4.8))
        for branch in samples[:, row]:
            for agent in range(branch.shape[1]):
                take = valid[row, :, agent]
                axis.plot(branch[take, agent, 0], branch[take, agent, 1], color=color, alpha=0.035, lw=0.65)
        for agent in range(target.shape[2]):
            take = valid[row, : frame + 1, agent]
            axis.plot(target[row, : frame + 1, agent, 0][take], target[row, : frame + 1, agent, 1][take], color="#222222", lw=1.4)
            axis.plot(samples[0, row, : frame + 1, agent, 0][take], samples[0, row, : frame + 1, agent, 1][take], color=color, lw=1.5)
        axis.plot(ego[row, : frame + 1, 0], ego[row, : frame + 1, 1], "--", color="#777777", lw=1.2)
        axis.set(xlim=(x_min - 5.0, x_max + 5.0), ylim=(y_min - 3.0, y_max + 3.0), xlabel="longitudinal position (m)", ylabel="lateral position (m)", title=f"{label} long-tail replay, t={frame * DT_S:.1f}s")
        figure.tight_layout()
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=110)
        plt.close(figure)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("P"))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=180, loop=0, optimize=False)


def _selected_events(target: np.ndarray, ego: np.ndarray, valid: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, int]:
    fields = traffic_fields(target, ego, valid)
    values = {
        "high_risk_following": fields["ttc_s"],
        "hard_braking": target[..., 4],
        "close_interaction": fields["gap_m"],
    }
    selected = {}
    for event in PLAYBACK_EVENTS:
        candidates = np.flatnonzero(masks[event])
        if not len(candidates):
            selected[event] = 0
            continue
        score = []
        for row in candidates:
            if event == "hard_braking":
                score.append(float(np.nanmin(np.where(valid[row], values[event][row], np.inf))))
            else:
                score.append(float(np.nanmin(values[event][row])))
        selected[event] = int(candidates[int(np.nanargmin(score))])
    return selected


def _plot_overview(path: Path, reports: dict[str, dict[str, Any]]) -> None:
    plt = _style()
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.3))
    names = [name for name, _label, _color in MODEL_SPECS]
    labels = [label for _name, label, _color in MODEL_SPECS]
    colors = [color for _name, _label, color in MODEL_SPECS]
    axes[0].bar(np.arange(len(names)), [reports[name]["trajectory_reconstruction"]["FDE_m"] for name in names], color=colors)
    axes[0].set_xticks(np.arange(len(names)), labels, rotation=18)
    axes[0].set_title("Deterministic 5 s FDE")
    axes[1].bar(np.arange(len(names)), [reports[name]["trajectory_reconstruction"]["minFDE_at_32_m"] for name in names], color=colors)
    axes[1].set_xticks(np.arange(len(names)), labels, rotation=18)
    axes[1].set_title("minFDE@32")
    axes[2].bar(np.arange(len(names)), [reports[name]["distribution_realism"]["traffic_feature_frechet_distance"] for name in names], color=colors)
    axes[2].set_xticks(np.arange(len(names)), labels, rotation=18)
    axes[2].set_title("Traffic-feature Fréchet")
    _save(figure, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ramp-config", default=str(ROOT / "world_model/scripts/configs/highd_ramp_world_model.yaml"))
    parser.add_argument("--firm-config", default=str(ROOT / "world_model/scripts/configs/highd_firm_world_model.yaml"))
    parser.add_argument("--semi-config", default=str(ROOT / "world_model/scripts/configs/highd_semi_markov_world_model.yaml"))
    parser.add_argument("--catk-config", default=str(ROOT / "world_model/scripts/configs/highd_cat_topk_world_model.yaml"))
    parser.add_argument("--qr-config", default=str(ROOT / "world_model/scripts/configs/highd_qr_world_model.yaml"))
    parser.add_argument("--ramp-checkpoint", default=str(ROOT / "results/highd_world_model/ramp_world_model/checkpoints/best_ramp_world_model.pt"))
    parser.add_argument("--firm-checkpoint", default=str(ROOT / "results/highd_world_model/firm_world_model/checkpoints/best_firm_world_model.pt"))
    parser.add_argument("--semi-checkpoint", default=str(ROOT / "results/highd_world_model/semi_markov_world_model/checkpoints/best_semi_markov_relational.pt"))
    parser.add_argument("--catk-checkpoint", default=str(ROOT / "results/highd_world_model/cat_topk_world_model/checkpoints/best_world_model.pt"))
    parser.add_argument("--qr-checkpoint", default=str(ROOT / "results/highd_world_model/qr_world_model/checkpoints/best_qr_world_model.pt"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/highd_world_model/long_tail_reproduction"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--branch-batch-size", type=int, default=4)
    parser.add_argument("--max-sequences", type=int, default=0, help="Development only; requires a non-default output directory.")
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    official_output = (ROOT / "results/highd_world_model/long_tail_reproduction").resolve()
    if args.max_sequences and output == official_output:
        raise ValueError("smoke runs require an explicit non-formal --output-dir")
    if args.branch_batch_size < 1:
        raise ValueError("--branch-batch-size must be positive")

    config_paths = {name: Path(path).resolve() for name, path in {
        "ramp_world_model": args.ramp_config, "firm_world_model": args.firm_config,
        "semi_markov_world_model": args.semi_config, "cat_topk_world_model": args.catk_config,
        "qr_world_model": args.qr_config,
    }.items()}
    configs = {name: load_yaml(path) for name, path in config_paths.items()}
    ramp_config = configs["ramp_world_model"]
    device = select_device(str(ramp_config.get("evaluation", {}).get("device", "auto")))
    set_seed(args.seed)
    cache_owner = sequence_cache_owner_dir(ramp_config, config_dir=config_paths["ramp_world_model"].parent)
    arrays, manifest = load_sequential_dataset(cache_owner)
    schema_raw = ramp_config["paths"]["flow_schema"]
    schema_path = Path(schema_raw)
    schema = FrozenLegacyFlowSchema.load(schema_path if schema_path.is_absolute() else config_paths["ramp_world_model"].parent / schema_path)
    arrays.update(ensure_frozen_flow_behavior_anchor_cache(cache_owner, arrays, manifest, schema))
    rows = np.flatnonzero((np.asarray(arrays["split_index"]) == 2) & np.asarray(arrays["is_evt_tail"], bool))
    if args.max_sequences:
        rows = rows[: args.max_sequences]
    if not len(rows):
        raise RuntimeError("the held-out sequence cache contains no EVT-tail events")
    target = np.asarray(arrays["agent_states"][rows, 25:150, 1:], np.float32)
    ego = np.asarray(arrays["agent_states"][rows, 25:150, 0], np.float32)
    valid = np.asarray(arrays["agent_valid"][rows, 25:150, 1:], bool)
    masks = event_masks(target, ego, valid)
    selected = _selected_events(target, ego, valid, masks)

    cat_arrays, cat_schema = load_world_model_dataset(dataset_dir_from_config(configs["cat_topk_world_model"], config_paths["cat_topk_world_model"].parent))
    cat_rows = legacy_sequence_rows(cat_arrays, np.asarray(arrays["sequence_id"])[rows], horizon_steps=int(cat_schema["horizon_steps"]), chunks=5)
    ramp = load_ramp_checkpoint(Path(args.ramp_checkpoint).resolve(), device=device)
    firm = load_firm_checkpoint(Path(args.firm_checkpoint).resolve(), device=device)
    semi = load_semi_markov_checkpoint(Path(args.semi_checkpoint).resolve(), device=device)
    semi.set_frozen_flow_schema(schema)
    cat, _ = load_cat_topk_checkpoint(str(Path(args.catk_checkpoint).resolve()), device)
    qr = load_qr_checkpoint(Path(args.qr_checkpoint).resolve(), device=device)

    output = ensure_dir(output)
    sample_parts = {name: [] for name, _label, _color in MODEL_SPECS}
    import torch
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            stop = min(start + args.batch_size, len(rows))
            print(f"Long-tail rollout {start + 1}-{stop}/{len(rows)} with K={NUM_SAMPLES}", flush=True)
            batch = _batch(arrays, rows[start:stop], device)
            sample_parts["ramp_world_model"].append(_sample_model(ramp, batch, seed=args.seed + 100_000 + start, branch_batch_size=args.branch_batch_size))
            sample_parts["firm_world_model"].append(_sample_model(firm, batch, seed=args.seed + 200_000 + start, branch_batch_size=args.branch_batch_size))
            sample_parts["semi_markov_world_model"].append(_sample_model(semi, batch, seed=args.seed + 300_000 + start, branch_batch_size=args.branch_batch_size))
            sample_parts["cat_topk_world_model"].append(_sample_cat_topk(cat, cat_arrays, cat_schema, cat_rows[start:stop], device=device, seed=args.seed + 400_000 + start, branch_batch_size=args.branch_batch_size))
            sample_parts["qr_world_model"].append(_sample_qr(qr, batch, seed=args.seed + 500_000 + start, branch_batch_size=args.branch_batch_size))
    samples = {name: np.concatenate(parts, axis=1) for name, parts in sample_parts.items()}
    checkpoints = {
        "ramp_world_model": Path(args.ramp_checkpoint).resolve(),
        "firm_world_model": Path(args.firm_checkpoint).resolve(),
        "semi_markov_world_model": Path(args.semi_checkpoint).resolve(),
        "cat_topk_world_model": Path(args.catk_checkpoint).resolve(),
        "qr_world_model": Path(args.qr_checkpoint).resolve(),
    }
    reports = {
        name: _model_report(samples[name], target, ego, valid, masks, seed=args.seed + index)
        for index, (name, _label, _color) in enumerate(MODEL_SPECS)
    }
    event_record = {
        event: {
            "tail_row_index": int(row), "sequence_id": str(np.asarray(arrays["sequence_id"])[rows[row]]),
            "selection": "largest observed severity within the named held-out event class",
        }
        for event, row in selected.items()
    }
    save_json(_json_value(event_record), output / "selected_events.json")
    manifest_report = {
        "protocol": {
            "name": "held-out highD EVT-tail conditional closed-loop reconstruction",
            "horizon_seconds": 5.0, "simulation_dt_s": DT_S, "num_stochastic_futures": NUM_SAMPLES,
            "fixed_conditions": ["logged one-second traffic history", "frozen B0", "road graph", "observed ego replay"],
            "num_sequences": int(len(rows)), "seed": int(args.seed),
            "cat_topk_information_asymmetric": True,
            "cat_topk_start_condition": "archived future-action summary",
            "qr_start_condition": "Flow-aligned B0 sidecar; current/history-only ego observations during rollout",
        },
        "sequence_cache": manifest,
        "checkpoints": {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in checkpoints.items()},
        "event_counts": {name: int(mask.sum()) for name, mask in masks.items()},
    }
    save_json(_json_value(manifest_report), output / "study_manifest.json")
    overview = ensure_dir(output / "overview")
    overview_report = {
        "protocol": manifest_report["protocol"],
        "models": {
            name: {
                "metrics": f"../{name}/metrics.json",
                "FDE_m": reports[name]["trajectory_reconstruction"]["FDE_m"],
                "minFDE_at_32_m": reports[name]["trajectory_reconstruction"]["minFDE_at_32_m"],
                "traffic_feature_frechet_distance": reports[name]["distribution_realism"]["traffic_feature_frechet_distance"],
            }
            for name, _label, _color in MODEL_SPECS
        },
    }
    save_json(_json_value(overview_report), overview / "long_tail_reconstruction_summary.json")
    _plot_overview(overview / "00_model_overview.png", reports)
    for name, label, color in MODEL_SPECS:
        model_dir = ensure_dir(output / name)
        figures, playbacks = ensure_dir(model_dir / "figures"), ensure_dir(model_dir / "event_playbacks")
        report = {
            "model": label,
            "checkpoint": {"path": str(checkpoints[name]), "sha256": file_sha256(checkpoints[name])},
            "information_conditions": {
                "strictly_information_symmetric": name != "cat_topk_world_model",
                "cat_topk_start_uses_archived_future_action_summary": name == "cat_topk_world_model",
            },
            **reports[name],
        }
        save_json(_json_value(report), model_dir / "metrics.json")
        _plot_examples(figures / "01_reconstruction_examples.png", samples[name], target, ego, valid, selected, color=color, label=label)
        _plot_trajectory_family(figures / "02_trajectory_family.png", samples[name], target, valid, color=color, label=label)
        _plot_kinematic(figures / "03_kinematic_realism.png", samples[name], target, ego, valid, report, color=color, label=label)
        _plot_interaction(figures / "04_interaction_realism.png", samples[name], target, ego, valid, report, color=color, label=label)
        _plot_distribution(figures / "05_distribution_realism.png", report, color=color, label=label)
        _plot_events(figures / "06_tail_event_breakdown.png", report, color=color, label=label)
        for event, row in selected.items():
            _gif(playbacks / f"{event}.gif", samples[name], target, ego, valid, row, color=color, label=label)
    print(overview / "long_tail_reconstruction_summary.json")


if __name__ == "__main__":
    main()

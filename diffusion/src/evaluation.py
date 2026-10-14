"""Open-loop trajectory reconstruction and distribution metrics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.plot_style import (
    GENERATED_COLOR,
    PAPER_FIGURE_DPI,
    PAPER_PANEL_LABELSIZE,
    REAL_COLOR,
    get_pyplot,
    style_axes,
)

from hierarchical_world_model.src.core.utils import (
    ensure_dir,
    save_json,
    select_device,
    set_seed,
)

from .data import (
    BackgroundTrajectoryDataset,
    decode_residual_trajectory,
    load_data_bundle,
    make_loader,
    split_rows,
    states_from_smooth_positions,
)
from .train import load_checkpoint

HORIZON_FRAMES = (25, 50, 75, 100, 125, 149)
HORIZON_NAMES = ("1.00s", "2.00s", "3.00s", "4.00s", "5.00s", "5.96s")
LOGGER = logging.getLogger(__name__)


def _new_component_error_totals(components: int) -> dict[str, np.ndarray | int]:
    return {
        "absolute": np.zeros(components, dtype=np.float64),
        "squared": np.zeros(components, dtype=np.float64),
        "count": 0,
    }


def _accumulate_component_errors(
    totals: dict[str, np.ndarray | int],
    generated: np.ndarray,
    target: np.ndarray,
    active: np.ndarray,
) -> None:
    error = np.asarray(generated, np.float64) - np.asarray(target, np.float64)[:, None]
    mask = np.broadcast_to(active[:, None, None, :, None], error.shape)
    for component in range(error.shape[-1]):
        selected = error[..., component][mask[..., component]]
        totals["absolute"][component] += np.abs(selected).sum()
        totals["squared"][component] += np.square(selected).sum()
    totals["count"] += int(mask[..., 0].sum())


def _component_error_metrics(
    totals: dict[str, np.ndarray | int], names: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    count = max(int(totals["count"]), 1)
    absolute = np.asarray(totals["absolute"], np.float64)
    squared = np.asarray(totals["squared"], np.float64)
    return {
        name: {
            "MAE": float(absolute[index] / count),
            "RMSE": float(np.sqrt(squared[index] / count)),
        }
        for index, name in enumerate(names)
    }


def _errors(
    samples: np.ndarray, target: np.ndarray, active: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    distance = np.linalg.norm(samples - target[:, None], axis=-1)
    mask = active[:, None, None, :]
    denominator = active.sum(axis=1, keepdims=True) * samples.shape[2]
    ade = (distance * mask).sum(axis=(2, 3)) / denominator.clip(min=1)
    fde = (distance[:, :, -1] * active[:, None]).sum(axis=2) / active.sum(
        axis=1, keepdims=True
    ).clip(min=1)
    return ade, fde


def _crossing_aligned_cutin_metrics(
    samples: np.ndarray,
    targets: np.ndarray,
    deterministic: np.ndarray,
    row_index: np.ndarray,
    cohort_path: Path,
) -> dict[str, float | int]:
    """Evaluate the indexed 4 s windows instead of a C0-aligned prefix."""
    cohort = np.load(cohort_path)
    row_to_local = {int(row): index for index, row in enumerate(row_index)}
    windows = []
    for row, slot, start in zip(
        cohort["sequence_row"],
        cohort["target_slot_index"],
        cohort["window_start_step"],
    ):
        local = row_to_local.get(int(row))
        if local is not None:
            windows.append((local, int(slot) - 1, int(start)))
    if not windows:
        return {"windows": 0, "sequences": 0}
    sample_ade, sample_fde, zero_ade, zero_fde, ensemble_ade, ensemble_fde = (
        [] for _ in range(6)
    )
    for sequence, slot, start in windows:
        stop = start + 100
        distance = np.linalg.norm(
            samples[sequence, :, start:stop, slot]
            - targets[sequence, None, start:stop, slot],
            axis=-1,
        )
        zero_distance = np.linalg.norm(
            deterministic[sequence, start:stop, slot]
            - targets[sequence, start:stop, slot],
            axis=-1,
        )
        ensemble_distance = np.linalg.norm(
            samples[sequence, :, start:stop, slot].mean(axis=0)
            - targets[sequence, start:stop, slot],
            axis=-1,
        )
        sample_ade.append(distance.mean(axis=1))
        sample_fde.append(distance[:, -1])
        zero_ade.append(zero_distance.mean())
        zero_fde.append(zero_distance[-1])
        ensemble_ade.append(ensemble_distance.mean())
        ensemble_fde.append(ensemble_distance[-1])
    ade = np.asarray(sample_ade)
    fde = np.asarray(sample_fde)
    return {
        "windows": len(windows),
        "sequences": len({sequence for sequence, _, _ in windows}),
        "zero_latent_ADE_m": float(np.mean(zero_ade)),
        "zero_latent_FDE_m": float(np.mean(zero_fde)),
        "sample_mean_ADE_m": float(ade.mean()),
        "sample_mean_FDE_m": float(fde.mean()),
        "min_ADE_m": float(ade.min(axis=1).mean()),
        "min_FDE_m": float(fde.min(axis=1).mean()),
        "ensemble_mean_ADE_m": float(np.mean(ensemble_ade)),
        "ensemble_mean_FDE_m": float(np.mean(ensemble_fde)),
    }


def _wasserstein(left: np.ndarray, right: np.ndarray) -> float:
    x = np.sort(np.asarray(left, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(right, dtype=np.float64).reshape(-1))
    count = max(len(x), len(y))
    quantiles = (np.arange(count) + 0.5) / count
    xq = np.interp(quantiles, (np.arange(len(x)) + 0.5) / len(x), x)
    yq = np.interp(quantiles, (np.arange(len(y)) + 0.5) / len(y), y)
    return float(np.mean(np.abs(xq - yq)))


def _ks(left: np.ndarray, right: np.ndarray) -> float:
    x = np.sort(np.asarray(left, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(right, dtype=np.float64).reshape(-1))
    pooled = np.sort(np.concatenate((x, y)))
    return float(
        np.max(
            np.abs(
                np.searchsorted(x, pooled, side="right") / max(len(x), 1)
                - np.searchsorted(y, pooled, side="right") / max(len(y), 1)
            )
        )
    )


def _action_fidelity(
    generated: np.ndarray, target: np.ndarray, active: np.ndarray, *, dt: float = 0.04
) -> dict[str, dict[str, float]]:
    generated_mask = np.broadcast_to(active[:, None, None, :, None], generated.shape)
    target_mask = np.broadcast_to(active[:, None, :, None], target.shape)
    fields = {
        "ax_mps2": (
            generated[..., 0][generated_mask[..., 0]],
            target[..., 0][target_mask[..., 0]],
        ),
        "ay_mps2": (
            generated[..., 1][generated_mask[..., 1]],
            target[..., 1][target_mask[..., 1]],
        ),
    }
    generated_jerk = np.diff(generated, axis=2) / float(dt)
    target_jerk = np.diff(target, axis=1) / float(dt)
    generated_jerk_mask = generated_mask[:, :, 1:]
    target_jerk_mask = target_mask[:, 1:]
    fields.update(
        {
            "jx_mps3": (
                generated_jerk[..., 0][generated_jerk_mask[..., 0]],
                target_jerk[..., 0][target_jerk_mask[..., 0]],
            ),
            "jy_mps3": (
                generated_jerk[..., 1][generated_jerk_mask[..., 1]],
                target_jerk[..., 1][target_jerk_mask[..., 1]],
            ),
        }
    )
    return {
        name: {"KS": _ks(real, predicted), "wasserstein": _wasserstein(real, predicted)}
        for name, (predicted, real) in fields.items()
    }


def _physical_rates(
    actions: np.ndarray,
    vx: np.ndarray,
    active: np.ndarray,
    *,
    dt: float = 0.04,
) -> dict[str, float]:
    plans = np.asarray(actions, dtype=np.float32)
    if plans.ndim == 4:
        plans = plans[:, None]
    mask = np.broadcast_to(active[:, None, None, :, None], plans.shape)
    jerk = np.diff(plans, axis=2) / float(dt)
    jerk_mask = mask[:, :, 1:]
    denominator = max(int(mask[..., 0].sum()), 1)
    jerk_denominator = max(int(jerk_mask[..., 0].sum()), 1)
    velocity = np.asarray(vx)
    if velocity.ndim == 3:
        velocity = velocity[:, None]
    negative_speed = int(((velocity < 0.0) & mask[..., 0]).sum())
    return {
        "nonfinite_rate": float((~np.isfinite(plans) & mask).sum() / (2 * denominator)),
        "ax_outside_minus8_4_rate": float(
            (((plans[..., 0] < -8.0) | (plans[..., 0] > 4.0)) & mask[..., 0]).sum()
            / denominator
        ),
        "abs_ay_above4_rate": float(
            ((np.abs(plans[..., 1]) > 4.0) & mask[..., 1]).sum() / denominator
        ),
        "abs_jx_above12_rate": float(
            ((np.abs(jerk[..., 0]) > 12.0) & jerk_mask[..., 0]).sum() / jerk_denominator
        ),
        "abs_jy_above8_rate": float(
            ((np.abs(jerk[..., 1]) > 8.0) & jerk_mask[..., 1]).sum() / jerk_denominator
        ),
        "negative_vx_rate": float(negative_speed / denominator),
    }


def _physical_fidelity(
    generated: np.ndarray,
    target: np.ndarray,
    generated_vx: np.ndarray,
    target_vx: np.ndarray,
    active: np.ndarray,
) -> dict[str, Any]:
    return {
        "thresholds": {
            "ax_mps2": [-8.0, 4.0],
            "abs_ay_mps2": 4.0,
            "abs_jx_mps3": 12.0,
            "abs_jy_mps3": 8.0,
        },
        "generated": _physical_rates(generated, generated_vx, active),
        "target": _physical_rates(target, target_vx, active),
    }


def _cutin_semantics(
    generated: np.ndarray,
    target: np.ndarray,
    initial_background: np.ndarray,
    initial_ego: np.ndarray,
    ego_future: np.ndarray,
    cutin_active: np.ndarray,
    *,
    lane_half_width_m: float = 1.8,
) -> dict[str, Any]:
    """Measure entry direction and >=70% post-entry lane retention."""
    target_with_draw = target[:, None]
    initial_relative_y = initial_background[..., 1] - initial_ego[:, None, 1]
    target_relative_y = target_with_draw[..., 1] - ego_future[:, None, :, None, 1]
    generated_relative_y = generated[..., 1] - ego_future[:, None, :, None, 1]

    def event_rows(relative_y: np.ndarray, active: np.ndarray) -> dict[str, np.ndarray]:
        per_agent = relative_y.transpose(0, 1, 3, 2)
        selected = per_agent[np.broadcast_to(active[:, None], per_agent.shape[:-1])]
        same_lane = np.abs(selected) <= float(lane_half_width_m)
        entered = same_lane.any(axis=1)
        first = np.argmax(same_lane, axis=1)
        frame = np.arange(same_lane.shape[1])[None]
        after_entry = frame >= first[:, None]
        retention = (same_lane & after_entry).sum(axis=1) / after_entry.sum(axis=1)
        return {
            "entered": entered,
            "retained": entered & (retention >= 0.7),
            "terminal_same_lane": same_lane[:, -1],
        }

    target_rows = event_rows(target_relative_y, cutin_active)
    generated_rows = event_rows(generated_relative_y, cutin_active)
    generated_displacement = (
        generated_relative_y[:, :, -1] - initial_relative_y[:, None]
    )[np.broadcast_to(cutin_active[:, None], generated_relative_y[:, :, -1].shape)]
    target_direction = np.broadcast_to(
        np.sign(target_relative_y[:, :, -1] - initial_relative_y[:, None]),
        generated_relative_y[:, :, -1].shape,
    )[np.broadcast_to(cutin_active[:, None], generated_relative_y[:, :, -1].shape)]
    direction_valid = np.abs(target_direction) > 0.0
    direction_match = np.sign(generated_displacement) == target_direction
    return {
        "cutin_agent_events": int(cutin_active.sum()),
        "generated_draw_events": int(len(generated_rows["entered"])),
        "lane_entry_rate": {
            "generated": float(generated_rows["entered"].mean()),
            "target": float(target_rows["entered"].mean()),
        },
        "post_entry_retention_70pct_rate": {
            "generated": float(generated_rows["retained"].mean()),
            "target": float(target_rows["retained"].mean()),
        },
        "terminal_same_lane_rate": {
            "generated": float(generated_rows["terminal_same_lane"].mean()),
            "target": float(target_rows["terminal_same_lane"].mean()),
        },
        "target_direction_match_rate": float(direction_match[direction_valid].mean()),
    }


def _risk_variables(
    background: np.ndarray,
    ego: np.ndarray,
    active: np.ndarray,
    vehicle_dimensions: np.ndarray,
    *,
    lane_half_width_m: float = 1.8,
) -> dict[str, np.ndarray]:
    """Compute bounded interaction-risk proxies for distribution comparison."""
    bg = np.asarray(background, dtype=np.float32)
    ego_states = np.asarray(ego, dtype=np.float32)
    dx = bg[..., 0] - ego_states[..., 0][..., None]
    dy = bg[..., 1] - ego_states[..., 1][..., None]
    dvx = bg[..., 2] - ego_states[..., 2][..., None]
    mask = np.broadcast_to(active[:, None, None, :], dx.shape)
    same_lane = np.abs(dy) <= float(lane_half_width_m)
    relevant = mask & same_lane
    gap = np.where(relevant, np.abs(dx), 200.0)
    closing = relevant & (dx * dvx < 0.0) & (np.abs(dvx) > 1.0e-3)
    ttc = np.where(closing, np.abs(dx) / np.maximum(np.abs(dvx), 1.0e-3), 20.0)
    dimensions = np.asarray(vehicle_dimensions, np.float32)
    pair_dimensions = 0.5 * (dimensions[:, :1] + dimensions[:, 1:])
    collision = (
        mask
        & (np.abs(dx) <= pair_dimensions[:, None, None, :, 0])
        & (np.abs(dy) <= pair_dimensions[:, None, None, :, 1])
    )
    return {
        "minimum_same_lane_center_gap_m": gap.min(axis=(-2, -1)),
        "minimum_ttc_s": np.minimum(ttc.min(axis=(-2, -1)), 20.0),
        "collision_proxy": collision.any(axis=(-2, -1)),
    }


def _risk_fidelity(
    generated: dict[str, np.ndarray], target: dict[str, np.ndarray]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("minimum_same_lane_center_gap_m", "minimum_ttc_s"):
        result[name] = {
            "generated_mean": float(np.mean(generated[name])),
            "target_mean": float(np.mean(target[name])),
            "KS": _ks(target[name], generated[name]),
            "wasserstein": _wasserstein(target[name], generated[name]),
        }
    generated_collision = float(np.mean(generated["collision_proxy"]))
    target_collision = float(np.mean(target["collision_proxy"]))
    result["collision_proxy_rate"] = {
        "generated": generated_collision,
        "target": target_collision,
        "generated_events": int(np.sum(generated["collision_proxy"])),
        "target_events": int(np.sum(target["collision_proxy"])),
        "rate_ratio": (
            generated_collision / target_collision if target_collision > 0.0 else None
        ),
    }
    return result


def _cohort_metrics(
    sample_positions: np.ndarray,
    target_positions: np.ndarray,
    active: np.ndarray,
    deterministic_positions: np.ndarray,
) -> dict[str, Any]:
    source_sequences = len(target_positions)
    modeled = np.asarray(active, bool).any(axis=1)
    sample_positions = sample_positions[modeled]
    target_positions = target_positions[modeled]
    active = active[modeled]
    deterministic_positions = deterministic_positions[modeled]
    sample_ade, sample_fde = _errors(sample_positions, target_positions, active)
    mean_positions = sample_positions.mean(axis=1, keepdims=True)
    mean_ade, mean_fde = _errors(mean_positions, target_positions, active)
    horizons: dict[str, dict[str, float]] = {}
    for name, frame in zip(HORIZON_NAMES, HORIZON_FRAMES):
        local_samples = sample_positions[:, :, :frame]
        local_target = target_positions[:, :frame]
        local_ade, local_fde = _errors(local_samples, local_target, active)
        local_mean_ade, local_mean_fde = _errors(
            local_samples.mean(axis=1, keepdims=True), local_target, active
        )
        horizons[name] = {
            "sample_mean_ADE_m": float(local_ade.mean()),
            "sample_mean_FDE_m": float(local_fde.mean()),
            "min_ADE_m": float(local_ade.min(axis=1).mean()),
            "min_FDE_m": float(local_fde.min(axis=1).mean()),
            "ensemble_mean_ADE_m": float(local_mean_ade.mean()),
            "ensemble_mean_FDE_m": float(local_mean_fde.mean()),
        }
    distinct_pair_distance = np.zeros(len(sample_positions), dtype=np.float64)
    distinct_endpoint_distance = np.zeros(len(sample_positions), dtype=np.float64)
    distinct_pair_count = 0
    pair_denominator = (active.sum(axis=1) * sample_positions.shape[2]).clip(min=1)
    for left in range(sample_positions.shape[1]):
        for right in range(left + 1, sample_positions.shape[1]):
            pair_distance = np.linalg.norm(
                sample_positions[:, left] - sample_positions[:, right], axis=-1
            )
            distinct_pair_distance += (pair_distance * active[:, None]).sum(
                axis=(1, 2)
            ) / pair_denominator
            distinct_endpoint_distance += (pair_distance[:, -1] * active).sum(
                axis=1
            ) / active.sum(axis=1).clip(min=1)
            distinct_pair_count += 1
    result = {
        "sequences": int(len(target_positions)),
        "source_sequences": int(source_sequences),
        "samples_per_condition": int(sample_positions.shape[1]),
        "sample_mean_ADE_m": float(sample_ade.mean()),
        "sample_mean_FDE_m": float(sample_fde.mean()),
        "min_ADE_m": float(sample_ade.min(axis=1).mean()),
        "min_FDE_m": float(sample_fde.min(axis=1).mean()),
        "ensemble_mean_ADE_m": float(mean_ade.mean()),
        "ensemble_mean_FDE_m": float(mean_fde.mean()),
        "mean_pairwise_sample_distance_m": float(
            (distinct_pair_distance / max(distinct_pair_count, 1)).mean()
        ),
        "mean_pairwise_endpoint_distance_m": float(
            (distinct_endpoint_distance / max(distinct_pair_count, 1)).mean()
        ),
        "horizons": horizons,
    }
    deterministic_ade, deterministic_fde = _errors(
        deterministic_positions[:, None], target_positions, active
    )
    result["zero_latent_ADE_m"] = float(deterministic_ade.mean())
    result["zero_latent_FDE_m"] = float(deterministic_fde.mean())
    for name, frame in zip(HORIZON_NAMES, HORIZON_FRAMES):
        local_ade, local_fde = _errors(
            deterministic_positions[:, None, :frame],
            target_positions[:, :frame],
            active,
        )
        result["horizons"][name]["zero_latent_ADE_m"] = float(local_ade.mean())
        result["horizons"][name]["zero_latent_FDE_m"] = float(local_fde.mean())
    return result


def _panel_labels(axes: np.ndarray | list[Any]) -> None:
    for label, axis in zip("abcdefghijklmnopqrstuvwxyz", np.asarray(axes).ravel()):
        text = getattr(axis, "text2D", axis.text)
        x_position = 0.035 if hasattr(axis, "text2D") else -0.11
        text(
            x_position,
            1.05,
            label,
            transform=axis.transAxes,
            fontsize=PAPER_PANEL_LABELSIZE,
            fontweight="bold",
        )


def _trajectory_pca(
    samples: np.ndarray,
    initial_positions: np.ndarray,
    targets: np.ndarray,
    active: np.ndarray,
    *,
    max_agents: int = 3_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project paired position-displacement trajectories with one shared PCA."""
    pairs = np.argwhere(np.asarray(active, bool))
    if len(pairs) > max_agents:
        rng = np.random.default_rng(20260812)
        pairs = pairs[rng.choice(len(pairs), size=max_agents, replace=False)]
    sequence, agent = pairs.T
    origin = initial_positions[sequence, agent][:, None]
    real = targets[sequence, :, agent] - origin
    generated = samples[sequence, 0, :, agent] - origin
    pooled = np.concatenate((real, generated), axis=0).reshape(2 * len(pairs), -1)
    scale = pooled.std(axis=0)
    retained = scale > 1.0e-6
    standardized = (pooled[:, retained] - pooled[:, retained].mean(axis=0)) / scale[
        retained
    ]
    _, singular_values, components = np.linalg.svd(standardized, full_matrices=False)
    projection = standardized @ components[:3].T
    explained = np.square(singular_values[:3]) / np.square(singular_values).sum()
    return projection[: len(pairs)], projection[len(pairs) :], explained, pairs


def _recording_mean_interval(
    values: np.ndarray, recording_ids: np.ndarray
) -> np.ndarray:
    """Bootstrap recordings rather than treating overlapping scenes as independent."""
    _, inverse = np.unique(recording_ids, return_inverse=True)
    count = np.bincount(inverse)
    total = np.bincount(inverse, weights=values)
    rng = np.random.default_rng(20260812)
    indices = rng.integers(0, len(count), size=(500, len(count)))
    means = total[indices].sum(axis=1) / count[indices].sum(axis=1)
    return np.quantile(means, (0.025, 0.975))


def _save_evaluation_diagnostics(
    summary: dict[str, Any],
    sample_ade: np.ndarray,
    recording_ids: np.ndarray,
    velocity_error: np.ndarray,
    real_projection: np.ndarray,
    generated_projection: np.ndarray,
    explained: np.ndarray,
    output: Path,
) -> None:
    """Summarize Test accuracy, sample utility, and trajectory coverage."""
    metrics = summary["metrics"]
    all_metrics = metrics["all"]
    sample_ade = np.asarray(sample_ade, np.float64)
    velocity_error = np.asarray(velocity_error, np.float64)
    per_condition = sample_ade.mean(axis=1)
    plt = get_pyplot()
    figure = plt.figure(figsize=(17.2, 8.8), constrained_layout=True)
    grid = figure.add_gridspec(
        2,
        4,
        width_ratios=(1.30, 1.30, 1.0, 1.0),
        wspace=0.18,
        hspace=0.18,
    )
    axes = np.asarray(
        [
            figure.add_subplot(grid[:, :2], projection="3d"),
            figure.add_subplot(grid[0, 2]),
            figure.add_subplot(grid[0, 3]),
            figure.add_subplot(grid[1, 2]),
            figure.add_subplot(grid[1, 3]),
        ],
        dtype=object,
    )

    # The shared projection is deliberately the dominant panel. Thin connectors
    # identify a fixed subset of paired highD/generated position trajectories.
    pooled_projection = np.concatenate((real_projection, generated_projection), axis=0)
    limits = np.quantile(pooled_projection, (0.005, 0.995), axis=0)
    margin = 0.04 * (limits[1] - limits[0])
    limits += np.stack((-margin, margin))
    real_visible = (
        (real_projection >= limits[0]) & (real_projection <= limits[1])
    ).all(1)
    generated_visible = (
        (generated_projection >= limits[0]) & (generated_projection <= limits[1])
    ).all(1)
    paired_visible = np.flatnonzero(real_visible & generated_visible)
    connector_rows = paired_visible[
        np.linspace(
            0, len(paired_visible) - 1, min(120, len(paired_visible)), dtype=int
        )
    ]
    for row in connector_rows:
        axes[0].plot(
            (real_projection[row, 0], generated_projection[row, 0]),
            (real_projection[row, 1], generated_projection[row, 1]),
            (real_projection[row, 2], generated_projection[row, 2]),
            color="#6F6F6F",
            alpha=0.10,
            linewidth=0.45,
            zorder=0,
        )
    axes[0].scatter(
        real_projection[real_visible, 0],
        real_projection[real_visible, 1],
        real_projection[real_visible, 2],
        s=8,
        alpha=0.20,
        color=REAL_COLOR,
        edgecolors="none",
        rasterized=True,
        label="highD",
    )
    axes[0].scatter(
        generated_projection[generated_visible, 0],
        generated_projection[generated_visible, 1],
        generated_projection[generated_visible, 2],
        s=8,
        alpha=0.24,
        color=GENERATED_COLOR,
        edgecolors="none",
        rasterized=True,
        label="Diffusion",
    )
    for projection, color in (
        (real_projection, REAL_COLOR),
        (generated_projection, GENERATED_COLOR),
    ):
        center = np.asarray(projection, np.float64).mean(axis=0)
        axes[0].scatter(
            *center,
            marker="X",
            s=90,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            depthshade=False,
            zorder=5,
        )
    for dimension, setter in enumerate(
        (axes[0].set_xlim, axes[0].set_ylim, axes[0].set_zlim)
    ):
        setter(*limits[:, dimension])
    axes[0].set(
        xlabel=f"PC1 ({100 * explained[0]:.1f}%)",
        ylabel=f"PC2 ({100 * explained[1]:.1f}%)",
        zlabel=f"PC3 ({100 * explained[2]:.1f}%)",
        title="Shared 3D position-trajectory embedding",
    )
    axes[0].view_init(elev=21, azim=-56)
    axes[0].set_box_aspect((1.0, 1.0, 0.78))
    axes[0].tick_params(labelsize=11, pad=1)
    axes[0].xaxis.labelpad = 5
    axes[0].yaxis.labelpad = 5
    axes[0].zaxis.labelpad = 5
    axes[0].legend(
        markerscale=2.7,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.94),
    )
    axes[0].text2D(
        0.02,
        0.02,
        "Initial-relative trajectories; central 99% per-axis limits",
        transform=axes[0].transAxes,
        fontsize=10.5,
        color="#555555",
    )
    axes[0].grid(True, color="#D9D9D9", linewidth=0.4, alpha=0.55)

    # Knot errors vanish by construction, not by free trajectory prediction.
    times = np.asarray((1.0, 2.0, 3.0, 4.0, 5.0, 5.96))
    horizon_keys = tuple(all_metrics["horizons"])
    time_curves = (
        (
            "Single-draw mean",
            "sample_mean_FDE_m",
            GENERATED_COLOR,
            "o",
        ),
        ("Ensemble mean", "ensemble_mean_FDE_m", REAL_COLOR, "s"),
        ("Best of 16", "min_FDE_m", "#54A24B", "^"),
    )
    for label, key, color, marker in time_curves:
        axes[1].plot(
            times,
            [all_metrics["horizons"][horizon][key] for horizon in horizon_keys],
            marker=marker,
            markersize=4.2,
            color=color,
            label=label,
        )
    for index, knot in enumerate((2.0, 4.0, 5.96)):
        axes[1].axvline(
            knot,
            color="#777777",
            linestyle=(0, (2, 2)),
            linewidth=0.8,
            alpha=0.65,
            label="Hard-constrained knot" if index == 0 else None,
        )
    axes[1].set(
        xlabel="Time (s)",
        ylabel="Displacement error at time $t$ (m)",
        title="Displacement error (knots enforced)",
        xticks=times,
        xticklabels=("1", "2", "3", "4", "5", "5.96"),
        ylim=(
            0.0,
            1.3
            * max(
                all_metrics["horizons"][horizon]["sample_mean_FDE_m"]
                for horizon in horizon_keys
            ),
        ),
    )
    axes[1].legend(frameon=False, fontsize=9.5, ncol=2, columnspacing=0.9)
    style_axes(axes[1])

    # Estimate best-of-K with every one of the 16 stored draws: for K < 16,
    # disjoint groups are averaged so the curve is not tied to a single draw.
    sample_count = sample_ade.shape[1]
    k_values = np.asarray([k for k in (1, 2, 4, 8, 16) if sample_count % k == 0])
    best_of_k = []
    best_of_k_ci = []
    for k in k_values:
        condition_values = (
            sample_ade.reshape(len(sample_ade), sample_count // k, k)
            .min(axis=2)
            .mean(axis=1)
        )
        best_of_k.append(condition_values.mean())
        best_of_k_ci.append(_recording_mean_interval(condition_values, recording_ids))
    best_of_k = np.asarray(best_of_k)
    best_of_k_ci = np.asarray(best_of_k_ci)
    axes[2].fill_between(
        k_values,
        best_of_k_ci[:, 0],
        best_of_k_ci[:, 1],
        color="#54A24B",
        alpha=0.18,
        linewidth=0,
        label="95% recording CI",
    )
    axes[2].plot(
        k_values,
        best_of_k,
        marker="o",
        markersize=4.8,
        color="#54A24B",
        label="Best-of-$K$ ADE",
    )
    improvement = 100.0 * (1.0 - best_of_k[-1] / best_of_k[0])
    axes[2].annotate(
        f"{improvement:.1f}% lower",
        xy=(k_values[-1], best_of_k[-1]),
        xytext=(-10, 16),
        textcoords="offset points",
        ha="right",
        color="#3B7D35",
        fontsize=10.5,
    )
    axes[2].set_xscale("log", base=2)
    axes[2].set(
        xlabel="Number of generated samples, $K$",
        ylabel="Best-of-$K$ ADE (m)",
        title="Utility of stochastic samples",
        xticks=k_values,
        xticklabels=[str(value) for value in k_values],
    )
    axes[2].legend(frameon=False, fontsize=9.5)
    style_axes(axes[2])

    ordered = np.sort(per_condition)
    cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
    axes[3].plot(ordered, cumulative, color=GENERATED_COLOR, linewidth=1.8, label="ADE")
    median, p95 = np.quantile(per_condition, (0.5, 0.95))
    axes[3].axvspan(median, p95, color=GENERATED_COLOR, alpha=0.08, linewidth=0)
    axes[3].scatter(
        (median, p95),
        (0.5, 0.95),
        color=(REAL_COLOR, "#E45756"),
        zorder=3,
    )
    axes[3].text(
        0.04,
        0.96,
        f"ADE median {median:.3f} m\nADE P95 {p95:.3f} m",
        transform=axes[3].transAxes,
        va="top",
        fontsize=10.0,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    axes[3].set(
        xlabel="Mean error per condition (m)",
        ylabel="Cumulative probability",
        title="Trajectory ADE distribution",
        xlim=(ordered[0] * 0.9, ordered[-1] * 1.04),
        ylim=(0.0, 1.01),
    )
    axes[3].set_xscale("log")
    from matplotlib.ticker import LogLocator, StrMethodFormatter

    axes[3].xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axes[3].xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    axes[3].legend(frameon=False, loc="lower right", fontsize=10.5)
    style_axes(axes[3])

    log_position = np.log10(per_condition)
    log_velocity = np.log10(velocity_error)
    density = axes[4].hexbin(
        log_position,
        log_velocity,
        gridsize=42,
        bins="log",
        mincnt=1,
        cmap="viridis",
        linewidths=0.08,
        rasterized=True,
    )
    slope, intercept = np.polyfit(log_position, log_velocity, 1)
    x_limits = np.quantile(log_position, (0.005, 0.995))
    y_limits = np.quantile(log_velocity, (0.005, 0.995))
    regression_x = np.linspace(*x_limits, 100)
    axes[4].plot(
        regression_x,
        slope * regression_x + intercept,
        color="white",
        linewidth=2.2,
        alpha=0.9,
    )
    axes[4].plot(
        regression_x,
        slope * regression_x + intercept,
        color="#303030",
        linewidth=0.9,
    )
    from matplotlib.ticker import MaxNLocator

    tick_locator = MaxNLocator(nbins=4)
    position_ticks = tick_locator.tick_values(*x_limits)
    velocity_ticks = tick_locator.tick_values(*y_limits)
    axes[4].set(
        xlabel="Expected ADE (m)",
        ylabel="Mean velocity error (m/s)",
        title="Position–velocity error density",
        xlim=x_limits,
        ylim=y_limits,
        xticks=position_ticks,
        xticklabels=[f"{10 ** value:.2g}" for value in position_ticks],
        yticks=velocity_ticks,
        yticklabels=[f"{10 ** value:.2g}" for value in velocity_ticks],
    )
    correlation = np.corrcoef(log_position, log_velocity)[0, 1]
    axes[4].text(
        0.04,
        0.94,
        f"log-scale $r={correlation:.2f}$",
        transform=axes[4].transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=10.5,
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": "#303030",
            "edgecolor": "none",
            "alpha": 0.75,
        },
    )
    colorbar = figure.colorbar(density, ax=axes[4], pad=0.02, fraction=0.05)
    colorbar.set_label("Condition count", fontsize=10.5)
    colorbar.ax.tick_params(labelsize=9)
    style_axes(axes[4], grid=False)

    _panel_labels(axes)
    figures = ensure_dir(output / "figures")
    figure.savefig(figures / "evaluation_diagnostics.png", dpi=PAPER_FIGURE_DPI)
    plt.close(figure)


@torch.no_grad()
def evaluate_background_diffusion(
    config: dict[str, Any],
    *,
    config_dir: Path,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    output = ensure_dir(config["paths"]["output_dir"])
    cohort_path = output / "cohorts/cut_in.npz"
    if not cohort_path.is_file():
        raise FileNotFoundError(
            f"missing {cohort_path}; run diffusion/scripts/prepare.py"
        )
    device = select_device(evaluation.get("device", "auto"))
    set_seed(int(evaluation.get("seed", 42)))
    torch.set_float32_matmul_precision("high")
    checkpoint_path = output / "checkpoints/best_background_diffusion.pt"
    model, state = load_checkpoint(checkpoint_path, device=device)
    model.eval()
    bundle = load_data_bundle(config, config_dir)
    contract = state["dataset_contract"]
    rows = split_rows(bundle.arrays, str(evaluation["split"]))
    dataset = BackgroundTrajectoryDataset(bundle, rows, contract)
    loader = make_loader(
        dataset,
        batch_size=int(evaluation.get("batch_size", 16)),
        shuffle=False,
        workers=int(evaluation.get("num_workers", 4)),
        seed=0,
    )
    draws = int(evaluation.get("samples_per_condition", 8))
    inference_steps = int(evaluation.get("ddim_steps", 20))
    generated_batches: list[np.ndarray] = []
    deterministic_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    active_batches: list[np.ndarray] = []
    tail_batches: list[np.ndarray] = []
    semantic_cutin_batches: list[np.ndarray] = []
    row_batches: list[np.ndarray] = []
    generated_action_batches: list[np.ndarray] = []
    target_action_batches: list[np.ndarray] = []
    generated_vx_batches: list[np.ndarray] = []
    target_vx_batches: list[np.ndarray] = []
    velocity_error_batches: list[np.ndarray] = []
    c0_batches: list[np.ndarray] = []
    ego_future_batches: list[np.ndarray] = []
    generated_risk_batches: list[dict[str, np.ndarray]] = []
    target_risk_batches: list[dict[str, np.ndarray]] = []
    velocity_error_totals = _new_component_error_totals(2)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(evaluation.get("seed", 42)))
    for batch_index, raw_batch in enumerate(loader, start=1):
        condition = raw_batch["condition"].to(device)
        target_mask = raw_batch["target_mask"].to(device)
        size = condition.shape[0]
        expanded_condition = condition.repeat_interleave(draws, dim=0)
        expanded_mask = target_mask.repeat_interleave(draws, dim=0)
        normalized = model.sample_ddim(
            expanded_condition,
            expanded_mask,
            inference_steps=inference_steps,
            generator=generator,
        )
        deterministic_normalized = model.sample_ddim(
            condition,
            target_mask,
            inference_steps=inference_steps,
            initial_noise=torch.zeros_like(target_mask, dtype=condition.dtype),
        )
        active = raw_batch["target_mask"][:, 0].numpy().reshape(size, 6, 2)[:, :, 0]
        c0 = raw_batch["c0_states"].numpy()[:, 1:]
        reference = raw_batch["trajectory_reference"].numpy()
        generated = decode_residual_trajectory(
            normalized.cpu().numpy().reshape(size, draws, 149, 12),
            reference[:, None],
            active[:, None],
            contract,
        )
        c0_batches.append(raw_batch["c0_states"].numpy())
        generated_states = states_from_smooth_positions(
            np.broadcast_to(c0[:, None], (size, draws, 6, 6)), generated
        )
        deterministic = decode_residual_trajectory(
            deterministic_normalized.cpu().numpy(), reference, active, contract
        )
        full_target = raw_batch["future_states"].numpy()
        target = full_target[:, :, 1:]
        _accumulate_component_errors(
            velocity_error_totals,
            generated_states[..., 2:4],
            target[..., 2:4],
            active,
        )
        velocity_distance = np.linalg.norm(
            generated_states[..., 2:4] - target[:, None, ..., 2:4], axis=-1
        )
        velocity_mask = active[:, None, None, :]
        velocity_denominator = (
            draws * generated_states.shape[2] * active.sum(axis=1)
        ).clip(min=1)
        velocity_error_batches.append(
            (velocity_distance * velocity_mask).sum(axis=(1, 2, 3))
            / velocity_denominator
        )
        generated_action_batches.append(generated_states[..., 4:6])
        generated_vx_batches.append(generated_states[..., 2])
        target_vx_batches.append(target[..., 2])
        target_actions = raw_batch["target_actions"].numpy()
        target_action_batches.append(target_actions)
        ego_future = full_target[:, None, :, 0]
        ego_future_batches.append(full_target[:, :, 0])
        generated_risk_batches.append(
            _risk_variables(
                generated_states, ego_future, active, raw_batch["vehicle_dimensions"]
            )
        )
        target_risk_batches.append(
            _risk_variables(
                target[:, None], ego_future, active, raw_batch["vehicle_dimensions"]
            )
        )
        generated_batches.append(generated)
        deterministic_batches.append(deterministic)
        target_batches.append(target[..., 0:2])
        active_batches.append(active)
        tail_batches.append(raw_batch["is_evt_tail"].numpy())
        semantic_cutin_batches.append(raw_batch["semantic_cutin_mask"].numpy())
        row_batches.append(raw_batch["row_index"].numpy())
        if batch_index % 5 == 0 or batch_index == len(loader):
            LOGGER.info("Evaluation batches %s/%s", batch_index, len(loader))
    samples = np.concatenate(generated_batches)
    deterministic = np.concatenate(deterministic_batches)
    targets = np.concatenate(target_batches)
    active = np.concatenate(active_batches).astype(bool)
    tail = np.concatenate(tail_batches).astype(bool)
    semantic_cutin = np.concatenate(semantic_cutin_batches).astype(bool)
    velocity_error = np.concatenate(velocity_error_batches)
    row_index = np.concatenate(row_batches)
    recording_ids = np.asarray(bundle.flow_arrays["recording_id"])[
        bundle.flow_row_for_sequence[row_index]
    ]
    metrics: dict[str, Any] = {
        "all": _cohort_metrics(samples, targets, active, deterministic),
        "velocity_reconstruction": _component_error_metrics(
            velocity_error_totals, ("vx_mps", "vy_mps")
        ),
    }
    metrics["crossing_aligned_cut_in"] = _crossing_aligned_cutin_metrics(
        samples,
        targets,
        deterministic,
        row_index,
        cohort_path,
    )
    following_active = active.copy()
    following_active[:, 2:] = False
    following_rows = following_active.any(axis=1)
    if following_rows.any():
        metrics["car_following"] = _cohort_metrics(
            samples[following_rows],
            targets[following_rows],
            following_active[following_rows],
            deterministic[following_rows],
        )
    cutin_active = semantic_cutin
    cutin_rows = cutin_active.any(axis=1)
    if cutin_rows.any():
        metrics["cut_in"] = _cohort_metrics(
            samples[cutin_rows],
            targets[cutin_rows],
            cutin_active[cutin_rows],
            deterministic[cutin_rows],
        )
    generated_actions = np.concatenate(generated_action_batches)
    target_actions = np.concatenate(target_action_batches)
    generated_vx = np.concatenate(generated_vx_batches)
    target_vx = np.concatenate(target_vx_batches)
    c0_states = np.concatenate(c0_batches)
    ego_future = np.concatenate(ego_future_batches)
    metrics["action_distribution_fidelity"] = _action_fidelity(
        generated_actions, target_actions, active
    )
    metrics["physical_feasibility"] = _physical_fidelity(
        generated_actions, target_actions, generated_vx, target_vx, active
    )
    if following_rows.any():
        metrics["car_following_action_distribution_fidelity"] = _action_fidelity(
            generated_actions[following_rows],
            target_actions[following_rows],
            following_active[following_rows],
        )
        metrics["car_following_physical_feasibility"] = _physical_fidelity(
            generated_actions[following_rows],
            target_actions[following_rows],
            generated_vx[following_rows],
            target_vx[following_rows],
            following_active[following_rows],
        )
    if cutin_rows.any():
        metrics["cut_in_action_distribution_fidelity"] = _action_fidelity(
            generated_actions[cutin_rows],
            target_actions[cutin_rows],
            cutin_active[cutin_rows],
        )
        metrics["cut_in_physical_feasibility"] = _physical_fidelity(
            generated_actions[cutin_rows],
            target_actions[cutin_rows],
            generated_vx[cutin_rows],
            target_vx[cutin_rows],
            cutin_active[cutin_rows],
        )
        metrics["cut_in_semantic_fidelity"] = _cutin_semantics(
            samples,
            targets,
            c0_states[:, 1:, 0:2],
            c0_states[:, 0, 0:2],
            ego_future,
            cutin_active,
            lane_half_width_m=1.0,
        )
    generated_risk = {
        key: np.concatenate([batch[key] for batch in generated_risk_batches])[
            active.any(axis=1)
        ]
        for key in generated_risk_batches[0]
    }
    target_risk = {
        key: np.concatenate([batch[key] for batch in target_risk_batches])[
            active.any(axis=1)
        ]
        for key in target_risk_batches[0]
    }
    metrics["risk_distribution_fidelity"] = _risk_fidelity(generated_risk, target_risk)
    for name, mask in (("evt_tail", tail), ("non_tail", ~tail)):
        if mask.any():
            metrics[name] = _cohort_metrics(
                samples[mask], targets[mask], active[mask], deterministic[mask]
            )
    summary = {
        "model": contract.get("name", "highd_background_trajectory_diffusion"),
        "role": "position_knot_conditioned_open_loop_diffusion",
        "checkpoint": "checkpoints/best_background_diffusion.pt",
        "checkpoint_epoch": int(state["epoch"]),
        "experiment_scope": "full",
        "ego_future_in_condition": bool(contract.get("ego_future_in_condition", True)),
        "target_representation": contract.get("target_representation"),
        "trajectory_interpolation": True,
        "knots_hard_enforced": True,
        "evaluation_scope": "oracle_sparse_position_knot_reconstruction",
        "end_to_end_plan_sampling_evaluated": False,
        "reference_aligned_horizons": {
            "car_following_seconds": 5.0,
            "cut_in_seconds": 4.0,
            "joint_world_seconds": 5.96,
        },
        "samples_per_condition": draws,
        "ddim_steps": inference_steps,
        "architecture": "factorized_spatiotemporal",
        "prediction_type": "v_prediction",
        "x0_clip_abs": model.config.x0_clip_abs,
        "split": str(evaluation.get("split", "test")),
        "figure_uncertainty": "recording-bootstrap 95% CI, 500 resamples, seed 20260812",
        "metrics": metrics,
    }
    summary["condition_disclosure"] = {
        "uses_future_background_position_knots": True,
        "knot_times_s": [2.0, 4.0, 5.96],
        "uses_future_ego": False,
        "interpretation": (
            "position-knot-conditioned reconstruction, not C0-only prediction"
        ),
        "endpoint_is_conditioned": True,
        "knots_hard_enforced": True,
    }
    summary["headline_metrics"] = {
        "mean_single_draw_ADE_m": float(metrics["all"]["sample_mean_ADE_m"]),
        "longitudinal_velocity_MAE_mps": float(
            metrics["velocity_reconstruction"]["vx_mps"]["MAE"]
        ),
        "lateral_velocity_MAE_mps": float(
            metrics["velocity_reconstruction"]["vy_mps"]["MAE"]
        ),
        "mean_pairwise_sample_distance_m": float(
            metrics["all"]["mean_pairwise_sample_distance_m"]
        ),
    }
    save_json(summary, output / "evaluation_summary.json")
    sample_ade, _ = _errors(samples, targets, active)
    modeled = active.any(axis=1)
    real_projection, generated_projection, explained, projection_pairs = (
        _trajectory_pca(samples, c0_states[:, 1:, :2], targets, active)
    )
    np.savez_compressed(
        output / "evaluation_per_sequence.npz",
        row_index=row_index[modeled],
        recording_id=recording_ids[modeled],
        is_evt_tail=tail[modeled],
        active_background=active[modeled],
        sample_ade_m=sample_ade[modeled],
        mean_velocity_vector_error_mps=velocity_error[modeled],
        target_endpoint=targets[modeled, -1],
        generated_endpoint=samples[modeled, 0, -1],
        trajectory_pca_highd=real_projection,
        trajectory_pca_diffusion=generated_projection,
        trajectory_pca_explained_variance_ratio=explained,
        trajectory_pca_sequence_row=row_index[projection_pairs[:, 0]],
        trajectory_pca_agent_slot=projection_pairs[:, 1] + 1,
    )
    _save_evaluation_diagnostics(
        summary,
        sample_ade[modeled],
        recording_ids[modeled],
        velocity_error[modeled],
        real_projection,
        generated_projection,
        explained,
        output,
    )
    return summary

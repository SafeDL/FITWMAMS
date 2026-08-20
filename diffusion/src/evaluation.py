"""Open-loop trajectory reconstruction and distribution metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from world_model.src.core.utils import (
    ensure_dir,
    save_json,
    select_device,
    set_seed,
)

from .data import (
    BackgroundTrajectoryDataset,
    load_data_bundle,
    make_loader,
    pilot_rows,
    smooth_position_residual,
    states_from_smooth_positions,
)
from .train import load_checkpoint

HORIZON_FRAMES = (25, 50, 75, 100, 125, 149)
HORIZON_NAMES = ("1.00s", "2.00s", "3.00s", "4.00s", "5.00s", "5.96s")


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


def states_from_actions(
    initial_states: np.ndarray, actions: np.ndarray, *, dt: float = 0.04
) -> np.ndarray:
    """Integrate highD Cartesian acceleration plans without hidden corrections."""
    control = np.asarray(actions, dtype=np.float32)
    initial = np.asarray(initial_states, dtype=np.float32)
    leading = control.shape[:-3]
    current = np.broadcast_to(initial, leading + initial.shape[-2:]).copy()
    output = np.empty(leading + control.shape[-3:-1] + (6,), dtype=np.float32)
    for frame in range(control.shape[-3]):
        acceleration = control[..., frame, :, :]
        current[..., 0:2] += (
            current[..., 2:4] * float(dt) + 0.5 * acceleration * float(dt) ** 2
        )
        current[..., 2:4] += acceleration * float(dt)
        current[..., 4:6] = acceleration
        output[..., frame, :, :] = current
    return output


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
    initial_vx: np.ndarray,
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
    negative_speed = 0
    for start in range(0, len(plans), 256):
        chunk = plans[start : start + 256]
        velocity = initial_vx[start : start + 256, None, None] + np.cumsum(
            chunk[..., 0], axis=2
        ) * float(dt)
        velocity_mask = np.broadcast_to(
            active[start : start + 256, None, None], velocity.shape
        )
        negative_speed += int(((velocity < 0.0) & velocity_mask).sum())
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
    initial_vx: np.ndarray,
    active: np.ndarray,
) -> dict[str, Any]:
    return {
        "thresholds": {
            "ax_mps2": [-8.0, 4.0],
            "abs_ay_mps2": 4.0,
            "abs_jx_mps3": 12.0,
            "abs_jy_mps3": 8.0,
        },
        "generated": _physical_rates(generated, initial_vx, active),
        "target": _physical_rates(target, initial_vx, active),
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
    *,
    lane_half_width_m: float = 1.8,
    collision_half_length_m: float = 4.5,
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
    collision = relevant & (np.abs(dx) <= float(collision_half_length_m))
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
    deterministic_positions: np.ndarray | None = None,
) -> dict[str, Any]:
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
    error_to_truth = np.linalg.norm(
        sample_positions - target_positions[:, None], axis=-1
    )
    first_term = (error_to_truth * active[:, None, None]).sum(axis=(2, 3))
    first_term /= (active.sum(axis=1) * sample_positions.shape[2]).clip(min=1)[:, None]
    coordinate_error = np.abs(sample_positions - target_positions[:, None])
    coordinate_first = (coordinate_error * active[:, None, None, :, None]).sum(
        axis=(2, 3, 4)
    )
    coordinate_denominator = (2 * active.sum(axis=1) * sample_positions.shape[2]).clip(
        min=1
    )
    coordinate_first /= coordinate_denominator[:, None]
    second_term = np.zeros(len(sample_positions), dtype=np.float64)
    coordinate_second = np.zeros(len(sample_positions), dtype=np.float64)
    pair_count = 0
    distinct_pair_distance = np.zeros(len(sample_positions), dtype=np.float64)
    distinct_endpoint_distance = np.zeros(len(sample_positions), dtype=np.float64)
    distinct_pair_count = 0
    pair_denominator = (active.sum(axis=1) * sample_positions.shape[2]).clip(min=1)
    for left in range(sample_positions.shape[1]):
        for right in range(sample_positions.shape[1]):
            pair_distance = np.linalg.norm(
                sample_positions[:, left] - sample_positions[:, right], axis=-1
            )
            second_term += (pair_distance * active[:, None]).sum(
                axis=(1, 2)
            ) / pair_denominator
            coordinate_second += (
                np.abs(sample_positions[:, left] - sample_positions[:, right])
                * active[:, None, :, None]
            ).sum(axis=(1, 2, 3)) / coordinate_denominator
            pair_count += 1
            if right > left:
                distinct_pair_distance += (pair_distance * active[:, None]).sum(
                    axis=(1, 2)
                ) / pair_denominator
                distinct_endpoint_distance += (pair_distance[:, -1] * active).sum(
                    axis=1
                ) / active.sum(axis=1).clip(min=1)
                distinct_pair_count += 1
    endpoint_real = (target_positions[:, -1] - target_positions[:, 0])[active]
    endpoint_generated = (sample_positions[:, :, -1] - sample_positions[:, :, 0])[
        np.broadcast_to(active[:, None], sample_positions.shape[:2] + active.shape[1:])
    ]
    result = {
        "sequences": int(len(target_positions)),
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
        "energy_score_m": float(
            (first_term.mean(axis=1) - 0.5 * second_term / pair_count).mean()
        ),
        "coordinate_CRPS_m": float(
            (
                coordinate_first.mean(axis=1) - 0.5 * coordinate_second / pair_count
            ).mean()
        ),
        "endpoint_dx_wasserstein_m": _wasserstein(
            endpoint_real[:, 0], endpoint_generated[:, 0]
        ),
        "endpoint_dy_wasserstein_m": _wasserstein(
            endpoint_real[:, 1], endpoint_generated[:, 1]
        ),
        "horizons": horizons,
    }
    if deterministic_positions is not None:
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


@torch.no_grad()
def evaluate_background_diffusion(
    config: dict[str, Any],
    *,
    config_dir: Path,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    model_output = Path(config["paths"]["output_dir"])
    output = ensure_dir(evaluation.get("output_dir", model_output))
    device = select_device(evaluation.get("device", "auto"))
    set_seed(int(evaluation.get("seed", 42)))
    checkpoint_path = (
        checkpoint or model_output / "checkpoints/best_background_diffusion.pt"
    )
    model, state = load_checkpoint(checkpoint_path, device=device)
    model.eval()
    bundle = load_data_bundle(config, config_dir)
    contract = state["dataset_contract"]
    rows = pilot_rows(
        bundle,
        str(evaluation.get("split", "test")),
        maximum=int(evaluation.get("max_sequences", 4096)),
        seed=int(evaluation.get("seed", 42)),
    )
    row_index_path = evaluation.get("row_index_path")
    if row_index_path:
        allowed = np.unique(np.load(row_index_path)["sequence_row"])
        rows = rows[np.isin(rows, allowed)]
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
    guidance_scale = float(evaluation.get("guidance_scale", 1.0))
    sampling_clip = evaluation.get("x0_clip_abs")
    generated_batches: list[np.ndarray] = []
    deterministic_batches: list[np.ndarray] = []
    reference_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    active_batches: list[np.ndarray] = []
    tail_batches: list[np.ndarray] = []
    semantic_cutin_batches: list[np.ndarray] = []
    row_batches: list[np.ndarray] = []
    generated_action_batches: list[np.ndarray] = []
    target_action_batches: list[np.ndarray] = []
    c0_batches: list[np.ndarray] = []
    ego_future_batches: list[np.ndarray] = []
    generated_risk_batches: list[dict[str, np.ndarray]] = []
    target_risk_batches: list[dict[str, np.ndarray]] = []
    velocity_error_totals = _new_component_error_totals(2)
    generator = (
        torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
    )
    generator.manual_seed(int(evaluation.get("seed", 42)))
    for raw_batch in loader:
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
            x0_clip_abs=sampling_clip,
            guidance_scale=guidance_scale,
        )
        deterministic_normalized = model.sample_ddim(
            condition,
            target_mask,
            inference_steps=inference_steps,
            initial_noise=torch.zeros_like(target_mask, dtype=condition.dtype),
            x0_clip_abs=sampling_clip,
            guidance_scale=guidance_scale,
        )
        active = raw_batch["target_mask"][:, 0].numpy().reshape(size, 6, 2)[:, :, 0]
        residual_mean = np.asarray(contract["position_residual"]["mean"], np.float32)
        residual_std = np.asarray(contract["position_residual"]["std"], np.float32)
        residual = normalized.cpu().numpy().reshape(size, draws, 149, 6, 2)
        residual = residual * residual_std + residual_mean
        residual = smooth_position_residual(residual)
        reference_positions = raw_batch["trajectory_reference"].numpy()
        reference_batches.append(reference_positions)
        generated = (reference_positions[:, None] + residual) * active[
            :, None, None, :, None
        ]
        c0 = raw_batch["c0_states"].numpy()[:, 1:]
        c0_batches.append(raw_batch["c0_states"].numpy())
        generated_states = states_from_smooth_positions(
            np.broadcast_to(c0[:, None], (size, draws, 6, 6)), generated
        )
        deterministic_residual = (
            deterministic_normalized.cpu().numpy().reshape(size, 149, 6, 2)
        )
        deterministic_residual = deterministic_residual * residual_std + residual_mean
        deterministic_residual = smooth_position_residual(deterministic_residual)
        deterministic = (reference_positions + deterministic_residual) * active[
            :, None, :, None
        ]
        full_target = raw_batch["future_states"].numpy()
        target = full_target[:, :, 1:]
        _accumulate_component_errors(
            velocity_error_totals,
            generated_states[..., 2:4],
            target[..., 2:4],
            active,
        )
        generated_action_batches.append(generated_states[..., 4:6])
        target_actions = raw_batch["target_actions"].numpy()
        target_action_batches.append(target_actions)
        ego_future = full_target[:, None, :, 0]
        ego_future_batches.append(full_target[:, :, 0])
        generated_risk_batches.append(
            _risk_variables(generated_states, ego_future, active)
        )
        target_risk_batches.append(_risk_variables(target[:, None], ego_future, active))
        generated_batches.append(generated)
        deterministic_batches.append(deterministic)
        target_batches.append(target[..., 0:2])
        active_batches.append(active)
        tail_batches.append(raw_batch["is_evt_tail"].numpy())
        semantic_cutin_batches.append(raw_batch["semantic_cutin_mask"].numpy())
        row_batches.append(raw_batch["row_index"].numpy())
    samples = np.concatenate(generated_batches)
    deterministic = np.concatenate(deterministic_batches)
    references = np.concatenate(reference_batches)
    targets = np.concatenate(target_batches)
    active = np.concatenate(active_batches).astype(bool)
    tail = np.concatenate(tail_batches).astype(bool)
    semantic_cutin = np.concatenate(semantic_cutin_batches).astype(bool)
    row_index = np.concatenate(row_batches)
    metrics: dict[str, Any] = {
        "all": _cohort_metrics(samples, targets, active, deterministic),
        "sparse_constraint_reference": _cohort_metrics(
            references[:, None], targets, active, references
        ),
        "velocity_reconstruction": _component_error_metrics(
            velocity_error_totals, ("vx_mps", "vy_mps")
        ),
    }
    cohort_dir = config.get("paths", {}).get("cutin_cohort_dir")
    if cohort_dir and (Path(cohort_dir) / "index.npz").exists():
        metrics["crossing_aligned_cut_in"] = _crossing_aligned_cutin_metrics(
            samples,
            targets,
            deterministic,
            row_index,
            Path(cohort_dir) / "index.npz",
        )
    if generated_action_batches:
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
        c0_states = np.concatenate(c0_batches)
        ego_future = np.concatenate(ego_future_batches)
        metrics["action_distribution_fidelity"] = _action_fidelity(
            generated_actions, target_actions, active
        )
        metrics["physical_feasibility"] = _physical_fidelity(
            generated_actions, target_actions, c0_states[:, 1:, 2], active
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
                c0_states[following_rows, 1:, 2],
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
                c0_states[cutin_rows, 1:, 2],
                cutin_active[cutin_rows],
            )
            metrics["cut_in_semantic_fidelity"] = _cutin_semantics(
                samples,
                targets,
                c0_states[:, 1:, 0:2],
                c0_states[:, 0, 0:2],
                ego_future,
                cutin_active,
            )
        semantic_cutin_rows_mask = semantic_cutin.any(axis=1)
        if semantic_cutin_rows_mask.any():
            metrics["semantic_cut_in"] = _cohort_metrics(
                samples[semantic_cutin_rows_mask],
                targets[semantic_cutin_rows_mask],
                semantic_cutin[semantic_cutin_rows_mask],
                deterministic[semantic_cutin_rows_mask],
            )
            metrics["semantic_cut_in_fidelity"] = _cutin_semantics(
                samples,
                targets,
                c0_states[:, 1:, 0:2],
                c0_states[:, 0, 0:2],
                ego_future,
                semantic_cutin,
                lane_half_width_m=1.0,
            )
        generated_risk = {
            key: np.concatenate([batch[key] for batch in generated_risk_batches])
            for key in generated_risk_batches[0]
        }
        target_risk = {
            key: np.concatenate([batch[key] for batch in target_risk_batches])
            for key in target_risk_batches[0]
        }
        metrics["risk_distribution_fidelity"] = _risk_fidelity(
            generated_risk, target_risk
        )
    for name, mask in (("evt_tail", tail), ("non_tail", ~tail)):
        if mask.any():
            metrics[name] = _cohort_metrics(
                samples[mask], targets[mask], active[mask], deterministic[mask]
            )
    summary = {
        "model": contract.get("name", "highd_background_trajectory_diffusion"),
        "role": "state_knot_conditioned_open_loop_diffusion",
        "checkpoint": "checkpoints/best_background_diffusion.pt",
        "checkpoint_epoch": int(state["epoch"]),
        "experiment_scope": (
            "full"
            if int(evaluation.get("max_sequences", 0)) <= 0
            and int(config.get("dataset", {}).get("max_train_sequences", 0)) <= 0
            and int(config.get("dataset", {}).get("max_val_sequences", 0)) <= 0
            else "pilot"
        ),
        "ego_future_in_condition": bool(contract.get("ego_future_in_condition", True)),
        "target_representation": contract.get("target_representation"),
        "trajectory_reference": contract.get("trajectory_reference"),
        "evaluation_scope": "oracle_sparse_state_knot_conditioned_reconstruction",
        "end_to_end_plan_sampling_evaluated": False,
        "reference_aligned_horizons": {
            "car_following_seconds": 5.0,
            "cut_in_seconds": 4.0,
            "joint_world_seconds": 5.96,
        },
        "samples_per_condition": draws,
        "ddim_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "architecture": model.config.denoiser,
        "prediction_type": model.config.prediction_type,
        "x0_clip_abs": (
            model.config.x0_clip_abs if sampling_clip is None else float(sampling_clip)
        ),
        "split": str(evaluation.get("split", "test")),
        "metrics": metrics,
    }
    summary["condition_disclosure"] = {
        "uses_future_background_state_knots": True,
        "knot_times_s": [2.0, 4.0, 5.96],
        "uses_future_ego": False,
        "interpretation": (
            "conditional trajectory reconstruction, not C0-only future prediction"
        ),
        "endpoint_is_conditioned": True,
    }
    action_ks = [
        item["KS"] for item in metrics.get("action_distribution_fidelity", {}).values()
    ]
    summary["headline_metrics"] = {
        "ADE_m": float(metrics["all"]["ensemble_mean_ADE_m"]),
        "FDE_m": float(metrics["all"]["ensemble_mean_FDE_m"]),
        "distribution_similarity": (
            float(1.0 - np.mean(action_ks)) if action_ks else float("nan")
        ),
    }
    historical = {
        "ADE_m": 0.059,
        "FDE_m": 0.103,
        "status": (
            "non-comparable reference from the removed 14,295-sequence protocol; "
            "not a reproduced result on the current split"
        ),
    }
    generated_physical = metrics.get("physical_feasibility", {}).get("generated", {})
    summary["reconstruction_gate"] = {
        "historical_target": historical,
        "ADE_passed": bool(
            summary["headline_metrics"]["ADE_m"] <= float(historical["ADE_m"])
        ),
        "FDE_passed": bool(
            summary["headline_metrics"]["FDE_m"] <= float(historical["FDE_m"])
        ),
        "physical_feasibility_passed": bool(
            generated_physical
            and all(float(value) == 0.0 for value in generated_physical.values())
        ),
    }
    save_json(summary, output / "evaluation_summary.json")
    sample_ade, sample_fde = _errors(samples, targets, active)
    np.savez_compressed(
        output / "evaluation_per_sequence.npz",
        row_index=row_index,
        is_evt_tail=tail,
        active_background=active,
        sample_ade_m=sample_ade,
        sample_fde_m=sample_fde,
        target_endpoint=targets[:, -1],
        generated_endpoint=samples[:, :, -1],
        zero_latent_endpoint=deterministic[:, -1],
    )
    return summary

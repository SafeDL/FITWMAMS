"""Read-only projection of the canonical highD sequence cache."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader, Dataset

from normalizing_flow.src.data import load_natural_dataset
from normalizing_flow.src.constraints import derived_modes
from normalizing_flow.src.features import (
    SLOT_NAMES,
    feature_index,
    feature_valid_from_slot_mask,
)
from normalizing_flow.src.sampling import normalize_features

from world_model.src.core.sequential_dataset import (
    load_sequential_dataset,
    sequence_cache_owner_dir,
)
from world_model.src.core.utils import load_json
from world_model.src.hiqr.data import cohort_manifest

ANCHOR_INDEX = 24
HORIZON_STEPS = 149
BACKGROUND_AGENTS = 6
STATE_AGENTS = 1 + BACKGROUND_AGENTS
C0_FEATURE_DIM = 40
MASK_DIM = 6
CONSTRAINT_AGENTS = BACKGROUND_AGENTS
CONSTRAINT_KNOT_INDICES = (50, 100, 149)
CONSTRAINT_FEATURES = tuple(
    f"{field}_{time_name}"
    for time_name in ("2s", "4s", "end")
    for field in ("dx_m", "dy_left_m", "dvx_mps", "dvy_left_mps")
)
CONSTRAINT_FEATURE_DIM = len(CONSTRAINT_FEATURES)
CONDITION_DIM = C0_FEATURE_DIM + MASK_DIM + CONSTRAINT_AGENTS * CONSTRAINT_FEATURE_DIM
MOTION_SMOOTH_WINDOW = 41
MOTION_SMOOTH_POLYORDER = 3
SPLIT_INDEX = {"train": 0, "val": 1, "test": 2}


@dataclass(frozen=True)
class DataBundle:
    arrays: dict[str, np.ndarray]
    state_mean: np.ndarray
    state_std: np.ndarray
    cohort_manifest: dict[str, Any]
    flow_arrays: dict[str, np.ndarray]
    flow_schema: dict[str, Any]
    flow_row_for_sequence: np.ndarray


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _state_normalization(
    arrays: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Fit state statistics on valid states from the canonical train split."""
    train = np.flatnonzero(np.asarray(arrays["split_index"]) == SPLIT_INDEX["train"])
    total = np.zeros(6, np.float64)
    square = np.zeros(6, np.float64)
    count = 0
    for start in range(0, len(train), 1024):
        rows = train[start : start + 1024]
        states = np.asarray(arrays["agent_states"][rows], np.float64)
        valid = np.asarray(arrays["agent_valid"][rows], bool)
        values = states[valid]
        total += values.sum(axis=0)
        square += np.square(values).sum(axis=0)
        count += len(values)
    if count == 0:
        raise RuntimeError("canonical train split contains no valid states")
    mean = total / count
    std = np.sqrt(np.maximum(square / count - np.square(mean), 1.0e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def load_data_bundle(config: dict[str, Any], config_dir: Path) -> DataBundle:
    cache_owner = sequence_cache_owner_dir(config, config_dir=config_dir)
    sequence_manifest = Path(cache_owner) / "sequence_cache/manifest.json"
    manifest = load_json(sequence_manifest)
    if not bool(manifest.get("lateral_event_integrity_required", False)):
        raise RuntimeError(
            f"{sequence_manifest} predates lateral-event integrity screening; "
            "rebuild the canonical highD sequence cache"
        )
    if not bool(manifest.get("background_slot_stability_required", False)):
        raise RuntimeError(
            f"{sequence_manifest} predates background-slot stability screening; "
            "rebuild natural data and the canonical highD sequence cache"
        )
    arrays, manifest = load_sequential_dataset(cache_owner)
    flow_schema_path = _resolve(config["paths"]["source_schema"], config_dir)
    flow_arrays, flow_schema = load_natural_dataset(flow_schema_path.parent)
    sequence_ids = np.asarray(arrays["sequence_id"]).astype(str)
    flow_ids = np.asarray(flow_arrays["segment_id"]).astype(str)
    if len(np.unique(sequence_ids)) != len(sequence_ids) or len(
        np.unique(flow_ids)
    ) != len(flow_ids):
        raise RuntimeError("sequence and Flow identifiers must be unique")
    flow_lookup = {identifier: index for index, identifier in enumerate(flow_ids)}
    try:
        flow_row = np.asarray([flow_lookup[item] for item in sequence_ids], np.int64)
    except KeyError as exc:
        raise RuntimeError(
            f"canonical sequence is absent from Flow data: {exc}"
        ) from exc
    if len(flow_lookup) != len(flow_row) or set(flow_ids) != set(sequence_ids):
        raise RuntimeError(
            "canonical sequences and scenario Flow rows must have identical sets"
        )
    if not np.array_equal(
        np.asarray(arrays["split_index"]),
        np.asarray(flow_arrays["split_index"])[flow_row],
    ):
        raise RuntimeError(
            "canonical and Flow recording-level split assignments differ"
        )
    manifest = {**manifest, "hiqr_cohort": cohort_manifest(arrays)}
    state_mean, state_std = _state_normalization(arrays)
    return DataBundle(
        arrays=arrays,
        state_mean=state_mean,
        state_std=state_std,
        cohort_manifest=manifest["hiqr_cohort"],
        flow_arrays=flow_arrays,
        flow_schema=flow_schema,
        flow_row_for_sequence=flow_row,
    )


def split_rows(
    arrays: dict[str, np.ndarray],
    split: str,
    *,
    maximum: int = 0,
    seed: int = 0,
) -> np.ndarray:
    if split not in SPLIT_INDEX:
        raise ValueError(f"unknown split {split!r}")
    rows = np.flatnonzero(np.asarray(arrays["split_index"]) == SPLIT_INDEX[split])
    np.random.default_rng(int(seed)).shuffle(rows)
    return rows[: int(maximum)] if maximum > 0 else rows


def pilot_rows(
    bundle: DataBundle,
    split: str,
    *,
    maximum: int,
    seed: int,
) -> np.ndarray:
    """Fixed-size deterministic pilot stratified by lane change, mask and EVT."""
    candidates = split_rows(bundle.arrays, split, seed=seed)
    if maximum <= 0 or len(candidates) <= maximum:
        return candidates
    flow_rows = bundle.flow_row_for_sequence[candidates]
    flow = bundle.flow_arrays
    modes = derived_modes(
        np.asarray(flow["trajectory_constraint"])[flow_rows],
        np.asarray(flow["slot_mask"])[flow_rows],
    )
    lane_change = (modes[..., 1] != 0).any(1)
    patterns = np.asarray(flow["mask_pattern"])[flow_rows]
    evt = np.asarray(bundle.arrays["is_evt_tail"])[candidates].astype(bool)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for row, is_lane, pattern, is_evt in zip(candidates, lane_change, patterns, evt):
        groups.setdefault((int(is_lane), int(pattern), int(is_evt)), []).append(
            int(row)
        )
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    leftovers: list[int] = []
    for key in sorted(groups):
        values = np.asarray(groups[key], np.int64)
        rng.shuffle(values)
        selected.append(int(values[0]))
        leftovers.extend(values[1:].tolist())
    if len(selected) > maximum:
        selected = rng.choice(np.asarray(selected), maximum, replace=False).tolist()
    elif len(selected) < maximum:
        remaining = np.asarray(leftovers, np.int64)
        rng.shuffle(remaining)
        selected.extend(remaining[: maximum - len(selected)].tolist())
    output = np.asarray(selected, np.int64)
    rng.shuffle(output)
    return output


def condition_vector(
    c0: np.ndarray,
    slot_mask: np.ndarray,
    trajectory_constraint_normalized: np.ndarray,
) -> np.ndarray:
    """Encode 40-D C0, slot mask and three long-horizon state knots."""
    initial = np.asarray(c0, np.float32).reshape(C0_FEATURE_DIM)
    slots = np.asarray(slot_mask, bool).reshape(CONSTRAINT_AGENTS)
    constraint = np.asarray(trajectory_constraint_normalized, np.float32).reshape(
        CONSTRAINT_AGENTS, CONSTRAINT_FEATURE_DIM
    )
    constraint *= slots[:, None]
    return np.concatenate((initial, slots.astype(np.float32), constraint.reshape(-1)))


def c0_states_from_physical_features(
    c0: np.ndarray,
    slot_mask: np.ndarray,
) -> np.ndarray:
    """Restore ego and six background local states from the physical 40-D C0."""
    values = np.asarray(c0, np.float32).reshape(C0_FEATURE_DIM)
    slots = np.asarray(slot_mask, bool).reshape(BACKGROUND_AGENTS)
    states = np.zeros((STATE_AGENTS, 6), np.float32)
    states[0, 2:] = values[
        [
            feature_index(None, "ego_vx_mps"),
            feature_index(None, "ego_vy_left_mps"),
            feature_index(None, "ego_ax_mps2"),
            feature_index(None, "ego_ay_left_mps2"),
        ]
    ]
    for slot, name in enumerate(SLOT_NAMES):
        if not slots[slot]:
            continue
        states[slot + 1] = (
            values[feature_index(name, "rel_x_m")],
            values[feature_index(name, "rel_y_left_m")],
            states[0, 2] + values[feature_index(name, "rel_vx_mps")],
            states[0, 3] + values[feature_index(name, "rel_vy_left_mps")],
            values[feature_index(name, "other_ax_mps2")],
            values[feature_index(name, "other_ay_left_mps2")],
        )
    return states


def prepare_external_condition(
    c0: np.ndarray,
    slot_mask: np.ndarray,
    trajectory_constraint: np.ndarray,
    *,
    flow_schema: dict[str, Any],
    diffusion_contract: dict[str, Any],
    inactive_seed: int | None = None,
    inactive_reference_normalized: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build the exact diffusion inputs from physical external quantities.

    ``c0`` may come from the scenario Flow or another scenario source.
    ``trajectory_constraint`` supplies the three background state knots.  The
    direct Flow workflow returns this ``[6, 12]`` array from
    ``p(K|C0,M)``.  The Flow
    density uses a standard-normal
    reference measure on undefined C0 coordinates.  External callers must
    either seed that nuisance reference or provide its normalized values.
    """
    values = np.asarray(c0, np.float32).reshape(C0_FEATURE_DIM)
    slots = np.asarray(slot_mask, bool).reshape(BACKGROUND_AGENTS)
    constraint = np.asarray(trajectory_constraint, np.float32)
    if constraint.shape != (BACKGROUND_AGENTS, CONSTRAINT_FEATURE_DIM):
        raise ValueError("trajectory_constraint must have shape [6, 12]")
    valid = feature_valid_from_slot_mask(flow_schema, slots)[0]
    if not np.isfinite(values[valid]).all():
        raise ValueError("active physical C0 features must be finite")
    if not np.isfinite(constraint[slots]).all():
        raise ValueError("active trajectory state knots must be finite")
    c0_normalized = normalize_features(values[None], valid[None], flow_schema)[0]
    invalid = ~valid
    if invalid.any():
        if inactive_reference_normalized is not None:
            reference_values = np.asarray(
                inactive_reference_normalized, np.float32
            ).reshape(C0_FEATURE_DIM)
            if not np.isfinite(reference_values[invalid]).all():
                raise ValueError("inactive normalized reference must be finite")
            c0_normalized[invalid] = reference_values[invalid]
        elif inactive_seed is not None:
            generator = np.random.default_rng(int(inactive_seed))
            c0_normalized[invalid] = generator.standard_normal(
                int(invalid.sum())
            ).astype(np.float32)
        else:
            raise ValueError(
                "inactive_seed or inactive_reference_normalized is required "
                "when the slot mask contains absent vehicles"
            )
    mean = np.asarray(diffusion_contract["trajectory_constraint"]["mean"], np.float32)
    std = np.asarray(diffusion_contract["trajectory_constraint"]["std"], np.float32)
    constraint_normalized = (constraint - mean) / std
    constraint_normalized[~slots] = 0.0
    c0_states = c0_states_from_physical_features(values, slots)
    reference = trajectory_reference_positions(c0_states[1:], constraint)
    target_mask = (
        np.broadcast_to(slots[None, :, None], (HORIZON_STEPS, BACKGROUND_AGENTS, 2))
        .reshape(HORIZON_STEPS, -1)
        .copy()
    )
    return {
        "condition": condition_vector(c0_normalized, slots, constraint_normalized),
        "target_mask": target_mask,
        "c0_states": c0_states,
        "trajectory_constraint": constraint.copy(),
        "trajectory_reference": reference,
    }


def prepare_flow_condition(
    sample: Any,
    index: int,
    *,
    flow_schema: dict[str, Any],
    diffusion_contract: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Convert one complete Flow sample into the exact diffusion input."""
    reference = np.asarray(sample.c0_normalized_reference)[int(index)]
    return prepare_external_condition(
        np.asarray(sample.c0)[int(index)],
        np.asarray(sample.slot_mask)[int(index)],
        np.asarray(sample.trajectory_constraint)[int(index)],
        flow_schema=flow_schema,
        diffusion_contract=diffusion_contract,
        inactive_reference_normalized=reference,
    )


def extract_trajectory_constraint(states: np.ndarray) -> np.ndarray:
    """Extract relative position/velocity at 2 s, 4 s and 5.96 s."""
    values = np.asarray(states, np.float32)
    if values.shape[-3:] != (HORIZON_STEPS + 1, BACKGROUND_AGENTS, 6):
        raise ValueError("states must end with [150, 6, 6]")
    initial_position = values[..., :1, :, :2]
    initial_velocity = values[..., :1, :, 2:4]
    fields = []
    for index in CONSTRAINT_KNOT_INDICES:
        fields.extend(
            (
                values[..., index, :, 0:1] - initial_position[..., 0, :, 0:1],
                values[..., index, :, 1:2] - initial_position[..., 0, :, 1:2],
                values[..., index, :, 2:3] - initial_velocity[..., 0, :, 0:1],
                values[..., index, :, 3:4] - initial_velocity[..., 0, :, 1:2],
            )
        )
    return np.concatenate(fields, axis=-1).astype(np.float32)


def trajectory_reference_positions(
    c0_background_states: np.ndarray,
    constraint: np.ndarray,
    *,
    dt_s: float = 0.04,
) -> np.ndarray:
    """Interpolate an absolute Cartesian path through declared state knots."""
    initial = np.asarray(c0_background_states, np.float32)
    values = np.asarray(constraint, np.float32)
    leading = values.shape[:-2]
    if initial.shape != leading + (BACKGROUND_AGENTS, 6):
        raise ValueError("C0 states and trajectory constraints do not align")
    if values.shape[-2:] != (BACKGROUND_AGENTS, CONSTRAINT_FEATURE_DIM):
        raise ValueError("trajectory constraint must end with [6, 12]")
    knot_frames = np.asarray((0, *CONSTRAINT_KNOT_INDICES), np.float32)
    knot_time = knot_frames * float(dt_s)
    knot_position = np.empty(leading + (4, BACKGROUND_AGENTS, 2), np.float32)
    knot_velocity = np.empty_like(knot_position)
    knot_position[..., 0, :, :] = initial[..., :, :2]
    knot_velocity[..., 0, :, :] = initial[..., :, 2:4]
    for knot in range(3):
        offset = 4 * knot
        knot_position[..., knot + 1, :, 0] = initial[..., :, 0] + values[..., :, offset]
        knot_position[..., knot + 1, :, 1] = (
            initial[..., :, 1] + values[..., :, offset + 1]
        )
        knot_velocity[..., knot + 1, :, 0] = (
            initial[..., :, 2] + values[..., :, offset + 2]
        )
        knot_velocity[..., knot + 1, :, 1] = (
            initial[..., :, 3] + values[..., :, offset + 3]
        )
    time = np.arange(1, HORIZON_STEPS + 1, dtype=np.float32) * float(dt_s)
    output = np.empty(leading + (HORIZON_STEPS, BACKGROUND_AGENTS, 2), np.float32)
    for interval in range(3):
        selected = (time > knot_time[interval]) & (
            time <= knot_time[interval + 1] + 1.0e-6
        )
        duration = knot_time[interval + 1] - knot_time[interval]
        ratio = (time[selected] - knot_time[interval]) / duration
        h00 = 2 * ratio**3 - 3 * ratio**2 + 1
        h10 = ratio**3 - 2 * ratio**2 + ratio
        h01 = -2 * ratio**3 + 3 * ratio**2
        h11 = ratio**3 - ratio**2
        shape = (1,) * len(leading) + (len(ratio), 1, 1)
        output[..., selected, :, :] = (
            h00.reshape(shape) * knot_position[..., interval, :, :][..., None, :, :]
            + h10.reshape(shape)
            * duration
            * knot_velocity[..., interval, :, :][..., None, :, :]
            + h01.reshape(shape)
            * knot_position[..., interval + 1, :, :][..., None, :, :]
            + h11.reshape(shape)
            * duration
            * knot_velocity[..., interval + 1, :, :][..., None, :, :]
        )
    return output


def smooth_position_residual(residual: np.ndarray) -> np.ndarray:
    """Project residual motion onto the selected low-frequency cubic basis."""
    values = np.asarray(residual, np.float32)
    time_axis = values.ndim - 3
    if values.shape[time_axis] != HORIZON_STEPS:
        raise ValueError("position residual must contain 149 future frames")
    return savgol_filter(
        values,
        MOTION_SMOOTH_WINDOW,
        MOTION_SMOOTH_POLYORDER,
        axis=time_axis,
        mode="interp",
    ).astype(np.float32)


def states_from_smooth_positions(
    c0_background_states: np.ndarray,
    positions: np.ndarray,
    *,
    dt_s: float = 0.04,
) -> np.ndarray:
    """Complete generated positions with derivatives of the same cubic basis."""
    initial = np.asarray(c0_background_states, np.float32)
    values = np.asarray(positions, np.float32)
    leading = values.shape[:-3]
    if initial.shape != leading + (BACKGROUND_AGENTS, 6):
        raise ValueError("C0 states and generated positions do not align")
    full_positions = np.concatenate((initial[..., None, :, :2], values), axis=-3)
    output = np.empty(values.shape[:-1] + (6,), np.float32)
    output[..., :2] = values
    for derivative, target_slice in ((1, slice(2, 4)), (2, slice(4, 6))):
        derived = savgol_filter(
            full_positions,
            MOTION_SMOOTH_WINDOW,
            MOTION_SMOOTH_POLYORDER,
            deriv=derivative,
            delta=dt_s,
            axis=-3,
            mode="interp",
        )
        output[..., target_slice] = derived[..., 1:, :, :]
    return output


def semantic_cutin_events(
    states: np.ndarray,
    valid: np.ndarray,
    *,
    lane_overlap_m: float = 1.0,
    initial_lane_offset_m: float = 1.8,
    minimum_post_steps: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Return strict modeled-world cut-in masks and lane-crossing indices."""
    values = np.asarray(states, dtype=np.float32)
    present = np.asarray(valid, dtype=bool)
    single = values.ndim == 3
    if single:
        values = values[None]
        present = present[None]
    if values.ndim != 4 or values.shape[2:] != (STATE_AGENTS, 6):
        raise ValueError("states must have shape [batch, time, 7, 6]")
    if present.shape != values.shape[:3]:
        raise ValueError("validity must match states")
    relative_y = values[:, :, 1:, 1] - values[:, :, :1, 1]
    relative_x = values[:, :, 1:, 0] - values[:, :, :1, 0]
    overlap = np.abs(relative_y) <= float(lane_overlap_m)
    entered = overlap.any(axis=1)
    first_entry = np.argmax(overlap, axis=1)
    frame = np.arange(values.shape[1])[None, :, None]
    after_entry = frame >= first_entry[:, None]
    retained = (overlap & after_entry).sum(axis=1) / after_entry.sum(axis=1) >= 0.7
    front_at_entry = (
        np.take_along_axis(relative_x, first_entry[:, None], axis=1)[:, 0] > 0.0
    )
    approached = np.min(np.abs(relative_y), axis=1) < (np.abs(relative_y[:, 0]) - 0.5)
    enough_post = first_entry <= values.shape[1] - int(minimum_post_steps)
    target_x = relative_x[:, :, :, None]
    other_x = relative_x[:, :, None, :]
    other_valid = present[:, :, None, 1:]
    candidate = np.arange(BACKGROUND_AGENTS)[None, None, :, None]
    other = np.arange(BACKGROUND_AGENTS)[None, None, None, :]
    post_window = (frame >= first_entry[:, None]) & (
        frame < first_entry[:, None] + int(minimum_post_steps)
    )
    blocked = (
        (other_x > 0.0)
        & (other_x < target_x)
        & other_valid
        & (candidate != other)
        & post_window[..., None]
    ).any(axis=(1, 3))
    semantic = (
        present.all(axis=1)[:, 1:]
        & (np.abs(relative_y[:, 0]) > float(initial_lane_offset_m))
        & entered
        & retained
        & overlap[:, -1]
        & front_at_entry
        & approached
        & enough_post
        & ~blocked
    )
    crossings = np.where(semantic, first_entry, -1).astype(np.int16)
    if single:
        return semantic[0], crossings[0]
    return semantic, crossings


def semantic_cutin_agents(
    states: np.ndarray,
    valid: np.ndarray,
    **kwargs: Any,
) -> np.ndarray:
    """Identify strict completed front cut-ins in fixed windows."""
    return semantic_cutin_events(states, valid, **kwargs)[0]


def semantic_cutin_rows(
    arrays: dict[str, np.ndarray], rows: np.ndarray, *, chunk_size: int = 512
) -> np.ndarray:
    selected = np.asarray(rows, dtype=np.int64)
    mask = np.zeros(len(selected), dtype=bool)
    for start in range(0, len(selected), int(chunk_size)):
        take = selected[start : start + int(chunk_size)]
        states = np.asarray(arrays["agent_states"][take, ANCHOR_INDEX:174])
        valid = np.asarray(arrays["agent_valid"][take, ANCHOR_INDEX:174])
        mask[start : start + len(take)] = semantic_cutin_agents(states, valid).any(
            axis=1
        )
    return mask


def fit_constraint_statistics(
    bundle: DataBundle,
    rows: np.ndarray,
    *,
    chunk_size: int = 512,
) -> dict[str, Any]:
    """Fit train-only normalization for constraints and position residuals."""
    constraint_sum = np.zeros(CONSTRAINT_FEATURE_DIM, dtype=np.float64)
    constraint_squared = np.zeros(CONSTRAINT_FEATURE_DIM, dtype=np.float64)
    residual_sum = np.zeros(2, dtype=np.float64)
    residual_squared = np.zeros(2, dtype=np.float64)
    agent_count = point_count = 0
    arrays = bundle.arrays
    for start in range(0, len(rows), int(chunk_size)):
        selected = rows[start : start + int(chunk_size)]
        states = np.asarray(
            arrays["agent_states"][selected, ANCHOR_INDEX:174, 1:], dtype=np.float32
        )
        active = np.asarray(arrays["agent_valid"][selected, ANCHOR_INDEX, 1:], bool)
        constraint = extract_trajectory_constraint(states)
        reference = trajectory_reference_positions(states[:, 0], constraint)
        residual = smooth_position_residual(states[:, 1:, :, :2] - reference)
        constraint_values = constraint[active]
        residual_values = residual[
            np.broadcast_to(active[:, None], residual.shape[:-1])
        ]
        constraint_sum += constraint_values.sum(axis=0, dtype=np.float64)
        constraint_squared += np.square(constraint_values, dtype=np.float64).sum(axis=0)
        residual_sum += residual_values.sum(axis=0, dtype=np.float64)
        residual_squared += np.square(residual_values, dtype=np.float64).sum(axis=0)
        agent_count += len(constraint_values)
        point_count += len(residual_values)
    if agent_count == 0 or point_count == 0:
        raise RuntimeError("no valid rows for trajectory normalization")
    constraint_mean = constraint_sum / agent_count
    constraint_variance = np.maximum(
        constraint_squared / agent_count - np.square(constraint_mean), 1.0e-6
    )
    residual_mean = residual_sum / point_count
    residual_variance = np.maximum(
        residual_squared / point_count - np.square(residual_mean), 1.0e-6
    )
    return {
        "constraint": {
            "feature_names": list(CONSTRAINT_FEATURES),
            "mean": constraint_mean.astype(np.float32).tolist(),
            "std": np.sqrt(constraint_variance).astype(np.float32).tolist(),
            "count": int(agent_count),
        },
        "position_residual": {
            "layout": "local_dx_dy_left",
            "mean": residual_mean.astype(np.float32).tolist(),
            "std": np.sqrt(residual_variance).astype(np.float32).tolist(),
            "count": int(point_count),
        },
    }


def trajectory_data_contract(
    bundle: DataBundle,
    statistics: dict[str, Any],
) -> dict[str, Any]:
    cache_manifest = (
        Path(bundle.arrays["agent_states"].filename).parent / "manifest.json"
    )
    return {
        "name": "highd_joint_background_trajectory_diffusion",
        "condition": "shared 40-D C0, slot mask and 2s/4s/end state knots",
        "condition_mode": "c0_long_horizon_state_knots",
        "condition_dim": CONDITION_DIM,
        "target": "149-frame joint six-background smooth position residual",
        "target_representation": "smooth_reference_relative_dx_dy_residual",
        "trajectory_reference": "piecewise_cubic_hermite_2s_4s_end",
        "motion_basis": {
            "name": "savitzky_golay_cubic",
            "window_frames": MOTION_SMOOTH_WINDOW,
            "polyorder": MOTION_SMOOTH_POLYORDER,
        },
        "ego_future_in_condition": False,
        "constraint_agents": CONSTRAINT_AGENTS,
        "trajectory_constraint": statistics["constraint"],
        "motion_seed_controls_diffusion_noise": True,
        "horizon_steps": HORIZON_STEPS,
        "dt_s": 0.04,
        "target_dim": BACKGROUND_AGENTS * 2,
        "state_mean": bundle.state_mean.tolist(),
        "state_std": bundle.state_std.tolist(),
        "position_residual": statistics["position_residual"],
        "sequence_manifest_sha256": hashlib.sha256(
            cache_manifest.read_bytes()
        ).hexdigest(),
        "cohort": bundle.cohort_manifest,
    }


class BackgroundTrajectoryDataset(Dataset):
    """Six-agent position residuals conditioned on C0 and sparse state knots."""

    def __init__(
        self,
        bundle: DataBundle,
        rows: np.ndarray,
        contract: dict[str, Any],
    ) -> None:
        self.bundle = bundle
        self.rows = np.asarray(rows, dtype=np.int64)
        self.residual_mean = np.asarray(
            contract["position_residual"]["mean"], dtype=np.float32
        )
        self.residual_std = np.asarray(
            contract["position_residual"]["std"], dtype=np.float32
        )
        self.constraint_mean = np.asarray(
            contract["trajectory_constraint"]["mean"], dtype=np.float32
        )
        self.constraint_std = np.asarray(
            contract["trajectory_constraint"]["std"], dtype=np.float32
        )
        if bool(contract.get("ego_future_in_condition", True)):
            raise ValueError(
                "future ego information is forbidden in diffusion conditions"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = int(self.rows[int(item)])
        arrays = self.bundle.arrays
        states = np.asarray(
            arrays["agent_states"][row, ANCHOR_INDEX:174], dtype=np.float32
        ).copy()
        valid = np.asarray(
            arrays["agent_valid"][row, ANCHOR_INDEX:174], dtype=bool
        ).copy()
        flow = self.bundle.flow_arrays
        flow_row = int(self.bundle.flow_row_for_sequence[row])
        c0 = np.asarray(flow["features_normalized"][flow_row], np.float32)
        slots = np.asarray(flow["slot_mask"][flow_row], bool)
        constraint = extract_trajectory_constraint(states[:, 1:])
        constraint_normalized = (
            constraint - self.constraint_mean
        ) / self.constraint_std
        condition = condition_vector(c0, slots, constraint_normalized)
        active = valid[0, 1:]
        trajectory_reference = trajectory_reference_positions(states[0, 1:], constraint)
        residual = smooth_position_residual(states[1:, 1:, :2] - trajectory_reference)
        mask = np.broadcast_to(active[None, :, None], residual.shape)
        normalized = ((residual - self.residual_mean) / self.residual_std) * mask
        return {
            "condition": torch.from_numpy(condition),
            "target": torch.from_numpy(normalized.reshape(HORIZON_STEPS, -1).copy()),
            "target_mask": torch.from_numpy(mask.reshape(HORIZON_STEPS, -1).copy()),
            "c0_states": torch.from_numpy(states[0].copy()),
            "future_states": torch.from_numpy(states[1:].copy()),
            "trajectory_constraint": torch.from_numpy(constraint),
            "trajectory_reference": torch.from_numpy(trajectory_reference),
            "target_actions": torch.from_numpy(
                np.asarray(arrays["actions_highd"][row], np.float32).copy()
            ),
            "semantic_cutin_mask": torch.from_numpy(
                semantic_cutin_agents(states, valid).copy()
            ),
            "row_index": torch.tensor(row, dtype=torch.long),
            "is_evt_tail": torch.tensor(bool(arrays["is_evt_tail"][row])),
        }


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator if shuffle else None,
        num_workers=max(0, int(workers)),
        persistent_workers=int(workers) > 0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

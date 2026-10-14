"""Read-only projection of the canonical highD sequence cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader, Dataset

from hierarchical_world_model.src.core.evaluation_scope import (
    scoped_canonical_trajectory,
    scoped_slot_mask,
)
from hierarchical_world_model.src.core.sequential_dataset import (
    cohort_manifest,
    load_sequential_dataset,
    sequence_cache_owner_dir,
)
from normalizing_flow.src.constraints import KNOT_FEATURE_NAMES
from normalizing_flow.src.data import load_scenario_dataset
from normalizing_flow.src.features import (
    SLOT_NAMES,
    build_feature_schema,
    feature_index,
    feature_valid_from_slot_mask,
)
from normalizing_flow.src.transforms import transform_features_for_model

ANCHOR_INDEX = 24
HORIZON_STEPS = 149
BACKGROUND_AGENTS = 6
STATE_AGENTS = 1 + BACKGROUND_AGENTS
C0_FEATURE_DIM = len(build_feature_schema().feature_names)
MASK_DIM = 6
CONSTRAINT_AGENTS = BACKGROUND_AGENTS
CONSTRAINT_KNOT_INDICES = (50, 100, 149)
CONSTRAINT_FEATURES = KNOT_FEATURE_NAMES
CONSTRAINT_FEATURE_DIM = len(CONSTRAINT_FEATURES)
CONDITION_DIM = C0_FEATURE_DIM + MASK_DIM + CONSTRAINT_AGENTS * CONSTRAINT_FEATURE_DIM
MOTION_SMOOTH_WINDOW = 41
MOTION_SMOOTH_POLYORDER = 3
SPLIT_INDEX = {"train": 0, "val": 1, "test": 2}


@dataclass(frozen=True)
class DataBundle:
    arrays: dict[str, np.ndarray]
    cohort_manifest: dict[str, Any]
    flow_arrays: dict[str, np.ndarray]
    flow_schema: dict[str, Any]
    flow_row_for_sequence: np.ndarray


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_data_bundle(config: dict[str, Any], config_dir: Path) -> DataBundle:
    cache_owner = sequence_cache_owner_dir(config, config_dir=config_dir)
    sequence_manifest = Path(cache_owner) / "sequence_cache/manifest.json"
    arrays, manifest = load_sequential_dataset(cache_owner)
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
    flow_schema_path = _resolve(config["paths"]["source_schema"], config_dir)
    flow_arrays, flow_schema = load_scenario_dataset(flow_schema_path.parent)
    sequence_ids = np.asarray(arrays["sequence_id"]).astype(str)
    flow_ids = np.asarray(flow_arrays["segment_id"]).astype(str)
    if len(np.unique(sequence_ids)) != len(sequence_ids) or len(
        np.unique(flow_ids)
    ) != len(flow_ids):
        raise RuntimeError("sequence and Flow identifiers must be unique")
    flow_lookup = {identifier: index for index, identifier in enumerate(flow_ids)}
    if set(flow_ids) != set(sequence_ids):
        raise RuntimeError(
            "canonical sequences and scenario Flow rows must have identical sets"
        )
    flow_row = np.asarray([flow_lookup[item] for item in sequence_ids], np.int64)
    if not np.array_equal(
        np.asarray(arrays["split_index"]),
        np.asarray(flow_arrays["split_index"])[flow_row],
    ):
        raise RuntimeError(
            "canonical and Flow recording-level split assignments differ"
        )
    return DataBundle(
        arrays=arrays,
        cohort_manifest=cohort_manifest(arrays),
        flow_arrays=flow_arrays,
        flow_schema=flow_schema,
        flow_row_for_sequence=flow_row,
    )


def split_rows(arrays: dict[str, np.ndarray], split: str) -> np.ndarray:
    if split not in SPLIT_INDEX:
        raise ValueError(f"unknown split {split!r}")
    return np.flatnonzero(np.asarray(arrays["split_index"]) == SPLIT_INDEX[split])


def condition_vector(
    c0: np.ndarray,
    slot_mask: np.ndarray,
    trajectory_constraint_normalized: np.ndarray,
) -> np.ndarray:
    """Encode initial states, slot mask and sparse position knots."""
    initial = np.asarray(c0, np.float32).reshape(C0_FEATURE_DIM)
    slots = np.asarray(slot_mask, bool).reshape(CONSTRAINT_AGENTS)
    constraint = np.asarray(trajectory_constraint_normalized, np.float32).reshape(
        CONSTRAINT_AGENTS, CONSTRAINT_FEATURE_DIM
    )
    constraint *= slots[:, None]
    return np.concatenate(
        (
            initial,
            slots.astype(np.float32),
            constraint.reshape(-1),
        )
    )


def normalize_c0_features(
    c0: np.ndarray,
    slot_mask: np.ndarray,
    flow_schema: dict[str, Any],
    diffusion_contract: dict[str, Any],
) -> np.ndarray:
    """Apply train-only Diffusion normalization and zero absent slot features."""
    values = np.asarray(c0, np.float32).reshape(1, C0_FEATURE_DIM)
    slots = np.asarray(slot_mask, bool).reshape(1, BACKGROUND_AGENTS)
    valid = feature_valid_from_slot_mask(flow_schema, slots)
    if not np.isfinite(values[valid]).all():
        raise ValueError("active physical C0 features must be finite")
    transformed = transform_features_for_model(
        values,
        valid,
        diffusion_contract["c0_normalization"]["feature_transforms"],
    )
    mean = np.asarray(diffusion_contract["c0_normalization"]["mean"], np.float32)
    std = np.asarray(diffusion_contract["c0_normalization"]["std"], np.float32)
    normalized = np.zeros_like(transformed, dtype=np.float32)
    normalized[valid] = ((transformed - mean) / std)[valid]
    return normalized[0]


def c0_states_from_physical_features(
    c0: np.ndarray,
    slot_mask: np.ndarray,
) -> np.ndarray:
    """Restore initial states; acceleration is produced by the controllers."""
    values = np.asarray(c0, np.float32).reshape(C0_FEATURE_DIM)
    slots = np.asarray(slot_mask, bool).reshape(BACKGROUND_AGENTS)
    states = np.zeros((STATE_AGENTS, 6), np.float32)
    states[0, 2:4] = values[
        [
            feature_index(None, "ego_vx_mps"),
            feature_index(None, "ego_vy_left_mps"),
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
            0.0,
            0.0,
        )
    return states


def prepare_external_condition(
    c0: np.ndarray,
    slot_mask: np.ndarray,
    trajectory_constraint: np.ndarray,
    *,
    flow_schema: dict[str, Any],
    diffusion_contract: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Build the exact diffusion inputs from physical external quantities.

    ``c0`` may come from the scenario Flow or another scenario source.
    ``trajectory_constraint`` supplies three background position knots. The
    Flow returns this ``[6, 6]`` array from ``p(K|C0,M)``. Diffusion applies
    its own natural-Train normalization. Undefined slot
    features are zero and never inherit the Flow density's reference noise.
    """
    values = np.asarray(c0, np.float32).reshape(C0_FEATURE_DIM)
    slots = np.asarray(slot_mask, bool).reshape(BACKGROUND_AGENTS)
    constraint = np.asarray(trajectory_constraint, np.float32)
    if constraint.shape != (BACKGROUND_AGENTS, CONSTRAINT_FEATURE_DIM):
        raise ValueError("trajectory_constraint must have shape [6, 6]")
    if not np.isfinite(constraint[slots]).all():
        raise ValueError("active trajectory position knots must be finite")
    c0_normalized = normalize_c0_features(
        values, slots, flow_schema, diffusion_contract
    )
    mean = np.asarray(diffusion_contract["trajectory_constraint"]["mean"], np.float32)
    std = np.asarray(diffusion_contract["trajectory_constraint"]["std"], np.float32)
    constraint_normalized = (constraint - mean) / std
    constraint_normalized[~slots] = 0.0
    c0_states = c0_states_from_physical_features(values, slots)
    target_mask = (
        np.broadcast_to(slots[None, :, None], (HORIZON_STEPS, BACKGROUND_AGENTS, 2))
        .reshape(HORIZON_STEPS, -1)
        .copy()
    )
    return {
        "condition": condition_vector(c0_normalized, slots, constraint_normalized),
        "target_mask": target_mask,
        "c0_states": c0_states,
        "trajectory_reference": trajectory_reference_positions(
            c0_states[1:], constraint
        ),
    }


def prepare_flow_condition(
    sample: Any,
    index: int,
    *,
    flow_schema: dict[str, Any],
    diffusion_contract: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Convert one complete Flow sample into the exact diffusion input."""
    return prepare_external_condition(
        np.asarray(sample.c0)[int(index)],
        np.asarray(sample.slot_mask)[int(index)],
        np.asarray(sample.trajectory_constraint)[int(index)],
        flow_schema=flow_schema,
        diffusion_contract=diffusion_contract,
    )


def extract_trajectory_constraint(states: np.ndarray) -> np.ndarray:
    """Extract relative positions at 2 s, 4 s and 5.96 s."""
    values = np.asarray(states, np.float32)
    if values.shape[-3:] != (HORIZON_STEPS + 1, BACKGROUND_AGENTS, 6):
        raise ValueError("states must end with [150, 6, 6]")
    initial_position = values[..., :1, :, :2]
    fields = []
    for index in CONSTRAINT_KNOT_INDICES:
        fields.extend(
            (
                values[..., index, :, 0:1] - initial_position[..., 0, :, 0:1],
                values[..., index, :, 1:2] - initial_position[..., 0, :, 1:2],
            )
        )
    return np.concatenate(fields, axis=-1).astype(np.float32)


def trajectory_reference_positions(
    c0_background_states: np.ndarray,
    constraint: np.ndarray,
    *,
    dt_s: float = 0.04,
) -> np.ndarray:
    """Build a shape-preserving reference through the initial and three positions."""
    initial = np.asarray(c0_background_states, np.float32)
    values = np.asarray(constraint, np.float32)
    knot_positions = np.concatenate(
        (
            initial[..., None, :, :2],
            initial[..., None, :, :2]
            + values.reshape(values.shape[:-1] + (3, 2)).swapaxes(-3, -2),
        ),
        axis=-3,
    )
    knot_times = np.asarray((0, *CONSTRAINT_KNOT_INDICES), np.float32) * dt_s
    times = np.arange(1, HORIZON_STEPS + 1, dtype=np.float32) * dt_s
    return PchipInterpolator(knot_times, knot_positions, axis=-3)(times).astype(
        np.float32
    )


def trajectory_residual_target(
    positions: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Project the training residual once, retaining the declared position knots."""
    residual = savgol_filter(
        np.asarray(positions, np.float32) - reference,
        MOTION_SMOOTH_WINDOW,
        MOTION_SMOOTH_POLYORDER,
        axis=-3,
        mode="interp",
    ).astype(np.float32)
    residual[..., np.asarray(CONSTRAINT_KNOT_INDICES) - 1, :, :] = 0.0
    return residual


def decode_residual_trajectory(
    normalized: np.ndarray,
    reference: np.ndarray,
    active: np.ndarray,
    contract: dict[str, Any],
) -> np.ndarray:
    """Add the generated residual without re-smoothing; keep conditioned knots exact."""
    values = np.asarray(normalized, np.float32)
    residual = values.reshape(values.shape[:-2] + (HORIZON_STEPS, 6, 2))
    mean = np.asarray(contract["position_residual"]["mean"], np.float32)
    std = np.asarray(contract["position_residual"]["std"], np.float32)
    residual = residual * std + mean
    residual[..., np.asarray(CONSTRAINT_KNOT_INDICES) - 1, :, :] = 0.0
    slots = np.asarray(active, bool)[..., None, :, None]
    return (residual + np.asarray(reference, np.float32)) * slots


def states_from_smooth_positions(
    c0_background_states: np.ndarray,
    positions: np.ndarray,
    *,
    dt_s: float = 0.04,
) -> np.ndarray:
    """Estimate velocity/acceleration for diagnostics without changing positions."""
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


def semantic_cutin_agents(
    states: np.ndarray,
    valid: np.ndarray,
    *,
    lane_overlap_m: float = 1.0,
    initial_lane_offset_m: float = 1.8,
    minimum_post_steps: int = 50,
) -> np.ndarray:
    """Identify strict completed front cut-ins in fixed windows."""
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
    return semantic[0] if single else semantic


def fit_constraint_statistics(
    bundle: DataBundle,
    rows: np.ndarray,
    *,
    chunk_size: int = 512,
) -> dict[str, Any]:
    """Fit Train-only C0, knot and single-smoothed position-residual statistics."""
    flow_rows = bundle.flow_row_for_sequence[rows]
    c0 = np.asarray(bundle.flow_arrays["features"])[flow_rows]
    slots = np.asarray(bundle.flow_arrays["slot_mask"])[flow_rows]
    c0_valid = feature_valid_from_slot_mask(bundle.flow_schema, slots)
    c0_model = transform_features_for_model(
        c0,
        c0_valid,
        bundle.flow_schema["model_feature_transforms"],
    ).astype(np.float64)
    c0_count = c0_valid.sum(axis=0)
    if not np.all(c0_count > 0):
        raise RuntimeError("natural-driving Train has an unobserved C0 feature")
    c0_sum = np.where(c0_valid, c0_model, 0.0).sum(axis=0)
    c0_squared = np.where(c0_valid, np.square(c0_model), 0.0).sum(axis=0)
    c0_mean = c0_sum / c0_count
    c0_variance = np.maximum(c0_squared / c0_count - np.square(c0_mean), 1.0e-6)

    constraint_sum = np.zeros(CONSTRAINT_FEATURE_DIM, dtype=np.float64)
    constraint_squared = np.zeros(CONSTRAINT_FEATURE_DIM, dtype=np.float64)
    position_sum = np.zeros(2, dtype=np.float64)
    position_squared = np.zeros_like(position_sum)
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
        residual = trajectory_residual_target(states[:, 1:, :, :2], reference)
        constraint_values = constraint[active]
        constraint_sum += constraint_values.sum(axis=0, dtype=np.float64)
        constraint_squared += np.square(constraint_values, dtype=np.float64).sum(axis=0)
        position_values = residual.transpose(0, 2, 1, 3)[active].reshape(-1, 2)
        position_sum += position_values.sum(axis=0, dtype=np.float64)
        position_squared += np.square(position_values, dtype=np.float64).sum(axis=0)
        agent_count += len(constraint_values)
        point_count += len(constraint_values) * HORIZON_STEPS
    if agent_count == 0 or point_count == 0:
        raise RuntimeError("no valid rows for trajectory normalization")
    constraint_mean = constraint_sum / agent_count
    constraint_variance = np.maximum(
        constraint_squared / agent_count - np.square(constraint_mean), 1.0e-6
    )
    position_mean = position_sum / point_count
    position_variance = np.maximum(
        position_squared / point_count - np.square(position_mean), 1.0e-6
    )
    return {
        "c0": {
            "feature_names": list(bundle.flow_schema["feature_names"]),
            "feature_transforms": list(bundle.flow_schema["model_feature_transforms"]),
            "mean": c0_mean.astype(np.float32).tolist(),
            "std": np.sqrt(c0_variance).astype(np.float32).tolist(),
            "valid_train_count": c0_count.astype(np.int64).tolist(),
            "fit_split": "train",
            "inactive_slot_policy": "zero",
        },
        "constraint": {
            "feature_names": list(CONSTRAINT_FEATURES),
            "mean": constraint_mean.astype(np.float32).tolist(),
            "std": np.sqrt(constraint_variance).astype(np.float32).tolist(),
            "count": int(agent_count),
        },
        "position_residual": {
            "layout": "local_dx_dy_left",
            "mean": position_mean.astype(np.float32).tolist(),
            "std": np.sqrt(position_variance).astype(np.float32).tolist(),
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
        "condition": "26-D C0, 6-D mask and 36-D position knots",
        "condition_mode": "c0_position_knots",
        "condition_dim": CONDITION_DIM,
        "c0_dim": C0_FEATURE_DIM,
        "target": "149-frame joint six-background single-smoothed position residual",
        "target_representation": "single_smoothed_pchip_position_residual",
        "trajectory_reference": "shape_preserving_cubic_position_2s_4s_end",
        "motion_basis": {
            "name": "savitzky_golay_cubic",
            "window_frames": MOTION_SMOOTH_WINDOW,
            "polyorder": MOTION_SMOOTH_POLYORDER,
        },
        "ego_future_in_condition": False,
        "constraint_agents": CONSTRAINT_AGENTS,
        "c0_normalization": statistics["c0"],
        "trajectory_constraint": statistics["constraint"],
        "motion_seed_controls_diffusion_noise": True,
        "horizon_steps": HORIZON_STEPS,
        "dt_s": 0.04,
        "target_dim": BACKGROUND_AGENTS * 2,
        "position_residual": statistics["position_residual"],
        "sequence_manifest": str(cache_manifest),
        "sequence_manifest_mtime_ns": cache_manifest.stat().st_mtime_ns,
        "cohort": bundle.cohort_manifest,
    }


class BackgroundTrajectoryDataset(Dataset):
    """Six-agent residual trajectories conditioned on C0 and position knots."""

    def __init__(
        self,
        bundle: DataBundle,
        rows: np.ndarray,
        contract: dict[str, Any],
        *,
        evaluation_scope: bool = False,
    ) -> None:
        self.bundle = bundle
        self.rows = np.asarray(rows, dtype=np.int64)
        self.position_mean = np.asarray(
            contract["position_residual"]["mean"], np.float32
        )
        self.position_std = np.asarray(contract["position_residual"]["std"], np.float32)
        self.constraint_mean = np.asarray(
            contract["trajectory_constraint"]["mean"], dtype=np.float32
        )
        self.constraint_std = np.asarray(
            contract["trajectory_constraint"]["std"], dtype=np.float32
        )
        self.contract = contract
        self.evaluation_scope = bool(evaluation_scope)
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
        if self.evaluation_scope:
            states, valid = scoped_canonical_trajectory(states, valid)
        flow = self.bundle.flow_arrays
        flow_row = int(self.bundle.flow_row_for_sequence[row])
        c0_physical = np.asarray(flow["features"][flow_row], np.float32).copy()
        slots = np.asarray(flow["slot_mask"][flow_row], bool)
        if self.evaluation_scope:
            slots = np.asarray(scoped_slot_mask(slots), bool)
        c0 = normalize_c0_features(
            c0_physical, slots, self.bundle.flow_schema, self.contract
        )
        constraint = extract_trajectory_constraint(states[:, 1:])
        constraint_normalized = (
            constraint - self.constraint_mean
        ) / self.constraint_std
        dimensions = np.asarray(flow["vehicle_dimensions"][flow_row], np.float32)
        condition = condition_vector(c0, slots, constraint_normalized)
        active = valid[0, 1:]
        reference = trajectory_reference_positions(states[0, 1:], constraint)
        residual = trajectory_residual_target(states[1:, 1:, :2], reference)
        mask = np.broadcast_to(active[None, :, None], residual.shape)
        normalized = ((residual - self.position_mean) / self.position_std) * mask
        return {
            "condition": torch.from_numpy(condition),
            "target": torch.from_numpy(normalized.reshape(HORIZON_STEPS, -1).copy()),
            "target_mask": torch.from_numpy(mask.reshape(HORIZON_STEPS, -1).copy()),
            "c0_states": torch.from_numpy(states[0].copy()),
            "future_states": torch.from_numpy(states[1:].copy()),
            "trajectory_reference": torch.from_numpy(reference),
            "vehicle_dimensions": torch.from_numpy(dimensions.copy()),
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

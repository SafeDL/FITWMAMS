"""First-second behavior anchors from the frozen 76-dimensional Flow."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from normalizing_flow.src.features import (
    EGO_FEATURES,
    SLOT_NAMES,
    TRAJECTORY_FEATURES,
    slot_feature_index,
    trajectory_feature_index,
)

from .schema import FLOW_ACTION_SUMMARY_FEATURES
from .utils import file_sha256


BEHAVIOR_ANCHOR_SECONDS = 1.0


@dataclass(frozen=True)
class FrozenLegacyFlowSchema:
    """Read-only contract for the frozen legacy 76-D Flow coordinates."""

    feature_names: tuple[str, ...]
    slot_names: tuple[str, ...]
    trajectory_features: tuple[str, ...]
    model_feature_transforms: tuple[str, ...]
    normalization_mean: np.ndarray
    normalization_std: np.ndarray
    anchor_feature_indices: np.ndarray
    schema_sha256: str
    source_path: Path

    @classmethod
    def load(cls, path: str | Path) -> "FrozenLegacyFlowSchema":
        source = Path(path)
        raw = source.read_bytes()
        schema = json.loads(raw)
        names = tuple(schema.get("feature_names", ()))
        slots = tuple(schema.get("slot_names", ()))
        trajectory = tuple(schema.get("trajectory_features", ()))
        transforms = tuple(schema.get("model_feature_transforms", ()))
        mean = np.asarray(schema.get("normalization", {}).get("mean", ()), np.float32)
        std = np.asarray(schema.get("normalization", {}).get("std", ()), np.float32)
        expected_indices = np.asarray([
            [trajectory_feature_index(slot, feature) for feature in FLOW_ACTION_SUMMARY_FEATURES]
            for slot in SLOT_NAMES
        ], np.int64)
        if (len(names), len(mean), len(std), len(transforms)) != (76, 76, 76, 76):
            raise ValueError("frozen Flow schema must contain exactly 76 features, transforms, means and stds")
        if slots != tuple(SLOT_NAMES) or trajectory != tuple(TRAJECTORY_FEATURES):
            raise ValueError("frozen Flow slot or trajectory feature order differs from the legacy contract")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0.0):
            raise ValueError("frozen Flow normalization is invalid")
        return cls(names, slots, trajectory, transforms, mean, std, expected_indices, hashlib.sha256(raw).hexdigest(), source)

    def verify_checkpoint(self, checkpoint: str | Path, expected_sha256: str | None = None) -> str:
        digest = file_sha256(checkpoint)
        if expected_sha256 is not None and digest != str(expected_sha256):
            raise ValueError("frozen Flow checkpoint SHA256 mismatch")
        return digest

    def standardize(self, anchor_raw: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Apply the exact legacy model-coordinate transform to six summaries."""
        if anchor_raw.shape[-2:] != (len(SLOT_NAMES), len(FLOW_ACTION_SUMMARY_FEATURES)):
            raise ValueError("behavior anchor must end in [six slots, six summaries]")
        if valid.shape != anchor_raw.shape[:-1]:
            raise ValueError("behavior anchor validity does not align with anchor values")
        indices = torch.as_tensor(self.anchor_feature_indices, device=anchor_raw.device)
        mean = torch.as_tensor(self.normalization_mean, device=anchor_raw.device, dtype=anchor_raw.dtype)[indices]
        std = torch.as_tensor(self.normalization_std, device=anchor_raw.device, dtype=anchor_raw.dtype)[indices]
        model = anchor_raw.clone()
        # The legacy Flow represents min_ax as softplus^{-1}(mean_ax-min_ax).
        gap = (anchor_raw[..., 2] - anchor_raw[..., 3]).clamp_min(1.0e-4)
        transformed_gap = torch.where(gap > 20.0, gap, torch.log(torch.expm1(gap) + 1.0e-4))
        model[..., 3] = transformed_gap
        out = (model - mean) / std
        return out * valid[..., None].to(dtype=out.dtype)

    def anchor_statistics(self) -> tuple[np.ndarray, np.ndarray]:
        return self.normalization_mean[self.anchor_feature_indices], self.normalization_std[self.anchor_feature_indices]


def behavior_anchor_from_flow_feature(
    feature_row: np.ndarray,
    slot_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the six per-slot first-second summaries from a legacy Flow row."""
    feature = np.asarray(feature_row, dtype=np.float32).reshape(-1)
    valid = np.asarray(slot_mask, dtype=bool).reshape(-1)
    if feature.size != 76 or valid.shape != (len(SLOT_NAMES),):
        raise ValueError("behavior anchors require one 76-D Flow row and a six-slot mask")
    anchor = np.zeros((len(SLOT_NAMES), len(FLOW_ACTION_SUMMARY_FEATURES)), dtype=np.float32)
    for slot_index, slot_name in enumerate(SLOT_NAMES):
        if valid[slot_index]:
            anchor[slot_index] = [
                feature[trajectory_feature_index(slot_name, name)]
                for name in FLOW_ACTION_SUMMARY_FEATURES
            ]
    return anchor, valid


def start_state_from_flow_feature(feature_row: np.ndarray, slot_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Directly unpack one raw Flow row for the behavior-anchored START mode.

    Training never calls this adapter: it reads the logged sequence cache.
    It exists solely for Flow end-to-end generation, where the sampled 76-D
    row already contains both the start scene and the six first-second
    summaries.  Keeping this conversion as a small value function avoids a
    separate "atomic initializer" object and its redundant lifecycle.
    """
    feature = np.asarray(feature_row, np.float32).reshape(-1)
    valid_slots = np.asarray(slot_mask, bool).reshape(-1)
    if feature.shape != (76,) or valid_slots.shape != (len(SLOT_NAMES),):
        raise ValueError("Flow START requires one raw 76-D row and one six-slot mask")
    if not np.isfinite(feature).all():
        raise ValueError("Flow START feature row contains a non-finite value")
    states = np.zeros((1 + len(SLOT_NAMES), 6), np.float32)
    valid = np.zeros((1 + len(SLOT_NAMES),), bool)
    ego = {name: feature[EGO_FEATURES.index(name)] for name in EGO_FEATURES}
    states[0] = (0.0, 0.0, ego["ego_vx_mps"], ego["ego_vy_left_mps"], ego["ego_ax_mps2"], ego["ego_ay_left_mps2"])
    valid[0] = True
    for index, slot in enumerate(SLOT_NAMES):
        if not valid_slots[index]:
            continue
        states[index + 1] = (
            feature[slot_feature_index(slot, "rel_x_m")], feature[slot_feature_index(slot, "rel_y_left_m")],
            ego["ego_vx_mps"] + feature[slot_feature_index(slot, "rel_vx_mps")],
            ego["ego_vy_left_mps"] + feature[slot_feature_index(slot, "rel_vy_left_mps")],
            feature[slot_feature_index(slot, "other_ax_mps2")], feature[slot_feature_index(slot, "other_ay_left_mps2")],
        )
        valid[index + 1] = True
    anchor, anchor_valid = behavior_anchor_from_flow_feature(feature, valid_slots)
    return states, valid, anchor, anchor_valid


def start_state_from_flow_tensor(
    feature_rows: torch.Tensor, slot_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch Torch form of :func:`start_state_from_flow_feature`.

    This is the shared Flow START adapter used by tensor world models.  It
    deliberately follows the raw Flow coordinate contract: a background
    velocity is stored relative to ego and must be reconstructed before any
    scene encoder sees it.
    """
    if feature_rows.ndim != 2 or feature_rows.shape[-1] != 76:
        raise ValueError("Flow START requires features with shape [batch, 76]")
    if not torch.isfinite(feature_rows).all():
        raise ValueError("Flow START feature rows contain non-finite values")
    batch = feature_rows.shape[0]
    if slot_mask is None:
        slot_mask = torch.ones((batch, len(SLOT_NAMES)), dtype=torch.bool, device=feature_rows.device)
    if slot_mask.shape != (batch, len(SLOT_NAMES)):
        raise ValueError("Flow START slot mask must have shape [batch, six slots]")
    valid_slots = slot_mask.bool()
    states = feature_rows.new_zeros((batch, 1 + len(SLOT_NAMES), 6))
    valid = torch.zeros((batch, 1 + len(SLOT_NAMES)), dtype=torch.bool, device=feature_rows.device)
    ego_indices = [EGO_FEATURES.index(name) for name in EGO_FEATURES]
    ego = feature_rows[:, ego_indices]
    states[:, 0, 2:6] = ego
    valid[:, 0] = True
    anchor = feature_rows.new_zeros((batch, len(SLOT_NAMES), len(FLOW_ACTION_SUMMARY_FEATURES)))
    for index, slot in enumerate(SLOT_NAMES):
        position_x = feature_rows[:, slot_feature_index(slot, "rel_x_m")]
        position_y = feature_rows[:, slot_feature_index(slot, "rel_y_left_m")]
        velocity_x = ego[:, 0] + feature_rows[:, slot_feature_index(slot, "rel_vx_mps")]
        velocity_y = ego[:, 1] + feature_rows[:, slot_feature_index(slot, "rel_vy_left_mps")]
        acceleration_x = feature_rows[:, slot_feature_index(slot, "other_ax_mps2")]
        acceleration_y = feature_rows[:, slot_feature_index(slot, "other_ay_left_mps2")]
        states[:, index + 1] = torch.stack(
            (position_x, position_y, velocity_x, velocity_y, acceleration_x, acceleration_y), dim=-1
        )
        anchor[:, index] = torch.stack(
            [feature_rows[:, trajectory_feature_index(slot, name)] for name in FLOW_ACTION_SUMMARY_FEATURES], dim=-1
        )
    states[:, 1:] *= valid_slots[..., None].to(dtype=states.dtype)
    valid[:, 1:] = valid_slots
    return states, valid, anchor * valid_slots[..., None].to(dtype=anchor.dtype), valid_slots


def summarize_first_second_states(states_26: torch.Tensor, valid_26: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact six Flow summaries and their all-26-frame validity.

    The Flow extractor uses anchor and ``anchor + 25`` as exact velocity
    endpoints and includes all 26 state samples in its acceleration window.
    This function is the single implementation used for logged targets and
    generated rollouts, so no action-integration proxy can drift from Flow.
    """
    if states_26.ndim != 4 or states_26.shape[1] != 26 or states_26.shape[-1] != 6 or valid_26.shape != states_26.shape[:-1]:
        raise ValueError("states must be [batch, frames, agents, 6] with matching validity")
    anchor_valid = valid_26.all(dim=1)
    ax, ay = states_26[..., 4], states_26[..., 5]
    raw = torch.stack((
        states_26[:, -1, :, 2] - states_26[:, 0, :, 2],
        states_26[:, -1, :, 3] - states_26[:, 0, :, 3],
        ax.mean(dim=1),
        ax.amin(dim=1),
        ax[:, -1],
        ay.mean(dim=1),
    ), dim=-1)
    # Replace inactive values rather than admitting partial windows into a
    # Flow condition.  This exactly matches the slot-level Flow contract.
    return raw * anchor_valid[..., None].to(dtype=raw.dtype), anchor_valid


def summarize_highd_states(states: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Compatibility alias; formal code must call the 26-state function."""
    return summarize_first_second_states(states, valid)[0]


class BehaviorAnchorControlPlan(nn.Module):
    """Project one Flow summary onto a smooth, bounded 25 Hz action plan."""

    def __init__(self, physics_steps: int = 25, knots: int = 5) -> None:
        super().__init__()
        if physics_steps < 2 or knots < 2:
            raise ValueError("behavior plan needs at least two frames and knots")
        positions = torch.linspace(0.0, float(knots - 1), physics_steps)
        left = positions.floor().long().clamp(max=knots - 1)
        right = (left + 1).clamp(max=knots - 1)
        fraction = positions - left.float()
        basis = torch.zeros((physics_steps, knots))
        basis[torch.arange(physics_steps), left] += 1.0 - fraction
        basis[torch.arange(physics_steps), right] += fraction
        smooth = torch.zeros((knots - 2, knots))
        for index in range(knots - 2):
            smooth[index, index : index + 3] = torch.tensor((1.0, -2.0, 1.0))
        self.register_buffer("basis", basis)
        self.register_buffer("smooth", smooth)

    def _solve(
        self,
        initial_acceleration: torch.Tensor,
        delta_velocity: torch.Tensor,
        mean_acceleration: torch.Tensor,
        final_acceleration: torch.Tensor | None,
        minimum_acceleration: torch.Tensor | None,
    ) -> torch.Tensor:
        """Solve a small ridge least-squares projection for every agent."""
        basis = self.basis.to(dtype=initial_acceleration.dtype, device=initial_acceleration.device)
        dt = 1.0 / float(basis.shape[0])
        rows = [dt * basis.sum(dim=0), basis.sum(dim=0) / float(basis.shape[0] + 1)]
        targets = [delta_velocity, mean_acceleration - initial_acceleration / float(basis.shape[0] + 1)]
        if final_acceleration is not None:
            rows.append(basis[-1])
            targets.append(final_acceleration)
        matrix = torch.stack(rows)
        target = torch.stack(targets, dim=-1)
        regularizer = 2.0e-2 * (self.smooth.T @ self.smooth).to(dtype=matrix.dtype, device=matrix.device)
        normal = matrix.T @ matrix + regularizer + 1.0e-5 * torch.eye(matrix.shape[1], dtype=matrix.dtype, device=matrix.device)
        solution = torch.linalg.solve(normal, matrix.T @ target.reshape(-1, target.shape[-1]).T).T
        curve = solution.reshape(*target.shape[:-1], -1) @ basis.T
        if minimum_acceleration is not None:
            # The minimum is an inequality-like statistic.  A soft-min row
            # yields the nearest smooth feasible curve without forcing
            # mutually inconsistent Flow samples to satisfy exact equalities.
            weights = torch.softmax(-curve.detach() / 0.35, dim=-1)
            min_row = torch.einsum("...t,tk->...k", weights, basis)
            full_target = torch.cat((target.reshape(-1, target.shape[-1]), (1.5 * minimum_acceleration).reshape(-1, 1)), dim=-1)
            # Solve each agent independently because the soft-min row differs.
            rows_per_agent = torch.cat((matrix.expand(full_target.shape[0], -1, -1), 1.5 * min_row.reshape(-1, 1, min_row.shape[-1])), dim=1)
            normal = rows_per_agent.transpose(1, 2) @ rows_per_agent + regularizer.unsqueeze(0)
            rhs = rows_per_agent.transpose(1, 2) @ full_target.unsqueeze(-1)
            solution = torch.linalg.solve(normal, rhs).squeeze(-1)
            curve = (solution @ basis.T).reshape(*target.shape[:-1], -1)
        return curve

    def forward(self, initial_states: torch.Tensor, anchor: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Return Cartesian highD actions ``[B, 25, background_agents, 2]``."""
        if initial_states.ndim != 3 or anchor.ndim != 3 or anchor.shape[-1] != 6:
            raise ValueError("initial states and behavior anchor have incompatible shapes")
        if initial_states.shape[:2] != (anchor.shape[0], anchor.shape[1] + 1) or valid.shape != anchor.shape[:2]:
            raise ValueError("behavior plan requires ego plus one entry per Flow slot")
        initial = initial_states[:, 1:, 4:6]
        longitudinal = self._solve(initial[..., 0], anchor[..., 0], anchor[..., 2], anchor[..., 4], anchor[..., 3])
        lateral = self._solve(initial[..., 1], anchor[..., 1], anchor[..., 5], None, None)
        actions = torch.stack((longitudinal, lateral), dim=-1).permute(0, 2, 1, 3)
        actions[..., 0] = actions[..., 0].clamp(-8.0, 4.0)
        actions[..., 1] = actions[..., 1].clamp(-4.0, 4.0)
        return actions * valid[:, None, :, None].to(dtype=actions.dtype)

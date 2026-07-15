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
    SLOT_NAMES,
    TRAJECTORY_FEATURES,
    trajectory_feature_index,
)

from .schema import FLOW_ACTION_SUMMARY_FEATURES


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
        return cls(names, slots, trajectory, transforms, mean, std, expected_indices, hashlib.sha256(raw).hexdigest())

    def verify_checkpoint(self, checkpoint: str | Path, expected_sha256: str | None = None) -> str:
        digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
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


class BehaviorAnchorEncoder(nn.Module):
    """Mask-aware per-vehicle and scene-level encoding of a behavior anchor."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros((len(SLOT_NAMES), len(FLOW_ACTION_SUMMARY_FEATURES))))
        self.register_buffer("std", torch.ones((len(SLOT_NAMES), len(FLOW_ACTION_SUMMARY_FEATURES))))
        self.per_agent = nn.Sequential(
            nn.Linear(len(FLOW_ACTION_SUMMARY_FEATURES), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape == (len(FLOW_ACTION_SUMMARY_FEATURES),):
            mean = mean.expand_as(self.mean)
        if std.shape == (len(FLOW_ACTION_SUMMARY_FEATURES),):
            std = std.expand_as(self.std)
        if mean.shape != self.mean.shape or std.shape != self.std.shape:
            raise ValueError("behavior-anchor statistics must be [six slots, six summaries]")
        self.mean.copy_(mean.to(device=self.mean.device, dtype=self.mean.dtype))
        self.std.copy_(std.to(device=self.std.device, dtype=self.std.dtype).clamp_min(1.0e-3))

    def normalize(self, anchor: torch.Tensor) -> torch.Tensor:
        if anchor.shape[-1] != len(FLOW_ACTION_SUMMARY_FEATURES) or anchor.shape[-2] > len(SLOT_NAMES):
            raise ValueError("behavior-anchor shape is incompatible with the fixed Flow slots")
        return (anchor - self.mean[: anchor.shape[-2]]) / self.std[: anchor.shape[-2]]

    def forward(self, anchor: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if anchor.ndim != 3 or anchor.shape[-1] != len(FLOW_ACTION_SUMMARY_FEATURES):
            raise ValueError("behavior anchor must be [batch, agents, 6]")
        if valid.shape != anchor.shape[:-1]:
            raise ValueError("behavior anchor validity must align with anchors")
        # Callers pass frozen-Flow model coordinates.  Keeping normalization
        # outside this module prevents silently applying the min-ax transform
        # twice and makes the raw/std boundary explicit.
        agents = self.per_agent(anchor) * valid[..., None].to(dtype=anchor.dtype)
        scene = agents.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=anchor.dtype)
        return agents, scene

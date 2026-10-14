"""Tail-scene density over initial states and sparse position plans."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .constraints import (
    KNOT_FEATURE_NAMES,
    KNOT_TIMES_S,
    constraint_from_model_coordinates,
    constraint_to_model_coordinates,
)
from .features import SLOT_NAMES, feature_index, feature_valid_from_slot_mask
from .model import build_maf_flow
from .sampling import inverse_normalize_features, normalize_features

CHECKPOINT_SCHEMA = "highd_tail_scenario_flow"


@dataclass
class ScenarioBatch:
    c0: np.ndarray
    slot_mask: np.ndarray
    trajectory_constraint: np.ndarray
    trajectory_constraint_valid: np.ndarray
    vehicle_dimensions: np.ndarray
    c0_normalized_reference: np.ndarray | None = None
    constraint_normalized_reference: np.ndarray | None = None


def normalize_constraint(
    values: np.ndarray,
    valid: np.ndarray,
    schema: dict[str, Any],
    c0: np.ndarray,
) -> np.ndarray:
    norm = schema["long_horizon_constraint"]["normalization"]
    mean = np.asarray(norm["mean"], np.float32)
    std = np.asarray(norm["std"], np.float32)
    output = np.zeros_like(values, dtype=np.float32)
    present = np.asarray(valid, bool)
    model = constraint_to_model_coordinates(values, c0)
    output[present] = ((model - mean) / std)[present]
    return output


def inverse_constraint(
    normalized: np.ndarray,
    slot_mask: np.ndarray,
    schema: dict[str, Any],
    c0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    norm = schema["long_horizon_constraint"]["normalization"]
    values = np.asarray(normalized, np.float32) * np.asarray(norm["std"], np.float32)
    values += np.asarray(norm["mean"], np.float32)
    values = constraint_from_model_coordinates(values, c0)
    valid = np.broadcast_to(np.asarray(slot_mask, bool)[..., None], values.shape).copy()
    values[~valid] = 0.0
    return values, valid


def project_physical_c0(
    values: np.ndarray, slot_mask: np.ndarray, dimensions: np.ndarray
) -> np.ndarray:
    """Project generated initial position and velocity states to road semantics."""
    output = np.asarray(values, np.float32).copy()
    slots = np.asarray(slot_mask, bool)
    ego_vx = feature_index(None, "ego_vx_mps")
    ego_vy = feature_index(None, "ego_vy_left_mps")
    output[:, ego_vx] = np.clip(output[:, ego_vx], 0.0, 70.0)
    output[:, ego_vy] = np.clip(output[:, ego_vy], -10.0, 10.0)
    for slot, name in enumerate(SLOT_NAMES):
        active = slots[:, slot]
        dx = feature_index(name, "rel_x_m")
        dy = feature_index(name, "rel_y_left_m")
        dvx = feature_index(name, "rel_vx_mps")
        dvy = feature_index(name, "rel_vy_left_mps")
        minimum_x = 0.5 * (dimensions[:, 0, 0] + dimensions[:, slot + 1, 0]) + 0.01
        if name.startswith("same"):
            if "front" in name:
                output[active, dx] = np.maximum(output[active, dx], minimum_x[active])
            else:
                output[active, dx] = np.minimum(output[active, dx], -minimum_x[active])
        else:
            output[active, dx] = (
                np.maximum(output[active, dx], 0.01)
                if "front" in name
                else np.minimum(output[active, dx], -0.01)
            )
        minimum_y = 0.5 * (dimensions[:, 0, 1] + dimensions[:, slot + 1, 1]) + 0.01
        if name.startswith("left"):
            output[active, dy] = np.maximum(output[active, dy], minimum_y[active])
        elif name.startswith("right"):
            output[active, dy] = np.minimum(output[active, dy], -minimum_y[active])
        other_vx = output[:, ego_vx] + output[:, dvx]
        other_vy = output[:, ego_vy] + output[:, dvy]
        output[active, dvx] += np.clip(other_vx[active], 0.0, 70.0) - other_vx[active]
        output[active, dvy] += np.clip(other_vy[active], -10.0, 10.0) - other_vy[active]
    positions = np.stack(
        [
            output[
                :,
                [feature_index(name, "rel_x_m"), feature_index(name, "rel_y_left_m")],
            ]
            for name in SLOT_NAMES
        ],
        axis=1,
    )
    for left, right in combinations(range(len(SLOT_NAMES)), 2):
        separation = positions[:, right, 0] - positions[:, left, 0]
        required_x = (
            0.5 * (dimensions[:, left + 1, 0] + dimensions[:, right + 1, 0]) + 0.01
        )
        required_y = 0.5 * (dimensions[:, left + 1, 1] + dimensions[:, right + 1, 1])
        overlap = (
            slots[:, left]
            & slots[:, right]
            & (np.abs(separation) < required_x)
            & (np.abs(positions[:, right, 1] - positions[:, left, 1]) <= required_y)
        )
        direction = np.where(separation >= 0.0, 1.0, -1.0)
        correction = direction * (required_x - np.abs(separation))
        right_x = feature_index(SLOT_NAMES[right], "rel_x_m")
        output[overlap, right_x] += correction[overlap]
        positions[overlap, right, 0] += correction[overlap]
    valid = feature_valid_from_slot_mask(
        {"feature_names": list(range(output.shape[1]))}, slots
    )
    output[~valid] = 0.0
    return output


def _minimum_displacement(speed: np.ndarray, duration: float) -> np.ndarray:
    stop_time = speed / 8.0
    return np.where(
        stop_time >= duration,
        speed * duration - 4.0 * duration**2,
        speed**2 / 16.0,
    )


def project_constraint(
    values: np.ndarray, slot_mask: np.ndarray, c0: np.ndarray
) -> np.ndarray:
    """Make longitudinal position knots reachable with ordinary acceleration."""
    output = np.asarray(values, np.float32).copy()
    slots = np.asarray(slot_mask, bool)
    speed = np.stack(
        [
            c0[:, feature_index(None, "ego_vx_mps")]
            + c0[:, feature_index(name, "rel_vx_mps")]
            for name in SLOT_NAMES
        ],
        axis=1,
    )
    position = np.zeros_like(speed)
    previous_time = 0.0
    for knot, time in enumerate(KNOT_TIMES_S):
        duration = float(time - previous_time)
        index = 2 * knot
        displacement = output[..., index] - position
        lower = _minimum_displacement(speed, duration)
        upper = speed * duration + 2.0 * duration**2
        displacement = np.clip(displacement, lower, upper)
        position += displacement
        output[..., index] = position
        speed = np.clip(2.0 * displacement / duration - speed, 0.0, 70.0)
        previous_time = float(time)
    output[~np.broadcast_to(slots[..., None], output.shape)] = 0.0
    return output


def _sample_flow(
    flow: nn.Module,
    context: torch.Tensor,
    generator: torch.Generator | None = None,
    *,
    base_noise: torch.Tensor | None = None,
    chunk_size: int = 128,
) -> torch.Tensor:
    features = int(np.prod(flow._distribution._shape))
    if base_noise is None:
        if generator is None:
            raise ValueError("either generator or base_noise is required")
        noise = torch.randn(
            (len(context), features), generator=generator, device=context.device
        )
    else:
        noise = torch.as_tensor(base_noise, dtype=context.dtype, device=context.device)
        if tuple(noise.shape) != (len(context), features):
            raise ValueError(
                f"flow base noise must have shape {(len(context), features)}, "
                f"got {tuple(noise.shape)}"
            )
    output = []
    for start in range(0, len(context), chunk_size):
        selected = context[start : start + chunk_size]
        embedded = flow._embedding_net(selected)
        value, _ = flow._transform.inverse(
            noise[start : start + len(selected)], context=embedded
        )
        output.append(value)
    return torch.cat(output)


class DirectScenarioFlow(nn.Module):
    """Tail density ``p(M) p(C0|M) p(K|C0,M)``."""

    def __init__(
        self,
        *,
        c0_flow: nn.Module,
        constraint_flow: nn.Module,
        mask_log_prob: torch.Tensor,
        dimension_bank: torch.Tensor,
        dimension_pattern: torch.Tensor,
        schema: dict[str, Any],
    ) -> None:
        super().__init__()
        self.c0_flow = c0_flow
        self.constraint_flow = constraint_flow
        self.register_buffer("mask_log_prob", mask_log_prob.float())
        self.register_buffer("dimension_bank", dimension_bank.float())
        self.register_buffer("dimension_pattern", dimension_pattern.long())
        self.schema = schema

    def log_prob_tensors(
        self,
        *,
        c0_normalized: torch.Tensor,
        slot_mask: torch.Tensor,
        constraint_normalized: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bits = 2 ** torch.arange(6, device=slot_mask.device)
        pattern = (slot_mask.long() * bits).sum(1)
        mask = self.mask_log_prob[pattern]
        c0 = self.c0_flow.log_prob(c0_normalized, context=slot_mask.float())
        context = torch.cat((c0_normalized, slot_mask.float()), dim=-1)
        constraint = self.constraint_flow.log_prob(
            constraint_normalized.flatten(1), context=context
        )
        return {
            "mask_log_prob": mask,
            "c0_log_prob": c0,
            "k_log_prob": constraint,
            "joint_log_prob": mask + c0 + constraint,
        }

    def log_prob(self, batch: ScenarioBatch) -> dict[str, np.ndarray]:
        valid = feature_valid_from_slot_mask(self.schema, batch.slot_mask)
        c0_norm = (
            normalize_features(batch.c0, valid, self.schema)
            if batch.c0_normalized_reference is None
            else np.asarray(batch.c0_normalized_reference, np.float32)
        )
        k_norm = (
            normalize_constraint(
                batch.trajectory_constraint,
                batch.trajectory_constraint_valid,
                self.schema,
                batch.c0,
            )
            if batch.constraint_normalized_reference is None
            else np.asarray(batch.constraint_normalized_reference, np.float32)
        )
        device = next(self.parameters()).device
        with torch.no_grad():
            terms = self.log_prob_tensors(
                c0_normalized=torch.from_numpy(c0_norm).float().to(device),
                slot_mask=torch.from_numpy(batch.slot_mask).bool().to(device),
                constraint_normalized=torch.from_numpy(k_norm).float().to(device),
            )
        return {name: value.cpu().numpy() for name, value in terms.items()}

    def _structure_from_uniform(
        self, scenario_uniform: np.ndarray
    ) -> tuple[torch.Tensor, np.ndarray]:
        device = self.mask_log_prob.device
        uniform = torch.as_tensor(scenario_uniform, dtype=torch.float64, device=device)
        if uniform.ndim != 1 or ((uniform < 0.0) | (uniform >= 1.0)).any():
            raise ValueError("scenario_uniform must be a vector in [0, 1)")
        probability = self.mask_log_prob.exp().double()
        support = torch.nonzero(probability > 0.0, as_tuple=False).flatten()
        support_probability = probability[support]
        cumulative = torch.cumsum(support_probability, dim=0)
        support_index = torch.searchsorted(cumulative, uniform).clamp_max(
            len(support) - 1
        )
        pattern = support[support_index]
        lower = torch.where(
            support_index > 0,
            cumulative[(support_index - 1).clamp_min(0)],
            0.0,
        )
        fraction = ((uniform - lower) / support_probability[support_index]).clamp(
            0.0, 1.0 - 1e-12
        )
        dimensions = torch.empty(
            (len(pattern), 7, 2), dtype=torch.float32, device=device
        )
        for value in torch.unique(pattern):
            rows = torch.nonzero(pattern == value, as_tuple=False).flatten()
            bank = torch.nonzero(
                self.dimension_pattern == value, as_tuple=False
            ).flatten()
            if not len(bank):
                raise RuntimeError(
                    f"missing empirical dimensions for mask {int(value)}"
                )
            selected = (
                torch.floor(fraction[rows] * len(bank)).long().clamp_max(len(bank) - 1)
            )
            dimensions[rows] = self.dimension_bank[bank[selected]]
        bits = 2 ** torch.arange(6, device=device)
        slots = pattern[:, None].bitwise_and(bits[None]) > 0
        return slots, dimensions.cpu().numpy()

    @torch.no_grad()
    def sample_scenarios(self, n: int, scenario_seed: int) -> ScenarioBatch:
        root = np.random.SeedSequence(int(scenario_seed))
        uniform_seed, c0_seed, k_seed = (
            int(item.generate_state(1)[0]) for item in root.spawn(3)
        )
        uniform = np.random.default_rng(uniform_seed).random(int(n))
        slots, dimensions = self._structure_from_uniform(uniform)
        generator = torch.Generator(device=slots.device).manual_seed(c0_seed)
        c0_norm = _sample_flow(self.c0_flow, slots.float(), generator)
        return self._sample_given(c0_norm, slots, dimensions, k_seed)

    @torch.no_grad()
    def sample_initial_conditions_from_base_randomness(
        self, scenario_uniform: np.ndarray, c0_base_latent: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        slots, dimensions = self._structure_from_uniform(scenario_uniform)
        c0_norm = _sample_flow(
            self.c0_flow,
            slots.float(),
            base_noise=torch.as_tensor(c0_base_latent, device=slots.device),
        )
        c0 = inverse_normalize_features(c0_norm.cpu().numpy(), self.schema)
        slot_values = slots.cpu().numpy()
        c0[~feature_valid_from_slot_mask(self.schema, slot_values)] = 0.0
        c0 = project_physical_c0(c0, slot_values, dimensions)
        return c0.astype(np.float32), slot_values, dimensions.astype(np.float32)

    @torch.no_grad()
    def sample_constraints(
        self,
        c0: np.ndarray,
        slot_mask: np.ndarray,
        vehicle_dimensions: np.ndarray,
        n: int,
        scenario_seed: int,
    ) -> ScenarioBatch:
        c0_dim = len(self.schema["feature_names"])
        values = np.asarray(c0, np.float32).reshape(-1, c0_dim)
        slots = np.asarray(slot_mask, bool).reshape(-1, 6)
        dimensions = np.asarray(vehicle_dimensions, np.float32).reshape(-1, 7, 2)
        if len(values) == 1 and int(n) > 1:
            values = np.repeat(values, int(n), axis=0)
            slots = np.repeat(slots, int(n), axis=0)
            dimensions = np.repeat(dimensions, int(n), axis=0)
        valid = feature_valid_from_slot_mask(self.schema, slots)
        c0_norm = normalize_features(values, valid, self.schema)
        device = next(self.parameters()).device
        return self._sample_given(
            torch.from_numpy(c0_norm).float().to(device),
            torch.from_numpy(slots).bool().to(device),
            dimensions,
            int(scenario_seed),
            c0_override=values,
        )

    @torch.no_grad()
    def sample_constraints_from_base_randomness(
        self,
        c0: np.ndarray,
        slot_mask: np.ndarray,
        k_base_latent: np.ndarray,
        vehicle_dimensions: np.ndarray,
    ) -> ScenarioBatch:
        c0_dim = len(self.schema["feature_names"])
        values = np.asarray(c0, np.float32).reshape(-1, c0_dim)
        slots = np.asarray(slot_mask, bool).reshape(-1, 6)
        dimensions = np.asarray(vehicle_dimensions, np.float32).reshape(-1, 7, 2)
        valid = feature_valid_from_slot_mask(self.schema, slots)
        device = next(self.parameters()).device
        c0_norm = (
            torch.from_numpy(normalize_features(values, valid, self.schema))
            .float()
            .to(device)
        )
        slot_tensor = torch.from_numpy(slots).bool().to(device)
        context = torch.cat((c0_norm, slot_tensor.float()), dim=-1)
        k_norm = _sample_flow(
            self.constraint_flow,
            context,
            base_noise=torch.as_tensor(k_base_latent, device=device),
        ).reshape(-1, 6, len(KNOT_FEATURE_NAMES))
        return self._scenario_from_normalized(
            c0_norm, slot_tensor, k_norm, dimensions, c0_override=values
        )

    def _sample_given(
        self,
        c0_norm: torch.Tensor,
        slots: torch.Tensor,
        dimensions: np.ndarray,
        k_seed: int,
        c0_override: np.ndarray | None = None,
    ) -> ScenarioBatch:
        generator = torch.Generator(device=c0_norm.device).manual_seed(int(k_seed))
        context = torch.cat((c0_norm, slots.float()), dim=-1)
        k_norm = _sample_flow(self.constraint_flow, context, generator).reshape(
            -1, 6, len(KNOT_FEATURE_NAMES)
        )
        return self._scenario_from_normalized(
            c0_norm, slots, k_norm, dimensions, c0_override=c0_override
        )

    def _scenario_from_normalized(
        self,
        c0_norm: torch.Tensor,
        slots: torch.Tensor,
        k_norm: torch.Tensor,
        dimensions: np.ndarray,
        c0_override: np.ndarray | None = None,
    ) -> ScenarioBatch:
        slot_values = slots.cpu().numpy()
        dimensions = np.asarray(dimensions, np.float32)
        if c0_override is None:
            c0 = inverse_normalize_features(c0_norm.cpu().numpy(), self.schema)
            c0[~feature_valid_from_slot_mask(self.schema, slot_values)] = 0.0
            c0 = project_physical_c0(c0, slot_values, dimensions)
        else:
            c0 = np.asarray(c0_override, np.float32).copy()
            c0[~feature_valid_from_slot_mask(self.schema, slot_values)] = 0.0
        constraint, valid = inverse_constraint(
            k_norm.cpu().numpy(), slot_values, self.schema, c0
        )
        constraint = project_constraint(constraint, slot_values, c0)
        return ScenarioBatch(
            c0=c0.astype(np.float32),
            slot_mask=slot_values,
            trajectory_constraint=constraint.astype(np.float32),
            trajectory_constraint_valid=valid,
            vehicle_dimensions=dimensions,
            c0_normalized_reference=c0_norm.cpu().numpy(),
            constraint_normalized_reference=k_norm.cpu().numpy(),
        )


def build_direct_model(
    schema: dict[str, Any],
    model_cfg: dict[str, Any],
    repo_root: str | Path,
    mask_log_prob: torch.Tensor,
    dimension_bank: torch.Tensor,
    dimension_pattern: torch.Tensor,
) -> DirectScenarioFlow:
    c0_dim = len(schema["feature_names"])
    c0 = build_maf_flow(
        num_features=c0_dim,
        context_features=6,
        model_cfg=dict(model_cfg["c0_flow"]),
        repo_root=repo_root,
    )
    constraint = build_maf_flow(
        num_features=6 * len(KNOT_FEATURE_NAMES),
        context_features=c0_dim + 6,
        model_cfg=dict(model_cfg["constraint_flow"]),
        repo_root=repo_root,
    )
    return DirectScenarioFlow(
        c0_flow=c0,
        constraint_flow=constraint,
        mask_log_prob=mask_log_prob,
        dimension_bank=dimension_bank,
        dimension_pattern=dimension_pattern,
        schema=schema,
    )


def save_checkpoint(
    path: str | Path,
    model: DirectScenarioFlow,
    model_cfg: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "state_dict": model.state_dict(),
            "model_cfg": model_cfg,
            "schema": model.schema,
            "metrics": metrics,
        },
        Path(path),
    )


def load_checkpoint(
    path: str | Path,
    *,
    repo_root: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[DirectScenarioFlow, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload["checkpoint_schema"] != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not the maintained tail-scenario flow")
    state = payload["state_dict"]
    model = build_direct_model(
        payload["schema"],
        payload["model_cfg"],
        repo_root,
        state["mask_log_prob"],
        state["dimension_bank"],
        state["dimension_pattern"],
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model, payload

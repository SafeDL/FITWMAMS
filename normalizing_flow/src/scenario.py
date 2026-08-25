"""Direct density over highD initial states and sparse state constraints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .constraints import KNOT_FEATURE_NAMES
from .features import SLOT_NAMES, feature_index, feature_valid_from_slot_mask
from .model import build_maf_flow
from .sampling import inverse_normalize_features, normalize_features

CHECKPOINT_SCHEMA = "highd_direct_scenario_condition_flow"


@dataclass
class ScenarioBatch:
    c0: np.ndarray
    slot_mask: np.ndarray
    trajectory_constraint: np.ndarray
    trajectory_constraint_valid: np.ndarray
    c0_normalized_reference: np.ndarray | None = None
    constraint_normalized_reference: np.ndarray | None = None


def normalize_constraint(
    values: np.ndarray,
    valid: np.ndarray,
    schema: dict[str, Any],
) -> np.ndarray:
    norm = schema["long_horizon_constraint"]["normalization"]
    mean = np.asarray(norm["mean"], np.float32)
    std = np.asarray(norm["std"], np.float32)
    output = np.zeros_like(values, dtype=np.float32)
    present = np.asarray(valid, bool)
    output[present] = ((np.asarray(values, np.float32) - mean) / std)[present]
    return output


def inverse_constraint(
    normalized: np.ndarray,
    slot_mask: np.ndarray,
    schema: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    norm = schema["long_horizon_constraint"]["normalization"]
    values = np.asarray(normalized, np.float32) * np.asarray(norm["std"], np.float32)
    values += np.asarray(norm["mean"], np.float32)
    valid = np.broadcast_to(np.asarray(slot_mask, bool)[..., None], values.shape).copy()
    values[~valid] = 0.0
    return values, valid


def project_physical_c0(values: np.ndarray, slot_mask: np.ndarray) -> np.ndarray:
    """Project sampled initial states onto the fixed-slot physical contract."""
    output = np.asarray(values, np.float32).copy()
    slots = np.asarray(slot_mask, bool)
    ego_vx = feature_index(None, "ego_vx_mps")
    output[:, ego_vx] = np.clip(output[:, ego_vx], 0.0, 70.0)
    for name in ("ego_vy_left_mps", "ego_ax_mps2", "ego_ay_left_mps2"):
        index = feature_index(None, name)
        limit = 5.0 if name == "ego_ay_left_mps2" else 10.0
        output[:, index] = np.clip(output[:, index], -limit, limit)
    for slot, name in enumerate(SLOT_NAMES):
        active = slots[:, slot]
        dx = feature_index(name, "rel_x_m")
        dy = feature_index(name, "rel_y_left_m")
        dvx = feature_index(name, "rel_vx_mps")
        ax = feature_index(name, "other_ax_mps2")
        ay = feature_index(name, "other_ay_left_mps2")
        output[active, dx] = (
            np.maximum(output[active, dx], 4.81)
            if "front" in name
            else np.minimum(output[active, dx], -4.81)
        )
        if name.startswith("left"):
            output[active, dy] = np.maximum(output[active, dy], 1.91)
        elif name.startswith("right"):
            output[active, dy] = np.minimum(output[active, dy], -1.91)
        other_vx = output[:, ego_vx] + output[:, dvx]
        output[active, dvx] += np.clip(other_vx[active], 0.0, 70.0) - other_vx[active]
        output[active, ax] = np.clip(output[active, ax], -10.0, 10.0)
        output[active, ay] = np.clip(output[active, ay], -5.0, 5.0)
    return output


def project_constraint(
    values: np.ndarray,
    slot_mask: np.ndarray,
    c0: np.ndarray,
) -> np.ndarray:
    """Enforce finite monotone positions and legal knot speeds."""
    output = np.asarray(values, np.float32).copy()
    slots = np.asarray(slot_mask, bool)
    initial_vx = np.stack(
        [
            c0[:, feature_index(None, "ego_vx_mps")]
            + c0[:, feature_index(name, "rel_vx_mps")]
            for name in SLOT_NAMES
        ],
        axis=1,
    )
    longitudinal = np.maximum.accumulate(
        np.maximum(output[..., (0, 4, 8)], 0.0), axis=-1
    )
    output[..., (0, 4, 8)] = longitudinal
    for index in (2, 6, 10):
        speed = initial_vx + output[..., index]
        output[..., index] += np.clip(speed, 0.0, 70.0) - speed
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
    """Three-factor density `p(M) p(C0|M) p(K|C0,M)`."""

    def __init__(
        self,
        *,
        c0_flow: nn.Module,
        constraint_flow: nn.Module,
        mask_log_prob: torch.Tensor,
        schema: dict[str, Any],
    ) -> None:
        super().__init__()
        self.c0_flow = c0_flow
        self.constraint_flow = constraint_flow
        self.register_buffer("mask_log_prob", mask_log_prob.float())
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

    def log_prob(
        self,
        c0: np.ndarray | ScenarioBatch,
        slot_mask: np.ndarray | None = None,
        k: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        item = c0 if isinstance(c0, ScenarioBatch) else ScenarioBatch(
            np.asarray(c0),
            np.asarray(slot_mask),
            np.asarray(k),
            np.broadcast_to(np.asarray(slot_mask)[..., None], np.asarray(k).shape),
        )
        valid = feature_valid_from_slot_mask(self.schema, item.slot_mask)
        c0_norm = (
            normalize_features(item.c0, valid, self.schema)
            if item.c0_normalized_reference is None
            else np.asarray(item.c0_normalized_reference, np.float32)
        )
        k_norm = (
            normalize_constraint(
                item.trajectory_constraint,
                item.trajectory_constraint_valid,
                self.schema,
            )
            if item.constraint_normalized_reference is None
            else np.asarray(item.constraint_normalized_reference, np.float32)
        )
        device = next(self.parameters()).device
        with torch.no_grad():
            terms = self.log_prob_tensors(
                c0_normalized=torch.from_numpy(c0_norm).float().to(device),
                slot_mask=torch.from_numpy(item.slot_mask).bool().to(device),
                constraint_normalized=torch.from_numpy(k_norm).float().to(device),
            )
        return {name: value.cpu().numpy() for name, value in terms.items()}

    @torch.no_grad()
    def sample_scenarios(self, n: int, scenario_seed: int) -> ScenarioBatch:
        device = next(self.parameters()).device
        root = np.random.SeedSequence(int(scenario_seed))
        mask_seed, k_seed = (int(x.generate_state(1)[0]) for x in root.spawn(2))
        generator = torch.Generator(device=device).manual_seed(mask_seed)
        pattern = torch.multinomial(
            self.mask_log_prob.exp(),
            int(n),
            replacement=True,
            generator=generator,
        )
        bits = 2 ** torch.arange(6, device=device)
        slots = (pattern[:, None].bitwise_and(bits[None])) > 0
        c0_norm = _sample_flow(self.c0_flow, slots.float(), generator)
        return self._sample_given(c0_norm, slots, k_seed)

    @torch.no_grad()
    def sample_scenarios_from_base_randomness(
        self,
        scenario_uniform: np.ndarray,
        c0_base_latent: np.ndarray,
        k_base_latent: np.ndarray,
    ) -> ScenarioBatch:
        """Sample exactly from explicit base variables.

        ``scenario_uniform`` is a U(0, 1) draw for the categorical slot
        structure; C0 and K use the standard-normal base variables of their
        respective normalizing flows.  This is the public mutation boundary
        for rare-event simulation and deliberately keeps projection rules
        identical to ordinary sampling.
        """
        device = next(self.parameters()).device
        uniform = torch.as_tensor(scenario_uniform, dtype=torch.float64, device=device)
        if uniform.ndim != 1 or not torch.isfinite(uniform).all() or (
            (uniform < 0.0) | (uniform >= 1.0)
        ).any():
            raise ValueError("scenario_uniform must be a finite [batch] vector in [0, 1)")
        cumulative = torch.cumsum(self.mask_log_prob.exp().double(), dim=0)
        pattern = torch.searchsorted(cumulative, uniform).clamp_max(len(cumulative) - 1)
        bits = 2 ** torch.arange(6, device=device)
        slots = (pattern[:, None].bitwise_and(bits[None])) > 0
        c0_norm = _sample_flow(
            self.c0_flow,
            slots.float(),
            base_noise=torch.as_tensor(c0_base_latent, device=device),
        )
        context = torch.cat((c0_norm, slots.float()), dim=-1)
        k_norm = _sample_flow(
            self.constraint_flow,
            context,
            base_noise=torch.as_tensor(k_base_latent, device=device),
        ).reshape(-1, 6, 12)
        return self._scenario_from_normalized(c0_norm, slots, k_norm)

    @torch.no_grad()
    def sample_constraints(
        self,
        c0: np.ndarray,
        slot_mask: np.ndarray,
        n: int,
        scenario_seed: int,
    ) -> ScenarioBatch:
        c0 = np.asarray(c0, np.float32)
        slots = np.asarray(slot_mask, bool)
        if c0.ndim == 1:
            c0 = c0[None]
        if slots.ndim == 1:
            slots = slots[None]
        if len(c0) == 1 and int(n) > 1:
            c0 = np.repeat(c0, int(n), axis=0)
        if len(slots) == 1 and int(n) > 1:
            slots = np.repeat(slots, int(n), axis=0)
        if len(c0) != int(n) or len(slots) != int(n):
            raise ValueError("c0 and slot_mask must be singleton or have n rows")
        valid = feature_valid_from_slot_mask(self.schema, slots)
        c0_norm = normalize_features(c0, valid, self.schema)
        device = next(self.parameters()).device
        return self._sample_given(
            torch.from_numpy(c0_norm).float().to(device),
            torch.from_numpy(slots).bool().to(device),
            int(scenario_seed),
        )

    def _sample_given(
        self,
        c0_norm: torch.Tensor,
        slots: torch.Tensor,
        k_seed: int,
    ) -> ScenarioBatch:
        generator = torch.Generator(device=c0_norm.device).manual_seed(int(k_seed))
        context = torch.cat((c0_norm, slots.float()), dim=-1)
        k_norm = _sample_flow(self.constraint_flow, context, generator).reshape(
            -1, 6, 12
        )
        return self._scenario_from_normalized(c0_norm, slots, k_norm)

    def _scenario_from_normalized(
        self,
        c0_norm: torch.Tensor,
        slots: torch.Tensor,
        k_norm: torch.Tensor,
    ) -> ScenarioBatch:
        c0 = inverse_normalize_features(c0_norm.detach().cpu().numpy(), self.schema)
        c0[~feature_valid_from_slot_mask(self.schema, slots.cpu().numpy())] = 0.0
        c0 = project_physical_c0(c0, slots.cpu().numpy())
        constraint, valid = inverse_constraint(
            k_norm.cpu().numpy(), slots.cpu().numpy(), self.schema
        )
        constraint = project_constraint(constraint, slots.cpu().numpy(), c0)
        return ScenarioBatch(
            c0,
            slots.cpu().numpy(),
            constraint,
            valid,
            c0_norm.cpu().numpy(),
            k_norm.cpu().numpy(),
        )


def build_direct_model(
    schema: dict[str, Any],
    model_cfg: dict[str, Any],
    repo_root: str | Path,
    mask_log_prob: torch.Tensor,
) -> DirectScenarioFlow:
    c0 = build_maf_flow(
        num_features=40,
        context_features=6,
        model_cfg=dict(model_cfg["c0_flow"]),
        repo_root=repo_root,
    )
    constraint = build_maf_flow(
        num_features=6 * len(KNOT_FEATURE_NAMES),
        context_features=46,
        model_cfg=dict(model_cfg["constraint_flow"]),
        repo_root=repo_root,
    )
    return DirectScenarioFlow(
        c0_flow=c0,
        constraint_flow=constraint,
        mask_log_prob=mask_log_prob,
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
    if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not a direct scenario condition flow")
    state = payload["state_dict"]
    model = build_direct_model(
        payload["schema"], payload["model_cfg"], repo_root, state["mask_log_prob"]
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model, payload

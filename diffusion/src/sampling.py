"""Explicit latent-to-open-loop-trajectory decoding utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .data import smooth_position_residual, states_from_smooth_positions
from .model import BackgroundTrajectoryDiffusion


@torch.no_grad()
def decode_background_latents(
    model: BackgroundTrajectoryDiffusion,
    condition: torch.Tensor,
    target_mask: torch.Tensor,
    c0_background_states: np.ndarray,
    contract: dict[str, Any],
    latents: torch.Tensor,
    *,
    trajectory_reference: np.ndarray,
    inference_steps: int,
    guidance_scale: float = 1.0,
    x0_clip_abs: float | None = None,
) -> dict[str, np.ndarray]:
    """Decode explicit path latents into smooth Cartesian trajectories.

    This is the public subset-simulation boundary: the condition is fixed and
    every supplied latent deterministically maps to one six-background path.
    """
    if condition.shape != (1, model.config.condition_dim):
        raise ValueError(
            f"condition must have shape {(1, model.config.condition_dim)}, "
            f"got {tuple(condition.shape)}"
        )
    expected_latent = (
        model.config.horizon_steps,
        model.config.target_dim,
    )
    if latents.ndim != 3 or tuple(latents.shape[1:]) != expected_latent:
        raise ValueError(
            f"latents must have shape [draws, {expected_latent[0]}, "
            f"{expected_latent[1]}], got {tuple(latents.shape)}"
        )
    draws = latents.shape[0]
    condition = condition.to(device=latents.device, dtype=latents.dtype)
    target_mask = target_mask.to(device=latents.device)
    expanded_condition = condition.expand(draws, -1)
    expanded_mask = target_mask.expand(draws, -1, -1)
    normalized = model.sample_ddim(
        expanded_condition,
        expanded_mask,
        inference_steps=int(inference_steps),
        initial_noise=latents,
        x0_clip_abs=x0_clip_abs,
        guidance_scale=float(guidance_scale),
    )
    active = target_mask[0, 0].reshape(6, 2)[:, 0].cpu().numpy().astype(bool)
    residual_mean = np.asarray(contract["position_residual"]["mean"], np.float32)
    residual_std = np.asarray(contract["position_residual"]["std"], np.float32)
    residual = normalized.cpu().numpy().reshape(draws, 149, 6, 2)
    residual = residual * residual_std + residual_mean
    residual = smooth_position_residual(residual)
    reference = np.asarray(trajectory_reference, np.float32)
    if reference.shape != (149, 6, 2):
        raise ValueError("trajectory_reference must have shape [149, 6, 2]")
    positions = (reference[None] + residual) * active[None, None, :, None]
    initial = np.asarray(c0_background_states, dtype=np.float32)
    if initial.shape != (6, 6):
        raise ValueError(f"c0_background_states must be [6, 6], got {initial.shape}")
    expanded_initial = np.broadcast_to(initial, (draws, 6, 6))
    states = states_from_smooth_positions(expanded_initial, positions)
    states *= active[None, None, :, None]
    return {
        "normalized_position_residual": normalized.cpu().numpy(),
        "position_residual_m": residual,
        "background_states": states,
        "background_positions_xy": states[..., :2],
        "active_background": active,
    }

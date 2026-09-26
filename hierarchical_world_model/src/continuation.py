"""Explicit reference-plan continuation for an existing online world.

The caller supplies a new future plan only after the previous plan is fully
executed. This preserves every realized state, controller memory and past
Diffusion frame. It does not generate a future plan or claim Diffusion accuracy
beyond its trained 149-frame horizon.
"""

from __future__ import annotations

import torch

from .highway import HighwayEnvClosedLoopWorld


def append_future_reference(
    world: HighwayEnvClosedLoopWorld,
    future_xy: torch.Tensor,
) -> None:
    """Append [batch,future_frames,6,2] map-frame NPC positions at a boundary."""
    if world.reference is None or world.states is None:
        raise RuntimeError("reset the online world before appending a future plan")
    if world.reference_index != world.reference.shape[1]:
        raise ValueError("future reference may only be appended after plan exhaustion")
    if world.response_agent_innovations is None:
        raise RuntimeError("world response innovations are unavailable")
    tail = torch.as_tensor(future_xy, device=world.device, dtype=world.reference.dtype)
    if (
        tail.ndim != 4
        or tail.shape[0] != world.reference.shape[0]
        or tail.shape[1] < 1
        or tail.shape[2:] != (6, 2)
    ):
        raise ValueError("future_xy must be [batch,positive_frames,6,2]")
    if not bool(torch.isfinite(tail).all()):
        raise ValueError("future_xy must contain only finite positions")
    remaining = world.response_agent_innovations.shape[1] - world.reference_index
    if tail.shape[1] > remaining:
        raise ValueError("future plan exceeds pre-sampled response innovations")
    world.reference = torch.cat((world.reference, tail.detach().clone()), dim=1)

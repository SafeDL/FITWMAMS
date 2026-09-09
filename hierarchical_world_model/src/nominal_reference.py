"""Prefix-only executable nominal-world cache for response experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .composition import HierarchicalWorldSampler, SampledWorldBatch


@dataclass(frozen=True)
class NominalReference:
    """Executed reference, not a log-future completion or an experiment label."""
    states: torch.Tensor             # [B,149,7,6], post-step physical states
    background_actions: torch.Tensor # [B,149,6,2], exact plant inputs
    ego_actions: torch.Tensor        # [B,149,2], fixed nominal IDM inputs
    initial_states: torch.Tensor     # [B,7,6], physical state before action 0
    exogenous_digest: str | None = None


@torch.no_grad()
def build_nominal_reference(
    sampler: HierarchicalWorldSampler, sample: SampledWorldBatch, *,
    idm_config: dict[str, Any], steps: int = 149,
) -> NominalReference:
    """Run one frozen all-slot nominal world under its fixed IDM ego policy.

    ``sample`` must have come from explicit base randomness.  Thus no source
    in this routine accepts highD post-prefix state, target action, future knot
    or pending actual ego command.
    """
    if sample.exogenous_state is None:
        raise ValueError("nominal references require explicit exogenous state")
    # This is the common executable baseline: actions are constrained before
    # the plant advances, and the stored states are therefore re-simulated.
    world = sampler.create_world(sample, idm_config=idm_config, controller="common_physics")
    initial = world.observe()["agent_states"]
    states, background, ego = [], [], []
    for _ in range(int(steps)):
        nominal_ego = world.idm_actions()
        transition = world.advance_response(nominal_ego)
        states.append(transition["agent_state_frames"])
        background.append(transition["background_actions"])
        ego.append(transition["ego_actions"])
    return NominalReference(
        states=torch.cat(states, dim=1).detach(),
        background_actions=torch.cat(background, dim=1).detach(),
        ego_actions=torch.cat(ego, dim=1).detach(),
        initial_states=initial.detach(),
    )

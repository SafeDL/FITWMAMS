"""Load one frozen traffic model for all resets of a single ADS environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from hierarchical_world_model.src.highway import HighwayEnvClosedLoopWorld
from hierarchical_world_model.src.stochastic_drivers.online import (
    DEFAULT_MA_IDM_POSTERIOR,
    OnlineMAIDMController,
    sample_population_theta_from_world,
)
from hierarchical_world_model.src.train import load_checkpoint
from hierarchical_world_model.src.response_gate import ResponseGate

from .env import WorldEvent


class FrozenWorldFactory:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
        responsive_npcs: bool = True,
        posterior: str | Path = DEFAULT_MA_IDM_POSTERIOR,
        response_gate_checkpoint: str | Path | None = None,
        behavior_randomization: str = "W2",
        controller_kwargs: dict[str, Any] | None = None,
        spatial_progress_reference: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.model, _ = load_checkpoint(checkpoint, device=self.device)
        self.model.eval()
        self.responsive_npcs = bool(responsive_npcs)
        self.posterior = Path(posterior)
        self.behavior_randomization = str(behavior_randomization)
        self.controller_kwargs = dict(controller_kwargs or {})
        self.spatial_progress_reference = bool(spatial_progress_reference)
        if self.behavior_randomization not in {"W0", "W1", "W2"}:
            raise ValueError("behavior_randomization must be W0, W1, or W2")
        self.response_gate = (
            None
            if response_gate_checkpoint is None
            else ResponseGate.load(response_gate_checkpoint, device=self.device)
        )

    def __call__(self, event: WorldEvent) -> HighwayEnvClosedLoopWorld:
        controller = None
        if self.responsive_npcs:
            theta = sample_population_theta_from_world(
                self.posterior,
                event.exogenous_state,
                use_driver_latent=self.behavior_randomization != "W0",
            )
            controller = OnlineMAIDMController(
                theta,
                maneuver_latent=(
                    event.exogenous_state.maneuver_latent
                    if self.behavior_randomization == "W2" else None
                ),
                response_gate=self.response_gate,
                **self.controller_kwargs,
            )
        return HighwayEnvClosedLoopWorld(
            self.model,
            device=self.device,
            controller=controller,
            spatial_progress_reference=self.spatial_progress_reference,
        )

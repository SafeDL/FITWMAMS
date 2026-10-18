"""Reuse frozen highD NPC parameters for all resets of one ADS environment."""

from pathlib import Path
import numpy as np
import torch
from hierarchical_world_model.src.highway import HighwayEnvClosedLoopWorld
from hierarchical_world_model.src.takeover import TakeoverConfig, TakeoverController
from hierarchical_world_model.src.tracking import TrackingConfig
from hierarchical_world_model.src.core.utils import load_yaml
from hierarchical_world_model.src.config import WorldModelConfig
from hierarchical_world_model.src.shared_evaluation import load_shared_components
from hierarchical_world_model.src.stochastic_drivers.idm_population import sample_theta
from hierarchical_world_model.src.stochastic_drivers.online import (
    calibrated_mobil_parameters,
)

DEFAULT_WORLD_CONFIG = (
    Path(__file__).resolve().parents[2] / "hierarchical_world_model/config.yaml"
)


class FrozenWorldFactory:
    def __init__(
        self,
        config_path=DEFAULT_WORLD_CONFIG,
        *,
        device="cpu",
        responsive_npcs=True,
        behavior_randomization="W2",
        controller_kwargs=None,
        tracking_config=None,
        takeover_config=None
    ):
        self.device = torch.device(device)
        self.responsive_npcs = bool(responsive_npcs)
        self.behavior_randomization = str(behavior_randomization)
        if self.behavior_randomization not in {"W0", "W1", "W2"}:
            raise ValueError("behavior_randomization must be W0, W1, or W2")
        settings = load_yaml(config_path)
        self.cfg = WorldModelConfig(**settings["dynamics"])
        self.population, fit = load_shared_components()
        self.controller_kwargs = {
            **calibrated_mobil_parameters(fit["mobil"]),
            **(controller_kwargs or {}),
        }
        self.tracking_config = tracking_config or TrackingConfig(**settings["tracking"])
        self.takeover_config = takeover_config or TakeoverConfig(**settings["takeover"])

    def __call__(self, event):
        controller = None
        if self.responsive_npcs:
            latent = event.exogenous_state.driver_latent
            theta = sample_theta(
                self.population,
                (
                    latent
                    if self.behavior_randomization != "W0"
                    else np.zeros_like(latent)
                ),
            )
            controller = TakeoverController(
                theta,
                maneuver_latent=(
                    event.exogenous_state.maneuver_latent
                    if self.behavior_randomization == "W2"
                    else None
                ),
                takeover_config=self.takeover_config,
                **self.controller_kwargs
            )
        return HighwayEnvClosedLoopWorld(
            cfg=self.cfg,
            device=self.device,
            controller=controller,
            tracking_config=self.tracking_config,
        )

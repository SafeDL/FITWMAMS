"""Deterministic control realization used only by M1 START mode."""
from __future__ import annotations

import torch

from .initial_behavior_anchor import BehaviorAnchorControlPlan


class StartModeControl(BehaviorAnchorControlPlan):
    """Expand Flow's first-second summaries into bounded START controls.

    It has no trainable parameters.  The fixed interpolation and smoothness
    buffers make the same `(S0, B0)` produce the same 25 START controls;
    `AnchorResidualController` supplies the small graph-dependent correction.
    """

    def __init__(self, physics_steps: int = 25, knots: int = 5) -> None:
        super().__init__(physics_steps=physics_steps, knots=knots)
    def forward(self, initial_states: torch.Tensor, anchor: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return super().forward(initial_states, anchor, valid)

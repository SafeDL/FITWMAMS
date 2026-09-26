"""ADS-side control policies used only by reproducible intervention audits."""

from __future__ import annotations

import torch

from hierarchical_world_model.src.execution import hold_current_ego_action


class MaintainEntrySpeedADS:
    """Maintain the realized entry speed during a semantic lane-change action.

    The target speed is captured when the lane-change window begins. This keeps
    an incidental logged initial deceleration from stopping the ADS partway
    through the lateral manoeuvre; it is not an acceleration-dose experiment.
    """

    def __init__(self, *, start_frame: int = 25, speed_gain: float = 1.5) -> None:
        if start_frame < 0 or speed_gain <= 0.0:
            raise ValueError("start_frame must be nonnegative and speed_gain positive")
        self.start_frame = int(start_frame)
        self.speed_gain = float(speed_gain)
        self.target_speed: torch.Tensor | None = None

    def __call__(self, context: dict[str, torch.Tensor | int]) -> torch.Tensor:
        action = hold_current_ego_action(context).clone()
        index = int(context["reference_index"])
        if index < self.start_frame:
            return action
        states = context["agent_states"]
        if not isinstance(states, torch.Tensor):
            raise TypeError("agent_states must be a tensor")
        speed = torch.linalg.vector_norm(states[:, 0, 2:4], dim=-1)
        if self.target_speed is None:
            self.target_speed = speed.detach().clone()
        target = self.target_speed.to(speed)
        action[:, 0] = (self.speed_gain * (target - speed)).clamp(-8.0, 4.0)
        return action

"""IDM policy marker for formal HighwayEnv execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tools.idm_ego import IDM_PARAMETER_KEYS


@dataclass(frozen=True)
class HighwayEnvIDMPolicy:
    """Request that the world execute its ego with HighwayEnv ``IDMVehicle``.

    The policy intentionally does not reimplement IDM in torch. The formal
    world detects this marker, instantiates the local HighwayEnv vehicle and
    uses that vehicle's action at every response boundary.
    """

    config: dict[str, Any]

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "HighwayEnvIDMPolicy":
        """Keep only the HighwayEnv IDM parameters accepted by the marker."""
        allowed = {"target_speed", "enable_lane_change", *IDM_PARAMETER_KEYS}
        return cls({name: values[name] for name in allowed if name in values})

    @property
    def highway_env_idm_config(self) -> dict[str, Any]:
        """Expose a defensive copy for ``HighwayEnvTraffic`` construction."""
        return dict(self.config)

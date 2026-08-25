"""Hierarchical long-horizon and reactive traffic world model.

The public classes are loaded lazily so lightweight protocol/config tooling does
not import the training stack (and its optional accelerator dependencies).
"""

__all__ = [
    "HierarchicalWorldSampler",
    "DiffusionGuidedHiQR",
    "WorldModelConfig",
    "WorldExogenousState",
    "WorldRollout",
    "rollout_world",
]


def __getattr__(name: str):
    if name == "HierarchicalWorldSampler":
        from .src.composition import HierarchicalWorldSampler
        return HierarchicalWorldSampler
    if name == "DiffusionGuidedHiQR":
        from .src.model import DiffusionGuidedHiQR
        return DiffusionGuidedHiQR
    if name == "WorldModelConfig":
        from .src.config import WorldModelConfig
        return WorldModelConfig
    if name == "WorldExogenousState":
        from .src.randomness import WorldExogenousState
        return WorldExogenousState
    if name in {"WorldRollout", "rollout_world"}:
        from .src.execution import WorldRollout, rollout_world
        return {"WorldRollout": WorldRollout, "rollout_world": rollout_world}[name]
    raise AttributeError(name)

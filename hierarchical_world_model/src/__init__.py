"""Model, data and simulation components, exported lazily."""

__all__ = [
    "HierarchicalWorldSampler",
    "DiffusionGuidedHiQR",
    "WorldModelConfig",
    "WorldExogenousState",
    "WorldRollout",
    "rollout_world",
]


def __getattr__(name: str):
    if name == "WorldModelConfig":
        from .config import WorldModelConfig
        return WorldModelConfig
    if name == "HierarchicalWorldSampler":
        from .composition import HierarchicalWorldSampler
        return HierarchicalWorldSampler
    if name == "DiffusionGuidedHiQR":
        from .model import DiffusionGuidedHiQR
        return DiffusionGuidedHiQR
    if name == "WorldExogenousState":
        from .randomness import WorldExogenousState
        return WorldExogenousState
    if name in {"WorldRollout", "rollout_world"}:
        from .execution import WorldRollout, rollout_world
        return {"WorldRollout": WorldRollout, "rollout_world": rollout_world}[name]
    raise AttributeError(name)

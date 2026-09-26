"""Hierarchical traffic world and single-pass ADS simulation interface."""

__all__ = [
    "FactualDynamicsModel",
    "DiffusionGuidedHiQR",
    "WorldModelConfig",
    "HierarchicalWorldSampler",
    "rollout_world",
    "OnlineAccelerationWindowPolicy",
    "OnlineSemanticLaneChangePolicy",
]


def __getattr__(name: str):
    if name == "FactualDynamicsModel":
        from .src.model import FactualDynamicsModel
        return FactualDynamicsModel
    if name == "DiffusionGuidedHiQR":
        from .src.model import DiffusionGuidedHiQR
        return DiffusionGuidedHiQR
    if name == "WorldModelConfig":
        from .src.config import WorldModelConfig
        return WorldModelConfig
    if name == "HierarchicalWorldSampler":
        from .src.composition import HierarchicalWorldSampler
        return HierarchicalWorldSampler
    if name == "rollout_world":
        from .src.execution import rollout_world
        return rollout_world
    if name in {"OnlineAccelerationWindowPolicy", "OnlineSemanticLaneChangePolicy"}:
        from .src import ads_interventions
        return getattr(ads_interventions, name)
    raise AttributeError(name)

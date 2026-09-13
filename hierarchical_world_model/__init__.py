"""Maintained factual-transition and CIH-WM public interface."""

__all__ = [
    "CausalInfluenceHierarchicalWorldModel",
    "FactualDynamicsModel",
    "DiffusionGuidedHiQR",
    "WorldModelConfig",
]


def __getattr__(name: str):
    if name == "CausalInfluenceHierarchicalWorldModel":
        from .src.cih_model import CausalInfluenceHierarchicalWorldModel
        return CausalInfluenceHierarchicalWorldModel
    if name == "FactualDynamicsModel":
        from .src.model import FactualDynamicsModel
        return FactualDynamicsModel
    if name == "DiffusionGuidedHiQR":
        from .src.model import DiffusionGuidedHiQR
        return DiffusionGuidedHiQR
    if name == "WorldModelConfig":
        from .src.config import WorldModelConfig
        return WorldModelConfig
    raise AttributeError(name)

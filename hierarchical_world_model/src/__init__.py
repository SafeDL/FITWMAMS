"""Maintained factual-transition and CIH-WM components, exported lazily."""

__all__ = [
    "CausalInfluenceHierarchicalWorldModel",
    "FactualDynamicsModel",
    "DiffusionGuidedHiQR",
    "WorldModelConfig",
]


def __getattr__(name: str):
    if name == "WorldModelConfig":
        from .config import WorldModelConfig
        return WorldModelConfig
    if name == "CausalInfluenceHierarchicalWorldModel":
        from .cih_model import CausalInfluenceHierarchicalWorldModel
        return CausalInfluenceHierarchicalWorldModel
    if name == "FactualDynamicsModel":
        from .model import FactualDynamicsModel
        return FactualDynamicsModel
    if name == "DiffusionGuidedHiQR":
        from .model import DiffusionGuidedHiQR
        return DiffusionGuidedHiQR
    raise AttributeError(name)

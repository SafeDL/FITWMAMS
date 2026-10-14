"""Maintained hierarchical factual-transition components, exported lazily."""

__all__ = [
    "FactualDynamicsModel",
    "WorldModelConfig",
]


def __getattr__(name: str):
    if name == "WorldModelConfig":
        from .config import WorldModelConfig

        return WorldModelConfig
    if name == "FactualDynamicsModel":
        from .model import FactualDynamicsModel

        return FactualDynamicsModel
    raise AttributeError(name)

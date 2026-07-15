"""Core modules for the FiT-AMS traffic behavior world model."""

from .schema import (
    AGENT_NAMES,
    AGENT_STATE_FEATURES,
    ACTION_FEATURES,
    MODE_NAMES,
    SLOT_NAMES,
)
from .environment import CATKBackgroundEnvironment, WorldSamplingConfig

__all__ = [
    "ACTION_FEATURES",
    "AGENT_NAMES",
    "AGENT_STATE_FEATURES",
    "MODE_NAMES",
    "SLOT_NAMES",
    "CATKBackgroundEnvironment",
    "WorldSamplingConfig",
]
"""World-model implementations and shared interfaces.

Imports stay lazy so cache/preparation tools that only require NumPy/Pandas do
not need to import PyTorch.  Import semi-Markov classes from their modules.
"""

"""Causal, evaluator-supplied semantic goals for the T4a protocol."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

BEHAVIOR_CONDITION_MODES = ("brake", "recover", "lane_left", "lane_right")


def append_behavior_condition(
    features: np.ndarray,
    behavior_condition: Mapping[str, Any] | None,
) -> np.ndarray:
    """Append fixed goal channels to causal per-agent state features.

    A condition never carries a logged future, event onset, or endpoint.  It
    only specifies which background NPC should attempt one requested behavior.
    """
    values = np.asarray(features, np.float32)
    if values.ndim != 3:
        raise ValueError("features must be [history,agents,feature]")
    extra = np.zeros((*values.shape[:2], len(BEHAVIOR_CONDITION_MODES)), np.float32)
    if behavior_condition is None:
        return np.concatenate((values, extra), axis=-1)
    if not isinstance(behavior_condition, Mapping):
        raise ValueError("behavior_condition must be a mapping or None")
    if (
        "target_agent_index" not in behavior_condition
        or "mode" not in behavior_condition
    ):
        raise ValueError("behavior_condition requires target_agent_index and mode")
    target = int(behavior_condition["target_agent_index"])
    if not 0 < target < values.shape[1]:
        raise ValueError(
            "behavior_condition target_agent_index must identify a background NPC"
        )
    mode = str(behavior_condition["mode"])
    if mode not in BEHAVIOR_CONDITION_MODES:
        raise ValueError(f"unsupported behavior condition mode: {mode!r}")
    extra[:, target, BEHAVIOR_CONDITION_MODES.index(mode)] = 1.0
    return np.concatenate((values, extra), axis=-1)

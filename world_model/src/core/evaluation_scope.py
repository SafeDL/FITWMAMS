"""Shared full-population scope for highD world-model benchmarks.

All canonical background slots, including ``same_rear``, remain available to
inference, simulation, factual metrics, and risk metrics.  The scope still
provides an explicit contract so historical masked artifacts cannot be mixed
with future all-slot results.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .schema import SLOT_NAMES, slot_index


EVALUATION_SCOPE_SCHEMA = "highd_all_background"
EXCLUDED_EVALUATION_SLOTS: tuple[str, ...] = ()


def evaluation_scope_contract() -> dict[str, Any]:
    """Return JSON-ready provenance for every evaluation artifact."""
    return {
        "schema": EVALUATION_SCOPE_SCHEMA,
        "excluded_background_slots": list(EXCLUDED_EVALUATION_SLOTS),
        "training_population_modified": False,
        "semantics": (
            "all canonical background slots, including same_rear, enter model "
            "inference, simulation, metrics, risk, and visualization"
        ),
    }


def _indices(excluded_slots: Iterable[str]) -> tuple[int, ...]:
    return tuple(slot_index(str(name)) for name in excluded_slots)


def scoped_slot_mask(
    slot_mask: np.ndarray | torch.Tensor,
    *,
    excluded_slots: Iterable[str] = EXCLUDED_EVALUATION_SLOTS,
) -> np.ndarray | torch.Tensor:
    """Copy a ``[...,6]`` mask and invalidate excluded background slots."""
    if slot_mask.shape[-1] != len(SLOT_NAMES):
        raise ValueError(f"slot_mask must end in {len(SLOT_NAMES)} slots")
    output = (
        slot_mask.clone()
        if isinstance(slot_mask, torch.Tensor)
        else np.array(slot_mask, bool, copy=True)
    )
    for index in _indices(excluded_slots):
        output[..., index] = False
    return output


def scoped_agent_valid(
    agent_valid: np.ndarray | torch.Tensor,
    *,
    excluded_slots: Iterable[str] = EXCLUDED_EVALUATION_SLOTS,
) -> np.ndarray | torch.Tensor:
    """Copy a ``[...,7]`` ego-plus-background mask and apply the scope."""
    if agent_valid.shape[-1] != len(SLOT_NAMES) + 1:
        raise ValueError(f"agent_valid must end in {len(SLOT_NAMES) + 1} agents")
    output = (
        agent_valid.clone()
        if isinstance(agent_valid, torch.Tensor)
        else np.array(agent_valid, bool, copy=True)
    )
    for index in _indices(excluded_slots):
        output[..., index + 1] = False
    return output


def scoped_canonical_trajectory(
    states: np.ndarray | torch.Tensor,
    valid: np.ndarray | torch.Tensor,
    *,
    excluded_slots: Iterable[str] = EXCLUDED_EVALUATION_SLOTS,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """Copy and scope canonical ``[...,7,6]`` states and ``[...,7]`` validity."""
    if states.shape[-2:] != (len(SLOT_NAMES) + 1, 6):
        raise ValueError("states must end in [7,6]")
    scoped_valid = scoped_agent_valid(valid, excluded_slots=excluded_slots)
    scoped_states = (
        states.clone()
        if isinstance(states, torch.Tensor)
        else np.array(states, copy=True)
    )
    for index in _indices(excluded_slots):
        scoped_states[..., index + 1, :] = 0
    return scoped_states, scoped_valid


def require_evaluation_scope(config: dict[str, Any]) -> None:
    """Reject a run whose declared scope could silently differ from the code."""
    declared = config.get("evaluation_scope")
    expected = evaluation_scope_contract()
    if not isinstance(declared, dict):
        raise ValueError("configuration must declare evaluation_scope")
    if declared.get("schema") != expected["schema"]:
        raise ValueError("configuration evaluation_scope schema is not supported")
    if (
        tuple(declared.get("excluded_background_slots", ()))
        != EXCLUDED_EVALUATION_SLOTS
    ):
        raise ValueError("configuration evaluation scope does not match the full-population contract")


def require_scoped_evt_model(model_path: str | Path) -> dict[str, Any]:
    """Require the EVT calibration beside ``model_path`` to use this scope."""
    path = Path(model_path)
    summary_path = path.with_name("natural_evt_summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"scoped EVT summary is required beside the model: {summary_path}"
        )
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    scope = summary.get("evaluation_scope")
    if not isinstance(scope, dict) or (
        tuple(scope.get("excluded_risk_slots", ()))
        != EXCLUDED_EVALUATION_SLOTS
    ):
        raise ValueError("EVT model was not calibrated on the full background population")
    return summary

from __future__ import annotations

import numpy as np
import torch

from IDM_subset.src.metrics import collision_and_min_gap
from hierarchical_world_model.src.execution import trajectory_event_risk
from world_model.src.core.evaluation_scope import (
    evaluation_scope_contract,
    scoped_agent_valid,
    scoped_canonical_trajectory,
    scoped_slot_mask,
)


def test_scope_excludes_only_same_rear_without_mutating_inputs() -> None:
    slots = np.ones((2, 6), bool)
    agents = torch.ones((2, 7), dtype=torch.bool)
    scoped_slots = scoped_slot_mask(slots)
    scoped_agents = scoped_agent_valid(agents)
    assert slots.all() and bool(agents.all())
    assert scoped_slots[:, 1].tolist() == [False, False]
    assert scoped_slots[:, [0, 2, 3, 4, 5]].all()
    assert not bool(scoped_agents[:, 2].any())
    assert bool(scoped_agents[:, [0, 1, 3, 4, 5, 6]].all())


def test_scope_zeros_same_rear_for_model_input_and_keeps_training_source() -> None:
    states = np.ones((3, 150, 7, 6), np.float32)
    valid = np.ones((3, 150, 7), bool)
    scoped_states, scoped_valid = scoped_canonical_trajectory(states, valid)
    assert states[..., 2, :].all() and valid[..., 2].all()
    assert not scoped_states[..., 2, :].any()
    assert not scoped_valid[..., 2].any()
    assert scoped_states[..., [0, 1, 3, 4, 5, 6], :].all()


def test_same_rear_only_collision_is_absent_from_metrics_and_risk() -> None:
    states = np.zeros((1, 2, 7, 6), np.float32)
    states[..., 0, 2] = 20.0
    states[..., 2, 0] = -1.0
    states[..., 2, 2] = 25.0
    valid = np.zeros((1, 7), bool)
    valid[:, (0, 2)] = True
    collision, gap = collision_and_min_gap(states, valid)
    risk = trajectory_event_risk(states, valid)
    assert not collision[0]
    assert gap[0] == 1_000.0
    assert risk[0] == 0.0


def test_scope_contract_records_evaluation_only_population_change() -> None:
    contract = evaluation_scope_contract()
    assert contract["excluded_background_slots"] == ["same_rear"]
    assert contract["training_population_modified"] is False

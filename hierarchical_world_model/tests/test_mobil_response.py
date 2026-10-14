from __future__ import annotations

import torch

from hierarchical_world_model.src.influence_graph import build_dynamic_interaction_relations
from hierarchical_world_model.src.stochastic_drivers.mobil_response import (
    evaluate_mobil_candidates,
)


THETA = torch.tensor([33.0, 2.0, 1.2, 1.4, 2.0]).view(1, 1, 5).expand(1, 6, 5)


def _map() -> tuple[torch.Tensor, torch.Tensor]:
    maps = torch.zeros(1, 3, 5, 6)
    maps[..., 0] = torch.linspace(-100.0, 100.0, 5)
    maps[0, :, :, 1] = torch.tensor((-3.6, 0.0, 3.6))[:, None]
    maps[..., 2] = 1.0
    maps[..., 4] = 3.6
    return maps, torch.ones(1, 3, 5, dtype=torch.bool)


def test_mobil_selects_clear_side_for_persistent_slow_leader() -> None:
    states = torch.zeros(1, 7, 6)
    valid = torch.zeros(1, 7, dtype=torch.bool)
    valid[0, 1:5] = True
    states[0, 1, :3] = torch.tensor((0.0, 0.0, 25.0))
    states[0, 2, :3] = torch.tensor((14.0, 0.0, 10.0))
    states[0, 3, :3] = torch.tensor((35.0, 3.6, 25.0))
    # A fast, close rear vehicle makes the +y target fail MOBIL safety.
    states[0, 4, :3] = torch.tensor((-4.0, 3.6, 36.0))
    maps, map_valid = _map()
    relations = build_dynamic_interaction_relations(states, valid, maps, map_valid)
    decision = evaluate_mobil_candidates(states, valid, THETA, relations)
    assert decision.selected[0, 0]
    assert decision.best_side[0, 0].item() == -1
    assert decision.safe[0, 0, 0]
    assert not decision.safe[0, 0, 1]


def test_mobil_rejects_target_that_forces_excess_rear_braking() -> None:
    states = torch.zeros(1, 7, 6)
    valid = torch.zeros(1, 7, dtype=torch.bool)
    valid[0, 1:5] = True
    states[0, 1, :3] = torch.tensor((0.0, 0.0, 20.0))
    states[0, 2, :3] = torch.tensor((20.0, 0.0, 8.0))
    states[0, 3, :3] = torch.tensor((-2.0, -3.6, 38.0))
    states[0, 4, :3] = torch.tensor((-2.0, 3.6, 38.0))
    maps, map_valid = _map()
    relations = build_dynamic_interaction_relations(states, valid, maps, map_valid)
    decision = evaluate_mobil_candidates(
        states, valid, THETA, relations, safe_braking_mps2=3.0
    )
    assert not decision.safe[0, 0].any()
    assert not decision.selected[0, 0]


def test_mobil_accepts_actor_local_tensor_parameters() -> None:
    states = torch.zeros(1, 7, 6)
    valid = torch.zeros(1, 7, dtype=torch.bool)
    valid[0, 1:4] = True
    states[0, 1, :3] = torch.tensor((0.0, 0.0, 25.0))
    states[0, 2, :3] = torch.tensor((14.0, 0.0, 10.0))
    states[0, 3, :3] = torch.tensor((35.0, -3.6, 25.0))
    maps, map_valid = _map()
    relations = build_dynamic_interaction_relations(states, valid, maps, map_valid)
    low_threshold = evaluate_mobil_candidates(
        states,
        valid,
        THETA,
        relations,
        minimum_incentive_mps2=torch.zeros(1, 6),
    )
    high_threshold = evaluate_mobil_candidates(
        states,
        valid,
        THETA,
        relations,
        minimum_incentive_mps2=torch.full((1, 6), 1.0e4),
    )
    assert low_threshold.selected[0, 0]
    assert not high_threshold.selected.any()

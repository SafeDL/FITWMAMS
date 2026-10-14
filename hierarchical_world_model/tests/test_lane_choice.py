"""Causal lane-choice features use the current physical neighbor relation."""

import torch

from hierarchical_world_model.src.influence_graph import (
    build_dynamic_interaction_relations,
)
from hierarchical_world_model.src.stochastic_drivers.lane_choice import (
    lane_choice_features,
)


def test_no_current_leader_does_not_use_an_unrelated_ego_speed():
    states = torch.zeros(1, 7, 6)
    states[0, 0, :4] = torch.tensor([10.0, 3.6, 5.0, 0.0])
    states[0, 1, :4] = torch.tensor([0.0, 0.0, 20.0, 0.0])
    valid = torch.zeros(1, 7, dtype=torch.bool)
    valid[:, :2] = True
    maps = torch.zeros(1, 3, 8, 6)
    maps[..., 1] = torch.tensor([-3.6, 0.0, 3.6])[None, :, None]
    maps[..., 4] = 3.6
    relation = build_dynamic_interaction_relations(
        states,
        valid,
        maps,
        torch.ones(1, 3, 8, dtype=torch.bool),
    )
    features = lane_choice_features(states, torch.full((1, 7), 4.8), relation)
    assert relation.leader_index[0, 1] == -1
    torch.testing.assert_close(features[0, 0, :, 3], torch.zeros(2))

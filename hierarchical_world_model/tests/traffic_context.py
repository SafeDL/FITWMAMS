from dataclasses import replace

import torch

from hierarchical_world_model.src.config import WorldModelConfig
from hierarchical_world_model.src.influence_graph import (
    build_dynamic_interaction_relations,
)
from hierarchical_world_model.src.reaction_controller import ReactionControllerContext

THETA = torch.tensor([33.0, 2.0, 1.2, 1.4, 2.0]).expand(6, -1).clone()


def context():
    states = torch.zeros(1, 7, 6)
    states[0, :, 0] = torch.tensor([30.0, 0.0, -35.0, 50.0, -50.0, 50.0, -50.0])
    states[0, :, 1] = torch.tensor([0.0, 0.0, 0.0, 3.6, 3.6, -3.6, -3.6])
    states[..., 2] = 20.0
    valid = torch.ones(1, 7, dtype=torch.bool)
    maps = torch.zeros(1, 3, 8, 6)
    maps[..., 0] = torch.linspace(-200, 200, 8)
    maps[0, :, :, 1] = torch.tensor([-3.6, 0.0, 3.6])[:, None]
    maps[..., 4] = 3.6
    map_valid = torch.ones(1, 3, 8, dtype=torch.bool)
    path = states[:, None, 1:, :2] + torch.arange(1, 51)[
        None, :, None, None
    ] * torch.tensor([0.8, 0.0])
    base = torch.zeros(1, 1, 6, 2)
    return ReactionControllerContext(
        history=states[:, None],
        history_valid=valid[:, None],
        current=states,
        current_valid=valid,
        base_actions=base,
        cfg=WorldModelConfig(),
        planned_current_xy=states[:, 1:, :2],
        planned_path_xy=path,
        map_polylines=maps,
        map_polyline_valid=map_valid,
        vehicle_dimensions=torch.tensor([4.8, 1.8]).expand(1, 7, 2),
        interaction_relations=build_dynamic_interaction_relations(
            states, valid, maps, map_valid
        ),
        policy_standard_normal=torch.zeros(1, 6, 2),
        policy_initial_standard_normal=torch.zeros(1, 6, 2),
    )


def change_context(original, states):
    return replace(
        original,
        current=states,
        history=states[:, None],
        interaction_relations=build_dynamic_interaction_relations(
            states,
            original.current_valid,
            original.map_polylines,
            original.map_polyline_valid,
        ),
    )

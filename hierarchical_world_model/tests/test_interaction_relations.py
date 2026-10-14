from __future__ import annotations

import torch

from hierarchical_world_model.src.influence_graph import build_dynamic_interaction_relations


def _map(centers: tuple[float, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    maps = torch.zeros(1, len(centers), 4, 6)
    maps[..., 0] = torch.linspace(-50.0, 50.0, 4)
    maps[0, :, :, 1] = torch.tensor(centers)[:, None]
    maps[..., 2] = 1.0
    maps[..., 4] = 3.6
    return maps, torch.ones(1, len(centers), 4, dtype=torch.bool)


def test_dynamic_relations_update_after_lane_change_not_slot_role() -> None:
    states = torch.zeros(1, 7, 6)
    valid = torch.zeros(1, 7, dtype=torch.bool)
    valid[:, :4] = True
    states[0, 0, :3] = torch.tensor((20.0, 0.0, 20.0))
    states[0, 1, :3] = torch.tensor((0.0, 0.0, 22.0))
    states[0, 2, :3] = torch.tensor((15.0, 3.6, 18.0))
    states[0, 3, :3] = torch.tensor((-5.0, 3.6, 23.0))
    maps, map_valid = _map((0.0, 3.6))
    before = build_dynamic_interaction_relations(states, valid, maps, map_valid)
    assert before.leader_index[0, 1].item() == 0
    moved = states.clone()
    moved[0, 1, 1] = 3.6
    after = build_dynamic_interaction_relations(moved, valid, maps, map_valid)
    assert after.leader_index[0, 1].item() == 2


def test_dynamic_relations_are_lateral_mirror_symmetric() -> None:
    states = torch.zeros(1, 7, 6)
    valid = torch.zeros(1, 7, dtype=torch.bool)
    valid[:, :4] = True
    states[0, 0, :4] = torch.tensor((15.0, 0.0, 20.0, 0.7))
    states[0, 1, :4] = torch.tensor((0.0, 3.6, 22.0, -0.4))
    states[0, 2, :4] = torch.tensor((10.0, 3.6, 18.0, 0.0))
    states[0, 3, :4] = torch.tensor((-8.0, 0.0, 21.0, 0.0))
    maps, map_valid = _map((-3.6, 0.0, 3.6))
    original = build_dynamic_interaction_relations(states, valid, maps, map_valid)
    mirrored_states = states.clone()
    mirrored_states[..., 1] *= -1
    mirrored_states[..., 3] *= -1
    mirrored_maps = maps.clone()
    mirrored_maps[..., 1] *= -1
    mirrored = build_dynamic_interaction_relations(
        mirrored_states, valid, mirrored_maps, map_valid
    )
    torch.testing.assert_close(original.leader_index, mirrored.leader_index)
    torch.testing.assert_close(original.leader_gap_m, mirrored.leader_gap_m)
    torch.testing.assert_close(
        original.adjacent_front_index[..., 0], mirrored.adjacent_front_index[..., 1]
    )
    torch.testing.assert_close(
        original.adjacent_front_index[..., 1], mirrored.adjacent_front_index[..., 0]
    )


def test_dynamic_relations_use_mapped_lane_width() -> None:
    states = torch.zeros(1, 7, 6)
    valid = torch.zeros(1, 7, dtype=torch.bool)
    valid[0, :2] = True
    states[0, 0, :3] = torch.tensor((0.0, 0.0, 20.0))
    states[0, 1, :3] = torch.tensor((10.0, 2.0, 18.0))
    maps = torch.zeros(1, 1, 4, 6)
    maps[..., 0] = torch.linspace(-50.0, 50.0, 4)
    maps[..., 2] = 1.0
    map_valid = torch.ones(1, 1, 4, dtype=torch.bool)
    maps[..., 4] = 4.4
    wide = build_dynamic_interaction_relations(states, valid, maps, map_valid)
    maps[..., 4] = 2.0
    narrow = build_dynamic_interaction_relations(states, valid, maps, map_valid)
    assert wide.leader_index[0, 0].item() == 1
    assert narrow.leader_index[0, 0].item() == -1


def test_dynamic_relations_respect_divided_highway_adjacency() -> None:
    centers = (-8.43, -4.325, -0.22, 9.675, 13.735, 17.8)
    widths = (4.32, 3.89, 4.32, 4.23, 3.89, 4.24)
    maps = torch.zeros(1, len(centers), 4, 6)
    maps[..., 0] = torch.linspace(-50.0, 50.0, 4)
    maps[0, :, :, 1] = torch.tensor(centers)[:, None]
    maps[0, :, :, 4] = torch.tensor(widths)[:, None]
    map_valid = torch.ones(1, len(centers), 4, dtype=torch.bool)

    states = torch.zeros(1, 7, 6)
    valid = torch.zeros(1, 7, dtype=torch.bool)
    valid[0, :2] = True
    states[0, 0, :3] = torch.tensor((0.0, -0.22, 25.0))
    states[0, 1, :3] = torch.tensor((10.0, 9.675, 25.0))
    relations = build_dynamic_interaction_relations(states, valid, maps, map_valid)

    assert relations.adjacent_lane_index[0, 0].tolist() == [1, -1]
    assert relations.adjacent_lane_index[0, 1].tolist() == [-1, 4]


def test_influence_pair_gap_uses_pair_specific_lengths() -> None:
    from hierarchical_world_model.src.influence_graph import CausalInfluenceGraph

    parent = torch.zeros(1, 1, 6)
    child = torch.zeros(1, 1, 6)
    child[..., 0] = -10.0
    _, _, gap, _, _ = CausalInfluenceGraph._pair_metrics(
        parent, child, torch.tensor(((6.0,),)), torch.tensor(((2.0,),))
    )
    torch.testing.assert_close(gap, torch.tensor(((6.0,),)))

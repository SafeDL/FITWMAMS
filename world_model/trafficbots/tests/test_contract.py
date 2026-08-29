from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

from world_model.src.core.highd_metrics import semantic_cutin_agents
from world_model.trafficbots.data import adapt_highd_batch, make_loader, states_to_motion


def _fixture():
    states = np.zeros((150, 7, 6), np.float32)
    states[..., 2] = 20.0
    valid = np.zeros((150, 7), bool); valid[:, :2] = True
    polylines = np.zeros((8, 8, 6), np.float32)
    for lane in range(2):
        polylines[lane, :, 0] = np.arange(8) * 10.0
        polylines[lane, :, 1] = lane * 4.0
    map_valid = np.zeros((8, 8), bool); map_valid[:2] = True
    return states, valid, polylines, map_valid


def test_highd_adapter_fixed_contract_and_dummy_traffic_lights():
    states, valid, polylines, map_valid = _fixture()
    batch = adapt_highd_batch(states, valid, polylines, map_valid)
    assert batch["agent/valid"].shape == (1, 8, 150)
    assert batch["map/valid"].shape == (1, 16, 8)
    assert batch["map/type"].shape == (1, 16, 11)
    assert batch["map/type"][0, :2, 0].all()
    assert not batch["map/type"][0, 2:].any()
    assert not batch["agent/size"][0, 2:].any()
    assert batch["destination_candidate_valid"][0, :2, :2].all()
    assert not batch["tl_stop/valid"].any()


def test_s0_yaw_rate_never_uses_s1():
    states = np.zeros((1, 1, 2, 6), np.float32)
    states[..., 2] = 10.0
    states[0, 0, 1, 2:4] = (0.0, 10.0)
    pose, motion = states_to_motion(states, np.ones((1, 1, 2), bool))
    assert pose[0, 0, 0, 2] == 0.0
    assert motion[0, 0, 0, 2] == 0.0
    assert motion[0, 0, 1, 2] > 0.0


def test_low_speed_s0_uses_causal_zero_heading():
    states = np.zeros((1, 1, 2, 6), np.float32)
    states[0, 0, 0, 2:4] = (-.04, 0.0)
    states[0, 0, 1, 2:4] = (10.0, 0.0)
    pose, motion = states_to_motion(states, np.ones((1, 1, 2), bool))
    assert pose[0, 0, 0, 2] == 0.0
    assert motion[0, 0, 0, 2] == 0.0


def test_reverse_lane_is_not_destination_candidate():
    states, valid, polylines, map_valid = _fixture()
    polylines[1, :, 0] = np.arange(8, 0, -1) * 10.0
    batch = adapt_highd_batch(states, valid, polylines, map_valid)
    assert batch["destination_candidate_valid"][0, 0, 0]
    assert not batch["destination_candidate_valid"][0, 0, 1]


def test_invalid_canonical_slot_is_not_given_a_vehicle_footprint():
    states, valid, polylines, map_valid = _fixture()
    valid[:, 1] = False
    batch = adapt_highd_batch(states, valid, polylines, map_valid)
    assert not batch["agent/type"][0, 1].any()
    assert not batch["agent/size"][0, 1].any()


def test_future_spawn_does_not_reveal_agent_identity_at_s0():
    states, valid, polylines, map_valid = _fixture()
    valid[:, 1] = False
    valid[1:, 1] = True
    batch = adapt_highd_batch(states, valid, polylines, map_valid)
    assert not batch["agent/type"][0, 1].any()
    assert not batch["agent/role"][0, 1].any()
    assert not batch["agent/size"][0, 1].any()
    assert not batch["destination_candidate_valid"][0, 1].any()


def test_shared_semantic_cutin_and_seeded_loader_contract():
    states = np.zeros((1, 150, 7, 6), np.float32)
    valid = np.ones((1, 150, 7), bool)
    states[..., 2] = 20.0
    states[:, :, 1, 0] = 12.0
    states[:, :10, 1, 1] = 3.0
    assert semantic_cutin_agents(states, valid)[0, 0]
    dataset = TensorDataset(torch.arange(12))
    first = [int(value) for (value,) in make_loader(dataset, batch_size=1, shuffle=True, seed=19)]
    second = [int(value) for (value,) in make_loader(dataset, batch_size=1, shuffle=True, seed=19)]
    assert first == second

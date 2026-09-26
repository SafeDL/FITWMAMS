"""highD vehicles above 50 m/s must not be truncated by either rollout plant."""

import numpy as np
import torch

from hierarchical_world_model.src.highway import HighwayEnvTraffic
from hierarchical_world_model.src.model import (
    DiffusionGuidedHiQR,
    HIGHD_INTEGRATION_SPEED_MAX_MPS,
)


def test_hiqr_plant_preserves_highd_speed_above_50_mps() -> None:
    model = DiffusionGuidedHiQR()
    assert model.dynamics.cfg.speed_max_mps == HIGHD_INTEGRATION_SPEED_MAX_MPS
    state = torch.zeros((1, 7, 6), dtype=torch.float32)
    state[0, 1, 2] = 56.0
    control = torch.zeros((1, 7, 2), dtype=torch.float32)
    control[0, 1, 0] = 1.0
    valid = torch.zeros((1, 7), dtype=torch.bool)
    valid[0, 1] = True
    advanced = model.dynamics.step(state, control, valid, 0.04)
    assert torch.isclose(advanced[0, 1, 2], torch.tensor(56.04), atol=1.0e-4)


def test_highway_plant_matches_high_speed_hiqr_step() -> None:
    states = np.zeros((7, 6), np.float32)
    states[0, :3] = [0.0, 0.0, 56.0]
    states[1, :3] = [50.0, 3.6, 56.0]
    valid = np.zeros(7, bool)
    valid[:2] = True
    world = HighwayEnvTraffic()
    world.reset(states, valid)
    background = np.zeros((6, 2), np.float32)
    background[0, 0] = 1.0
    advanced = world.step(background, ego_action=np.zeros(2, np.float32)).states
    assert abs(float(advanced[1, 2]) - 56.04) < 1.0e-4
    assert abs(float(advanced[0, 2]) - 56.0) < 1.0e-4


def test_highway_idm_ego_is_not_forced_below_50_mps() -> None:
    states = np.zeros((7, 6), np.float32)
    states[0, :3] = [0.0, 0.0, 56.0]
    valid = np.zeros(7, bool)
    valid[0] = True
    world = HighwayEnvTraffic()
    world.reset(states, valid, idm_config={"target_speed": 56.0})
    advanced = world.step(np.zeros((6, 2), np.float32)).states
    assert advanced[0, 2] > 50.0

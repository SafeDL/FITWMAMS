from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch

from ads.common.instances import GeneratedSource
from tools.evt import GPDTailModel


class SmallSampler:
    generated_road: ClassVar = {"lane_centers_m": [-4.0, 0.0, 4.0], "lane_width_m": 4.0}

    def compose_exogenous(self, omega):
        n = omega.batch_size
        states = np.zeros((n, 7, 6), np.float32)
        states[:, 0, 2] = 20
        states[:, 1, 0] = 35 + omega.c0_base_latent[:, 0] * 2
        states[:, 1, 2] = 19
        valid = np.zeros((n, 7), bool)
        valid[:, :2] = True
        plan = (
            states[:, None, 1:, :2]
            + np.arange(149)[None, :, None, None] * 0.04 * states[:, None, 1:, 2:4]
        )
        return SimpleNamespace(
            initial_states=states,
            initial_valid=valid,
            soft_plan=plan.astype(np.float32),
            vehicle_dimensions=np.tile(np.asarray((4.8, 1.8), np.float32), (n, 7, 1)),
        )


class SmallWorld:
    def reset(self, states, valid, reference, maps, map_valid, **kwargs):
        from hierarchical_world_model.src.scene_geometry import SceneGeometry

        self.states = states.clone().float()
        self.valid = valid.numpy()[0]
        self.reference_index = 0
        self.geometry = SceneGeometry.from_map_polylines(
            maps.numpy()[0], map_valid.numpy()[0]
        )
        return self.observe()

    def observe(self):
        return {"agent_states": self.states, "reference_index": self.reference_index}

    def advance_response(self, action):
        action = action.numpy()[0]
        ego = self.states[0, 0]
        speed = max(0, float(torch.linalg.norm(ego[2:4])) + action[0] * 0.04)
        heading = float(torch.atan2(ego[3], ego[2])) + action[1] * 0.04
        old = ego[2:4].clone()
        ego[2:4] = torch.tensor((speed * np.cos(heading), speed * np.sin(heading)))
        ego[4:6] = (ego[2:4] - old) / 0.04
        self.states[0, :, :2] += self.states[0, :, 2:4] * 0.04
        self.reference_index += 1
        offroad = self.geometry.offroad(
            self.states[0, :, :2].numpy(), vehicle_width_m=1.8
        )
        return {
            "collision": torch.zeros(1, dtype=torch.bool),
            "crashed": torch.zeros((1, 7), dtype=torch.bool),
            "offroad": torch.as_tensor(offroad)[None],
            "takeover_owned": torch.zeros((1, 6), dtype=torch.bool),
        }


@pytest.fixture
def source():
    return GeneratedSource(
        SmallSampler(), {"frozen": "asset"}, master_seed=123, purpose="search"
    )


@pytest.fixture
def instance(source):
    return source.sample(1)[0]


@pytest.fixture
def evt():
    return GPDTailModel(0.5, 0.0, 1.0, 0.1, np.asarray((0, 0.1, 0.5)), {}, {}, [])


@pytest.fixture
def world_factory():
    return lambda event: SmallWorld()

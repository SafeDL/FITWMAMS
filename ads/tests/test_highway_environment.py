from __future__ import annotations

import numpy as np
import torch
from gymnasium.utils.env_checker import check_env

from ads.common.control import FASTER, LANE_LEFT
from ads.common.environment import ADSHighwayEnv
from ads.common.world import WorldEvent
from hierarchical_world_model.src.randomness import WorldExogenousState


class FakeWorld:
    def __init__(self) -> None:
        self.reference_index = 0

    def reset(self, states, valid, reference, maps, map_valid, **kwargs):
        del valid, reference, maps, map_valid, kwargs
        self.states = torch.as_tensor(states, dtype=torch.float32).clone()
        self.reference_index = 0
        return self.observe()

    def observe(self):
        return {"agent_states": self.states, "reference_index": self.reference_index}

    def advance_response(self, ego_action):
        action = torch.as_tensor(ego_action, dtype=torch.float32)
        dt = 0.04
        self.states[:, 0, 0] += self.states[:, 0, 2] * dt
        self.states[:, 0, 1] += self.states[:, 0, 3] * dt
        self.states[:, 0, 2] += action[:, 0] * dt
        self.states[:, 0, 3] += action[:, 1] * dt
        self.states[:, 0, 4] = action[:, 0]
        self.reference_index += 1
        return {
            "collision": torch.zeros(1, dtype=torch.bool),
            "offroad": torch.zeros(1, 7, dtype=torch.bool),
        }


class NPCollisionWorld(FakeWorld):
    def advance_response(self, ego_action):
        result = super().advance_response(ego_action)
        result["collision"] = torch.ones(1, dtype=torch.bool)
        result["crashed"] = torch.tensor(
            ((False, True, True, False, False, False, False),),
            dtype=torch.bool,
        )
        return result


def _event(frames: int = 7) -> WorldEvent:
    states = np.zeros((7, 6), np.float32)
    valid = np.zeros(7, bool)
    valid[:2] = True
    states[0, :3] = (0.0, 0.0, 20.0)
    states[1, :3] = (20.0, 0.0, 18.0)
    maps = np.zeros((2, 4, 6), np.float32)
    maps[..., 0] = np.linspace(-100.0, 100.0, 4)
    maps[0, :, 1] = 0.0
    maps[1, :, 1] = 4.0
    maps[..., 2] = 1.0
    maps[..., 4] = 4.0
    return WorldEvent(
        "e0",
        "e0",
        states,
        valid,
        np.zeros((frames, 6, 2), np.float32),
        maps,
        np.ones((2, 4), bool),
        WorldExogenousState.sample(1, seed=3, response_steps=frames),
    )


def test_ads_env_runs_five_substeps_then_exact_remainder() -> None:
    env = ADSHighwayEnv([_event()], lambda event: FakeWorld(), random_events=False)
    observation, info = env.reset()
    assert observation.shape == (49,)
    assert info["event_id"] == "e0"
    _, _, terminated, truncated, info = env.step(FASTER)
    assert not terminated and not truncated
    assert info["executed_physics_steps"] == 5
    _, _, terminated, truncated, info = env.step(LANE_LEFT)
    assert not terminated and truncated
    assert info["executed_physics_steps"] == 2
    assert info["structurally_valid_action"]


def test_ads_env_satisfies_gymnasium_contract() -> None:
    env = ADSHighwayEnv([_event(10)], lambda event: FakeWorld(), random_events=False)
    check_env(env, skip_render_check=True)


def test_npc_only_collision_is_diagnostic_not_ads_failure() -> None:
    env = ADSHighwayEnv(
        [_event(5)], lambda event: NPCollisionWorld(), random_events=False
    )
    env.reset()
    _, _, terminated, truncated, info = env.step(FASTER)
    assert not terminated and truncated
    assert not info["collision"]
    assert info["any_vehicle_collision"]
    assert info["npc_collision"]

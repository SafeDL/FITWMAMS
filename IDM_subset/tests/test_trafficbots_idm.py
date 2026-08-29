from __future__ import annotations

import numpy as np
import torch

from IDM_subset.src.trafficbots_randomness import TrafficBotsExogenousState
from IDM_subset.src.trafficbots_world import (
    _categorical_from_uniform,
    _initial_states_from_c0,
)
from IDM_subset.src.world_evaluator import WorldEvaluation
from IDM_subset.src.world_subset_simulation import run_world_subset_simulation
from normalizing_flow.src.features import feature_index


def test_trafficbots_exogenous_state_exact_replay_and_prior_mutation(tmp_path) -> None:
    state = TrafficBotsExogenousState.sample(4, seed=17)
    repeated = TrafficBotsExogenousState.sample(4, seed=17)
    for name, value in state.as_dict().items():
        assert np.array_equal(value, repeated.as_dict()[name])

    path = tmp_path / "trafficbots_world.npz"
    state.save(path)
    loaded = TrafficBotsExogenousState.load(path)
    for name, value in state.as_dict().items():
        assert np.array_equal(value, loaded.as_dict()[name])

    personality = state.pcn_mutate("personality_latent", beta=0.2, seed=19)
    assert not np.array_equal(personality.personality_latent, state.personality_latent)
    assert np.array_equal(personality.destination_uniform, state.destination_uniform)
    destination = state.pcn_mutate("destination_uniform", beta=0.2, seed=20)
    assert np.all((destination.destination_uniform >= 0.0) & (destination.destination_uniform < 1.0))
    assert not np.array_equal(destination.destination_uniform, state.destination_uniform)


def test_external_flow_c0_conversion_contains_no_future_k() -> None:
    c0 = np.zeros((1, 40), np.float32)
    slots = np.asarray([[True, False, False, False, False, False]])
    c0[0, feature_index(None, "ego_vx_mps")] = 25.0
    c0[0, feature_index("same_front", "rel_x_m")] = 18.0
    c0[0, feature_index("same_front", "rel_y_left_m")] = 0.2
    c0[0, feature_index("same_front", "rel_vx_mps")] = -2.0
    c0[0, feature_index("same_front", "other_ax_mps2")] = -0.5
    states = _initial_states_from_c0(c0, slots)
    assert states.shape == (1, 7, 6)
    assert states[0, 0, 2] == 25.0
    assert np.allclose(states[0, 1], [18.0, 0.2, 23.0, 0.0, -0.5, 0.0])
    assert not states[0, 2:].any()


def test_destination_inverse_cdf_uses_explicit_uniform_and_validity() -> None:
    probability = torch.tensor(
        [[
            [0.25, 0.75, 0.0],
            [0.10, 0.20, 0.70],
        ]]
    )
    valid = torch.tensor([[True, False]])
    selected = _categorical_from_uniform(
        probability, np.asarray([[0.249, 0.95]]), valid
    )
    assert selected.tolist() == [[0, 0]]
    selected = _categorical_from_uniform(
        probability, np.asarray([[0.251, 0.01]]), valid
    )
    assert selected.tolist() == [[1, 0]]


def test_generic_subset_engine_accepts_trafficbots_probability_space() -> None:
    def sample_worlds(count: int, seed: int) -> TrafficBotsExogenousState:
        return TrafficBotsExogenousState.sample(count, seed=seed)

    def evaluate(worlds: TrafficBotsExogenousState) -> WorldEvaluation:
        score = worlds.personality_latent[:, 0, 0].astype(np.float64)
        return WorldEvaluation(
            evt_score=score,
            event_risk=np.maximum(score, 0.0),
            collision=np.zeros(worlds.batch_size, dtype=bool),
            min_gap_m=np.full(worlds.batch_size, 10.0),
            numerical_valid=np.ones(worlds.batch_size, dtype=bool),
        )

    result = run_world_subset_simulation(
        evaluate,
        num_samples=20,
        p0=0.2,
        max_levels=2,
        failure_threshold=4.0,
        response_steps=1,
        scene_dim=0,
        agent_dim=16,
        mutation_blocks=("personality_latent", "destination_uniform"),
        pcn_beta=0.2,
        mcmc_steps=1,
        seed=23,
        sample_worlds=sample_worlds,
    )
    assert len(result.levels) == 2
    assert isinstance(result.levels[-1].worlds, TrafficBotsExogenousState)
    assert np.isfinite(result.probability)

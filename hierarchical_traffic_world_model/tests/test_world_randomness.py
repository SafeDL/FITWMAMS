"""Contracts for explicit, prior-preserving world base randomness."""

import numpy as np

from hierarchical_traffic_world_model.src.world_randomness import WorldExogenousState


def test_explicit_world_randomness_is_seed_replayable_and_block_mutable(tmp_path):
    first = WorldExogenousState.sample(2, seed=71, response_steps=9)
    replay = WorldExogenousState.sample(2, seed=71, response_steps=9)
    assert all(np.array_equal(first.as_dict()[name], replay.as_dict()[name]) for name in first.as_dict())
    path = tmp_path / "world.npz"
    first.save(path)
    restored = WorldExogenousState.load(path)
    assert all(np.array_equal(first.as_dict()[name], restored.as_dict()[name]) for name in first.as_dict())
    proposal = first.pcn_mutate("diffusion_noise", beta=0.3, seed=17)
    assert not np.array_equal(first.diffusion_noise, proposal.diffusion_noise)
    assert np.array_equal(first.scenario_uniform, proposal.scenario_uniform)


def test_scenario_block_mutation_remains_a_valid_uniform_block():
    state = WorldExogenousState.sample(3, seed=5, response_steps=1)
    proposal = state.pcn_mutate("scenario", beta=1.0, seed=6)
    assert np.all((proposal.scenario_uniform >= 0.0) & (proposal.scenario_uniform < 1.0))
    assert not np.array_equal(state.scenario_uniform, proposal.scenario_uniform)

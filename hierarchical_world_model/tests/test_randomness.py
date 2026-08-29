"""Contracts for explicit, prior-preserving world base randomness."""

import numpy as np

from hierarchical_world_model.src.randomness import (
    WORLD_RANDOM_BLOCKS,
    WorldExogenousState,
)


def test_explicit_randomness_is_seed_replayable_and_block_mutable(tmp_path):
    first = WorldExogenousState.sample(2, seed=71, response_steps=9)
    replay = WorldExogenousState.sample(2, seed=71, response_steps=9)
    assert tuple(first.as_dict()) == WORLD_RANDOM_BLOCKS
    assert all(np.array_equal(first.as_dict()[name], replay.as_dict()[name]) for name in first.as_dict())
    path = tmp_path / "world.npz"
    first.save(path)
    restored = WorldExogenousState.load(path)
    assert all(np.array_equal(first.as_dict()[name], restored.as_dict()[name]) for name in first.as_dict())
    proposal = first.pcn_mutate("diffusion_noise", beta=0.3, seed=17)
    assert not np.array_equal(first.diffusion_noise, proposal.diffusion_noise)
    assert np.array_equal(first.scenario_uniform, proposal.scenario_uniform)


def test_scene_innovations_are_compressed_without_becoming_the_horizon_source():
    state = WorldExogenousState.sample(2, seed=18, response_steps=149)
    assert state.response_steps == 149
    assert state.scene_innovations.shape == (2, 6, 16)
    assert state.agent_response_innovations.shape == (2, 149, 7, 16)
    state.validate(response_steps=149)


def test_world_blocks_have_stable_independent_streams():
    short = WorldExogenousState.sample(1, seed=22, response_steps=1)
    long = WorldExogenousState.sample(1, seed=22, response_steps=149)
    assert np.array_equal(short.scenario_uniform, long.scenario_uniform)
    assert np.array_equal(short.c0_base_latent, long.c0_base_latent)
    assert np.array_equal(short.k_base_latent, long.k_base_latent)
    assert np.array_equal(short.diffusion_noise, long.diffusion_noise)


def test_scenario_block_mutation_remains_a_valid_uniform_block():
    state = WorldExogenousState.sample(3, seed=5, response_steps=1)
    proposal = state.pcn_mutate("scenario", beta=1.0, seed=6)
    assert np.all((proposal.scenario_uniform >= 0.0) & (proposal.scenario_uniform < 1.0))
    assert not np.array_equal(state.scenario_uniform, proposal.scenario_uniform)


def test_scene_innovations_respect_custom_scene_refresh_responses():
    base_steps = 10
    state = WorldExogenousState.sample(
        2, seed=44, response_steps=base_steps, scene_refresh_responses=2
    )
    assert state.scene_innovations.shape == (2, 5, 16)

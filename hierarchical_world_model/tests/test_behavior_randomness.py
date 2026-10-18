from __future__ import annotations

import numpy as np
import pytest

from hierarchical_world_model.src.empirical_context import EmpiricalKWorldState
from hierarchical_world_model.src.randomness import (
    WORLD_RANDOM_BLOCKS,
    WORLD_RANDOM_STREAMS,
    WorldExogenousState,
)
from hierarchical_world_model.src.stochastic_drivers.idm_population import (
    MODEL_PATH,
    sample_theta,
)
from hierarchical_world_model.src.stochastic_drivers.online import (
    mobil_parameters_from_maneuver_latent,
)


def test_empirical_k_random_streams_replay_and_reject_invalid_lengths() -> None:
    first = EmpiricalKWorldState.sample(2, seed=19, response_steps=8)
    replay = EmpiricalKWorldState.sample(2, seed=19, response_steps=8)
    for name, value in first.as_dict().items():
        np.testing.assert_array_equal(value, replay.as_dict()[name])

    with pytest.raises(ValueError, match="must be positive"):
        EmpiricalKWorldState.sample(1, seed=19, response_steps=0)


def test_world_random_stream_ids_are_explicit_and_unique() -> None:
    assert set(WORLD_RANDOM_STREAMS) == set(WORLD_RANDOM_BLOCKS)
    assert len(set(WORLD_RANDOM_STREAMS.values())) == len(WORLD_RANDOM_BLOCKS)


def test_randomness_concatenation_preserves_blocks() -> None:
    worlds = [
        WorldExogenousState.sample(1, seed=seed, response_steps=8) for seed in (3, 7)
    ]
    combined = WorldExogenousState.concatenate(worlds)
    combined.validate()
    for name, value in worlds[0].as_dict().items():
        np.testing.assert_array_equal(
            getattr(combined, name),
            np.concatenate([getattr(world, name) for world in worlds]),
        )

    contexts = [
        EmpiricalKWorldState.sample(1, seed=seed, response_steps=8) for seed in (3, 7)
    ]
    combined_context = EmpiricalKWorldState.concatenate(contexts)
    combined_context.validate()
    for name, value in contexts[0].as_dict().items():
        np.testing.assert_array_equal(
            getattr(combined_context, name),
            np.concatenate([getattr(context, name) for context in contexts]),
        )


def test_world_exogenous_archive_roundtrip_preserves_blocks(tmp_path) -> None:
    world = WorldExogenousState.sample(2, seed=15, response_steps=12)
    path = tmp_path / "world_randomness.npz"
    world.save(path)
    restored = WorldExogenousState.load(path)
    for name, value in world.as_dict().items():
        np.testing.assert_array_equal(getattr(restored, name), value)


def test_actor_local_driver_mutation_changes_only_selected_driver() -> None:
    world = WorldExogenousState.sample(2, seed=4, response_steps=8)
    mask = np.zeros((2, 6), bool)
    mask[0, 3] = True
    proposal = world.mutate("driver", beta=0.25, seed=9, actor_mask=mask)
    changed = np.any(proposal.driver_latent != world.driver_latent, axis=-1)
    assert np.array_equal(changed, mask)
    assert np.array_equal(proposal.maneuver_latent, world.maneuver_latent)
    assert np.array_equal(
        proposal.policy_response_innovations, world.policy_response_innovations
    )


def test_future_response_mutation_preserves_prefix_and_other_actors() -> None:
    world = WorldExogenousState.sample(1, seed=5, response_steps=12)
    mask = np.asarray((False, True, False, False, False, False))
    proposal = world.mutate(
        "policy", beta=0.4, seed=10, actor_mask=mask, time_range=(7, 12)
    )
    assert np.array_equal(
        proposal.policy_response_innovations[:, :7],
        world.policy_response_innovations[:, :7],
    )
    changed = proposal.policy_response_innovations != world.policy_response_innovations
    assert not changed[:, :, 0].any()  # ego innovation is never selected by an NPC mask
    assert changed[:, 7:, 1].any()
    assert not changed[:, 7:, 2].any()


def test_driver_latent_maps_continuously_to_fixed_episode_parameters() -> None:
    world = WorldExogenousState.sample(1, seed=6, response_steps=2)
    first = sample_theta(dict(np.load(MODEL_PATH)), world.driver_latent)
    values = world.as_dict()
    values["driver_latent"][0, 0, 0] += 1.0e-3
    second_world = world.replace_arrays(values)
    second = sample_theta(dict(np.load(MODEL_PATH)), second_world.driver_latent)
    assert np.array_equal(first[0, 1:], second[0, 1:])
    assert np.linalg.norm(first[0, 0] - second[0, 0]) < 0.02
    assert not np.array_equal(first[0, 0], second[0, 0])


def test_actor_local_maneuver_mutation_changes_only_selected_traits() -> None:
    world = WorldExogenousState.sample(1, seed=12, response_steps=2)
    mask = np.zeros((1, 6), bool)
    mask[0, 2] = True
    proposal = world.mutate("maneuver", beta=0.2, seed=13, actor_mask=mask)
    before = mobil_parameters_from_maneuver_latent(world.maneuver_latent)
    after = mobil_parameters_from_maneuver_latent(proposal.maneuver_latent)
    for name in before:
        changed = (before[name] != after[name]).numpy()
        assert np.array_equal(changed, mask)

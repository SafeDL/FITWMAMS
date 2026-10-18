import numpy as np

from hierarchical_world_model.src.randomness import WORLD_RANDOM_BLOCKS
from hierarchical_world_model.src.shared_evaluation import (
    recording_interval,
    scene_randomness,
)


def test_recording_bootstrap_keeps_constant_scene_error():
    rows = [{"recording_id": index // 2, "error": 0.25} for index in range(4)]
    np.testing.assert_allclose(recording_interval(rows, "error"), [0.25, 0.25])


def test_world_randomness_is_stable_across_batches_and_order():
    together = scene_randomness(["nat_04_51_100", "nat_49_16_200"], frames=3)
    separate = scene_randomness(["nat_49_16_200"], frames=3)
    for name in WORLD_RANDOM_BLOCKS:
        np.testing.assert_array_equal(
            getattr(together, name)[1:], getattr(separate, name)
        )
    assert not np.array_equal(together.driver_latent[0], together.driver_latent[1])


def test_persistent_driver_draws_do_not_depend_on_response_horizon():
    short = scene_randomness(["nat_04_51_100", "nat_49_16_200"], frames=1)
    full = scene_randomness(["nat_04_51_100", "nat_49_16_200"], frames=149)
    for name in ("driver_latent", "maneuver_latent", "diffusion_noise"):
        np.testing.assert_array_equal(getattr(short, name), getattr(full, name))

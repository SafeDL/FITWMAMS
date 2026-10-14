import numpy as np
import torch

from hierarchical_world_model.src.npc_experiments.road import neighbors
from hierarchical_world_model.src.stochastic_drivers.idm_population import (
    fit_parameters,
    idm_acceleration,
    sample_theta,
    torch_idm,
)


def test_torch_and_numpy_idm_agree_and_do_not_brake_for_a_receding_leader():
    theta = np.asarray([35.0, 2.0, 1.0, 1.0, 2.0])
    gap = np.asarray([20.0, 60.0])
    speed = np.asarray([20.0, 25.0])
    leaders = np.asarray([40.0, 20.0])
    expected = idm_acceleration(gap, speed, leaders, theta)
    actual = torch_idm(*(torch.from_numpy(x) for x in (gap, speed, leaders, theta)))
    np.testing.assert_allclose(actual.numpy(), expected)
    assert expected[0] > 0.0


def test_parameter_fit_reconstructs_an_identifiable_synthetic_driver():
    rng = np.random.default_rng(1)
    data = {
        "gap": rng.uniform(15.0, 150.0, 200),
        "speed": rng.uniform(8.0, 35.0, 200),
        "leader_speed": rng.uniform(8.0, 35.0, 200),
    }
    theta = np.asarray([36.0, 2.0, 1.2, 1.1, 2.2])
    data["acceleration"] = idm_acceleration(**data, theta=theta)
    fitted = fit_parameters(data)
    np.testing.assert_allclose(np.exp(fitted), theta, rtol=0.01)


def test_float32_cache_observations_do_not_zero_the_optimizer_jacobian():
    rng = np.random.default_rng(2)
    data = {
        "gap": rng.uniform(20.0, 150.0, 400).astype(np.float32),
        "speed": rng.uniform(10.0, 35.0, 400).astype(np.float32),
        "leader_speed": rng.uniform(10.0, 35.0, 400).astype(np.float32),
    }
    theta = np.asarray([38.0, 2.5, 1.3, 1.3, 2.4])
    data["acceleration"] = idm_acceleration(**data, theta=theta).astype(np.float32)
    fitted = fit_parameters(data)
    np.testing.assert_allclose(np.exp(fitted), theta, rtol=0.01)


def test_parameter_draws_replay_the_same_episode_latents():
    population = {"log_mean": np.log([33, 2, 1.2, 1, 2]), "log_cov": np.eye(5) * 0.02}
    latent = np.random.default_rng(3).standard_normal((2, 6, 5))
    np.testing.assert_array_equal(
        sample_theta(population, latent), sample_theta(population, latent)
    )
    assert not np.array_equal(
        sample_theta(population, latent), sample_theta(population, -latent)
    )


def test_leader_assignment_uses_lane_and_actual_vehicle_lengths():
    states = np.zeros((1, 7, 6), np.float32)
    states[0, :, 0] = [0, 10, 30, 20, 5, 40, 50]
    states[0, 3, 1] = 3.6
    valid = np.ones((1, 7), bool)
    dimensions = np.tile([4.8, 1.8], (1, 7, 1)).astype(np.float32)
    dimensions[0, 2, 0] = 10.0
    _, leader, gap, _, _ = neighbors(
        states, valid, dimensions[0], np.array([0.0, 3.6])
    )
    assert leader[0, 1] == 2
    np.testing.assert_allclose(gap[0, 1], 12.6)
    assert leader[0, 3] == -1

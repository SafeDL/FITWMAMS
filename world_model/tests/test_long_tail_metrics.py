import numpy as np

from world_model.src.core.long_tail_metrics import (
    collision_metrics,
    empirical_distance,
    feature_distribution_distance,
    kinematic_reconstruction_metrics,
    speed_kl_divergence,
    traffic_fields,
    trajectory_metrics,
)


def _states(batch=2, frames=4, agents=2):
    value = np.zeros((batch, frames, agents, 6), np.float32)
    value[..., 2] = 10.0
    value[..., 0] = np.arange(frames, dtype=np.float32)[None, :, None] * 0.4
    return value


def test_identical_trajectory_and_distribution_metrics_are_zero():
    target = _states()
    valid = np.ones(target.shape[:-1], bool)
    samples = np.stack([target, target])
    metric = trajectory_metrics(samples, target, valid)
    assert metric["ADE_m"] == 0.0
    assert metric["minFDE_at_32_m"] == 0.0
    assert speed_kl_divergence(np.array([5.0, 10.0]), np.array([5.0, 10.0])) < 1.0e-10
    assert empirical_distance(np.array([1.0, 2.0]), np.array([1.0, 2.0]))["wasserstein_1"] == 0.0


def test_low_speed_curvature_is_masked():
    target = _states(batch=1, frames=3, agents=1)
    target[..., 2:4] = 0.0
    predicted = target.copy()
    predicted[..., 5] = 10.0
    result = kinematic_reconstruction_metrics(predicted, target, np.ones(target.shape[:-1], bool))
    assert np.isnan(result["curvature_mae_m_inv"])


def test_collision_uses_all_vehicle_pairs():
    background = _states(batch=1, frames=2, agents=2)
    background[..., 0, 0] = 0.0
    background[..., 1, 0] = 2.0
    ego = np.zeros((1, 2, 6), np.float32)
    ego[..., 2] = 10.0
    fields = traffic_fields(background, ego, np.ones(background.shape[:-1], bool))
    assert collision_metrics(fields)["collision_episode_rate"] == 1.0


def test_identical_feature_distributions_have_near_zero_distance():
    states = _states(batch=3, frames=8, agents=2)
    ego = np.zeros((3, 8, 6), np.float32)
    ego[..., 2] = 10.0
    valid = np.ones(states.shape[:-1], bool)
    result = feature_distribution_distance(states, ego, valid, states, ego, valid, seed=7)
    assert result["traffic_feature_frechet_distance"] < 1.0e-7
    assert abs(result["mmd_rbf"]) < 1.0e-7

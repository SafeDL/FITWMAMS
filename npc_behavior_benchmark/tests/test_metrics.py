import numpy as np

from npc_behavior_benchmark.evaluation.metrics import (
    ensemble_channel_metrics,
    fair_crps,
    fair_energy_score,
    trajectory_errors,
)
from npc_behavior_benchmark.scripts.evaluate_events import _behavior_brier


def test_perfect_ensemble_has_zero_ade_and_fes():
    target = np.arange(24, dtype=np.float64).reshape(3, 2, 4)
    samples = np.repeat(target[None], 4, axis=0)
    valid = np.ones((3, 2), bool)
    assert fair_energy_score(samples, target, valid, np.ones(4)) == 0.0
    metrics = trajectory_errors(samples, target, valid)
    assert metrics["sample_mean_ADE_m"] == 0.0
    assert metrics["joint_min_ADE_m"] == 0.0


def test_fair_energy_excludes_diagonal_pair_distances():
    target = np.zeros((1, 1, 1))
    samples = np.asarray([[[[0.0]]], [[[2.0]]]])
    # First term is 1; ordered off-diagonal correction is 1.
    assert np.isclose(
        fair_energy_score(samples, target, np.ones((1, 1), bool), np.ones(1)), 0.0
    )
    assert np.isclose(fair_crps(np.asarray([0.0, 2.0]), 0.0), 0.0)


def test_joint_minade_uses_one_world_for_all_agents():
    target = np.zeros((1, 2, 4))
    samples = np.zeros((2, 1, 2, 4))
    samples[0, 0, 1, 0] = 10.0
    samples[1, 0, 0, 0] = 10.0
    result = trajectory_errors(samples, target, np.ones((1, 2), bool))
    assert result["joint_min_ADE_m"] == 5.0


def test_deterministic_ensemble_has_no_pairwise_reward():
    target = np.zeros((2, 1, 2))
    prediction = np.ones_like(target)
    samples = np.repeat(prediction[None], 4, axis=0)
    score = fair_energy_score(samples, target, np.ones((2, 1), bool), np.ones(2))
    assert np.isclose(score, 1.0)


def test_channel_calibration_uses_fair_crps_and_linear_interval():
    target = np.asarray([[0.0, 1.0], [2.0, 3.0]])
    samples = np.stack((target - 1.0, target + 1.0))
    metrics = ensemble_channel_metrics(
        samples, target, np.ones(2, bool), ("x", "v"), level=0.9
    )
    # For two symmetric members: first term 1, fair pair correction 1.
    assert np.isclose(metrics["fair_CRPS_x"], 0.0)
    assert metrics["coverage_90_x"] == 1.0
    assert np.isclose(metrics["interval_width_90_v"], 1.8)


def test_behavior_brier_is_zero_when_all_samples_match_class():
    initial = np.zeros(6)
    initial[2] = 10.0
    target = np.zeros((75, 2, 6))
    target[:, 1, 2] = 10.0
    target[-1, 1, 1] = 3.5
    samples = np.repeat(target[None], 4, axis=0)
    metrics = _behavior_brier(samples, target, 1, initial)
    assert metrics["lane_behavior_brier"] == 0.0
    assert metrics["braking_behavior_brier"] == 0.0

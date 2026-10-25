from __future__ import annotations

from pathlib import Path

import numpy as np

from ads.idm.src.idm_policy import HighwayEnvIDMPolicy
from ads.idm.src.metrics import collision_and_min_gap
from ads.idm.src.world_evaluator import WorldEvaluation
from ads.idm.src.world_subset_simulation import (
    _mutated_world,
    run_world_subset_simulation,
)
from hierarchical_world_model.src.empirical_context import EmpiricalKWorldState
from hierarchical_world_model.src.randomness import WorldExogenousState
from tools.idm_ego import IDM_PARAMETER_KEYS, load_idm_ego_config


def test_actual_length_changes_collision_and_clearance() -> None:
    states = np.zeros((2, 1, 7, 6), np.float32)
    states[:, :, 1, 0] = 7.0
    valid = np.zeros((2, 7), bool)
    valid[:, :2] = True
    dimensions = np.full((2, 7, 2), (4.8, 1.8), np.float32)
    dimensions[1, 1, 0] = 16.0

    collision, gap = collision_and_min_gap(states, valid, dimensions)

    np.testing.assert_array_equal(collision, [False, True])
    np.testing.assert_allclose(gap, [2.2, 0.0])


def test_idm_policy_is_a_highway_env_marker() -> None:
    policy = HighwayEnvIDMPolicy.from_dict(
        {"target_speed": 30.0, "COMFORT_ACC_MAX": 3.0}
    )
    assert policy.highway_env_idm_config == {
        "target_speed": 30.0,
        "COMFORT_ACC_MAX": 3.0,
    }


def test_shared_idm_config_contains_only_executed_parameters() -> None:
    path = Path(__file__).resolve().parents[3] / "tools/idm_ego.yaml"
    values = load_idm_ego_config(path)
    assert set(values) == {"target_speed", "enable_lane_change", *IDM_PARAMETER_KEYS}
    assert HighwayEnvIDMPolicy.from_dict(values).highway_env_idm_config == values
    assert values["target_speed"] == 40.0
    assert values["enable_lane_change"] is False


def test_current_world_subset_uses_explicit_world_blocks() -> None:
    def evaluate(worlds: WorldExogenousState) -> WorldEvaluation:
        score = worlds.diffusion_noise[:, 0, 0].astype(np.float64)
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
        max_levels=3,
        failure_threshold=3.0,
        response_steps=1,
        mutation_blocks=("diffusion_noise",),
        pcn_beta=0.3,
        mcmc_steps=2,
        seed=17,
    )
    assert len(result.levels) == 3
    assert result.total_evaluations == 100
    assert result.proposal_evaluations == 80
    assert np.isfinite(result.probability)
    assert result.levels[-1].worlds.batch_size == 20


def test_joint_mutation_sweeps_each_requested_randomness_block() -> None:
    worlds = WorldExogenousState.sample(3, seed=11, response_steps=1)
    proposal = _mutated_world(
        worlds,
        blocks=("scenario", "diffusion_noise", "policy_response_innovations"),
        beta=0.3,
        rng=np.random.default_rng(12),
    )
    assert not np.array_equal(proposal.scenario_uniform, worlds.scenario_uniform)
    assert not np.array_equal(proposal.diffusion_noise, worlds.diffusion_noise)
    assert not np.array_equal(
        proposal.policy_response_innovations,
        worlds.policy_response_innovations,
    )
    assert np.array_equal(proposal.c0_base_latent, worlds.c0_base_latent)


def test_tracking_npc_subset_replacement_preserves_response_horizon() -> None:
    def evaluate(values: WorldExogenousState) -> WorldEvaluation:
        score = values.c0_base_latent[:, 0].astype(np.float64)
        return WorldEvaluation(
            evt_score=score,
            event_risk=np.maximum(score, 0.0),
            collision=np.zeros(values.batch_size, bool),
            min_gap_m=np.ones(values.batch_size),
            numerical_valid=np.ones(values.batch_size, bool),
        )

    result = run_world_subset_simulation(
        evaluate,
        num_samples=4,
        p0=0.25,
        max_levels=2,
        failure_threshold=10.0,
        response_steps=51,
        mutation_blocks=("c0_base_latent",),
        pcn_beta=0.2,
        mcmc_steps=1,
        seed=32,
        sample_worlds=lambda count, seed: WorldExogenousState.sample(
            count,
            seed=seed,
            response_steps=51,
        ),
    )
    assert len(result.levels) == 2
    for level in result.levels:
        level.worlds.validate()
        assert level.worlds.response_steps == 51


def test_empirical_k_world_mutation_never_resamples_c0_or_k() -> None:
    worlds = EmpiricalKWorldState.sample(3, seed=41, response_steps=1)
    proposal = _mutated_world(
        worlds,
        blocks=("test_context", "diffusion_noise"),
        beta=0.3,
        rng=np.random.default_rng(42),
    )
    assert not np.array_equal(proposal.test_row_uniform, worlds.test_row_uniform)
    assert not np.array_equal(proposal.diffusion_noise, worlds.diffusion_noise)
    assert proposal.as_dict().keys() == {
        "test_row_uniform",
        "diffusion_noise",
        "driver_latent",
        "maneuver_latent",
        "policy_response_innovations",
        "policy_calibration_innovations",
    }

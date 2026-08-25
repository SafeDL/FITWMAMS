from __future__ import annotations

import numpy as np

from hierarchical_world_model.src.randomness import WorldExogenousState
from IDM_subset.src.idm_policy import HighwayEnvIDMPolicy
from IDM_subset.src.world_evaluator import WorldEvaluation
from IDM_subset.src.world_subset_simulation import run_world_subset_simulation
from IDM_subset.src.world_subset_simulation import _mutated_world


def test_idm_policy_is_a_highway_env_marker() -> None:
    policy = HighwayEnvIDMPolicy.from_dict(
        {"target_speed": 30.0, "COMFORT_ACC_MAX": 3.0}
    )
    assert policy.highway_env_idm_config == {
        "target_speed": 30.0,
        "COMFORT_ACC_MAX": 3.0,
    }


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
        scene_dim=2,
        agent_dim=2,
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
    worlds = WorldExogenousState.sample(
        3, seed=11, response_steps=1, scene_dim=2, agent_dim=2
    )
    proposal = _mutated_world(
        worlds,
        blocks=("scenario", "diffusion_noise", "agent_response_innovations"),
        beta=0.3,
        rng=np.random.default_rng(12),
    )
    assert not np.array_equal(proposal.scenario_uniform, worlds.scenario_uniform)
    assert not np.array_equal(proposal.diffusion_noise, worlds.diffusion_noise)
    assert not np.array_equal(
        proposal.agent_response_innovations,
        worlds.agent_response_innovations,
    )
    assert np.array_equal(proposal.c0_base_latent, worlds.c0_base_latent)

"""Contracts for the maintained human-response A2 candidate."""

from __future__ import annotations

import torch

from hierarchical_world_model.src.human_response_training import (
    conditional_energy_distance,
    loo_energy_rewards,
    mechanism_auxiliary_loss,
)
from hierarchical_world_model.src.reaction_controller import HumanResponseA2Controller
from hierarchical_world_model.src.randomness import WorldExogenousState


def test_cold_inactive_mapping_is_exact_hiqr_passthrough():
    base = torch.tensor([[[-.2, .3]]])
    mapped = HumanResponseA2Controller.map_final_action(
        base=base, rule=base - 2.0, authority=torch.zeros_like(base),
        active=torch.zeros_like(base, dtype=torch.bool), mechanism_allowed=torch.zeros_like(base, dtype=torch.bool),
        raw=torch.full((1, 1, 3), 3.0), previous_correction=torch.zeros_like(base),
        minimum=-8.0, maximum=4.0, dt_s=.04,
    )
    torch.testing.assert_close(mapped["final"], base, rtol=0.0, atol=1.0e-6)


def test_correction_release_obeys_induced_jerk_and_converges():
    base = torch.zeros(1, 1)
    previous = torch.full_like(base, -1.0)
    first = HumanResponseA2Controller.map_final_action(
        base=base, rule=base, authority=torch.zeros_like(base), active=torch.zeros_like(base, dtype=torch.bool),
        mechanism_allowed=torch.zeros_like(base, dtype=torch.bool), raw=torch.zeros(1, 1, 3),
        previous_correction=previous, minimum=-8.0, maximum=4.0, dt_s=.04,
    )
    assert bool(first["release"].item())
    assert float((first["final"] - previous).abs()) <= .48 + 1.e-6
    second = HumanResponseA2Controller.map_final_action(
        base=base, rule=base, authority=torch.zeros_like(base), active=torch.zeros_like(base, dtype=torch.bool),
        mechanism_allowed=torch.zeros_like(base, dtype=torch.bool), raw=torch.zeros(1, 1, 3),
        previous_correction=first["final"], minimum=-8.0, maximum=4.0, dt_s=.04,
    )
    assert float(second["final"].abs()) < float(first["final"].abs())


def test_synthetic_mechanism_loss_has_gradients_for_both_branches():
    intervention = torch.tensor([[-1.0]], requires_grad=True)
    baseline = torch.tensor([[.2]], requires_grad=True)
    loss, _ = mechanism_auxiliary_loss(
        model_intervention=intervention, model_baseline=baseline,
        idm_intervention=torch.tensor([[-1.4]]), idm_baseline=torch.tensor([[.1]]),
    )
    loss.backward()
    assert intervention.grad is not None and baseline.grad is not None
    assert intervention.grad.abs().sum() > 0 and baseline.grad.abs().sum() > 0


def test_loo_energy_reward_matches_naive_definition():
    futures = torch.tensor([[[0., 0.]], [[1., 0.]], [[2., 0.]]])
    human = torch.tensor([[[0., 0.]], [[1., 0.]]])
    reward = loo_energy_rewards(futures, human, torch.ones(2))
    scores = []
    for index in range(3):
        keep = torch.cat((futures[:index], futures[index + 1:]))
        scores.append(-conditional_energy_distance(keep, human, torch.ones(2)))
    expected = torch.stack(scores).mean() - torch.stack(scores)
    torch.testing.assert_close(reward, expected)


def test_extra_policy_randomness_does_not_change_legacy_blocks():
    state = WorldExogenousState.sample(1, seed=91, response_steps=3)
    assert state.policy_response_extra_innovations.shape == (1, 3, 6, 1)
    # Version-two stream construction is preserved for old A2 noise.
    import hashlib
    legacy_seed = int.from_bytes(hashlib.sha256(b"world_rng:2:91:policy_response_innovations").digest()[:8], "little")
    legacy = torch.from_numpy(__import__("numpy").random.default_rng(legacy_seed).standard_normal((1, 3, 6, 2), dtype=__import__("numpy").float32))
    torch.testing.assert_close(torch.from_numpy(state.policy_response_innovations), legacy)

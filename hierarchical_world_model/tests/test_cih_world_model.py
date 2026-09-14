"""Contracts for the maintained CIH-WM response policy."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hierarchical_world_model.src.cih_model import load_cih_method_config
from hierarchical_world_model.src.human_response_training import (
    conditional_energy_distance,
    loo_energy_rewards,
    mechanism_auxiliary_loss,
    prefix_loo_event_energy_rewards,
    event_energy_score,
    legacy_event_energy_score,
)
from hierarchical_world_model.src.reaction_controller import (
    CausalInfluenceResponsePolicy,
    ResponsePriorProposal,
    HumanResponseCalibrator,
)
from hierarchical_world_model.src.influence_graph import CausalInfluenceGraph
from hierarchical_world_model.src.reaction_evidence import event_identity
from world_model.src.core.evaluation_scope import scoped_canonical_trajectory


ROOT = Path(__file__).resolve().parents[2]


def test_cih_contract_keeps_same_rear_for_response_and_disables_unproven_channels():
    config = load_cih_method_config(
        ROOT / "hierarchical_world_model/config/cih_world_model.yaml"
    )
    scopes = config["evaluation"]["scopes"]
    assert scopes["released_factual_comparison"]["excluded_slots"] == []
    assert scopes["response_factual_reconstruction"]["excluded_slots"] == []
    assert scopes["ads_closed_loop"]["excluded_slots"] == []
    assert config["response_channels"]["lateral"]["enabled"] is False
    assert config["influence"]["enable_secondary"] is False
    assert config["method"]["ood_use_nominal_shadow"] is False
    assert "causal_shadow_delta_gain" not in config["method"]
    assert "paired_ego_gate_strength" not in config["method"]


def test_cih_merges_the_frozen_mechanism_guided_response_prior_into_one_chain():
    config = load_cih_method_config(
        ROOT / "hierarchical_world_model/config/cih_world_model.yaml"
    )
    assert config["method"]["response_prior"] == {
        "name": "mechanism_guided_response_prior", "frozen": True
    }
    assert (ROOT / config["paths"]["response_prior_lineage"]).is_file()
    payload = torch.load(
        ROOT / config["paths"]["response_prior_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    assert payload["schema"] == "reaction_residual_ppo"
    assert payload["controller_mode"] == "rl_residual_idm"


def test_full_population_scope_keeps_same_rear_available_to_the_factual_layer():
    states = torch.ones(1, 2, 7, 6)
    valid = torch.ones(1, 2, 7, dtype=torch.bool)
    scoped_states, scoped_valid = scoped_canonical_trajectory(states, valid)
    assert bool(scoped_valid[..., 2].all())
    assert bool(scoped_states[..., 2, :].all())


def test_cold_inactive_mapping_is_exact_factual_dynamics_passthrough():
    base = torch.tensor([[[-.2, .3]]])
    mapped = CausalInfluenceResponsePolicy.map_calibration_tensors(
        nominal=base, prior_correction=torch.zeros_like(base), authority=torch.zeros_like(base),
        active=torch.zeros_like(base, dtype=torch.bool), raw=torch.full((1, 1, 2), 3.0),
        previous_calibration=torch.zeros_like(base), minimum=-8.0, maximum=4.0, dt_s=.04,
    )
    torch.testing.assert_close(mapped["final"], base, rtol=0.0, atol=1.0e-6)


def test_event_gate_no_longer_changes_the_runtime_mapping():
    nominal = torch.tensor([[[-1.1]]])
    mapped = CausalInfluenceResponsePolicy.map_calibration_tensors(
        nominal=nominal,
        prior_correction=torch.tensor([[[-.9]]]),
        authority=torch.ones_like(nominal),
        active=torch.ones_like(nominal, dtype=torch.bool),
        raw=torch.zeros(1, 1, 2),
        previous_calibration=torch.zeros_like(nominal),
        minimum=-8.0,
        maximum=4.0,
        dt_s=.04,
        event_gate=torch.zeros_like(nominal),
    )
    torch.testing.assert_close(mapped["requested_total_correction"], torch.tensor([[[-.9]]]))
    assert mapped["final"].item() == pytest.approx(-.68)


def test_correction_release_obeys_induced_jerk_and_converges():
    base = torch.zeros(1, 1)
    previous = torch.full_like(base, -1.0)
    first = CausalInfluenceResponsePolicy.map_calibration_tensors(
        nominal=base, prior_correction=torch.zeros_like(base), authority=torch.zeros_like(base), active=torch.zeros_like(base, dtype=torch.bool),
        raw=torch.zeros(1, 1, 2), previous_calibration=previous, minimum=-8.0, maximum=4.0, dt_s=.04,
    )
    assert bool(first["release"].item())
    assert float((first["final"] - previous).abs()) <= .48 + 1.e-6
    second = CausalInfluenceResponsePolicy.map_calibration_tensors(
        nominal=base, prior_correction=torch.zeros_like(base), authority=torch.zeros_like(base), active=torch.zeros_like(base, dtype=torch.bool),
        raw=torch.zeros(1, 1, 2), previous_calibration=first["calibration"], minimum=-8.0, maximum=4.0, dt_s=.04,
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


def test_prefix_rewards_are_applied_at_declared_temporal_prefixes():
    futures = torch.zeros(3, 25, 2)
    observed = torch.zeros(25, 2)
    values = prefix_loo_event_energy_rewards(futures, observed, torch.ones(2))
    assert tuple(values) == (5, 10, 25)
    assert all(value.shape == (3,) for value in values.values())


def test_zero_adapter_mean_preserves_the_frozen_prior_proposal():
    nominal = torch.tensor([[[-1.1]]])
    proposal = ResponsePriorProposal(
        hidden=torch.zeros(1, 1, 128), distribution=torch.distributions.Normal(torch.zeros(1, 1, 2), torch.ones(1, 1, 2)),
        raw_action=torch.zeros(1, 1, 2), value=torch.zeros(1, 1), features=torch.zeros(1, 1, 172),
        rule_action_ax=torch.tensor([[[-1.4]]]), base_action_ax=torch.tensor([[[-.2]]]), authority=torch.ones(1, 1),
        active=torch.ones(1, 1, dtype=torch.bool), nominal_action_ax=nominal, correction_ax=torch.tensor([[[-.9]]]),
    )
    mapped = CausalInfluenceResponsePolicy.map_calibration(
        proposal=proposal, raw=torch.zeros(1, 1, 2), previous_calibration=torch.zeros(1, 1),
        minimum=-8.0, maximum=4.0, dt_s=.04,
    )
    torch.testing.assert_close(mapped["requested_total_correction"], proposal.correction_ax)
    assert mapped["final"].item() == pytest.approx(-.68)


def test_retention_mapping_contains_factual_and_prior_endpoints():
    nominal = torch.tensor([[-1.1, -1.1]])
    correction = torch.tensor([[-.9, -.9]])
    mapped = CausalInfluenceResponsePolicy.map_calibration_tensors(
        nominal=nominal, prior_correction=correction,
        active=torch.ones_like(nominal, dtype=torch.bool), authority=torch.ones_like(nominal),
        raw=torch.tensor([[[-1., 0.], [0., 0.]]]), previous_calibration=torch.zeros_like(nominal),
        minimum=-8., maximum=4., dt_s=.04, jerk_limit_mps3=100.,
    )
    torch.testing.assert_close(mapped["requested_total_correction"], torch.tensor([[0., -.9]]))


def test_fair_energy_excludes_diagonal_finite_ensemble_bias():
    futures = torch.tensor([[[0., 0.]], [[2., 0.]]])
    observed = torch.tensor([[1., 0.]])
    fair = event_energy_score(futures, observed, torch.ones(2))
    legacy = legacy_event_energy_score(futures, observed, torch.ones(2))
    assert fair.item() == pytest.approx(0.0)
    assert legacy.item() == pytest.approx(0.5)


def test_clearance_counter_reaches_recovery_and_preserves_relation():
    graph = CausalInfluenceGraph(stable_release_frames=3, recovery_frames=2)
    current = torch.zeros(1, 7, 6)
    valid = torch.ones(1, 7, dtype=torch.bool)
    current[:, 1, 0] = -15.0
    current[:, 0, 2] = 8.0
    current[:, 1, 2] = 10.0
    current[:, 0, 4] = -1.0
    history = current[:, None].repeat(1, 2, 1, 1)
    state = graph.update(current, valid, history, None)
    assert state.phase[0, 0].item() == 1
    assert state.parent[0, 0].item() == 0
    for distance in (55.0, 56.0, 57.0):
        previous = current.clone()
        current = current.clone()
        current[:, 1, 0] = -distance
        current[:, 0, 4] = 0.0
        history = torch.stack((previous, current), dim=1)
        state = graph.update(current, valid, history, state)
    assert state.phase[0, 0].item() == 2
    assert state.parent[0, 0].item() == 0
    assert state.role[0, 0].item() != 0


def test_event_identity_contains_pair_and_recording_without_collisions():
    keys = {
        event_identity(1, 7, 8, 100), event_identity(2, 7, 8, 100),
        event_identity(1, 9, 8, 100), event_identity(1, 7, 10, 100),
        event_identity(1, 7, 8, 101),
    }
    assert len(keys) == 5
    assert all(len(key) == 64 for key in keys)


def test_every_runtime_trainable_parameter_has_policy_or_value_path():
    adapter = HumanResponseCalibrator()
    assert all(not parameter.requires_grad for parameter in adapter.event_gate.parameters())
    features = torch.randn(3, adapter.feature_dim)
    distribution, value = adapter.distribution_and_value(features)
    (distribution.mean.square().mean() + value.square().mean() + distribution.scale.mean()).backward()
    trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    assert trainable and all(parameter.grad is not None for parameter in trainable)

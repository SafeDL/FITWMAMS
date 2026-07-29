import numpy as np
import torch

from world_model.src.firm import (
    FIRMBackgroundEnvironment,
    FIRMConfig,
    FIRMWorldModel,
    FIRMWorldRandomness,
)
from world_model.src.firm.action_flow import JointActionFlow
from world_model.src.traffic_graph.graph_schema import DynamicTrafficGraph


def _batch(batch_size=2):
    states = torch.zeros(batch_size, 150, 7, 6)
    states[..., 2] = 22.0
    return {
        "agent_states": states,
        "agent_valid": torch.ones(batch_size, 150, 7, dtype=torch.bool),
        "ego_index": torch.zeros(batch_size, dtype=torch.long),
        "actions_highd": torch.zeros(batch_size, 125, 6, 2),
        "behavior_anchor_raw": torch.zeros(batch_size, 6, 6),
        "behavior_anchor_valid": torch.ones(batch_size, 6, dtype=torch.bool),
    }


def _environment(cfg=None):
    torch.manual_seed(4)
    graph = DynamicTrafficGraph(
        timestamp=0.0,
        agent_ids=np.arange(7),
        agent_states=np.zeros((7, 6), np.float32),
        agent_valid=np.ones(7, bool),
        ego_index=0,
    )
    graph.agent_states[:, 2] = 22.0
    environment = FIRMBackgroundEnvironment(
        FIRMWorldModel(
            cfg or FIRMConfig(hidden_dim=32, world_latent_dim=8, action_flow_layers=2)
        )
    )
    environment.reset(graph, FIRMWorldRandomness(seed=8))
    return environment


def test_firm_uses_single_start_frame_and_no_map_parameters():
    environment = _environment()
    snapshot = environment.snapshot()
    assert len(snapshot["history_states"]) == 1
    assert not any("map" in key for key in environment.model.state_dict())


def test_firm_first_plan_ignores_future_highd_labels():
    model = FIRMWorldModel(
        FIRMConfig(
            hidden_dim=32,
            world_latent_dim=8,
            action_flow_layers=2,
        )
    ).eval()
    batch = _batch()
    changed = {key: value.clone() for key, value in batch.items()}
    changed["agent_states"][:, 25:] += 1000.0
    changed["actions_highd"] += 1000.0
    with torch.no_grad():
        original = model._closed_loop(batch, response_steps=1, deterministic=True)
        perturbed = model._closed_loop(changed, response_steps=1, deterministic=True)
    assert torch.allclose(original["joint_control_plans"], perturbed["joint_control_plans"])


def test_firm_joint_action_flow_scores_executed_prefix():
    model = FIRMWorldModel(FIRMConfig(hidden_dim=32, world_latent_dim=8, action_flow_layers=2))
    output = model.forward_training(_batch(), response_steps=1)
    assert torch.isfinite(output["loss"])
    assert output["joint_control_plans"].shape == (2, 1, 25, 6, 2)
    assert output["joint_jerk_plans"].shape == (2, 1, 25, 6, 2)
    assert torch.isfinite(output["prefix_nll"])


def test_firm_keeps_per_vehicle_centres_in_rollout():
    cfg = FIRMConfig(
        hidden_dim=32,
        world_latent_dim=8,
        action_flow_layers=2,
    )
    model = FIRMWorldModel(cfg)
    assert model.plan_field is not None
    rollout = model._closed_loop(_batch(), response_steps=1, deterministic=True)
    centres = rollout["raw_joint_jerk_centres"]
    assert centres is not None
    assert centres.shape == (2, 1, 25, 6, 2)
    assert torch.isfinite(centres).all()


def test_firm_horizon_consistency_supervises_the_complete_plan():
    cfg = FIRMConfig(
        hidden_dim=32,
        world_latent_dim=8,
        action_flow_layers=2,
        plan_horizon_control_weight=0.2,
        plan_horizon_state_weight=0.2,
    )
    output = FIRMWorldModel(cfg).forward_training(_batch(), response_steps=1)
    assert torch.isfinite(output["plan_horizon_control"])
    assert torch.isfinite(output["plan_horizon_position"])


def test_firm_action_flow_uses_a_reversible_conditional_centre():
    model = FIRMWorldModel(FIRMConfig(hidden_dim=32, world_latent_dim=8, action_flow_layers=2))
    flow = model.action_flow
    context = torch.zeros(2, 32)
    noise = torch.zeros(2, 5, 6, 2)
    with torch.no_grad():
        flow.base_location[-1].bias[0] = 0.2
    jerks = flow.sample(context, noise)
    valid = torch.ones(2, 5, 6, dtype=torch.bool)
    assert jerks[..., 0].abs().max() > 0.0
    assert torch.isfinite(flow.nll(jerks, valid, context))


def test_firm_residual_flow_preserves_execution_centre_and_score_routing():
    kwargs = {
        "context_dim": 8,
        "execute_frames": 5,
        "layers": 2,
    }
    flow = JointActionFlow(**kwargs)
    with torch.no_grad():
        for layer in flow.layers:
            layer.network[-1].bias[: flow.dimensions].fill_(3.0)
    context = torch.zeros(2, 8, requires_grad=True)
    zero_noise = torch.zeros(2, 5, 6, 2)
    assert torch.allclose(flow.sample(context, zero_noise), torch.zeros_like(zero_noise))
    valid = torch.ones(2, 5, 6, dtype=torch.bool)
    score = flow.nll(zero_noise, valid, context)
    score.backward()
    assert context.grad is None
    assert flow.base_location[-1].weight.grad is None
    assert flow.base_scale[-1].weight.grad is not None


def test_firm_start_consumes_the_flow_condition():
    model = FIRMWorldModel(FIRMConfig(hidden_dim=32, world_latent_dim=8, action_flow_layers=2))
    batch = _batch()
    current, valid = batch["agent_states"][:, 24], batch["agent_valid"][:, 24]
    ego = torch.nn.functional.one_hot(batch["ego_index"], 7).bool()
    zero = model.initialize(
        current,
        valid,
        ego,
        behavior_anchor=batch["behavior_anchor_raw"],
        behavior_anchor_valid=batch["behavior_anchor_valid"],
        flow_latent=torch.zeros(2, 76),
    )
    shifted = model.initialize(
        current,
        valid,
        ego,
        behavior_anchor=batch["behavior_anchor_raw"],
        behavior_anchor_valid=batch["behavior_anchor_valid"],
        flow_latent=torch.ones(2, 76),
    )
    assert not torch.allclose(zero["continuous_memory"], shifted["continuous_memory"])


def test_firm_keeps_the_flow_condition_in_every_action_context():
    model = FIRMWorldModel(FIRMConfig(hidden_dim=32, world_latent_dim=8, action_flow_layers=2))
    batch = _batch()
    current, valid = batch["agent_states"][:, 24], batch["agent_valid"][:, 24]
    ego = torch.nn.functional.one_hot(batch["ego_index"], 7).bool()
    zero = model.initialize(
        current, valid, ego, flow_latent=torch.zeros(2, 76)
    )
    shifted = model.initialize(
        current, valid, ego, flow_latent=torch.ones(2, 76)
    )
    common = (
        current[:, None], valid[:, None], current, valid, ego,
        zero["continuous_memory"], zero["world_latent"],
    )
    noise = torch.zeros(2, 5, 6, 2)
    first = model.plan_step(*common, zero["flow_embedding"], None, None, noise)
    second = model.plan_step(*common, shifted["flow_embedding"], None, None, noise)
    assert not torch.allclose(first["action_context"], second["action_context"])


def test_firm_snapshot_restores_continuous_random_world():
    environment = _environment()
    environment.step(np.array([0.0, 0.0, 22.0, 0.0, 0.0, 0.0], np.float32))
    snapshot = environment.snapshot()
    assert "flow_embedding" in snapshot
    expected = environment.step(np.array([1.0, 0.0, 22.0, 0.0, 0.0, 0.0], np.float32))
    environment.restore(snapshot)
    actual = environment.step(np.array([1.0, 0.0, 22.0, 0.0, 0.0, 0.0], np.float32))
    assert np.allclose(expected["background_states"], actual["background_states"])
    assert np.allclose(expected["world_latent"], actual["world_latent"])

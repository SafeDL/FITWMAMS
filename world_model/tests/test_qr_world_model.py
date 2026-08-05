"""Architecture, jointness, Flow lifecycle, and causal-rollout tests for QR-WM."""

from __future__ import annotations

import inspect
from unittest.mock import patch

import torch
from torch import nn

from normalizing_flow.src.features import slot_feature_index
from world_model.src.core.initial_behavior_anchor import start_state_from_flow_feature, start_state_from_flow_tensor
from world_model.src.qr.config import QRWorldModelConfig
from world_model.src.qr.environment import (
    BatchedQRWorldModelEnvironment,
    FlowStartMetadata,
    QRWorldModelEnvironment,
    WorldRandomness,
)
from world_model.src.qr.model import QueryRefineWorldModel
from world_model.src.qr.train import (
    _roll_fde,
    _tensorboard_writer,
    _write_tensorboard_epoch,
    load_qr_checkpoint,
    require_canonical_qr_checkpoint,
)


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    # 24 invalid compatibility entries, then S0..S149.  This is the formal
    # START(25 tick)+ROLL(124 tick) QR cache layout.
    frames, agents, maps, points = 174, 7, 8, 8
    state = torch.zeros(batch_size, frames, agents, 6)
    state[..., 0] = torch.arange(frames).view(1, frames, 1) * 0.8
    state[..., 0] += torch.arange(agents).view(1, 1, agents) * 12.0
    state[..., 1] = (torch.arange(agents).view(1, 1, agents) % 3 - 1) * 3.6
    state[..., 2] = 20.0 + torch.arange(agents).view(1, 1, agents) * 0.2
    state[..., 4] = 0.05
    state[..., 5] = 0.01
    return {
        "agent_states": state,
        "agent_valid": torch.cat((
            torch.zeros(batch_size, 24, agents, dtype=torch.bool),
            torch.ones(batch_size, frames - 24, agents, dtype=torch.bool),
        ), dim=1),
        "ego_index": torch.zeros(batch_size, dtype=torch.long),
        "map_polylines": torch.randn(batch_size, maps, points, 6),
        "map_polyline_valid": torch.ones(batch_size, maps, points, dtype=torch.bool),
        "lane_graph_edges": torch.tensor([[[0, 1, 0], [1, 2, 1]]], dtype=torch.long).expand(batch_size, -1, -1).clone(),
        "actions_highd": torch.zeros(batch_size, 149, agents - 1, 2),
        "is_evt_tail": torch.zeros(batch_size, dtype=torch.bool),
    }


def _model() -> QueryRefineWorldModel:
    return QueryRefineWorldModel(
        QRWorldModelConfig(
            hidden_dim=32, behavior_latent_dim=8, plan_frames=25, execute_frames=5, refinement_iterations=2,
            attention_layers=2, num_heads=4,
        )
    )


def test_qr_training_rollout_has_refined_receding_buffer_and_gradients() -> None:
    model, batch = _model(), _batch()
    result = model.forward_training(batch, response_steps=2, tbptt_steps=1)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert model.joint_refiner.blocks[0].agents.in_proj_weight.grad is not None
    assert model.joint_refiner.action_embedding[0].weight.grad is not None
    rollout = model.rollout_reconstruction(batch, response_steps=2, deterministic=True)
    assert rollout["predicted_states"].shape == (2, 10, 7, 6)
    assert rollout["background_future_actions"].shape == (2, 2, 25, 6, 2)
    assert rollout["refined_background_future_states"].shape == (2, 2, 25, 6, 6)
    assert not torch.equal(rollout["background_future_actions_before_refinement"], rollout["background_future_actions"])
    masks = rollout["background_future_action_masks"]
    assert masks["refinable"].shape == (2, 2, 25, 6)
    assert rollout["executed_background_action_masks"].shape == (2, 2, 5, 6)
    assert not hasattr(model, "world_update")
    assert not hasattr(model, "memory")
    assert len(model.encoder.relation_blocks) == model.cfg.attention_layers
    assert len(model.joint_refiner.blocks) == model.cfg.attention_layers


def test_qr_full_protocol_is_one_second_start_plus_4p96_second_roll() -> None:
    model, batch = _model().eval(), _batch()
    rollout = model.rollout_reconstruction(batch, deterministic=True)
    assert model.cfg.response_steps == 30
    assert rollout["predicted_states"].shape == (2, 149, 7, 6)
    assert rollout["target_states"].shape == (2, 149, 7, 6)
    assert rollout["background_future_actions"].shape == (2, 30, 25, 6, 2)
    assert rollout["start_reconstruction_frames"] == 25
    assert rollout["roll_frames"] == 124
    assert rollout["total_frames"] == 149
    # The final 5 Hz response contains only the four recorded highD transitions
    # S145->S146 through S148->S149; it must not supervise a fabricated fifth.
    assert rollout["executed_background_action_masks"][:, -1, 4].sum() == 0


def test_formal_rollouts_and_checkpoint_selection_use_true_start_initialization() -> None:
    model, batch = _model().eval(), _batch()
    with patch.object(model.encoder, "encode_start", wraps=model.encoder.encode_start) as encode_start:
        model.rollout_reconstruction(batch, response_steps=2, deterministic=True)
    # One call initializes latent/memory and one encodes the first plan; the
    # second 5 Hz response must instead use the temporal history route.
    assert encode_start.call_count == 2

    class OneBatchLoader:
        def __init__(self, values: dict[str, torch.Tensor]) -> None:
            self.values = values
            self.field_names = tuple(values)

        def __iter__(self):
            yield tuple(self.values[name] for name in self.field_names)

    with patch.object(model, "rollout_reconstruction", wraps=model.rollout_reconstruction) as rollout:
        _roll_fde(model, OneBatchLoader(batch), torch.device("cpu"), response_steps=2)
    assert rollout.call_args.kwargs["start_mode"] is True


def test_training_uses_one_complete_start_to_roll_path_per_batch() -> None:
    model, batch = _model(), _batch()
    with patch.object(model, "supervised_terms", wraps=model.supervised_terms) as terms:
        result = model.forward_training(batch, response_steps=2, tbptt_steps=1)
    assert terms.call_count == 1
    assert terms.call_args.kwargs["start_mode"] is True
    assert terms.call_args.kwargs["training"] is True
    assert torch.isfinite(result["loss"])


def test_qr_inference_does_not_read_background_future() -> None:
    model, batch = _model(), _batch()
    model.eval()
    reference = model.rollout_reconstruction(batch, response_steps=2, deterministic=True)["predicted_states"]
    altered = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    altered["agent_states"][:, 25:, 1:] += 10_000.0
    altered["actions_highd"] += 1_000.0
    candidate = model.rollout_reconstruction(altered, response_steps=2, deterministic=True)["predicted_states"]
    assert torch.equal(reference, candidate)


def test_joint_refiner_has_cross_agent_action_gradients() -> None:
    model = _model().eval()
    refiner, h, z = model.joint_refiner, model.cfg.hidden_dim, model.cfg.behavior_latent_dim
    actions = torch.randn(1, 25, 6, 2, requires_grad=True)
    residual = refiner.residual(
        actions, torch.randn(1, 25, 6, 6), torch.randn(1, 7, h), torch.randn(1, h), torch.randn(1, h),
        torch.randn(1, 7, z), torch.randn(1, 4, h), torch.ones(1, 4, dtype=torch.bool),
        torch.ones(1, 25, 6, dtype=torch.bool),
    )
    residual[:, :, 0].square().mean().backward()
    assert actions.grad is not None
    assert actions.grad[:, :, 1].abs().sum() > 0


def test_b0_changes_only_start_initialization_and_flow_rollout() -> None:
    model, batch = _model().eval(), _batch()
    batch["behavior_anchor_raw"] = torch.zeros(2, 6, 6)
    batch["behavior_anchor_valid"] = torch.ones(2, 6, dtype=torch.bool)
    first = model.rollout_reconstruction(batch, response_steps=2, deterministic=True)
    changed = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    changed["behavior_anchor_raw"][:, 0, 0] = 3.0
    second = model.rollout_reconstruction(changed, response_steps=2, deterministic=True)
    assert not torch.equal(first["background_future_actions"][:, 0], second["background_future_actions"][:, 0])
    assert "raw_anchor" not in model.plan_step.__code__.co_varnames

    feature = torch.zeros(2, 76)
    maps = torch.randn(2, 4, 5, 6)
    slots = torch.ones(2, 6, dtype=torch.bool)

    def start_actions(value: torch.Tensor) -> torch.Tensor:
        current, valid, anchor = model.flow_condition_to_scene(value, slots)
        ego = torch.zeros_like(valid); ego[:, 0] = True
        start = model.initialize_start(
            current, valid, ego, maps, torch.ones(2, 4, 5, dtype=torch.bool),
            torch.zeros(2, 1, 3, dtype=torch.long), anchor, slots,
        )
        return model.plan_step(
            current[:, None], valid[:, None], current, valid, ego, maps,
            torch.ones(2, 4, 5, dtype=torch.bool), torch.zeros(2, 1, 3, dtype=torch.long),
            previous_memory=start["scene_memory"],
            start_anchor_actions=start["start_anchor_actions"],
            start_behavior_seed=start["start_behavior_seed"], start_mode=True,
        )["background_future_actions"]

    flow_first = start_actions(feature)
    feature[:, 40] = 3.0
    assert not torch.equal(flow_first, start_actions(feature))


def test_observed_ego_state_conditions_background_and_invalid_slots_stay_masked() -> None:
    model, batch = _model().eval(), _batch()
    batch["agent_valid"][:, :, 6] = False
    rollout = model.rollout_reconstruction(batch, response_steps=2, deterministic=True)
    assert not rollout["background_future_action_masks"]["refinable"][..., -1].any()
    assert torch.equal(
        rollout["background_future_actions"][..., -1, :],
        torch.zeros_like(rollout["background_future_actions"][..., -1, :]),
    )


def test_qr_relation_value_path_excludes_invalid_pair_slots() -> None:
    """Invalid slots must not contribute learned relation values to valid tokens."""
    encoder = _model().encoder.eval()
    encoder.relation_blocks = nn.ModuleList()
    with torch.no_grad():
        for module in (encoder.current_mlp, encoder.relation_mlp, encoder.cross_attention):
            for parameter in module.parameters():
                parameter.zero_()
        # Make the learned value depend only on pairwise longitudinal displacement.
        encoder.relation_mlp[0].weight[0, 0] = 1.0
        encoder.relation_mlp[2].weight[0, 0] = 1.0

    current = torch.zeros(1, 7, 6)
    current_valid = torch.tensor([[True, True, False, False, False, False, False]])
    ego_mask = torch.tensor([[True, False, False, False, False, False, False]])
    map_polylines = torch.zeros(1, 1, 1, 6)
    map_polyline_valid = torch.zeros(1, 1, 1, dtype=torch.bool)
    baseline = encoder.encode_start(current, current_valid, ego_mask, map_polylines, map_polyline_valid)[0]

    # This creates a large relation value only for pairs whose target is invalid.
    current[:, 2, 0] = 100_000.0
    changed = encoder.encode_start(current, current_valid, ego_mask, map_polylines, map_polyline_valid)[0]

    assert torch.equal(changed[:, :2], baseline[:, :2])


def test_start_mix_is_convex_decaying_and_start_summary_is_trained() -> None:
    model, batch = _model(), _batch()
    fresh = torch.ones(1, 25, 6, 2)
    anchor = torch.full_like(fresh, -3.0)
    fresh[..., 1] = 0.1
    anchor[..., 1] = -0.3
    mixed = model._mix_start_actions(fresh, anchor)
    assert torch.allclose(mixed[:, 0, :, 0], torch.full_like(mixed[:, 0, :, 0], -2.0))
    assert torch.allclose(mixed[:, 0, :, 1], torch.full_like(mixed[:, 0, :, 1], -0.2))
    assert torch.allclose(mixed[:, -1], fresh[:, -1])
    absent_anchor, _ = model._start_anchor(batch["agent_states"][:, 24], batch["agent_valid"][:, 24], None, None)
    assert absent_anchor is None
    batch["behavior_anchor_raw"] = torch.zeros(2, 6, 6)
    batch["behavior_anchor_valid"] = torch.ones(2, 6, dtype=torch.bool)
    start = model.supervised_terms(batch, response_steps=5, start_mode=True, training=True)
    assert torch.isfinite(start["start_summary"])
    terms = model.forward_training(batch, response_steps=2, tbptt_steps=1)
    assert "start_summary" in terms
    assert torch.isfinite(terms["loss"])


def test_qr_does_not_read_future_ego_before_it_is_observed() -> None:
    model, batch = _model().eval(), _batch()
    assert not hasattr(model, "rollout")
    assert "ego_future_controls" not in inspect.signature(model.rollout_reconstruction).parameters
    assert "ego_future" not in " ".join(inspect.signature(model.plan_step).parameters)
    assert "behavior_latent" not in inspect.signature(model.plan_step).parameters
    assert "ego" not in " ".join(inspect.signature(model.scene_memory.forward).parameters)
    assert "noise_level" not in inspect.signature(model.joint_refiner.residual).parameters
    assert not hasattr(model.joint_refiner, "noise_embedding")
    assert not any("noise" in name or "denois" in name for name in model.cfg.__dataclass_fields__)
    assert torch.equal(model.joint_refiner.action_scale, torch.tensor((1.5, 0.15)))
    try:
        model.rollout_reconstruction(batch, response_steps=2, ego_future_controls=torch.zeros(2, 10, 2))
    except TypeError:
        pass
    else:
        raise AssertionError("QR-WM must not accept a future ego-control argument")
    reference = model.rollout_reconstruction(batch, response_steps=2, deterministic=True)
    repeated = model.rollout_reconstruction(batch, response_steps=2, deterministic=True)
    assert torch.equal(reference["background_future_actions"], repeated["background_future_actions"])
    changed = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    changed["agent_states"][:, 25:, 0, :4] += 10_000.0
    candidate = model.rollout_reconstruction(changed, response_steps=2, deterministic=True)
    assert torch.equal(reference["background_future_actions"][:, 0], candidate["background_future_actions"][:, 0])
    assert torch.equal(reference["predicted_states"][:, :5, 1:], candidate["predicted_states"][:, :5, 1:])


def test_qr_explicit_behavior_latent_noise_controls_stochastic_reconstruction() -> None:
    model, batch = _model().eval(), _batch()
    noise = torch.linspace(-1.0, 1.0, 2 * 2 * 7 * model.cfg.behavior_latent_dim).reshape(
        2, 2, 7, model.cfg.behavior_latent_dim
    )
    torch.manual_seed(11)
    first = model.rollout_reconstruction(
        batch, response_steps=2, deterministic=False, behavior_standard_normal=noise
    )
    torch.manual_seed(99_999)
    repeated = model.rollout_reconstruction(
        batch, response_steps=2, deterministic=False, behavior_standard_normal=noise
    )
    assert torch.equal(first["predicted_states"], repeated["predicted_states"])
    changed = model.rollout_reconstruction(
        batch, response_steps=2, deterministic=False, behavior_standard_normal=-noise
    )
    assert not torch.equal(first["predicted_states"], changed["predicted_states"])


def test_qr_samples_a_new_conditional_innovation_for_each_response() -> None:
    """Changing only z_1 must leave the realised START response unchanged."""
    model, batch = _model().eval(), _batch()
    noise = torch.linspace(-1.0, 1.0, 2 * 2 * 7 * model.cfg.behavior_latent_dim).reshape(
        2, 2, 7, model.cfg.behavior_latent_dim
    )
    changed_noise = noise.clone()
    changed_noise[:, 1] *= -1.0
    first = model.rollout_reconstruction(
        batch, response_steps=2, deterministic=False, behavior_standard_normal=noise,
    )
    changed = model.rollout_reconstruction(
        batch, response_steps=2, deterministic=False, behavior_standard_normal=changed_noise,
    )
    assert torch.equal(first["behavior_latent"][:, 0], changed["behavior_latent"][:, 0])
    assert torch.equal(first["background_future_actions"][:, 0], changed["background_future_actions"][:, 0])
    assert not torch.equal(first["behavior_latent"][:, 1], changed["behavior_latent"][:, 1])
    assert not torch.equal(first["background_future_actions"][:, 1], changed["background_future_actions"][:, 1])


def test_qr_snapshot_restore_and_branch_resampling_are_causal_and_replayable() -> None:
    """AMS children share a prefix but draw an auditable new ROLL innovation."""
    model, batch = _model().eval(), _batch()
    metadata = FlowStartMetadata(
        slot_valid=torch.ones(6, dtype=torch.bool).numpy(), map_polylines=batch["map_polylines"][0].numpy(),
        map_polyline_valid=batch["map_polyline_valid"][0].numpy(), lane_graph_edges=batch["lane_graph_edges"][0].numpy(),
        primary_slot_index=0, event_structure=[1, 0], mask_pattern=63, event_structure_id=0,
        event_structure_log_prob=-0.2, conditional_log_prob=-1.3, log_prob=-1.5,
    )
    c0, b0 = torch.zeros(40), torch.zeros(6, 6)
    c0[0] = 20.0
    parent = QRWorldModelEnvironment(model)
    parent.reset_from_flow(c0, b0, metadata, deterministic=False, world_randomness=WorldRandomness(seed=31))
    for _ in range(model.cfg.execute_frames):
        parent.step(torch.zeros(2))
    snapshot = parent.snapshot()
    assert snapshot.response_index == 1

    # Plain restore is exact replay under the parent's original innovation stream.
    replay = QRWorldModelEnvironment(model)
    replay.restore(snapshot)
    parent_next, replay_next = parent.step(torch.zeros(2)), replay.step(torch.zeros(2))
    assert torch.equal(
        torch.from_numpy(parent_next["background_future_actions"]),
        torch.from_numpy(replay_next["background_future_actions"]),
    )

    positive = WorldRandomness(innovation_standard_normal=torch.ones(7, model.cfg.behavior_latent_dim))
    negative = WorldRandomness(innovation_standard_normal=-torch.ones(7, model.cfg.behavior_latent_dim))
    left, right, repeated = (QRWorldModelEnvironment(model) for _ in range(3))
    left.branch_from_snapshot(snapshot, positive)
    right.branch_from_snapshot(snapshot, negative)
    repeated.branch_from_snapshot(snapshot, positive)
    left_next, right_next, repeated_next = (
        left.step(torch.zeros(2)), right.step(torch.zeros(2)), repeated.step(torch.zeros(2)),
    )
    assert torch.equal(
        torch.from_numpy(left_next["background_future_actions"]),
        torch.from_numpy(repeated_next["background_future_actions"]),
    )
    assert not torch.equal(
        torch.from_numpy(left_next["background_future_actions"]),
        torch.from_numpy(right_next["background_future_actions"]),
    )
    roll_audit = left_next["trace"]["world_randomness"]["response_innovations"][-1]
    assert roll_audit["response_index"] == 1 and roll_audit["kind"] == "roll"
    assert left_next["trace"]["world_randomness"]["branch_resampling"][-1]["response_index"] == 1

    # The vectorized path keeps the same row-wise branch semantics for AMS.
    features = torch.zeros(2, 76); features[:, 0] = torch.tensor((20.0, 22.0))
    slots = torch.ones(2, 6, dtype=torch.bool)
    batched = BatchedQRWorldModelEnvironment(model)
    batched.reset_from_flow_batch(
        features, slots, batch["map_polylines"], batch["map_polyline_valid"], batch["lane_graph_edges"],
        deterministic=False, world_randomness=(WorldRandomness(seed=41), WorldRandomness(seed=42)),
    )
    for _ in range(model.cfg.execute_frames):
        batched.step(torch.zeros(2, 2))
    batched_snapshot = batched.snapshot()
    batched_child = BatchedQRWorldModelEnvironment(model)
    batched_child.branch_from_snapshot(
        batched_snapshot,
        (
            WorldRandomness(innovation_standard_normal=torch.ones(7, model.cfg.behavior_latent_dim)),
            WorldRandomness(innovation_standard_normal=-torch.ones(7, model.cfg.behavior_latent_dim)),
        ),
    )
    batched_next = batched_child.step(torch.zeros(2, 2))
    assert [row["response_innovations"][-1]["response_index"] for row in batched_next["world_randomness"]] == [1, 1]
    assert all("branch_resampling" in row for row in batched_next["world_randomness"])


def test_observed_ego_token_and_online_environment() -> None:
    model, batch = _model().eval(), _batch()
    refiner, h, z = model.joint_refiner, model.cfg.hidden_dim, model.cfg.behavior_latent_dim
    agents = torch.randn(1, 7, h, requires_grad=True)
    residual = refiner.residual(
        torch.randn(1, 25, 6, 2), torch.randn(1, 25, 6, 6), agents, torch.randn(1, h), torch.randn(1, h),
        torch.randn(1, 7, z), torch.randn(1, 4, h), torch.ones(1, 4, dtype=torch.bool),
        torch.ones(1, 25, 6, dtype=torch.bool),
    )
    residual.square().mean().backward()
    assert agents.grad is not None and agents.grad[:, 0].abs().sum() > 0

    metadata = FlowStartMetadata(
        slot_valid=torch.ones(6, dtype=torch.bool).numpy(), map_polylines=batch["map_polylines"][0].numpy(),
        map_polyline_valid=batch["map_polyline_valid"][0].numpy(), lane_graph_edges=batch["lane_graph_edges"][0].numpy(),
        primary_slot_index=0, event_structure=[1, 0], mask_pattern=63, event_structure_id=0, event_structure_log_prob=-0.2,
        conditional_log_prob=-1.3, log_prob=-1.5,
    )
    c0, b0 = torch.zeros(40), torch.zeros(6, 6)
    c0[0] = 20.0
    left, right = QRWorldModelEnvironment(model), QRWorldModelEnvironment(model)
    assert left.reset_from_flow(c0, b0, metadata)["flow_metadata"]["log_prob"] == -1.5
    right.reset_from_flow(c0, b0, metadata)
    quiet_action, brake_action = torch.tensor([0.0, 0.0]), torch.tensor([-4.0, 0.0])
    first_left, first_right = left.step(quiet_action), right.step(brake_action)
    assert torch.equal(
        torch.from_numpy(first_left["background_future_actions"]),
        torch.from_numpy(first_right["background_future_actions"]),
    )
    # The action is not a QR-WM input, but the physical ego trajectory from
    # this response is part of the next 5 Hz causal condition.
    for _ in range(model.cfg.execute_frames - 1):
        left.step(quiet_action)
        right.step(brake_action)
    quiet = left.step(quiet_action)
    braking = right.step(brake_action)
    assert quiet["response_index"] == 1
    assert quiet["planner_updated"] and braking["planner_updated"]
    assert not torch.equal(torch.from_numpy(quiet["background_future_actions"]), torch.from_numpy(braking["background_future_actions"]))


def test_online_environment_advances_joint_physics_at_25hz_and_replans_at_5hz() -> None:
    from unittest.mock import patch

    model, batch = _model().eval(), _batch()
    metadata = FlowStartMetadata(
        slot_valid=torch.ones(6, dtype=torch.bool).numpy(), map_polylines=batch["map_polylines"][0].numpy(),
        map_polyline_valid=batch["map_polyline_valid"][0].numpy(), lane_graph_edges=batch["lane_graph_edges"][0].numpy(),
        primary_slot_index=0, event_structure=[1, 0], mask_pattern=63, event_structure_id=0,
        event_structure_log_prob=-0.2, conditional_log_prob=-1.3, log_prob=-1.5,
    )
    c0, b0 = torch.zeros(40), torch.zeros(6, 6)
    c0[0] = 20.0
    action = torch.tensor([2.0, 0.0])
    environment = QRWorldModelEnvironment(model)
    environment.reset_from_flow(c0, b0, metadata)
    with patch.object(model, "plan_step", wraps=model.plan_step) as planner:
        ticks = [environment.step(action) for _ in range(model.cfg.execute_frames + 1)]

    assert planner.call_count == 2
    assert [tick["planner_updated"] for tick in ticks] == [True, False, False, False, False, True]
    assert [tick["executed_plan_frame"] for tick in ticks[:5]] == list(range(model.cfg.execute_frames))
    first = torch.from_numpy(ticks[0]["agent_states"])[0]
    dt = model.cfg.simulation_dt_s
    assert torch.allclose(first, torch.tensor([20.0 * dt + 0.5 * 2.0 * dt ** 2, 0.0, 20.0 + 2.0 * dt, 0.0, 2.0, 0.0]))
    for index in range(model.cfg.execute_frames):
        assert torch.allclose(
            torch.from_numpy(ticks[index]["applied_background_actions"]),
            torch.from_numpy(ticks[0]["background_future_actions"])[index],
        )
    assert ticks[4]["response_index"] == 1


def test_advance_response_matches_five_physics_steps_for_single_and_batched_environments() -> None:
    model, batch = _model().eval(), _batch()
    metadata = FlowStartMetadata(
        slot_valid=torch.ones(6, dtype=torch.bool).numpy(), map_polylines=batch["map_polylines"][0].numpy(),
        map_polyline_valid=batch["map_polyline_valid"][0].numpy(), lane_graph_edges=batch["lane_graph_edges"][0].numpy(),
        primary_slot_index=0, event_structure=[1, 0], mask_pattern=63, event_structure_id=0,
        event_structure_log_prob=-0.2, conditional_log_prob=-1.3, log_prob=-1.5,
    )
    c0, b0 = torch.zeros(40), torch.zeros(6, 6)
    c0[0] = 20.0
    actions = torch.tensor(((1.0, 0.0), (0.5, 0.01), (0.0, 0.0), (-0.5, -0.01), (-1.0, 0.0)))
    stepped, grouped = QRWorldModelEnvironment(model), QRWorldModelEnvironment(model)
    stepped.reset_from_flow(c0, b0, metadata)
    grouped.reset_from_flow(c0, b0, metadata)
    expected = torch.stack([torch.from_numpy(stepped.step(action)["agent_states"]) for action in actions])
    actual = torch.from_numpy(grouped.advance_response(actions)["agent_state_frames"])
    assert torch.allclose(actual, expected, atol=1.0e-6)

    features = torch.zeros(2, 76); features[:, 0] = torch.tensor((20.0, 22.0))
    slots = torch.ones(2, 6, dtype=torch.bool)
    batch_actions = actions.unsqueeze(0).expand(2, -1, -1).clone()
    stepped_batch, grouped_batch = BatchedQRWorldModelEnvironment(model), BatchedQRWorldModelEnvironment(model)
    for environment in (stepped_batch, grouped_batch):
        environment.reset_from_flow_batch(
            features, slots, batch["map_polylines"], batch["map_polyline_valid"], batch["lane_graph_edges"], deterministic=True,
        )
    stepped_ticks = [
        stepped_batch.step(batch_actions[:, index])
        for index in range(model.cfg.execute_frames)
    ]
    assert stepped_ticks[0]["planner_updated"]
    assert torch.equal(stepped_ticks[0]["applied_ego_action"], batch_actions[:, 0])
    assert stepped_ticks[0]["background_future_actions"].shape == (2, 25, 6, 2)
    expected_batch = torch.stack([tick["agent_states"] for tick in stepped_ticks], dim=1)
    actual_batch = grouped_batch.advance_response(batch_actions)["agent_state_frames"]
    assert torch.allclose(actual_batch, expected_batch, atol=1.0e-6)


def test_advance_response_accepts_the_final_four_tick_highd_prefix() -> None:
    model, batch = _model().eval(), _batch()
    metadata = FlowStartMetadata(
        slot_valid=torch.ones(6, dtype=torch.bool).numpy(), map_polylines=batch["map_polylines"][0].numpy(),
        map_polyline_valid=batch["map_polyline_valid"][0].numpy(), lane_graph_edges=batch["lane_graph_edges"][0].numpy(),
        primary_slot_index=0, event_structure=[1, 0], mask_pattern=63, event_structure_id=0,
        event_structure_log_prob=-0.2, conditional_log_prob=-1.3, log_prob=-1.5,
    )
    c0, b0 = torch.zeros(40), torch.zeros(6, 6)
    c0[0] = 20.0
    actions = torch.tensor(((1.0, 0.0), (0.5, 0.01), (0.0, 0.0), (-0.5, -0.01)))
    stepped, grouped = QRWorldModelEnvironment(model), QRWorldModelEnvironment(model)
    stepped.reset_from_flow(c0, b0, metadata)
    grouped.reset_from_flow(c0, b0, metadata)
    expected = torch.stack([torch.from_numpy(stepped.step(action)["agent_states"]) for action in actions])
    actual = torch.from_numpy(grouped.advance_response(actions)["agent_state_frames"])
    assert torch.allclose(actual, expected, atol=1.0e-6)


def test_flow_start_has_no_synthetic_history_and_keeps_metadata() -> None:
    from unittest.mock import patch

    model, batch = _model().eval(), _batch()
    metadata = FlowStartMetadata(
        slot_valid=torch.ones(6, dtype=torch.bool).numpy(), map_polylines=batch["map_polylines"][0].numpy(),
        map_polyline_valid=batch["map_polyline_valid"][0].numpy(), lane_graph_edges=batch["lane_graph_edges"][0].numpy(),
        primary_slot_index=0, event_structure=[1, 0], mask_pattern=63, event_structure_id=0,
        event_structure_log_prob=-0.2, conditional_log_prob=-1.3, log_prob=-1.5,
    )
    environment = QRWorldModelEnvironment(model)
    with patch.object(model.encoder, "_temporal_tokens", wraps=model.encoder._temporal_tokens) as temporal:
        observation = environment.reset_from_flow(torch.tensor([20.0] + [0.0] * 39), torch.zeros(6, 6), metadata)
        environment.step(torch.zeros(2))
    assert not temporal.called
    assert observation["flow_metadata"]["log_prob"] == -1.5


def test_batched_flow_environment_matches_independent_deterministic_environments() -> None:
    model, batch = _model().eval(), _batch()
    features = torch.zeros(2, 76); features[:, 0] = torch.tensor((20.0, 22.0))
    slots = torch.ones(2, 6, dtype=torch.bool)
    batched = BatchedQRWorldModelEnvironment(model)
    batched.reset_from_flow_batch(
        features, slots, batch["map_polylines"], batch["map_polyline_valid"], batch["lane_graph_edges"],
        deterministic=True,
    )
    actions = torch.zeros(2, model.cfg.execute_frames, 2)
    actual = batched.advance_response(actions)
    expected = []
    for index in range(2):
        metadata = FlowStartMetadata(
            slot_valid=slots[index].numpy(), map_polylines=batch["map_polylines"][index].numpy(),
            map_polyline_valid=batch["map_polyline_valid"][index].numpy(), lane_graph_edges=batch["lane_graph_edges"][index].numpy(),
            primary_slot_index=0, event_structure=[1, 0], mask_pattern=63, event_structure_id=0,
            event_structure_log_prob=-0.2, conditional_log_prob=-1.3, log_prob=-1.5,
        )
        environment = QRWorldModelEnvironment(model)
        environment.reset_from_flow(features[index, :40], features[index, 40:].reshape(6, 6), metadata, deterministic=True)
        expected.append(torch.from_numpy(environment.advance_response(actions[index])["agent_state_frames"]))
    assert torch.allclose(actual["agent_state_frames"].cpu(), torch.stack(expected), atol=1.0e-6)
    assert actual["physics_step_index"] == model.cfg.execute_frames
    assert actual["response_index"] == 1
    assert actual["planning_updates"].tolist() == [True, False, False, False, False]
    assert torch.equal(
        actual["applied_background_action_frames"],
        actual["background_future_actions"][:, :model.cfg.execute_frames],
    )


def test_stochastic_world_seed_is_replayable_and_batch_rows_remain_independent() -> None:
    model, batch = _model().eval(), _batch()
    features = torch.zeros(2, 76)
    features[:, 0] = torch.tensor((20.0, 22.0))
    slots = torch.ones(2, 6, dtype=torch.bool)
    actions = torch.zeros(2, model.cfg.execute_frames, 2)
    controls = (WorldRandomness(seed=1001), WorldRandomness(seed=1002))
    batched = BatchedQRWorldModelEnvironment(model)
    batched.reset_from_flow_batch(
        features, slots, batch["map_polylines"], batch["map_polyline_valid"], batch["lane_graph_edges"],
        deterministic=False, world_randomness=controls,
    )
    actual = batched.advance_response(actions)
    assert [row["seed"] for row in batched.world_randomness_audit] == [1001, 1002]
    expected = []
    for index, control in enumerate(controls):
        metadata = FlowStartMetadata(
            slot_valid=slots[index].numpy(), map_polylines=batch["map_polylines"][index].numpy(),
            map_polyline_valid=batch["map_polyline_valid"][index].numpy(), lane_graph_edges=batch["lane_graph_edges"][index].numpy(),
            primary_slot_index=0, event_structure=[1, 0], mask_pattern=63, event_structure_id=0,
            event_structure_log_prob=-0.2, conditional_log_prob=-1.3, log_prob=-1.5,
        )
        environment = QRWorldModelEnvironment(model)
        torch.manual_seed(77 + index)  # Explicit world seeds must ignore global RNG state.
        observation = environment.reset_from_flow(
            features[index, :40], features[index, 40:].reshape(6, 6), metadata,
            deterministic=False, world_randomness=control,
        )
        assert observation["world_randomness"]["seed"] == control.seed
        expected.append(torch.from_numpy(environment.advance_response(actions[index])["agent_state_frames"]))
    assert torch.allclose(actual["agent_state_frames"].cpu(), torch.stack(expected), atol=1.0e-6)
    try:
        QRWorldModelEnvironment(model).reset_from_flow(
            features[0, :40], features[0, 40:].reshape(6, 6), metadata,
            deterministic=False,
        )
    except ValueError as exc:
        assert "explicit WorldRandomness" in str(exc)
    else:
        raise AssertionError("stochastic QR environments must reject implicit global RNG")


def test_flow_metadata_requires_a_matching_slot_mask() -> None:
    metadata = FlowStartMetadata(
        slot_valid=torch.tensor([True, False, False, False, False, False]).numpy(),
        map_polylines=torch.zeros(2, 2, 6).numpy(),
        map_polyline_valid=torch.ones(2, 2, dtype=torch.bool).numpy(),
        lane_graph_edges=torch.zeros(1, 3, dtype=torch.long).numpy(),
        primary_slot_index=0, event_structure=[1], mask_pattern=3, event_structure_id=0,
        event_structure_log_prob=-0.2, conditional_log_prob=-1.3, log_prob=-1.5,
    )
    try:
        metadata.validate()
    except ValueError as exc:
        assert "mask_pattern" in str(exc)
    else:
        raise AssertionError("Flow metadata must reject an inconsistent slot mask")


def test_flow_replay_control_adapter_recovers_25hz_speed_and_heading_changes() -> None:
    from world_model.src.qr.flow_evaluation import replay_states_to_ego_controls

    initial = torch.tensor([[0.0, 0.0, 10.0, 0.0, 0.0, 0.0]]).numpy()
    future = torch.tensor([[[0.4, 0.0, 10.04, 0.0, 0.0, 0.0], [0.8, 0.0, 0.0, 10.08, 0.0, 0.0]]]).numpy()
    controls = replay_states_to_ego_controls(initial, future, dt_s=0.04)
    assert torch.allclose(torch.from_numpy(controls[0, 0]), torch.tensor([1.0, 0.0]), atol=1.0e-5)
    assert controls[0, 1, 1] > 30.0


def test_qr_flow_adapter_restores_absolute_background_velocity_numerically() -> None:
    feature = torch.zeros(1, 76)
    feature[:, 0:4] = torch.tensor([[31.0, -2.0, 0.3, -0.4]])
    feature[:, slot_feature_index("same_front", "rel_x_m")] = 18.0
    feature[:, slot_feature_index("same_front", "rel_y_left_m")] = 3.6
    feature[:, slot_feature_index("same_front", "rel_vx_mps")] = -7.0
    feature[:, slot_feature_index("same_front", "rel_vy_left_mps")] = 1.5
    feature[:, slot_feature_index("same_front", "other_ax_mps2")] = -1.0
    feature[:, slot_feature_index("same_front", "other_ay_left_mps2")] = 0.5
    slots = torch.tensor([[True, False, False, False, False, False]])
    scene, valid, anchor = start_state_from_flow_tensor(feature, slots)[:3]
    numpy_scene, numpy_valid, numpy_anchor, _ = start_state_from_flow_feature(feature[0].numpy(), slots[0].numpy())
    assert torch.allclose(scene[0], torch.from_numpy(numpy_scene))
    assert torch.equal(valid[0], torch.from_numpy(numpy_valid))
    assert torch.allclose(anchor[0], torch.from_numpy(numpy_anchor))
    assert torch.equal(scene[0, 1, 2:4], torch.tensor([24.0, -0.5]))
    assert not valid[0, 2]
    qr_scene, _, _ = QueryRefineWorldModel.flow_condition_to_scene(feature, slots)
    assert torch.equal(qr_scene, scene)


def test_incompatible_checkpoint_model_type_is_rejected(tmp_path) -> None:
    checkpoint = tmp_path / "obsolete.pt"
    torch.save({"model_type": "obsolete_qr_world_model"}, checkpoint)
    try:
        load_qr_checkpoint(checkpoint)
    except ValueError as exc:
        assert "model_type" in str(exc)
    else:
        raise AssertionError("incompatible checkpoint must not load into QR-WM")


def test_qr_evaluation_requires_canonical_checkpoint_training_contract(tmp_path) -> None:
    historical = tmp_path / "historical.pt"
    torch.save(_model().checkpoint_payload(), historical)
    try:
        require_canonical_qr_checkpoint(load_qr_checkpoint(historical))
    except RuntimeError as exc:
        assert "raw-150-state START+ROLL" in str(exc)
    else:
        raise AssertionError("checkpoint without canonical training metadata must be rejected")

    canonical = tmp_path / "canonical.pt"
    model = _model()
    model.flow_schema_sha256 = "frozen-flow-schema-for-test"
    payload = model.checkpoint_payload()
    payload["training_protocol"] = {
        "sequence_cache_format": "qr_start_roll_raw150",
        "total_transition_frames": 149,
        "start_reconstruction_frames": 25,
        "roll_transition_frames": 124,
        "flow_b0_start_only": True,
        "start_encoder": "C0_plus_map_without_synthetic_history",
        "canonical_rollout_initialization": "encode_start_for_train_validation_selection_and_held_out",
        "start_semantics": "segment_start_behavior_reconstruction_not_risk_event_onset",
        "independent_roll_auxiliary": False,
        "response_conditioned_innovations": True,
        "innovation_sampling": "one_conditional_behavior_latent_per_5hz_response",
        "flow_schema_sha256": "frozen-flow-schema-for-test",
    }
    torch.save(payload, canonical)
    require_canonical_qr_checkpoint(load_qr_checkpoint(canonical))


def test_qr_tensorboard_records_batch_loss_and_epoch_metrics(tmp_path) -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tensorboard_writer, log_dir = _tensorboard_writer(tmp_path, {"tensorboard": True})
    assert tensorboard_writer is not None
    assert log_dir == tmp_path / "tensorboard"
    tensorboard_writer.add_scalar("batch/train/loss", 1.25, 0)
    _write_tensorboard_epoch(
        tensorboard_writer,
        {
            "epoch": 1,
            "rollout_seconds": 5.96,
            "train_loss": 1.0, "train_start_summary": 0.25, "train_plan_position": 1.1,
            "val_position": 0.5,
            "selection_metric": 1.75,
        },
    )
    tensorboard_writer.close()

    events = EventAccumulator(str(log_dir))
    events.Reload()
    tags = events.Tags()["scalars"]
    assert {
        "batch/train/loss", "epoch/train/loss", "epoch/train/start_summary", "epoch/train/plan_position",
        "epoch/validation/position", "selection/validation_fde_m",
    } <= set(tags)

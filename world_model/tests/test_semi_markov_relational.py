"""Dependency-light smoke tests for the semi-Markov relational path."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from world_model.src.graph_builder import DynamicTrafficGraphBuilder
from world_model.src.initial_behavior_anchor import FrozenLegacyFlowSchema, behavior_anchor_from_flow_feature, summarize_first_second_states
from world_model.src.legacy_flow_initializer import graph_and_anchor_from_legacy_flow
from world_model.src.adapters.round_adapter import RoundGraphAdapter
from world_model.src.adapters.highd_adapter import HighDGraphAdapter
from world_model.src.semi_markov_environment import SemiMarkovBackgroundEnvironment, WorldRandomness
from world_model.src.semi_markov_model import SemiMarkovRelationalWorldModel, SemiMarkovWorldModelConfig
from world_model.src.semi_markov_evaluation import _counterfactual_ego_batch
from world_model.src.metrics import physical_diagnostics
from world_model.src.sequential_dataset import (
    combine_sequence_caches,
    ensure_frozen_flow_behavior_anchor_cache,
    load_sequential_dataset,
    prepare_round_sequential_dataset,
    write_dynamic_sequence_cache,
)
from world_model.src.semi_markov_train import _load_initial_state, _loader, _prototypes, _teacher_forcing_ratio


class SemiMarkovRelationalTests(unittest.TestCase):
    def test_exact_26_point_flow_summary_and_validity(self):
        import torch
        states = torch.zeros((1, 26, 1, 6))
        states[0, :, 0, 2] = torch.linspace(10.0, 12.0, 26)
        states[0, :, 0, 3] = torch.linspace(-1.0, 2.0, 26)
        states[0, :, 0, 4] = torch.linspace(-3.0, 2.0, 26)
        states[0, :, 0, 5] = 0.5
        summary, valid = summarize_first_second_states(states, torch.ones((1, 26, 1), dtype=torch.bool))
        torch.testing.assert_close(summary[0, 0], torch.tensor((2.0, 3.0, -0.5, -3.0, 2.0, 0.5)))
        self.assertTrue(bool(valid[0, 0]))
        partial = torch.ones((1, 26, 1), dtype=torch.bool); partial[:, 7] = False
        summary, valid = summarize_first_second_states(states, partial)
        self.assertFalse(bool(valid[0, 0])); torch.testing.assert_close(summary, torch.zeros_like(summary))

    def test_atomic_flow_initializer_rejects_external_graph_mismatch(self):
        from normalizing_flow.src.features import EGO_FEATURES, SLOT_NAMES, slot_feature_index
        schema = FrozenLegacyFlowSchema.load(Path(__file__).resolve().parents[2] / "results/highd_tail_flow_best/dataset_schema.json")
        feature = np.zeros(76, np.float32); feature[EGO_FEATURES.index("ego_vx_mps")] = 20.0
        slot = SLOT_NAMES[0]
        for name, value in {"rel_x_m": 20.0, "rel_y_left_m": 0.0, "rel_vx_mps": -1.0, "rel_vy_left_mps": 0.0, "other_ax_mps2": 0.0, "other_ay_left_mps2": 0.0}.items():
            feature[slot_feature_index(slot, name)] = value
        scene = graph_and_anchor_from_legacy_flow(feature, np.array([True, False, False, False, False, False]), 0, {"frozen_flow_schema": schema})
        self.assertEqual(scene.primary_agent_index, 1)
        np.testing.assert_array_equal(scene.graph.agent_valid[1:], [True, False, False, False, False, False])
    def _batch(self):
        import torch
        b, n = 2, 4
        states = torch.zeros((b, 150, n, 6), dtype=torch.float32)
        states[..., 0, 2] = 20.0
        states[..., 1, 0] = 22.0; states[..., 1, 2] = 19.0
        states[..., 2, 0] = -18.0; states[..., 2, 2] = 21.0
        states[..., 3, 1] = 3.6; states[..., 3, 2] = 20.0
        lanes = torch.zeros((b, 3, 8, 6), dtype=torch.float32)
        lanes[:, 0, :, 1] = -3.6; lanes[:, 1, :, 1] = 0.0; lanes[:, 2, :, 1] = 3.6
        lanes[..., 2] = 1.0
        return {
            "agent_states": states, "agent_valid": torch.ones((b, 150, n), dtype=torch.bool),
            "ego_index": torch.zeros((b,), dtype=torch.long), "map_polylines": lanes,
            "map_polyline_valid": torch.ones((b, 3, 8), dtype=torch.bool),
            "lane_graph_edges": torch.tensor([[[0, 1, 1], [1, 0, 1], [1, 2, 1], [2, 1, 1]]], dtype=torch.long).expand(b, -1, -1).clone(),
            "actions_highd": torch.zeros((b, 125, n - 1, 2), dtype=torch.float32),
        }

    def test_training_loss_and_causal_prior_rollout_are_finite(self):
        import torch
        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(hidden_dim=24, num_latent_states=4, use_conflict_zones=True))
        batch = self._batch()
        batch["conflict_zone_features"] = torch.tensor([[[0.0, 0.0, 4.0, 1.0]]], dtype=torch.float32).expand(2, -1, -1).clone()
        batch["conflict_zone_valid"] = torch.ones((2, 1), dtype=torch.bool)
        output = model.forward_training(batch, teacher_forcing_ratio=0.5)
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertTrue(torch.isfinite(output["prior_roll_loss"]))
        self.assertTrue(torch.isfinite(output["prior_endpoint_roll_loss"]))
        self.assertTrue(torch.isfinite(output["prior_control_loss"]))
        self.assertTrue(torch.isfinite(output["late_prior_control_loss"]))
        output["loss"].backward()
        rollout = model.rollout_prior(batch, deterministic=True)
        self.assertEqual(tuple(rollout["predicted_states"].shape), (2, 125, 4, 6))

    def test_behavior_anchor_affects_only_the_first_second(self):
        import torch
        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(
            hidden_dim=24, num_latent_states=4, variant="m1",
        ))
        raw_anchor = torch.ones((1, 3, 6), requires_grad=True)
        residual = model._anchor_residual_controls(
            torch.zeros((1, 4, 24)), torch.zeros((1, 24)), torch.zeros((1, 24)), raw_anchor,
            torch.zeros((1, 4, 6)), [], torch.zeros((1, 5, 3, 2)), 0,
            torch.ones((1, 4), dtype=torch.bool),
        )
        # Only the residual output layer is zero-initialized, so M1 starts as
        # exactly the nominal plan without blocking gradients in its encoders.
        torch.testing.assert_close(residual, torch.zeros_like(residual), atol=0.0, rtol=0.0)
        self.assertEqual(model.cfg.behavior_anchor_response_steps, 5)

    def test_behavior_anchored_training_uses_only_logged_first_second(self):
        import torch
        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(
            hidden_dim=24, num_latent_states=4, variant="m1",
        ))
        batch = self._batch()
        batch["actions_highd"][:, :25, :, 0] = 0.5
        output = model.forward_training(batch, teacher_forcing_ratio=0.0, rollout_response_steps=5)
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertTrue(torch.isfinite(output["anchor_loss"]))
        output["loss"].backward()

    def test_cold_start_ignores_unavailable_pre_anchor_history(self):
        import torch
        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(
            hidden_dim=24, num_latent_states=4, variant="m1", cold_start_history=True,
        )).eval()
        batch = self._batch()
        changed = {name: value.clone() for name, value in batch.items()}
        changed["agent_states"][:, :24, 1:, :2] += 50.0
        with torch.no_grad():
            baseline = model.rollout_prior(batch, deterministic=True)["predicted_states"]
            repeated = model.rollout_prior(changed, deterministic=True)["predicted_states"]
        torch.testing.assert_close(baseline, repeated, atol=1.0e-6, rtol=0.0)

    def test_behavior_anchor_flow_extraction_and_environment_expiry(self):
        from normalizing_flow.src.features import SLOT_NAMES, trajectory_feature_index

        feature = np.zeros(76, np.float32)
        for index, name in enumerate(("delta_vx_1s_mps", "delta_vy_left_1s_mps", "mean_ax_1s_mps2", "min_ax_1s_mps2", "final_ax_1s_mps2", "mean_ay_left_1s_mps2")):
            feature[trajectory_feature_index(SLOT_NAMES[0], name)] = index + 1
        anchor, anchor_valid = behavior_anchor_from_flow_feature(feature, np.asarray([True, False, False, False, False, False]))
        np.testing.assert_allclose(anchor[0], np.arange(1, 7, dtype=np.float32))
        self.assertTrue(anchor_valid[0])
        self.assertFalse(anchor_valid[1])

        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(
            hidden_dim=24, num_latent_states=4, variant="m1",
        ))
        states = np.asarray([[0, 0, 20, 0, 0, 0], [25, 0, 19, 0, 0, 0]], np.float32)
        valid = np.ones(2, bool)
        builder = DynamicTrafficGraphBuilder()
        lanes, lane_valid, lane_edges = builder.straight_lane_map(states, valid)
        graph = builder.graph_at(
            timestamp=0.0, agent_ids=np.arange(2), states=states, valid=valid,
            ego_index=0, primary_agent_index=1, map_polylines=lanes,
            map_polyline_valid=lane_valid, lane_graph_edges=lane_edges,
        )
        environment = SemiMarkovBackgroundEnvironment(model)
        environment.reset(graph, behavior_anchor=np.ones((1, 6), np.float32))
        for _ in range(5):
            environment.step(states[0])
        snapshot = environment.snapshot()
        self.assertFalse(snapshot["behavior_anchor_active"])
        restored = SemiMarkovBackgroundEnvironment(model)
        restored.restore(snapshot)
        self.assertFalse(restored.snapshot()["behavior_anchor_active"])

    def test_fixed_teacher_forcing_ratio_overrides_short_continuation_schedule(self):
        scheduled = _teacher_forcing_ratio({"min_teacher_forcing": 0.0}, epoch=1, schedule_epochs=4, stage_one_epochs=0)
        fixed = _teacher_forcing_ratio({"fixed_teacher_forcing_ratio": 0.0}, epoch=1, schedule_epochs=4, stage_one_epochs=0)
        self.assertEqual(scheduled, 0.75)
        self.assertEqual(fixed, 0.0)
        with self.assertRaises(ValueError):
            _teacher_forcing_ratio({"fixed_teacher_forcing_ratio": 1.1}, epoch=1, schedule_epochs=1, stage_one_epochs=0)

    def test_random_length_tbptt_rollout_is_finite(self):
        import torch
        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(hidden_dim=24, num_latent_states=4))
        output = model.forward_training(
            self._batch(), teacher_forcing_ratio=0.0,
            rollout_response_steps=5, tbptt_response_steps=2,
        )
        self.assertEqual(tuple(output["predicted_states"].shape), (2, 25, 4, 6))
        self.assertEqual(int(output["rollout_response_steps"].item()), 5)
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()

    def test_counterfactual_response_probe_replaces_only_future_ego(self):
        import torch
        batch = self._batch()
        changed = _counterfactual_ego_batch(batch, kind="accelerate")
        self.assertTrue(torch.equal(changed["agent_states"][:, :25], batch["agent_states"][:, :25]))
        self.assertTrue(torch.equal(changed["agent_states"][:, :, 1:], batch["agent_states"][:, :, 1:]))
        self.assertGreater(float((changed["agent_states"][:, 25:, 0, 2] - batch["agent_states"][:, 25:, 0, 2]).abs().max()), 0.0)
        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(hidden_dim=24, num_latent_states=4))
        with torch.no_grad():
            baseline = model.rollout_prior(batch, deterministic=True)
            response = model.rollout_prior(changed, deterministic=True)
        self.assertTrue(torch.isfinite(response["controls"]).all())
        self.assertEqual(tuple(baseline["controls"].shape), tuple(response["controls"].shape))

    def test_causal_prior_rollout_does_not_read_future_background_validity(self):
        import torch
        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(hidden_dim=24, num_latent_states=4))
        reference = self._batch()
        altered = {name: value.clone() for name, value in reference.items()}
        # Keep all initial history and all ego observations identical, but
        # remove future background validity.  This is a target-mask change
        # only; a causal background rollout must not change its generated
        # states in response.
        altered["agent_valid"][:, 25:, 1:] = False
        with torch.no_grad():
            expected = model.rollout_prior(reference, deterministic=True)
            actual = model.rollout_prior(altered, deterministic=True)
        torch.testing.assert_close(expected["predicted_states"], actual["predicted_states"], atol=1.0e-6, rtol=0.0)

    def test_modal_duration_rollout_has_no_hidden_uniform_draw(self):
        import torch
        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(hidden_dim=24, num_latent_states=4))
        batch = self._batch()
        with torch.no_grad():
            first = model.rollout_prior(batch, seed=11, deterministic=True, deterministic_duration=True)
            second = model.rollout_prior(batch, seed=29, deterministic=True, deterministic_duration=True)
        torch.testing.assert_close(first["predicted_states"], second["predicted_states"], atol=1.0e-6, rtol=0.0)

    def test_dynamic_physical_diagnostic_requires_lateral_body_overlap(self):
        states = np.zeros((1, 1, 1, 6), np.float32)
        ego = np.zeros((1, 1, 6), np.float32)
        states[0, 0, 0, 0] = 1.0  # longitudinal body projection overlaps
        states[0, 0, 0, 1] = 3.6  # but this is an adjacent lane
        adjacent = physical_diagnostics(states, np.ones((1, 1, 1), bool), ego_future_states=ego, slot_names=None)
        self.assertFalse(adjacent["semantic_diagnostic_available"])
        self.assertEqual(adjacent["negative_gap_rate"], 0.0)
        states[0, 0, 0, 1] = 0.0
        overlapping = physical_diagnostics(states, np.ones((1, 1, 1), bool), ego_future_states=ego, slot_names=None)
        self.assertGreater(overlapping["negative_gap_rate"], 0.0)

    def test_checkpoint_prototypes_include_dynamic_change_statistics(self):
        batch = self._batch()
        field_names = tuple(batch)

        class OneBatch:
            def __iter__(self):
                return iter([tuple(batch[name] for name in field_names)])

        loader = OneBatch()
        loader.field_names = field_names
        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(hidden_dim=24, num_latent_states=4))
        report = _prototypes(model, loader, "cpu")
        for name in (
            "relation_edge_change_rate_by_state", "lane_assignment_change_rate_by_state",
            "primary_interaction_change_rate_by_state", "posterior_hard_boundary_duration_histogram",
        ):
            self.assertIn(name, report)
        self.assertEqual(len(report["relation_edge_change_rate_by_state"]), 4)

    def test_highd_checkpoint_can_initialize_optional_round_conflict_attention(self):
        highd = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(hidden_dim=24, num_latent_states=4, use_conflict_zones=False))
        target = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(hidden_dim=24, num_latent_states=4, use_conflict_zones=True))
        transfer = _load_initial_state(target, {"model_config": highd.config_payload(), "state_dict": highd.state_dict()})
        self.assertTrue(transfer["missing"])
        self.assertTrue(all(name.startswith(("encoder.conflict_", "encoder.ac_edge.")) for name in transfer["missing"]))

    def test_stepwise_latent_path_disables_duration_learning(self):
        import torch
        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(
            hidden_dim=24, num_latent_states=4, learn_duration=False,
        ))
        output = model.forward_training(self._batch(), rollout_response_steps=5)
        self.assertTrue(torch.allclose(output["posterior_boundary_probs"], torch.ones_like(output["posterior_boundary_probs"])))
        self.assertEqual(float(output["duration_nll"]), 0.0)
        rollout = model.rollout_prior(self._batch(), deterministic=True)
        self.assertTrue(all(all(duration == 1 for duration in item) for item in rollout["latent_durations"]))

    def test_roll_wrapper_matches_five_response_steps(self):
        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(hidden_dim=24, num_latent_states=4))
        states = np.asarray([[0, 0, 20, 0, 0, 0], [25, 0, 19, 0, 0, 0], [-20, 0, 21, 0, 0, 0]], np.float32)
        valid = np.ones(3, bool)
        builder = DynamicTrafficGraphBuilder()
        lanes, lane_valid, lane_edges = builder.straight_lane_map(states, valid)
        graph = builder.graph_at(timestamp=0.0, agent_ids=np.arange(3), states=states, valid=valid, ego_index=0,
            primary_agent_index=1, map_polylines=lanes, map_polyline_valid=lane_valid, lane_graph_edges=lane_edges)
        first = SemiMarkovBackgroundEnvironment(model); first.reset(graph, WorldRandomness(seed=7))
        wrapped = first.roll(np.repeat(states[:1], 25, axis=0), np.ones(25, bool))
        second = SemiMarkovBackgroundEnvironment(model); second.reset(graph, WorldRandomness(seed=7))
        direct = []
        for ego in second._causal_ego_extrapolation(states[0], 5, model.cfg.response_interval_s):
            direct.append(second.step(ego)["background_states"])
        np.testing.assert_allclose(wrapped["background_states"], np.concatenate(direct), atol=1e-6)

    def test_environment_forwards_conflict_map_features(self):
        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(
            hidden_dim=24, num_latent_states=4, use_conflict_zones=True,
        ))
        states = np.asarray([[0, 0, 20, 0, 0, 0], [25, 0, 19, 0, 0, 0]], np.float32)
        valid = np.ones(2, bool)
        builder = DynamicTrafficGraphBuilder()
        lanes, lane_valid, lane_edges = builder.straight_lane_map(states, valid)
        zones = np.asarray([[12.0, 0.0, 3.0, 1.0]], np.float32)
        graph = builder.graph_at(
            timestamp=0.0, agent_ids=np.arange(2), states=states, valid=valid,
            ego_index=0, primary_agent_index=1, map_polylines=lanes,
            map_polyline_valid=lane_valid, lane_graph_edges=lane_edges,
            conflict_zone_features=zones, conflict_zone_valid=np.ones(1, bool),
        )
        environment = SemiMarkovBackgroundEnvironment(model)
        environment.reset(graph, WorldRandomness(seed=7))
        batch, _ = environment._tensors()
        np.testing.assert_allclose(batch["conflict_zone_features"].cpu().numpy()[0], zones)
        self.assertTrue(bool(batch["conflict_zone_valid"].cpu().numpy()[0, 0]))
        result = environment.step(states[0])
        self.assertTrue(np.isfinite(result["background_states"]).all())

    def test_environment_integrates_every_physics_rate_control_curve_point(self):
        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(hidden_dim=24, num_latent_states=4))
        states = np.asarray([[0, 0, 20, 0, 0, 0], [25, 0, 19, 0, 0, 0]], np.float32)
        valid = np.ones(2, bool)
        builder = DynamicTrafficGraphBuilder()
        lanes, lane_valid, lane_edges = builder.straight_lane_map(states, valid)
        graph = builder.graph_at(
            timestamp=0.0, agent_ids=np.arange(2), states=states, valid=valid,
            ego_index=0, primary_agent_index=1, map_polylines=lanes,
            map_polyline_valid=lane_valid, lane_graph_edges=lane_edges,
        )
        environment = SemiMarkovBackgroundEnvironment(model)
        environment.reset(graph, WorldRandomness(seed=7))
        result = environment.step(states[0])
        self.assertEqual(tuple(result["control_curve"].shape), (5, 1, 2))
        self.assertTrue(np.isfinite(result["background_states"]).all())

    def test_snapshot_restore_preserves_latent_rng_history_and_trace(self):
        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(hidden_dim=24, num_latent_states=4))
        states = np.asarray([[0, 0, 20, 0, 0, 0], [25, 0, 19, 0, 0, 0], [-20, 0, 21, 0, 0, 0]], np.float32)
        valid = np.ones(3, bool)
        builder = DynamicTrafficGraphBuilder()
        lanes, lane_valid, lane_edges = builder.straight_lane_map(states, valid)
        graph = builder.graph_at(timestamp=0.0, agent_ids=np.arange(3), states=states, valid=valid, ego_index=0,
            primary_agent_index=1, map_polylines=lanes, map_polyline_valid=lane_valid, lane_graph_edges=lane_edges)
        source = SemiMarkovBackgroundEnvironment(model, model_checkpoint_hash="test-checkpoint")
        source.reset(graph, WorldRandomness(seed=17, event_structure={"primary": 1}, flow_base_sample=[0.25]))
        ego_path = source._causal_ego_extrapolation(states[0], 5, model.cfg.response_interval_s)
        source.step(ego_path[0]); source.step(ego_path[1])
        snapshot = source.snapshot()
        expected = [source.step(ego) for ego in ego_path[2:]]

        restored = SemiMarkovBackgroundEnvironment(model)
        restored.restore(snapshot)
        actual = [restored.step(ego) for ego in ego_path[2:]]
        for left, right in zip(expected, actual):
            np.testing.assert_allclose(left["background_states"], right["background_states"], atol=1e-6)
            self.assertEqual(left["latent_state"], right["latent_state"])
            self.assertEqual(left["remaining_duration"], right["remaining_duration"])
        self.assertEqual(expected[-1]["trace"], actual[-1]["trace"])
        self.assertEqual(actual[-1]["trace"]["model_checkpoint_hash"], "test-checkpoint")
        self.assertEqual(actual[-1]["trace"]["event_structure"], {"primary": 1})

    def test_merge_and_cross_topology_are_preserved_in_agent_edges(self):
        builder = DynamicTrafficGraphBuilder()
        states = np.asarray([[0, 0, 10, 0, 0, 0], [0, 8, 10, 0, 0, 0], [0, 16, 10, 0, 0, 0]], np.float32)
        valid = np.ones(3, bool)
        lanes = np.zeros((3, 3, 6), np.float32)
        lanes[:, :, 0] = np.asarray([-5.0, 0.0, 5.0], np.float32)
        lanes[0, :, 1] = 0.0; lanes[1, :, 1] = 8.0; lanes[2, :, 1] = 16.0
        lanes[:, :, 2] = 1.0
        graph = builder.graph_at(
            timestamp=0.0, agent_ids=np.arange(3), states=states, valid=valid, ego_index=0, primary_agent_index=1,
            map_polylines=lanes, map_polyline_valid=np.ones((3, 3), bool),
            lane_graph_edges=np.asarray([[0, 1, 2], [1, 2, 4]], np.int64),
        )
        feature_by_pair = {tuple(pair): feature for pair, feature in zip(graph.aa_edge_index.T, graph.aa_edge_features)}
        self.assertEqual(int(feature_by_pair[(0, 1)][6]), 2)  # merge
        self.assertEqual(int(feature_by_pair[(1, 2)][6]), 4)  # cross

    def test_round_adapter_preserves_variable_agents_and_curved_map(self):
        adapter = RoundGraphAdapter(top_r_lanes=2)
        timestamps = np.asarray([0.0, 0.2, 0.4], np.float32)
        states = np.zeros((3, 4, 6), np.float32)
        states[..., 2] = 8.0
        valid = np.asarray([[True, True, False, True], [True, True, True, True], [True, False, True, True]])
        polylines = np.zeros((3, 5, 6), np.float32)
        for lane in range(3):
            angle = np.linspace(lane * 0.15, lane * 0.15 + 0.4, 5)
            polylines[lane, :, 0] = 20.0 * np.cos(angle)
            polylines[lane, :, 1] = 20.0 * np.sin(angle)
            polylines[lane, :, 2] = -np.sin(angle)
            polylines[lane, :, 3] = np.cos(angle)
            polylines[lane, :, 4] = 3.6
        sequence = adapter.adapt(
            sequence_id="round-smoke", recording_id="round-recording", ego_id="42", timestamps=timestamps,
            agent_ids=np.asarray([42, 10, 11, 12]), agent_states=states, agent_valid=valid, ego_index=0,
            primary_agent_index=1, map_polylines=polylines, map_polyline_valid=np.ones((3, 5), bool),
            lane_graph_edges=np.asarray([[0, 1, 2], [1, 2, 2], [2, 0, 2]]), split="test",
        )
        self.assertEqual(sequence.agent_states.shape, (3, 4, 6))
        self.assertEqual(sequence.agent_lane_candidates.shape, (3, 4, 2))
        self.assertTrue(np.all(sequence.agent_lane_candidates[~valid] == -1))
        self.assertGreaterEqual(sequence.conflict_zone_features.shape[0], 1)
        graph = adapter.builder.graph_at(
            timestamp=float(timestamps[1]), agent_ids=sequence.agent_ids,
            states=sequence.agent_states[1], valid=sequence.agent_valid[1], ego_index=sequence.ego_index,
            primary_agent_index=sequence.primary_agent_index, map_polylines=sequence.map_polylines,
            map_polyline_valid=sequence.map_polyline_valid, lane_graph_edges=sequence.lane_graph_edges,
            conflict_zone_features=sequence.conflict_zone_features, conflict_zone_valid=sequence.conflict_zone_valid,
        )
        self.assertGreater(graph.ac_edge_index.shape[1], 0)

    def test_round_adapter_reads_standard_tracks_csv_and_vector_map(self):
        import json
        import tempfile
        from pathlib import Path

        import pandas as pd

        tracks = pd.DataFrame([
            {"frame": 100, "trackId": 7, "xCenter": 0.0, "yCenter": 0.0, "xVelocity": 7.0, "yVelocity": 0.0, "xAcceleration": 0.1, "yAcceleration": 0.0},
            {"frame": 101, "trackId": 7, "xCenter": 0.3, "yCenter": 0.0, "xVelocity": 7.0, "yVelocity": 0.0, "xAcceleration": 0.1, "yAcceleration": 0.0},
            {"frame": 100, "trackId": 9, "xCenter": 5.0, "yCenter": 1.0, "xVelocity": 6.0, "yVelocity": 0.0, "xAcceleration": 0.0, "yAcceleration": 0.0},
            {"frame": 101, "trackId": 9, "xCenter": 5.2, "yCenter": 1.0, "xVelocity": 6.0, "yVelocity": 0.0, "xAcceleration": 0.0, "yAcceleration": 0.0},
        ])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracks_path = root / "00_tracks.csv"; tracks.to_csv(tracks_path, index=False)
            map_path = root / "00_vector_map.json"
            map_path.write_text(json.dumps({
                "polylines": [[[-10.0, 0.0], [0.0, 0.0], [10.0, 0.0]], [[0.0, -10.0], [0.0, 0.0], [0.0, 10.0]]],
                "lane_graph_edges": [[0, 1, 4]],
            }), encoding="utf-8")
            sequence = RoundGraphAdapter(top_r_lanes=2).adapt_from_files(
                tracks_csv=tracks_path, vector_map=map_path, ego_id=7, start_frame=100, num_frames=2,
                recording_id="00", split="test", primary_agent_id=9,
            )
        self.assertEqual(sequence.agent_ids.tolist(), [7, 9])
        self.assertTrue(sequence.agent_valid.all())
        self.assertEqual(sequence.primary_agent_index, 1)
        self.assertGreaterEqual(len(sequence.conflict_zone_features), 1)

    def test_round_adapter_reads_official_lanelet2_osm_map(self):
        import tempfile
        from pathlib import Path

        # Three directed Lanelet2 lanelets: lane 0 succeeds to lane 1, while
        # lane 2 crosses lane 1.  Metric x/y tags keep the fixture compact;
        # production rounD OSM nodes may instead use lat/lon and are projected
        # to the same local metric geometry by the adapter.
        fixture = """<osm version=\"0.6\">
<node id=\"1\"><tag k=\"x\" v=\"0\"/><tag k=\"y\" v=\"1\"/></node><node id=\"2\"><tag k=\"x\" v=\"10\"/><tag k=\"y\" v=\"1\"/></node>
<node id=\"3\"><tag k=\"x\" v=\"0\"/><tag k=\"y\" v=\"-1\"/></node><node id=\"4\"><tag k=\"x\" v=\"10\"/><tag k=\"y\" v=\"-1\"/></node>
<node id=\"5\"><tag k=\"x\" v=\"10\"/><tag k=\"y\" v=\"1\"/></node><node id=\"6\"><tag k=\"x\" v=\"20\"/><tag k=\"y\" v=\"1\"/></node>
<node id=\"7\"><tag k=\"x\" v=\"10\"/><tag k=\"y\" v=\"-1\"/></node><node id=\"8\"><tag k=\"x\" v=\"20\"/><tag k=\"y\" v=\"-1\"/></node>
<node id=\"9\"><tag k=\"x\" v=\"14\"/><tag k=\"y\" v=\"-5\"/></node><node id=\"10\"><tag k=\"x\" v=\"14\"/><tag k=\"y\" v=\"5\"/></node>
<node id=\"11\"><tag k=\"x\" v=\"16\"/><tag k=\"y\" v=\"-5\"/></node><node id=\"12\"><tag k=\"x\" v=\"16\"/><tag k=\"y\" v=\"5\"/></node>
<way id=\"20\"><nd ref=\"1\"/><nd ref=\"2\"/></way><way id=\"21\"><nd ref=\"3\"/><nd ref=\"4\"/></way>
<way id=\"22\"><nd ref=\"5\"/><nd ref=\"6\"/></way><way id=\"23\"><nd ref=\"7\"/><nd ref=\"8\"/></way>
<way id=\"24\"><nd ref=\"9\"/><nd ref=\"10\"/></way><way id=\"25\"><nd ref=\"11\"/><nd ref=\"12\"/></way>
<relation id=\"100\"><member type=\"way\" ref=\"20\" role=\"left\"/><member type=\"way\" ref=\"21\" role=\"right\"/><tag k=\"type\" v=\"lanelet\"/></relation>
<relation id=\"101\"><member type=\"way\" ref=\"22\" role=\"left\"/><member type=\"way\" ref=\"23\" role=\"right\"/><tag k=\"type\" v=\"lanelet\"/></relation>
<relation id=\"102\"><member type=\"way\" ref=\"24\" role=\"left\"/><member type=\"way\" ref=\"25\" role=\"right\"/><tag k=\"type\" v=\"lanelet\"/></relation>
</osm>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "round_map.osm"
            path.write_text(fixture, encoding="utf-8")
            polylines, valid, edges, zones, zone_valid = RoundGraphAdapter.load_vector_map(path)
        self.assertEqual(polylines.shape[0], 3)
        self.assertTrue(valid.all())
        self.assertTrue(any(tuple(edge) == (0, 1, 0) for edge in edges.tolist()))
        self.assertTrue(any(int(edge[2]) == 4 for edge in edges.tolist()))
        self.assertGreaterEqual(len(zones), 1)
        self.assertTrue(zone_valid.any())

    def test_round_preparation_writes_trainable_dynamic_sequence_cache(self):
        import json
        import tempfile
        from pathlib import Path

        import pandas as pd

        rows = []
        for frame in range(150):
            rows.extend((
                {"frame": frame, "trackId": 7, "xCenter": 0.28 * frame, "yCenter": 0.0, "xVelocity": 7.0, "yVelocity": 0.0, "xAcceleration": 0.1, "yAcceleration": 0.0},
                {"frame": frame, "trackId": 9, "xCenter": 8.0 + 0.24 * frame, "yCenter": 1.0, "xVelocity": 6.0, "yVelocity": 0.0, "xAcceleration": 0.0, "yAcceleration": 0.0},
            ))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracks_path = root / "00_tracks.csv"; pd.DataFrame(rows).to_csv(tracks_path, index=False)
            map_path = root / "00_vector_map.json"
            map_path.write_text(json.dumps({
                "polylines": [[[-20.0, 0.0], [0.0, 0.0], [20.0, 0.0]], [[0.0, -20.0], [0.0, 0.0], [0.0, 20.0]]],
                "lane_graph_edges": [[0, 1, 4]],
            }), encoding="utf-8")
            output = root / "cache"
            config = {
                "paths": {"output_dir": str(output), "sequence_cache_dir": str(output), "round_tracks_csv": str(tracks_path), "round_vector_map": str(map_path)},
                "dataset": {"adapter": "round", "sequence_frames": 150, "sequence_stride_frames": 25, "frame_rate_hz": 25.0, "max_sequences": 1},
                "graph": {"top_r_lanes": 2, "lane_width_m": 3.6}, "split": {"seed": 42},
            }
            manifest = prepare_round_sequential_dataset(config, config_dir=root)
            arrays, loaded = load_sequential_dataset(output)
            self.assertEqual(manifest["cache_version"], "semi_markov_sequence_v4_dynamic_graph_conflicts")
            self.assertEqual(loaded["num_sequences"], 1)
            self.assertEqual(arrays["agent_states"].shape[1], 150)
            self.assertIn("conflict_zone_features", arrays)
            self.assertTrue(arrays["conflict_zone_valid"].any())
            split = {0: "train", 1: "val", 2: "test"}[int(arrays["split_index"][0])]
            loader = _loader(arrays, split, batch_size=1, maximum=0, shuffle=False, seed=1)
            self.assertIn("conflict_zone_features", loader.field_names)

    def test_joint_cache_pads_highway_and_round_graph_capacities(self):
        import tempfile
        from pathlib import Path

        adapter = RoundGraphAdapter(top_r_lanes=2)
        timestamp = np.arange(150, dtype=np.float32) / 25.0
        lanes = np.zeros((2, 3, 6), np.float32)
        lanes[0, :, 0] = (-5.0, 0.0, 5.0); lanes[1, :, 1] = (-5.0, 0.0, 5.0)
        lanes[..., 2] = 1.0
        def sequence(name, agents, split):
            states = np.zeros((150, agents, 6), np.float32)
            states[..., 2] = 8.0
            for index in range(agents): states[:, index, 0] = 5.0 * index
            return adapter.adapt(
                sequence_id=name, recording_id=name, ego_id="0", timestamps=timestamp,
                agent_ids=np.arange(agents), agent_states=states, agent_valid=np.ones((150, agents), bool), ego_index=0,
                primary_agent_index=1 if agents > 1 else -1, map_polylines=lanes,
                map_polyline_valid=np.ones((2, 3), bool), lane_graph_edges=np.asarray([[0, 1, 4]], np.int64), split=split,
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, joint = root / "first", root / "second", root / "joint"
            write_dynamic_sequence_cache([sequence("highway", 2, "train")], output_dir=first, source_dataset="highD", adapter="highd")
            write_dynamic_sequence_cache([sequence("round", 4, "test")], output_dir=second, source_dataset="rounD", adapter="round")
            manifest = combine_sequence_caches([first, second], output_dir=joint)
            arrays, _ = load_sequential_dataset(joint)
        self.assertEqual(manifest["num_sequences"], 2)
        self.assertEqual(arrays["agent_states"].shape[2], 4)
        self.assertEqual(arrays["actions_highd"].shape[2], 3)
        self.assertIn("conflict_zone_features", arrays)

    def test_frozen_flow_anchor_sidecar_is_reused_by_m1_batches(self):
        """Formal M1 batches read B0 instead of reconstructing it per step."""
        import tempfile
        import torch

        adapter = RoundGraphAdapter(top_r_lanes=2)
        timestamp = np.arange(150, dtype=np.float32) / 25.0
        states = np.zeros((150, 7, 6), np.float32)
        states[..., 2] = 10.0
        states[24:50, 1:, 2] = np.linspace(10.0, 12.0, 26, dtype=np.float32)[:, None]
        states[24:50, 1:, 4] = np.linspace(-1.0, 1.0, 26, dtype=np.float32)[:, None]
        lanes = np.zeros((2, 3, 6), np.float32); lanes[..., 2] = 1.0
        sequence = adapter.adapt(
            sequence_id="anchor", recording_id="anchor", ego_id="0", timestamps=timestamp,
            agent_ids=np.arange(7), agent_states=states, agent_valid=np.ones((150, 7), bool), ego_index=0,
            primary_agent_index=1, map_polylines=lanes, map_polyline_valid=np.ones((2, 3), bool),
            lane_graph_edges=np.asarray([[0, 1, 4]], np.int64), split="train",
        )
        schema = FrozenLegacyFlowSchema.load(Path(__file__).resolve().parents[2] / "results/highd_tail_flow_best/dataset_schema.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            write_dynamic_sequence_cache([sequence], output_dir=root, source_dataset="highD", adapter="highd")
            arrays, manifest = load_sequential_dataset(root)
            cached = ensure_frozen_flow_behavior_anchor_cache(root, arrays, manifest, schema)
            repeated = ensure_frozen_flow_behavior_anchor_cache(root, arrays, manifest, schema)
            self.assertIsInstance(repeated["behavior_anchor_raw"], np.memmap)
            np.testing.assert_allclose(cached["behavior_anchor_raw"][0, 0, :5], (2.0, 0.0, 0.0, -1.0, 1.0), atol=1.0e-6)
            arrays.update(cached)
            loader = _loader(arrays, "train", batch_size=1, maximum=0, shuffle=False, seed=1)
            values = next(iter(loader))
            batch = {name: value for name, value in zip(loader.field_names, values)}
            model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(hidden_dim=24, num_latent_states=4, variant="m1"))
            model.set_frozen_flow_schema(schema)
            raw, standardized, _agents, valid = model._batch_behavior_anchor(batch)
            torch.testing.assert_close(raw, batch["behavior_anchor_raw"])
            torch.testing.assert_close(standardized, batch["behavior_anchor_std"])
            self.assertTrue(bool(valid.all()))

    def test_curved_polyline_assignment_uses_nearest_geometry(self):
        import torch
        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(hidden_dim=24, num_latent_states=4))
        # Lane 0 curves through the agent but has a mean y=5m.  A y-average
        # heuristic would incorrectly prefer lane 1 (mean y=0.5m).
        lanes = torch.zeros((1, 2, 3, 6), dtype=torch.float32)
        lanes[0, 0, :, :2] = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 15.0]])
        lanes[0, 1, :, :2] = torch.tensor([[0.0, 0.5], [10.0, 0.5], [20.0, 0.5]])
        lanes[..., 2] = 1.0
        states = torch.tensor([[[10.0, 0.0, 8.0, 0.0, 0.0, 0.0]]])
        candidates = model._lane_candidates(states, lanes, torch.ones((1, 2, 3), dtype=torch.bool), top_r=1)
        self.assertEqual(int(candidates[0, 0, 0]), 0)

    def test_highd_recording_lane_metadata_is_localized(self):
        polylines, valid, edges = HighDGraphAdapter.map_from_recording_metadata(
            {"upperLaneMarkings": np.asarray([0.0, 3.5, 7.0]), "lowerLaneMarkings": np.asarray([10.0, 13.5, 17.0])},
            ego_global_y_m=5.25,
        )
        self.assertEqual(polylines.shape, (4, 8, 6))
        self.assertTrue(valid.all())
        self.assertEqual(edges.shape, (4, 3))
        self.assertAlmostEqual(float(polylines[0, :, 1].mean()), -3.5)
        mirrored, _, _ = HighDGraphAdapter.map_from_recording_metadata(
            {"upperLaneMarkings": np.asarray([0.0, 3.5]), "lowerLaneMarkings": np.asarray([10.0, 13.5])},
            ego_global_y_m=-1.75, lateral_sign=-1.0,
        )
        self.assertAlmostEqual(float(mirrored[0, :, 1].mean()), 0.0)

if __name__ == "__main__":
    unittest.main()

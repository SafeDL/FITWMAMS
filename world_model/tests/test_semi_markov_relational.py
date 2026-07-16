"""Core regression tests for the active highD M0/M1 world-model path."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from world_model.src.adapters.highd_adapter import HighDGraphAdapter
from world_model.src.graph_builder import DynamicTrafficGraphBuilder
from world_model.src.initial_behavior_anchor import (
    FrozenLegacyFlowSchema,
    behavior_anchor_from_flow_feature,
    start_state_from_flow_feature,
    summarize_first_second_states,
)
from world_model.src.semi_markov_environment import SemiMarkovBackgroundEnvironment, WorldRandomness
from world_model.src.semi_markov_model import SemiMarkovRelationalWorldModel, SemiMarkovWorldModelConfig
from world_model.src.semi_markov_state import SemiMarkovConfig, SemiMarkovLatentState
from world_model.src.sequential_dataset import (
    ensure_frozen_flow_behavior_anchor_cache,
    load_sequential_dataset,
    write_dynamic_sequence_cache,
)
from world_model.src.semi_markov_train import _loader


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
        partial = torch.ones((1, 26, 1), dtype=torch.bool)
        partial[:, 7] = False
        summary, valid = summarize_first_second_states(states, partial)
        self.assertFalse(bool(valid[0, 0]))
        torch.testing.assert_close(summary, torch.zeros_like(summary))

    def test_direct_flow_start_adapter_preserves_slot_order(self):
        from normalizing_flow.src.features import EGO_FEATURES, SLOT_NAMES, slot_feature_index

        feature = np.zeros(76, np.float32)
        feature[EGO_FEATURES.index("ego_vx_mps")] = 20.0
        for name, value in {
            "rel_x_m": 20.0, "rel_y_left_m": 0.0, "rel_vx_mps": -1.0,
            "rel_vy_left_mps": 0.0, "other_ax_mps2": 0.0, "other_ay_left_mps2": 0.0,
        }.items():
            feature[slot_feature_index(SLOT_NAMES[0], name)] = value
        states, valid, anchor, anchor_valid = start_state_from_flow_feature(
            feature, np.array([True, False, False, False, False, False]),
        )
        self.assertEqual(tuple(states[1]), (20.0, 0.0, 19.0, 0.0, 0.0, 0.0))
        np.testing.assert_array_equal(valid[1:], [True, False, False, False, False, False])
        np.testing.assert_array_equal(anchor_valid, valid[1:])
        self.assertEqual(anchor.shape, (6, 6))

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

    def test_training_loss_and_roll_mode_are_finite(self):
        import torch

        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(hidden_dim=24, num_latent_states=4))
        output = model.forward_training(self._batch(), teacher_forcing_ratio=0.5)
        for name in ("loss", "prior_roll_loss", "prior_endpoint_roll_loss", "prior_control_loss", "late_prior_control_loss"):
            self.assertTrue(torch.isfinite(output[name]))
        output["loss"].backward()
        rollout = model.rollout_roll_mode(self._batch(), deterministic=True)
        self.assertEqual(tuple(rollout["predicted_states"].shape), (2, 125, 4, 6))

    def test_start_residual_is_zero_at_initialization(self):
        import torch

        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(hidden_dim=24, num_latent_states=4, variant="m1"))
        residual = model._anchor_residual_controls(
            torch.zeros((1, 4, 24)), torch.zeros((1, 24)), torch.zeros((1, 24)),
            torch.ones((1, 3, 6), requires_grad=True), torch.zeros((1, 4, 6)), [],
            torch.zeros((1, 5, 3, 2)), 0, torch.ones((1, 4), dtype=torch.bool),
        )
        # Only the final residual layer is zero-initialized: M1 therefore
        # begins as the deterministic START controls.
        torch.testing.assert_close(residual, torch.zeros_like(residual), atol=0.0, rtol=0.0)
        self.assertEqual(model.cfg.behavior_anchor_response_steps, 5)

    def test_start_feedback_uses_frozen_flow_coordinates(self):
        """START feedback must never subtract physical values from B0_std."""
        import torch

        schema = FrozenLegacyFlowSchema.load(
            Path(__file__).resolve().parents[2] / "results/highd_tail_flow_best/dataset_schema.json"
        )
        model = SemiMarkovRelationalWorldModel(
            SemiMarkovWorldModelConfig(hidden_dim=24, num_latent_states=4, variant="m1")
        )
        model.set_frozen_flow_schema(schema)
        initial = torch.zeros((1, 7, 6)); initial[:, :, 2] = 20.0; initial[:, :, 4] = -1.0
        generated = initial.clone(); generated[:, :, 2] += 0.4; generated[:, :, 4] = -0.5
        raw = torch.tensor([[[0.4, 0.0, -0.75, -1.0, -0.5, 0.0]] * 6])
        expected = schema.standardize(raw, torch.ones((1, 6), dtype=torch.bool))
        actual = model._realized_prefix_anchor(initial, [generated], torch.ones((1, 7), dtype=torch.bool))
        torch.testing.assert_close(actual, expected)

    def test_duration_hazard_stays_at_the_completed_phase_start(self):
        """A phase ending at t must not use t's new scene/state as input."""
        import torch

        latent = SemiMarkovLatentState(SemiMarkovConfig(hidden_dim=2, num_states=2))
        one_hot = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
        boundaries = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        latent.posterior = lambda _scene: (one_hot, one_hot, boundaries, torch.zeros((1, 4)), torch.zeros((1, 4, 2)))
        calls: list[tuple[float, float]] = []

        def recorded_hazard(scene, _state, age):
            calls.extend(zip(scene[:, 0].tolist(), age.tolist()))
            return scene[:, 0] * 0.0

        latent.hazard_logits = recorded_hazard
        scene = torch.tensor([[[10.0, 0.0], [11.0, 0.0], [12.0, 0.0], [13.0, 0.0]]])
        latent.training_terms(scene, scene)
        self.assertEqual(calls, [(10.0, 1.0), (10.0, 2.0), (12.0, 1.0), (12.0, 2.0)])

    def test_behavior_anchored_training_uses_only_logged_first_second(self):
        import torch

        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(hidden_dim=24, num_latent_states=4, variant="m1"))
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
            baseline = model.rollout_roll_mode(batch, deterministic=True)["predicted_states"]
            repeated = model.rollout_roll_mode(changed, deterministic=True)["predicted_states"]
        torch.testing.assert_close(baseline, repeated, atol=1.0e-6, rtol=0.0)

    def test_behavior_anchor_flow_extraction_and_environment_expiry(self):
        from normalizing_flow.src.features import SLOT_NAMES, trajectory_feature_index

        feature = np.zeros(76, np.float32)
        for index, name in enumerate(("delta_vx_1s_mps", "delta_vy_left_1s_mps", "mean_ax_1s_mps2", "min_ax_1s_mps2", "final_ax_1s_mps2", "mean_ay_left_1s_mps2")):
            feature[trajectory_feature_index(SLOT_NAMES[0], name)] = index + 1
        anchor, anchor_valid = behavior_anchor_from_flow_feature(feature, np.asarray([True, False, False, False, False, False]))
        np.testing.assert_allclose(anchor[0], np.arange(1, 7, dtype=np.float32))
        self.assertTrue(anchor_valid[0]); self.assertFalse(anchor_valid[1])

        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(hidden_dim=24, num_latent_states=4, variant="m1"))
        states = np.asarray([[0, 0, 20, 0, 0, 0], [25, 0, 19, 0, 0, 0]], np.float32)
        builder = DynamicTrafficGraphBuilder(); lanes, lane_valid, lane_edges = builder.straight_lane_map(states, np.ones(2, bool))
        graph = builder.graph_at(timestamp=0.0, agent_ids=np.arange(2), states=states, valid=np.ones(2, bool), ego_index=0,
            primary_agent_index=1, map_polylines=lanes, map_polyline_valid=lane_valid, lane_graph_edges=lane_edges)
        environment = SemiMarkovBackgroundEnvironment(model)
        environment.reset(graph, behavior_anchor=np.ones((1, 6), np.float32))
        for _ in range(5):
            environment.step(states[0])
        snapshot = environment.snapshot()
        self.assertFalse(snapshot["behavior_anchor_active"])
        restored = SemiMarkovBackgroundEnvironment(model); restored.restore(snapshot)
        self.assertFalse(restored.snapshot()["behavior_anchor_active"])

    def test_random_length_tbptt_rollout_is_finite(self):
        import torch

        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(hidden_dim=24, num_latent_states=4))
        output = model.forward_training(self._batch(), teacher_forcing_ratio=0.0, rollout_response_steps=5, tbptt_response_steps=2)
        self.assertEqual(tuple(output["predicted_states"].shape), (2, 25, 4, 6))
        self.assertEqual(int(output["rollout_response_steps"].item()), 5)
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()

    def test_roll_mode_does_not_read_future_background_validity(self):
        import torch

        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(hidden_dim=24, num_latent_states=4))
        reference = self._batch()
        altered = {name: value.clone() for name, value in reference.items()}
        altered["agent_valid"][:, 25:, 1:] = False
        with torch.no_grad():
            expected = model.rollout_roll_mode(reference, deterministic=True)
            actual = model.rollout_roll_mode(altered, deterministic=True)
        torch.testing.assert_close(expected["predicted_states"], actual["predicted_states"], atol=1.0e-6, rtol=0.0)

    def test_roll_wrapper_matches_five_response_steps(self):
        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(hidden_dim=24, num_latent_states=4))
        states = np.asarray([[0, 0, 20, 0, 0, 0], [25, 0, 19, 0, 0, 0], [-20, 0, 21, 0, 0, 0]], np.float32)
        builder = DynamicTrafficGraphBuilder(); lanes, lane_valid, lane_edges = builder.straight_lane_map(states, np.ones(3, bool))
        graph = builder.graph_at(timestamp=0.0, agent_ids=np.arange(3), states=states, valid=np.ones(3, bool), ego_index=0,
            primary_agent_index=1, map_polylines=lanes, map_polyline_valid=lane_valid, lane_graph_edges=lane_edges)
        first = SemiMarkovBackgroundEnvironment(model); first.reset(graph, WorldRandomness(seed=7))
        wrapped = first.roll(np.repeat(states[:1], 25, axis=0), np.ones(25, bool))
        second = SemiMarkovBackgroundEnvironment(model); second.reset(graph, WorldRandomness(seed=7))
        direct = [second.step(ego)["background_states"] for ego in second._constant_velocity_ego_extrapolation(states[0], 5, model.cfg.response_interval_s)]
        np.testing.assert_allclose(wrapped["background_states"], np.concatenate(direct), atol=1e-6)

    def test_snapshot_restore_preserves_latent_rng_history_and_trace(self):
        model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(hidden_dim=24, num_latent_states=4))
        states = np.asarray([[0, 0, 20, 0, 0, 0], [25, 0, 19, 0, 0, 0], [-20, 0, 21, 0, 0, 0]], np.float32)
        builder = DynamicTrafficGraphBuilder(); lanes, lane_valid, lane_edges = builder.straight_lane_map(states, np.ones(3, bool))
        graph = builder.graph_at(timestamp=0.0, agent_ids=np.arange(3), states=states, valid=np.ones(3, bool), ego_index=0,
            primary_agent_index=1, map_polylines=lanes, map_polyline_valid=lane_valid, lane_graph_edges=lane_edges)
        source = SemiMarkovBackgroundEnvironment(model, model_checkpoint_hash="test-checkpoint")
        source.reset(graph, WorldRandomness(seed=17, event_structure={"primary": 1}, flow_base_sample=[0.25]))
        ego_path = source._constant_velocity_ego_extrapolation(states[0], 5, model.cfg.response_interval_s)
        source.step(ego_path[0]); source.step(ego_path[1]); snapshot = source.snapshot()
        expected = [source.step(ego) for ego in ego_path[2:]]
        restored = SemiMarkovBackgroundEnvironment(model); restored.restore(snapshot)
        actual = [restored.step(ego) for ego in ego_path[2:]]
        for left, right in zip(expected, actual):
            np.testing.assert_allclose(left["background_states"], right["background_states"], atol=1e-6)
            self.assertEqual(left["latent_state"], right["latent_state"])
            self.assertEqual(left["remaining_duration"], right["remaining_duration"])
        self.assertEqual(expected[-1]["trace"], actual[-1]["trace"])
        self.assertEqual(actual[-1]["trace"]["model_checkpoint_hash"], "test-checkpoint")

    def test_frozen_flow_anchor_sidecar_is_reused_by_m1_batches(self):
        """Formal M1 batches load B0; they do not recompute it during training."""
        import tempfile
        import torch

        adapter = HighDGraphAdapter(top_r_lanes=2)
        timestamp = np.arange(150, dtype=np.float32) / 25.0
        states = np.zeros((150, 7, 6), np.float32); states[..., 2] = 10.0
        states[24:50, 1:, 2] = np.linspace(10.0, 12.0, 26, dtype=np.float32)[:, None]
        states[24:50, 1:, 4] = np.linspace(-1.0, 1.0, 26, dtype=np.float32)[:, None]
        lanes = np.zeros((2, 3, 6), np.float32); lanes[..., 2] = 1.0
        sequence = adapter.adapt(
            sequence_id="anchor", recording_id="anchor", ego_id="0", timestamps=timestamp,
            agent_states=states, agent_valid=np.ones((150, 7), bool), primary_agent_index=0,
            split="train", map_override=(lanes, np.ones((2, 3), bool), np.asarray([[0, 1, 4]], np.int64)),
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
            batch = dict(zip(loader.field_names, next(iter(loader))))
            model = SemiMarkovRelationalWorldModel(SemiMarkovWorldModelConfig(hidden_dim=24, num_latent_states=4, variant="m1"))
            model.set_frozen_flow_schema(schema)
            raw, standardized, valid = model._batch_behavior_anchor(batch)
            torch.testing.assert_close(raw, batch["behavior_anchor_raw"])
            torch.testing.assert_close(standardized, batch["behavior_anchor_std"])
            self.assertTrue(bool(valid.all()))

    def test_highd_recording_lane_metadata_is_localized(self):
        polylines, valid, edges = HighDGraphAdapter.map_from_recording_metadata(
            {"upperLaneMarkings": np.asarray([0.0, 3.5, 7.0]), "lowerLaneMarkings": np.asarray([10.0, 13.5, 17.0])},
            ego_global_y_m=5.25,
        )
        self.assertEqual(polylines.shape, (4, 8, 6)); self.assertTrue(valid.all()); self.assertEqual(edges.shape, (4, 3))
        self.assertAlmostEqual(float(polylines[0, :, 1].mean()), -3.5)


if __name__ == "__main__":
    unittest.main()

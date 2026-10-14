import numpy as np

from hierarchical_world_model.src.randomness import WORLD_RANDOM_BLOCKS
from hierarchical_world_model.src.shared_evaluation import (
    factual_records,
    scene_randomness,
    summarize_records,
)


def test_world_randomness_is_stable_across_batches_and_order():
    together = scene_randomness(["nat_04_51_100", "nat_49_16_200"], frames=3)
    separate = scene_randomness(["nat_49_16_200"], frames=3)
    for name in WORLD_RANDOM_BLOCKS:
        np.testing.assert_array_equal(
            getattr(together, name)[1:], getattr(separate, name)
        )
    assert not np.array_equal(together.driver_latent[0], together.driver_latent[1])


def test_persistent_driver_draws_do_not_depend_on_response_horizon():
    short = scene_randomness(["nat_04_51_100", "nat_49_16_200"], frames=1)
    full = scene_randomness(["nat_04_51_100", "nat_49_16_200"], frames=149)
    for name in ("driver_latent", "maneuver_latent", "diffusion_noise"):
        np.testing.assert_array_equal(getattr(short, name), getattr(full, name))


def test_factual_summary_preserves_failed_rows_and_masks_absent_vehicles():
    logged = np.zeros((2, 174, 7, 6), np.float32)
    valid = np.zeros((2, 174, 7), bool)
    valid[:, :, :2] = True
    valid[1, :, 2] = True
    states = logged[:, 25:].copy()
    states[0, :, 1, 0] = 1.0
    states[1, :, 1:3, 0] = 2.0
    states[:, :, 3:, 0] = 1000.0
    trace = {
        "states": states,
        "actions": np.zeros((2, 149, 6, 2)),
        "base_actions": np.zeros((2, 149, 6, 2)),
        "crashed": np.zeros((2, 149, 7), bool),
        "offroad": np.zeros((2, 149, 7), bool),
        "pairs": np.zeros((2, 149, 7, 7), bool),
        "retired": np.zeros((2, 149, 6), bool),
        "initial_offroad": np.zeros((2, 7), bool),
        "logged_offroad": np.zeros((2, 149, 7), bool),
        "boundary_clearance_m": np.ones((2, 149, 7)),
        "vehicle_width_m": np.full((2, 7), 1.8),
        "target_y": np.zeros((2, 149, 6)),
        "committed": np.zeros((2, 149, 6), bool),
    }
    trace["pairs"][0, -1, 0, 1] = True
    trace["offroad"][0, :, 3] = True  # An absent actor does not count.
    trace["boundary_clearance_m"][0, :, 3] = -1000
    trace["offroad"][1, :, 2] = True
    trace["boundary_clearance_m"][1, :, 2] = -0.02
    trace["initial_offroad"][1, 2] = True
    trace["logged_offroad"][1, :, 2] = True
    trace["retired"][0, 20:, 0] = True
    trace["target_y"][0, 30:, 0] = 3.6
    trace["committed"][0, 30:, 0] = True  # Starting is not a goal flip.
    trace["target_y"][0, 50:, 0] = 0.0
    records, errors, jerks = factual_records([17, 23], logged, valid, trace)
    report = summarize_records(records, [errors], [jerks])
    assert [row["row"] for row in records] == [17, 23]
    assert report["sequences"] == 2
    assert report["actor_frames"] == 3 * 149
    assert report["ADE_m"] == 5 / 3
    assert report["FDE_m"] == 5 / 3
    assert report["P95_m"] == 2
    assert report["collision"] == 1
    assert report["offroad"] == 1
    assert report["new_offroad"] == 0
    assert report["logged_offroad"] == 1
    assert report["introduced_offroad"] == 0
    assert report["npc_offroad"] == 1
    assert report["npc_logged_offroad"] == 1
    assert report["npc_new_offroad"] == 0
    assert report["npc_introduced_offroad"] == 0
    assert all(
        report[f"ego_{name}"] == 0
        for name in ("offroad", "new_offroad", "logged_offroad", "introduced_offroad")
    )
    assert report["retired_actors"] == 1
    assert report["retired_before_last_second"] == 1
    assert report["committed_target_changes"] == 1
    assert report["boundary_severity"]["scene"]["maximum_overhang_m"] == 0.02

    trace["offroad"][0, 50, 1] = True
    trace["boundary_clearance_m"][0, 50, 1] = -0.05
    trace["logged_offroad"][
        0, 51, 1
    ] = True  # A different frame cannot excuse this violation.
    changed, errors, jerks = factual_records([17, 23], logged, valid, trace)
    report = summarize_records(changed, [errors], [jerks])
    assert report["introduced_offroad"] == 1
    assert report["new_offroad"] == 1
    assert report["npc_introduced_offroad"] == 1
    assert report["ego_introduced_offroad"] == 0
    assert (
        report["boundary_severity"]["npc"]["introduced_overhang_at_most_0_1m_sequences"]
        == 1
    )
    assert report["boundary_severity"]["scene"]["center_outside_road_sequences"] == 0
    assert changed[0]["maximum_introduced_boundary_overhang_m"] == 0.05

    trace["offroad"][1, 60, 0] = True
    trace["boundary_clearance_m"][1, 60, 0] = -1.1
    trace["offroad"][0, 70, 0] = True
    trace["boundary_clearance_m"][0, 70, 0] = -0.03
    changed, errors, jerks = factual_records([17, 23], logged, valid, trace)
    report = summarize_records(changed, [errors], [jerks])
    assert report["introduced_offroad"] == 2
    assert report["ego_introduced_offroad"] == 2
    assert report["npc_introduced_offroad"] == 1
    # Actor groups overlap at scene level; their counts must not be added.
    assert report["offroad"] == 2
    assert report["ego_offroad"] == 2 and report["npc_offroad"] == 2
    assert (
        report["boundary_severity"]["ego"]["introduced_overhang_over_0_1m_sequences"]
        == 1
    )
    assert report["boundary_severity"]["ego"]["center_outside_road_sequences"] == 1
    assert report["boundary_severity"]["npc"]["center_outside_road_sequences"] == 0

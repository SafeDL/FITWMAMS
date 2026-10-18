"""Admission rationale, physical side interactions, and persistent maneuver intent."""

from dataclasses import replace
import torch
import pytest

from hierarchical_world_model.src.motion import motion_forecast
from hierarchical_world_model.src.stochastic_drivers.driver_policy import (
    physical_leader_observation,
    anticipated_entry_features,
)
from hierarchical_world_model.src.takeover import TakeoverConfig, TakeoverController
from hierarchical_world_model.tests.traffic_context import context, THETA


def with_history(c, previous):
    return replace(
        c,
        history=torch.stack((previous, c.current), 1),
        history_valid=c.current_valid[:, None].expand(-1, 2, -1),
    )


def test_rotated_disjoint_parallel_body_is_not_an_idm_leader():
    states = torch.tensor(
        [[[0.0, 0.0, 30.0, 0.9, 0.0, 0.0], [4.7, -1.9, 30.0, 0.9, 0.0, 0.0]]]
    )
    valid = torch.ones(1, 2, dtype=torch.bool)
    sizes = torch.tensor([4.8, 1.8]).expand(1, 2, 2)
    gap, _, _, leader = physical_leader_observation(states, valid, sizes)
    assert leader[0, 0] == -1 and gap[0, 0] == 1e6
    features = anticipated_entry_features(states, valid, sizes)
    assert features[0, 0, 0] == 1 and features[0, 0, 1] == 0
    assert features[0, 1, 2] < 0  # Signed rearward position is retained.


def test_observed_lateral_acceleration_anticipates_entry_without_waiting():
    c = context()
    states = c.current.clone()
    states[:, 0, 1], states[:, 0, 3] = 3.0, -0.1
    past = states.clone()
    past[:, 0, 3] = -0.02
    c = with_history(replace(c, current=states), past)
    cold = anticipated_entry_features(states, c.current_valid, c.vehicle_dimensions)
    trend = anticipated_entry_features(
        states,
        c.current_valid,
        c.vehicle_dimensions,
        history=c.history,
        history_valid=c.history_valid,
    )
    assert cold[0, 1, 0] == 0
    assert trend[0, 1, 0] == 1 and trend[0, 1, 1] > 0
    times = torch.tensor([0.4, 0.8, 1.2])
    prediction, _ = motion_forecast(
        states, None, None, times, history=c.history, history_valid=c.history_valid
    )
    torch.testing.assert_close(
        prediction[:, 2, :, 1] - prediction[:, 1, :, 1],
        prediction[:, 1, :, 1] - prediction[:, 0, :, 1],
    )


def acceleration_scene(plan_acceleration=0.0):
    c = context()
    states = c.current.clone()
    states[:, 0, 2] += 0.04
    past = states.clone()
    past[..., 0] -= past[..., 2] * 0.04
    past[:, 0, 2] -= 0.04
    times = torch.arange(1, 51) * 0.04
    plan = c.planned_path_xy.clone()
    plan[:, :, 0, 0] += 0.5 * plan_acceleration * times.square()
    return with_history(replace(c, current=states, planned_path_xy=plan), past)


def test_leader_acceleration_requires_remaining_plan_mismatch():
    cfg = TakeoverConfig(following_scale_s=0.01, interaction_threshold=1.0)
    policy = TakeoverController(THETA, takeover_config=cfg)
    result = policy(acceleration_scene())
    assert result.takeover_reason[0, 0] == 4
    assert result.takeover_influencer[0, 0] == 0
    assert result.takeover_score[0, 0] == 0  # No collision is needed.
    assert not TakeoverController(THETA, takeover_config=cfg)(
        acceleration_scene(1.0)
    ).takeover_started[0, 0]
    for _ in range(3):
        assert not policy(acceleration_scene()).takeover_started[0, 0]


@pytest.mark.parametrize("gap,speed", [(100.0, 20.04), (15.0, 30.0)])
def test_distant_or_fast_departing_front_vehicle_is_not_a_following_constraint(
    gap, speed
):
    c = acceleration_scene()
    states = c.current.clone()
    states[:, 0, 0], states[:, 0, 2] = gap + 4.8, speed
    previous = states.clone()
    previous[..., 0] -= states[..., 2] * 0.04
    previous[:, 0, 2] -= 0.04
    c = with_history(replace(c, current=states), previous)
    cfg = TakeoverConfig(following_scale_s=0.001, interaction_threshold=1.0)
    result = TakeoverController(THETA, takeover_config=cfg)(c)
    assert not result.takeover_started[0, 0]
    assert result.takeover_following_score_s[0, 0] == 0


def release_scene(plan_acceleration=0.0):
    c = context()
    states = c.current.clone()
    states[:, 0, 1], states[:, 0, 3] = 1.84, 0.2
    past = states.clone()
    past[..., 0] -= 0.8
    past[:, 0, 1] = 1.79
    times = torch.arange(1, 51) * 0.04
    plan = c.planned_path_xy.clone()
    plan[:, :, 0, 0] += 0.5 * plan_acceleration * times.square()
    return with_history(replace(c, current=states, planned_path_xy=plan), past)


def test_actual_corridor_release_has_a_former_leader_and_suppresses_planned_response():
    cfg = TakeoverConfig(release_scale_per_s=0.5, interaction_threshold=1.0)
    result = TakeoverController(THETA, takeover_config=cfg)(release_scene())
    assert result.takeover_reason[0, 0] == 5
    assert result.takeover_influencer[0, 0] == 0
    assert result.takeover_interaction_time_s[0, 0] == 0
    assert not TakeoverController(THETA, takeover_config=cfg)(
        release_scene(1.0)
    ).takeover_started[0, 0]
    c = release_scene()
    past = c.history.clone()
    past[:, 0, 0, 1] = 3.6  # This was an unrelated adjacent vehicle.
    assert not TakeoverController(THETA, takeover_config=cfg)(
        replace(c, history=past)
    ).takeover_started[0, 0]


def test_slow_mid_maneuver_uses_current_segment_not_final_lane_and_restores():
    c = context()
    states = c.current.clone()
    states[:, 1, 1], states[:, 1, 3] = 1.6, 0.1
    plan = c.planned_path_xy.clone()
    plan[:, :25, 0, 1] = torch.linspace(1.62, 3.6, 25)
    plan[:, 25:, 0, 1] = torch.linspace(3.6, 0.0, 25)
    c = replace(
        c,
        current=states,
        history=states[:, None],
        planned_current_xy=states[:, 1:, :2],
        planned_path_xy=plan,
    )
    policy = TakeoverController(THETA)
    policy(c)
    assert policy.plan_target_y[0, 0] == pytest.approx(3.6)
    policy._adopt_current_maneuver(
        c, torch.tensor([[True, False, False, False, False, False]]), torch.zeros(1, 6)
    )
    assert policy._autonomous_lane_target_y[0, 0] == pytest.approx(3.6)
    restored = TakeoverController(THETA)
    restored.restore_runtime(policy.snapshot_runtime())
    torch.testing.assert_close(restored.plan_progress, policy.plan_progress)
    torch.testing.assert_close(restored.plan_target_y, policy.plan_target_y)
    torch.testing.assert_close(restored.previous_actual, policy.previous_actual)


def test_missing_history_prefix_cannot_invent_a_lane_change():
    c = context()
    states = c.current.clone()
    states[:, 1, 1] = 3.6
    plan = c.planned_path_xy.clone()
    plan[:, :, 0, 1] = 3.6
    history = torch.stack((torch.zeros_like(states), states), 1)
    valid = torch.stack((torch.zeros_like(c.current_valid), c.current_valid), 1)
    c = replace(
        c,
        current=states,
        history=history,
        history_valid=valid,
        planned_path_xy=plan,
        planned_current_xy=states[:, 1:, :2],
    )
    policy = TakeoverController(THETA)
    policy(c)
    assert policy.plan_source_y[0, 0] == pytest.approx(3.6)
    assert not policy.plan_started[0, 0]


def test_lateral_error_samples_are_reflected_with_observed_motion():
    from hierarchical_world_model.src.error_trajectories import ErrorTrajectoryBank
    from hierarchical_world_model.src.plan_applicability import forecast_bundle

    c = context()
    states = c.current.clone()
    states[:, 0, 1], states[:, 0, 3] = 2.0, 0.4
    c = replace(c, current=states, history=states[:, None])
    mirrored = states.clone()
    mirrored[..., 1] *= -1
    mirrored[..., 3] *= -1
    plan = c.planned_path_xy.clone()
    plan[..., 1] *= -1
    origin = c.planned_current_xy.clone()
    origin[..., 1] *= -1
    other = replace(
        c,
        current=mirrored,
        history=mirrored[:, None],
        planned_path_xy=plan,
        planned_current_xy=origin,
    )
    bank = ErrorTrajectoryBank.nominal()
    bank.samples[5, ..., 1] = 0.2
    bank.samples[5, ..., 2] = 0.01
    owned = torch.zeros(1, 6, dtype=torch.bool)
    a, b = forecast_bundle(c, owned, bank), forecast_bundle(other, owned, bank)
    torch.testing.assert_close(a[0][..., 1], -b[0][..., 1])
    torch.testing.assert_close(a[2][:, 0, ..., 1:], -b[2][:, 0, ..., 1:])


def test_natural_thresholds_keep_conflict_and_noncollision_budgets_separate(
    tmp_path, monkeypatch
):
    import numpy as np
    import yaml
    from hierarchical_world_model.scripts import calibrate_bridge as calibration

    monkeypatch.setattr(calibration, "ROOT", tmp_path)
    monkeypatch.setattr(calibration, "OUTPUT", tmp_path)
    directory = tmp_path / "hierarchical_world_model"
    directory.mkdir()
    (directory / "config.yaml").write_text(yaml.safe_dump({"takeover": {}}))
    train = dict(
        rows=np.arange(20),
        valid=np.ones((20, 1), bool),
        interaction=np.ones((20, 1, 2), np.float32),
    )
    val = dict(
        rows=np.arange(1000),
        conflict=np.zeros((1000, 1), np.float32),
        interaction=np.zeros((1000, 1, 2), np.float32),
    )
    val["conflict"][:4] = 1
    val["interaction"][:, 0, 0] = np.arange(1000)
    result = calibration.select(train, val)
    assert result["conflict_scenes"] == 4
    assert result["noncollision_scenes"] == 10
    assert result["combined_scenes"] == 14
    assert result["noncollision_scene_rate"] == 0.01

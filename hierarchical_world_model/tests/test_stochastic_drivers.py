from __future__ import annotations

import numpy as np
import pytest

from hierarchical_world_model.src.stochastic_drivers import (
    LongitudinalObservation,
    LongitudinalPrefix,
    create_driver_session,
    verify_driver_assets,
)


def test_project_state_conversion_and_five_frame_action_hold() -> None:
    observation = LongitudinalObservation.from_project_states(
        np.asarray([0., 0., 18., 0., 0., 0.]),
        np.asarray([30., 0., 17.5, 0., 0., 0.]),
        length_sum_m=4.8,
    )
    assert observation == LongitudinalObservation(25.2, 18., 17.5, 0.)
    session = create_driver_session("ma_idm", seed=7)
    session.reset(observation, seed=7)
    commands = [session.step(observation) for _ in range(6)]
    assert [command.updated for command in commands] == [True] + [False] * 4 + [True]
    assert len({command.acceleration_mps2 for command in commands[:5]}) == 1


def test_snapshot_restores_stochastic_decision_exactly() -> None:
    observation = LongitudinalObservation(25., 18., 17.5)
    session = create_driver_session("dynamic_ar5", seed=11)
    session.reset(observation, seed=11)
    for _ in range(5):
        session.step(observation)
    snapshot = session.snapshot()
    first = session.step(observation)
    session.restore(snapshot)
    repeated = session.step(observation)
    assert first == repeated and first.updated


def test_registry_preserves_scientific_acceptance_boundaries() -> None:
    inventory = verify_driver_assets()
    assert inventory["b_idm"]["artifact_present"]
    assert inventory["ma_idm"]["artifact_present"]
    assert inventory["dynamic_ar5"]["artifact_present"]
    assert inventory["multi_regime"]["evidence_status"] == "rejected_not_well_calibrated"
    with pytest.raises(RuntimeError, match="not deployment-approved"):
        create_driver_session("multi_regime", seed=3)


def test_unaccepted_multi_regime_remains_executable_for_research() -> None:
    observation = LongitudinalObservation(25., 18., 17.5)
    session = create_driver_session(
        "multi_regime", seed=3, style_id=1, allow_unaccepted=True
    )
    session.reset(observation, seed=3)
    assert np.isfinite(session.step(observation).acceleration_mps2)


def test_ma_idm_restores_completed_prefix_memory_and_records_provenance() -> None:
    frames = 126
    speed = np.linspace(17.0, 18.0, frames)
    prefix = LongitudinalPrefix.from_native_series(
        gap_m=np.linspace(28.0, 24.0, frames),
        ego_speed_mps=speed,
        leader_speed_mps=np.full(frames, 17.5),
    )
    assert len(prefix.time_s) == 25
    # The final action [120,125] is complete at the origin and no sample after
    # native frame 125 is required.
    assert prefix.time_s[-1] == pytest.approx(4.8)
    session = create_driver_session(
        "ma_idm", seed=13, prefix_observations=prefix,
    )
    session.reset(LongitudinalObservation(24.0, 18.0, 17.5), seed=13)
    assert len(session.driver._gp.times) == 25
    assert session.driver._gp.times[-1] == pytest.approx(-0.2)
    assert session.metadata["parameter_mode"] == "population"
    assert session.metadata["prefix_memory_initialized"] is True


def test_ma_idm_prefix_personalization_is_explicit_and_reproducible() -> None:
    time = np.arange(8, dtype=float) * 0.2
    prefix = LongitudinalPrefix(
        time,
        np.linspace(30.0, 24.0, 8),
        np.linspace(17.0, 18.0, 8),
        np.full(8, 17.5),
        np.full(8, 0.4),
    )
    left = create_driver_session(
        "ma_idm", seed=19, prefix_observations=prefix,
        personalize_from_prefix=True, personalization_candidates=16,
    )
    right = create_driver_session(
        "ma_idm", seed=19, prefix_observations=prefix,
        personalize_from_prefix=True, personalization_candidates=16,
    )
    np.testing.assert_allclose(left.driver.theta, right.driver.theta)
    assert left.metadata["parameter_mode"] == "prefix_personalized"
    assert left.metadata["posterior_artifact"].endswith("ma_idm_all_251.npz")

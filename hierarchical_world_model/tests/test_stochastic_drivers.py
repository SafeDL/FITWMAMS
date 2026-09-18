from __future__ import annotations

import numpy as np
import pytest

from hierarchical_world_model.src.stochastic_drivers import (
    LongitudinalObservation,
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

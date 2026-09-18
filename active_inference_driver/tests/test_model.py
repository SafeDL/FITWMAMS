from __future__ import annotations

import numpy as np

from active_inference_driver.config import ActiveInferenceConfig
from active_inference_driver.model import ActiveInferenceDriver, surprise_step


def test_surprise_accumulation_is_deterministic_and_has_no_decay() -> None:
    cfg = ActiveInferenceConfig()
    first, surprise = surprise_step(.2, -100., cfg)
    second, _ = surprise_step(first, 0., cfg)
    assert surprise == 100.
    assert second == first


def test_driver_keeps_a_5_tick_decision_contract() -> None:
    cfg = ActiveInferenceConfig().reduced()
    assert cfg.native_ticks_per_decision == 5
    driver = ActiveInferenceDriver(cfg)
    driver.reset(gap_m=30., ego_speed_mps=20., leader_speed_mps=20., seed=3)
    first = driver.decide(gap_m=30., ego_speed_mps=20., leader_speed_mps=20.)
    assert np.isfinite(first.action)
    assert len(driver.state.policy) == cfg.horizon


def test_pedal_change_inserts_coast_action() -> None:
    cfg = ActiveInferenceConfig().reduced()
    driver = ActiveInferenceDriver(cfg)
    driver.reset(gap_m=20., ego_speed_mps=20., leader_speed_mps=18., seed=4)
    constrained = driver._constrain_plan(np.array([-4., -4.]), 1.)
    assert constrained[0] == cfg.coast_acceleration_mps2


def test_pedal_ablation_allows_direct_crossing() -> None:
    cfg = ActiveInferenceConfig(enforce_pedal_constraint=False).reduced()
    driver = ActiveInferenceDriver(cfg)
    driver.reset(gap_m=20., ego_speed_mps=20., leader_speed_mps=18., seed=5)
    constrained = driver._constrain_plan(np.array([-4.]), 1.)
    assert constrained[0] < cfg.coast_acceleration_mps2


def test_full_replan_only_occurs_after_evidence_threshold() -> None:
    """Equation (14) extends a policy until deterministic E reaches one."""
    cfg = ActiveInferenceConfig().reduced()
    driver = ActiveInferenceDriver(cfg)
    driver.reset(gap_m=30., ego_speed_mps=20., leader_speed_mps=20., seed=6)
    state = driver.state
    assert state is not None
    state.decision_count = 1
    state.evidence = .3
    calls: list[bool] = []
    driver._update_belief = lambda *_: None  # type: ignore[method-assign]
    driver._target_predictions = lambda: (np.zeros((cfg.horizon, cfg.particles)),
                                          np.zeros((cfg.horizon, cfg.particles)))  # type: ignore[method-assign]

    def plan(_: float, *, full_replan: bool) -> np.ndarray:
        calls.append(full_replan)
        return state.policy

    driver._plan = plan  # type: ignore[method-assign]
    driver._pragmatic_value = lambda *_: np.array([0.])  # type: ignore[method-assign]
    trace = driver.decide(gap_m=30., ego_speed_mps=20., leader_speed_mps=20.)
    assert not trace.replanned
    assert calls[-1] is False
    assert state.evidence == .3

    state.evidence = .999
    driver._pragmatic_value = lambda *_: np.array([-1_000_000.])  # type: ignore[method-assign]
    trace = driver.decide(gap_m=30., ego_speed_mps=20., leader_speed_mps=20.)
    assert trace.replanned
    assert calls[-1] is True
    assert state.evidence == 0.

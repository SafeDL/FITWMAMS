from __future__ import annotations

import numpy as np

from process_highD.src.safety_envelope_risk import (
    SEI_COMPONENT_NAMES,
    SafetyEnvelopeRiskOptions,
    pairwise_safety_envelope_intrusion,
)


def _pair(*, other_x: float, other_y: float, other_vx: float, other_vy: float):
    zero = np.zeros(1, dtype=np.float32)
    return pairwise_safety_envelope_intrusion(
        ego_x=zero,
        ego_y=zero,
        ego_vx=np.asarray([25.0], dtype=np.float32),
        ego_vy=zero,
        ego_ax=zero,
        ego_ay=zero,
        other_x=np.asarray([other_x], dtype=np.float32),
        other_y=np.asarray([other_y], dtype=np.float32),
        other_vx=np.asarray([other_vx], dtype=np.float32),
        other_vy=np.asarray([other_vy], dtype=np.float32),
        other_ax=zero,
        other_ay=zero,
        ego_length=4.8,
        ego_width=1.8,
        other_length=4.8,
        other_width=1.8,
        valid=np.ones(1, dtype=bool),
        options=SafetyEnvelopeRiskOptions(),
    )


def test_longitudinal_closing_creates_longitudinal_risk() -> None:
    score, components = _pair(
        other_x=18.0, other_y=0.0, other_vx=10.0, other_vy=0.0
    )
    index = SEI_COMPONENT_NAMES.index("sei_longitudinal_deficit")
    assert score[0] > 0.0
    assert components[0, index] > 0.0


def test_lateral_closing_creates_lateral_risk() -> None:
    score, components = _pair(
        other_x=0.0, other_y=4.0, other_vx=25.0, other_vy=-2.0
    )
    index = SEI_COMPONENT_NAMES.index("sei_lateral_deficit")
    assert score[0] > 0.0
    assert components[0, index] > 0.0


def test_larger_vehicle_dimensions_raise_intrusion() -> None:
    zero = np.zeros(1, dtype=np.float32)
    common = {
        "ego_x": zero,
        "ego_y": zero,
        "ego_vx": np.asarray([25.0], dtype=np.float32),
        "ego_vy": zero,
        "ego_ax": zero,
        "ego_ay": zero,
        "other_x": np.asarray([25.0], dtype=np.float32),
        "other_y": zero,
        "other_vx": np.asarray([25.0], dtype=np.float32),
        "other_vy": zero,
        "other_ax": zero,
        "other_ay": zero,
        "ego_length": 4.8,
        "ego_width": 1.8,
        "other_width": 1.8,
        "valid": np.ones(1, dtype=bool),
        "options": SafetyEnvelopeRiskOptions(),
    }
    car, _ = pairwise_safety_envelope_intrusion(
        **common, other_length=4.8
    )
    truck, _ = pairwise_safety_envelope_intrusion(
        **common, other_length=16.0
    )
    assert truck[0] > car[0]

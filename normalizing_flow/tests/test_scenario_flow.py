from __future__ import annotations

import numpy as np
import pytest
import torch

from normalizing_flow.src.constraints import derived_modes
from normalizing_flow.src.features import build_feature_schema, feature_index
from normalizing_flow.src.scenario import DirectScenarioFlow, project_constraint
from normalizing_flow.src import scenario as scenario_module


class _GaussianFlow(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def log_prob(self, values, context=None):
        del context
        return -0.5 * values.square().sum(dim=1) + self.bias


def test_joint_log_prob_is_sum_of_three_factors() -> None:
    model = DirectScenarioFlow(
        c0_flow=_GaussianFlow(),
        constraint_flow=_GaussianFlow(),
        mask_log_prob=torch.full((64,), -np.log(64), dtype=torch.float32),
        schema={},
    )
    terms = model.log_prob_tensors(
        c0_normalized=torch.zeros(2, 40),
        slot_mask=torch.tensor(
            [[1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0]], dtype=torch.bool
        ),
        constraint_normalized=torch.zeros(2, 6, 12),
    )
    expected = sum(
        terms[name] for name in ("mask_log_prob", "c0_log_prob", "k_log_prob")
    )
    assert torch.allclose(terms["joint_log_prob"], expected)


def test_future_constraint_cannot_change_c0_or_mask_factors() -> None:
    model = DirectScenarioFlow(
        c0_flow=_GaussianFlow(),
        constraint_flow=_GaussianFlow(),
        mask_log_prob=torch.full((64,), -np.log(64), dtype=torch.float32),
        schema={},
    )
    arguments = {
        "c0_normalized": torch.randn(3, 40),
        "slot_mask": torch.ones(3, 6, dtype=torch.bool),
    }
    first = model.log_prob_tensors(
        **arguments, constraint_normalized=torch.zeros(3, 6, 12)
    )
    changed = model.log_prob_tensors(
        **arguments, constraint_normalized=torch.ones(3, 6, 12)
    )
    torch.testing.assert_close(first["mask_log_prob"], changed["mask_log_prob"])
    torch.testing.assert_close(first["c0_log_prob"], changed["c0_log_prob"])
    assert not torch.equal(first["k_log_prob"], changed["k_log_prob"])


def test_constraint_projection_enforces_speed_and_position_contract() -> None:
    values = np.zeros((1, 6, 12), np.float32)
    values[..., (0, 4, 8)] = (30.0, 10.0, -5.0)
    values[..., (2, 6, 10)] = (-30.0, 80.0, 1.0)
    slots = np.ones((1, 6), bool)
    c0 = np.zeros((1, 40), np.float32)
    c0[:, 0] = 20.0
    projected = project_constraint(values, slots, c0)
    assert np.all(np.diff(projected[..., (0, 4, 8)], axis=-1) >= 0.0)
    speed = 20.0 + projected[..., (2, 6, 10)]
    assert np.all((speed >= 0.0) & (speed <= 70.0))


def test_modes_are_derived_only_from_k() -> None:
    k = np.zeros((1, 6, 12), np.float32)
    k[0, 0, 10] = -1.0
    k[0, 0, 9] = 3.6
    mask = np.ones((1, 6), bool)
    modes = derived_modes(k, mask)
    assert tuple(modes[0, 0]) == (0, 1)


def test_conditional_scenario_preserves_logged_c0_without_projection() -> None:
    feature_names = list(build_feature_schema().feature_names)
    schema = {
        "feature_names": feature_names,
        "normalization": {
            "mean": [0.0] * 40,
            "std": [1.0] * 40,
        },
        "model_feature_transforms": [],
        "long_horizon_constraint": {
            "normalization": {
                "mean": [0.0] * 12,
                "std": [1.0] * 12,
            }
        },
    }
    model = DirectScenarioFlow(
        c0_flow=_GaussianFlow(),
        constraint_flow=_GaussianFlow(),
        mask_log_prob=torch.full((64,), -np.log(64), dtype=torch.float32),
        schema=schema,
    )
    slots = torch.tensor([[True, False, False, False, False, False]])
    logged = np.zeros((1, 40), np.float32)
    rel_x = feature_index("same_front", "rel_x_m")
    logged[0, rel_x] = 0.125
    scenario = model._scenario_from_normalized(
        torch.from_numpy(logged),
        slots,
        torch.zeros(1, 6, 12),
        c0_override=logged,
    )
    np.testing.assert_array_equal(scenario.c0, logged)
    generated = model._scenario_from_normalized(
        torch.from_numpy(logged), slots, torch.zeros(1, 6, 12)
    )
    assert generated.c0[0, rel_x] == np.float32(4.81)


def test_initial_condition_boundary_never_evaluates_future_k_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = {
        "feature_names": list(build_feature_schema().feature_names),
        "normalization": {"mean": [0.0] * 40, "std": [1.0] * 40},
        "model_feature_transforms": [],
    }
    c0_flow = _GaussianFlow()
    constraint_flow = _GaussianFlow()
    mask_log_prob = torch.full((64,), float("-inf"))
    mask_log_prob[1] = 0.0
    model = DirectScenarioFlow(
        c0_flow=c0_flow,
        constraint_flow=constraint_flow,
        mask_log_prob=mask_log_prob,
        schema=schema,
    )

    def fake_sample(flow, context, generator=None, *, base_noise=None, chunk_size=128):
        del context, generator, chunk_size
        if flow is constraint_flow:
            raise AssertionError("future-knot flow must not be evaluated")
        return torch.as_tensor(base_noise, dtype=torch.float32)

    monkeypatch.setattr(scenario_module, "_sample_flow", fake_sample)
    c0, slots = model.sample_initial_conditions_from_base_randomness(
        np.asarray([0.5]), np.zeros((1, 40), np.float32)
    )
    assert c0.shape == (1, 40)
    assert slots.tolist() == [[True, False, False, False, False, False]]

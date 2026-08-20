from __future__ import annotations

import numpy as np
import torch

from normalizing_flow.src.constraints import derived_modes
from normalizing_flow.src.scenario import DirectScenarioFlow, project_constraint


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

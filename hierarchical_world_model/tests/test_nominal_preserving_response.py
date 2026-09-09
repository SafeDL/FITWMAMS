"""P1 numerical contracts for the isolated nominal-preserving QP primitive."""
from __future__ import annotations

import numpy as np
import torch

from hierarchical_world_model.src.decision_cost import DecisionCostInputs, affine_cost_residuals
from hierarchical_world_model.src.nominal_preserving_controller import solve_calibrated_qp, solve_hinge_calibrated_qp
from world_model.src.core.evaluation_scope import scoped_slot_mask


def _problem(dtype=torch.float64):
    torch.manual_seed(20260907)
    batch, horizon = 3, 25
    # A zero nominal and previous action are strictly feasible for all bounds.
    nominal = torch.zeros((batch, horizon), dtype=dtype)
    previous = torch.zeros(batch, dtype=dtype)
    matrix = torch.randn((batch, horizon, horizon), dtype=dtype)
    hessian = 0.01 * torch.bmm(matrix.transpose(1, 2), matrix)
    linear = torch.randn((batch, horizon), dtype=dtype) * 0.1
    return nominal, hessian, linear, previous


def test_nominal_identity_float64_and_parameter_gradient() -> None:
    nominal, hessian, linear, previous = _problem()
    hessian.requires_grad_()
    action, diagnostic = solve_calibrated_qp(nominal, hessian, hessian, linear, linear, previous)
    assert float(diagnostic.nominal_error.max()) <= 1.0e-8
    assert float(diagnostic.feasible_residual.max()) <= 1.0e-8
    gradient = torch.autograd.grad(action.square().sum(), hessian)[0]
    assert float(gradient.abs().max()) <= 1.0e-8


def test_nominal_identity_float32_and_non_nominal_response() -> None:
    nominal, hessian, linear, previous = _problem(torch.float32)
    action, diagnostic = solve_calibrated_qp(nominal, hessian, hessian, linear, linear, previous)
    assert float(diagnostic.nominal_error.max()) <= 1.0e-5
    assert float(diagnostic.feasible_residual.max()) <= 1.0e-5
    altered_linear = linear - 2.0
    changed, changed_diagnostic = solve_calibrated_qp(nominal, hessian, hessian, altered_linear, linear, previous)
    assert float(changed.abs().max()) > 1.0e-4
    assert float(changed_diagnostic.feasible_residual.max()) <= 1.0e-5


def test_cost_bases_are_affine_and_causal() -> None:
    values = DecisionCostInputs(
        gap_m=torch.tensor([12.0]), speed_mps=torch.tensor([20.0]),
        leader_speed_mps=torch.tensor([18.0]), leader_acceleration_mps2=torch.tensor([-1.0]),
        reference_speed_mps=torch.tensor([22.0]),
    )
    matrix, intercept = affine_cost_residuals(values, 25)
    assert matrix.shape == (1, 25, 3, 25)
    assert intercept.shape == (1, 25, 3)
    # Affinity check avoids hidden dependence on a candidate future action.
    left, right = torch.randn(1, 25), torch.randn(1, 25)
    residual = lambda action: torch.einsum("btch,bh->btc", matrix, action) + intercept
    assert torch.allclose(residual(0.25 * left + 0.75 * right), 0.25 * residual(left) + 0.75 * residual(right), atol=1e-6)


def test_hinge_qp_keeps_identical_nominal_history() -> None:
    nominal = torch.zeros((1, 25), dtype=torch.float64)
    previous = torch.zeros(1, dtype=torch.float64)
    history = DecisionCostInputs(*(torch.tensor([value], dtype=torch.float64) for value in (12., 20., 18., -1., 22.)))
    weights = torch.ones((1, 3), dtype=torch.float64)
    action, diagnostics = solve_hinge_calibrated_qp(nominal, history, history, weights, weights, previous)
    assert float(diagnostics.nominal_error.max()) <= 1.0e-8
    assert float(diagnostics.feasible_residual.max()) <= 1.0e-8


def test_research_scope_keeps_same_rear_without_changing_the_default() -> None:
    slots = np.ones((2, 6), dtype=bool)
    assert not scoped_slot_mask(slots)[0, 1]
    assert scoped_slot_mask(slots, excluded_slots=())[0, 1]

"""Differentiable constrained QP primitive for the research-only response layer.

This module is intentionally independent of the released controller factory.
It supplies the mathematical P1 contract before any integration with the
frozen world model: the nominal-gradient calibration makes a feasible nominal
action the unique optimizer when the actual and nominal quadratic costs agree.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .decision_cost import DecisionCostInputs, affine_cost_residuals


@dataclass(frozen=True)
class QPDiagnostics:
    feasible_residual: torch.Tensor
    nominal_error: torch.Tensor
    iterations: int


def physical_inequalities(
    nominal: torch.Tensor,
    previous_action: torch.Tensor,
    *,
    dt_s: float = 0.04,
    acceleration_min: float = -8.0,
    acceleration_max: float = 4.0,
    jerk_limit: float = 12.0,
    initial_speed: torch.Tensor | None = None,
    speed_min: float = 0.0,
    speed_max: float = 50.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return batched ``G, h`` for box, jerk and optional speed constraints.

    The caller must establish that ``nominal`` is feasible; this function does
    not project it, which avoids silently changing the nominal reference.
    """
    if nominal.ndim != 2 or previous_action.shape != nominal.shape[:1]:
        raise ValueError("nominal must be [batch,horizon], previous_action [batch]")
    batch, horizon = nominal.shape
    eye = torch.eye(horizon, dtype=nominal.dtype, device=nominal.device)
    rows = [eye, -eye]
    rhs = [
        nominal.new_full((batch, horizon), float(acceleration_max)),
        nominal.new_full((batch, horizon), -float(acceleration_min)),
    ]
    difference = torch.zeros((horizon, horizon), dtype=nominal.dtype, device=nominal.device)
    difference[0, 0] = 1.0
    if horizon > 1:
        index = torch.arange(1, horizon, device=nominal.device)
        difference[index, index] = 1.0
        difference[index, index - 1] = -1.0
    jerk_step = float(jerk_limit) * float(dt_s)
    rows.extend((difference, -difference))
    first = nominal.new_zeros((batch, horizon))
    first[:, 0] = previous_action
    rhs.extend((first + jerk_step, -first + jerk_step))
    if initial_speed is not None:
        if initial_speed.shape != (batch,):
            raise ValueError("initial_speed must be [batch]")
        integration = torch.tril(torch.ones((horizon, horizon), dtype=nominal.dtype, device=nominal.device)) * float(dt_s)
        rows.extend((integration, -integration))
        rhs.extend((
            nominal.new_full((batch, horizon), float(speed_max)) - initial_speed[:, None],
            initial_speed[:, None] - float(speed_min),
        ))
    return torch.cat(rows, dim=0).expand(batch, -1, -1), torch.cat(rhs, dim=1)


def solve_calibrated_qp(
    nominal: torch.Tensor,
    actual_q: torch.Tensor,
    nominal_q: torch.Tensor,
    linear_actual: torch.Tensor,
    linear_nominal: torch.Tensor,
    previous_action: torch.Tensor,
    *,
    initial_speed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, QPDiagnostics]:
    """Solve the calibrated strongly-convex QP with qpth.

    ``actual_q`` and ``nominal_q`` are PSD cost Hessians.  The objective is
    ``.5||a-bar||² + V_actual(a)-V_nominal(bar)-grad V_nominal(bar)·(a-bar)``.
    Constants are omitted because they do not affect the solution.
    """
    try:
        from qpth.qp import QPFunction
    except ImportError as error:  # pragma: no cover - preflight catches this
        raise RuntimeError("qpth is required; see requirements-nominal-response.txt") from error
    if nominal.ndim != 2:
        raise ValueError("all action tensors must have shape [batch,horizon]")
    # qpth's primal-dual iterations do not meet the required 1e-5 nominal
    # tolerance in float32 on this Torch version.  Solve in float64 and cast
    # only the executed action back; autograd retains the cast path.
    original_dtype = nominal.dtype
    if nominal.dtype == torch.float32:
        nominal = nominal.double()
        actual_q = actual_q.double()
        nominal_q = nominal_q.double()
        linear_actual = linear_actual.double()
        linear_nominal = linear_nominal.double()
        previous_action = previous_action.double()
        initial_speed = None if initial_speed is None else initial_speed.double()
    batch, horizon = nominal.shape
    expected_matrix = (batch, horizon, horizon)
    if actual_q.shape != expected_matrix or nominal_q.shape != expected_matrix:
        raise ValueError("quadratic costs must be [batch,horizon,horizon]")
    if linear_actual.shape != nominal.shape or linear_nominal.shape != nominal.shape:
        raise ValueError("linear costs must be [batch,horizon]")
    identity = torch.eye(horizon, dtype=nominal.dtype, device=nominal.device).expand(batch, -1, -1)
    q = identity + actual_q
    nominal_gradient = torch.bmm(nominal_q, nominal[..., None]).squeeze(-1) + linear_nominal
    # qpth uses .5*x'Q*x + p'x.  Expand only nonconstant terms.
    p = -nominal + linear_actual - nominal_gradient
    g, h = physical_inequalities(nominal, previous_action, initial_speed=initial_speed)
    # qpth 0.0.16 requires explicit empty equality tensors (rather than
    # ``None``) even when the QP has no equality constraints.
    empty_a = nominal.new_empty((batch, 0, horizon))
    empty_b = nominal.new_empty((batch, 0))
    solution = QPFunction(verbose=-1, eps=1.0e-12, maxIter=100, notImprovedLim=20)(q, p, g, h, empty_a, empty_b)
    residual = (torch.bmm(g, solution[..., None]).squeeze(-1) - h).clamp_min(0).amax(dim=1)
    executed = solution.to(original_dtype)
    return executed, QPDiagnostics(
        feasible_residual=residual.to(original_dtype),
        nominal_error=(executed - nominal.to(original_dtype)).abs().amax(dim=1),
        iterations=100,
    )


def solve_hinge_calibrated_qp(
    nominal: torch.Tensor, actual: DecisionCostInputs, nominal_history: DecisionCostInputs,
    actual_weights: torch.Tensor, nominal_weights: torch.Tensor, previous_action: torch.Tensor,
) -> tuple[torch.Tensor, QPDiagnostics]:
    """Solve the specified three-basis positive-part-square QP exactly.

    The auxiliary variables encode each hinge.  ``nominal_history`` is
    separate so the first-order calibration is evaluated at ``bar H`` rather
    than incorrectly detached or reused from the factual history.
    """
    from qpth.qp import QPFunction
    original_dtype = nominal.dtype
    if nominal.dtype == torch.float32:
        nominal, previous_action = nominal.double(), previous_action.double()
        actual_weights, nominal_weights = actual_weights.double(), nominal_weights.double()
        actual = DecisionCostInputs(*(item.double() for item in actual.__dict__.values()))
        nominal_history = DecisionCostInputs(*(item.double() for item in nominal_history.__dict__.values()))
    batch, horizon = nominal.shape
    if actual_weights.shape != (batch, 3) or nominal_weights.shape != (batch, 3):
        raise ValueError("cost weights must be [batch,3]")
    if (actual_weights <= 0).any() or (nominal_weights <= 0).any():
        raise ValueError("cost weights must be strictly positive")
    ma, ba = affine_cost_residuals(actual, horizon)
    mn, bn = affine_cost_residuals(nominal_history, horizon)
    ma, mn = ma.permute(0, 2, 1, 3).reshape(batch, 3 * horizon, horizon), mn.permute(0, 2, 1, 3).reshape(batch, 3 * horizon, horizon)
    ba, bn = ba.permute(0, 2, 1).reshape(batch, 3 * horizon), bn.permute(0, 2, 1).reshape(batch, 3 * horizon)
    wa = actual_weights[:, :, None].expand(-1, -1, horizon).reshape(batch, 3 * horizon)
    wn = nominal_weights[:, :, None].expand(-1, -1, horizon).reshape(batch, 3 * horizon)
    nominal_residual = torch.bmm(mn, nominal[..., None]).squeeze(-1) + bn
    nominal_gradient = torch.bmm((2.0 * wn * nominal_residual.clamp_min(0))[..., None, :], mn).squeeze(1)
    dimensions = 4 * horizon
    q = nominal.new_zeros((batch, dimensions, dimensions))
    q[:, :horizon, :horizon] = torch.eye(horizon, dtype=nominal.dtype, device=nominal.device)
    q[:, horizon:, horizon:] = torch.diag_embed(2.0 * wa)
    p = nominal.new_zeros((batch, dimensions))
    p[:, :horizon] = -nominal - nominal_gradient
    g_phys, h_phys = physical_inequalities(nominal, previous_action)
    padded = nominal.new_zeros((batch, g_phys.shape[1], dimensions)); padded[:, :, :horizon] = g_phys
    # M a - s <= -b and -s <= 0.
    hinge = nominal.new_zeros((batch, 3 * horizon, dimensions)); hinge[:, :, :horizon] = ma; hinge[:, :, horizon:] = -torch.eye(3 * horizon, dtype=nominal.dtype, device=nominal.device)
    nonnegative = nominal.new_zeros((batch, 3 * horizon, dimensions)); nonnegative[:, :, horizon:] = -torch.eye(3 * horizon, dtype=nominal.dtype, device=nominal.device)
    g = torch.cat((padded, hinge, nonnegative), dim=1)
    h = torch.cat((h_phys, -ba, nominal.new_zeros((batch, 3 * horizon))), dim=1)
    empty_a, empty_b = nominal.new_empty((batch, 0, dimensions)), nominal.new_empty((batch, 0))
    solution = QPFunction(verbose=-1, eps=1e-12, maxIter=100, notImprovedLim=20)(q, p, g, h, empty_a, empty_b)
    action = solution[:, :horizon].to(original_dtype)
    residual = (torch.bmm(g, solution[..., None]).squeeze(-1) - h).clamp_min(0).amax(1).to(original_dtype)
    return action, QPDiagnostics(residual, (action - nominal.to(original_dtype)).abs().amax(1), 100)

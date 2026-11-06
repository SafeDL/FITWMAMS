"""Project the actual optimizer displacement onto two retention halfspaces."""

import itertools

import torch


def project_displacement(displacement, gradients):
    original = displacement.double()
    normals = []
    for gradient in gradients:
        gradient = gradient.double()
        norm = gradient.norm()
        if norm > 1e-12:
            normals.append(gradient / norm)
    if not normals:
        return original, {"active_constraints": 0, "normalized_dots": []}
    matrix = torch.stack(normals)
    tolerance = 1e-10 * max(1.0, float(original.norm()))
    candidates = []
    for size in range(len(normals) + 1):
        for active in itertools.combinations(range(len(normals)), size):
            value = original
            if active:
                rows = matrix[list(active)]
                multipliers = torch.linalg.pinv(rows @ rows.T) @ (rows @ original)
                if (multipliers < -tolerance).any():
                    continue
                value = original - rows.T @ multipliers
            if (matrix @ value <= tolerance).all():
                candidates.append((float((value - original).square().sum()), value))
    if not candidates:
        raise ArithmeticError("No feasible retention displacement found")
    projected = min(candidates, key=lambda item: item[0])[1]
    return projected, {
        "active_constraints": len(normals),
        "normalized_dots": (matrix @ projected).tolist(),
        "projection_norm": float((projected - original).norm()),
    }


def flatten_gradients(parameters):
    return torch.cat(
        [
            (torch.zeros_like(p) if p.grad is None else p.grad).detach().reshape(-1)
            for p in parameters
        ]
    )


@torch.no_grad()
def constrain_step(parameters, before, gradients):
    after = torch.cat([p.detach().reshape(-1) for p in parameters])
    displacement, report = project_displacement(
        after.double() - before.double(), gradients
    )
    normals = [
        gradient.double() / gradient.double().norm()
        for gradient in gradients
        if gradient.double().norm() > 1e-12
    ]
    # A boundary projection can round to the wrong side of a halfspace when
    # stored in FP32. Move a few rounding units into the feasible cone instead
    # of repeatedly projecting onto the same unrepresentable boundary.
    inward = None
    if normals:
        matrix = torch.stack(normals)
        direction = -matrix.sum(0)
        decrease = -(matrix @ direction)
        if decrease.min() > 1e-8:
            inward = direction / decrease.min()
    boundary = displacement.clone()
    margin = 0.0
    for attempt in range(16):
        values = (before.double() + displacement).to(before.dtype)
        actual = values.double() - before.double()
        dots = [
            (
                float(g.double().dot(actual) / g.double().norm())
                if g.double().norm() > 1e-12
                else 0.0
            )
            for g in gradients
        ]
        if max(dots, default=0.0) <= 1e-9:
            break
        if inward is not None:
            margin = max(2 * margin, 2 * max(dots), 2e-9)
            displacement = boundary + margin * inward
        else:
            # Opposite normals have no strict interior. Keep their common
            # boundary and reduce the update if it cannot be represented.
            displacement, _ = project_displacement(actual, gradients)
            if attempt >= 3:
                displacement *= 0.5
    else:
        values, dots = before, [0.0 for _ in gradients]
        report["rejected_for_rounding"] = True
    offset = 0
    for parameter in parameters:
        parameter.copy_(
            values[offset : offset + parameter.numel()].reshape_as(parameter)
        )
        offset += parameter.numel()
    report["actual_normalized_dots"] = dots
    report["actual_step_norm"] = float((values - before).norm())
    report["rounding_attempts"] = attempt
    report["rounding_margin"] = margin
    return report

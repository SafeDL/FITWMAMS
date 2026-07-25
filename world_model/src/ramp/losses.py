"""Losses that supervise plans in physical and relational state space."""

from __future__ import annotations

import torch


def masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return (value * valid.float()).sum() / valid.float().sum().clamp_min(1.0)


def candidate_energy(
    plans: torch.Tensor,
    target_controls: torch.Tensor,
    plan_states: torch.Tensor,
    target_states: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    control = (plans - target_controls[:, None]).abs().mean(dim=-1)
    state = (
        (plan_states[..., :4] - target_states[:, None, :, 1:, :4]).abs().mean(dim=-1)
    )
    weight = valid[:, None].float()
    return ((control + state) * weight).sum(dim=(2, 3)) / weight.sum(
        dim=(2, 3)
    ).clamp_min(1.0)


def mixture_loss(
    energy: torch.Tensor, probabilities: torch.Tensor, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    responsibility = torch.softmax(-energy / max(float(temperature), 1.0e-4), dim=-1)
    mixture = (
        -float(temperature)
        * torch.logsumexp(
            torch.log(probabilities.clamp_min(1.0e-8)) - energy / float(temperature),
            dim=-1,
        ).mean()
    )
    calibration = (
        -(responsibility * torch.log(probabilities.clamp_min(1.0e-8)))
        .sum(dim=-1)
        .mean()
    )
    return mixture, calibration

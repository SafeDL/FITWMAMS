"""Objectives shared by the human-response A2 training workflow."""

from __future__ import annotations

import torch
from torch.nn import functional


def normalized_response(response: torch.Tensor, response_iqr: torch.Tensor) -> torch.Tensor:
    """Scale acceleration and absolute jerk with train-only robust scales."""
    return response / response_iqr.to(response)[None, None, :]


def conditional_energy_distance(
    futures: torch.Tensor, human_reference: torch.Tensor, response_iqr: torch.Tensor,
) -> torch.Tensor:
    """Two-sample Energy Distance over response sequences."""
    model = normalized_response(futures, response_iqr).flatten(1)
    human = normalized_response(human_reference, response_iqr).flatten(1)
    model_human = torch.cdist(model, human).mean()
    model_model = torch.cdist(model, model).mean()
    human_human = torch.cdist(human, human).mean()
    return 2.0 * model_human - model_model - human_human


def loo_energy_rewards(
    futures: torch.Tensor, human_reference: torch.Tensor, response_iqr: torch.Tensor,
) -> torch.Tensor:
    """Return the exact leave-one-out contribution reward for each future."""
    if len(futures) < 3:
        raise ValueError("LOO Energy reward requires at least three futures")
    scores = []
    for index in range(len(futures)):
        kept = torch.cat((futures[:index], futures[index + 1:]), dim=0)
        scores.append(-conditional_energy_distance(kept, human_reference, response_iqr))
    score = torch.stack(scores)
    return score.mean() - score


def event_energy_score(
    futures: torch.Tensor, observed: torch.Tensor, response_iqr: torch.Tensor,
) -> torch.Tensor:
    """Proper Energy Score for one observed response and stochastic futures."""
    model = normalized_response(futures, response_iqr).flatten(1)
    target = normalized_response(observed[None], response_iqr).flatten(1)
    return torch.cdist(model, target).mean() - 0.5 * torch.cdist(model, model).mean()


def loo_event_energy_rewards(
    futures: torch.Tensor, observed: torch.Tensor, response_iqr: torch.Tensor,
) -> torch.Tensor:
    """Exact contribution reward for an event-level Energy Score."""
    if len(futures) < 3:
        raise ValueError("LOO Event Energy Score requires at least three futures")
    scores = []
    for index in range(len(futures)):
        kept = torch.cat((futures[:index], futures[index + 1:]), dim=0)
        scores.append(event_energy_score(kept, observed, response_iqr))
    score = torch.stack(scores)
    return score - score.mean()


def prefix_loo_event_energy_rewards(
    futures: torch.Tensor, observed: torch.Tensor, response_iqr: torch.Tensor,
    prefixes: tuple[int, ...] = (5, 10, 25), weights: tuple[float, ...] = (.25, .35, .40),
) -> dict[int, torch.Tensor]:
    if len(prefixes) != len(weights) or abs(sum(weights) - 1.0) > 1.e-6:
        raise ValueError("prefix weights must align and sum to one")
    return {
        int(prefix): float(weight) * loo_event_energy_rewards(futures[:, :prefix], observed[:prefix], response_iqr)
        for prefix, weight in zip(prefixes, weights)
    }


def mechanism_auxiliary_loss(
    *, model_intervention: torch.Tensor, model_baseline: torch.Tensor,
    idm_intervention: torch.Tensor, idm_baseline: torch.Tensor,
    threshold: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiable paired mechanism regularizer for detached rollout states."""
    model_delta = (model_intervention - model_baseline).clamp(-4.0, 4.0)
    idm_delta = (idm_intervention - idm_baseline).clamp(-4.0, 4.0)
    selected = idm_delta.abs() > float(threshold)
    if not selected.any():
        zero = model_delta.sum() * 0.0
        return zero, {"direction": zero, "magnitude": zero, "selected": selected.float().sum()}
    direction = functional.relu(-idm_delta[selected].sign() * model_delta[selected]).mean()
    magnitude = functional.huber_loss(model_delta[selected], idm_delta[selected])
    return direction + 0.1 * magnitude, {
        "direction": direction,
        "magnitude": magnitude,
        "selected": selected.float().sum(),
    }


def controller_induced_jerk(correction: torch.Tensor, dt_s: float = 0.04) -> torch.Tensor:
    """Finite-difference jerk attributable to the post-HiQR controller."""
    return (correction[..., 1:] - correction[..., :-1]).abs() / float(dt_s)

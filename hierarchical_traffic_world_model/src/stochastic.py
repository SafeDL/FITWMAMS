"""Structured stochastic state and causal interaction components.

The modules in this file are intentionally opt-in.  They turn the previous
highly-correlated Gaussian innovation into a persistent, graph-coupled latent
process, while keeping the random stream supplied by the caller explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class LatentTransition:
    """One conditionally-normalizing-flow transition of driver state."""

    state: torch.Tensor
    innovation_log_prob: torch.Tensor
    graph_message: torch.Tensor


class GraphCoupledLatentTransition(nn.Module):
    """Persistent driver states coupled through the observed traffic graph.

    A pair of conditional affine-coupling layers is a small exact conditional
    flow.  The context is built from directed relative-state edges, so one
    vehicle's latent transition can depend on the current state of nearby
    vehicles without observing any future ego signal.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, persistence: float) -> None:
        super().__init__()
        if latent_dim % 2:
            raise ValueError("conditional latent flow requires an even latent dimension")
        self.latent_dim = latent_dim
        self.persistence = float(persistence)
        self.node = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.edge = nn.Sequential(
            nn.Linear(7, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim)
        )
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        context_dim = 4 * latent_dim
        half = latent_dim // 2
        self.first = nn.Sequential(
            nn.Linear(context_dim + half, latent_dim), nn.SiLU(), nn.Linear(latent_dim, 2 * half)
        )
        self.second = nn.Sequential(
            nn.Linear(context_dim + half, latent_dim), nn.SiLU(), nn.Linear(latent_dim, 2 * half)
        )

    def _graph_message(
        self,
        state: torch.Tensor,
        node_features: torch.Tensor,
        current: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        # Edge i <- j: relative distance, velocity, closing speed, and an
        # exponentially decaying geometric affinity.  All are causal values
        # from the realized state at this response boundary.
        delta_position = current[:, :, None, :2] - current[:, None, :, :2]
        delta_velocity = current[:, :, None, 2:4] - current[:, None, :, 2:4]
        distance = torch.linalg.vector_norm(delta_position, dim=-1)
        closing = (delta_velocity[..., 0] * delta_position[..., 0]).sign() * (
            delta_velocity[..., 0].abs()
        )
        affinity = torch.exp(-distance / 35.0)
        edge_features = torch.stack(
            (
                delta_position[..., 0] / 40.0,
                delta_position[..., 1] / 8.0,
                delta_velocity[..., 0] / 15.0,
                delta_velocity[..., 1] / 5.0,
                closing / 15.0,
                affinity,
                (delta_position[..., 1].abs() < 1.8).to(current.dtype),
            ),
            dim=-1,
        )
        node = self.node(node_features)
        edge = self.edge(edge_features)
        key = self.key(node)[:, None] + edge
        query = self.query(node)[:, :, None]
        logits = (query * key).sum(dim=-1) / math.sqrt(self.latent_dim)
        edge_valid = valid[:, :, None] & valid[:, None, :]
        diagonal = torch.eye(current.shape[1], device=current.device, dtype=torch.bool)
        edge_valid = edge_valid & ~diagonal[None]
        # Softmax of an entirely masked row is undefined.  A zero message is
        # the correct result for an isolated or invalid vehicle.
        logits = logits.masked_fill(~edge_valid, -1.0e4)
        weights = torch.softmax(logits, dim=-1) * edge_valid.to(logits.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.einsum("bij,bjd->bid", weights, state + node)

    def forward(
        self,
        previous: torch.Tensor | None,
        innovation: torch.Tensor,
        node_features: torch.Tensor,
        scene_latent: torch.Tensor,
        current: torch.Tensor,
        valid: torch.Tensor,
    ) -> LatentTransition:
        state = torch.zeros_like(innovation) if previous is None else previous
        message = self._graph_message(state, node_features, current, valid)
        node = self.node(node_features)
        scene = scene_latent[:, None].expand_as(node)
        context = torch.cat((state, message, node, scene), dim=-1)
        first, second = innovation.chunk(2, dim=-1)
        shift, log_scale = self.first(torch.cat((context, second), dim=-1)).chunk(2, dim=-1)
        log_scale = 0.25 * torch.tanh(log_scale)
        first = first * log_scale.exp() + 0.35 * torch.tanh(shift)
        shift, second_log_scale = self.second(torch.cat((context, first), dim=-1)).chunk(2, dim=-1)
        second_log_scale = 0.25 * torch.tanh(second_log_scale)
        second = second * second_log_scale.exp() + 0.35 * torch.tanh(shift)
        flowed = torch.cat((first, second), dim=-1)
        if previous is None:
            next_state = flowed
        else:
            next_state = self.persistence * state + math.sqrt(
                1.0 - self.persistence**2
            ) * flowed
        log_base = -0.5 * (innovation.square() + math.log(2.0 * math.pi)).sum(-1)
        log_det = (log_scale + second_log_scale).sum(-1)
        mask = valid.to(next_state.dtype)
        return LatentTransition(
            state=next_state * mask[..., None],
            innovation_log_prob=(log_base - log_det) * mask,
            graph_message=message * mask[..., None],
        )


class CausalInteractionResponseField(nn.Module):
    """Latent-conditioned local response gain for already realized ego changes."""

    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.field = nn.Sequential(
            # The signed, already-realized intervention memory makes brake
            # and acceleration response modes identifiable to the field.  It
            # is causal: the value only aggregates controls committed before
            # this response boundary.
            nn.Linear(3 * hidden_dim + 2 * latent_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Starts as an exact multiplicative identity.  This lets an E1
        # checkpoint be used as the causal-control baseline in a pilot.
        nn.init.zeros_(self.field[-1].weight)
        nn.init.zeros_(self.field[-1].bias)

    def forward(
        self,
        agent_features: torch.Tensor,
        filter_features: torch.Tensor,
        preview_features: torch.Tensor,
        behavior_latent: torch.Tensor,
        graph_message: torch.Tensor,
        intervention_memory: torch.Tensor,
    ) -> torch.Tensor:
        signed_memory = intervention_memory[:, None, None].expand(
            -1, agent_features.shape[1], 1
        ) / 4.0
        feature = torch.cat(
            (
                agent_features,
                filter_features,
                preview_features,
                behavior_latent,
                graph_message,
                signed_memory,
            ),
            dim=-1,
        )
        # Bounded around one: the field can model heterogeneous response
        # strength but cannot reverse the structural, direction-safe prior.
        return 1.0 + 0.5 * torch.tanh(self.field(feature).squeeze(-1))

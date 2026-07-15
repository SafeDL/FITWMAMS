"""Discrete semi-Markov latent interaction state and duration hazards."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SemiMarkovConfig:
    num_states: int = 12
    hidden_dim: int = 128
    max_duration_steps: int = 30
    posterior_temperature: float = 1.0
    boundary_supervision_weight: float = 0.25
    prototype_dim: int = 4
    prototype_weight: float = 1.0
    state_bootstrap_weight: float = 5.0


class SemiMarkovLatentState(nn.Module):
    """Causal prior, full-sequence posterior, and discrete hazard duration.

    Durations are measured in response-update steps.  The terminal response
    state is treated as right-censored: its objective contains survival
    probability, never a fabricated transition at the end of a six-second
    observation window.
    """

    def __init__(self, cfg: SemiMarkovConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h, k = int(cfg.hidden_dim), int(cfg.num_states)
        self.embedding = nn.Embedding(k, h)
        # Learned state prototypes are kept in observed interaction-descriptor
        # space.  This is a VQ-style commitment signal, not an entropy or
        # diversity heuristic: a single code cannot reconstruct distinct
        # natural-driving interaction patterns.
        self.prototype = nn.Embedding(k, int(cfg.prototype_dim))
        self.register_buffer("descriptor_centroids", torch.zeros((k, int(cfg.prototype_dim))))
        self.prior = nn.Sequential(nn.Linear(h * 2, h), nn.SiLU(), nn.Linear(h, k))
        self.hazard = nn.Sequential(nn.Linear(h * 2 + 1, h), nn.SiLU(), nn.Linear(h, 1))
        self.posterior_rnn = nn.GRU(h, h, batch_first=True, bidirectional=True)
        self.posterior_z = nn.Sequential(nn.Linear(h * 2, h), nn.SiLU(), nn.Linear(h, k))
        self.posterior_boundary = nn.Sequential(nn.Linear(h * 2, h), nn.SiLU(), nn.Linear(h, 1))
        # An initially sticky posterior prevents the common degenerate solution
        # where latent states are resampled at every 0.2-second response step.
        nn.init.constant_(self.posterior_boundary[-1].bias, -2.0)

    def state_embedding(self, probabilities: torch.Tensor) -> torch.Tensor:
        return probabilities @ self.embedding.weight

    @torch.no_grad()
    def set_descriptor_centroids(self, centroids: torch.Tensor) -> None:
        if tuple(centroids.shape) != tuple(self.descriptor_centroids.shape):
            raise ValueError("descriptor centroid shape does not match the latent codebook")
        self.descriptor_centroids.copy_(centroids.to(device=self.descriptor_centroids.device, dtype=self.descriptor_centroids.dtype))

    def prior_logits(self, scene: torch.Tensor, previous_state: torch.Tensor) -> torch.Tensor:
        return self.prior(torch.cat((scene, self.state_embedding(previous_state)), dim=-1))

    def hazard_logits(self, scene: torch.Tensor, state: torch.Tensor, elapsed_steps: torch.Tensor) -> torch.Tensor:
        elapsed = elapsed_steps.float().unsqueeze(-1) / max(float(self.cfg.max_duration_steps), 1.0)
        return self.hazard(torch.cat((scene, self.state_embedding(state), elapsed), dim=-1)).squeeze(-1)

    def posterior(self, full_scene: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded, _ = self.posterior_rnn(full_scene)
        state_logits = self.posterior_z(encoded) / max(float(self.cfg.posterior_temperature), 1.0e-4)
        z_soft = F.softmax(state_logits, dim=-1)
        # The decoder receives an actual discrete state path.  The straight-
        # through Gumbel sample permits reconstruction gradients to make states
        # useful while the KL below remains well-defined on q_soft.
        if self.training:
            z = F.gumbel_softmax(state_logits, tau=max(float(self.cfg.posterior_temperature), 1.0e-4), hard=True, dim=-1)
        else:
            z = F.one_hot(state_logits.argmax(dim=-1), num_classes=self.cfg.num_states).to(dtype=z_soft.dtype)
        boundary_logits = self.posterior_boundary(encoded).squeeze(-1)
        boundary = torch.sigmoid(boundary_logits)
        # A sequence begins a new state by definition.  It has no preceding
        # duration event, so it is excluded from duration supervision below.
        boundary = torch.cat((torch.ones_like(boundary[:, :1]), boundary[:, 1:]), dim=1)
        return z_soft, z, boundary, boundary_logits, state_logits

    @staticmethod
    def propagated_states(z: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        """Differentiably preserve state identity until a learned boundary."""
        outputs = [z[:, 0]]
        for step in range(1, z.shape[1]):
            b = boundary[:, step : step + 1]
            outputs.append((1.0 - b) * outputs[-1] + b * z[:, step])
        return torch.stack(outputs, dim=1)

    def training_terms(
        self,
        causal_scene: torch.Tensor,
        full_scene: torch.Tensor,
        boundary_target: torch.Tensor | None = None,
        interaction_descriptor: torch.Tensor | None = None,
        state_target: torch.Tensor | None = None,
        *,
        force_stepwise: bool = False,
        initial_scene: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return posterior paths and the KL/duration/right-censor objectives."""
        q_z_soft, q_z_sample, q_boundary, boundary_logits, state_logits = self.posterior(full_scene)
        if force_stepwise:
            # B1 removes duration persistence while retaining a scene-level
            # categorical interaction state at every response update.
            q_boundary = torch.ones_like(q_boundary)
            q_z = q_z_sample
        else:
            q_z = self.propagated_states(q_z_sample, q_boundary)
        previous = torch.cat((q_z[:, :1], q_z[:, :-1]), dim=1)
        prior_scene = causal_scene
        if initial_scene is not None:
            if initial_scene.shape != causal_scene[:, 0].shape:
                raise ValueError("initial_scene must be [batch, hidden_dim]")
            prior_scene = torch.cat((initial_scene[:, None], causal_scene[:, 1:]), dim=1)
        prior_logits = self.prior_logits(prior_scene.reshape(-1, prior_scene.shape[-1]), previous.reshape(-1, previous.shape[-1]))
        prior_logits = prior_logits.reshape(*causal_scene.shape[:2], -1)
        log_prior = F.log_softmax(prior_logits, dim=-1)
        # The KL uses the categorical posterior probabilities rather than its
        # straight-through sample so it remains a genuine prior/posterior term.
        kl = (q_z_soft * (q_z_soft.clamp_min(1.0e-8).log() - log_prior)).sum(dim=-1).mean()

        elapsed = torch.ones_like(q_boundary)
        # Expected elapsed state age.  At a boundary it resets to one response
        # interval; otherwise it increments smoothly.
        for step in range(1, elapsed.shape[1]):
            elapsed[:, step] = (1.0 - q_boundary[:, step]) * (elapsed[:, step - 1] + 1.0) + q_boundary[:, step]
        hazard_logits = self.hazard_logits(
            prior_scene.reshape(-1, prior_scene.shape[-1]), q_z.reshape(-1, q_z.shape[-1]), elapsed.reshape(-1)
        ).reshape_as(q_boundary)
        # Transition targets exist only after the initial state and before the
        # window end.  The final observed state is right-censored.
        if q_boundary.shape[1] > 1 and not force_stepwise:
            duration_nll = F.binary_cross_entropy_with_logits(
                hazard_logits[:, 1:-1], q_boundary[:, 1:-1], reduction="mean"
            ) if q_boundary.shape[1] > 2 else hazard_logits.new_zeros(())
            censor = -F.logsigmoid(-hazard_logits[:, -1]).mean()
        else:
            duration_nll, censor = hazard_logits.new_zeros(()), hazard_logits.new_zeros(())
        if boundary_target is None or q_boundary.shape[1] <= 1 or force_stepwise:
            posterior_boundary_nll = hazard_logits.new_zeros(())
        else:
            if boundary_target.shape != q_boundary.shape:
                raise ValueError("boundary_target must align with response steps")
            target_values = boundary_target[:, 1:].float()
            positive = target_values.sum()
            negatives = target_values.numel() - positive
            # Behaviour changes are naturally sparse at 0.2 s resolution.
            # Balance the posterior likelihood so the valid all-survival
            # solution cannot win solely through class frequency.
            positive_weight = (negatives / positive.clamp_min(1.0)).clamp(1.0, 20.0)
            posterior_boundary_nll = F.binary_cross_entropy_with_logits(
                boundary_logits[:, 1:], target_values, pos_weight=positive_weight, reduction="mean"
            )
        if interaction_descriptor is None:
            prototype_reconstruction = hazard_logits.new_zeros(())
        else:
            if interaction_descriptor.shape[:2] != q_z.shape[:2] or interaction_descriptor.shape[-1] != self.cfg.prototype_dim:
                raise ValueError("interaction_descriptor must have shape [B, response_steps, prototype_dim]")
            prototype_prediction = q_z @ self.prototype.weight
            prototype_reconstruction = F.smooth_l1_loss(prototype_prediction, interaction_descriptor, reduction="mean")
        if state_target is None:
            state_bootstrap_nll = hazard_logits.new_zeros(())
        else:
            if state_target.shape != q_boundary.shape:
                raise ValueError("state_target must align with response steps")
            state_bootstrap_nll = F.cross_entropy(
                state_logits.reshape(-1, self.cfg.num_states), state_target.reshape(-1).long(), reduction="mean"
            )
        # Useful audit values, not an entropy/diversity loss.
        switches = q_boundary[:, 1:].mean() if q_boundary.shape[1] > 1 else q_boundary.new_zeros(())
        return {
            "posterior_state_probs": q_z,
            "posterior_raw_state_probs": q_z_soft,
            "posterior_boundary_probs": q_boundary,
            "prior_logits": prior_logits,
            "hazard_logits": hazard_logits,
            "latent_kl": kl,
            "duration_nll": duration_nll,
            "censor_nll": censor,
            "posterior_boundary_nll": posterior_boundary_nll,
            "boundary_target_rate": boundary_target[:, 1:].float().mean() if boundary_target.shape[1] > 1 else hazard_logits.new_zeros(()),
            "prototype_reconstruction": prototype_reconstruction,
            "state_bootstrap_nll": state_bootstrap_nll,
            "switch_rate": switches,
        }

    @torch.no_grad()
    def sample_state_and_duration(
        self,
        scene: torch.Tensor,
        previous_state: int | None,
        uniform_state: float,
        uniform_duration: float,
    ) -> tuple[int, int, torch.Tensor]:
        """Inverse-CDF sampling from externally supplied uniforms.

        The method consumes exactly one state and one duration uniform per
        latent segment, which is required for ADS-independent replay.
        """
        device = scene.device
        k = int(self.cfg.num_states)
        prev = torch.zeros((1, k), device=device)
        if previous_state is not None:
            prev[0, int(previous_state)] = 1.0
        logits = self.prior_logits(scene.reshape(1, -1), prev)
        probs = F.softmax(logits, dim=-1)[0]
        cdf = torch.cumsum(probs, dim=0)
        z = int(torch.searchsorted(cdf, torch.tensor(float(uniform_state), device=device).clamp(0, 1 - 1e-7)).item())
        state = F.one_hot(torch.tensor([z], device=device), num_classes=k).float()
        remaining = float(uniform_duration)
        duration = int(self.cfg.max_duration_steps)
        for elapsed in range(1, int(self.cfg.max_duration_steps) + 1):
            h = torch.sigmoid(self.hazard_logits(scene.reshape(1, -1), state, torch.tensor([elapsed], device=device)))[0]
            if remaining <= float(h.item()):
                duration = elapsed
                break
            remaining = (remaining - float(h.item())) / max(1.0 - float(h.item()), 1.0e-8)
        return z, duration, probs

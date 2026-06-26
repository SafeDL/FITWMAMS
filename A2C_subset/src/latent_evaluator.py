"""Evaluate latent MPC episodes with highway-env closed-loop rollouts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from A2C_subset.src.closed_loop_runner import ClosedLoopFollowingRunner
from A2C_subset.src.frozen_diffusion_sampler import FrozenDiffusionSampler


@dataclass
class LatentEvaluation:
    score: float
    actions: np.ndarray
    metrics: dict[str, float]
    trace: list[dict[str, float]]
    context_index: int


class LatentMpcEpisodeEvaluator:
    """Evaluate one context index and one latent sequence as an MPC episode."""

    def __init__(
        self,
        sampler: FrozenDiffusionSampler,
        runner: ClosedLoopFollowingRunner,
        contexts: Any,
        config: dict[str, Any],
        *,
        inference_steps: int | None = None,
    ) -> None:
        if not contexts:
            raise ValueError("contexts must not be empty")
        self.sampler = sampler
        self.runner = runner
        self.contexts = contexts
        self.config = config
        self.inference_steps = inference_steps
        env_cfg = config.get("env", {})
        self.episode_steps = int(env_cfg.get("episode_steps", 200))
        if self.episode_steps <= 0:
            raise ValueError("env.episode_steps must be positive")

    @property
    def context_count(self) -> int:
        return len(self.contexts)

    @property
    def plan_latent_shape(self) -> tuple[int, int]:
        cfg = self.sampler.prior.model.denoiser.cfg
        return int(cfg.horizon_steps), int(cfg.action_dim)

    @property
    def latent_shape(self) -> tuple[int, ...]:
        return self.plan_latent_shape

    def decode_plan(
        self,
        context: dict[str, Any],
        latent: np.ndarray,
    ) -> np.ndarray:
        latent = np.asarray(latent, dtype=np.float32)
        if latent.shape != self.plan_latent_shape:
            raise ValueError(
                f"Expected plan latent shape {self.plan_latent_shape}, "
                f"got {latent.shape}"
            )
        with torch.no_grad():
            sample = self.sampler.sample_from_noise(
                torch.from_numpy(
                    np.asarray(context["scenario_conditions"], dtype=np.float32)[None]
                ).float(),
                torch.from_numpy(latent[None]).float(),
                inference_steps=self.inference_steps,
            )
        return sample.raw_actions[0].detach().cpu().numpy().astype(np.float32)

    def decode_plans(
        self,
        context_indices: np.ndarray,
        latents: np.ndarray,
        *,
        batch_size: int,
    ) -> list[np.ndarray]:
        context_indices = np.asarray(context_indices, dtype=np.int64)
        latents = np.asarray(latents, dtype=np.float32)
        if latents.ndim != 3 or tuple(latents.shape[1:]) != self.plan_latent_shape:
            raise ValueError(
                f"Expected batched latent shape [N, {self.plan_latent_shape[0]}, "
                f"{self.plan_latent_shape[1]}], got {latents.shape}"
            )
        if int(context_indices.shape[0]) != int(latents.shape[0]):
            raise ValueError(
                "context_indices and latents must contain the same number of samples"
            )
        batch_size = max(1, int(batch_size))
        plans: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, int(latents.shape[0]), batch_size):
                end = min(start + batch_size, int(latents.shape[0]))
                conditions = np.stack(
                    [
                        np.asarray(
                            self.contexts[int(context_idx)]["scenario_conditions"],
                            dtype=np.float32,
                        )
                        for context_idx in context_indices[start:end]
                    ],
                    axis=0,
                )
                sample = self.sampler.sample_from_noise(
                    torch.from_numpy(conditions).float(),
                    torch.from_numpy(latents[start:end]).float(),
                    inference_steps=self.inference_steps,
                )
                decoded = sample.raw_actions.detach().cpu().numpy().astype(np.float32)
                plans.extend([decoded[idx].copy() for idx in range(decoded.shape[0])])
        return plans

    def evaluate_decoded_plan(
        self,
        context_index: int,
        plan: np.ndarray,
    ) -> LatentEvaluation:
        if context_index < 0 or context_index >= len(self.contexts):
            raise IndexError(f"context_index out of range: {context_index}")
        context = self.contexts[int(context_index)]
        result = self.runner.rollout(
            context,
            fixed_plan=np.asarray(plan, dtype=np.float32),
            episode_steps=self.episode_steps,
        )
        if result.actions is None:
            raise RuntimeError("Rolling rollout did not return actions")
        metrics = dict(result.metrics)
        metrics.update(
            {
                "context_index": float(context_index),
                "recording_id": float(context.get("recording_id", -1)),
                "event_steps_config": float(self.episode_steps),
                "executed_steps": float(len(result.trace)),
            }
        )
        return LatentEvaluation(
            score=float(result.risk_score),
            actions=result.actions.astype(np.float32),
            metrics=metrics,
            trace=list(result.trace),
            context_index=int(context_index),
        )

    def evaluate(
        self,
        context_index: int,
        z: np.ndarray,
    ) -> LatentEvaluation:
        if context_index < 0 or context_index >= len(self.contexts):
            raise IndexError(f"context_index out of range: {context_index}")
        latent = np.asarray(z, dtype=np.float32)
        if latent.shape != self.latent_shape:
            raise ValueError(
                f"Expected latent shape {self.latent_shape}, "
                f"got {latent.shape}"
            )
        context = self.contexts[int(context_index)]
        plan = self.decode_plan(context, latent)
        return self.evaluate_decoded_plan(context_index, plan)

"""PPO policy loader for PPO highway-env checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = ROOT / "PPO_subset" / "weights" / "ppo" / "model.zip"


class PPOPolicyLoadError(RuntimeError):
    """Raised when the PPO policy checkpoint cannot be loaded."""


def _resolve_checkpoint(path: str | Path | None) -> Path:
    checkpoint = Path(path) if path else DEFAULT_CHECKPOINT_PATH
    if not checkpoint.is_absolute():
        checkpoint = (ROOT / checkpoint).resolve()
    return checkpoint


class PPOPolicy:
    """Stable-baselines3 PPO policy with a TREAD-compatible reset/act API."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        deterministic: bool = True,
        device: str = "auto",
    ) -> None:
        self.checkpoint_path = _resolve_checkpoint(checkpoint_path)
        self.deterministic = bool(deterministic)
        self.device = str(device)
        self._model: Any | None = None
        self._load()

    def _load(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "PPO checkpoint not found: "
                f"{self.checkpoint_path}. Configure ppo_policy.checkpoint_path."
            )
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise PPOPolicyLoadError(
                "stable-baselines3 is required for PPO inference. "
                "Activate the tread conda environment or install stable-baselines3."
            ) from exc
        try:
            self._model = PPO.load(str(self.checkpoint_path), device=self.device)
        except Exception as exc:
            raise PPOPolicyLoadError(
                f"Failed to load PPO checkpoint from {self.checkpoint_path}: {exc}"
            ) from exc

    def reset(self) -> None:
        """Reset policy state. The PPO MLP policy is memoryless."""

    @property
    def observation_shape(self) -> tuple[int, int]:
        return (5, 5)

    def act(self, observation: np.ndarray) -> int:
        if self._model is None:
            raise PPOPolicyLoadError("PPO policy is not loaded")
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != self.observation_shape:
            if obs.size != int(np.prod(self.observation_shape)):
                raise ValueError(
                    f"PPO observation must have shape {self.observation_shape}, "
                    f"got {obs.shape}"
                )
            obs = obs.reshape(self.observation_shape)
        action, _ = self._model.predict(obs, deterministic=self.deterministic)
        return int(np.asarray(action).item())

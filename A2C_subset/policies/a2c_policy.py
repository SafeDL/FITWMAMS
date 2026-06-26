"""A2C policy loader for the reference D2RL highway-env checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = ROOT / "A2C_subset" / "weights" / "a2c" / "model.zip"


class A2CPolicyLoadError(RuntimeError):
    """Raised when the A2C policy checkpoint cannot be loaded."""


def _resolve_checkpoint(path: str | Path | None) -> Path:
    checkpoint = Path(path) if path else DEFAULT_CHECKPOINT_PATH
    if not checkpoint.is_absolute():
        checkpoint = (ROOT / checkpoint).resolve()
    return checkpoint


class A2CPolicy:
    """Stable-baselines3 A2C policy with a TREAD-compatible reset/act API."""

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
                "A2C checkpoint not found: "
                f"{self.checkpoint_path}. Configure a2c_policy.checkpoint_path."
            )
        try:
            from stable_baselines3 import A2C
            from stable_baselines3.common.policies import ActorCriticPolicy
            from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
            import torch.nn as nn
        except ImportError as exc:
            raise A2CPolicyLoadError(
                "stable-baselines3 is required for A2C inference. "
                "Activate the tread conda environment or install stable-baselines3."
            ) from exc

        class D2RLNetwork(BaseFeaturesExtractor):
            def __init__(self, observation_space: Any, features_dim: int = 256) -> None:
                super().__init__(observation_space, features_dim)
                self.flatten = nn.Flatten()
                self.hidden = nn.Sequential(
                    nn.Linear(int(np.prod(observation_space.shape)), features_dim),
                    nn.ReLU(),
                )
                self.d2rl1 = nn.Sequential(
                    nn.Linear(features_dim, features_dim),
                    nn.ReLU(),
                )
                self.d2rl2 = nn.Sequential(
                    nn.Linear(features_dim, features_dim),
                    nn.ReLU(),
                )

            def forward(self, observations: Any) -> Any:
                x = self.flatten(observations)
                x1 = self.hidden(x)
                x2 = self.d2rl1(x1)
                return self.d2rl2(x1 + x2)

        class D2RLA2CPolicy(ActorCriticPolicy):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(
                    *args,
                    **kwargs,
                    features_extractor_class=D2RLNetwork,
                    features_extractor_kwargs={"features_dim": 256},
                )

        try:
            self._model = A2C.load(
                str(self.checkpoint_path),
                device=self.device,
                custom_objects={"policy_class": D2RLA2CPolicy},
            )
        except Exception as exc:
            raise A2CPolicyLoadError(
                f"Failed to load A2C checkpoint from {self.checkpoint_path}: {exc}"
            ) from exc

    def reset(self) -> None:
        """Reset policy state. The A2C actor-critic policy is memoryless."""

    @property
    def observation_shape(self) -> tuple[int, int]:
        return (5, 5)

    def act(self, observation: np.ndarray) -> int:
        if self._model is None:
            raise A2CPolicyLoadError("A2C policy is not loaded")
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != self.observation_shape:
            if obs.size != int(np.prod(self.observation_shape)):
                raise ValueError(
                    f"A2C observation must have shape {self.observation_shape}, "
                    f"got {obs.shape}"
                )
            obs = obs.reshape(self.observation_shape)
        action, _ = self._model.predict(obs, deterministic=self.deterministic)
        return int(np.asarray(action).item())

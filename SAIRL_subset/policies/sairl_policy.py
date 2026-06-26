"""SAIRL policy adapter for TREAD long-tail subset evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = ROOT / "SAIRL_subset" / "weights" / "sairl" / "model.npz"
DEFAULT_ACTIONS = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}


class SAIRLPolicyLoadError(RuntimeError):
    """Raised when the SAIRL policy checkpoint cannot be loaded."""


def _resolve_checkpoint(path: str | Path | None) -> Path:
    checkpoint = Path(path) if path else DEFAULT_CHECKPOINT_PATH
    if not checkpoint.is_absolute():
        checkpoint = (ROOT / checkpoint).resolve()
    return checkpoint


def _checkpoint_prefix_exists(path: Path) -> bool:
    if path.suffix in {".npz", ".pt", ".pth"}:
        return path.exists()
    return path.exists() or path.with_suffix(path.suffix + ".index").exists()


def _import_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise SAIRLPolicyLoadError(
            "PyTorch is required for SAIRLPolicy inference. Activate the tread "
            "environment or install torch."
        ) from exc
    return torch, nn


class _TorchSAIRLActor:
    """PyTorch mirror of the TensorFlow discrete policy MLP."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_units: tuple[int, ...],
        weights_path: Path,
    ) -> None:
        torch, nn = _import_torch()
        layers: list[Any] = []
        last_dim = int(obs_dim)
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            last_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, int(action_dim)))
        self.torch = torch
        self.model = nn.Sequential(*layers)
        self.model.eval()
        self._load_npz(weights_path)

    def _load_npz(self, weights_path: Path) -> None:
        values = np.load(weights_path, allow_pickle=False)
        linear_layers = [
            module
            for module in self.model
            if module.__class__.__name__ == "Linear"
        ]
        expected = len(linear_layers)
        for idx, layer in enumerate(linear_layers):
            weight_key = f"w{idx}"
            bias_key = f"b{idx}"
            if weight_key not in values or bias_key not in values:
                raise SAIRLPolicyLoadError(
                    f"Converted SAIRL weight file {weights_path} is missing "
                    f"{weight_key}/{bias_key}; expected {expected} dense layers."
                )
            weight = np.asarray(values[weight_key], dtype=np.float32)
            bias = np.asarray(values[bias_key], dtype=np.float32)
            if weight.shape != tuple(layer.weight.detach().numpy().T.shape):
                raise SAIRLPolicyLoadError(
                    f"SAIRL weight {weight_key} has shape {weight.shape}, expected "
                    f"{tuple(layer.weight.detach().numpy().T.shape)}"
                )
            if bias.shape != tuple(layer.bias.detach().numpy().shape):
                raise SAIRLPolicyLoadError(
                    f"SAIRL bias {bias_key} has shape {bias.shape}, expected "
                    f"{tuple(layer.bias.detach().numpy().shape)}"
                )
            with self.torch.no_grad():
                layer.weight.copy_(self.torch.from_numpy(weight.T))
                layer.bias.copy_(self.torch.from_numpy(bias))

    def action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        with self.torch.no_grad():
            logits = self.model(self.torch.from_numpy(obs))
            probs = self.torch.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy()[0].astype(np.float32)


def convert_tensorflow_checkpoint_to_npz(
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Convert the TensorFlow 1.x SAIRL policy weights into a small NPZ file."""

    checkpoint = _resolve_checkpoint(checkpoint_path)
    if not _checkpoint_prefix_exists(checkpoint):
        raise FileNotFoundError(f"SAIRL checkpoint not found: {checkpoint}")
    target = Path(output_path) if output_path else checkpoint.with_name(
        checkpoint.name + ".policy_net.npz"
    )
    if not target.is_absolute():
        target = (ROOT / target).resolve()
    try:
        import tensorflow.compat.v1 as tf
    except ImportError as exc:
        raise SAIRLPolicyLoadError(
            "TensorFlow 1.x compatibility API is required to convert the original "
            "SAIRL checkpoint. Install tensorflow in the tread environment or pass "
            "an already converted .npz policy weight file."
        ) from exc

    reader = tf.train.NewCheckpointReader(str(checkpoint))
    variable_map = reader.get_variable_to_shape_map()
    dense_weights: list[tuple[int, np.ndarray, np.ndarray]] = []
    for name in sorted(variable_map):
        if not name.startswith("policy/policy_net/dense"):
            continue
        if not name.endswith("/kernel"):
            continue
        stem = name[: -len("/kernel")]
        bias_name = stem + "/bias"
        if bias_name not in variable_map:
            continue
        dense_suffix = stem.rsplit("dense", 1)[-1]
        dense_idx = 0 if dense_suffix == "" else int(dense_suffix.lstrip("_"))
        dense_weights.append(
            (
                dense_idx,
                reader.get_tensor(name).astype(np.float32),
                reader.get_tensor(bias_name).astype(np.float32),
            )
        )
    dense_weights.sort(key=lambda item: item[0])
    if len(dense_weights) != 4:
        raise SAIRLPolicyLoadError(
            "Expected 4 policy dense layers in the SAIRL TensorFlow checkpoint, "
            f"found {len(dense_weights)} at {checkpoint}"
        )
    payload: dict[str, np.ndarray] = {}
    for idx, (_dense_idx, weight, bias) in enumerate(dense_weights):
        payload[f"w{idx}"] = weight
        payload[f"b{idx}"] = bias
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **payload)
    LOGGER.info("Converted SAIRL TensorFlow checkpoint to %s", target)
    return target


class SAIRLPolicy:
    """Discrete SAIRL policy with a stable ``reset``/``act`` interface."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        obs_dim: int = 35,
        action_dim: int = 5,
        hidden_units: tuple[int, ...] = (96, 96, 96),
        deterministic: bool = True,
        seed: int = 0,
    ) -> None:
        self.checkpoint_path = _resolve_checkpoint(checkpoint_path)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_units = tuple(int(item) for item in hidden_units)
        self.deterministic = bool(deterministic)
        self.rng = np.random.default_rng(int(seed))
        self._actor: _TorchSAIRLActor | None = None
        self._load()

    def _load(self) -> None:
        if self.checkpoint_path.suffix != ".npz":
            raise SAIRLPolicyLoadError(
                "SAIRLPolicy expects local NPZ weights. Run "
                "SAIRL_subset/scripts/convert_sairl_checkpoint.py first if you "
                "only have a TensorFlow checkpoint."
            )
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"SAIRL NPZ weights not found: {self.checkpoint_path}")
        self._actor = _TorchSAIRLActor(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_units=self.hidden_units,
            weights_path=self.checkpoint_path,
        )
        LOGGER.info("Loaded SAIRL policy weights from %s", self.checkpoint_path)

    def reset(self) -> None:
        """Reset policy state. The SAIRL MLP is memoryless."""

    def act(self, observation: np.ndarray) -> int:
        if self._actor is None:
            raise SAIRLPolicyLoadError("SAIRLPolicy is not loaded")
        probs = self._actor.action_probabilities(observation)
        if self.deterministic:
            return int(np.argmax(probs))
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs / max(float(np.sum(probs)), 1.0e-12)
        return int(self.rng.choice(np.arange(self.action_dim), p=probs))

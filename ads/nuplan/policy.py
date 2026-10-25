"""Shared lifecycle for strictly loaded native nuPlan planning networks."""

import json
from pathlib import Path

import numpy as np
import torch

from ads.common.instances import file_hash
from ads.common.randomness import PolicyRandomContext


def collate(samples, device):
    return {
        key: torch.as_tensor(
            np.stack([sample[key] for sample in samples]), device=device
        )
        for key in samples[0]
    }


def strip_module(state):
    return {key.removeprefix("module."): value for key, value in state.items()}


def prefixed_state(network):
    return {
        "module." + key: value.detach().cpu()
        for key, value in network.state_dict().items()
    }


class NativePlannerPolicy:
    algorithm = None

    def __init__(self, network, config, *, device="cpu", optimizer_state=None):
        self.network = network.to(device)
        self.config, self.device = config, torch.device(device)
        self.optimizer_state = optimizer_state
        self.training_state = None
        self.ema_state = None
        self.reset(PolicyRandomContext(self.algorithm))

    def reset(self, context):
        self.random_context, self.decision = context, 0

    @torch.no_grad()
    def predict(self, observation, deterministic=True):
        self.network.eval()
        with self.random_context.step(
            self.decision, self.device, use=self.algorithm + "_native_inference"
        ):
            output = self._predict(observation)
        self.decision += 1
        trajectory = np.asarray(output, np.float32)
        if trajectory.shape != (80, 3) or not np.isfinite(trajectory).all():
            raise ValueError(f"{self.algorithm} returned an invalid native trajectory")
        return trajectory, None

    def export_native(self, path):
        from ads.nuplan.assets import official_path

        path = Path(path)
        if path.resolve().is_relative_to(
            official_path(self.algorithm).parent.resolve()
        ):
            # File-based planners also write their configuration beside the weight.
            raise ValueError("native export cannot write into official artifacts")
        self.save(path, optimizer=False)
        metadata = {
            "algorithm": self.algorithm,
            "sha256": file_hash(path),
            "format": "official native checkpoint",
            "model_config": self.config,
        }
        path.with_name(path.name + ".source.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

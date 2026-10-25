"""Strict import and native export of official nuPlan CaRL weights."""

from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from ads.carl.vendor.config import GlobalConfig
from ads.carl.vendor.model import PPOPolicy
from ads.common.randomness import PolicyRandomContext


def observation_space():
    return spaces.Dict(
        {
            "bev_semantics": spaces.Box(0, 255, (9, 256, 256), dtype=np.uint8),
            "measurements": spaces.Box(-np.inf, np.inf, (10,), dtype=np.float32),
            "value_measurements": spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),
        }
    )


class CaRLPolicy:
    algorithm = "carl"

    def __init__(self, network, *, device="cpu", optimizer_state=None):
        self.device = torch.device(device)
        self.network = network.to(self.device)
        self.optimizer_state = optimizer_state
        self.repair_state = None
        self.reset(PolicyRandomContext(self.algorithm))

    def reset(self, context):
        self.random_context, self.decision = context, 0

    @classmethod
    def load(cls, path, *, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = GlobalConfig()
        if isinstance(payload, dict) and payload.get("schema") == "ads_carl_v1":
            config.__dict__.update(payload["config"])
            state = payload["state_dict"]
            optimizer = payload["optimizer"]
        else:
            state, optimizer = payload, None
        network = PPOPolicy(
            observation_space(),
            spaces.Box(-1, 1, (2,), dtype=np.float32),
            config=config,
        )
        network.load_state_dict(state, strict=True)
        policy = cls(network, device=device, optimizer_state=optimizer)
        if isinstance(payload, dict) and payload.get("schema") == "ads_carl_v1":
            policy.repair_state = payload.get("repair")
        return policy

    def tensors(self, observation, *, batched=False):
        return {
            k: torch.as_tensor(v if batched else v[None], device=self.device)
            for k, v in observation.items()
        }

    @torch.no_grad()
    def predict(self, observation, deterministic=True):
        self.network.eval()
        with self.random_context.step(self.decision, self.device, use="beta_action"):
            action = self.network(
                self.tensors(observation), deterministic=deterministic
            )[0]
        self.decision += 1
        return action[0].cpu().numpy(), None

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "ads_carl_v1",
                "state_dict": self.network.state_dict(),
                "config": self.network.config.__dict__,
                "optimizer": self.optimizer_state,
                "repair": self.repair_state,
            },
            path,
        )

    def export_native(self, path):
        """Save the raw state_dict accepted by the official nuPlan PPOPlanner."""
        path = Path(path)
        if path.resolve().is_relative_to(
            (Path(__file__).parent / "artifacts").resolve()
        ):
            raise ValueError("native export cannot write into official artifacts")
        native = PPOPolicy(
            observation_space(),
            spaces.Box(-1, 1, (2,), dtype=np.float32),
            config=GlobalConfig(),
        )
        native.load_state_dict(self.network.state_dict(), strict=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(native.state_dict(), path)

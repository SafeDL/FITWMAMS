"""Official PlanT architecture with causal highway object and road inputs."""

import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch
from omegaconf import OmegaConf

from ads.common.randomness import PolicyRandomContext
from ads.plant2.vendor.model import HFLM


def collate(samples, device, *, training=False):
    max_objects = max(1, max(len(s["objects"]) for s in samples))
    indices = np.zeros((len(samples), max_objects), np.int64)
    objects, targets = [np.zeros(7, np.float32)], [np.full(4, -999, np.int64)]
    for row, sample in enumerate(samples):
        start = len(objects)
        indices[row, : len(sample["objects"])] = np.arange(
            start, start + len(sample["objects"])
        )
        objects.extend(sample["objects"])
        if training:
            targets.extend(sample["forecast"])
    batch = {
        key: torch.as_tensor(np.stack([s[key] for s in samples]), device=device)
        for key in ("route_original", "speed_limit", "input_ego_speed", "BEV")
    }
    batch.update(
        idxs=torch.as_tensor(indices, device=device),
        x_objs=torch.as_tensor(np.stack(objects), device=device),
        y_objs=torch.as_tensor(np.stack(targets), device=device) if training else None,
    )
    return batch


class PlanTPolicy:
    algorithm = "plant2"

    def __init__(self, network, cfg, *, device="cpu", optimizer_state=None):
        self.cfg = cfg
        self.device = torch.device(device)
        self.network = network.to(self.device)
        self.optimizer_state = optimizer_state
        self.reset(PolicyRandomContext(self.algorithm))

    def reset(self, context):
        self.random_context, self.decision = context, 0

    @classmethod
    def load(cls, path, *, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") == "ads_plant_v1":
            cfg = OmegaConf.create(payload["config"])
            state = payload["state_dict"]
            optimizer = payload["optimizer"]
        else:
            cfg = payload["hyper_parameters"]["cfg"]
            cfg = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
            state = {
                k.removeprefix("model."): v
                for k, v in payload["state_dict"].items()
                if k.startswith("model.")
            }
            optimizer = None
        # Hydra run names/paths in the original checkpoint need its training
        # process's resolvers. The architecture only consumes the model section.
        cfg = OmegaConf.create(
            {"model": OmegaConf.to_container(cfg.model, resolve=True)}
        )
        if (
            cfg.model.waypoints.representation != "path+wps"
            or cfg.model.waypoints.generator != "linear"
        ):
            raise ValueError(
                "this adaptation requires the official path+wps/linear checkpoint"
            )
        network = HFLM(cfg.model.network, cfg)
        network.load_state_dict(state, strict=True)
        return cls(network, cfg, device=device, optimizer_state=optimizer)

    @torch.no_grad()
    def predict(self, observation, deterministic=True):
        self.network.eval()
        with self.random_context.step(
            self.decision, self.device, use="trajectory_inference"
        ):
            _, _, (path, waypoints, _), _ = self.network(
                collate([observation], self.device)
            )
        self.decision += 1
        return (path[0].cpu().numpy(), waypoints[0].cpu().numpy()), None

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "ads_plant_v1",
                "config": OmegaConf.to_container(self.cfg, resolve=True),
                "state_dict": self.network.state_dict(),
                "optimizer": self.optimizer_state,
            },
            path,
        )

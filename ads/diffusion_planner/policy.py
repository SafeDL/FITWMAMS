"""Official Diffusion Planner, including its normalizers and DPM solver."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ads.diffusion_planner.vendor.diffusion_planner.model.diffusion_planner import (
    Diffusion_Planner,
)
from ads.diffusion_planner.vendor.diffusion_planner.utils.normalizer import (
    ObservationNormalizer,
    StateNormalizer,
)
from ads.nuplan.policy import NativePlannerPolicy, collate, prefixed_state, strip_module


class DiffusionPlannerPolicy(NativePlannerPolicy):
    algorithm = "diffusion_planner"

    @classmethod
    def load(cls, path, *, device="cpu"):
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = payload.get("ads_model_config")
        if config is None:
            config = json.loads((path.parent / "args.json").read_text())
        args = SimpleNamespace(**config)
        args.device = str(device)
        args.state_normalizer = StateNormalizer(**config["state_normalizer"])
        args.observation_normalizer = ObservationNormalizer(
            {
                key: {name: torch.tensor(value) for name, value in stats.items()}
                for key, stats in config["observation_normalizer"].items()
            }
        )
        args.guidance_fn = None
        network = Diffusion_Planner(args)
        network.load_state_dict(strip_module(payload["ema_state_dict"]), strict=True)
        policy = cls(
            network, config, device=device, optimizer_state=payload.get("ads_optimizer")
        )
        policy.args = args
        policy.repair_state = payload.get("ads_repair")
        policy.training_state = strip_module(
            payload["model"] if policy.repair_state else payload["ema_state_dict"]
        )
        if policy.repair_state is None:
            policy.optimizer_state = None
        policy.ema_state = {
            key: value.detach().cpu().clone()
            for key, value in policy.network.state_dict().items()
        }
        return policy

    def inputs(self, samples):
        inputs = collate(samples, self.device)
        inputs.pop("ego_agent_past", None)
        return self.args.observation_normalizer(inputs)

    def denoise(self, inputs, trajectories, time):
        """Native DiT prediction at supplied noise/time, independent of sampler mode."""
        context = self.network.encoder(inputs)["encoding"]
        neighbors = inputs["neighbor_agents_past"][
            :, : self.args.predicted_neighbor_num, -1, :4
        ]
        missing = (neighbors == 0).all(-1)
        batch, agents = trajectories.shape[:2]
        return self.network.decoder.decoder.dit(
            trajectories.reshape(batch, agents, -1),
            time,
            context,
            inputs["route_lanes"],
            missing,
        ).reshape(batch, agents, -1, 4)

    def _predict(self, observation):
        features = {key: value.copy() for key, value in observation["features"].items()}
        # The native nuPlan inference builder intentionally leaves ego dynamics zero.
        features["ego_current_state"] = np.array(
            (0, 0, 1, 0, 0, 0, 0, 0, 0, 0), np.float32
        )
        _, output = self.network(self.inputs([features]))
        future = output["prediction"][0, 0].cpu().numpy()
        return np.column_stack((future[:, :2], np.arctan2(future[:, 3], future[:, 2])))

    def save(self, path, *, optimizer=True):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = prefixed_state(self.network)
        raw = (
            state
            if self.training_state is None
            else {
                "module." + key: value.cpu()
                for key, value in self.training_state.items()
            }
        )
        torch.save(
            {
                "model": raw,
                "ema_state_dict": state,
                "ads_model_config": self.config,
                "ads_optimizer": self.optimizer_state if optimizer else None,
                "ads_repair": self.repair_state,
            },
            path,
        )
        (path.parent / "args.json").write_text(json.dumps(self.config, indent=2) + "\n")

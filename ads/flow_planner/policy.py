"""Official Flow Planner network, flow ODE, normalization and action assembly."""

from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from ads.flow_planner.vendor.flow_planner.data.dataset.nuplan import NuPlanDataSample
from ads.nuplan.policy import NativePlannerPolicy, collate, prefixed_state, strip_module


class FlowPlannerPolicy(NativePlannerPolicy):
    algorithm = "flow_planner"

    @classmethod
    def load(cls, path, *, device="cpu"):
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = OmegaConf.load(path.parent / "model_config.yaml")
        cfg.device = str(device)
        dataset_config = OmegaConf.load(
            Path(__file__).parent
            / "vendor/flow_planner/script/data/dataset/nuplan_data.yaml"
        )
        cfg.data = {"dataset": {"train": dataset_config["train"]}}
        model_config = OmegaConf.to_container(cfg.model, resolve=True)

        def targets(value):
            if isinstance(value, dict):
                return {key: targets(item) for key, item in value.items()}
            if isinstance(value, list):
                return [targets(item) for item in value]
            return (
                "ads.flow_planner.vendor.flow_planner." + value[len("flow_planner.") :]
                if isinstance(value, str) and value.startswith("flow_planner.")
                else value
            )

        network = instantiate(targets(model_config), device=str(device))
        state = payload.get("ema_state_dict", payload)
        network.load_state_dict(strip_module(state), strict=True)
        policy = cls(
            network,
            OmegaConf.to_container(cfg, resolve=False),
            device=device,
            optimizer_state=payload.get("ads_optimizer"),
        )
        policy.native_config = cfg
        policy.training_state = (
            strip_module(payload["model"]) if "model" in payload else None
        )
        policy.ema_state = {
            key: value.detach().cpu().clone()
            for key, value in policy.network.state_dict().items()
        }
        return policy

    def inputs(self, samples, future=None):
        values = collate(samples, self.device)
        return NuPlanDataSample(
            batched=True,
            ego_past=values["ego_agent_past"],
            ego_current=values["ego_current_state"],
            neighbor_past=values["neighbor_agents_past"],
            lanes=values["lanes"],
            lanes_speedlimit=values["lanes_speed_limit"],
            lanes_has_speedlimit=values["lanes_has_speed_limit"],
            routes=values["route_lanes"],
            routes_speedlimit=values["route_lanes_speed_limit"],
            routes_has_speedlimit=values["route_lanes_has_speed_limit"],
            map_objects=values["static_objects"],
            ego_future=torch.tensor([], device=self.device)
            if future is None
            else future,
        )

    def _predict(self, observation):
        output = self.network(
            self.inputs([observation["features"]]),
            mode="inference",
            use_cfg=True,
            cfg_weight=1.0,
        )
        output = (
            output[0, 0].cpu().numpy() if output.ndim == 4 else output[0].cpu().numpy()
        )
        return np.column_stack((output[:, :2], np.arctan2(output[:, 3], output[:, 2])))

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
                "ads_optimizer": self.optimizer_state if optimizer else None,
            },
            path,
        )
        # The official constructor does not pass its device to the root network.
        self.native_config.model.device = "${device}"
        OmegaConf.save(self.native_config, path.parent / "model_config.yaml")

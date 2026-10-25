"""DTPP's official conditional predictor, learned cost and two-stage tree planner."""

import math
from pathlib import Path

import torch
from torch import nn

from ads.dtpp.vendor.scenario_tree_prediction import Decoder, Encoder
from ads.nuplan.policy import NativePlannerPolicy, collate


class DTPPPolicy(NativePlannerPolicy):
    algorithm = "dtpp"

    @classmethod
    def load(cls, path, *, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        encoder, decoder = Encoder(), Decoder()
        encoder.load_state_dict(payload["encoder"], strict=True)
        decoder.load_state_dict(payload["decoder"], strict=True)
        return cls(
            nn.ModuleDict({"encoder": encoder, "decoder": decoder}),
            {"neighbors": 10, "max_branch": 30, "first_stage_s": 3, "future_s": 8},
            device=device,
            optimizer_state=payload.get("ads_optimizer"),
        )

    def reset(self, context):
        super().reset(context)
        self.map_api = None
        self.tree = None

    def _plan(self, observation):
        from ads.dtpp.vendor.trajectory_tree_planner import TreePlanner
        from ads.nuplan.scene import planner_input

        initialization, current = planner_input(
            observation, self.decision, self.map_api
        )
        self.map_api = initialization.map_api
        if self.tree is None:
            self.tree = TreePlanner(
                str(self.device), self.network["encoder"], self.network["decoder"]
            )
            # The native 120 m crop cannot cover eight seconds at highway speeds.
            self.tree.max_path_len = math.ceil(
                max(lane.baseline_path.length for lane in self.map_api.lanes)
            )
        ego, detections = current.history.current_state
        features = collate([observation["features"]], self.device)
        plan = self.tree.plan(
            self.decision,
            ego,
            features,
            self.map_api.block,
            [self.map_api.block],
            [lane.id for lane in self.map_api.lanes],
            [],
            detections,
        )
        return plan

    def _predict(self, observation):
        return self._plan(observation).detach().cpu().numpy()

    def save(self, path, *, optimizer=True):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder": {
                    k: v.detach().cpu()
                    for k, v in self.network["encoder"].state_dict().items()
                },
                "decoder": {
                    k: v.detach().cpu()
                    for k, v in self.network["decoder"].state_dict().items()
                },
                "ads_optimizer": self.optimizer_state if optimizer else None,
            },
            path,
        )

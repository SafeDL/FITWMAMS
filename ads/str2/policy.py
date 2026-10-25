"""Published STR2 MoE planning network and its native STR/PDM closed planner."""

import json
import shutil
from dataclasses import fields
from pathlib import Path

import torch
from safetensors.torch import load_file

from ads.nuplan.policy import NativePlannerPolicy, collate


class STR2Policy(NativePlannerPolicy):
    algorithm = "str2"

    @classmethod
    def load(cls, path, *, device="cpu"):
        from ads.str2.vendor.transformer4planning.models.backbone.mixtral import (
            STR_Mixtral,
            STRMixtralConfig,
        )
        from ads.str2.vendor.transformer4planning.utils.args import ModelArguments

        path = Path(path)
        saved = json.loads((path / "config.json").read_text())
        names = {f.name for f in fields(ModelArguments)}
        args = ModelArguments(
            **{key: value for key, value in saved.items() if key in names}
        )
        config = STRMixtralConfig()
        config.update_by_model_args(args)
        config.update(saved)
        # Transformers 4.49 materializes this in __init__; the 4.44 release derived it.
        config.head_dim = config.hidden_size // config.num_attention_heads
        config.traj_cluster_file = str(
            (path / Path(saved["traj_cluster_file"]).name).resolve()
        )
        if not Path(config.traj_cluster_file).is_file():
            raise FileNotFoundError("STR2 requires its official trajectory vocabulary")
        # Identical attention equations/parameters using Torch; no FlashAttention build is required.
        config._attn_implementation = "eager"
        network = STR_Mixtral(config)
        network.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
        policy = cls(network, saved, device=device)
        policy.source_directory = path
        optimizer = path / "ads_optimizer.pth"
        if optimizer.exists():
            policy.optimizer_state = torch.load(
                optimizer, map_location="cpu", weights_only=False
            )
        return policy

    def reset(self, context):
        super().reset(context)
        self.closed_planner = None
        self.map_api = None

    def inputs(self, samples, target=None):
        inputs = collate(samples, self.device)
        inputs["trajectory_label"] = (
            torch.zeros((len(samples), 80, 4), device=self.device)
            if target is None
            else target
        )
        return inputs

    def _predict(self, observation):
        from ads.str2.postprocess import make_closed_planner, postprocess

        output = self.network.generate(**self.inputs([observation["features"]]))
        trajectory = output["traj_logits"][0].detach().cpu().float().numpy()
        trajectory = (
            trajectory[:, [0, 1, 3]] if trajectory.shape[-1] == 4 else trajectory
        )
        if self.closed_planner is None:
            self.closed_planner, self.map_api = make_closed_planner(observation)
        return postprocess(
            self.closed_planner, self.map_api, observation, trajectory, self.decision
        )

    def save(self, path, *, optimizer=True):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        from safetensors.torch import save_file

        save_file(
            {
                key: value.detach().cpu().contiguous()
                for key, value in self.network.state_dict().items()
            },
            str(path / "model.safetensors"),
        )
        (path / "config.json").write_text(json.dumps(self.config, indent=2) + "\n")
        if self.network.generation_config is not None:
            self.network.generation_config.to_json_file(path / "generation_config.json")
        vocabulary = Path(self.network.config.traj_cluster_file)
        destination = path / vocabulary.name
        if destination.resolve() != vocabulary.resolve():
            shutil.copyfile(vocabulary, destination)
        if optimizer and self.optimizer_state is not None:
            torch.save(self.optimizer_state, path / "ads_optimizer.pth")

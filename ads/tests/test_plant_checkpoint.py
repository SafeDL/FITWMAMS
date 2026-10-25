import torch
from omegaconf import OmegaConf

from ads.plant2.policy import PlanTPolicy


def test_saved_adaptation_does_not_depend_on_original_hydra_process(
    tmp_path, monkeypatch
):
    class SmallNetwork(torch.nn.Module):
        def __init__(self, *args):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    monkeypatch.setattr("ads.plant2.policy.HFLM", SmallNetwork)
    source = tmp_path / "official.ckpt"
    cfg = OmegaConf.create(
        {
            "wandb_name": "${hydra:job.name}",
            "model": {
                "waypoints": {"representation": "path+wps", "generator": "linear"},
                "network": {},
            },
        }
    )
    torch.save(
        {
            "hyper_parameters": {"cfg": cfg},
            "state_dict": {"model.weight": torch.ones(1)},
        },
        source,
    )
    policy = PlanTPolicy.load(source)
    output = tmp_path / "adapted.ckpt"
    policy.save(output)
    restored = PlanTPolicy.load(output)
    assert list(restored.cfg) == ["model"]
    torch.testing.assert_close(restored.network.weight, policy.network.weight)

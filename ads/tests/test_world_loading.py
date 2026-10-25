"""Check real distribution loading, including the fitted natural Flow asset."""

from pathlib import Path

import pytest
import yaml

from ads.common.setup import REPO_ROOT, load_world
from hierarchical_world_model.src.composition import HierarchicalWorldSampler


@pytest.mark.parametrize("distribution", ["tail_generated", "natural_generated"])
def test_registered_generator_loads_its_declared_population(distribution):
    settings = yaml.safe_load(
        (REPO_ROOT / "ads/configs/risk_stratified_repair.yaml").read_text()
    )
    if (
        distribution == "natural_generated"
        and not (REPO_ROOT / settings["evaluation"]["natural_flow_checkpoint"]).exists()
    ):
        pytest.skip("natural Flow is fitted by the formal experiment stage")
    settings["runtime"]["device"] = "cpu"
    sampler, _, _, assets = load_world(settings, distribution=distribution)
    assert sampler.flow.training is False
    assert len(assets["flow_checkpoint"]) == 64


def test_natural_checkpoint_cannot_silently_replace_tail_flow():
    checkpoint = REPO_ROOT / "results/natural_flow/checkpoints/scenario_flow.pt"
    if not checkpoint.exists():
        pytest.skip("natural Flow has not been fitted")
    with pytest.raises(ValueError, match="evt_tail Flow checkpoint"):
        HierarchicalWorldSampler(
            flow_checkpoint=checkpoint,
            diffusion_checkpoint=Path("unused"),
            diffusion_contract=Path("unused"),
            repo_root=REPO_ROOT,
        )

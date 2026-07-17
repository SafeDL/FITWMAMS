"""Contract tests for the checkpoint-independent BARS reproduction setup."""
from __future__ import annotations

from pathlib import Path

from world_model.scripts.train_bars_m2_v5 import materialize_reproduction_configs
from world_model.src.utils import load_yaml


def test_materialized_reproduction_uses_only_same_run_artifacts(tmp_path: Path) -> None:
    m1 = Path("world_model/scripts/configs/highd_behavior_anchored_semi_markov.yaml").resolve()
    m2 = Path("world_model/scripts/configs/highd_bars_m2_plan_carry_3s.yaml").resolve()
    paths = materialize_reproduction_configs(m1, m2, tmp_path / "run")
    m1, m2 = load_yaml(paths["m1"]), load_yaml(paths["m2"])
    root = (tmp_path / "run").resolve()

    assert m1["paths"]["output_dir"] == str(root / "bars_m1")
    assert "incumbent_reference_checkpoint" not in m1["training"]
    assert m2["paths"]["output_dir"] == str(root / "bars_m2_v5")
    assert m2["training"]["incumbent_reference_checkpoint"] == str(
        root / "bars_m1" / "checkpoints" / "best_semi_markov_relational.pt"
    )
    assert Path(m2["paths"]["flow_checkpoint"]).name == "best_tail_conditional_maf.pt"
    assert m1["paths"]["sequence_cache_dir"].endswith("semi_markov_relational_full")
    assert "highd_evt_config" not in m1["paths"]
    assert load_yaml(root / "reproduction_manifest.yaml")["no_historical_bars_checkpoint_inputs"] is True

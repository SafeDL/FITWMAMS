#!/usr/bin/env python3
"""Train and evaluate BARS-M2 v5 from prepared world-model inputs.

The upstream highD/EVT, Flow and sequence-cache pipelines are intentionally
outside this command.  It creates new BARS-M1 and BARS-M2 weights only: M1 is
trained from random initialization, and M2 is initialized from that same run's
M1 rather than from a historical BARS checkpoint.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.initial_behavior_anchor import FrozenLegacyFlowSchema
from world_model.src.semi_markov_evaluation import evaluate_semi_markov_world_model
from world_model.src.semi_markov_train import train_semi_markov_world_model
from world_model.src.sequential_dataset import load_sequential_dataset, sequence_cache_owner_dir
from world_model.src.utils import load_yaml, setup_logging


STAGES = ("validate", "m1", "m2", "evaluate")


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _write_yaml(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize_reproduction_configs(
    m1_template_path: Path, m2_template_path: Path, output: Path,
) -> dict[str, Path]:
    """Write isolated M1/M2 configs containing only upstream prepared inputs."""
    m1_template_path, m2_template_path = Path(m1_template_path).resolve(), Path(m2_template_path).resolve()
    output = Path(output).resolve()
    config_dir = output / "configs"
    m1_template, m2_template = load_yaml(m1_template_path), load_yaml(m2_template_path)
    seed = int(m1_template.get("training", {}).get("seed", 42))
    device = str(m1_template.get("training", {}).get("device", "auto"))
    template_paths = dict(m1_template["paths"])
    inputs = {
        key: _resolve(template_paths[key], m1_template_path.parent)
        for key in ("sequence_cache_dir", "legacy_dataset_dir", "flow_checkpoint", "flow_schema")
    }

    def bars_config(template: dict[str, Any], name: str) -> tuple[dict[str, Any], Path]:
        cfg = deepcopy(template)
        paths = dict(cfg["paths"])
        paths.update({
            "legacy_dataset_dir": str(inputs["legacy_dataset_dir"]),
            "sequence_cache_dir": str(inputs["sequence_cache_dir"]),
            "output_dir": str(output / name),
            "flow_checkpoint": str(inputs["flow_checkpoint"]),
            "flow_schema": str(inputs["flow_schema"]),
            "flow_checkpoint_sha256": str(template_paths["flow_checkpoint_sha256"]),
            "flow_schema_sha256": str(template_paths["flow_schema_sha256"]),
        })
        # The sequence cache is already materialized, so BARS reproduction
        # must not reach back into the raw highD preprocessing pipeline.
        paths.pop("highd_evt_config", None)
        cfg["paths"] = paths
        cfg["split"] = {**dict(cfg.get("split", {})), "seed": seed}
        cfg["training"] = {**dict(cfg.get("training", {})), "seed": seed, "device": device}
        cfg["evaluation"] = {**dict(cfg.get("evaluation", {})), "seed": 123, "device": device}
        cfg["evaluation"].pop("legacy_paired_baseline_summary", None)
        cfg["evaluation"].pop("legacy_paired_long_horizon_baseline_summary", None)
        path = config_dir / f"highd_{name}.yaml"
        return cfg, path

    m1, m1_path = bars_config(m1_template, "bars_m1")
    m1["training"].pop("incumbent_reference_checkpoint", None)
    m1["evaluation"].pop("incumbent_reference_summary", None)
    _write_yaml(m1, m1_path)

    m2, m2_path = bars_config(m2_template, "bars_m2_v5")
    m1_checkpoint = output / "bars_m1" / "checkpoints" / "best_semi_markov_relational.pt"
    m2["training"]["incumbent_reference_checkpoint"] = str(m1_checkpoint)
    m2["evaluation"]["incumbent_reference_summary"] = str(output / "bars_m1" / "semi_markov_evaluation_summary.json")
    _write_yaml(m2, m2_path)

    manifest_path = output / "reproduction_manifest.yaml"
    _write_yaml({
        "source_configs": {"m1": str(m1_template_path), "m2": str(m2_template_path)}, "output_dir": str(output),
        "prepared_inputs": {key: str(value) for key, value in inputs.items()},
        "configs": {"m1": str(m1_path), "m2": str(m2_path)},
        "stages": list(STAGES), "no_historical_bars_checkpoint_inputs": True,
    }, manifest_path)
    return {"root": output, "m1": m1_path, "m2": m2_path}


def _require(path: Path, stage: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"stage {stage!r} requires a prepared input or earlier-stage artifact: {path}")


def _validate_inputs(config: dict[str, Any], config_path: Path) -> None:
    paths = config["paths"]
    checkpoint, schema_path = Path(paths["flow_checkpoint"]), Path(paths["flow_schema"])
    _require(checkpoint, "validate")
    _require(schema_path, "validate")
    schema = FrozenLegacyFlowSchema.load(schema_path)
    if schema.schema_sha256 != str(paths["flow_schema_sha256"]):
        raise ValueError("prepared Flow schema SHA-256 does not match the reproduction config")
    schema.verify_checkpoint(checkpoint, paths["flow_checkpoint_sha256"])
    owner = sequence_cache_owner_dir(config, config_dir=config_path.parent)
    arrays, manifest = load_sequential_dataset(owner)
    if bool(manifest.get("bounded_development_cache", True)):
        raise RuntimeError("BARS reproduction requires the complete prepared sequence cache")
    if not len(arrays.get("sequence_id", ())):
        raise RuntimeError("prepared BARS sequence cache is empty")


def run(m1_template_path: Path, m2_template_path: Path, output: Path, stages: tuple[str, ...]) -> dict[str, Path]:
    paths = materialize_reproduction_configs(m1_template_path, m2_template_path, output)
    selected = set(STAGES if "all" in stages else stages)
    m1, m2 = load_yaml(paths["m1"]), load_yaml(paths["m2"])
    if "validate" in selected:
        _validate_inputs(m1, paths["m1"])
    if "m1" in selected:
        _validate_inputs(m1, paths["m1"])
        train_semi_markov_world_model(m1, config_dir=paths["m1"].parent)
    if "m2" in selected:
        initial = paths["root"] / "bars_m1" / "checkpoints" / "best_semi_markov_relational.pt"
        _require(initial, "m2")
        train_semi_markov_world_model(m2, config_dir=paths["m2"].parent, initial_checkpoint=initial)
    if "evaluate" in selected:
        for cfg, path in ((m1, paths["m1"]), (m2, paths["m2"])):
            checkpoint = Path(cfg["paths"]["output_dir"]) / "checkpoints" / "best_semi_markov_relational.pt"
            _require(checkpoint, "evaluate")
            evaluate_semi_markov_world_model(cfg, config_dir=path.parent, checkpoint=checkpoint)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config_dir = Path(__file__).parent / "configs"
    parser.add_argument("--m1-config", default=str(config_dir / "highd_behavior_anchored_semi_markov.yaml"))
    parser.add_argument("--m2-config", default=str(config_dir / "highd_bars_m2_plan_carry_3s.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/highd_world_model/bars_m2_v5_reproduction"))
    parser.add_argument("--stages", nargs="+", choices=("all", *STAGES), default=["all"])
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    paths = run(Path(args.m1_config), Path(args.m2_config), Path(args.output_dir), tuple(args.stages))
    print(paths["root"] / "reproduction_manifest.yaml")


if __name__ == "__main__":
    main()

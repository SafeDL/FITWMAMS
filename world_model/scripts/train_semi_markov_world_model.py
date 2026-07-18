#!/usr/bin/env python3
"""Train and evaluate the standalone Semi-Markov World Model from prepared inputs."""
from __future__ import annotations

import argparse
from copy import deepcopy
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


STAGES = ("validate", "train", "evaluate")


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _write_yaml(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def materialize_reproduction_config(template_path: Path, output: Path) -> dict[str, Path]:
    """Write an isolated standalone-model config using only prepared inputs."""
    template_path, output = Path(template_path).resolve(), Path(output).resolve()
    template = load_yaml(template_path)
    config = deepcopy(template)
    template_paths = dict(template["paths"])
    paths = dict(config["paths"])
    for key in ("legacy_dataset_dir", "sequence_cache_dir", "flow_checkpoint", "flow_schema"):
        paths[key] = str(_resolve(template_paths[key], template_path.parent))
    paths["output_dir"] = str(output)
    paths.pop("highd_evt_config", None)
    config["paths"] = paths

    config_path = output / "configs" / "highd_semi_markov_world_model.yaml"
    _write_yaml(config, config_path)
    manifest_path = output / "reproduction_manifest.yaml"
    _write_yaml({
        "source_config": str(template_path),
        "output_dir": str(output),
        "prepared_inputs": {key: paths[key] for key in ("legacy_dataset_dir", "sequence_cache_dir", "flow_checkpoint", "flow_schema")},
        "config": str(config_path),
        "stages": list(STAGES),
        "historical_semi_markov_checkpoint_inputs": False,
    }, manifest_path)
    return {"root": output, "config": config_path}


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
        raise RuntimeError("Semi-Markov World Model reproduction requires the complete prepared sequence cache")
    if not len(arrays.get("sequence_id", ())):
        raise RuntimeError("prepared Semi-Markov World Model sequence cache is empty")


def run(
    template_path: Path, output: Path, stages: tuple[str, ...], *, initial_checkpoint: Path | None = None,
) -> dict[str, Path]:
    paths = materialize_reproduction_config(template_path, output)
    selected = set(STAGES if "all" in stages else stages)
    config = load_yaml(paths["config"])
    if "validate" in selected:
        _validate_inputs(config, paths["config"])
    if "train" in selected:
        _validate_inputs(config, paths["config"])
        train_semi_markov_world_model(
            config, config_dir=paths["config"].parent, initial_checkpoint=initial_checkpoint,
        )
    if "evaluate" in selected:
        checkpoint = paths["root"] / "checkpoints" / "best_semi_markov_relational.pt"
        _require(checkpoint, "evaluate")
        evaluate_semi_markov_world_model(config, config_dir=paths["config"].parent, checkpoint=checkpoint)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    config_dir = Path(__file__).parent / "configs"
    parser.add_argument("--config", default=str(config_dir / "highd_semi_markov_world_model.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/highd_world_model/semi_markov_world_model_reproduction"))
    parser.add_argument("--initial-checkpoint", help="Optional compatible checkpoint for a continuation or new plan heads.")
    parser.add_argument("--stages", nargs="+", choices=("all", *STAGES), default=["all"])
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    paths = run(
        Path(args.config), Path(args.output_dir), tuple(args.stages),
        initial_checkpoint=None if args.initial_checkpoint is None else Path(args.initial_checkpoint).resolve(),
    )
    print(paths["root"] / "reproduction_manifest.yaml")


if __name__ == "__main__":
    main()

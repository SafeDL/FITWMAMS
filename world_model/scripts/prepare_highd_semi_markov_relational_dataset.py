#!/usr/bin/env python3
"""Prepare one six-second dynamic-graph sequence per highD natural segment."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from world_model.src.initial_behavior_anchor import FrozenLegacyFlowSchema
from world_model.src.sequential_dataset import ensure_frozen_flow_behavior_anchor_cache, load_sequential_dataset, prepare_sequential_dataset, sequence_cache_owner_dir
from world_model.src.utils import load_yaml, setup_logging
CONFIG = Path(__file__).resolve().parent / "configs" / "highd_behavior_anchored_semi_markov.yaml"
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG)); parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--max-sequences", type=int, default=None); parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(); setup_logging(args.log_level); path = Path(args.config).resolve(); config = load_yaml(path)
    prepare_sequential_dataset(config, config_dir=path.parent, rebuild=args.rebuild, max_sequences=args.max_sequences)
    if config.get("model", {}).get("variant") == "m1":
        schema_value = config.get("paths", {}).get("flow_schema")
        if not schema_value:
            raise ValueError("M1 sequence preparation requires paths.flow_schema")
        schema_path = Path(schema_value)
        schema = FrozenLegacyFlowSchema.load(schema_path if schema_path.is_absolute() else (path.parent / schema_path).resolve())
        owner = sequence_cache_owner_dir(config, config_dir=path.parent)
        arrays, manifest = load_sequential_dataset(owner)
        ensure_frozen_flow_behavior_anchor_cache(owner, arrays, manifest, schema)
if __name__ == "__main__": main()

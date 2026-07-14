#!/usr/bin/env python3
"""Prepare one six-second dynamic-graph sequence per highD natural segment."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from world_model.src.sequential_dataset import prepare_sequential_dataset
from world_model.src.utils import load_yaml, setup_logging
CONFIG = Path(__file__).resolve().parent / "configs" / "highd_semi_markov_relational.yaml"
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG)); parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--max-sequences", type=int, default=None); parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(); setup_logging(args.log_level); path = Path(args.config).resolve()
    prepare_sequential_dataset(load_yaml(path), config_dir=path.parent, rebuild=args.rebuild, max_sequences=args.max_sequences)
if __name__ == "__main__": main()

#!/usr/bin/env python3
"""Train the highD Semi-Markov Relational Traffic World Model."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from world_model.src.semi_markov_train import train_semi_markov_world_model
from world_model.src.utils import load_yaml, setup_logging
CONFIG = Path(__file__).resolve().parent / "configs" / "highd_behavior_anchored_semi_markov.yaml"
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG)); parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rebuild-dataset", action="store_true"); parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-train-sequences", type=int, default=0); parser.add_argument("--max-val-sequences", type=int, default=0)
    parser.add_argument("--initial-checkpoint", default=None, help="Continue optimization from a compatible semi-Markov checkpoint.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(); setup_logging(args.log_level); path = Path(args.config).resolve()
    train_semi_markov_world_model(load_yaml(path), config_dir=path.parent, epochs=args.epochs, rebuild_dataset=args.rebuild_dataset,
        max_sequences=args.max_sequences, max_train_sequences=args.max_train_sequences, max_val_sequences=args.max_val_sequences,
        initial_checkpoint=args.initial_checkpoint)
if __name__ == "__main__": main()

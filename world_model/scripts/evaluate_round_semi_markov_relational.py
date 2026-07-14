#!/usr/bin/env python3
"""Evaluate a causal-prior Semi-Markov rollout on held-out rounD sequences."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.semi_markov_evaluation import evaluate_semi_markov_world_model
from world_model.src.utils import load_yaml, setup_logging

CONFIG = Path(__file__).resolve().parent / "configs" / "round_semi_markov_relational.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    path = Path(args.config).resolve()
    evaluate_semi_markov_world_model(
        load_yaml(path), config_dir=path.parent,
        checkpoint=Path(args.checkpoint).resolve() if args.checkpoint else None,
        max_sequences=args.max_sequences,
    )


if __name__ == "__main__":
    main()

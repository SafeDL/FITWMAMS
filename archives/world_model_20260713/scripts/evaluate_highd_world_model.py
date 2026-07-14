#!/usr/bin/env python3
"""评价唯一的 catk_topk 世界模型。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.evaluation import evaluate_world_model  # noqa: E402
from world_model.src.utils import load_yaml, setup_logging  # noqa: E402


CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "highd_world_model.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-branches", type=int, default=16)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    evaluate_world_model(
        config,
        config_dir=config_path.parent,
        checkpoint=Path(args.checkpoint).resolve() if args.checkpoint else None,
        max_samples=args.max_samples,
        num_branches=args.num_branches,
    )


if __name__ == "__main__":
    main()

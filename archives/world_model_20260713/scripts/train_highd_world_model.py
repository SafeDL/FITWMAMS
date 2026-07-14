#!/usr/bin/env python3
"""训练唯一的 catk_topk 世界模型。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.train import train_world_model  # noqa: E402
from world_model.src.utils import load_yaml, setup_logging  # noqa: E402


CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "highd_world_model.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    train_world_model(
        config,
        config_dir=config_path.parent,
        epochs=args.epochs if args.epochs > 0 else None,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        rebuild_dataset=bool(args.rebuild_dataset),
        dataset_max_segments=args.max_segments if args.max_segments > 0 else None,
    )


if __name__ == "__main__":
    main()

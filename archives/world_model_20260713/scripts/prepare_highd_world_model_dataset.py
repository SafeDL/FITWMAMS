#!/usr/bin/env python3
"""构建唯一世界模型实现使用的 highD START/ROLL 数据集。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.data import build_world_model_dataset  # noqa: E402
from world_model.src.utils import load_yaml, setup_logging  # noqa: E402


CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "highd_world_model.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    build_world_model_dataset(
        config,
        config_dir=config_path.parent,
        max_segments=args.max_segments if args.max_segments > 0 else None,
        rebuild=bool(args.rebuild),
    )


if __name__ == "__main__":
    main()

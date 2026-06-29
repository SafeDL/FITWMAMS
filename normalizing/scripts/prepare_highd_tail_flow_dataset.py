#!/usr/bin/env python3
"""Build the highD EVT-tail dataset used by the normalizing flow.

This single entry point also refreshes natural-tail contexts when requested.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from normalizing.src.data import build_tail_flow_dataset  # noqa: E402
from normalizing.src.utils import load_yaml, setup_logging  # noqa: E402


CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "highd_tail_flow_best.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--rebuild-tail-contexts", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    build_tail_flow_dataset(
        config,
        config_dir=config_path.parent,
        rebuild_tail_contexts=bool(args.rebuild_tail_contexts),
    )


if __name__ == "__main__":
    main()

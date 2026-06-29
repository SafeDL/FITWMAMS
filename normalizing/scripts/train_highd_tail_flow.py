#!/usr/bin/env python3
"""Train the highD EVT-tail event normalizing flow."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from normalizing.src.train import train_tail_flow  # noqa: E402
from normalizing.src.utils import load_yaml, setup_logging  # noqa: E402


CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "highd_tail_flow_best.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--tensorboard-log-dir", default=None)
    parser.add_argument("--clear-tensorboard", action="store_true")
    args = parser.parse_args()
    setup_logging(args.log_level)
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    tensorboard_cfg = dict(config.get("tensorboard", {}))
    if args.no_tensorboard:
        tensorboard_cfg["enabled"] = False
    if args.tensorboard_log_dir:
        tensorboard_cfg["log_dir"] = args.tensorboard_log_dir
    if args.clear_tensorboard:
        tensorboard_cfg["clear_existing"] = True
    config["tensorboard"] = tensorboard_cfg
    train_tail_flow(config, config_dir=config_path.parent, repo_root=ROOT)


if __name__ == "__main__":
    main()

"""Run the complete, resumable TrafficBotsV1.5-HighD experiment.

The process deliberately evaluates only after Lightning has selected a
validation-best checkpoint.  It is the unattended entry point for the full
train/test baseline, not a pilot runner.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from world_model.trafficbots.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and formally evaluate TrafficBotsV1.5-HighD")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config_path = args.config or Path("world_model/trafficbots/config/highd.yaml")
    config = load_config(config_path)
    command = [sys.executable, "-m"]
    subprocess.run([*command, "world_model.trafficbots.scripts.train", "--config", str(config_path)], check=True)
    checkpoint = Path(config["paths"]["output_dir"]) / "checkpoints" / "best.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"full training completed without a validation-best checkpoint: {checkpoint}")
    subprocess.run([
        *command, "world_model.trafficbots.scripts.evaluate", "--config", str(config_path),
        "--checkpoint", str(checkpoint),
    ], check=True)
    subprocess.run([*command, "world_model.trafficbots.scripts.verify_full", "--config", str(config_path)], check=True)


if __name__ == "__main__":
    main()

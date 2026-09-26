"""Freeze and audit the benchmark-v1 scenario manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import torch

from npc_behavior_benchmark.data.manifest import build_dataset_manifest, load_benchmark_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/benchmark_v1.yaml",
    )
    args = parser.parse_args()
    manifest = build_dataset_manifest(args.config)
    config, config_path = load_benchmark_config(args.config)
    output = Path(config["paths"]["output_dir"])
    protocols = output / "protocols"
    protocols.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, protocols / "benchmark_v1.yaml")
    split_audit = {
        "benchmark_id": config["benchmark"]["id"],
        "split_policy": config["benchmark"]["split_policy"],
        "split_summary": manifest["split_summary"],
        **manifest["recording_split_audit"],
    }
    (output / "split_audit.json").write_text(json.dumps(split_audit, indent=2) + "\n")
    try:
        git_head = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=config_path.parents[2], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = "unknown"
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "git_head_at_prepare": git_head,
        "target_source_commit": config["benchmark"]["source_commit"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    (output / "software_environment.json").write_text(
        json.dumps(environment, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "dataset_manifest": manifest,
                "split_audit": split_audit,
                "software_environment": environment,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

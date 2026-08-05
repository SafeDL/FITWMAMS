#!/usr/bin/env python3
"""Prepare HiQR-owned START metadata without modifying the QR cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.core.cli_config import materialize_config  # noqa: E402
from world_model.src.core.initial_behavior_anchor import (  # noqa: E402
    FrozenLegacyFlowSchema,
)
from world_model.src.core.sequential_dataset import (  # noqa: E402
    sequence_cache_owner_dir,
)
from world_model.src.core.utils import setup_logging  # noqa: E402
from world_model.src.hiqr.data import build_hiqr_start_sidecar  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs/highd_hiqr_world_model.yaml"),
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "results/highd_world_model/hiqr_world_model")
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    config, path = materialize_config(
        Path(args.config),
        Path(args.output_dir),
        config_name=Path(args.config).name,
        resolve_path_keys=("sequence_cache_dir", "flow_schema", "source_dataset_dir"),
    )
    schema = FrozenLegacyFlowSchema.load(config["paths"]["flow_schema"])
    print(
        build_hiqr_start_sidecar(
            cache_owner=sequence_cache_owner_dir(config, config_dir=path.parent),
            output_dir=config["paths"]["output_dir"],
            flow_schema=schema,
            source_dataset_dir=config["paths"]["source_dataset_dir"],
        )
    )


if __name__ == "__main__":
    main()

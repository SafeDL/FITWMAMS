#!/usr/bin/env python3
"""Evaluate HiQR-v2 with Normalizing-Flow starts and replayed ADS controls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.core.cli_config import materialize_config  # noqa: E402
from world_model.src.core.sequential_dataset import (  # noqa: E402
    sequence_cache_owner_dir,
)
from world_model.src.core.utils import select_device, setup_logging  # noqa: E402
from world_model.src.hiqr_v2.flow_evaluation import (  # noqa: E402
    evaluate_hiqr_v2_flow_ads,
)
from world_model.src.hiqr_v2.train import (  # noqa: E402
    load_hiqr_v2_checkpoint,
    require_canonical_hiqr_v2_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs/highd_hiqr_v2_world_model.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results/highd_world_model/hiqr_v2_world_model"),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-starts", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--worlds-per-start", type=int, default=1)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    config, path = materialize_config(
        Path(args.config),
        Path(args.output_dir),
        config_name=Path(args.config).name,
        resolve_path_keys=(
            "sequence_cache_dir",
            "flow_schema",
            "source_dataset_dir",
            "v1_sidecar_output_dir",
        ),
    )
    output = Path(config["paths"]["output_dir"])
    checkpoint = args.checkpoint or (output / "checkpoints/best_hiqr_v2_world_model.pt")
    model = load_hiqr_v2_checkpoint(
        checkpoint.resolve(), device=select_device(str(config["evaluation"].get("device", "auto")))
    )
    require_canonical_hiqr_v2_checkpoint(model)
    evaluate_hiqr_v2_flow_ads(
        model,
        repo_root=ROOT,
        cache_owner=sequence_cache_owner_dir(config, config_dir=path.parent),
        output_dir=output / "flow_ads_evaluation",
        max_starts=args.max_starts,
        deterministic=not args.stochastic,
        worlds_per_start=args.worlds_per_start,
    )


if __name__ == "__main__":
    main()

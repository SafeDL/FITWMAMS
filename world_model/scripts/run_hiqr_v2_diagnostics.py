#!/usr/bin/env python3
"""Plan or run fixed-cohort, 12-epoch HiQR-v2 causal ablations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.core.cli_config import materialize_config  # noqa: E402
from world_model.src.core.utils import setup_logging  # noqa: E402
from world_model.src.hiqr_v2.diagnostics import (
    DIAGNOSTIC_VARIANTS,
    run_hiqr_v2_diagnostics,
)  # noqa: E402


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
    parser.add_argument(
        "--v1-checkpoint",
        type=Path,
        default=ROOT
        / "results/highd_world_model/hiqr_world_model/checkpoints/last_training_state.pt",
    )
    parser.add_argument(
        "--variant", choices=["all", *DIAGNOSTIC_VARIANTS], default="all"
    )
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument(
        "--run-training",
        action="store_true",
        help="execute the six isolated 12-epoch runs; omitted means plan and V1 baseline only",
    )
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
    run_hiqr_v2_diagnostics(
        config,
        config_dir=path.parent,
        v1_checkpoint=args.v1_checkpoint.resolve(),
        variants=args.variant,
        max_sequences=args.max_sequences,
        train_variants=args.run_training,
    )


if __name__ == "__main__":
    main()

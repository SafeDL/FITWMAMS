#!/usr/bin/env python3
"""Generate the all-highD-EVT-tail Flow × QR-WM composition study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.core.utils import save_json, setup_logging
from world_model.src.qr.flow_evaluation import evaluate_flow_composition
from world_model.scripts.plot_reconstruction_result_summaries import (
    plot_tail_interaction_distribution,
    plot_tail_sampling_and_runtime,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "results/highd_world_model/qr_world_model/checkpoints/best_qr_world_model.pt"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results/highd_world_model/long_tail_reproduction"),
    )
    parser.add_argument(
        "--flow-start-batch-size", type=int, default=96,
        help="Independent Flow starts evaluated together (four QR futures per start).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    report = evaluate_flow_composition(
        checkpoint=checkpoint,
        output_dir=output_dir,
        flow_start_batch_size=args.flow_start_batch_size,
    )
    figures = [
        plot_tail_interaction_distribution(output_dir),
        plot_tail_sampling_and_runtime(output_dir),
    ]
    save_json(
        {
            "study": "all-highD EVT-tail Flow × QR-WM end-to-end composition",
            "protocol": report["protocol"],
            "artifacts": {
                "evaluation": "flow_composition_evaluation.json",
                "start_audit": "flow_start_audit.npz",
                "figures": [str(path.relative_to(output_dir)) for path in figures],
            },
            "comparison_target": (
                "Compare interaction-feature distributions between all highD EVT-tail "
                "futures and Flow × QR-WM synthetic futures; this is not a paired trajectory reconstruction."
            ),
        },
        output_dir / "study_manifest.json",
    )


if __name__ == "__main__":
    main()

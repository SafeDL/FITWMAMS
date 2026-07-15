#!/usr/bin/env python3
"""Evaluate frozen highD Flow samples through a trained M1 world model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.flow_end_to_end_evaluation import evaluate_frozen_flow_end_to_end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--flow-samples", default=str(ROOT / "results/highd_tail_flow_best/samples/generated_samples.npz"))
    parser.add_argument("--flow-schema", default=str(ROOT / "results/highd_tail_flow_best/dataset_schema.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=256)
    args = parser.parse_args()
    evaluate_frozen_flow_end_to_end(
        checkpoint=args.checkpoint, flow_samples=args.flow_samples, flow_schema=args.flow_schema,
        output_path=args.output, max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

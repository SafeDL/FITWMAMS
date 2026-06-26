#!/usr/bin/env python3
"""Convert the reference SAIRL TensorFlow checkpoint to PyTorch/NPZ weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.utils import setup_logging
from SAIRL_subset.policies.sairl_policy import convert_tensorflow_checkpoint_to_npz


DEFAULT_TF_CHECKPOINT_PATH = (
    ROOT
    / "ref_code"
    / "Safe_imitation_learning-master"
    / "trained_models"
    / "SAIRL"
    / "model_1"
    / "195model.ckpt"
)
DEFAULT_OUTPUT_PATH = (
    ROOT / "SAIRL_subset" / "weights" / "sairl" / "model.npz"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        default=str(DEFAULT_TF_CHECKPOINT_PATH),
        help="TensorFlow checkpoint prefix, without .index/.data suffix.",
    )
    parser.add_argument(
        "--output_path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Converted NPZ output path.",
    )
    args = parser.parse_args()
    setup_logging("INFO")
    output = convert_tensorflow_checkpoint_to_npz(
        args.checkpoint_path,
        args.output_path,
    )
    print(output)


if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse
from dynamic_ar_idm.data import prepare_full_highd

parser = argparse.ArgumentParser(description="Prepare all qualifying highD following pairs at native 25 Hz")
parser.add_argument("--raw-dir", default="highD_dataset/Matlab/data")
parser.add_argument("--output-dir", default="dynamic_ar_idm/artifacts/full_data")
parser.add_argument("--minimum-seconds", type=float, default=50.)
args = parser.parse_args()
print(prepare_full_highd(args.raw_dir, args.output_dir, minimum_seconds=args.minimum_seconds))

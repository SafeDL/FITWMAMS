from __future__ import annotations
import argparse
from dynamic_ar_idm.data import prepare_paper_highd

parser = argparse.ArgumentParser(description="Prepare published highD cohort at native 25 Hz")
parser.add_argument("--raw-dir", default="highD_dataset/Matlab/data")
parser.add_argument("--output-dir", default="dynamic_ar_idm/artifacts/data")
args = parser.parse_args()
print(prepare_paper_highd(args.raw_dir, args.output_dir))

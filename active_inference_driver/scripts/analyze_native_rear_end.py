"""Run the authors' released rear-end analysis and plotting script externally."""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    environment = {**os.environ, "PYTHONPATH": str(args.source_dir)}
    subprocess.run([str(args.python), str(args.source_dir / "Analysis_rear_end.py")],
                   cwd=args.results_dir, env=environment, check=True)


if __name__ == "__main__":
    main()

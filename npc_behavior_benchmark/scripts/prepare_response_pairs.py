"""Build Idea A's train-only MA-IDM response-difference cache."""

from __future__ import annotations
import argparse, json
from pathlib import Path
from npc_behavior_benchmark.data.response_pairs import build_response_pair_cache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/benchmark_v1.yaml",
    )
    p.add_argument("--teacher-futures", type=int, default=4)
    a = p.parse_args()
    print(json.dumps(build_response_pair_cache(a.config, a.teacher_futures), indent=2))


if __name__ == "__main__":
    main()

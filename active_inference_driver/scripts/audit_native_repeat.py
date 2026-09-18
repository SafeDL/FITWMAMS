"""Check same-backend, same-seed native trajectory determinism."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np


ARRAYS = ("eta", "o", "b", "w", "a_disc", "a_cont", "a_cont_init", "v_init", "v")


def _load(run_dir: Path, experiment: int) -> dict[str, np.ndarray]:
    path = run_dir / "Results_rear_end" / f"Exp_{experiment}" / f"Exp_{experiment}.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", type=int, default=0)
    args = parser.parse_args()
    first, repeat = _load(args.first, args.experiment), _load(args.repeat, args.experiment)
    differences = {}
    for name in ARRAYS:
        if first[name].shape != repeat[name].shape:
            raise RuntimeError(f"{name}: shape mismatch")
        differences[name] = float(np.max(np.abs(first[name] - repeat[name])))
    maximum = max(differences.values())
    report = {
        "experiment": args.experiment,
        "same_backend_same_seed": True,
        "tolerance": 1e-5,
        "maximum_abs_trajectory_difference": maximum,
        "passes": maximum <= 1e-5,
        "max_abs_by_array": differences,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Compare locally executed native rear-end trajectories to the OSF archive."""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import pickle
from zipfile import ZipFile

import numpy as np


ARRAYS = ("eta", "o", "b", "w", "a_disc", "a_cont", "a_cont_init", "v_init", "v")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-results", type=Path, required=True)
    parser.add_argument("--official-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiments", type=int, default=28)
    args = parser.parse_args()
    records = []
    with ZipFile(args.official_archive) as archive:
        for index in range(args.experiments):
            local_path = args.local_results / "Results_rear_end" / f"Exp_{index}" / f"Exp_{index}.pkl"
            with local_path.open("rb") as handle:
                local = pickle.load(handle)
            official_member = f"Results_rear_end/Exp_{index}/Exp_{index}.pkl"
            official = pickle.load(BytesIO(archive.read(official_member)))
            errors = {}
            for name in ARRAYS:
                if local[name].shape != official[name].shape:
                    raise RuntimeError(f"Exp {index} {name}: shape mismatch")
                errors[name] = float(np.max(np.abs(local[name] - official[name])))
            records.append({"experiment": index, "max_abs_by_array": errors,
                            "max_abs": max(errors.values())})
    maximum = max(record["max_abs"] for record in records)
    report = {"experiments": args.experiments, "tolerance": 1e-5,
              "maximum_abs_trajectory_difference": maximum,
              "passes_same_backend_gate": maximum <= 1e-5, "per_experiment": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "per_experiment"}, indent=2))


if __name__ == "__main__":
    main()

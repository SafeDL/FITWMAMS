"""Export a repaired checkpoint and its native nuPlan handoff manifest."""

import argparse
import json
from pathlib import Path

from ads.common.instances import file_hash
from ads.common.reporting import write_json
from ads.common.setup import load_policy
from ads.nuplan.assets import ADS, PUBLISHED, official_path


def export_policy(algorithm, source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    initial = (
        (ADS / "carl/artifacts/official.pth")
        if algorithm == "carl"
        else official_path(algorithm)
    )
    initial = initial.resolve()
    if source == destination or destination.is_relative_to(initial.parent):
        raise ValueError(
            "native export must preserve the training checkpoint and official artifact directory"
        )
    load_policy(algorithm, source, device="cpu").export_native(destination)
    sidecar = destination.with_name(destination.name + ".source.json")
    record = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    record.update(
        algorithm=algorithm,
        source_checkpoint=str(source),
        source_sha256=file_hash(source),
        export_sha256=file_hash(destination),
        format="official_native_checkpoint",
    )
    write_json(sidecar, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("algorithm", choices=("carl", *PUBLISHED))
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            export_policy(args.algorithm, args.source, args.destination), indent=2
        )
    )


if __name__ == "__main__":
    main()

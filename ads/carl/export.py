"""Export a trained CaRL policy for the unchanged official nuPlan loader."""

import argparse
from pathlib import Path

from ads.carl.policy import CaRLPolicy
from ads.common.instances import file_hash
from ads.common.reporting import write_json


def export_policy(checkpoint, output):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    official = (Path(__file__).parent / "artifacts/official.pth").resolve()
    if checkpoint == output or output.is_relative_to(official.parent):
        raise ValueError(
            "native export must preserve the training checkpoint and official artifact directory"
        )
    model = CaRLPolicy.load(checkpoint)
    model.export_native(output)
    record = {
        "format": "nuplan_carl_state_dict",
        "source_checkpoint": str(checkpoint),
        "source_sha256": file_hash(checkpoint),
        "export_sha256": file_hash(output),
    }
    write_json(output.with_suffix(".json"), record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(export_policy(args.checkpoint, args.output))


if __name__ == "__main__":
    main()

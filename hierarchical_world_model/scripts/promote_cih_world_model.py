#!/usr/bin/env python3
"""Create the final CIH-WM acceptance record from validation and test evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"required artifact is absent: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("acceptance artifact is immutable and will not be overwritten")
    validation, test = read(args.validation), read(args.test)
    same_checkpoint = (
        Path(str(validation.get("checkpoint", ""))).resolve()
        == Path(str(test.get("checkpoint", ""))).resolve()
        and bool(validation.get("checkpoint_sha256"))
        and validation.get("checkpoint_sha256") == test.get("checkpoint_sha256")
    )
    provenance = test.get("training_provenance", {})
    passed = bool(
        validation.get("split") == "validation" and validation.get("complete_split") is True
        and validation.get("accepted") is True and validation.get("evaluation_passed") is True
        and test.get("split") == "test" and test.get("complete_split") is True
        and test.get("evaluation_passed") is True
        and validation.get("human_response_distribution", {}).get("passed") is True
        and test.get("human_response_distribution", {}).get("passed") is True
        and provenance.get("passed") is True and same_checkpoint
    )
    record = {
        "schema": "cih_world_model_acceptance",
        "accepted": passed,
        "candidate": test.get("checkpoint"),
        "validation_artifact": str(args.validation.resolve()),
        "test_artifact": str(args.test.resolve()),
        "training_provenance": provenance,
        "gates": {
            "training_provenance": provenance.get("passed") is True,
            "complete_accepted_validation": validation.get("accepted") is True and validation.get("evaluation_passed") is True,
            "complete_one_shot_test": test.get("evaluation_passed") is True,
            "same_checkpoint": same_checkpoint,
            "human_sequence_distribution": (
                validation.get("human_response_distribution", {}).get("passed") is True
                and test.get("human_response_distribution", {}).get("passed") is True
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

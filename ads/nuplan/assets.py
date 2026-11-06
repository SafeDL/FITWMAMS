"""Pinned published nuPlan checkpoints, configurations and their provenance."""

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

from ads.common.instances import file_hash

ADS = Path(__file__).resolve().parents[1]
PUBLISHED = {
    "diffusion_planner": {
        "checkpoint": "model.pth",
        "files": {
            "model.pth": (
                "https://huggingface.co/ZhengYinan2001/Diffusion-Planner/resolve/main/model.pth",
                "7a441df91ebe1c912d8262010c40486da24f425f757e2b4228072e251ab67d45",
            ),
            "args.json": (
                "https://huggingface.co/ZhengYinan2001/Diffusion-Planner/resolve/main/args.json",
                "7e62b89a50953f133d55484777e54490f7f24e58feec1efcf696bcc7b91bdf10",
            ),
        },
    },
}


def official_path(algorithm):
    return ADS / algorithm / "artifacts" / PUBLISHED[algorithm]["checkpoint"]


def install(algorithm):
    directory = ADS / algorithm / "artifacts"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (url, expected) in PUBLISHED[algorithm]["files"].items():
        target = directory / name
        if not target.exists():
            temporary = target.with_name(target.name + ".part")
            with (
                urllib.request.urlopen(url, timeout=180) as source,
                temporary.open("wb") as destination,
            ):
                shutil.copyfileobj(source, destination, 4 * 1024 * 1024)
            temporary.replace(target)
        digest = file_hash(target)
        if expected is not None and digest != expected:
            raise ValueError(
                f"published checkpoint/configuration hash differs: {target}"
            )
        records.append(
            {
                "path": name,
                "official_url": url,
                "sha256": digest,
                "bytes": target.stat().st_size,
            }
        )
    record = {
        "algorithm": algorithm,
        "files": records,
        "source": json.loads((ADS / algorithm / "vendor/SOURCE.json").read_text()),
        "checkpoint_sha256": file_hash(official_path(algorithm)),
    }
    (directory / "official.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("algorithm", choices=tuple(PUBLISHED))
    args = parser.parse_args()
    print(json.dumps(install(args.algorithm), indent=2))


if __name__ == "__main__":
    main()

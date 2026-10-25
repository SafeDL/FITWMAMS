"""Pinned official policy weights; no external evaluation data is downloaded."""

import json
import shutil
import urllib.request
from pathlib import Path

from ads.common.instances import file_hash

ROOT = Path(__file__).resolve().parent
CARL_COMMIT = "2677d1477193ecb5eb41ad8d7e4eda78057fd8a9"
CARL_VARIANT = "nuplan_51892_1B"
PLANT_REVISION = "e81b261f8b92b20225d020fe9b7a0470c1434d96"


def official_assets():
    assets = [
        (
            ROOT / "carl/artifacts/official.pth",
            f"https://raw.githubusercontent.com/autonomousvision/CaRL/{CARL_COMMIT}/nuPlan/checkpoints/{CARL_VARIANT}/model_best.pth",
        )
    ]
    assets.append(
        (
            ROOT / "plant2/artifacts/official_1.ckpt",
            f"https://huggingface.co/SimonGer/PlanT2/resolve/{PLANT_REVISION}/epoch%3D029_final_1.ckpt",
        )
    )
    return assets


def prepare_official_assets():
    records = []
    for path, url in official_assets():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            print(f"Downloading {path.relative_to(ROOT)}", flush=True)
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                temporary.open("wb") as stream,
            ):
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            temporary.rename(path)
        record = {"url": url, "sha256": file_hash(path), "bytes": path.stat().st_size}
        provenance = path.with_suffix(".json")
        if provenance.exists() and json.loads(provenance.read_text()) != record:
            raise ValueError(f"official artifact changed: {path}")
        provenance.write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
    return records


if __name__ == "__main__":
    prepare_official_assets()

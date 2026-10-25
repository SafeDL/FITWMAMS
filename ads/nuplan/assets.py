"""Pinned published nuPlan checkpoints, configurations and their provenance."""

import argparse
import json
import shutil
import tarfile
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
    "dtpp": {
        "checkpoint": "base_model.pth",
        "files": {
            "base_model.pth": (
                "https://raw.githubusercontent.com/MCZhi/DTPP/4d721b634c7e0e5b6c6be867c1f8cc456d5d11b7/base_model.pth",
                "aba84322e6e26ca4a3f33133a4feb43901e7c666d2db17234255cc832a49c073",
            )
        },
    },
    "flow_planner": {
        "checkpoint": "model.pth",
        "files": {
            "model.pth": (
                "https://huggingface.co/ttwhy/flow-planner/resolve/main/model.pth",
                "9231b4c02968c9e42dd2881ad3e7dd4bd362796f7b2c0158e0458c4ab3bde88e",
            ),
            "model_config.yaml": (
                "https://huggingface.co/ttwhy/flow-planner/resolve/main/model_config.yaml",
                "2914c4f174e7fba8baf9ee7d42300c640a285f9d6fa13d36b78f0a3cbb9a03fe",
            ),
        },
    },
    "str2": {
        "checkpoint": "checkpoint-297000",
        "files": {
            "str_open.tar.gz": (
                "https://huggingface.co/JohnZhan/StateTransformer2/resolve/main/str_open.tar.gz",
                "0d346779bb1910f2cfd82c265237a45fb318ae0956c4cf323bb684a08ce0a849",
            )
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
    if algorithm == "str2" and not official_path(algorithm).exists():
        with tarfile.open(directory / "str_open.tar.gz") as archive:
            for member in archive:
                parts = Path(member.name).parts
                if "checkpoint-297000" not in parts or not member.isfile():
                    continue
                relative = Path(*parts[parts.index("checkpoint-297000") :])
                if relative.name in (
                    "optimizer.pt",
                    "scheduler.pt",
                    "training_args.bin",
                    "trainer_state.json",
                ) or relative.name.startswith("rng_state"):
                    continue
                target = directory / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with (
                    archive.extractfile(member) as source,
                    target.open("wb") as destination,
                ):
                    shutil.copyfileobj(source, destination, 4 * 1024 * 1024)
    if algorithm == "str2":
        vocabulary = official_path(algorithm) / "kmeans_traj_8s_512.csv"
        url = "https://raw.githubusercontent.com/Tsinghua-MARS-Lab/StateTransformer/b82f9bcf5c9056d0fc5afe9da3350d9bd1c5a9c5/kmeans_traj_8s_512.csv"
        if not vocabulary.exists():
            urllib.request.urlretrieve(url, vocabulary)
        if (
            file_hash(vocabulary)
            != "4fef7267cd8ce31b07f27effaaed3d530de8384f4b50c438fb16d3d97ea686b2"
        ):
            raise ValueError(
                "STR2 trajectory vocabulary differs from the published file"
            )
        records.append(
            {
                "path": str(vocabulary.relative_to(directory)),
                "official_url": url,
                "sha256": file_hash(vocabulary),
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

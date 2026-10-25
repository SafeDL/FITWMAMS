"""Plan and fetch a small official Train subset without downloading ZIP archives."""

import argparse
import hashlib
import io
import json
import random
import shutil
import urllib.parse
import urllib.request
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

WORKSPACE = Path("/media/hp/DATA/nuplan_workspace")
MANIFEST = WORKSPACE / "preparation/nuplan_train_source_pool.json"
ARCHIVE_ROOT = (
    "https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1"
)
ARCHIVES = ("boston", "pittsburgh", "singapore", *(f"vegas_{i}" for i in range(1, 7)))
SEED = 20260926


class RangeFile(io.RawIOBase):
    """Seek in a remote ZIP using bounded, verified HTTP range responses."""

    def __init__(self, url):
        self.url = url
        self.position = 0
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=30) as response:
            self.size = int(response.headers["Content-Length"])
            self.etag = response.headers["ETag"]

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        origins = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        self.position = origins[whence] + offset
        if self.position < 0:
            raise ValueError("negative HTTP file position")
        return self.position

    def read(self, size=-1):
        remaining = max(0, self.size - self.position)
        size = remaining if size < 0 else min(size, remaining)
        if not size:
            return b""
        if size > 8 * 1024**2:
            raise ValueError("HTTP ZIP reads must stay below 8 MiB")
        start, end = self.position, self.position + size - 1
        # Separate cache entries prevent proxies from reusing a different range.
        query = urllib.parse.urlencode({"fitwmams_range": f"{start}-{end}"})
        request = urllib.request.Request(
            self.url + "?" + query,
            headers={"Range": f"bytes={start}-{end}"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            expected = f"bytes {start}-{end}/{self.size}"
            if response.status != 206 or response.headers["Content-Range"] != expected:
                raise ValueError("server did not return the requested byte range")
            data = response.read(size + 1)
        if len(data) != size:
            raise ValueError("incomplete HTTP range response")
        self.position += size
        return data


def session_id(log_name):
    return log_name.rsplit("_", 2)[0]


def read_archive(part):
    url = f"{ARCHIVE_ROOT}/nuplan-v1.1_train_{part}.zip"
    city = "las_vegas" if part.startswith("vegas_") else part
    records = []
    with RangeFile(url) as source, zipfile.ZipFile(source) as archive:
        for member in sorted(archive.infolist(), key=lambda item: item.filename):
            if member.filename.endswith(".db"):
                records.append(
                    {
                        "city": city,
                        "archive": part,
                        "session": session_id(Path(member.filename).stem),
                        "log": Path(member.filename).stem,
                        "member": member.filename,
                        "bytes": member.file_size,
                        "compressed_bytes": member.compress_size,
                        "crc32": member.CRC,
                    }
                )
        info = {
            "part": part,
            "city": city,
            "url": url,
            "bytes": source.size,
            "etag": source.etag,
        }
    return info, records


def make_plan(budget_gib=100):
    if budget_gib <= 0:
        raise ValueError("storage budget must be positive")
    splitter = WORKSPACE / (
        "nuplan-devkit/nuplan/planning/script/config/common/splitter/nuplan.yaml"
    )
    splits = yaml.safe_load(splitter.read_text())["log_splits"]
    train = set(splits["train"])
    excluded = {session_id(name) for name in splits["val"] + splits["test"]}
    archives, by_city = [], defaultdict(list)
    with ThreadPoolExecutor(max_workers=4) as workers:
        for info, records in workers.map(read_archive, ARCHIVES):
            archives.append(info)
            by_city[info["city"]].extend(r for r in records if r["log"] in train)
    raw_budget = int(budget_gib * 1024**3 / 2)
    total_source_logs = sum(len(rows) for rows in by_city.values())
    rng = random.Random(SEED)
    selected = []
    for city in sorted(by_city):
        groups = defaultdict(list)
        for row in by_city[city]:
            if row["session"] not in excluded:
                groups[row["session"]].append(row)
        sessions = sorted(groups)
        rng.shuffle(sessions)
        city_budget = raw_budget * len(by_city[city]) // total_source_logs
        city_rows, used = [], 0
        for session in sessions:
            row = dict(rng.choice(groups[session]))
            if used + row["bytes"] > city_budget:
                continue
            # Separate driving sessions, not adjacent windows, define the split.
            row["purpose"] = "development" if len(city_rows) % 5 == 0 else "replay"
            city_rows.append(row)
            used += row["bytes"]
        if len(city_rows) < 5:
            raise ValueError(
                f"{city}: budget cannot cover separate replay/development sessions"
            )
        excluded.update(row["session"] for row in city_rows)
        selected.extend(city_rows)
        print(
            f"{city}: {len(city_rows)} source sessions, {used / 1024**3:.2f} GiB",
            flush=True,
        )
    plan = {
        "schema": "nuplan_train_source_pool_v2",
        "seed": SEED,
        "storage_budget_bytes": int(budget_gib * 1024**3),
        "raw_budget_bytes": raw_budget,
        "cache_and_temporary_reserve_bytes": int(budget_gib * 1024**3) - raw_budget,
        "splitter_sha256": hashlib.sha256(splitter.read_bytes()).hexdigest(),
        "archives": archives,
        "logs": selected,
        "raw_bytes": sum(row["bytes"] for row in selected),
        "compressed_bytes": sum(row["compressed_bytes"] for row in selected),
        "scope": "Candidate source pool; scenario types and temporal coverage still require preprocessing",
        "selection": "Train city proportions, shuffled separate sessions; no per-file size preference",
    }
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != plan:
        roots = WORKSPACE / "data/nuplan-v1.1/splits"
        if any(
            any((roots / name).glob("*.db"))
            for name in ("train_replay", "train_development")
        ):
            raise ValueError(
                "source DBs already exist; account for their storage before replanning"
            )
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(plan, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: plan[key]
                for key in ("raw_bytes", "compressed_bytes", "storage_budget_bytes")
            }
        )
    )
    print(f"Source plan saved: {MANIFEST}; no DB files downloaded.")
    return plan


def file_crc(path):
    crc = 0
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024**2), b""):
            crc = zlib.crc32(data, crc)
    return crc


def download():
    plan = json.loads(MANIFEST.read_text())
    root = WORKSPACE / "data/nuplan-v1.1/splits"
    for info in plan["archives"]:
        with RangeFile(info["url"]) as source, zipfile.ZipFile(source) as archive:
            if source.size != info["bytes"] or source.etag != info["etag"]:
                raise ValueError(
                    "official archive changed since the subset was planned"
                )
            for row in (r for r in plan["logs"] if r["archive"] == info["part"]):
                directory = root / ("train_" + row["purpose"])
                directory.mkdir(exist_ok=True)
                destination = directory / (row["log"] + ".db")
                if destination.exists():
                    if destination.stat().st_size != row["bytes"] or (
                        file_crc(destination) != row["crc32"]
                    ):
                        raise ValueError(
                            f"existing DB failed verification: {destination}"
                        )
                    print(f"Verified existing: {destination.name}", flush=True)
                    continue
                member = archive.getinfo(row["member"])
                if (member.file_size, member.compress_size, member.CRC) != (
                    row["bytes"],
                    row["compressed_bytes"],
                    row["crc32"],
                ):
                    raise ValueError("ZIP member differs from the subset manifest")
                temporary = destination.with_suffix(".db.part")
                with archive.open(member) as stream, temporary.open("wb") as output:
                    shutil.copyfileobj(stream, output, 4 * 1024**2)
                if temporary.stat().st_size != row["bytes"]:
                    raise ValueError("extracted DB length differs")
                temporary.replace(destination)
                print(f"Downloaded: {row['purpose']}/{destination.name}", flush=True)
    replay = sum(row["purpose"] == "replay" for row in plan["logs"])
    development = len(plan["logs"]) - replay
    print(
        f"Prepared {replay} replay DBs and {development} development DBs; Val unchanged."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "download"))
    parser.add_argument(
        "--budget-gib", type=float, default=100, help="total raw/cache storage budget"
    )
    args = parser.parse_args()
    if args.action == "plan":
        make_plan(args.budget_gib)
    else:
        download()


if __name__ == "__main__":
    main()

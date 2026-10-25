"""Select isolated native Train scenes and encode them with the official processor."""

import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ads.common.instances import file_hash, json_hash
from ads.common.reporting import write_json
from ads.nuplan.prepare_data import SEED, WORKSPACE, session_id
from ads.nuplan.verify_data import REPORT

ROOT = WORKSPACE / "preparation/retention"
MANIFEST = ROOT / "scenes.json"
COUNTS = {"replay": 2000, "development": 200}
CONFIG = Path(__file__).resolve().parent / "artifacts/args.json"


def scene_candidates(record):
    """Non-overlapping 2 s history / 8 s future windows, independent of ADS risk."""
    with sqlite3.connect(Path(record["path"]).as_uri() + "?mode=ro", uri=True) as db:
        frames = db.execute(
            "SELECT token,timestamp FROM lidar_pc ORDER BY timestamp"
        ).fetchall()
        times = np.asarray([row[1] for row in frames], np.int64)
        offset = (
            int(hashlib.sha256(record["log"].encode()).hexdigest()[:8], 16) % 1_000_000
        )
        anchors = np.arange(
            times[0] + 2_100_000 + offset, times[-1] - 8_100_000, 12_000_000
        )
        candidates = []
        for stamp in anchors:
            token, timestamp = frames[int(np.searchsorted(times, stamp))]
            tags = sorted(
                row[0]
                for row in db.execute(
                    "SELECT type FROM scenario_tag WHERE lidar_pc_token=?", (token,)
                )
            )
            candidates.append(
                {
                    key: record[key]
                    for key in ("log", "city", "purpose", "path", "map_name", "sha256")
                }
                | {
                    "token": token.hex(),
                    "timestamp": timestamp,
                    "tags": tags,
                    "session": session_id(record["log"]),
                }
            )
    rng = np.random.default_rng(int(json_hash([SEED, record["log"]])[:16], 16))
    rng.shuffle(candidates)
    return candidates


def select_scenes():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    verification = json.loads(REPORT.read_text())
    if verification["status"] != "passed":
        raise ValueError("nuPlan Train source verification must pass first")
    groups = defaultdict(lambda: defaultdict(list))
    for record in verification["logs"]:
        groups[record["purpose"]][record["city"]].append(record)
    output = {}
    for purpose, count in COUNTS.items():
        cities = groups[purpose]
        total = sum(map(len, cities.values()))
        weights = {city: count * len(logs) / total for city, logs in cities.items()}
        quotas = {city: int(value) for city, value in weights.items()}
        for city in sorted(cities, key=lambda c: (-(weights[c] - quotas[c]), c))[
            : count - sum(quotas.values())
        ]:
            quotas[city] += 1
        selected, reserve = [], []
        for city, records in sorted(cities.items()):
            queues = deque(
                deque(scene_candidates(row))
                for row in sorted(records, key=lambda r: r["log"])
            )
            candidates = []
            while queues:
                queue = queues.popleft()
                if queue:
                    candidates.append(queue.popleft())
                if queue:
                    queues.append(queue)
            selected.extend(candidates[: quotas[city]])
            reserve.extend(candidates[quotas[city] :])
            print(
                f"Selected {purpose}/{city}: {quotas[city]}, reserve {len(candidates) - quotas[city]}",
                flush=True,
            )
        if len(selected) != count:
            raise ValueError("Insufficient distinct temporal windows")
        output[purpose] = {"selected": selected, "reserve": reserve}
    sessions = [{row["session"] for row in output[p]["selected"]} for p in COUNTS]
    if sessions[0] & sessions[1]:
        raise ValueError("Retention and development sessions overlap")
    manifest = {
        "seed": SEED,
        "source_verification_sha256": file_hash(REPORT),
        "selection": "city proportions, log round-robin, seeded windows separated by 12 s; no ADS scoring",
        "scenes": output,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(MANIFEST, manifest)
    return manifest


def native_scenario(row):
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import (
        NuPlanScenario,
    )
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
        ScenarioExtractionInfo,
    )

    return NuPlanScenario(
        data_root=str(WORKSPACE / "data"),
        log_file_load_path=row["path"],
        initial_lidar_token=row["token"],
        initial_lidar_timestamp=row["timestamp"],
        scenario_type="retention",
        map_root=str(WORKSPACE / "data/maps"),
        map_version="nuplan-maps-v1.0",
        map_name=row["map_name"],
        scenario_extraction_info=ScenarioExtractionInfo(
            scenario_duration=10.0, subsample_ratio=0.5
        ),
        ego_vehicle_parameters=get_pacifica_parameters(),
    )


def encode_scene(row):
    from ads.diffusion_planner.vendor.diffusion_planner.data_process.data_processor import (
        DataProcessor,
    )

    directory = ROOT / "diffusion_planner" / row["purpose"]
    directory.mkdir(parents=True, exist_ok=True)
    identity = json_hash([row["log"], row["token"]])
    path = directory / (identity + ".npz")
    if path.exists():
        return {
            "scene_id": identity,
            "path": str(path),
            "sha256": file_hash(path),
            "source": row,
        }

    class Capture(DataProcessor):
        def save_to_disk(self, directory, data):
            self.data = data

    processor = Capture(SimpleNamespace(**json.loads(CONFIG.read_text())))
    try:
        scenario = native_scenario(row)
        processor.work([scenario])
        data = processor.data
        if data["ego_agent_future"].shape != (80, 3):
            raise ValueError("Incomplete ego future")
        if not np.any(data["route_lanes"]):
            raise ValueError("No valid native route")
        for value in data.values():
            if isinstance(value, np.ndarray) and not np.isfinite(value).all():
                raise ValueError("Non-finite native feature")
    except (ValueError, RuntimeError, IndexError, AssertionError) as error:
        return {
            "scene_id": identity,
            "source": row,
            "excluded": type(error).__name__ + ": " + str(error),
        }
    np.savez_compressed(path, **data)
    return {
        "scene_id": identity,
        "path": str(path),
        "sha256": file_hash(path),
        "source": row,
    }


def prepare(workers=4):
    manifest = select_scenes()
    report_path = ROOT / "diffusion_planner/materials.json"
    completed = {purpose: [] for purpose in COUNTS}
    excluded = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for purpose, count in COUNTS.items():
            chosen = manifest["scenes"][purpose]["selected"]
            for index, result in enumerate(
                executor.map(encode_scene, chosen, chunksize=1), 1
            ):
                if "excluded" in result:
                    excluded.append(result)
                else:
                    completed[purpose].append(result)
                if index % 25 == 0 or index == len(chosen):
                    print(
                        f"Native DP {purpose}: {index}/{count}, valid {len(completed[purpose])}",
                        flush=True,
                    )
            needs = Counter(
                row["source"]["city"]
                for row in excluded
                if row["source"]["purpose"] == purpose
            )
            for row in manifest["scenes"][purpose]["reserve"]:
                if not needs[row["city"]]:
                    continue
                result = executor.submit(encode_scene, row).result()
                if "excluded" in result:
                    excluded.append(result)
                else:
                    completed[purpose].append(result)
                    needs[row["city"]] -= 1
                if len(completed[purpose]) == count:
                    break
            if len(completed[purpose]) != count:
                raise ValueError(
                    f"Not enough qualified {purpose} scenes: {len(completed[purpose])}"
                )
    coverage = {}
    for purpose, rows in completed.items():
        coverage[purpose] = {
            "scenes": len(rows),
            "logs": len({r["source"]["log"] for r in rows}),
            "cities": dict(Counter(r["source"]["city"] for r in rows)),
            "tags": dict(Counter(tag for r in rows for tag in r["source"]["tags"])),
        }
    report = {
        "status": "passed",
        "manifest_sha256": file_hash(MANIFEST),
        "model_config_sha256": file_hash(CONFIG),
        "coverage": coverage,
        "scenes": completed,
        "excluded": excluded,
        "nuplan_simulation_performed": False,
    }
    write_json(report_path, report)
    print(f"Native material preparation complete: {report_path}", flush=True)
    return report


def load_training_samples(report_path=ROOT / "diffusion_planner/materials.json"):
    report = json.loads(Path(report_path).read_text())
    if (
        report["status"] != "passed"
        or report["model_config_sha256"] != file_hash(CONFIG)
        or report["manifest_sha256"] != file_hash(MANIFEST)
    ):
        raise ValueError("Native material/config verification differs")
    samples = []
    keys = (
        "ego_current_state",
        "neighbor_agents_past",
        "static_objects",
        "lanes",
        "lanes_speed_limit",
        "lanes_has_speed_limit",
        "route_lanes",
        "route_lanes_speed_limit",
        "route_lanes_has_speed_limit",
    )
    for row in report["scenes"]["replay"]:
        if (
            row["source"]["purpose"] != "replay"
            or file_hash(row["path"]) != row["sha256"]
        ):
            raise ValueError("Native retention material identity mismatch")
        with np.load(row["path"], allow_pickle=False) as data:
            neighbors = data["neighbor_agents_future"][:10]
            target = np.concatenate(
                (data["ego_agent_future"][None], neighbors), axis=0
            ).astype(np.float32)
            valid = np.concatenate(
                (np.ones((1, 80), bool), np.any(neighbors != 0, axis=-1)), axis=0
            )
            samples.append(
                {
                    "features": {key: data[key].copy() for key in keys},
                    "target": target,
                    "future_valid": valid,
                    "scene_id": row["scene_id"],
                    "domain": "native",
                    "source": "nuplan_train",
                }
            )
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(args.workers)


if __name__ == "__main__":
    main()

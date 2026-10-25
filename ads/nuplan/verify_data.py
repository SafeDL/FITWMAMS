"""Offline CRC, SQLite, split and native nuPlan readback checks."""

import hashlib
import json
import sqlite3
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version

import yaml

from ads.nuplan.prepare_data import MANIFEST, WORKSPACE, session_id

REPORT = WORKSPACE / "preparation/nuplan_train_verification.json"
TABLES = {
    "log",
    "scene",
    "lidar_pc",
    "ego_pose",
    "lidar_box",
    "scenario_tag",
    "traffic_light_status",
}


def database_path(row):
    return (
        WORKSPACE
        / "data/nuplan-v1.1/splits"
        / ("train_" + row["purpose"])
        / (row["log"] + ".db")
    )


def verify_database(row):
    path = database_path(row)
    if path.stat().st_size != row["bytes"]:
        raise ValueError(f"DB size differs from the download manifest: {path}")
    crc, digest = 0, hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024**2), b""):
            crc = zlib.crc32(block, crc)
            digest.update(block)
    if crc != row["crc32"]:
        raise ValueError(f"DB CRC differs from the official ZIP member: {path}")
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        tables = {
            record[0]
            for record in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if TABLES - tables:
            raise ValueError(f"Missing nuPlan tables in {path}: {TABLES - tables}")
        first, last, frames = connection.execute(
            "SELECT MIN(timestamp),MAX(timestamp),COUNT(*) FROM lidar_pc"
        ).fetchone()
        if frames == 0 or first == last:
            raise ValueError(f"No usable frame sequence in {path}")
        if connection.execute("SELECT 1 FROM ego_pose LIMIT 1").fetchone() is None:
            raise ValueError(f"No ego states in {path}")
        map_name = connection.execute("SELECT map_version FROM log").fetchone()[0]
        tags = dict(
            connection.execute("SELECT type,COUNT(*) FROM scenario_tag GROUP BY type")
        )
        probe = connection.execute(
            "SELECT hex(token),timestamp FROM lidar_pc "
            "WHERE timestamp BETWEEN ? AND ? "
            "ORDER BY ABS(timestamp-?) LIMIT 1",
            (first + 2_000_000, last - 10_000_000, (first + last) // 2),
        ).fetchone()
    return {
        "log": row["log"],
        "purpose": row["purpose"],
        "city": row["city"],
        "path": str(path),
        "bytes": row["bytes"],
        "crc32": crc,
        "sha256": digest.hexdigest(),
        "map_name": map_name,
        "frames": frames,
        "duration_s": (last - first) / 1e6,
        "scenario_tag_counts": tags,
        "probe": {"token": probe[0].lower(), "timestamp": probe[1]} if probe else None,
    }


def native_readback(record):
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import (
        NuPlanScenario,
    )
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
        ScenarioExtractionInfo,
    )

    scenario = NuPlanScenario(
        data_root=str(WORKSPACE / "data"),
        log_file_load_path=record["path"],
        initial_lidar_token=record["probe"]["token"],
        initial_lidar_timestamp=record["probe"]["timestamp"],
        scenario_type="data_readback",
        map_root=str(WORKSPACE / "data/maps"),
        map_version="nuplan-maps-v1.0",
        map_name=record["map_name"],
        scenario_extraction_info=ScenarioExtractionInfo(
            scenario_duration=10.0, subsample_ratio=0.5
        ),
        ego_vehicle_parameters=get_pacifica_parameters(),
    )
    ego = scenario.get_ego_state_at_iteration(0)
    past = list(scenario.get_ego_past_trajectory(0, 2.0, num_samples=20))
    future = list(scenario.get_ego_future_trajectory(0, 8.0, num_samples=80))
    current = scenario.get_tracked_objects_at_iteration(0)
    future_objects = list(scenario.get_future_tracked_objects(0, 8.0, num_samples=8))
    lights = list(scenario.get_traffic_light_status_at_iteration(0))
    route = scenario.get_route_roadblock_ids()
    nearby = scenario.map_api.get_proximal_map_objects(
        ego.rear_axle.point,
        80.0,
        [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
    )
    if len(past) != 20 or len(future) != 80 or len(future_objects) != 8:
        raise ValueError(
            f"Native history/future sampling is incomplete: {record['path']}"
        )
    lanes = sum(len(objects) for objects in nearby.values())
    if not lanes or not route:
        raise ValueError(f"Native map/route unavailable: {record['path']}")
    return {
        "log": record["log"],
        "city": record["city"],
        "purpose": record["purpose"],
        "past_ego_samples": len(past),
        "future_ego_samples": len(future),
        "current_objects": len(current.tracked_objects.tracked_objects),
        "future_object_frames": len(future_objects),
        "traffic_light_records": len(lights),
        "route_roadblocks": len(route),
        "nearby_lanes_and_connectors": lanes,
    }


def verify():
    plan = json.loads(MANIFEST.read_text())
    splitter = WORKSPACE / (
        "nuplan-devkit/nuplan/planning/script/config/common/splitter/nuplan.yaml"
    )
    if hashlib.sha256(splitter.read_bytes()).hexdigest() != plan["splitter_sha256"]:
        raise ValueError("Official split configuration changed since planning")
    splits = yaml.safe_load(splitter.read_text())["log_splits"]
    reserved = {session_id(name) for name in splits["val"] + splits["test"]}
    rows = plan["logs"]
    if len({row["session"] for row in rows}) != len(rows):
        raise ValueError("Source sessions overlap")
    if any(
        row["log"] not in splits["train"] or row["session"] in reserved for row in rows
    ):
        raise ValueError("Source pool overlaps reserved Val/Test sessions")
    results, partials = [], []
    for name in ("train_replay", "train_development"):
        partials.extend((WORKSPACE / "data/nuplan-v1.1/splits" / name).glob("*.part"))
    if partials:
        raise ValueError(f"Unfinished DB downloads: {partials}")
    with ThreadPoolExecutor(max_workers=4) as workers:
        for index, result in enumerate(workers.map(verify_database, rows), 1):
            results.append(result)
            if index % 25 == 0 or index == len(rows):
                print(f"CRC and SQLite verified: {index}/{len(rows)}", flush=True)
    maps = json.loads((WORKSPACE / "data/maps/nuplan-maps-v1.0.json").read_text())
    tag_counts, purposes, cities = Counter(), Counter(), Counter()
    probes = {}
    for record in results:
        map_name = record["map_name"]
        map_file = (
            WORKSPACE / "data/maps" / map_name / maps[map_name]["version"] / "map.gpkg"
        )
        if not map_file.is_file():
            raise FileNotFoundError(map_file)
        tag_counts.update(record["scenario_tag_counts"])
        purposes[record["purpose"]] += 1
        cities[record["city"]] += 1
        if record["probe"]:
            probes.setdefault((record["city"], record["purpose"]), record)
    if len(probes) != len(cities) * len(purposes):
        raise ValueError(
            "Not all city/purpose combinations have a native readback window"
        )
    readbacks = []
    for record in probes.values():
        readbacks.append(native_readback(record))
        print(
            f"Native readback verified: {record['city']}/{record['purpose']}",
            flush=True,
        )
    report = {
        "status": "passed",
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "verified_logs": len(results),
        "verified_bytes": sum(record["bytes"] for record in results),
        "by_purpose": dict(purposes),
        "by_city": dict(cities),
        "scenario_tag_counts": dict(tag_counts),
        "native_readbacks": readbacks,
        "logs": results,
        "environment": {
            name: version(name)
            for name in ("numpy", "torch", "nuplan-devkit", "pyogrio", "rasterio")
        },
        "checks": [
            "official_split",
            "session_isolation",
            "size",
            "zip_crc32",
            "sqlite_tables",
            "maps",
            "native_readback",
        ],
        "training_or_simulation": False,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verification passed: {REPORT}")
    return report


if __name__ == "__main__":
    verify()

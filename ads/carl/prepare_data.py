"""Encode original nuPlan observations with CaRL's official inference builder."""

import inspect
import json
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from ads.carl.policy import observation_space
from ads.common.instances import file_hash, json_hash
from ads.common.reporting import write_json
from ads.diffusion_planner.prepare_data import (
    COUNTS,
    ROOT,
    native_scenario,
    select_scenes,
)


def native_inputs(scenario):
    from nuplan.planning.simulation.history.simulation_history_buffer import (
        SimulationHistoryBuffer,
    )
    from nuplan.planning.simulation.planner.abstract_planner import (
        PlannerInitialization,
        PlannerInput,
    )
    from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
        SimulationIteration,
    )

    ego = list(scenario.get_ego_past_trajectory(0, time_horizon=2.0, num_samples=20))
    tracked = list(
        scenario.get_past_tracked_objects(0, time_horizon=2.0, num_samples=20)
    )
    if len(ego) != 20 or len(tracked) != 20:
        raise ValueError("CaRL native retention requires complete recorded history")
    ego.append(scenario.get_ego_state_at_iteration(0))
    tracked.append(scenario.get_tracked_objects_at_iteration(0))
    history = SimulationHistoryBuffer.initialize_from_list(
        21, ego, tracked, sample_interval=scenario.database_interval
    )
    current = PlannerInput(
        SimulationIteration(scenario.get_time_point(0), 0),
        history,
        list(scenario.get_traffic_light_status_at_iteration(0)),
    )
    initialization = PlannerInitialization(
        list(scenario.get_route_roadblock_ids()),
        scenario.get_mission_goal(),
        scenario.map_api,
    )
    return current, initialization


def native_observation(scenario):
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    current, initialization = native_inputs(scenario)
    if not initialization.route_roadblock_ids:
        raise ValueError("Native CaRL route has no roadblocks")
    for roadblock in initialization.route_roadblock_ids:
        block = initialization.map_api.get_map_object(
            roadblock, SemanticMapLayer.ROADBLOCK
        )
        block = block or initialization.map_api.get_map_object(
            roadblock, SemanticMapLayer.ROADBLOCK_CONNECTOR
        )
        if block is None:
            raise ValueError(
                f"Native CaRL route roadblock {roadblock} is absent from map"
            )

    from carl_nuplan.planning.gym.environment.helper.environment_area import (
        RectangleEnvironmentArea,
    )
    from carl_nuplan.planning.gym.environment.observation_builder.default.default_observation_builder import (
        DefaultObservationBuilder,
    )
    from carl_nuplan.planning.gym.environment.observation_builder.default.default_renderer import (
        DefaultRenderer,
    )

    area = RectangleEnvironmentArea()
    builder = DefaultObservationBuilder(area, DefaultRenderer(area), inference=True)
    # Match the native planner's first decision, including its zero previous action.
    observation = builder.build_observation(
        current, initialization, {"last_action": np.zeros(2, np.float32)}
    )
    if not observation_space().contains(observation) or any(
        not np.isfinite(value).all() for value in observation.values()
    ):
        raise ValueError("Invalid native CaRL observation")
    return observation


def encode_scene(row, directory):
    identity = json_hash([row["log"], row["token"]])
    path = Path(directory) / (identity + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            observation = {name: cached[name] for name in cached.files}
        if not observation_space().contains(observation):
            raise ValueError("Cached native CaRL observation has a different schema")
    else:
        observation = native_observation(native_scenario(row))
        np.savez_compressed(path, **observation)
    return {
        "scene_id": identity,
        "path": str(path),
        "sha256": file_hash(path),
        "source": row,
        "history_samples": 21,
        "future_labels": False,
        "builder": "official DefaultObservationBuilder(inference=True)",
    }


def native_source_fingerprint():
    import carl_nuplan

    package = Path(inspect.getfile(carl_nuplan)).parent
    return json_hash(
        {
            str(path.relative_to(package)): file_hash(path)
            for path in sorted(package.rglob("*.py"))
        }
    )


def qualified_scene(row, directory):
    try:
        return encode_scene(row, directory)
    except (ValueError, RuntimeError, IndexError, AssertionError) as error:
        return {"source": row, "excluded": type(error).__name__ + ": " + str(error)}


def prepare():
    manifest = select_scenes()
    directory = ROOT / "carl"
    directory.mkdir(parents=True, exist_ok=True)
    protocol = {
        "source_selection_sha256": json_hash(manifest),
        "encoder_sha256": file_hash(__file__),
        "official_sources_sha256": native_source_fingerprint(),
        "counts": COUNTS,
        "history_seconds": 2.0,
        "history_samples": 20,
        "future_labels": False,
    }
    header = directory / "protocol.json"
    if header.exists():
        if json.loads(header.read_text()) != protocol:
            raise ValueError("Native CaRL preprocessing protocol changed")
    else:
        write_json(header, protocol)
    destination = directory / "materials.json"
    if destination.exists():
        report = json.loads(destination.read_text())
        if report["protocol_sha256"] != file_hash(header):
            raise ValueError("Native CaRL material protocol changed")
        for purpose, count in COUNTS.items():
            entries = report["scenes"][purpose]
            if len(entries) != count or any(
                entry["source"]["purpose"] != purpose
                or file_hash(entry["path"]) != entry["sha256"]
                for entry in entries
            ):
                raise ValueError("Native CaRL material identity or purpose differs")
        return report
    completed, excluded = {}, []
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        for purpose, count in COUNTS.items():
            rows = manifest["scenes"][purpose]["selected"]
            if len(rows) != count or any(row["purpose"] != purpose for row in rows):
                raise ValueError(
                    "Native CaRL source selection has a different purpose or size"
                )
            encode = partial(qualified_scene, directory=directory / purpose)
            completed[purpose] = []
            needs = Counter()
            for index, result in enumerate(executor.map(encode, rows), 1):
                if "excluded" in result:
                    excluded.append(result)
                    needs[result["source"]["city"]] += 1
                else:
                    completed[purpose].append(result)
                if index % 25 == 0 or index == count:
                    print(
                        f"Native CaRL {purpose}: {index}/{count}, valid {len(completed[purpose])}",
                        flush=True,
                    )
            attempted = {(row["log"], row["token"]) for row in rows}
            for row in manifest["scenes"][purpose]["reserve"]:
                if not any(needs.values()):
                    break
                identity = (row["log"], row["token"])
                if not needs[row["city"]] or identity in attempted:
                    continue
                if row["purpose"] != purpose:
                    raise ValueError("Native CaRL reserve has a different data purpose")
                attempted.add(identity)
                result = executor.submit(encode, row).result()
                if "excluded" in result:
                    excluded.append(result)
                else:
                    completed[purpose].append(result)
                    needs[row["city"]] -= 1
                if not any(needs.values()):
                    break
            if len(completed[purpose]) != count:
                raise ValueError("Insufficient qualified native CaRL source scenes")
    replay_sessions = {entry["source"]["session"] for entry in completed["replay"]}
    development_sessions = {
        entry["source"]["session"] for entry in completed["development"]
    }
    if replay_sessions & development_sessions:
        raise ValueError("CaRL native retention and development sessions overlap")
    report = {
        "status": "passed",
        "protocol_sha256": file_hash(header),
        "scenes": completed,
        "excluded": excluded,
        "coverage": {
            purpose: {
                "scenes": len(entries),
                "logs": len({r["source"]["log"] for r in entries}),
                "cities": dict(Counter(r["source"]["city"] for r in entries)),
            }
            for purpose, entries in completed.items()
        },
        "nuplan_simulation_performed": False,
        "ads_training_performed": False,
        "future_labels": False,
    }
    write_json(destination, report)
    return report


if __name__ == "__main__":
    prepare()

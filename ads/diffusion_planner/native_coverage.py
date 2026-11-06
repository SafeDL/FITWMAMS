"""Describe available driving behaviour from recorded futures, without relabelling."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from ads.common.instances import file_hash
from ads.common.reporting import write_json


def behaviour_indicators(features, future):
    speed = np.linalg.norm(np.diff(future[:, :2], axis=0), axis=-1) / 0.1
    heading = np.unwrap(future[:, 2])
    displacement = future[-1, :2]
    stopped = speed < 0.5
    moving = speed > 2.0
    past = features["neighbor_agents_past"][:, -1]
    return {
        "recorded_stationary": bool(speed.max() < 0.5),
        "recorded_start_from_stop": bool(stopped[:20].any() and moving[20:].any()),
        "recorded_braking_to_stop": bool(moving[:40].any() and stopped[40:].any()),
        "recorded_turn": bool(np.abs(heading).max() >= np.pi / 8),
        "straight_road_lateral_shift_candidate": bool(
            np.abs(heading).max() < np.pi / 12
            and abs(displacement[1]) > 2
            and displacement[0] > 20
        ),
        "front_vehicle_following_candidate": bool(
            ((past[:, 0] > 0) & (past[:, 0] < 50) & (np.abs(past[:, 1]) < 2.5)).any()
            and moving.any()
        ),
    }


def audit(report_path, destination):
    report = json.loads(Path(report_path).read_text())
    coverage = {}
    for purpose, rows in report["scenes"].items():
        tags, indicators, cases = Counter(), Counter(), []
        for row in rows:
            with np.load(row["path"], allow_pickle=False) as data:
                flags = behaviour_indicators(data, data["ego_agent_future"])
            tags.update(row["source"]["tags"])
            indicators.update({name: int(value) for name, value in flags.items()})
            cases.append(
                {
                    "scene_id": row["scene_id"],
                    "city": row["source"]["city"],
                    "official_tags": row["source"]["tags"],
                    "trajectory_indicators": flags,
                }
            )
        coverage[purpose] = {
            "scenes": len(rows),
            "official_tags": dict(tags),
            "trajectory_indicators": dict(indicators),
            "cases": cases,
        }
    result = {
        "materials_sha256": file_hash(report_path),
        "coverage": coverage,
        "definition": {
            "sample_interval_s": 0.1,
            "stopped_mps": 0.5,
            "moving_mps": 2.0,
            "turn_heading_rad": float(np.pi / 8),
            "straight_heading_rad": float(np.pi / 12),
            "lateral_displacement_m": 2.0,
            "forward_displacement_m": 20.0,
            "front_vehicle_x_m": [0, 50],
            "front_vehicle_abs_y_m": 2.5,
        },
        "interpretation": "Geometric behaviour candidates describe source coverage; they do not replace official scenario labels or alter training selection.",
    }
    write_json(destination, result)
    return result


if __name__ == "__main__":
    import yaml

    from ads.common.setup import REPO_ROOT
    from ads.diffusion_planner.expert_materials import CONFIG

    settings = yaml.safe_load(CONFIG.read_text())
    result = audit(
        settings["native_materials"],
        REPO_ROOT / settings["output_dir"] / "native_coverage.json",
    )
    print(
        {
            purpose: values["trajectory_indicators"]
            for purpose, values in result["coverage"].items()
        }
    )

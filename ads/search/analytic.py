"""Two cheap high-dimensional search checks; no ADS or traffic model is run."""

from types import SimpleNamespace
from typing import ClassVar

import numpy as np

from ads.common.evaluation import EpisodeResult
from ads.common.instances import GeneratedSource
from ads.search.engine import SearchConfig, SearchEngine


class AnalyticSampler:
    generated_road: ClassVar = {"lane_centers_m": [-4.0, 0.0, 4.0], "lane_width_m": 4.0}

    def compose_exogenous(self, omega):
        states = np.zeros((omega.batch_size, 7, 6), np.float32)
        states[:, :, 2] = 20
        states[:, 1, 0] = 40
        valid = np.zeros((omega.batch_size, 7), bool)
        valid[:, :2] = True
        return SimpleNamespace(
            initial_states=states,
            initial_valid=valid,
            soft_plan=np.zeros((omega.batch_size, 149, 6, 2), np.float32),
            vehicle_dimensions=np.tile(
                np.array([4.8, 1.8], np.float32), (omega.batch_size, 7, 1)
            ),
        )


def performance(instance, problem):
    values = instance.event.exogenous_state.c0_base_latent[0]
    x, y = map(float, values[:2])
    if problem == "unbalanced":
        left = ((x + 2.7) / 0.3) ** 2 + (y / 0.5) ** 2 < 1
        right = ((x - 2.2) / 0.8) ** 2 + (y / 1.1) ** 2 < 1
        risk = max(
            np.exp(-(((x + 2.7) / 1.5) ** 2)), np.exp(-(((x - 2.2) / 1.8) ** 2))
        ) * np.exp(-y * y / 2)
    elif problem == "weak_early_branch":
        left, right = x < -3 and abs(y) < 0.8, x > 2 and abs(y) < 1.5
        risk = max(0.15 * np.exp(-x), np.exp(x)) * np.exp(-y * y / 2)
    else:
        raise ValueError("unknown analytic search problem")
    failure = bool(left or right)
    return EpisodeResult(
        instance,
        {
            "safety_failure": failure,
            "collision": failure,
            "event_risk": float(risk),
            "evt_score": min(float(risk), 1.0),
            "branch_id": "root",
            "interaction_type": "same_explanation",
            "descriptor": [x, y, 0, 0, 0.5],
            "critical_slot": 1,
            "critical_frame": 75,
            "physics_steps": 149,
            "elapsed_s": 0.0,
            "analytic_region": "left" if left else "right" if right else "safe",
        },
        np.empty((0, 7, 6)),
        np.empty((0, 2)),
        [],
    )


def compare(problem, *, config=None):
    config = config or SearchConfig()
    result = {}
    for method in ("subset", "guided"):
        source = GeneratedSource(
            AnalyticSampler(),
            {"analytic": problem},
            master_seed=20260926,
            purpose="search",
        )
        run = SearchEngine(
            source, lambda instance: performance(instance, problem), config=config
        ).run(method)
        regular = [
            r for r in run.records if r.record["execution_kind"] != "branch_check"
        ]
        result[method] = {
            **run.summary(),
            "known_regions_found": sorted(
                {r.record["analytic_region"] for r in regular if r.failure}
            ),
            "ambient_dimension": sum(
                value.size
                for value in regular[0]
                .instance.event.exogenous_state.as_dict()
                .values()
            ),
            "active_dimensions": 2,
        }
    return result


if __name__ == "__main__":
    import json

    print(
        json.dumps(
            {
                problem: compare(problem)
                for problem in ("unbalanced", "weak_early_branch")
            },
            indent=2,
        )
    )

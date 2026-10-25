"""Verify the current tread runtime against saved generator and world replays."""

import json

import numpy as np
import torch
import yaml

from ads.common.reporting import write_json
from ads.common.setup import REPO_ROOT, load_world
from ads.common.world import WorldEvent
from hierarchical_world_model.src.randomness import (
    WORLD_RANDOM_BLOCKS,
    WorldExogenousState,
)

REFERENCE = REPO_ROOT / "results/risk_stratified_repair_history/runtime_reference"
REPORT = REPO_ROOT / "results/risk_stratified_repair_history/runtime_check.json"


def replay(instance, factory):
    event = instance.event
    world = factory(event)
    world.reset(
        torch.as_tensor(event.initial_states)[None],
        torch.as_tensor(event.valid)[None],
        torch.as_tensor(event.soft_reference)[None],
        torch.as_tensor(event.map_polylines)[None],
        torch.as_tensor(event.map_polyline_valid)[None],
        exogenous_state=event.exogenous_state,
        vehicle_dimensions=event.vehicle_dimensions[None],
    )
    states = [np.asarray(world.observe()["agent_states"].cpu())[0]]
    failures = []
    for _ in range(len(event.soft_reference)):
        result = world.advance_response(torch.zeros((1, 2)))
        states.append(np.asarray(result["agent_states"].cpu())[0])
        failures.append((bool(result["crashed"][0, 0]), bool(result["offroad"][0, 0])))
    return np.stack(states), np.asarray(failures)


def main():
    settings = yaml.safe_load(
        (REPO_ROOT / "ads/configs/risk_stratified_repair.yaml").read_text()
    )
    torch.set_num_threads(1)
    sampler, factory, _, assets = load_world(settings)
    reference = json.loads((REFERENCE / "runtime.json").read_text())
    for key, digest in reference["assets"].items():
        if (
            key not in ("world_implementation", "generation_contract")
            and assets[key] != digest
        ):
            raise ValueError(f"frozen world asset differs from the reference: {key}")
    reports = []
    for index in range(4):
        # These are immutable numeric reference fixtures, not campaign instances.
        with np.load(REFERENCE / f"instance_{index}.npz", allow_pickle=False) as data:
            event = WorldEvent(
                f"runtime_reference_{index}",
                "runtime_reference",
                **{
                    name: data[name].copy()
                    for name in (
                        "initial_states",
                        "valid",
                        "soft_reference",
                        "map_polylines",
                        "map_polyline_valid",
                        "vehicle_dimensions",
                    )
                },
                exogenous_state=WorldExogenousState(
                    **{name: data["rng_" + name].copy() for name in WORLD_RANDOM_BLOCKS}
                ),
            )
        from types import SimpleNamespace

        instance = SimpleNamespace(event=event)
        recomposed = sampler.compose_exogenous(event.exogenous_state)
        np.testing.assert_allclose(
            recomposed.initial_states[0],
            instance.event.initial_states,
            atol=1e-3,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            recomposed.soft_plan[0], instance.event.soft_reference, atol=1e-2, rtol=1e-5
        )
        states, failures = replay(instance, factory)
        with np.load(REFERENCE / f"trace_{index}.npz") as old:
            np.testing.assert_allclose(states, old["states"], atol=1e-3, rtol=1e-5)
            np.testing.assert_array_equal(failures, old["failures"])
            reports.append(
                {
                    "case": index,
                    "max_state_difference": float(np.abs(states - old["states"]).max()),
                    "max_soft_plan_difference": float(
                        np.abs(
                            recomposed.soft_plan[0] - instance.event.soft_reference
                        ).max()
                    ),
                }
            )
    write_json(
        REPORT,
        {
            "status": "passed",
            "source_torch": reference["torch"],
            "target_torch": torch.__version__,
            "assets": assets,
            "cases": reports,
            "state_atol": 1e-3,
            "soft_plan_atol": 1e-2,
        },
    )
    print(f"Runtime replay check passed: {REPORT}")


if __name__ == "__main__":
    main()

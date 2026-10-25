"""Registered experiment units; evaluation depends on every final policy freeze."""

from dataclasses import dataclass
from pathlib import Path

NATIVE_PLANNERS = ("diffusion_planner", "dtpp", "flow_planner", "str2")
SUPERVISED_ALGORITHMS = ("plant2", *NATIVE_PLANNERS)
POLICY_SUFFIX = {
    "dqn": ".zip",
    "ppo": ".zip",
    "carl": ".pth",
    "plant2": ".ckpt",
    "diffusion_planner": ".pth",
    "dtpp": ".pth",
    "flow_planner": ".pth",
    "str2": "",
}


def training_budget(config, algorithm, *, baseline):
    if baseline:
        # All eight initial policies are imported, never trained by this protocol.
        return (
            "gradient_updates"
            if algorithm in SUPERVISED_ALGORITHMS
            else "actual_steps",
            0,
        )
    if algorithm in SUPERVISED_ALGORITHMS:
        return "gradient_updates", config["repair"]["plant_updates"]
    return "actual_steps", config["repair"]["rl_steps"]


@dataclass(frozen=True)
class Task:
    stage: str
    algorithm: str | None = None
    seed: int | None = None
    method: str | None = None
    round_index: int | None = None

    @property
    def key(self):
        return "_".join(
            str(v)
            for v in (
                self.stage,
                self.algorithm,
                self.seed,
                self.method,
                self.round_index,
            )
            if v is not None
        )

    def directory(self, output):
        output = Path(output)
        if self.stage == "baseline":
            return output / self.algorithm / str(self.seed) / "baseline"
        if self.stage == "repair":
            return (
                output
                / self.algorithm
                / str(self.seed)
                / self.method
                / f"round_{self.round_index}"
            )
        return output / "units" / self.key

    def command(self, python, config):
        command = [str(python), "-u", "-m", "ads.worker", self.stage]
        if self.algorithm is not None:
            command += [self.algorithm, str(self.seed)]
        if self.method is not None:
            command += ["--method", self.method]
        if self.round_index is not None:
            command += ["--round", str(self.round_index)]
        return command + ["--config", str(config)]


def experiment_graph(config):
    graph = {}
    final = set()
    for algorithm in config["algorithms"]:
        for seed in config["seeds"]:
            baseline = Task("baseline", algorithm, seed)
            graph[baseline] = set()
            for method in config["methods"]:
                previous = baseline
                for round_index in range(1, config["rounds"] + 1):
                    task = Task("repair", algorithm, seed, method, round_index)
                    graph[task] = {previous}
                    previous = task
                final.add(previous)
    freeze = Task("freeze")
    graph[freeze] = final
    evaluations = set()
    for algorithm in config["algorithms"]:
        for seed in config["seeds"]:
            for method in ("initial", *config["methods"]):
                task = Task("iid", algorithm, seed, method)
                graph[task] = {freeze}
                evaluations.add(task)
            for method in config["methods"]:
                task = Task("diagnostic", algorithm, seed, method)
                graph[task] = {freeze}
                evaluations.add(task)
    graph[Task("report")] = evaluations
    return graph

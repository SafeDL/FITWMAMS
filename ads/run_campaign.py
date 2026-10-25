"""Run the fixed experiment as isolated, dependency-ordered local processes."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from ads.common.instances import file_hash, json_hash
from ads.common.reporting import write_json
from ads.common.setup import REPO_ROOT
from ads.tasks import (
    POLICY_SUFFIX,
    SUPERVISED_ALGORITHMS,
    experiment_graph,
    training_budget,
)

STR2_INFERENCE_WORKERS = 3


def process_alive(pid):
    try:
        status = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return status not in ("Z", "X")


class Scheduler:
    def __init__(self, config, *, workers=1):
        self.config_path = Path(config).resolve()
        self.config = yaml.safe_load(self.config_path.read_text())
        self.output = REPO_ROOT / self.config["output_dir"]
        self.output.mkdir(parents=True, exist_ok=True)
        protocol = self.output / "protocol.json"
        if (
            protocol.exists()
            and json.loads(protocol.read_text())["torch"] != torch.__version__
        ):
            raise ValueError(
                "launch the scheduler with the registered ADS Python environment"
            )
        self.graph = experiment_graph(self.config)
        self.workers = workers
        self.running, self.failed, self.children = {}, set(), {}
        self.hash_cache = {}

    def digest(self, path):
        files = (
            sorted(p for p in path.rglob("*") if p.is_file())
            if path.is_dir()
            else [path]
        )
        stamp = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)
        cached = self.hash_cache.get(path)
        if cached is None or cached[0] != stamp:
            self.hash_cache[path] = (stamp, file_hash(path))
        return self.hash_cache[path][1]

    def training_done(self, task):
        directory = task.directory(self.output)
        path = directory / "training.json"
        if not path.exists():
            return False
        metadata = json.loads(path.read_text())
        protocol = json.loads((self.output / "protocol.json").read_text())
        if metadata["protocol_hash"] != json_hash(protocol):
            raise ValueError(f"training protocol mismatch: {task.key}")
        suffix = POLICY_SUFFIX[task.algorithm]
        checkpoint = directory / ("policy" + suffix)
        if self.digest(checkpoint) != metadata["policy_sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {task.key}")
        key, target = training_budget(
            self.config, task.algorithm, baseline=task.stage == "baseline"
        )
        actual = metadata["learning"][key]
        skipped = metadata["learning"].get("skipped") == "no_qualified_demonstrations"
        if (
            task.algorithm == "dtpp"
            and task.stage == "repair"
            and not skipped
            and metadata["learning"].get("dtpp_attention_mask_layout")
            != "batch_then_head"
        ):
            raise ValueError("DTPP checkpoint predates the batch-mask correction")
        if actual != target and not (
            task.algorithm in SUPERVISED_ALGORITHMS and actual == 0 and skipped
        ):
            raise ValueError(f"incomplete learning budget: {task.key}")
        manifest = directory / "inputs/curriculum.json"
        if not manifest.exists():
            return False
        if self.digest(manifest) != metadata["curriculum_sha256"]:
            raise ValueError(f"curriculum hash mismatch: {task.key}")
        if task.stage == "baseline":
            return metadata["learning"][key] == 0
        summary = directory / "search_summary.json"
        if not summary.exists() or not (directory / "archive/archive.json").exists():
            return False
        evaluations = json.loads(summary.read_text())["evaluations"]
        budget = self.config["search"]["budget"]
        if not 0 < evaluations <= budget or (
            self.config["search"].get("fixed_budget", False) and evaluations != budget
        ):
            raise ValueError(f"incomplete search budget: {task.key}")
        return True

    def done(self, task):
        if task.stage in ("baseline", "repair"):
            return self.training_done(task)
        if task.stage == "freeze":
            from ads.common.evaluation_sampling import COHORTS

            return (self.output / "frozen_policies.json").exists() and all(
                len(list((self.output / "instances" / cohort).glob("*.npz")))
                == self.config["evaluation"]["instances"]
                for cohort in COHORTS
            )
        if task.stage == "iid":
            from ads.common.evaluation_sampling import COHORTS

            freeze = self.output / "frozen_policies.json"
            if not freeze.exists():
                return False
            frozen = json.loads(freeze.read_text())
            key = f"{task.algorithm}/{task.seed}/{task.method}"
            for cohort in COHORTS:
                path = (
                    self.output
                    / "evaluation"
                    / cohort
                    / task.algorithm
                    / str(task.seed)
                    / (task.method + ".json")
                )
                if not path.exists():
                    return False
                value = json.loads(path.read_text())
                if (
                    value["policy_sha256"] != frozen["policies"][key]["sha256"]
                    or len(value["records"]) != self.config["evaluation"]["instances"]
                ):
                    return False
            return True
        if task.stage == "diagnostic":
            path = task.directory(self.output) / "result.json"
            return (
                path.exists()
                and json.loads(path.read_text()).get("schema")
                == "ads_archive_diagnostics_v2"
            )
        return (
            (self.output / "audit.json").exists()
            and (self.output / "evaluation/summary.json").exists()
            and (self.output / "reports/costs.csv").exists()
        )

    def existing_worker(self, task):
        path = task.directory(self.output) / "unit_status.json"
        if not path.exists():
            return None
        status = json.loads(path.read_text())
        pid = status["pid"]
        if status["status"] != "running" or not process_alive(pid):
            return None
        command = (
            Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        )
        expected = f"ads.worker {task.stage}"
        if task.algorithm is not None:
            expected += f" {task.algorithm} {task.seed}"
        if expected not in command:
            raise ValueError(f"worker PID belongs to a different command: {task.key}")
        if task.method is not None and f"--method {task.method}" not in command:
            raise ValueError(f"worker method differs: {task.key}")
        if (
            task.round_index is not None
            and f"--round {task.round_index}" not in command
        ):
            raise ValueError(f"worker round differs: {task.key}")
        return status

    def search_prepared(self, task):
        if task.stage != "repair":
            return False
        directory = task.directory(self.output)
        path = directory / "unit_status.json"
        if not path.exists():
            return False
        status = json.loads(path.read_text())
        return (
            status.get("search_only") is True
            and status["status"] == "search_complete"
            and all(
                (directory / name).exists()
                for name in (
                    "search_summary.json",
                    "inputs/curriculum.json",
                    "archive/archive.json",
                )
            )
        )

    def launch(self, task):
        logs = self.output / "unit_logs"
        logs.mkdir(exist_ok=True)
        log = logs / (task.key + ".log")
        with log.open("ab", buffering=0) as stream:
            child = subprocess.Popen(
                task.command(sys.executable, self.config_path),
                cwd=REPO_ROOT,
                env=dict(os.environ, USE_TF="0", HF_HUB_OFFLINE="1"),
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.children[task] = child
        self.running[task] = child.pid
        print(f"Started {task.key}, PID {child.pid}", flush=True)

    def can_launch(self, task):
        if len(self.running) >= self.workers:
            return False
        # STR2 keeps the full 800M-parameter Adam state and needs the GPU alone
        # during repair. Other units can share the mostly CPU world rollouts.
        needs_gpu_alone = task.stage == "repair" and task.algorithm == "str2"
        if needs_gpu_alone and self.running:
            return False
        # Each measured STR2 inference process uses about 5,400 MiB on this
        # 24 GiB GPU; leave room for the other evaluation workers.
        if (
            task.algorithm == "str2"
            and task.stage in ("iid", "diagnostic")
            and sum(active.algorithm == "str2" for active in self.running)
            >= STR2_INFERENCE_WORKERS
        ):
            return False
        return not any(
            active.stage == "repair" and active.algorithm == "str2"
            for active in self.running
        )

    def run(self):
        with (self.output / "scheduler.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while True:
                completed = set()
                for task in self.graph:
                    if task in self.running:
                        continue
                    status = self.existing_worker(task)
                    if self.done(task):
                        completed.add(task)
                    elif status:
                        self.running[task] = status["pid"]
                for task, pid in list(self.running.items()):
                    child = self.children.get(task)
                    alive = (
                        child.poll() is None
                        if child is not None
                        else process_alive(pid)
                    )
                    if alive:
                        continue
                    self.running.pop(task)
                    if self.done(task):
                        completed.add(task)
                        print(f"Completed {task.key}", flush=True)
                    elif self.search_prepared(task):
                        print(
                            f"Prepared search for {task.key}; optimizer still pending",
                            flush=True,
                        )
                    else:
                        self.failed.add(task)
                        print(
                            f"FAILED {task.key}; other independent units continue",
                            flush=True,
                        )
                priority = {
                    "baseline": 0,
                    "repair": 1,
                    "freeze": 2,
                    "iid": 3,
                    "diagnostic": 4,
                    "report": 5,
                }
                ordered = sorted(
                    self.graph,
                    key=lambda task: (
                        priority[task.stage],
                        task.algorithm not in SUPERVISED_ALGORITHMS,
                        task.round_index or 0,
                        task.key,
                    ),
                )
                for task in ordered:
                    dependencies = self.graph[task]
                    if len(self.running) >= self.workers:
                        break
                    if (
                        task not in completed
                        and task not in self.running
                        and task not in self.failed
                        and dependencies <= completed
                        and self.can_launch(task)
                    ):
                        self.launch(task)
                payload = {
                    "pid": os.getpid(),
                    "status": "running",
                    "completed_units": len(completed),
                    "total_units": len(self.graph),
                    "workers": self.workers,
                    "running": {task.key: pid for task, pid in self.running.items()},
                    "failed": sorted(task.key for task in self.failed),
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                }
                if len(completed) == len(self.graph):
                    write_json(
                        self.output / "status.json", {**payload, "status": "complete"}
                    )
                    return
                if not self.running:
                    write_json(
                        self.output / "status.json", {**payload, "status": "failed"}
                    )
                    raise RuntimeError(
                        "no runnable units remain; inspect failed unit logs"
                    )
                write_json(self.output / "status.json", payload)
                time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "ads/configs/risk_stratified_repair.yaml",
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    Scheduler(args.config, workers=args.workers).run()


if __name__ == "__main__":
    main()

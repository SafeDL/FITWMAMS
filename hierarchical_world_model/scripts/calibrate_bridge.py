"""Full Train scaling and natural Val selection for the one-time handover."""

import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import yaml
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src import bridge_evaluation as experiment
from hierarchical_world_model.src.core.utils import file_sha256, load_yaml, save_json
from hierarchical_world_model.src.data import ego_controls, prepare_experiment_data
from hierarchical_world_model.src.error_trajectories import MODEL_PATH
from hierarchical_world_model.src.highway import HighwayEnvClosedLoopWorld
from hierarchical_world_model.src.plan_applicability import (
    conflict_scores,
    forecast_bundle,
    interaction_scores,
)
from hierarchical_world_model.src.reaction_controller import ReactionControllerOutput
from hierarchical_world_model.src.shared_evaluation import scene_randomness
from hierarchical_world_model.src.takeover import TakeoverConfig, TakeoverController
from hierarchical_world_model.src.tracking import TrackingConfig

OUTPUT = ROOT / "results/hierarchical_world_model/bridge"
BATCH = 128
WORKERS = 16
_DATA = None


class ApplicabilityMonitor(TakeoverController):
    """Measure all first-admission candidates without changing the tracker trace."""

    def forward(self, context, *, deterministic=False):
        valid = context.current_valid[:, 1:]
        if self.owned is None:
            self.owned = torch.zeros_like(valid)
            self.maximum_conflict = context.current.new_zeros(valid.shape)
            self.maximum_interaction = context.current.new_zeros((*valid.shape, 2))
        self._update_plan_maneuver(context)
        forecast = forecast_bundle(context, self.owned, self.error_bank)
        conflict, _, _ = conflict_scores(
            context, self.owned, self.error_bank, forecast=forecast
        )
        previous = (
            (context.history[:, -2], context.history_valid[:, -2])
            if context.history.shape[1] > 1
            else None
        )
        raw, _, _ = interaction_scores(context, forecast, previous, self.plan_started)
        raw = torch.where(valid[..., None], raw, 0)
        self.maximum_conflict = torch.maximum(self.maximum_conflict, conflict)
        self.maximum_interaction = torch.maximum(self.maximum_interaction, raw)
        return ReactionControllerOutput(
            actions=context.base_actions,
            alpha=valid.float() * 0,
            delta_ax=valid.float() * 0,
            active=self.owned,
        )


def initialize(split):
    global _DATA
    torch.set_num_threads(1)
    threadpool_limits(1)
    config = load_yaml(ROOT / "hierarchical_world_model/config.yaml")
    data = prepare_experiment_data(config, ROOT)
    rows, plans = experiment.load_plans(data, split)
    _DATA = config, data.bundle.arrays, rows, plans, split


def score_batch(offset):
    config, arrays, rows, plans, split = _DATA
    path = OUTPUT / ".cache" / "scores" / split / f"{offset}.npz"
    if path.exists():
        return offset
    chosen = rows[offset : offset + BATCH]
    log = np.asarray(arrays["agent_states"][chosen])
    valid = np.asarray(arrays["agent_valid"][chosen])
    monitor = ApplicabilityMonitor(
        torch.ones(6, 5),
        takeover_config=TakeoverConfig(error_trajectories_path=str(MODEL_PATH)),
    )
    world = HighwayEnvClosedLoopWorld(
        controller=monitor, tracking_config=TrackingConfig(**config["tracking"])
    )
    world.reset(
        log[:, 24],
        valid[:, 24],
        plans[offset : offset + len(chosen)],
        np.asarray(arrays["map_polylines"][chosen]),
        np.asarray(arrays["map_polyline_valid"][chosen]),
        vehicle_dimensions=np.asarray(arrays["vehicle_dimensions"][chosen]),
        exogenous_state=scene_randomness(
            arrays["sequence_id"][chosen], seed=experiment.SEEDS[0]
        ),
        initial_history=log[:, :25],
        initial_history_valid=valid[:, :25],
    )
    controls = ego_controls(log[:, 24:173, 0], log[:, 25:174, 0], 0.04)
    collided = np.zeros(len(chosen), bool)
    for frame in range(149):
        result = world.advance_response(torch.tensor(controls[:, frame]))
        collided |= result["crashed"].any(-1).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rows=chosen,
        conflict=monitor.maximum_conflict.numpy(),
        interaction=monitor.maximum_interaction.numpy(),
        valid=valid[:, 24, 1:],
        collided=collided,
    )
    return offset


def collect(split):
    initialize(split)
    rows = _DATA[2]
    jobs = list(range(0, len(rows), BATCH))
    with ProcessPoolExecutor(
        max_workers=WORKERS,
        mp_context=get_context("spawn"),
        initializer=initialize,
        initargs=(split,),
    ) as pool:
        for number, offset in enumerate(pool.map(score_batch, jobs)):
            print(
                "Natural applicability", split, number + 1, "/", len(jobs), flush=True
            )
    blocks = [
        dict(np.load(OUTPUT / ".cache" / "scores" / split / f"{offset}.npz"))
        for offset in jobs
    ]
    values = {name: np.concatenate([b[name] for b in blocks]) for name in blocks[0]}
    np.savez_compressed(OUTPUT / f"{split}_scores.npz", **values)
    return values


def select(train, val):
    raw = train["interaction"][train["valid"]]
    scales = []
    for channel in range(2):
        positive = raw[:, channel][raw[:, channel] > 0]
        if not len(positive):
            raise ValueError("Train contains no valid interaction event for scaling")
        scales.append(float(np.quantile(positive, 0.95)))
    conflict = val["conflict"].max(1)
    combined = (val["interaction"] / np.array(scales, dtype=np.float32)).max((1, 2))
    candidates = [
        {"threshold": k / 16, "scene_rate": float(np.mean(conflict >= k / 16))}
        for k in range(1, 17)
    ]
    qualified = [c for c in candidates if c["scene_rate"] <= 0.005]
    report = {
        "status": "failed_no_conflict_threshold",
        "train_scenes": len(train["rows"]),
        "val_scenes": len(val["rows"]),
        "scaling": "95th percentile of positive Train vehicle-episode maximum raw scores",
        "following_scale_s": scales[0],
        "release_scale_per_s": scales[1],
        "conflict_candidates": candidates,
        "conflict_scene_budget": 0.005,
        "noncollision_scene_budget": 0.01,
        "test_or_ADS_failures_used": False,
        "raw_units": ["seconds", "inverse_seconds"],
        "score_support_is_collision_probability": False,
    }
    if not qualified:
        save_json(report, OUTPUT / "selection.json")
        return report
    q = qualified[0]["threshold"]
    # The 1% budget combines the two noncollision reasons; conflict has its own budget.
    fixed = conflict >= q
    allowed = int(np.floor(0.01 * len(combined)))
    remaining = np.sort(combined)
    excluded = remaining[max(0, len(remaining) - allowed - 1)]
    threshold = float(np.nextafter(np.float32(excluded), np.float32(np.inf)))
    admitted = fixed | (combined >= threshold)
    report.update(
        status="selected_on_natural_data",
        support_threshold=q,
        interaction_threshold=threshold,
        conflict_scenes=int(fixed.sum()),
        noncollision_scenes=int((combined >= threshold).sum()),
        noncollision_scene_rate=float((combined >= threshold).mean()),
        combined_scenes=int(admitted.sum()),
        combined_scene_rate=float(admitted.mean()),
        train_valid_vehicle_episodes=int(train["valid"].sum()),
        threshold_comparison="inclusive >=; smallest representable threshold above the excluded Val score",
    )
    config_path = ROOT / "hierarchical_world_model/config.yaml"
    config = load_yaml(config_path)
    config["takeover"].update(
        support_threshold=q,
        interaction_threshold=threshold,
        following_scale_s=scales[0],
        release_scale_per_s=scales[1],
        error_trajectories_path="results/hierarchical_world_model/model_assets/interaction_errors.npz",
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    save_json(report, OUTPUT / "selection.json")
    return report


def main():
    train, val = collect("train"), collect("val")
    report = select(train, val)
    print(report, flush=True)
    report["source_sha256"] = {
        str(p.relative_to(ROOT)): file_sha256(p)
        for p in [
            Path(__file__),
            ROOT / "hierarchical_world_model/src/plan_applicability.py",
            ROOT / "hierarchical_world_model/src/takeover.py",
            MODEL_PATH,
        ]
    }
    save_json(report, OUTPUT / "selection.json")


if __name__ == "__main__":
    main()

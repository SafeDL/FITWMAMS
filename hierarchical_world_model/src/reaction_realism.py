"""Supported highD reaction references and stratified MLOO scoring.

This module deliberately keeps distribution alignment outside the frozen
HiQR world model.  It mines *observed* ego-braking events from a supplied
split, records only causal post-event quantities, and exposes a small,
serialisable reference table for PPO post-training.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from diffusion.src.data import ANCHOR_INDEX


FEATURE_NAMES = (
    "final_ax_mps2", "abs_jerk_mps3", "speed_mps", "gap_m", "closing_mps", "ttc_s",
)
BRAKE_EDGES = (0.5, 2.0, 4.0, 6.0, 8.0001)
TTC_EDGES = (0.0, 2.0, 4.0, 10.0001)
DEFAULT_WINDOW_FRAMES = 25
DEFAULT_ROLLOUT_STEPS = 149


def support_cell(brake_mps2: float, ttc_s: float) -> int:
    """Return the fixed brake/TTC support cell, or ``-1`` when outside it."""
    brake = float(brake_mps2)
    ttc = float(np.clip(ttc_s, 0.0, 10.0))
    brake_index = int(np.searchsorted(BRAKE_EDGES, brake, side="right") - 1)
    ttc_index = int(np.searchsorted(TTC_EDGES, ttc, side="right") - 1)
    if not (0 <= brake_index < len(BRAKE_EDGES) - 1 and 0 <= ttc_index < len(TTC_EDGES) - 1):
        return -1
    return brake_index * (len(TTC_EDGES) - 1) + ttc_index


def cell_label(cell: int) -> str:
    if cell < 0:
        return "unsupported"
    columns = len(TTC_EDGES) - 1
    brake, ttc = divmod(int(cell), columns)
    return f"brake_{BRAKE_EDGES[brake]:g}_{BRAKE_EDGES[brake + 1]:g}_ttc_{TTC_EDGES[ttc]:g}_{TTC_EDGES[ttc + 1]:g}"


@dataclass(frozen=True)
class SupportedEventPool:
    """Replayable highD ego-parent events, indexed into a local array view."""

    row_index: np.ndarray
    onset_step: np.ndarray
    child_slot: np.ndarray
    cell: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.row_index)
        if any(len(value) != count for value in (self.onset_step, self.child_slot, self.cell)):
            raise ValueError("supported-event fields must have equal length")

    def by_cell(self, cell: int) -> np.ndarray:
        return np.flatnonzero(self.cell == int(cell))


@dataclass(frozen=True)
class ReactionRealismReference:
    """Train-only distribution reference plus replayable supported events."""

    distributions: dict[int, np.ndarray]
    scales: dict[int, np.ndarray]
    event_counts: dict[int, int]
    supported_cells: tuple[int, ...]
    events: SupportedEventPool
    source_rows_sha256: str
    window_frames: int = DEFAULT_WINDOW_FRAMES
    minimum_events: int = 100
    source_split: str = "unspecified"
    replay_radius_m: float = 50.0

    @property
    def schema(self) -> str:
        return "highd_reaction_realism_reference_v1"

    def is_supported(self, cell: int) -> bool:
        return int(cell) in self.supported_cells

    def with_supported_cells(
        self, cells: Iterable[int], *, minimum_events: int | None = None,
    ) -> "ReactionRealismReference":
        """Return an evaluation view restricted to train-admitted cells.

        Held-out splits may have too few events to establish support on their
        own.  They can still provide target distributions for cells that the
        train split already admitted.  Keeping that policy here avoids
        duplicating a fragile dataclass reconstruction in training and
        evaluation entry points.
        """
        allowed = {int(cell) for cell in cells}
        supported = tuple(
            cell for cell in self.supported_cells
            if cell in allowed and cell in self.distributions
        )
        return replace(
            self,
            supported_cells=supported,
            minimum_events=(self.minimum_events if minimum_events is None else int(minimum_events)),
        )

    def save(self, directory: str | Path) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "event_row_index": self.events.row_index.astype(np.int64),
            "event_onset_step": self.events.onset_step.astype(np.int16),
            "event_child_slot": self.events.child_slot.astype(np.int8),
            "event_cell": self.events.cell.astype(np.int16),
        }
        for cell, values in self.distributions.items():
            arrays[f"distribution_{cell}"] = np.asarray(values, np.float32)
            arrays[f"scale_{cell}"] = np.asarray(self.scales[cell], np.float32)
        np.savez_compressed(root / "reaction_realism_reference.npz", **arrays)
        manifest = {
            "schema": self.schema,
            "feature_names": FEATURE_NAMES,
            "brake_edges_mps2": BRAKE_EDGES,
            "ttc_edges_s": TTC_EDGES,
            "window_frames": self.window_frames,
            "minimum_events": self.minimum_events,
            "event_counts": {str(key): int(value) for key, value in self.event_counts.items()},
            "supported_cells": [int(cell) for cell in self.supported_cells],
            "supported_cell_labels": [cell_label(cell) for cell in self.supported_cells],
            "source_rows_sha256": self.source_rows_sha256,
            "source_split": self.source_split,
            "replay_radius_m": self.replay_radius_m,
        }
        (root / "reaction_realism_reference.json").write_text(json.dumps(manifest, indent=2) + "\n")

    @classmethod
    def load(cls, directory: str | Path) -> "ReactionRealismReference":
        root = Path(directory)
        manifest = json.loads((root / "reaction_realism_reference.json").read_text())
        if manifest.get("schema") != "highd_reaction_realism_reference_v1":
            raise ValueError("unsupported reaction realism reference schema")
        with np.load(root / "reaction_realism_reference.npz") as payload:
            distributions = {
                int(key.removeprefix("distribution_")): payload[key].copy()
                for key in payload.files if key.startswith("distribution_")
            }
            scales = {
                int(key.removeprefix("scale_")): payload[key].copy()
                for key in payload.files if key.startswith("scale_")
            }
            events = SupportedEventPool(
                payload["event_row_index"].copy(), payload["event_onset_step"].copy(),
                payload["event_child_slot"].copy(), payload["event_cell"].copy(),
            )
        return cls(
            distributions=distributions, scales=scales,
            event_counts={int(key): int(value) for key, value in manifest["event_counts"].items()},
            supported_cells=tuple(int(value) for value in manifest["supported_cells"]),
            events=events, source_rows_sha256=str(manifest["source_rows_sha256"]),
            window_frames=int(manifest["window_frames"]), minimum_events=int(manifest["minimum_events"]),
            source_split=str(manifest.get("source_split", "unspecified")),
            replay_radius_m=float(manifest.get("replay_radius_m", 50.0)),
        )


def _event_features(states: np.ndarray, valid: np.ndarray, frame: int, child_slot: int, window: int) -> np.ndarray | None:
    """Extract the causal-lagged post-event window for one follower.

    The parent brake is committed at ``frame``.  A controller can first react
    at ``frame + 1``, so the target window starts there and records the state
    *after* each corresponding follower action, matching HighwayEnv telemetry.
    """
    start = int(frame) + 1
    stop = start + int(window)
    # The final action needs state ``stop``; jerk additionally needs t-1.
    if frame < 1 or stop >= len(states):
        return None
    child = int(child_slot)
    needed = valid[frame:start + int(window) + 1, (0, child)]
    if not bool(needed.all()):
        return None
    follower = states[start:stop, child]
    next_parent = states[start + 1:stop + 1, 0]
    next_follower = states[start + 1:stop + 1, child]
    previous_follower = states[start - 1:stop - 1, child]
    action = np.clip((next_follower[:, 2] - follower[:, 2]) / .04, -8.0, 4.0)
    previous_action = np.clip((follower[:, 2] - previous_follower[:, 2]) / .04, -8.0, 4.0)
    gap = next_parent[:, 0] - next_follower[:, 0] - 4.8
    closing = next_follower[:, 2] - next_parent[:, 2]
    ttc = np.where(closing > 1.e-4, gap / np.maximum(closing, 1.e-4), 10.0)
    return np.stack((
        action, np.abs(action - previous_action) / .04, next_follower[:, 2], gap,
        closing, np.clip(ttc, 0.0, 10.0),
    ), axis=-1).astype(np.float32)


def build_reaction_realism_reference(
    arrays: dict[str, np.ndarray], source_rows: np.ndarray, *,
    minimum_events: int = 100, window_frames: int = DEFAULT_WINDOW_FRAMES,
    allowed_cells: Iterable[int] | None = None,
    rollout_steps: int = DEFAULT_ROLLOUT_STEPS,
    source_split: str = "unspecified",
    replay_radius_m: float = 50.0,
) -> ReactionRealismReference:
    """Mine observed ego braking events without using future values as inputs.

    Future frames are used only after the event to form an offline target
    distribution.  The event selection itself uses the committed parent
    acceleration and same-tick gap/TTC.  At most one event per source sequence
    and cell is retained, so support is measured in independent sequences.
    """
    states_all = np.asarray(arrays["agent_states"], np.float32)
    valid_all = np.asarray(arrays["agent_valid"], bool)
    rows = np.asarray(source_rows, np.int64)
    if len(states_all) != len(rows):
        raise ValueError("source_rows must align with local reaction arrays")
    allowed = None if allowed_cells is None else {int(value) for value in allowed_cells}
    features: dict[int, list[np.ndarray]] = {}
    events: dict[int, list[tuple[int, int, int]]] = {}
    # Need a full post-event window and one previous state for jerk.
    first = max(int(ANCHOR_INDEX), 1)
    # PPO never observes response actions beyond this horizon. Do not admit
    # an event which cannot supply the entire post-event scorer window.
    last = min(
        states_all.shape[1] - int(window_frames) - 2,
        int(ANCHOR_INDEX) + int(rollout_steps) - int(window_frames) - 1,
    )
    claimed_source_cells: set[tuple[int, int]] = set()
    for local_row, (states, valid) in enumerate(zip(states_all, valid_all)):
        seen: set[int] = set()
        for frame in range(first, last + 1):
            if not (valid[frame, 0] and valid[frame + 1, 0] and valid[frame - 1, 0]):
                continue
            ego = states[frame, 0]
            previous_ego = states[frame - 1, 0]
            next_ego = states[frame + 1, 0]
            ego_ax = float((next_ego[2] - ego[2]) / .04)
            previous_ax = float((ego[2] - previous_ego[2]) / .04)
            brake = -ego_ax
            # An onset is band-specific: a long sustained brake is not
            # repeatedly counted as independent support.
            candidates: list[tuple[float, int, float]] = []
            for child in range(1, 7):
                if not valid[frame, child]:
                    continue
                follower = states[frame, child]
                gap = float(ego[0] - follower[0] - 4.8)
                closing = float(follower[2] - ego[2])
                if not (0.1 < gap < float(replay_radius_m) and abs(float(ego[1] - follower[1])) < 1.8):
                    continue
                ttc = float(np.clip(gap / closing, 0.0, 10.0)) if closing > 1.e-4 else 10.0
                cell = support_cell(brake, ttc)
                if cell < 0 or cell in seen or (allowed is not None and cell not in allowed):
                    continue
                lower = BRAKE_EDGES[cell // (len(TTC_EDGES) - 1)]
                upper = BRAKE_EDGES[cell // (len(TTC_EDGES) - 1) + 1]
                if lower <= -previous_ax < upper:
                    continue
                candidates.append((gap, child, ttc))
            # One closest following relation per sequence/cell prevents a
            # dense local recording from dominating either support or W1.
            for _, child, ttc in sorted(candidates):
                cell = support_cell(brake, ttc)
                source_cell = (int(rows[local_row]), cell)
                if cell in seen or source_cell in claimed_source_cells:
                    continue
                values = _event_features(states, valid, frame, child, window_frames)
                if values is None:
                    continue
                seen.add(cell)
                claimed_source_cells.add(source_cell)
                features.setdefault(cell, []).append(values)
                events.setdefault(cell, []).append((local_row, frame - int(ANCHOR_INDEX), child))
    counts = {cell: len(value) for cell, value in events.items()}
    supported = tuple(sorted(
        cell for cell, count in counts.items()
        if count >= int(minimum_events) and (allowed is None or cell in allowed)
    ))
    distributions: dict[int, np.ndarray] = {}
    scales: dict[int, np.ndarray] = {}
    row_index: list[int] = []
    onset_step: list[int] = []
    child_slot: list[int] = []
    event_cell: list[int] = []
    for cell, values in features.items():
        # Evaluation references may retain an allowed but sparsely sampled
        # cell; only the training table decides whether it is supported.
        flattened = np.concatenate(values, axis=0).astype(np.float32)
        distributions[cell] = flattened
        iqr = np.quantile(flattened, .75, axis=0) - np.quantile(flattened, .25, axis=0)
        scales[cell] = np.maximum(iqr, 1.e-3).astype(np.float32)
        if cell in supported:
            for row, onset, child in events[cell]:
                row_index.append(row); onset_step.append(onset); child_slot.append(child); event_cell.append(cell)
    digest = hashlib.sha256(rows.tobytes()).hexdigest()
    return ReactionRealismReference(
        distributions=distributions, scales=scales, event_counts=counts,
        supported_cells=supported,
        events=SupportedEventPool(
            np.asarray(row_index, np.int64), np.asarray(onset_step, np.int16),
            np.asarray(child_slot, np.int8), np.asarray(event_cell, np.int16),
        ),
        source_rows_sha256=digest, window_frames=int(window_frames), minimum_events=int(minimum_events),
        source_split=str(source_split),
        replay_radius_m=float(replay_radius_m),
    )


def wasserstein_1d(left: np.ndarray, right: np.ndarray) -> float:
    left = np.sort(np.asarray(left, np.float64).reshape(-1))
    right = np.sort(np.asarray(right, np.float64).reshape(-1))
    count = min(len(left), len(right))
    if count == 0:
        return float("inf")
    quantiles = np.linspace(0.0, 1.0, count)
    return float(np.abs(np.quantile(left, quantiles) - np.quantile(right, quantiles)).mean())


def realism_metric(trajectories: np.ndarray, reference: ReactionRealismReference, cell: int) -> tuple[float, np.ndarray]:
    """Return the six-feature similarity and its per-feature W1 values."""
    values = np.asarray(trajectories, np.float32)
    if values.ndim != 3 or values.shape[-1] != len(FEATURE_NAMES):
        raise ValueError("trajectories must be [rollouts, frames, six_features]")
    target = reference.distributions.get(int(cell))
    scale = reference.scales.get(int(cell))
    if target is None or scale is None or not len(values):
        return 0.0, np.full(len(FEATURE_NAMES), np.inf, np.float32)
    # Each trajectory is represented by the same number of frames, so the
    # concatenation gives every rollout equal weight in the empirical CDF.
    w1 = np.asarray([
        wasserstein_1d(values[..., index], target[:, index])
        for index in range(len(FEATURE_NAMES))
    ], np.float32)
    return float(np.exp(-w1 / scale).mean()), w1


def mloo_rewards(trajectories: np.ndarray, reference: ReactionRealismReference, cell: int, *, clip: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute permutation-invariant, standardized leave-one-out rewards."""
    values = np.asarray(trajectories, np.float32)
    count = len(values)
    if count < 2:
        return np.zeros(count, np.float32), np.zeros(count, np.float32)
    leave_one_out = np.asarray([
        realism_metric(np.delete(values, index, axis=0), reference, cell)[0]
        for index in range(count)
    ], np.float32)
    raw = leave_one_out.mean() - leave_one_out
    std = float(raw.std())
    normalized = raw / max(std, 1.e-6)
    # Clipping alone can destroy the defining zero-sum MLOO baseline. Center
    # after clipping and, if necessary, rescale back into the advertised
    # interval; this preserves both the group baseline and bounded advantage.
    bounded = np.clip(normalized, -float(clip), float(clip))
    bounded = bounded - bounded.mean()
    maximum = float(np.abs(bounded).max())
    if maximum > float(clip):
        bounded *= float(clip) / maximum
    return bounded.astype(np.float32), leave_one_out

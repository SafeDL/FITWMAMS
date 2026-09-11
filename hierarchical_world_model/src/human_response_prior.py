"""Train-defined conditional human-response references.

This module is deliberately offline-only.  Its peak and dose fields help
select comparable human events for rewards and reports, but never enter the
online traffic controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .reaction_evidence import EVALUATION_FRAMES, FEATURE_NAMES, ReactionEventReference, event_window


DESCRIPTOR_NAMES = (
    "gap_m", "closing_mps", "ttc_s", "follower_previous_acceleration_mps2",
    "follower_speed_mps", "leader_onset_acceleration_mps2", "leader_short_brake_dose_mps",
)
RESPONSE_NAMES = ("acceleration_mps2", "abs_jerk_mps3")
SUPPORT_LABELS = (
    "empirically_supported", "weakly_supported", "unsupported_by_training_evidence",
)


@dataclass(frozen=True)
class HumanResponsePrior:
    """A compact, train-scaled human response library for one split."""

    split: str
    descriptor: np.ndarray
    response: np.ndarray
    recording_id: np.ndarray
    leader_id: np.ndarray
    follower_id: np.ndarray
    event_key: np.ndarray
    metrics: np.ndarray
    descriptor_median: np.ndarray
    descriptor_iqr: np.ndarray
    response_iqr: np.ndarray
    support_label: np.ndarray
    supported_distance: float
    weak_distance: float

    def save(self, directory: str | Path) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / "human_response_prior.npz", **self.__dict__)
        manifest = {
            "schema_name": "human_response_prior",
            "schema_version": 1,
            "split": self.split,
            "descriptor_names": DESCRIPTOR_NAMES,
            "response_names": RESPONSE_NAMES,
            "response_frames": EVALUATION_FRAMES,
            "online_safe_fields": [
                "gap_m", "closing_mps", "ttc_s", "follower_previous_acceleration_mps2", "follower_speed_mps",
            ],
            "offline_only_fields": [
                "leader_peak_brake_mps2", "leader_brake_dose_mps", "leader_brake_ramp_mps3",
                "leader_brake_duration_s", "response_latency_s", "follower_peak_brake_mps2",
                "follower_peak_abs_jerk_mps3", "follower_brake_dose_mps", "recovery_time_s",
            ],
            "neighbors": 16,
            "minimum_recordings": 5,
            "supported_distance": float(self.supported_distance),
            "weak_distance": float(self.weak_distance),
            "support_labels": list(SUPPORT_LABELS),
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    @classmethod
    def load(cls, directory: str | Path) -> "HumanResponsePrior":
        root = Path(directory)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("schema_name") != "human_response_prior" or manifest.get("schema_version") != 1:
            raise ValueError("unsupported human-response prior")
        with np.load(root / "human_response_prior.npz") as values:
            return cls(**{name: values[name].copy() for name in cls.__dataclass_fields__})

    def normalized_descriptor(self, descriptor: np.ndarray) -> np.ndarray:
        return (np.asarray(descriptor, np.float32) - self.descriptor_median) / self.descriptor_iqr

    def neighbor_indices(
        self, descriptor: np.ndarray, *, recording_id: int, leader_id: int, follower_id: int,
        event_key: int, neighbors: int = 16,
    ) -> np.ndarray:
        """Return nearest train events after every leakage exclusion."""
        distance = np.square(self.normalized_descriptor(self.descriptor) - self.normalized_descriptor(descriptor)).sum(-1)
        allowed = (
            (self.recording_id != int(recording_id))
            & ~((self.leader_id == int(leader_id)) & (self.follower_id == int(follower_id)))
            & (self.event_key != int(event_key))
        )
        candidates = np.flatnonzero(allowed)
        if len(candidates) < neighbors:
            return np.empty(0, np.int64)
        selected = candidates[np.argpartition(distance[candidates], neighbors - 1)[:neighbors]]
        return selected[np.argsort(distance[selected], kind="stable")]


def _acceleration(states: np.ndarray) -> np.ndarray:
    return (states[1:, :, 2] - states[:-1, :, 2]) / 0.04


def _event_arrays(
    reference: ReactionEventReference, arrays: dict[str, np.ndarray], *, ego_leader_only: bool,
) -> dict[str, np.ndarray]:
    events = reference.events
    indices = events.indices(reference.supported_cells)
    if ego_leader_only:
        indices = indices[events.leader_slot[indices] == 0]
    states_all = np.asarray(arrays["agent_states"], np.float32)
    rows = np.asarray(arrays["row_index"], np.int64)
    lookup = {int(row): index for index, row in enumerate(rows)}
    kept = np.asarray([index for index in indices if int(events.row_index[index]) in lookup], np.int64)
    descriptor, metrics = [], []
    for index in kept:
        local = lookup[int(events.row_index[index])]
        onset = int(events.local_onset_frame[index])
        leader, follower = int(events.leader_slot[index]), int(events.follower_slot[index])
        states = states_all[local]
        acceleration = _acceleration(states)
        leader_future = acceleration[onset:onset + EVALUATION_FRAMES, leader]
        follower_future = acceleration[onset:onset + EVALUATION_FRAMES, follower]
        follower_previous = float(acceleration[onset - 1, follower])
        gap = float(states[onset, leader, 0] - states[onset, follower, 0] - 4.8)
        closing = float(states[onset, follower, 2] - states[onset, leader, 2])
        ttc = gap / closing if gap > 0.0 and closing > 1.0e-4 else 10.0
        onset_ax = float(leader_future[0])
        dose = float(np.maximum(-leader_future[:10], 0.0).sum() * 0.04)
        peak = float(np.maximum(-leader_future, 0.0).max())
        duration = float((leader_future < -0.5).sum() * 0.04)
        ramp = float(np.abs(np.diff(leader_future[:10])).max(initial=0.0) / 0.04)
        follower_peak = float(np.minimum(follower_future, 0.0).min())
        follower_jerk = np.abs(np.diff(np.concatenate(([follower_previous], follower_future)))) / 0.04
        crossed = np.flatnonzero(follower_future <= follower_previous - 0.1)
        latency = float(crossed[0] * 0.04) if len(crossed) else float(EVALUATION_FRAMES * 0.04)
        recovery = float(np.flatnonzero(follower_future >= follower_previous - 0.1)[-1] * 0.04) if np.any(follower_future >= follower_previous - 0.1) else float(EVALUATION_FRAMES * 0.04)
        descriptor.append((gap, closing, np.clip(ttc, 0.0, 10.0), follower_previous, states[onset, follower, 2], onset_ax, dose))
        metrics.append((peak, dose, ramp, duration, latency, follower_peak, float(follower_jerk.max()), float(np.maximum(-follower_future, 0.0).sum() * 0.04), recovery))
    response = event_window(events)[kept, :, :2]
    return {
        "indices": kept,
        "descriptor": np.asarray(descriptor, np.float32),
        "response": np.asarray(response, np.float32),
        "metrics": np.asarray(metrics, np.float32),
    }


def build_human_response_priors(
    *, train_reference: ReactionEventReference, split_reference: ReactionEventReference,
    train_arrays: dict[str, np.ndarray], split_arrays: dict[str, np.ndarray], split: str,
) -> tuple[HumanResponsePrior, dict[str, int | float]]:
    """Build a split query set with all support thresholds frozen from train."""
    library = _event_arrays(train_reference, train_arrays, ego_leader_only=False)
    query = _event_arrays(split_reference, split_arrays, ego_leader_only=True)
    train_query = _event_arrays(train_reference, train_arrays, ego_leader_only=True)
    median = np.median(library["descriptor"], axis=0).astype(np.float32)
    iqr = np.maximum(np.quantile(library["descriptor"], .75, axis=0) - np.quantile(library["descriptor"], .25, axis=0), 1.0e-3).astype(np.float32)
    response_iqr = np.maximum(
        np.quantile(library["response"].reshape(-1, 2), .75, axis=0)
        - np.quantile(library["response"].reshape(-1, 2), .25, axis=0), 1.0e-3,
    ).astype(np.float32)
    train_events = train_reference.events
    library_keys = (
        train_events.absolute_onset_frame[library["indices"]]
        + 10_000_000 * train_events.recording_id[library["indices"]]
    )
    query_events = split_reference.events

    provisional = HumanResponsePrior(
        split=split,
        descriptor=library["descriptor"], response=library["response"],
        recording_id=train_events.recording_id[library["indices"]],
        leader_id=train_events.leader_id[library["indices"]],
        follower_id=train_events.follower_id[library["indices"]], event_key=library_keys,
        metrics=library["metrics"], descriptor_median=median, descriptor_iqr=iqr,
        response_iqr=response_iqr, support_label=np.empty(0, "U32"), supported_distance=0.0, weak_distance=0.0,
    )

    def distances(values: dict[str, np.ndarray], reference: ReactionEventReference) -> tuple[np.ndarray, np.ndarray]:
        result, records = [], []
        for row, index in enumerate(values["indices"]):
            event = reference.events
            key = int(event.absolute_onset_frame[index] + 10_000_000 * event.recording_id[index])
            selected = provisional.neighbor_indices(
                values["descriptor"][row], recording_id=int(event.recording_id[index]),
                leader_id=int(event.leader_id[index]), follower_id=int(event.follower_id[index]), event_key=key,
            )
            if len(selected):
                result.append(float(np.linalg.norm(provisional.normalized_descriptor(values["descriptor"][row]) - provisional.normalized_descriptor(provisional.descriptor[selected[-1]]))))
                records.append(len(np.unique(provisional.recording_id[selected])))
            else:
                result.append(float("inf")); records.append(0)
        return np.asarray(result, np.float32), np.asarray(records, np.int16)

    train_distance, _ = distances(train_query, train_reference)
    supported_distance, weak_distance = np.quantile(train_distance[np.isfinite(train_distance)], (.95, .99))
    split_distance, split_records = distances(query, split_reference)
    labels = np.full(len(query["indices"]), SUPPORT_LABELS[2], "U32")
    labels[(split_distance <= weak_distance) & (split_records >= 3)] = SUPPORT_LABELS[1]
    labels[(split_distance <= supported_distance) & (split_records >= 5)] = SUPPORT_LABELS[0]
    selected_events = split_reference.events
    keys = selected_events.absolute_onset_frame[query["indices"]] + 10_000_000 * selected_events.recording_id[query["indices"]]
    prior = HumanResponsePrior(
        split=split, descriptor=query["descriptor"], response=query["response"],
        recording_id=selected_events.recording_id[query["indices"]], leader_id=selected_events.leader_id[query["indices"]],
        follower_id=selected_events.follower_id[query["indices"]], event_key=keys,
        metrics=query["metrics"], descriptor_median=median, descriptor_iqr=iqr, response_iqr=response_iqr,
        support_label=labels, supported_distance=float(supported_distance), weak_distance=float(weak_distance),
    )
    return prior, {
        "events": int(len(labels)), "empirically_supported": int((labels == SUPPORT_LABELS[0]).sum()),
        "weakly_supported": int((labels == SUPPORT_LABELS[1]).sum()),
        "unsupported_by_training_evidence": int((labels == SUPPORT_LABELS[2]).sum()),
    }

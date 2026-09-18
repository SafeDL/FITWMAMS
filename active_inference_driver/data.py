"""Small, explicit loader for the project's 25 Hz highD cohort artifact."""
from __future__ import annotations

from pathlib import Path
import numpy as np


DEFAULT_DATASET = Path("dynamic_ar_idm/artifacts/full_data/full_highd_25hz.npz")


def load_highd_pairs(path: str | Path = DEFAULT_DATASET) -> list[dict[str, np.ndarray]]:
    """Load longitudinal highD pairs without importing another model package."""
    with np.load(Path(path), allow_pickle=False) as raw:
        data = {name: raw[name] for name in raw.files}
    pairs = []
    for index, (start, stop) in enumerate(zip(data["offsets"][:-1], data["offsets"][1:])):
        pair = {key: np.asarray(data[key][start:stop], float) for key in
                ("follower_x", "follower_v", "leader_x", "leader_v")}
        pair.update({key: data[key][index] for key in
                     ("pair_no", "recording_id", "follower_id", "leader_id", "length_sum")})
        pair["gap"] = pair["leader_x"] - pair["follower_x"] - float(pair["length_sum"])
        pairs.append(pair)
    return pairs


def leader_brake_events(pair: dict[str, np.ndarray], *, threshold_mps2: float = -1.0,
                        pre_s: float = 2.0, post_s: float = 5.0) -> list[int]:
    """Return separated 25 Hz indices where an observed leader starts braking."""
    acceleration = np.diff(pair["leader_v"], prepend=pair["leader_v"][0]) / .04
    candidates = np.flatnonzero(acceleration <= threshold_mps2)
    minimum = int(pre_s / .04); maximum = len(pair["gap"]) - int(post_s / .04)
    events, last = [], -10**9
    for point in candidates:
        if minimum <= point < maximum and point - last >= int(post_s / .04):
            events.append(int(point)); last = int(point)
    return events

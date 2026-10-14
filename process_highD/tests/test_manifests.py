from __future__ import annotations

import pandas as pd

from process_highD.scripts.build_manifests import select_manifest


def test_manifest_selection_is_stable_and_interaction_deduplicated() -> None:
    frame = pd.DataFrame(
        {
            "event_id": [f"e{index}" for index in range(8)],
            "source_event_id": [f"e{index}" for index in range(8)],
            "split": ["val"] * 8,
            "interaction_group": [0, 0, 1, 2, 3, 4, 5, 6],
        }
    )
    first = select_manifest(frame, split="val", size=5, seed=7)
    second = select_manifest(frame, split="val", size=5, seed=7)
    assert first["event_id"].tolist() == second["event_id"].tolist()
    assert first["interaction_group"].is_unique

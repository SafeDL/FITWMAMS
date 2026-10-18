from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ads.dqn.evaluate import _publish_evaluation_artifacts


def test_evaluation_publication_is_complete_immutable_and_identifies_algorithm(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluation.csv"
    frame = pd.DataFrame(
        [{"event_id": "evt-1", "failure": False, "progress_m": 12.5}]
    )
    metadata = {"algorithm": "dqn", "events": 1, "input_artifacts": {}}

    _publish_evaluation_artifacts(output, frame, metadata)

    assert pd.read_csv(output).equals(frame)
    assert json.loads(output.with_suffix(".json").read_text(encoding="utf-8")) == metadata
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _publish_evaluation_artifacts(output, frame, metadata)


def test_evaluation_metadata_is_last_commit_marker(tmp_path: Path) -> None:
    output = tmp_path / "evaluation.csv"
    output.with_suffix(".json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _publish_evaluation_artifacts(output, pd.DataFrame(), {"algorithm": "dqn"})
    assert not output.exists()

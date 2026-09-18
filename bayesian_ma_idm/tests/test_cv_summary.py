from __future__ import annotations

import json

import numpy as np

from bayesian_ma_idm.scripts.summarize_full_cv import METRICS, summarize


def test_summary_is_pair_weighted(tmp_path) -> None:
    for model, values in {"b_idm": (1., 3.), "ma_idm": (2., 4.)}.items():
        for recording, pairs, value in ((25, 1, values[0]), (26, 3, values[1])):
            payload = {"model": model, "fit_backend": "test", "pairs": pairs, "anchors": pairs,
                       "futures": 4, "evaluation_recordings": [recording], "horizons": {}}
            for horizon in ("3.0", "5.0"):
                payload["horizons"][horizon] = {metric: {"mean": value} for metric in METRICS}
                payload["horizons"][horizon]["position_interval_calibration"] = {
                    "0.5": {"mean": value / 10}, "0.9": {"mean": value / 5}}
            (tmp_path / f"{model}_holdout_{recording}.json").write_text(json.dumps(payload))
    output = summarize(tmp_path, tmp_path / "summary.json")
    result = json.loads(output.read_text())
    assert result["models"]["b_idm"]["pair_weighted_horizons"]["3.0"]["position_rmse_m"] == 2.5
    assert result["models"]["ma_idm"]["pair_weighted_horizons"]["5.0"]["position_rmse_m"] == 3.5
    assert np.isclose(result["models"]["b_idm"]["pair_weighted_horizons"]["3.0"]["position_interval_calibration"]["0.9"], .5)

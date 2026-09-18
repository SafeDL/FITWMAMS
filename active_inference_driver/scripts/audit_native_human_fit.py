"""Compare authors' native rear-end analysis with the paper Source Data.

The local input must be produced by the released ``Analysis_rear_end.py``.
This keeps the piecewise-linear brake-response extraction identical to the
paper implementation; this script only compares its final table to Source
Data, it does not re-estimate response times.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = ("RT0", "RT1", "RT_MAE", "RT_MAE_std", "A0", "A1", "A_MAE", "A_MAE_std")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-results", type=Path, required=True,
                        help="run directory containing Results_rear_end")
    parser.add_argument("--source-data-rear", type=Path, required=True,
                        help="paper Figure_6-Source_data/Rear directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results_dir = args.local_results / "Results_rear_end"
    local = pd.read_csv(results_dir / "Analysis_setting_approximations.csv")
    settings_path = results_dir / "Analysis_settings_rear_end.xlsx"
    if len(local) != 1:
        settings = pd.read_excel(settings_path)
        if "EA_mode" not in settings or len(settings) != len(local):
            raise RuntimeError("cannot identify full-model row in multi-setting analysis")
        local = local.loc[settings["EA_mode"].to_numpy() == 1]
    if len(local) != 1:
        raise RuntimeError("expected exactly one EA_mode=Surprise full-model row")
    source = pd.read_csv(args.source_data_rear / "Analysis_setting_approximations.csv")
    if len(source) <= 7:
        raise RuntimeError("paper source table lacks documented full-model setting #7")
    local_row = {key: float(local.iloc[0][key]) for key in METRICS}
    source_row = {key: float(source.iloc[7][key]) for key in METRICS}
    delta = {key: local_row[key] - source_row[key] for key in METRICS}
    gate = {
        "reaction_time_mae_increment_limit_s": 0.10,
        "deceleration_mae_increment_limit_mps2": 0.20,
        "reaction_time_mae_increment_s": delta["RT_MAE"],
        "deceleration_mae_increment_mps2": delta["A_MAE"],
        "passes": delta["RT_MAE"] <= 0.10 and delta["A_MAE"] <= 0.20,
    }
    report = {
        "analysis": "released Analysis_rear_end.py (piecewise-linear velocity response extractor)",
        "paper_source": "Figure_6-Source_data/Rear; its data-source document maps full model to setting #7",
        "local_full_model": local_row,
        "paper_full_model": source_row,
        "local_minus_paper": delta,
        "pre_registered_engineering_gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

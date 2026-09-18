from driver_reproduction.scorecard import ROOT, build_scorecard, write_scorecard


def test_scorecard_preserves_comparison_boundaries(tmp_path):
    scorecard = build_scorecard(ROOT)

    rows = {row["model"]: row for row in scorecard["natural_rollouts"]}
    assert rows["b_idm"]["benchmark_id"] == rows["ma_idm"]["benchmark_id"]
    assert rows["dynamic_ar5"]["benchmark_id"] != rows["ma_idm"]["benchmark_id"]
    assert rows["multi_regime_b_idm"]["comparison_status"] == "rejected_not_well_calibrated"
    assert len(scorecard["counterfactual_braking"]["rows"]) == 4

    output = write_scorecard(tmp_path / "scorecard.json", ROOT)
    assert output.is_file()

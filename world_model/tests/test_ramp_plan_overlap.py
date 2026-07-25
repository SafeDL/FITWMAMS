from world_model.src.ramp import RAMPConfig


def test_plan_overlap_contract():
    cfg = RAMPConfig()
    assert cfg.plan_frames == 25 and cfg.execute_frames == 5
    assert cfg.plan_frames - cfg.execute_frames == 20

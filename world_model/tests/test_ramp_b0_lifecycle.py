from world_model.src.ramp import RAMPConfig, RAMPWorldModel


def test_b0_is_start_only_by_response_index():
    model = RAMPWorldModel(RAMPConfig(hidden_dim=32))
    assert model.cfg.execute_frames * 5 == model.cfg.plan_frames

from world_model.src.ramp import RAMPConfig, RAMPWorldModel
from test_ramp_shapes import _batch


def test_rollout_has_generated_autoregressive_prefix():
    output = RAMPWorldModel(RAMPConfig(hidden_dim=32)).rollout_roll_mode(_batch(), deterministic=True)
    assert output["predicted_states"].shape[1] == 125
    assert output["continuous_memory"].shape[1] == 25

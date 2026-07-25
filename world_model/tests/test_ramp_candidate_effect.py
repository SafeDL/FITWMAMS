import torch
from world_model.src.ramp import RAMPConfig, RAMPWorldModel
from test_ramp_shapes import _batch


def test_candidate_zero_is_nominal_and_residuals_start_at_zero():
    output = RAMPWorldModel(RAMPConfig(hidden_dim=32)).forward_training(_batch(), response_steps=1)
    plans = output["candidate_control_plans"][:, 0]
    assert torch.allclose(plans[:, 0], plans[:, 1], atol=1e-6)

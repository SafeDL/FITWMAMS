import torch
from world_model.src.ramp import RAMPConfig, RAMPWorldModel
from test_ramp_shapes import _batch


def test_first_plan_ignores_future_background_labels():
    model = RAMPWorldModel(RAMPConfig(hidden_dim=32)).eval(); batch = _batch()
    changed = {key: value.clone() for key, value in batch.items()}
    changed["agent_states"][:, 25:, 1:] += 1000.0
    with torch.no_grad():
        a = model.forward_training(batch, response_steps=1)["candidate_control_plans"]
        b = model.forward_training(changed, response_steps=1)["candidate_control_plans"]
    assert torch.allclose(a, b)

import torch
from world_model.src.ramp import RAMPConfig, RAMPWorldModel
from test_ramp_shapes import _batch


def test_candidate_probability_is_scene_joint_not_per_vehicle():
    model = RAMPWorldModel(RAMPConfig(hidden_dim=32))
    output = model.forward_training(_batch(), response_steps=1)
    assert output["candidate_probabilities"].shape[-1] == 8
    assert output["candidate_probabilities"].ndim == 3

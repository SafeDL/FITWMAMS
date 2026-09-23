import torch

from interactive_behavior_world_model.evaluation.rule_rollout import _rule_action


def test_rule_background_brakes_for_close_lead():
    states = torch.zeros((1, 7, 6))
    states[0, :, 2] = 20.0
    states[0, 1, 0] = 0.0
    states[0, 2, 0] = 8.0
    states[0, 2, 2] = 0.0
    active = torch.zeros((1, 7), dtype=torch.bool)
    active[0, 1:3] = True
    action = _rule_action(
        states,
        active,
        torch.full((1, 7), 4.5),
        torch.full((1, 7), 20.0),
        torch.zeros((1, 7)),
    )
    assert action[0, 1, 0] < -1.0

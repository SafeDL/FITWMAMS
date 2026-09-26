"""The diagnostic NPC brake changes one realized leader action only."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from hierarchical_world_model.scripts.evaluate_online_npc_to_npc_highd import (
    _ForcedNPCBrakeController,
)
from hierarchical_world_model.src.reaction_controller import ReactionControllerContext
from hierarchical_world_model.src.stochastic_drivers.online import OnlineMAIDMController


def test_forced_npc_brake_is_confined_to_selected_slot_and_window() -> None:
    theta = np.tile(np.array([33.0, 2.0, 1.2, 1.4, 2.0], np.float32), (6, 1))
    states = torch.zeros(1, 7, 6)
    states[0, :, 2] = 20.0
    states[0, 0, 0] = -60.0
    states[0, 1, 0] = -20.0
    states[0, 2, 0] = 20.0
    valid = torch.zeros(1, 7, dtype=torch.bool)
    valid[0, :3] = True
    base = torch.zeros(1, 1, 6, 2)
    context = ReactionControllerContext(
        history=states[:, None], history_valid=valid[:, None],
        current=states, current_valid=valid,
        committed_ego_controls=torch.zeros(1, 1, 2),
        base_actions=base, reference_actions=base,
        intervention_trigger=torch.zeros(1), intervention_memory=torch.zeros(1),
        lateral_intervention_memory=torch.zeros(1),
        agent_style_state=torch.zeros(1, 7, 1), response_field_gain=None,
        response_sensitivity_bounds=None, adapter_gain=None,
        reaction_enabled=None, cfg=SimpleNamespace(),
    )
    normal = OnlineMAIDMController(theta)
    forced = _ForcedNPCBrakeController(theta, np.asarray([1]), -8.0)
    for frame in (24, 25, 49, 50):
        current = replace(context, response_index=frame)
        expected = normal(current, deterministic=True).actions
        actual = forced(current, deterministic=True).actions
        if frame in (25, 49):
            assert actual[0, 0, 1, 0].item() == -8.0
            expected = expected.clone()
            expected[0, 0, 1, 0] = -8.0
        torch.testing.assert_close(actual, expected)

import torch
from world_model.src.ramp import RAMPConfig, RAMPWorldModel


def _batch(batch=2):
    return {"agent_states": torch.randn(batch, 150, 7, 6), "agent_valid": torch.ones(batch, 150, 7, dtype=torch.bool), "ego_index": torch.zeros(batch, dtype=torch.long),
            "map_polylines": torch.randn(batch, 2, 4, 6), "map_polyline_valid": torch.ones(batch, 2, 4, dtype=torch.bool), "lane_graph_edges": torch.tensor([[[-1, -1, -1]],] * batch),
            "actions_highd": torch.randn(batch, 125, 6, 2), "is_evt_tail": torch.zeros(batch, dtype=torch.bool)}


def test_ramp_shapes():
    output = RAMPWorldModel(RAMPConfig(hidden_dim=32)).forward_training(_batch(), response_steps=1)
    assert output["candidate_control_plans"].shape == (2, 1, 8, 25, 6, 2)
    assert output["predicted_candidate_states"].shape == (2, 1, 8, 25, 6, 6)
    assert output["candidate_probabilities"].shape == (2, 1, 8)
    assert output["predicted_states"].shape == (2, 5, 7, 6)


def test_nominal_stage_has_finite_single_candidate_loss():
    output = RAMPWorldModel(RAMPConfig(hidden_dim=32)).forward_training(_batch(), response_steps=1, active_candidates=1)
    assert torch.isfinite(output["loss"])
    assert output["candidate_control_plans"].shape[2] == 1


def test_batched_stochastic_rollout_inverse_cdf_sampling():
    rollout = RAMPWorldModel(RAMPConfig(hidden_dim=32)).rollout_roll_mode(_batch(), deterministic=False, seed=7)
    assert rollout["selected_candidate_index"].shape == (2, 25)
    assert int(rollout["selected_candidate_index"].min()) >= 0
    assert int(rollout["selected_candidate_index"].max()) < 8


def test_all_candidate_plans_respect_hard_framewise_jerk_limits():
    cfg = RAMPConfig(hidden_dim=32)
    plans = RAMPWorldModel(cfg).forward_training(_batch(), response_steps=1)["candidate_control_plans"][:, 0]
    delta = plans[:, :, 1:] - plans[:, :, :-1]
    limit = torch.tensor((cfg.max_longitudinal_jerk, cfg.max_yaw_jerk)) * cfg.simulation_dt_s
    assert bool((delta.abs() <= limit + 1.0e-6).all())

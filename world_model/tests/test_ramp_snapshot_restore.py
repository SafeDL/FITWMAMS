import numpy as np
import torch
from world_model.src.traffic_graph.graph_schema import DynamicTrafficGraph
from world_model.src.ramp import RAMPBackgroundEnvironment, RAMPConfig, RAMPWorldModel, RAMPWorldRandomness


def _environment():
    torch.manual_seed(123)
    graph = DynamicTrafficGraph(timestamp=0., agent_ids=np.arange(7), agent_states=np.zeros((7, 6), np.float32), agent_valid=np.ones(7, bool), ego_index=0,
                                map_polylines=np.zeros((1, 2, 6), np.float32), map_polyline_valid=np.ones((1, 2), bool), lane_graph_edges=np.array([[-1, -1, -1]], np.int64))
    environment = RAMPBackgroundEnvironment(RAMPWorldModel(RAMPConfig(hidden_dim=32)))
    environment.reset(graph, RAMPWorldRandomness(plan_uniforms=[.2, .8]))
    return environment


def test_snapshot_restore_replays_next_step():
    env = _environment(); env.step(np.zeros(6, np.float32)); snapshot = env.snapshot()
    assert {"previous_candidate_index", "previous_candidate_probabilities", "previous_relation_summary", "behavior_anchor_state", "plan_uniforms_remaining", "rng_state"} <= set(snapshot)
    expected = env.step(np.zeros(6, np.float32)); env.restore(snapshot); actual = env.step(np.zeros(6, np.float32))
    assert expected["candidate_index"] == actual["candidate_index"]
    assert np.allclose(expected["background_states"], actual["background_states"])

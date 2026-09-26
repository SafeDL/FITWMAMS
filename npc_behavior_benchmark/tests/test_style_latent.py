import numpy as np
import torch

from npc_behavior_benchmark.evaluation.rollout import rollout_shared_bc


class _RecordingModel(torch.nn.Module):
    style_dim = 2

    def __init__(self):
        super().__init__()
        self.styles = []

    def forward(self, features, history_valid, style=None):
        self.styles.append(style.detach().cpu().clone())
        shape = (len(features), 6, 2)
        return features.new_zeros(shape), features.new_full(shape, -4.0)


def _run(style_resampling):
    model = _RecordingModel()
    history = np.zeros((1, 25, 7, 6), np.float32)
    history[0, :, :, 0] = np.arange(7, dtype=np.float32)[None] * 10.0
    history[0, :, :, 2] = 10.0
    valid = np.ones((1, 25, 7), bool)
    ego = np.repeat(history[:, -1:, 0], 10, axis=1)
    maps = np.zeros((1, 1, 2, 5), np.float32)
    maps[..., 4] = 4.0
    rollout_shared_bc(
        model,
        {
            "feature_mean": np.zeros(12, np.float32),
            "feature_std": np.ones(12, np.float32),
            "style_resampling": style_resampling,
        },
        history,
        valid,
        ego,
        np.arange(7)[None],
        np.full((1, 7), 4.5, np.float32),
        np.full((1, 7), 1.8, np.float32),
        maps,
        np.ones((1, 1, 2), bool),
        np.asarray(["scene"]),
        benchmark_id="test",
        fit_seed=19,
        futures=1,
        device=torch.device("cpu"),
    )
    return model.styles


def test_persistent_style_is_fixed_across_decisions():
    styles = _run("persistent")
    assert len(styles) == 2
    torch.testing.assert_close(styles[0], styles[1])


def test_a3_style_is_resampled_across_decisions():
    styles = _run("per_decision")
    assert len(styles) == 2
    assert not torch.equal(styles[0], styles[1])

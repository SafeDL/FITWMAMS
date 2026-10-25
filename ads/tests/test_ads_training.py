from stable_baselines3 import DQN

from ads.common.repair import RepairEnvironment, train_discrete


def test_dqn_continuation_warms_new_buffer_against_cumulative_steps(
    instance, world_factory, tmp_path
):
    env = RepairEnvironment(
        "dqn", {"normal": [(instance, "prior")]}, world_factory, round_index=0
    )
    base = DQN(
        "MlpPolicy",
        env,
        learning_starts=0,
        buffer_size=128,
        batch_size=4,
        policy_kwargs={"net_arch": [16, 16]},
        device="cpu",
    )
    base.num_timesteps = 100
    path = tmp_path / "base.zip"
    base.save(path)
    trained, report = train_discrete(
        "dqn", env, steps=8, seed=0, device="cpu", base=path
    )
    assert trained.learning_starts == 102
    assert trained.replay_buffer.size() == 8
    assert trained._n_updates > 0
    assert report["actual_steps"] == 8

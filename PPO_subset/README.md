# PPO_subset：PPO 长尾安全评估

`PPO_subset/` 在 `IDM_subset/` 的长尾评估管线上只替换 ego ADS：IDM ego 被替换为
stable-baselines3 PPO policy，其余场景分布、冻结扩散模型、EVT 阈值、风险评分和
final-level playback 机制保持同口径。

默认权重位于：

```text
PPO_subset/weights/ppo/model.zip
```

该权重是本目录内置的 PPO checkpoint。运行时只加载权重做推理测试，不在 subset
simulation 中训练或微调 PPO。

## 被测 ADS

PPO policy 通过 `stable_baselines3.PPO.load()` 加载，输入为 5x5 kinematics observation：

```text
[presence, x, y, vx, vy] x ego/target/nearest vehicles
```

输出是 highway-env MDP 离散动作，当前保留 PPO 原始 5 动作空间：

```text
0 LANE_LEFT
1 IDLE
2 LANE_RIGHT
3 FASTER
4 SLOWER
```

因此 PPO 在 following 和 cut-in 测试中都允许选择换道、加速、减速或保持动作；环境车道数仍由
对应 YAML 固定：following 为 1 车道，cut-in 为 2 车道。

## 公平性约束

```text
same highD tail scenario-condition distribution
same diffusion checkpoint and DDIM sampler
same initial-state reconstruction from empirical tail contexts
same EVT/GPD failure threshold
same closed-loop risk scoring
same final-level playback selection
```

与 IDM 的主要差异只在 ego ADS。当前 PPO following 为加快评估，YAML 默认
`subset_simulation.num_samples = 1000`、Monte Carlo `num_samples = 10000`；cut-in 默认
`num_samples = 1000`、Monte Carlo `num_samples = 20000`。

## 运行

```bash
conda activate tread
```

following：

```bash
python PPO_subset/scripts/run_monte_carlo_following.py
python PPO_subset/scripts/run_subset_following.py
python PPO_subset/scripts/play_final_level_following.py --no-gif
```

cut-in：

```bash
python PPO_subset/scripts/run_monte_carlo_cutin.py
python PPO_subset/scripts/run_subset_cutin.py
python PPO_subset/scripts/play_final_level_cutin.py --no-gif
```

常用覆盖项：

```bash
python PPO_subset/scripts/run_subset_following.py \
  --checkpoint_path PPO_subset/weights/ppo/model.zip \
  --deterministic \
  --seed 42 \
  --num_samples 1000
```

`--checkpoint_path` 可指向新的 PPO 权重；`--deterministic/--no-deterministic` 控制
stable-baselines3 推理方式。

## 当前结果

```text
following subset:      p = 0.18140000, se = 0.00183686
following Monte Carlo: p = 0.18300000
cut-in subset:         p = 0.02510000, se = 0.00137113
cut-in Monte Carlo:    p = 0.03175000
```

压缩结果写入：

```text
PPO_subset/results/following/ppo_following_result.json
PPO_subset/results/cutin/ppo_cutin_result.json
```

完整诊断仍保留在同目录的 `latent_subset_summary.json`、`latent_subset_level_stats.csv`、
`latent_subset_samples.npz`、`latent_subset_top_cases.json` 和
`global_risk_exposure_comparison.*`。

## 主要文件

```text
PPO_subset/policies/ppo_policy.py
PPO_subset/policies/ppo_observation_adapter.py
PPO_subset/src/closed_loop_runner.py
PPO_subset/src/result_payload.py
PPO_subset/scripts/configs/latent_subset_following.yaml
PPO_subset/scripts/configs/latent_subset_cutin.yaml
PPO_subset/weights/ppo/model.zip
tools/subset_entrypoints.py
```

`tools/subset_entrypoints.py` 提供共享 CLI 入口；PPO 脚本只声明 subset 名称、事件类型、
默认配置和压缩结果文件名。

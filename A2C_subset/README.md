# A2C_subset：A2C 长尾安全评估

`A2C_subset/` 复用 `IDM_subset/` 的 latent-space subset simulation 管线，在同一 highD 长尾
scenario-condition 分布、同一冻结扩散模型、同一 EVT/GPD 人类风险阈值和同一 subset 默认超参数下，
只把被测 ego ADS policy 替换为 A2C learned policy。

默认实现使用 `A2C_subset` 内置的 stable-baselines3 A2C checkpoint。该权重来自参考
项目 `logs_a2c/best_model/best_model.zip`，并保留参考 notebook 中的 D2RL-style
features extractor：

```text
A2C_subset/weights/a2c/model.zip
```

后续新增其他策略时，应在 `weights/` 下按算法单独放置权重，并新增对应算法前缀文件。

运行前使用项目环境：

```bash
conda activate tread
```

需要安装 `stable-baselines3`、`torch`、`gymnasium`，并使用仓库内的本地 `HighwayEnv/` package。
如果 checkpoint 不存在或 `stable-baselines3` 不可用，`A2CPolicy` 会直接报错。

## 公平性约束

- 不重新训练或修改扩散模型；
- 不修改 highD tail context / scenario-condition sampler；
- 不修改 EVT/GPD 阈值、风险变量和 mileage return-period 映射；
- 不修改 subset simulation 默认 `N`、`p0`、proposal、MCMC 接受率统计和 stopping rule；
- 默认保留 A2C checkpoint 的原始 5 个离散动作：
  `LANE_LEFT`、`IDLE`、`LANE_RIGHT`、`FASTER`、`SLOWER`。

## 运行

Car-following subset simulation：

```bash
python A2C_subset/scripts/run_subset_following.py
```

Car-following Monte Carlo baseline：

```bash
python A2C_subset/scripts/run_monte_carlo_following.py
```

Cut-in subset simulation：

```bash
python A2C_subset/scripts/run_subset_cutin.py
```

Cut-in Monte Carlo baseline：

```bash
python A2C_subset/scripts/run_monte_carlo_cutin.py
```

常用覆盖项：

```bash
python A2C_subset/scripts/run_subset_following.py \
  --checkpoint_path A2C_subset/weights/a2c/model.zip \
  --seed 42 \
  --num_samples 3000 \
  --p0 0.2 \
  --proposal_std 0.12
```

## 输出

Subset 输出：

```text
A2C_subset/results/following/latent_subset_summary.json
A2C_subset/results/following/a2c_following_result.json
A2C_subset/results/cutin/latent_subset_summary.json
A2C_subset/results/cutin/a2c_cutin_result.json
```

Monte Carlo 输出：

```text
A2C_subset/results/monte_carlo_following/latent_monte_carlo_summary.json
A2C_subset/results/monte_carlo_cutin/latent_monte_carlo_summary.json
```

完整诊断保存在同目录的 `latent_subset_level_stats.csv`、`latent_subset_samples.npz`、
`latent_subset_top_cases.json` 和 `global_risk_exposure_comparison.*`。压缩结果 JSON 包含：

```text
ads
policy
event_type
failure_event
fairness_checks
subset
monte_carlo
global_exposure
source_files
```

`policy` 字段会记录 `policy_name`、`policy_type`、`backend`、`checkpoint_path` 和 `deterministic`，
用于审计不同 ADS policy 是否在同一长尾测试分布下比较。

## 主要文件

- `policies/a2c_policy.py`：加载 D2RL-style A2C checkpoint，并提供 `reset()` / `act()`。
- `policies/a2c_observation_adapter.py`：生成 A2C 训练时使用的 5x5 kinematics observation。
- `src/closed_loop_runner.py`：用 `A2CEgoVehicle` 替换 baseline IDM ego，保留扩散 adversary 计划、
  风险评分和 trace 记录。
- `src/script_entrypoints.py`：统一 subset、Monte Carlo 和 final-level playback 的脚本入口逻辑。
- `scripts/configs/latent_subset_*.yaml`：复用 baseline 输入路径，输出到 `A2C_subset/results/`。

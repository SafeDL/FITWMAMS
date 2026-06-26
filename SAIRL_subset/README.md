# SAIRL_subset：SAIRL 长尾安全评估

`SAIRL_subset/` 在 `IDM_subset/` 的长尾评估管线上只替换 ego ADS：IDM ego 被替换为
SAIRL policy，其余 highD 长尾分布、冻结扩散模型、初始状态重构、EVT 阈值和风险评分保持同口径。

默认权重位于：

```text
SAIRL_subset/weights/sairl/model.npz
```

该文件是从参考 TensorFlow checkpoint 转换得到的本地 NPZ 权重。正常评估只加载该权重做推理，
不在 subset simulation 中训练或微调 SAIRL。

## 被测 ADS

SAIRL policy 当前实现为一个离散 MLP actor：

```text
input:  7 vehicles x [presence, x, y, vx, vy] = 35 dims
hidden: 96, 96, 96
output: 5 discrete highway-env MDP actions
```

动作空间保留原始 5 个离散动作：

```text
0 LANE_LEFT
1 IDLE
2 LANE_RIGHT
3 FASTER
4 SLOWER
```

因此 SAIRL 在 cut-in 测试中允许换道避撞；following 仍由 YAML 固定为 1 车道，cut-in 为
2 车道。

若只有外部 TensorFlow checkpoint，可显式转换：

```bash
python SAIRL_subset/scripts/convert_sairl_checkpoint.py
```

## 公平性约束

```text
same highD tail scenario-condition distribution
same diffusion checkpoint and DDIM sampler
same initial-state reconstruction from empirical tail contexts
same EVT/GPD failure threshold
same closed-loop risk scoring
same final-level playback selection
```

与 IDM 的主要差异只在 ego ADS。当前 SAIRL following 默认 `subset_simulation.num_samples = 3000`、
Monte Carlo `num_samples = 20000`；cut-in 默认 `num_samples = 1000`、Monte Carlo
`num_samples = 20000`。

## 运行

```bash
conda activate tread
```

following：

```bash
python SAIRL_subset/scripts/run_monte_carlo_following.py
python SAIRL_subset/scripts/run_subset_following.py
python SAIRL_subset/scripts/play_final_level_following.py --no-gif
```

cut-in：

```bash
python SAIRL_subset/scripts/run_monte_carlo_cutin.py
python SAIRL_subset/scripts/run_subset_cutin.py
python SAIRL_subset/scripts/play_final_level_cutin.py --no-gif
```

常用覆盖项：

```bash
python SAIRL_subset/scripts/run_subset_following.py \
  --checkpoint_path SAIRL_subset/weights/sairl/model.npz \
  --seed 42 \
  --num_samples 3000
```

`--seed` 会同时覆盖 simulation/context seed 和 SAIRL policy seed。

## 当前结果

```text
following subset:      p = 0.17946667, se = 0.00110831
following Monte Carlo: p = 0.16855000
cut-in subset:         p = 0.03530000, se = 0.00151126
cut-in Monte Carlo:    p = 0.03610000
```

压缩结果写入：

```text
SAIRL_subset/results/following/sairl_following_result.json
SAIRL_subset/results/cutin/sairl_cutin_result.json
```

完整诊断仍保留在同目录的 `latent_subset_summary.json`、`latent_subset_level_stats.csv`、
`latent_subset_samples.npz`、`latent_subset_top_cases.json` 和
`global_risk_exposure_comparison.*`。

## 主要文件

```text
SAIRL_subset/policies/sairl_policy.py
SAIRL_subset/src/closed_loop_runner.py
SAIRL_subset/src/result_payload.py
SAIRL_subset/scripts/convert_sairl_checkpoint.py
SAIRL_subset/scripts/configs/latent_subset_following.yaml
SAIRL_subset/scripts/configs/latent_subset_cutin.yaml
SAIRL_subset/weights/sairl/model.npz
tools/subset_entrypoints.py
```

`convert_sairl_checkpoint.py` 不是评估入口，只用于从参考 checkpoint 重建本地 NPZ 权重。

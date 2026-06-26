# IDM_subset：IDM 长尾安全评估

`IDM_subset/` 是当前长尾安全评估的传统模型 baseline。它在同一 highD
长尾 scenario-condition 分布、同一冻结扩散模型和同一 EVT 风险阈值下，评估
IDM ego vehicle 对扩散生成 adversary 轨迹的闭环响应。

测试空间为：

```text
scenario condition ~ highD tail scenario-condition distribution
z ~ N(0, I)
adversary actions = deterministic DDIM(scenario condition, z)
score = S_EVT(Y_sim)
```

扩散模型只用于生成被测场景中的对手车动作；被测 ADS 是 highway-env 中由
`tools/idm_ego.yaml` 参数化的 IDM ego。following 使用 125 步、1 车道配置；
cut-in 使用 100 步、2 车道配置。两类事件都不做 rolling reconditioning。

默认 diffusion checkpoint：

```text
following: results/diffusion_natural/following/checkpoints/best_noise_mse_train_val_test.pt
cut-in:    results/diffusion_natural/cutin/checkpoints/best_noise_mse_train_val_test.pt
```

配置指向不存在的 checkpoint 时会直接报错，不维护旧权重 fallback。

## 评分口径

following 使用闭环轨迹的 `Y_long_sim`，cut-in 使用 `Y_cutin_sim`，再分别通过
highD EVT/GPD 模型映射为 `S_EVT(Y_sim)`。默认失效事件为：

```text
Y_sim > x_c, x_c = 5.0
failure threshold = S_EVT(x_c)
```

该失效概率表示：

```text
P_ADS(failure | sampled from highD tail scenario-condition distribution)
```

它不是完整 highD 自然驾驶分布上的直接碰撞概率。完整数据集暴露强度由
`global_risk_exposure_comparison.*` 通过 highD independent tail peak 暴露率映射。

## 运行

运行环境：

```bash
conda activate tread
```

following：

```bash
python IDM_subset/scripts/run_monte_carlo_following.py
python IDM_subset/scripts/run_subset_following.py
python IDM_subset/scripts/play_final_level_following.py --no-gif
```

cut-in：

```bash
python IDM_subset/scripts/run_monte_carlo_cutin.py
python IDM_subset/scripts/run_subset_cutin.py
python IDM_subset/scripts/play_final_level_cutin.py --no-gif
```

当前 IDM subset 入口只接受 `--config`；实验默认值写在 YAML 中。Monte Carlo 入口可覆盖
`--num_samples`、`--seed` 和 `--output_dir`。

## 当前默认配置

```text
following:
  lanes_count = 1
  episode_steps = 125
  subset num_samples = 3000, p0 = 0.2, max_levels = 8
  monte_carlo num_samples = 200000

cut-in:
  lanes_count = 2
  episode_steps = 100
  subset num_samples = 1000, p0 = 0.1, max_levels = 8
  monte_carlo num_samples = 10000
```

## 当前结果

```text
following subset:      p = 0.00249067, se = 0.00006763
following Monte Carlo: p = 0.00241000, se = 0.00010964
cut-in subset:         p = 0.00680000, se = 0.00079609
cut-in Monte Carlo:    p = 0.00650000, se = 0.00080360
```

## 主要文件

```text
IDM_subset/scripts/configs/latent_subset_following.yaml
IDM_subset/scripts/configs/latent_subset_cutin.yaml
IDM_subset/scripts/run_subset_following.py
IDM_subset/scripts/run_subset_cutin.py
IDM_subset/scripts/run_monte_carlo_following.py
IDM_subset/scripts/run_monte_carlo_cutin.py
IDM_subset/scripts/play_final_level_following.py
IDM_subset/scripts/play_final_level_cutin.py
IDM_subset/src/closed_loop_runner.py
IDM_subset/src/latent_subset_runner.py
IDM_subset/src/subset_simulation.py
IDM_subset/src/final_level_playback.py
tools/subset_entrypoints.py
tools/idm_ego.yaml
```

`tools/subset_entrypoints.py` 提供三类共享 CLI 入口：subset simulation、Monte Carlo
baseline 和 final-level playback。各 subset 目录中的脚本只保留事件类型、默认配置路径和输出命名。

## 输出

```text
IDM_subset/results/following/latent_subset_summary.json
IDM_subset/results/following/global_risk_exposure_comparison.json
IDM_subset/results/following/global_risk_exposure_comparison.csv
IDM_subset/results/following/latent_subset_level_stats.csv
IDM_subset/results/following/latent_subset_top_cases.json
IDM_subset/results/following/latent_subset_samples.npz
IDM_subset/results/following/final_level_playbacks/
IDM_subset/results/monte_carlo_following/latent_monte_carlo_summary.json
IDM_subset/results/monte_carlo_following/latent_monte_carlo_stats.csv
IDM_subset/results/monte_carlo_following/latent_monte_carlo_samples.npz
IDM_subset/results/cutin/latent_subset_summary.json
IDM_subset/results/cutin/global_risk_exposure_comparison.json
IDM_subset/results/cutin/global_risk_exposure_comparison.csv
IDM_subset/results/cutin/latent_subset_level_stats.csv
IDM_subset/results/cutin/latent_subset_top_cases.json
IDM_subset/results/cutin/latent_subset_samples.npz
IDM_subset/results/cutin/final_level_playbacks/
IDM_subset/results/monte_carlo_cutin/latent_monte_carlo_summary.json
IDM_subset/results/monte_carlo_cutin/latent_monte_carlo_stats.csv
IDM_subset/results/monte_carlo_cutin/latent_monte_carlo_samples.npz
```

`latent_subset_samples.npz`、`latent_monte_carlo_samples.npz` 和
`final_level_playbacks/` 是可复现实验产物，可由对应脚本重建。

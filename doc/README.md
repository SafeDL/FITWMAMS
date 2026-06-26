# TREAD

TREAD 是一个基于 highD 自然驾驶数据的长尾安全测试实验工程。当前主线覆盖
following 和 cut-in 两类事件，把 highD 事件抽取、自然动作 diffusion prior、
peak-level EVT 标定和 latent-space subset simulation 串成可复现实验链路。

```text
process_highD/   highD 事件抽取、风险缓存、EVT 拟合、exposure 和 tail context 构造
diffusion/       训练 highD following/cut-in 场景的自然动作扩散先验
IDM_subset/          在 context + diffusion latent 空间中执行闭环 subset simulation
tools/           共享 IO、风险评分、EVT、context、exposure、绘图和 IDM 配置
```

`ad-rss-lib/` 不是当前主流程依赖；当前风险评分和历史 proxy-risk 辅助函数集中在
`tools/risk.py`。

## Goal 文档

用于后续 `/goal` 任务的约束模板：

```text
doc/following_diffusion_goal.md
doc/cutin_diffusion_goal.md
```

## 运行环境

默认从仓库根目录运行：

```bash
conda activate tread
```

主流程脚本基本不暴露 CLI 参数，实验默认值写在脚本常量或配置文件中：

```text
process_highD/scripts/configs/highd_default.yaml
diffusion/scripts/configs/natural_following.yaml
diffusion/scripts/configs/natural_cutin.yaml
IDM_subset/scripts/configs/latent_subset_following.yaml
IDM_subset/scripts/configs/latent_subset_cutin.yaml
```

highD 原始 CSV 默认读取：

```text
highD_dataset/Matlab/data/
```

即包含 `XX_tracks.csv`、`XX_tracksMeta.csv`、`XX_recordingMeta.csv` 的目录。

## 风险口径

following 使用纵向风险变量 `Y_long`。它由 `TTC`、`THW`、`gap`、`DRAC` 的
softmax pooling 项组成，并叠加 collision、near-collision 和 hard-brake 项：
在 highD 暴露率建模中，`Y_long` 基于完整不等长原始 following 事件计算；125 步定长
窗口只用于 diffusion 数据、长尾条件采样和回放对齐。

```text
longitudinal_proxy =
  w_ttc * softmax_pool(1 / TTC)
+ w_thw * softmax_pool(1 / THW)
+ w_gap * softmax_pool(1 / gap)
+ w_drac * softmax_pool(DRAC)

Y_long =
  longitudinal_proxy
+ collision_bonus * collision
+ near_collision_weight * near_collision
+ hard_brake_weight * hard_brake
```

cut-in 使用统一的 `Y_cutin`。语义上先要求目标车横向进入 ego 车道并保持在目标车道，
且 `require_front_at_cutin: true` 时，切入时刻目标车必须位于 ego 前方；不满足该
cut-in 门控的轨迹默认记为 `Y_cutin = 0`。通过门控后，纵向安全风险从切入帧
`risk_start_index` 一直计算到驾驶事件窗口结束，并与 following 使用同一套
`Y_long` 公式和权重；横向风险只额外加入切入后短窗口内的 LTG 项
（默认 `ltg_window_steps=5`）。EVT 在 decluster 后的 following `Y_long` peaks 和 cut-in
`Y_cutin = Y_long + LTG` peaks 上分别拟合 POT/GPD。闭环仿真的输出分数为：

```text
risk_score = S_EVT(Y_sim) = -log P_EVT(Y > Y_sim)
```

这个分数表示相对 highD 自然 peak 尾部分布的极端程度，不是 ADS 碰撞概率。

当前 IDM_subset 默认失效目标为：

```text
Y_sim > x_c,  x_c = 5.0
failure_threshold = S_EVT(x_c)
```

该目标由 `IDM_subset/scripts/configs/latent_subset_following.yaml` 或
`IDM_subset/scripts/configs/latent_subset_cutin.yaml` 中的 `evt` 配置决定。following 和
cut-in 当前默认都使用 `collision_critical_level: 5.0`。

## 数据与默认过滤

`process_highD/scripts/extract_highd_events.py` 抽取 following 和 cut-in 事件，并
同步写出风险缓存和 context 缓存。默认配置中：

- 采样频率为 `25 Hz`。
- 事件窗口长度为 `128` 帧。
- following 需要同一 preceding vehicle 持续 `128` 帧。
- cut-in 原始事件筛选要求目标车完成相邻车道切入、cross 后多数时间在 ego 前方且与
  ego 同车道，并默认要求目标车是 ego 在目标车道的最近前车；后续风险评分还会再次
  用 `is_cutin` 和 `is_front_cutin` 门控长尾风险。
- 默认要求 ego 和 lead 都是 passenger car。
- diffusion 数据集按 recording id 划分，避免同一 recording 的窗口跨 split 泄漏。
  following 默认 `train / val / test = 0.70 / 0.15 / 0.15`，cut-in 默认
  `0.75 / 0.15 / 0.10`，随机种子均为 `42`。

`diffusion/` 只学习自然动作分布，不使用安全分数作为训练目标。following 默认动作
表示为 lead vehicle jerk；cut-in 默认动作表示为 target vehicle `[ax, ay]`。两条
链路都使用 anchor-frame `scenario_conditions` 作为唯一 diffusion 条件，该向量包含
初始关系和参考窗口的压缩动作/轨迹摘要；不再输入 rolling `context_states`、
`context_features` 或 `relative_history`。训练目标为 DDPM noise prediction，推理和
IDM_subset 中使用 DDIM deterministic sampling：

```text
same scenario condition + same latent z -> same action trajectory
```

## Tail Contexts

following 和 cut-in 的 tail context 构建入口按场景拆开：

```text
following: context_source = independent_tail_peaks
cut_in:    context_source = independent_tail_peaks
following context_generation_method = gaussian_copula
cut-in condition_generation_method = gaussian_copula
include_empirical_contexts = true
empirical_context_limit = null
num_synthetic_contexts = 5000
```

含义是：

1. following 从 `highd_independent_tail_peaks.csv` 中取 decluster 后的
   highD independent tail peaks。
2. cut-in 从 `highd_independent_tail_peaks.csv` 中取 decluster 后的
   highD independent tail peaks。
3. `empirical_context_limit = null` 表示保留全部 matched empirical tail
   contexts；若设为正整数，则先无放回抽取对应数量的 empirical contexts。
4. following 默认额外生成 `5000` 个 synthetic contexts。生成方式是在 tail
   `scenario_conditions` 的联合分布上拟合 Gaussian copula，再用最近邻 empirical
   context 重构 `initial_states`。
5. cut-in 默认写出 `scenario_condition_distribution.npz`，并用同一套条件分布、
   cut-in diffusion prior 输出 `5000` 个 generated scenarios；默认先采样 `5000`
   个 conditions，如果语义硬筛选后不足，再从同一联合分布补采样并继续解码，直到
   达到目标数量或达到 refill 上限。语义 cut-in、车道保持、front-at-entry 和
   collision-free 等检查作为后处理 masks/指标写入 generated 文件和 summary。
   默认不再因语义检查把 sampled conditions 预先按固定倍率隐式放大。
6. 输出中用 `source_type`、`synthetic_context`、`context_model_method`、
   `base_event_id` 和 `context_feature_distance` 区分 empirical 与 synthetic
   contexts。

因此 IDM_subset 默认估计的是：

```text
P_context,z(Y_sim > x_c | context sampled from highD tail scenario-condition distribution)
```

而不是严格的：

```text
P_context,z(Y_sim > x_c | context in finite empirical highD tail peaks)
```

这样做的目的，是缓解纯 empirical 少量失效样本导致的 Markov chain 坍缩；解释结果时
应把 context 分布理解为 highD tail scenario-condition 联合分布的平滑近似。若需要最干净的经验分布
解释，可把 `context_generation_method` 改为 `empirical`，并保留
`empirical_context_limit = null`。

following 重建从 tail context 的 `initial_states` 开始，由 diffusion prior 一次性生成
125 步 lead jerk。cut-in diffusion 泛化先从 tail scenario-condition 联合分布采样
条件；其中的初始 gap、相对速度、横向偏移、目标车初始横纵向速度/加速度决定
`initial_states` 中可由 condition 表达的部分，其余几何/未建模状态沿用最近邻 highD
tail 事件重构。`initial_states` 只用于 target `[ax, ay]` 轨迹积分、评价和后处理，
不作为 denoiser 条件输入。

## 推荐运行顺序

1. 抽取 highD 事件、following 风险缓存和 exposure per recording：

```bash
python process_highD/scripts/extract_highd_events.py
```

2. 构建 diffusion 数据集、训练自然先验、评估自然性：

```bash
python process_highD/scripts/build_natural_dataset.py
python diffusion/scripts/train_following_diffusion.py
python diffusion/scripts/evaluate_following_prior.py
```

3. 拟合 highD following peak EVT、估计 exposure、构造 tail contexts：

```bash
python process_highD/scripts/estimate_following_exposure.py
python process_highD/scripts/select_following_tail_contexts.py
```

4. 执行 latent-space subset simulation：

```bash
python IDM_subset/scripts/run_monte_carlo_following.py
python IDM_subset/scripts/run_subset_following.py
python IDM_subset/scripts/play_final_level_following.py --no-gif
```

当前 following IDM_subset 默认使用 `num_samples=10000, p0=0.1, max_levels=8` 并开启
adaptive stop。following diffusion 噪声空间为 `[125, 1] = 125` 维；加上 7 维
scenario conditions，联合输入空间为 132 维。运行入口会在结束日志中打印实际闭环仿真
evaluator 调用次数和唯一 scenario context 数。

cut-in 分支：

```bash
python process_highD/scripts/extract_highd_events.py
python process_highD/scripts/build_natural_dataset.py \
  --config diffusion/scripts/configs/natural_cutin.yaml
python diffusion/scripts/train_cutin_diffusion.py
python diffusion/scripts/evaluate_cutin_prior.py
python process_highD/scripts/estimate_cutin_exposure.py
python process_highD/scripts/select_cutin_tail_contexts.py
python IDM_subset/scripts/run_monte_carlo_cutin.py
python IDM_subset/scripts/run_subset_cutin.py
```

当前 cut-in 长尾重建主输出是
`results/highd_cutin_tail/generated/diffusion_generated_scenarios.npz` 及其图表/GIF；
当前 cut-in MC 默认使用 10000 个独立样本；cut-in IDM_subset 默认使用
`num_samples=1000, p0=0.1, max_levels=8` 并开启 adaptive stop。`max_levels=8`
是最大允许层数；当前 `x_c=5` 目标下通常在 2 层后因失效样本数足够而停止，以避免过深
条件化造成 final-level scenario context 坍缩。两个入口都会在结束日志中打印实际闭环仿真
evaluator 调用次数和唯一 scenario context 数。cut-in 扩散噪声空间为 `[100, 2] = 200`
维；加上 10 维 scenario conditions，联合输入空间为 210 维。

可选回放：

```bash
python process_highD/scripts/play_following_tail_events.py
python process_highD/scripts/play_cutin_tail_events.py
python IDM_subset/scripts/play_final_level_following.py
python IDM_subset/scripts/play_final_level_cutin.py
```

`play_following_tail_events.py` 和 `play_cutin_tail_events.py` 不暴露 CLI 配置，
分别回放 following/cut-in diffusion 泛化场景并输出 GIF。
`SCENARIO_SELECTION = "all"` 表示全部场景；设为整数表示按
`RANDOM_SEED` 随机采样这么多个场景；设为 tuple/list 表示精确指定 generated index。
播放使用生成 NPZ 中的等长轨迹，同时用 `base_event_id` 反查 highD 记录补充动态背景车。共享回放逻辑位于
`process_highD/src/event_playback.py`。

`play_final_level_following.py` 和 `play_final_level_cutin.py` 复现 subset simulation
最后一层发现的危险闭环样本。它们读取 `latent_subset_samples.npz` 中最后一层保存的
`scenario_conditions`、`initial_states`、`context_index`、latent 解码后的 `actions`
和 `action_mask`，并用 `latent_subset_summary.json` 中的 `failure_threshold` 筛选
`score >= failure_threshold` 的案例。默认按完整测试输入去重，而不是按
`context_index` 去重；同一 empirical context 下的不同 latent/action plan 仍可作为不同
危险测试场景。脚本随后用固定随机种子无放回抽取案例，默认
`--num-cases 10 --random-seed 42`。输出目录默认为
`results/subset_simulation_{following,cutin}/final_level_playbacks/`。

## 主要输出

```text
results/highd_events/events.csv
results/highd_events/following_event_scores.csv
results/highd_events/following_event_contexts.npz
results/highd_events/following_event_segments.npz
results/highd_events/cutin_event_scores.csv
results/highd_events/cutin_event_contexts.npz
results/highd_events/following_event_cache_summary.json
results/highd_events/exposure_per_recording.csv

results/diffusion_natural/following/dataset.npz
results/diffusion_natural/following/dataset_normalized.npz
results/diffusion_natural/following/feature_schema.json
results/diffusion_natural/following/normalization_stats.json
results/diffusion_natural/following/train_val_test_split.json
results/diffusion_natural/following/checkpoints/best_noise_mse_train_val_test.pt
results/diffusion_natural/following/checkpoints/final_train_val_test.pt
results/diffusion_natural/following/training_summary.json
results/diffusion_natural/following/tensorboard/
results/diffusion_natural/following/naturalness_summary.json
results/diffusion_natural/cutin/checkpoints/best_noise_mse_train_val_test.pt
results/diffusion_natural/cutin/checkpoints/final_train_val_test.pt

results/highd_following_tail/evt/longitudinal_peak_evt_model.json
results/highd_following_tail/evt/longitudinal_peak_evt_summary.json
results/highd_following_tail/exposure/highd_exposure_summary.json
results/highd_following_tail/exposure/highd_independent_tail_peaks.csv
results/highd_following_tail/contexts/scenario_condition_distribution.npz
results/highd_following_tail/contexts/tail_contexts.npz
results/highd_following_tail/contexts/tail_context_summary.json
results/highd_following_tail/generated/diffusion_generated_scenarios.npz
results/highd_following_tail/generated/figures/
results/highd_following_tail/generated/event_playbacks/
results/highd_cutin_tail/evt/cutin_peak_evt_model.json
results/highd_cutin_tail/evt/cutin_peak_evt_summary.json
results/highd_cutin_tail/exposure/highd_cutin_exposure_summary.json
results/highd_cutin_tail/exposure/highd_independent_tail_peaks.csv
results/highd_cutin_tail/contexts/scenario_condition_distribution.npz
results/highd_cutin_tail/contexts/scenario_condition_distribution_summary.json
results/highd_cutin_tail/generated/diffusion_generated_scenarios.npz
results/highd_cutin_tail/generated/figures/
results/highd_cutin_tail/generated/event_playbacks/

results/subset_simulation_following/latent_subset_summary.json
results/subset_simulation_following/latent_subset_level_stats.csv
results/subset_simulation_following/latent_subset_top_cases.json
results/subset_simulation_following/global_risk_exposure_comparison.json
results/subset_simulation_following/global_risk_exposure_comparison.csv
results/subset_simulation_following/latent_subset_samples.npz
results/subset_simulation_following/final_level_playbacks/
results/monte_carlo_following/latent_monte_carlo_summary.json
results/monte_carlo_following/latent_monte_carlo_stats.csv
results/monte_carlo_following/latent_monte_carlo_top_cases.json
results/monte_carlo_following/latent_monte_carlo_samples.npz
results/subset_simulation_cutin/latent_subset_summary.json
results/subset_simulation_cutin/latent_subset_level_stats.csv
results/subset_simulation_cutin/latent_subset_top_cases.json
results/subset_simulation_cutin/global_risk_exposure_comparison.json
results/subset_simulation_cutin/global_risk_exposure_comparison.csv
results/subset_simulation_cutin/latent_subset_samples.npz
results/subset_simulation_cutin/final_level_playbacks/
results/monte_carlo_cutin/latent_monte_carlo_summary.json
results/monte_carlo_cutin/latent_monte_carlo_stats.csv
results/monte_carlo_cutin/latent_monte_carlo_top_cases.json
results/monte_carlo_cutin/latent_monte_carlo_samples.npz

results/paper_experiments/following/following_gpd_diagnostic_panel.png
results/paper_experiments/following/following_safety_threshold_inverse_calibration.png
results/paper_experiments/following/following_tail_diffusion_generalization_panel.png
results/paper_experiments/following/following_subset_level_score_histograms.png
results/paper_experiments/cutin/cutin_gpd_diagnostic_panel.png
results/paper_experiments/cutin/cutin_safety_threshold_inverse_calibration.png
results/paper_experiments/cutin/cutin_tail_diffusion_generalization_panel.png
results/paper_experiments/cutin/cutin_subset_level_score_histograms.png
```

`results/highd_events/cutin_event_contexts.npz` 保存变长度 raw cut-in event
context。每个事件的 anchor 从 `[15,20,25,30,35,45,50]` 帧 pre-cross 候选中确定，
并保证 `cross_frame` 后至少 2 秒；`Y_cutin` 基于这段变长度真实轨迹评分。cut-in
diffusion 数据集再整理为固定 100 帧训练窗口，且同样保证 cross 后不少于 2 秒。

`latent_subset_summary.json` 中最重要的字段是：

- `probability`: subset 估计的条件失效概率。
- `probability_target`: 该概率对应的 context 分布解释。
- `probability_estimate_kind`: 标准估计、低可靠性标准估计或 guarded 诊断估计。
- `reliability`: final level 的 unique context/state、最大占比和 acceptance rate 诊断。
- `mileage_return_period`: 在 strictness 条件满足时，把条件概率乘以 highD tail peak
  exposure rate 后得到的里程或时间回报周期。

cut-in Monte Carlo 基线由 `IDM_subset/scripts/run_monte_carlo_cutin.py` 运行。
它与 cut-in subset 使用同一个 scenario-condition 联合分布和 diffusion latent 空间，
但只做独立同分布直接采样，用于对比 subset simulation 的稀有事件估计效率。

论文图由 `results/build_following_paper_experiments.py` 和
`results/build_cutin_paper_experiments.py` 从已有 JSON/CSV/NPZ/PNG 结果重建，不重训模型、
不重跑 EVT 或 subset simulation。following diffusion generalization panel 复用
`process_highD` 的 tail condition/segment cache 口径；所有 paper figures 统一使用
`tools/plot_style.py` 中的 300 dpi、Times-compatible serif 和 STIX mathtext 样式。阈值反解图中
$L^\star$ 和 $x^\star_e$ 使用不同线型/颜色，避免把目标重现里程和反解风险阈值混为同一含义。

## 子集模拟可靠性

`IDM_subset` 使用标准 subset simulation 概率估计，并同时输出可靠性诊断。如果 final
level 的 unique context/state 太少、最大 context/state 占比过高，或 MH acceptance
rate 过低，`strict_probability_interpretation` 会变为 `false`。

## 版本控制约定

大型模型、数组、逐样本结果和可再生成文件应由 `.gitignore` 忽略。README 中列出的
结果路径是运行产物说明，不表示这些文件都应该提交。

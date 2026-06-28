# highD 长尾驾驶事件 Normalizing Flow 建模

`normalizing/` 对 `process_highD/` 中 EVT 已筛选出的 highD 长尾自然驾驶事件建模，并从该联合分布中采样新的长尾事件。它不是尾部事件分类器，也不重新学习 EVT 风险分层；训练集已经全部是 EVT-tail 事件。

默认生成逻辑是无外部用户条件的联合采样：

```text
p(event | EVT-tail) = p_hat(e | EVT-tail) * p_flow(y | e, EVT-tail)
```

其中 `e` 是离散事件结构，`y` 是连续局部交通状态和未来 1 秒轨迹摘要。默认采样时，程序先从 EVT-tail 数据集的经验离散结构分布中抽样 `e`，再用 conditional normalizing flow 抽样连续变量 `y`。因此下游仿真不需要预先提供 context。

## 入口脚本

当前只保留三个入口，避免重复 wrapper：

```bash
python normalizing/scripts/prepare_highd_tail_flow_dataset.py
python normalizing/scripts/train_highd_tail_flow.py
python normalizing/scripts/evaluate_highd_tail_flow.py
```

职责划分：

- `prepare_highd_tail_flow_dataset.py`：从上游 EVT 长尾自然驾驶事件中整理 flow 数据集；tail context CSV 不存在或要求重建时，会调用上游 EVT tail context 选择逻辑。
- `train_highd_tail_flow.py`：只训练 conditional MAF，写 checkpoint、`training_history.csv`、`training_summary.json` 和 TensorBoard 训练曲线。
- `evaluate_highd_tail_flow.py`：计算 NLL/基线，生成等量样本，写样本接口，绘制诊断图，写 `evaluation_summary.json`。

## 建模目标

本模块服务两个下游用途：

- 生成 ADS 场景测试的初始局部交通配置。
- 为世界模型 START 条件提供第一秒轨迹级别摘要，使世界模型继续生成所有交通车未来动作。

模型不生成完整未来轨迹。它只生成每个活跃交通车 slot 的 1 秒动作摘要，完整轨迹仍由世界模型滚动生成。

## 不学习的内容

以下信息不作为 normalizing flow 的随机变量学习：

- 车辆尺寸、车道宽度等 highD 固定几何信息。
- `event_risk` 或 EVT 风险分层，因为训练集已经全部是 EVT-tail 事件。
- `lane_count`、`ego_lane_ordinal`、左右车道存在性等 lane topology 条件。
- ego lane offset 和 ego speed 标量。ego 默认位于当前车道中心，速度可由 `ego_vx_mps`、`ego_vy_left_mps` 推出。

lane topology 不写入 normalizing flow 数据集；如需回放或道路几何审计，应回到 `process_highD/` 的上游结果。Flow 只使用 slot mask 和 primary slot 作为离散事件结构。

## 离散事件结构

离散事件结构是 12 维：

```text
6 个 slot mask bit + 6 个 primary-slot one-hot bit
```

slot 固定顺序：

```text
same_front
same_rear
left_front
left_rear
right_front
right_rear
```

`slot mask` 表示 anchor 时刻该语义 slot 是否存在交通车。`primary_slot` 来自 EVT 峰值交互对象，用于标识主要长尾交互对象；但轨迹摘要为所有活跃 slot 建模，不只建模 primary vehicle。

CSV 和图中的 `mask_pattern` 是 6 个 mask bit 的整数编码：

```text
mask_pattern = sum(mask_i * 2**i)
```

它只是紧凑审计标签，便于统计和画图；整数大小没有连续物理意义。

## Anchor 时刻

anchor 时刻是一个 150 帧自然驾驶片段的起点。`process_highD/` 当前配置为 25 Hz、6 秒窗口，因此每个片段长度为 150 帧。EVT 风险轨迹在窗口内计算，事件风险取窗口内风险峰值；normalizing flow 在 anchor 时刻抽取初始局部状态，并额外抽取 anchor 后 1 秒内的交通车动作摘要。

## 连续特征

当前连续目标向量为 76 维。

Ego 动力学 4 维：

```text
ego_vx_mps
ego_vy_left_mps
ego_ax_mps2
ego_ay_left_mps2
```

每个 slot 的 anchor 初始局部状态 6 维，共 36 维：

```text
<slot>_rel_x_m
<slot>_rel_y_left_m
<slot>_rel_vx_mps
<slot>_rel_vy_left_mps
<slot>_other_ax_mps2
<slot>_other_ay_left_mps2
```

每个 slot 的未来 1 秒动作摘要 6 维，共 36 维：

```text
<slot>_delta_vx_1s_mps
<slot>_delta_vy_left_1s_mps
<slot>_mean_ax_1s_mps2
<slot>_min_ax_1s_mps2
<slot>_final_ax_1s_mps2
<slot>_mean_ay_left_1s_mps2
```

`final_ax_1s_mps2` 替代了旧的 `braking_duration_1s`。后者是 25 Hz 阈值累计量，离散台阶明显，不适合连续 normalizing flow；`final_ax_1s_mps2` 是连续 endpoint 动作状态，和 `mean_ax/min_ax/delta_vx` 组合后更适合世界模型恢复第一秒纵向动作趋势。

非活跃 slot 在固定宽度 tensor 中写为 0，并通过 `slot_mask` 和 `feature_valid` 区分。不要把非活跃 slot 的 0 当成真实车辆状态。

## Flow 技术路线

默认模型是 conditional Masked Autoregressive Flow，连续变换层使用 rational-quadratic spline：

```text
model.type: conditional_maf
model.transform_type: rq_spline
num_layers: 6
hidden_features: 160
num_bins: 8
```

连续 flow 学习：

```text
p_flow(y | slot_mask, primary_slot, EVT-tail)
```

完整联合概率在导出时加上经验离散结构概率：

```text
joint_log_prob = continuous_log_prob + event_structure_log_prob
```

训练前会做确定性模型坐标变换，再做均值/方差归一化。原始审计特征仍保存在 `dataset.npz` 中。

当前变换统计：

```text
identity:                              70
positive_mean_minus_min_ax_softplus:    6
```

`positive_mean_minus_min_ax_softplus` 用于稳定表达 `mean_ax_1s_mps2 - min_ax_1s_mps2 >= 0`。

## NLL 解释

这里的 NLL 是连续密度的 `-log p(x)`，不是离散概率的 `-log P(x)`。连续概率密度可以大于 1，所以 `log p(x)` 可以为正，NLL 就可以为负。不同特征变换、归一化坐标和维度会改变 NLL 的绝对数值；因此 NLL 主要用于同一数据坐标、同一 split 下比较模型，而不是按“必须为正”来解释。

## 输出文件

默认输出目录：

```text
results/highd_tail_normalizing_flow/
```

核心文件：

```text
dataset.npz
dataset_schema.json
checkpoints/best_tail_conditional_maf.pt
checkpoints/baseline_realnvp.pt
training_history.csv
training_summary.json
evaluation_summary.json
diagnostics/nll_comparison.csv
diagnostics/feature_distribution_metrics.csv
diagnostics/joint_probability_scores.csv
diagnostics/visualization_summary.json
samples/generated_samples.npz
samples/generated_samples.csv
samples/generated_samples_interfaces.json
```

审计图：

```text
figures/tail_c0_marginal_distributions.png
figures/tail_c0_all_marginal_distributions.png
figures/tail_c0_all_feature_distribution_errors.png
figures/tail_c0_joint_probability_tail_vs_generated.png
figures/tail_c0_correlation_tail_vs_generated.png
figures/tail_c0_all_correlation_tail_vs_generated.png
figures/tail_c0_probability_diagnostics.png
figures/tail_c0_context_occupancy_tail_vs_generated.png
```

图表含义：

- `tail_c0_marginal_distributions.png`：关键变量的一维边缘分布。
- `tail_c0_all_marginal_distributions.png`：全部 76 个连续变量的一维边缘分布网格。
- `tail_c0_all_feature_distribution_errors.png`：全部 76 个变量的 KS 和 Wasserstein 误差条形图。
- `tail_c0_joint_probability_tail_vs_generated.png`：关键变量的两两联合散点和 KDE 轮廓。
- `tail_c0_correlation_tail_vs_generated.png`：关键变量 Pearson 相关矩阵。
- `tail_c0_all_correlation_tail_vs_generated.png`：全部 76 维相关矩阵、生成矩阵和差值矩阵。
- `tail_c0_probability_diagnostics.png`：联合 log probability、NLL CDF、主要 mask pattern 下的 NLL 分布，以及 held-out test NLL 基线对比。
- `tail_c0_context_occupancy_tail_vs_generated.png`：slot mask pattern 和 primary slot 的离散占用率对比。

## 世界模型接口

`samples/generated_samples_interfaces.json` 中每条样本包含：

- `ads_initialization`：ego 和活跃背景车的局部坐标初始配置。
- `world_model_start_condition`：`mode="START"`，包含 ego 状态、每个 slot 的初始状态、非活跃 slot 的 `null`、每个活跃交通车的 `action_1s_summary`，以及总表 `traffic_action_1s_summary`。

`primary_interaction_1s_summary` 只是 primary slot 的便捷视图，不是唯一动作条件。世界模型应优先读取 `traffic_action_1s_summary`，因为它覆盖所有活跃交通车。

## 当前实验结果

当前审计结果来自 `results/highd_tail_normalizing_flow/`。

数据集：

```text
EVT-tail 样本数:        2209
连续特征维度:           76
离散事件结构维度:       12
split:                 train 1550 / val 330 / test 329
全量复现参考 split:     all
生成样本数:             2209
```

held-out continuous conditional NLL，越低越好：

```text
conditional rq-spline MAF    train -100.3875  val -68.2119  test -82.7773
GMM                          train -125.6069  val -43.3915  test -65.9791
Gaussian                     train    1.4248  val  16.2438  test   6.7505
Copula                       train   30.3386  val  36.5070  test  32.6475
Unconditional RealNVP        train   78.2931  val  81.1770  test  79.1017
```

结论：当前 conditional rq-spline MAF 在 held-out test NLL 上优于 GMM、Gaussian、Copula 和 unconditional RealNVP。GMM 的 train NLL 更低但 test NLL 较差，说明 GMM 对训练集拟合更强、泛化弱于当前 flow。

全量 2209 vs 2209 分布复现指标：

```text
mean per-feature KS:          0.1342
mean Wasserstein:             0.5439
Pearson corr MAE:             0.0556
mask occupancy L1:            0.0697
invalid_rate:                 0.0000
overlap_rate:                 0.0000
negative_gap_rate:            0.0000
semantic_error_rate:          0.0000
sampling rejection_rate:      0.2575
```

相对上一版改进：

- 移除 `braking_duration_1s` 后，最差 KS 不再由离散刹车时长主导。
- mean KS 从约 0.155 降到 0.134。
- mask occupancy L1 从约 0.097 降到 0.070。
- rejection rate 从约 0.301 降到 0.257。

主要剩余不足：

- 误差最大的变量集中在低频 `left_rear` slot。真实数据中该类样本较少，条件密度估计天然更不稳定。
- 全 76 维两两散点矩阵不可读，因此当前完整审计采用全边缘、全误差条形图和全相关矩阵；关键变量仍保留两两联合散点图。

因此，当前结果适合用于默认长尾测试初始条件和世界模型 START 条件。如果后续重点测试稀有侧后方交互，应考虑对低频 slot 做重采样增强或分 slot 校准。

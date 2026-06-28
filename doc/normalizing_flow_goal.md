# Normalizing Flow 长尾驾驶事件分布建模改进目标

## 1. 核心目标

本阶段的核心目标改为：

```text
尽最大可能提升 highD EVT 长尾驾驶事件的联合分布拟合效果，
并保证最终效果优于当前 normalizing/ 方法。
```

当前 [normalizing/](../normalizing/) 已经形成可运行基线：它对 EVT 筛选后的 highD 长尾自然驾驶事件建模，默认学习和采样：

```text
p(event | EVT-tail) = p_hat(e | EVT-tail) * p_flow(y | e, EVT-tail)
```

其中：

- `e` 是离散事件结构，包括 6 个 slot mask bit 和 6 个 primary-slot one-hot bit；
- `y` 是 76 维连续变量，包括 anchor 时刻局部交通状态和每个活跃 slot 的未来 1 秒动作摘要；
- 最终采样不要求用户预先提供 context，而是先采样 EVT-tail 离散事件结构，再采样连续状态。

本阶段不再以“实现一个可运行 flow”为终点，而是以**在严格 EVT-tail 测试集上的联合分布建模效果持续优于当前基线**为终点。

运行环境：

```bash
conda activate tread
```

默认从仓库根目录运行：

```bash
python normalizing/scripts/prepare_highd_tail_flow_dataset.py \
  --config normalizing/scripts/configs/highd_tail_flow.yaml
python normalizing/scripts/train_highd_tail_flow.py \
  --config normalizing/scripts/configs/highd_tail_flow.yaml
python normalizing/scripts/evaluate_highd_tail_flow.py \
  --config normalizing/scripts/configs/highd_tail_flow.yaml
```

---

## 2. 当前基线

当前基线位于：

```text
normalizing/
results/highd_tail_normalizing_flow/
```

当前模型：

```text
conditional Masked Autoregressive Flow
rational-quadratic spline transform
6 layers
160 hidden features
2 residual blocks per transform
dropout 0.02
8 spline bins
tail bound 4.0
batch norm disabled
```

当前 76 维连续目标：

```text
4 ego dynamics:
  ego_vx_mps
  ego_vy_left_mps
  ego_ax_mps2
  ego_ay_left_mps2

6 traffic slots:
  same_front
  same_rear
  left_front
  left_rear
  right_front
  right_rear

6 anchor local-state features per slot:
  rel_x_m
  rel_y_left_m
  rel_vx_mps
  rel_vy_left_mps
  other_ax_mps2
  other_ay_left_mps2

6 future-1s action-summary features per slot:
  delta_vx_1s_mps
  delta_vy_left_1s_mps
  mean_ax_1s_mps2
  min_ax_1s_mps2
  final_ax_1s_mps2
  mean_ay_left_1s_mps2
```

因此：

```text
4 + 6 * 6 + 6 * 6 = 76
```

当前 12 维离散事件结构 context 只包含：

```text
6 slot active mask bits
6 primary-slot one-hot bits
```

清理后的基线不再把以下内容作为 flow 学习目标或条件：

```text
vehicle dimensions
lane width
lane topology / lane existence
ego lane offset
EVT risk bin / risk level
```

`event_risk` 仅作为审计 metadata 保留在数据集中，不作为条件变量。inactive slot 的连续特征为 0 placeholder，必须通过 `slot_mask` 和 `feature_valid` 判断是否参与评估和接口恢复。

当前评估口径：

```text
EVT-tail samples:      2209
generated samples:     2209
reference split:       all
event-structure split: all
held-out split:        train 1550 / val 330 / test 329
sample count policy:   match_reference
invalid rejection:     enabled
```

当前基线结果：

```text
conditional rq-spline MAF    train -100.3875  val -68.2119  test -82.7773
GMM                          train -125.6069  val -43.3915  test -65.9791
Gaussian                     train    1.4248  val  16.2438  test   6.7505
Copula                       train   30.3386  val  36.5070  test  32.6475
Unconditional RealNVP        train   78.4393  val  81.3103  test  79.2735
```

当前全量 EVT-tail 复现指标：

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

后续任何新方法都必须保留这些指标作为对照，并在严格 EVT-tail 测试和全量 EVT-tail 复现上证明改进。

---

## 3. 当前主要问题

当前模型整体可用，但性能被少数低频 slot 和部分特征拖累。

按当前诊断结果，误差最大的 slot 是：

```text
left_rear   mean KS 0.253
left_front  mean KS 0.190
same_rear   mean KS 0.147
right_rear  mean KS 0.134
same_front  mean KS 0.040
right_front mean KS 0.052
```

最差特征主要集中在 `left_rear`：

```text
left_rear_mean_ax_1s_mps2
left_rear_other_ax_mps2
left_rear_min_ax_1s_mps2
left_rear_rel_vy_left_mps
left_rear_delta_vy_left_1s_mps
left_rear_delta_vx_1s_mps
```

Wasserstein 误差最大的特征主要是纵向相对距离：

```text
left_rear_rel_x_m
left_front_rel_x_m
right_rear_rel_x_m
same_rear_rel_x_m
```

这说明：

1. 主流 `same_front`、`right_front` 场景拟合较好；
2. 稀有侧后方、侧前方 slot 数据量不足；
3. 全局 likelihood 训练容易被高频 slot 主导；
4. `rel_x_m` 物理尺度较大，几米偏差会明显拉高 Wasserstein；
5. rejection sampling 会进一步放大稀有 slot 的占比偏差。

---

## 4. 关于是否使用全量自然驾驶数据

可以使用更多数据，但不建议直接用全量自然驾驶样本训练一个默认采样的无条件 flow。

直接训练会得到：

```text
p(x | natural)
```

而本阶段目标是：

```text
p(x | EVT-tail)
```

如果普通自然驾驶样本占绝大多数，无条件 flow 会被普通场景主导，默认采样也会更像普通交通，而不是 EVT-tail 事件。这会削弱长尾事件联合分布建模效果。

因此，本阶段推荐使用更多数据的方式是：

1. **全量自然驾驶预训练 + 严格 EVT-tail 微调**；
2. **扩大 near-tail 数据集训练 + 严格 EVT-tail 测试**；
3. **条件 flow 显式建模 risk/tail 条件，采样时固定 tail 条件**；
4. **自然分布模型 + tail density-ratio 或 tail classifier 重加权**。

不推荐：

```text
直接用 process_highD 全量自然驾驶片段训练无条件 flow，
然后把它当作 EVT-tail 生成模型。
```

---

## 5. 推荐改进路线

### 5.1 全量自然驾驶预训练 + EVT-tail 微调

目标：

```text
先学习通用交通几何、slot 关系和动力学范围，
再将最终分布拉回严格 EVT-tail。
```

训练流程：

```text
stage 1: train p_flow(x | natural)
stage 2: initialize from stage 1
stage 3: finetune p_flow(x | EVT-tail)
stage 4: evaluate only on EVT-tail
```

要求：

- stage 1 可以使用 `process_highD/` 产生的全量自然驾驶片段；
- stage 2/3 必须使用严格 EVT-tail 或加权 tail 数据；
- 最终 checkpoint 必须以 EVT-tail validation NLL 选择；
- 最终默认采样必须仍然表示 `p(event | EVT-tail)`。

预期收益：

- 改善低频 slot 的基础动力学拟合；
- 降低物理无效样本比例；
- 改善 `left_rear/left_front` 的加速度和相对速度分布。

风险：

- 如果微调不足，模型会保留普通 natural bias；
- 如果预训练和 tail schema 不一致，会引入接口复杂度；
- 需要严格记录最终 checkpoint 的训练阶段和数据来源。

验收：

- EVT-tail test NLL 必须优于当前 `-82.7773`；
- mean KS 必须优于当前 `0.1342`；
- `left_rear` mean KS 应明显下降；
- physical invalid rate 必须保持 `0.0` 或不显著变差。

### 5.2 扩大 near-tail 训练集

目标：

```text
增加接近 EVT 阈值的困难样本，缓解 2209 条严格 tail 数据不足。
```

可选策略：

- 使用 EVT threshold 附近的 exceedance-neighborhood；
- 使用风险分数 top-k，k 大于严格 tail 数量；
- 使用较低风险阈值构建 near-tail pool；
- 对严格 EVT-tail 样本赋更高权重。

训练目标可以写成：

```text
weighted NLL = sum_i w_i * -log p(x_i | e_i)
```

其中：

- 严格 EVT-tail 样本权重最高；
- near-tail 样本用于补充交通结构和低频 slot；
- 普通样本权重应显著低于 tail 样本。

验收：

- 只在严格 EVT-tail held-out 上选模型；
- 不允许用 near-tail 测试集替代严格 EVT-tail 测试；
- 默认采样分布仍必须接近严格 EVT-tail，而不是 near-tail 总体。

### 5.3 稀有 slot 重加权

目标：

```text
降低 high-frequency slots 对 likelihood 的支配，
提升 left_rear、left_front、same_rear 等低频 slot 的拟合。
```

推荐实现：

- 按 slot active count 计算权重；
- 按 mask pattern 计算权重；
- 对含有 `left_rear`、`left_front` 的样本提高 loss 权重；
- 使用 balanced minibatch，保证每个 batch 中稀有 slot 出现比例不至于过低。

注意：

- 重加权会改变训练目标；
- 必须报告加权前后的默认采样分布；
- 如果默认采样使用经验 event-structure 分布，则连续 flow 的重加权不应改变 `p_hat(e | EVT-tail)`。

验收：

- `left_rear`、`left_front` 的 KS/Wasserstein 明显下降；
- 总体 mean KS 不得变差；
- mask occupancy L1 不得显著变差。

### 5.4 分 slot 或共享 slot flow

目标：

```text
让低频 slot 获得更适合自身几何语义的局部密度模型。
```

可选方式：

- 一个全局 flow + slot-specific calibration；
- 按 slot group 拆分局部 flow；
- left/right 镜像标准化后共享侧向 slot flow；
- front/rear 使用正 gap 表达，再由 slot 语义恢复方向。

推荐优先尝试：

```text
front/rear rel_x_m -> log(longitudinal_gap)
left/right rel_y_left_m -> lane-relative signed offset normalization
```

原因：

- `rel_x_m` 是当前 Wasserstein 最大来源；
- gap 取 log 后尺度更稳定；
- front/rear 的符号由 slot name 决定，不需要 flow 自己学习符号边界。

### 5.5 改进采样阶段的 event-structure 配额

当前默认采样从经验 `p_hat(e | EVT-tail)` 抽样，再做物理 rejection。rejection 后低频 slot 可能被进一步削弱。

推荐改进：

```text
先按真实 EVT-tail mask/primary-slot 计数设定目标 quota，
每个 event structure 独立采样直到达到 quota，
最后合并生成样本。
```

验收：

- mask occupancy L1 应低于当前 `0.0697`；
- primary-slot occupancy L1 应单独报告；
- 稀有 event structure 不应因 rejection 被系统性丢失。

### 5.6 特征去噪和物理变换

建议继续保留：

```text
mean_ax - min_ax >= 0
```

建议新增或尝试：

- `rel_x_m` 改为 positive gap 的 log 坐标；
- 对 acceleration 使用轻度平滑后的统计量；
- 对 lateral velocity/acceleration 做 robust clipping 或 rank-gaussian transform；
- 对 `final_ax_1s_mps2` 与 `mean_ax_1s_mps2` 增加一致性检查；
- 对 slot 初始状态和 1 秒动作摘要增加跨特征物理一致性约束。

---

## 6. 不推荐路线

以下路线不应作为主线：

1. 直接用全量 natural 数据训练无条件 flow，然后直接采样作为 tail；
2. 只追求 train NLL 更低，而不看 EVT-tail held-out test；
3. 用未来真实风险或未来完整轨迹作为 flow 的输入条件；
4. 静默裁剪生成样本以通过物理检查；
5. 为了改善指标而改变 EVT 阈值或 tail 定义；
6. 只看关键 6 个变量图，而不看全 76 维诊断。

---

## 7. 评估协议

所有改进必须统一使用以下评估协议。

### 7.1 密度指标

必须报告：

```text
train NLL
val NLL
test NLL
NLL by mask_pattern
NLL by slot group
```

NLL 是连续密度的 `-log p(x)`，可以为负。它只能在同一坐标、同一数据划分和同一目标分布下比较。

最低要求：

```text
test NLL < -82.7773
```

同时必须优于：

```text
Gaussian
GMM
Copula
Unconditional RealNVP
```

### 7.2 分布匹配指标

必须报告：

```text
mean per-feature KS
mean Wasserstein
Pearson corr MAE
mask occupancy L1
primary-slot occupancy L1
slot-wise mean KS
slot-wise mean Wasserstein
```

最低要求：

```text
mean KS < 0.1342
mask occupancy L1 < 0.0697
```

推荐目标：

```text
left_rear mean KS 显著低于 0.253
left_front mean KS 显著低于 0.190
same_rear mean KS 显著低于 0.147
```

### 7.3 物理合法性指标

必须报告：

```text
invalid_rate
overlap_rate
negative_gap_rate
semantic_error_rate
slot_action_summary_out_of_range
sampling rejection_rate
```

最低要求：

```text
invalid_rate = 0.0 after rejection
overlap_rate = 0.0 after rejection
semantic_error_rate = 0.0 after rejection
```

推荐目标：

```text
sampling rejection_rate < 0.2575
```

### 7.4 可视化要求

必须生成：

```text
all 76 feature marginal grid
all 76 feature KS/Wasserstein bar plot
all 76 feature correlation heatmap
selected joint scatter/KDE
probability diagnostics
event-structure occupancy
```

全 76 维两两散点矩阵不作为强制要求，因为不可读；应使用全边缘、全误差和全相关矩阵做完整审计。

---

## 8. 数据策略

后续应支持三类数据来源：

### 8.1 Strict Tail

严格 EVT-tail 数据，当前为 2209 条。

用途：

- 最终测试；
- 最终采样目标；
- 最终 finetune；
- 论文或报告中的主结果。

### 8.2 Near Tail

低于严格 EVT 阈值但接近尾部的高风险样本。

用途：

- 扩充训练；
- 改善低频 slot；
- 支持加权训练。

约束：

- 不得替代 strict tail 测试；
- 不得让默认生成分布漂移到 near-tail 总体。

### 8.3 Full Natural

`process_highD/` 筛选后的全量自然驾驶片段。

用途：

- 预训练；
- 学习通用交通几何和动力学；
- 训练条件 flow 或密度比模型。

约束：

- 不得直接作为无条件 tail 生成分布；
- 最终 checkpoint 必须经过 tail finetune 或 tail 条件化；
- 最终效果必须按 strict tail 评价。

---

## 9. 实验优先级

建议按以下顺序实施。

### Experiment A: Quota Sampling

只改采样，不改训练。

目的：

- 判断当前连续 flow 是否已经足够，主要问题是否来自 rejection 后的 event-structure 偏差。

成功标准：

- mask occupancy L1 下降；
- low-frequency active count 更接近真实；
- mean KS 不变差。

### Experiment B: Rare-slot Weighted Tail Training

只用 strict tail，但训练 loss 加 rare-slot 权重。

目的：

- 直接改善 `left_rear/left_front/same_rear`。

成功标准：

- `left_rear` 和 `left_front` mean KS 下降；
- test NLL 和 mean KS 优于当前基线；
- rejection rate 不上升。

### Experiment C: Feature Transform Upgrade

改 `rel_x_m` 为 positive gap/log-gap 表达。

目的：

- 降低纵向距离 Wasserstein；
- 改善 front/rear 物理边界。

成功标准：

- `*_rel_x_m` 或其逆变换后的 Wasserstein 降低；
- overlap/negative gap 维持 0；
- rejection rate 下降。

### Experiment D: Near-tail Weighted Training

引入 near-tail pool，严格 tail 加权。

目的：

- 扩大低频 slot 训练样本；
- 避免 full natural 普通样本过度稀释。

成功标准：

- strict tail test NLL 优于当前；
- low-frequency slot 指标优于当前；
- 默认生成仍接近 strict tail。

### Experiment E: Full-natural Pretrain + Tail Finetune

使用 full natural 数据预训练，再 strict tail 微调。

目的：

- 学习更稳健的基础交通分布；
- 通过 tail finetune 保持目标分布。

成功标准：

- strict tail test NLL 最优；
- mean KS、mask L1、rejection rate 同时优于当前；
- 低频 slot 指标明显改善。

### Experiment F: Conditional Risk Flow

训练包含 risk/tail 条件的条件 flow。

目标分布：

```text
p(x | risk_bin, event_structure)
```

采样时固定：

```text
risk_bin = EVT-tail
```

成功标准：

- strict tail 条件采样优于当前；
- risk 条件不会泄露未来完整轨迹；
- 能解释自然、near-tail、strict-tail 之间的分布差异。

---

## 10. 输出和审计要求

每个实验都必须输出独立目录，例如：

```text
results/highd_tail_normalizing_flow_<experiment_name>/
```

每个目录必须包含：

```text
dataset_schema.json
training_summary.json
evaluation_summary.json
training_history.csv
diagnostics/feature_distribution_metrics.csv
diagnostics/nll_comparison.csv
diagnostics/joint_probability_scores.csv
diagnostics/visualization_summary.json
figures/
samples/generated_samples.npz
samples/generated_samples.csv
samples/generated_samples_interfaces.json
```

实验报告必须包含：

- 与当前 `normalizing/` 基线的指标对比；
- 哪些 slot 改善；
- 哪些特征恶化；
- 是否改变目标分布；
- 是否使用 full natural 或 near-tail；
- 最终 checkpoint 的选择标准；
- 是否保留默认 `p(event | EVT-tail)` 采样语义。

---

## 11. 最终验收标准

最终方法必须满足：

1. 严格 EVT-tail held-out test NLL 优于当前 `-82.7773`；
2. 全量 EVT-tail 2209 vs 2209 mean KS 优于当前 `0.1342`；
3. mask occupancy L1 优于当前 `0.0697`；
4. sampling rejection rate 优于当前 `0.2575`，或在其他核心指标显著改善时不得明显恶化；
5. `left_rear/left_front/same_rear` 等低频 slot 的 slot-wise 指标明显优于当前；
6. physical invalid rate、overlap rate、semantic error rate 维持为 0；
7. 默认采样仍表示 EVT-tail 条件分布；
8. 生成样本接口仍可用于 ADS initialization 和 World Model START condition；
9. 不改变 `process_highD/` 已固定的 EVT 风险定义和严格 tail 测试集；
10. 所有实验可复现，有固定配置、随机种子、完整诊断图和指标文件。

如果使用 full natural 数据，最终论文或报告中必须明确：

```text
full natural 只用于预训练、条件建模或 density-ratio 校正；
最终目标分布仍是 p(event | EVT-tail)。
```

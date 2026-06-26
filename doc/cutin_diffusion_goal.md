# Cut-in Diffusion 修复 Goal 模板

## Goal

`/goal` 改善 TREAD 工程中 cut-in 驾驶事件的条件扩散模型训练、重建与长尾泛化效果，使其在保持 ADS-agnostic 场景生成设定的前提下，能够生成与 highD EVT cut-in 长尾事件在统计意义上更接近的 target vehicle 轨迹。
代码的运行环境是系统的: conda activate tread

最终状态应满足：

- cut-in diffusion 在 test/val split 上具备稳定的去噪与 x0 重建能力；
- 由 `select_cutin_tail_contexts.py` 生成的 `diffusion_generated_scenarios.npz` 在目标车横向机动、纵向速度变化、最终横向偏移、语义 cut-in 成功率等指标上明显接近 EVT independent tail peaks；
- `distribution_similarity_summary.json` 中的关键指标相比当前基线显著改善。

---

## 由以下测试或数据证据验证

### 1. 重新运行 cut-in 主流程

```bash
python process_highD/scripts/build_natural_dataset.py --config diffusion/scripts/configs/natural_cutin.yaml
python diffusion/scripts/train_cutin_diffusion.py
python diffusion/scripts/evaluate_cutin_prior.py  # 检查在测试集合上的扩散模型效果
python process_highD/scripts/estimate_cutin_exposure.py  # 获取长尾驾驶事件
python process_highD/scripts/select_cutin_tail_contexts.py  # 联合概率分布建模+批量采样生成
python process_highD/scripts/play_cutin_tail_events.py  # 生成可视化视频，检查语义有效性和多样性
```

### 2. 检查训练结果

检查：

```text
results/diffusion_natural
results/highd_cutin_tail
results/diffusion_natural/cutin/training_summary.json
results/diffusion_natural/cutin/training_history.json
```

验证：

- `val_noise_mse`、`val_x0_l1`、`fixed_eval_noise_mse` 不得出现发散；
- best checkpoint 必须来自 validation split，而不是由明显过拟合的 late epoch 产生；
- 若修改 loss，必须同时报告 train/val 上以下指标的变化：
  - `cutin_constraint_loss`
  - `end_y_loss`
  - `post_lane_loss`
  - `lateral_jerk_loss`

### 3. 检查生成结果摘要

检查：

```text
results/highd_cutin_tail/generated/diffusion_generated_scenarios_summary.json
```

验证：

- `output_semantic_cutin_rate` 应较当前基线提高；
- `candidate_overlap_rate`、`candidate_post_remain_rate`、`candidate_front_at_overlap_rate`、`candidate_collision_free_rate` 不应因修复而显著恶化；
- 如果修改 `guidance_scale` 或 rejection 参数，必须说明它们是否只是后处理修正，还是确实改善了模型本身生成质量。

### 4. 检查分布相似性报告

检查：

```text
results/highd_cutin_tail/generated/figures/distribution_similarity_summary.json
```

验证：

- `intrinsic_trajectory_metrics` 中以下指标的 KS statistic 与 Wasserstein distance 应整体下降：
  - `total_lateral_displacement`
  - `max_abs_lateral_velocity`
  - `mean_abs_lateral_accel`
  - `max_abs_lateral_jerk`
  - `target_speed_change`
  - `final_lateral_offset`
- `scenario_start_target_speed` 与 `scenario_end_target_speed` 的分布不应出现明显漂移；
- 如果某个指标恶化，必须解释其原因，并说明是否属于真实性、多样性或语义有效性之间的可接受权衡。

### 5. 检查生成样本文件

检查：

```text
results/highd_cutin_tail/generated/diffusion_generated_scenarios.npz
```

验证：

- `actions` 的形状、单位和范围必须仍然对应 target `[ax, ay]`；
- `target_trajectory` 必须能由 `initial_states` 和 `actions` 积分得到；
- `scenario_conditions` 的字段顺序必须与 `diffusion/src/features.py` 中 `CUTIN_SCENARIO_CONDITION_KEYS` 完全一致。

---

## 同时必须保持以下不能破坏的约束

1. 不允许重新引入 `rolling context_states`、`context_features` 或 `relative_history` 作为 denoiser 输入。当前模型必须继续只以 `scenario_conditions` 作为 diffusion 条件。

2. 不允许把以下变量作为 diffusion 训练条件：

```text
tail_level
EVT_score
risk_score
collision_label
failure_label
```

3. 不允许把 ADS 闭环响应变量作为 cut-in diffusion 条件输入。`initial_states` 只能用于轨迹积分、trajectory loss 或 sampling guidance，不得作为 denoiser 的直接条件输入。

4. 不允许把 car-following 和 cut-in 合并成统一模型。本次只修复 cut-in 链路。

5. 不允许改变 cut-in diffusion 的动作语义：输出仍然必须是 target vehicle `[ax, ay]`。

6. 不允许引入以下静态地图或采集来源特征作为模型条件：

```text
map_id
lane_width
road_curvature
recording_id
number_of_lanes
```

7. 不允许为了提高 `semantic_cutin_rate` 而仅仅依赖强 rejection 后处理。优先改善训练数据构造、条件特征一致性、trajectory loss、归一化、采样与评价的一致性。

8. 不允许破坏 following 链路、EVT 拟合链路、exposure 估计链路和 subset simulation 的既有接口。

9. 不允许修改输出文件命名和主要字段，除非同时更新所有读取这些字段的代码和 README 说明。

10. 不允许让 generated tail distribution 与 empirical tail distribution 的接近性仅靠裁剪 `final_lateral_offset` 或手工改写 conditions 实现；生成轨迹本身必须通过积分后的 `target_trajectory` 表现出合理 cut-in 行为。

---

## 只允许使用以下限定范围、文件或工具

### 优先修改 cut-in 相关文件,且不得改变主流程默认输出结构。

### 不允许的操作

- 不要修改 following diffusion 的配置、特征、训练逻辑，除非是修复共享工具中的明显 bug，且必须证明 following 输出不受影响；
- 不要引入新的外部大型依赖；
- 不要使用联网下载数据；
- 不要把生成质量优化写成论文式注释；
- 代码修改必须有可运行证据和明确输出文件。

---

## 在尝试迭代期间，AI 应按以下方式选择下一步

### 1. 先诊断数据与条件是否一致，再改模型结构

先检查：

- `CUTIN_SCENARIO_CONDITION_KEYS` 与 `feature_schema.json` 是否一致；
- `dataset.npz` 中 `scenario_conditions`、`initial_states`、`future_states`、`actions` 是否能互相重建；
- 由真实 `actions` 积分出的 target trajectory 是否能复现 `future_states`；
- condition 中：
  - `final_lateral_offset`
  - `time_to_cross`
  - `target_speed_change`
  是否与真实 `future_states` 一致。

### 2. 如果真实 actions 积分都无法复现真实 future_states

优先修复：

```text
action 构造
积分器
坐标系
dt
clip 范围
anchor 选择
```

不要优先修改神经网络。

### 3. 如果真实 actions 可复现真实 future_states，但 diffusion 采样差

优先检查：

- action normalization 是否过强或 std 过小；
- trajectory loss 是否作用在归一化 x0 还是反归一化 actions 上；
- `cutin_trajectory_loss` 的权重是否过弱或过强；
- `x0_weight`、`smooth_weight`、`lateral_jerk_weight` 是否造成过度平滑；
- DDIM `inference_steps` 是否与训练 `diffusion_steps` 对齐。

### 4. 如果单个 condition 下生成结果不稳定或语义失败

优先做条件一致性诊断：

- 生成轨迹重新计算出的 `final_lateral_offset` 是否接近 `condition[7]`；
- 生成轨迹在 `condition[8]` 对应时刻是否接近 ego lane；
- 生成轨迹的 `target_speed_change` 是否接近 `condition[9]`；

### 5. 如果 select_tail 生成分布与 EVT tail 分布不一致

必须区分两类问题：

1. **condition distribution sampling 问题**  
   Gaussian copula 生成的 `scenario_conditions` 已经偏离 empirical tail。

2. **diffusion reconstruction/generation 问题**  
   condition 本身接近 tail，但生成 trajectory 偏离 tail。

只有在证明是哪一类问题后，才能修改对应模块。

### 6. 每次迭代只做一个主要假设变更

例如：

```text
只改 loss 权重
只改 condition 特征
只改 guidance
只改 action/integration
```

每一项都应单独做 ablation。

### 7. 每次修改后必须记录

- 修改内容；
- 预期改善的指标；
- 实际运行命令；
- 实际指标变化；
- 是否接受该修改。

### 8. 如果两个目标冲突，优先级顺序为

1. 真实 actions 积分一致性；
2. semantic cut-in 有效性；
3. 生成分布与 EVT tail intrinsic metrics 的相似性；
4. latent 多样性；
5. 训练 loss 数值下降。

### 9. 不要只根据 train_noise_mse 判断成功

必须结合：

```text
积分后的轨迹指标
tail distribution similarity
semantic cut-in validity
```

### 10. 如果需要调整 cut-in 条件特征

必须保持低维、非冗余、ADS-agnostic，并同步更新：

```text
diffusion/src/features.py
feature_schema.json 生成逻辑
process_highD/src/cutin_tail_generation.py 中 CONDITION_KEYS、TAIL_FEATURE_NAMES、_tail_feature、_reconstruct_cutin_state
相关验证和可视化代码
```

---

## 如果遇到阻碍或无路可走，必须停下并报告

### 1. 当前失败属于哪一类

```text
数据构造失败
坐标系/积分不一致
条件特征不一致
模型训练不足
trajectory loss 设计不合理
copula tail condition sampling 偏移
DDIM sampling/guidance 问题
semantic rejection 过严或过松
```

### 2. 已运行内容

必须报告：

- 已运行的命令；
- 产生的关键输出文件路径；
- 关键指标数值。

### 3. 已排除假设

说明：

- 哪些假设已经被排除；
- 排除依据是什么。

### 4. 修改记录

说明：

- 哪些文件被修改过；
- 修改是否可回滚。

### 5. 下一步候选方向

至少给出两个候选方向，并说明各自风险：

```text
继续优化 training/loss
修改 cut-in condition 特征
修复 action integration
修改 copula condition sampling
改变 guidance/rejection
增加诊断脚本
```

### 6. 证据限制

如果没有足够数据证据，不允许声称问题已经解决；只能说明当前证据支持或不支持某个假设。

---

## 核心诊断原则

本任务的核心不是直接调参，而是先把误差拆成两段：

\[
p_{tail}(C) \rightarrow \hat{p}(C)
\]

和：

\[
(C,z) \rightarrow A_{0:H} \rightarrow \hat{S}
\]

如果第一段偏了，修 copula/条件采样；如果第二段偏了，修 diffusion/重建/trajectory loss。

# Car-following Diffusion 修复 Goal 模板

## Goal

`/goal` 改善 TREAD 工程中 car-following 驾驶事件的条件扩散模型训练、重建与长尾泛化效果，使其在保持 ADS-agnostic 场景生成设定的前提下，能够生成与 highD EVT longitudinal following 长尾事件在统计意义上更接近的 lead vehicle 纵向轨迹。不要破坏cut-in事件的处理

代码的运行环境是系统的：

```bash
conda activate tread
```

最终状态应满足：

- following diffusion 在 test/val split 上具备稳定的去噪与 x0 重建能力；
- 由 `select_following_tail_contexts.py` 生成的 `diffusion_generated_scenarios.npz` 能完整保存 following diffusion rollouts；
- 生成结果在 lead vehicle 加速度、jerk、速度变化、最小 gap、TTC/THW、`y_long` 长尾风险等指标上明显接近 highD EVT independent tail peaks；
- 不破坏 cut-in 链路、EVT 拟合链路、exposure 估计链路和 subset simulation 的既有接口。

当前代码中需要特别确认的已知缺口：

- `process_highD/src/following_tail_generation.py` 中 `_generate_diffusion_rollouts()` 已支持 following 的 `lead_trajectory`/`acceleration` 生成逻辑，`select_following_tail_contexts.py` 在 `generate_diffusion_rollouts=true` 时可以输出 following diffusion 泛化场景。

---

## 由以下测试或数据证据验证

### 1. 重新运行 following 主流程

```bash
python process_highD/scripts/extract_highd_events.py
python process_highD/scripts/build_natural_dataset.py --config diffusion/scripts/configs/natural_following.yaml
python diffusion/scripts/train_following_diffusion.py
python diffusion/scripts/evaluate_following_prior.py
python process_highD/scripts/estimate_following_exposure.py
python process_highD/scripts/select_following_tail_contexts.py
python process_highD/scripts/play_following_tail_events.py
```

如果只验证 diffusion 局部修复，可先运行：

```bash
python process_highD/scripts/build_natural_dataset.py --config diffusion/scripts/configs/natural_following.yaml
python diffusion/scripts/train_following_diffusion.py
python diffusion/scripts/evaluate_following_prior.py
```

### 2. 检查训练结果

检查：

```text
results/diffusion_natural/following
results/diffusion_natural/following/training_summary.json
results/diffusion_natural/following/training_history.json
results/diffusion_natural/following/naturalness_summary.json
```

验证：

- `val_noise_mse`、`val_x0_l1`、`fixed_eval_noise_mse` 不得出现发散；
- best checkpoint 必须来自 validation split，而不是由明显过拟合的 late epoch 产生；
- `naturalness_summary.json` 中以下 sections 必须存在且数值有限：
  - `action_distribution`
  - `physical_feasibility`
  - `trajectory_naturalness`
  - `interaction_naturalness`
  - `record_conditioned_rollout_shift`
  - `conditional_sample_quality`
  - `diversity`
  - `trajectory_reconstruction`
- 如果修改 loss、normalization 或 action 表示，必须同时报告 train/val 上 `noise_mse`、`x0_l1` 与 evaluation 中 action/trajectory/interaction 指标的变化。

### 3. 检查生成结果摘要

检查：

```text
results/highd_following_tail/contexts/tail_context_summary.json
results/highd_following_tail/generated/diffusion_generated_scenarios_summary.json
results/highd_following_tail/generated/diffusion_generated_scenarios.npz
```

验证：

- `tail_context_summary.json` 中 `scenario` 必须为 `following`；
- `tail_feature_names` 必须对应 following 的 7 维条件特征；
- `diffusion_generation` 不应为 `null`；
- `diffusion_generated_scenarios_summary.json` 中 `num_generated_scenarios` 应与配置中的 `num_diffusion_scenarios` 一致；
- 生成文件必须包含：
  - `scenario_conditions`
  - `initial_states`
  - `actions`
  - `acceleration`
  - `lead_trajectory`
  - `context_index`
- `actions` 的语义必须保持为 lead vehicle longitudinal `jerk`，形状应为 `[N, H, 1]`；
- `acceleration` 必须由 initial lead acceleration 和 `actions` 积分得到，并受 `ax_min`/`ax_max` 限制；
- `lead_trajectory` 必须由 `initial_states[:, 1]` 和 `acceleration` 积分得到。

### 4. 检查 following 生成输出

following select 阶段只输出 scenario conditions 和 generated lead/adversary
trajectory；闭环 ego 响应和 interaction risk 指标应在 playback/仿真阶段计算。

验证：

- `diffusion_generated_scenarios.npz` 包含 `lead_trajectory`、`actions`、`acceleration`
  和 `scenario_conditions`；
- 不包含 select 阶段生成的 `ego_trajectory`；
- `actions`、`acceleration` 和 `lead_trajectory` 的积分关系一致；
- interaction/risk metrics、collision rate 和 near-collision rate 由
  `play_following_tail_events.py` 或后续闭环仿真阶段基于 highway-env IDM ego 计算。

### 5. 检查数据构造一致性

检查：

```text
results/diffusion_natural/following/dataset.npz
results/diffusion_natural/following/dataset_normalized.npz
results/diffusion_natural/following/feature_schema.json
results/diffusion_natural/following/normalization_stats.json
```

验证：

- `feature_schema.json` 中：
  - `event_type` 必须为 `following`
  - `conditioning_mode` 必须为 `anchor_scenario`
  - `model_input_keys` 只能包含 `scenario_conditions`
  - `action_representation` 必须为当前配置中的 `jerk`
  - `action_keys` 必须对应 lead longitudinal jerk
- `scenario_conditions` 的字段顺序必须与 `diffusion/src/features.py` 中 `FOLLOWING_SCENARIO_CONDITION_KEYS` 完全一致：

```text
ego_vx_0
initial_gap
initial_delta_v
lead_ax_0
lead_speed_change
lead_min_ax
lead_braking_duration
```

- 真实 `actions` 必须能按以下链路重建 `future_states[:, 1]` 的 lead trajectory：

```text
lead_ax_0 + cumsum(jerk * dt) -> lead_ax
initial lead state + lead_ax integration -> lead_trajectory
```

- condition 中以下变量必须能由真实 `initial_states` 和 `future_states` 重新计算并与保存值一致：
  - `initial_gap`
  - `initial_delta_v`
  - `lead_ax_0`
  - `lead_speed_change`
  - `lead_min_ax`
  - `lead_braking_duration`

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
y_long
evt_tail_probability
```

3. 不允许把 ADS 闭环响应变量作为 following diffusion 条件输入。`initial_states` 只能用于轨迹积分、trajectory loss、evaluation 或 sampling guidance，不得作为 denoiser 的直接条件输入。

4. 不允许把 car-following 和 cut-in 合并成统一模型。本次只修复 following 链路。

5. 不允许改变 following diffusion 的动作语义：输出仍然必须是 lead vehicle longitudinal `jerk`，除非单独完成 acceleration 表示的 ablation，并同步更新所有 schema、adapter、evaluation、subset 调用。

6. 不允许引入以下静态地图或采集来源特征作为模型条件：

```text
map_id
lane_width
road_curvature
recording_id
number_of_lanes
```

7. 不允许为了提高 tail risk 相似性而仅仅依赖强 rejection 后处理。优先改善数据构造、condition 一致性、jerk/acceleration 解码、归一化、采样与评价的一致性。

8. 不允许破坏 cut-in 链路、EVT 拟合链路、exposure 估计链路和 subset simulation 的既有接口。

9. 不允许修改输出文件命名和主要字段，除非同时更新所有读取这些字段的代码和 README 说明。

10. 不允许让 generated tail distribution 与 empirical tail distribution 的接近性仅靠裁剪 `initial_gap`、`lead_min_ax` 或手工改写 conditions 实现；生成轨迹本身必须通过积分后的 `lead_trajectory` 表现出合理 following 行为。

---

## 只允许使用以下限定范围、文件或工具

### 优先修改 following 相关文件，且不得改变主流程默认输出结构

优先范围：

```text
diffusion/scripts/configs/natural_following.yaml
diffusion/scripts/train_following_diffusion.py
diffusion/scripts/evaluate_following_prior.py
diffusion/src/data.py
diffusion/src/features.py
diffusion/src/evaluation.py
diffusion/src/kinematics.py
process_highD/scripts/select_following_tail_contexts.py
process_highD/src/following_tail_generation.py
tools/highd_longitudinal.py
tools/diffusion_adapter.py
IDM_subset/scripts/configs/latent_subset_following.yaml
```

### 不允许的操作

- 不要修改 cut-in diffusion 的配置、特征、训练逻辑，除非是修复共享工具中的明显 bug，且必须证明 cut-in 输出不受影响；
- 不要引入新的外部大型依赖；
- 不要使用联网下载数据；
- 不要把生成质量优化写成论文式注释；
- 代码修改必须有可运行证据和明确输出文件。

---

## 在尝试迭代期间，AI 应按以下方式选择下一步

### 1. 先诊断数据与条件是否一致，再改模型结构

先检查：

- `FOLLOWING_SCENARIO_CONDITION_KEYS` 与 `feature_schema.json` 是否一致；
- `dataset.npz` 中 `scenario_conditions`、`initial_states`、`future_states`、`actions` 是否能互相重建；
- 由真实 `actions` 解码出的 lead acceleration 是否能复现 `future_states[:, 1, 4]` 的主要趋势；
- 由真实 acceleration 积分出的 lead trajectory 是否能复现 `future_states[:, 1]`；
- condition 中 `lead_speed_change`、`lead_min_ax`、`lead_braking_duration` 是否与真实 `future_states` 一致。

### 2. 如果真实 actions 积分都无法复现真实 future_states

优先修复：

```text
jerk 构造
acceleration 解码
dt
savgol smoothing
clip 范围
anchor 选择
坐标系
```

不要优先修改神经网络。

### 3. 如果真实 actions 可复现真实 future_states，但 diffusion 采样差

优先检查：

- action normalization 是否过强或 std 过小；
- jerk 表示是否造成积分漂移；
- `x0_weight`、`smooth_weight` 是否造成过度平滑；
- DDIM `inference_steps` 是否与训练 `diffusion.steps` 对齐；
- train/val/test split 是否因 recording 分布差异导致条件外推。

### 4. 如果单个 condition 下生成结果不稳定或语义失败

优先做条件一致性诊断：

- 生成轨迹重新计算出的 `lead_speed_change` 是否接近 condition；
- 生成轨迹的 `lead_min_ax` 是否接近 condition；
- 生成轨迹的 `lead_braking_duration` 是否接近 condition；
- 生成的 `initial_gap`、`min_gap`、`min_ttc` 是否在物理合理范围；
- `acceleration` clip rate 是否过高，导致长尾风险由硬裁剪产生。

### 5. 如果 select_tail 生成分布与 EVT tail 分布不一致

必须区分两类问题：

1. **condition distribution sampling 问题**  
   Gaussian copula 生成的 `scenario_conditions` 已经偏离 empirical tail。

2. **diffusion reconstruction/generation 问题**  
   condition 本身接近 tail，但生成 lead trajectory、interaction risk 或 `y_long` 偏离 tail。

只有在证明是哪一类问题后，才能修改对应模块。

### 6. 每次迭代只做一个主要假设变更

例如：

```text
只改 following rollout guard
只改 jerk/acceleration 解码
只改 loss 权重
只改 condition 特征
只改 guidance
只改 tail context sampler
```

每一项都应单独做 ablation。

### 7. 每次修改后必须记录

- 修改内容；
- 预期改善的指标；
- 实际运行命令；
- 实际指标变化；
- 是否接受该修改。

### 8. 如果两个目标冲突，优先级顺序为

1. 真实 actions 解码与积分一致性；
2. following 物理可行性；
3. gap/TTC/THW/y_long interaction risk 相似性；
4. 生成分布与 EVT tail intrinsic metrics 的相似性；
5. latent 多样性；
6. 训练 loss 数值下降。

### 9. 不要只根据 train_noise_mse 判断成功

必须结合：

```text
积分后的 lead trajectory 指标
interaction naturalness
tail distribution similarity
collision/near-collision validity
```

### 10. 如果需要调整 following 条件特征

必须保持低维、非冗余、ADS-agnostic，并同步更新：

```text
diffusion/src/features.py
diffusion/src/data.py
diffusion/src/evaluation.py
process_highD/src/following_tail_generation.py
results/diffusion_natural/following/feature_schema.json 的重建流程
diffusion/README.md
IDM_subset/README.md 或相关 following 配置说明
```

调整后必须重建 dataset、normalization stats 和 checkpoint，旧 checkpoint 不得混用。

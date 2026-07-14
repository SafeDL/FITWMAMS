# 实现说明

## 活动模型

唯一活动实现为 `catk_topk`，配置位于：

```text
world_model/scripts/configs/highd_world_model.yaml
```

它使用共享 START/ROLL Transformer 编码器和 CAT-K `K=8` 联合背景车动作头。`NominalCATKDecoder` 先生成内部候选并取其 MAP 动作，作为外层候选 `0`；候选 `1--7` 是场景级 token 控制的联合残差行为。外层将候选 `0` 的 logit 固定为其余候选最大 logit 加 `nominal_logit_margin`，因此 argmax 复现名义分支，categorical 仍可采样全部八个索引。新训练从随机初始化开始，不读取外部初始化路径或基线权重；`resume_from_checkpoint` 只允许续训同一输出目录的 `latest_world_model.pt`。它们都只读取既有 START/ROLL 状态张量，不引入 ADS、ego future 或新的 latent。中间候选、一次性合成工具和重复配置已删除；正式配对比较脚本保留用于复核历史结果与冻结基线。

## 环境随机变量

世界模型测试空间写为：

```text
Omega_test = E x Z_flow x Xi_world
```

`E=(slot_mask, primary_slot)` 与 `Z_flow` 属于归一化流初始场景采样。`Xi_world` 是 CAT-K 在每个一秒 chunk 选择的候选索引。`CATKTopKWorldModel.sample_actions_with_xi()` 显式返回：

- `actions`：被选候选的背景车动作；
- `candidate_index`：本 chunk 的 `Xi_world`；
- `candidate_probabilities`：当前状态下的候选概率。

环境默认按 categorical 分布采样候选，不添加连续动作噪声。若需要确定性环境，应在构造 `WorldSamplingConfig` 时将 `candidate_selection` 设为 `argmax`；此时 `Xi_world` 退化为确定值。调用 `start(candidate_index=...)` 或 `roll(..., candidate_index=...)` 时，显式索引优先于采样设置；完整复现实例应存储该索引序列，而不是只存储随机种子。

## 正式环境接口

`CATKBackgroundEnvironment` 位于 `world_model/src/environment.py`。

1. `reset_from_flow_sample()` 接收完整 Flow 场景样本：连续 76 维特征、slot mask、primary slot。
2. `start()` 根据该初始样本生成第一秒背景车动作，不接收 ego future。
3. `roll()` 只接收已经发生的 ego 历史状态 `[25, 6]` 与有效掩码 `[25]`，重算关系特征后生成下一秒背景车动作。

环境只建模背景车。ego 当前/历史状态是外部边界条件，不是 ADS 模型输入。接口没有 ADS 对象、ego future、风险轨迹或 EVT 标签参数。

## 关系特征

关系特征是固定函数：

```text
relation(t) = g(ego(t), background(t), primary_slot)
```

它包含相对位置、gap、相对速度、closing speed、截断 TTC、截断 DRAC、primary-slot 标志和 slot 有效标志。START 由 Flow 初始物理状态计算，ROLL 由当前外部 ego 与世界模型背景状态重新计算。它是模型输入的确定性重参数化，不是测试空间变量，也不由 Flow 单独采样。

数据构造、重建评测和正式环境调用 `world_model.src.rollout.build_relation_features_from_current()`。多 chunk 训练使用等价的 Torch 实现以保持状态转移可微；两者均采用相同的固定车辆几何、截断阈值和特征顺序。

## 固定环境约定

当前环境的六个背景车辆槽位在 `c0` 固定为 `same_front`、`same_rear`、`left_front`、`left_rear`、`right_front`、`right_rear`。`slot_mask` 定义初始有效性，`primary_slot` 定义主要交互槽位。环境 rollout 不进行槽位重分配，也没有车辆新增或主动消失机制；这是当前 `catk_topk` 的建模边界。

状态为 `[x_m, y_left_m, vx_mps, vy_left_mps, ax_mps2, ay_left_mps2]`。START 使用初始 ego 原点，ROLL 输入输出保持同一持续局部坐标系，内部只在编码时重心化至当前 ego。固定物理常数为车辆长 4.8 m、宽 1.9 m、车道宽 3.6 m；固定采样频率为 25 Hz，历史与每个输出 chunk 均为 25 帧（1 秒）。

## 样本数来源

`process_highD/` 生成 161,314 个 6 秒自然驾驶片段及其 EVT 标记；它不生成 339 万个片段。世界模型构造器对每个片段生成 1 个 START 样本与 20 个 ROLL 样本，ROLL 时刻为 1.0--4.8 秒、步长 0.2 秒。因此训练缓存总数为 3,387,594。

所有数据构造均通过 `process_highD.src.preprocess.prepare_recording()` 读取 highD。该函数集中执行读取、方向统一、异常标记和 25 Hz 重采样，EVT、归一化流、世界模型和真实片段回放不再各自维护重复预处理流程。

## logged-ego 重建评测

`world_model/src/evaluation.py` 中的 START->ROLL replay 保留用于 highD 重建验证。它在进入 ROLL 时使用 logged ground-truth ego future，因此只能说明背景车在真实 ego 条件下的重建误差，不能作为 ADS 测试环境接口或 ADS 闭环结果。

保留 checkpoint 的晋升比较采用 test paired bootstrap（2,000 次）。候选相对冻结基线的 EVT-tail START ADE、FDE、gap MAE，以及 logged-ego START->ROLL ADE 的点差和单侧 95% 上界均为 `0`。该结果是历史证据；从零训练得到的新 checkpoint 必须重新通过相同门槛。冻结基线位于：

```text
results/highd_world_model/catk_topk_baseline/checkpoints/best_world_model.pt
```

当前 checkpoint 的历史重建指标：

| 指标 | 数值 |
|---|---:|
| EVT-tail ADE | 0.028524 m |
| EVT-tail gap MAE | 0.026337 m |
| logged-ego START->ROLL ADE | 0.055516 m |

`high_risk_persistence` 已删除，因为它是与 canonical `risk_trace` 不一致的 TTC/DRAC 阈值代理指标。`gap`、`TTC` 和 `DRAC` 的重建误差仍可作为交通交互重建诊断，但不构成未来 ADS 风险结论。

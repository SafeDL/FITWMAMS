# 世界模型目标

## 当前结论

`world_model/` 的唯一活动模型名为 `catk_topk`。正式模型与冻结基线分别为：

```text
results/highd_world_model/catk_topk/checkpoints/best_world_model.pt
results/highd_world_model/catk_topk_baseline/checkpoints/best_world_model.pt
```

正式模型为 CAT-K。候选 `0` 来自可训练名义解码器的内部 MAP 动作，候选 `1--7` 为可采样的联合残差意图；外层以 `nominal_logit_margin` 保证候选 `0` 是外层 argmax。新训练从随机初始化开始，不读取外部初始化 checkpoint 或基线权重。保留的历史 checkpoint 在其产生协议下，EVT-tail START 的 ADE、FDE、gap MAE，以及 logged-ego START->ROLL ADE 相对冻结基线的差值和单侧 95% 上界均为 `0`。

该结论只覆盖确定性 `argmax` 重建路径。候选 `1--7` 的多样性、概率校准和 ADS 下游价值仍需在 categorical 采样协议下单独验证，不能将名义锚点的零差值表述为残差候选已经优于基线。

## 测试空间

自然驾驶长尾事件由 `process_highD/` 固定筛选，归一化流生成完整初始场景：

```text
Omega_tail = E x Z_flow x Xi_world
E          = (slot_mask, primary_slot)
s0         = Flow(E, z_flow)
Xi_world   = (xi_0, ..., xi_J-1), xi_j in {0, ..., 7}
```

固定 Flow checkpoint、CAT-K checkpoint、候选温度、候选采样规则、积分器、坐标和槽位规则、车辆几何常数与 episode 时间边界后，`(E, z_flow, Xi_world)` 唯一确定背景交通随机过程。若不显式提供 `Xi_world`，环境以 `world_seed` 驱动 categorical 采样；若显式传入每个 chunk 的 `candidate_index`，该索引序列优先于随机采样。

`relation(t) = g(ego(t), background(t), primary_slot)` 是当前状态的确定性变换，包含相对位置、gap、相对速度、closing speed、截断 TTC、截断 DRAC、主槽位与有效性。它是模型输入，不是额外随机变量；不同 ADS 的已发生 ego 历史会导致同一环境动力学产生不同响应。

## 固定接口与边界

- 保持 `reset_from_flow_sample(...)`、`start()`、`roll(...)` 的调用方式不变。
- START 接收 Flow 的 76 维连续场景、`slot_mask` 与 `primary_slot_index`；首秒动作摘要属于该初始样本。
- ROLL 只接收已经发生的 ego 历史 `[25, 6]` 与有效掩码；动作摘要固定为零，关系特征由当前状态重算。
- 每个 chunk 输出 `actions_mps2[25, 6, 2]`、`background_states[25, 6, 6]`、`background_valid[25, 6]`、`candidate_index`、`candidate_probabilities[8]`。
- 固定六个背景车槽位、静态 `slot_mask`、持续局部坐标系、25 Hz、每 chunk 一秒、固定积分器与车辆几何。
- 禁止向模型输入 ADS 身份、ADS 网络特征、ego future、风险标签、EVT 标签或 `risk_trace`。
- 环境不引入连续动作噪声；世界模型随机性仅来自每个 chunk 的离散 `Xi_world`。

## CAT-K 设计

1. 时空条件编码：每个 agent 的历史状态先经时间 Transformer 编码，再经全局车辆交互与相对图注意力建模 ego--背景车、背景车--背景车关系。
2. 场景级意图：六个背景车 token 池化为场景表示，产生八维 `candidate_probabilities`。一个 `candidate_index` 联合控制全部六辆背景车，不存在逐槽位独立采样。
3. 名义分支与残差：`NominalCATKDecoder` 先生成内部候选并取其 MAP 动作，作为外层候选 `0`；残差解码器的候选 `1--7` 由场景级 token 产生。外层将候选 `0` 的 logit 置为其余候选最大 logit 加 `nominal_logit_margin`，故 `argmax` 固定选择候选 `0`；categorical 采样仍可选择八个候选。
4. 物理残差动作：每个候选输出每车、每动作维度的 5 个 jerk 样条控制点。控制点展开为 25 帧有界 jerk，再从当前加速度积分得到连续动作；纵向动作限制为 `[-8, 4] m/s2`，横向限制为 `[-4, 4] m/s2`。
5. 软多模态训练：动作距离与候选概率共同形成 mixture loss；energy、多样性、平滑、MAP 一致性与概率熵项约束候选塌缩和过度抖动。
6. 多 chunk 训练：同一自然驾驶片段按 START 和连续 ROLL 位置展开最多五个一秒 chunk。下一 chunk 使用模型生成的背景车历史，并仅将该时刻已经发生的 logged ego 历史作为外部条件。

## 从零复现

- 数据构造完成后，`train_highd_world_model.py` 直接根据唯一活动配置构造 `CATKTopKWorldModel`；不存在版本选择字段、外部初始化 checkpoint 或基线权重依赖。`resume_from_checkpoint` 仅用于续训同一输出目录的 `latest_world_model.pt`，不参与从零复现。
- 名义解码器通过直通 MAP 选择接收混合损失和多 chunk 状态一致性损失的梯度，因此随机初始化时也会与残差分支共同收敛。
- 保留的 checkpoint 与评测 JSON 是历史性能证据。未来从零训练得到的新 checkpoint 必须重新执行固定 paired-bootstrap，并满足“不弱于冻结基线”的验收条件。

## 验收与保留结果

- 任何后续模型都必须在固定的数据缓存、划分、候选数、温度、随机种子和积分器下与冻结基线配对比较。
- EVT-tail ADE、FDE、gap MAE 和 logged-ego START->ROLL ADE 的点估计及单侧 95% bootstrap 上界均不得高于冻结基线。
- 必须同时报告 2--5 chunk 模型状态重建、候选熵、有效候选数、成对轨迹距离、mixture NLL 与概率--责任差异。
- logged-ego replay 仅是背景交通重建评测，不能作为 ADS 闭环测试结果。

`results/highd_world_model/` 只保留当前 CAT-K 正式产物、冻结基线和共享缓存；中间候选、重复配置和 TensorBoard 日志已删除。活动配置仅为：

```text
world_model/scripts/configs/highd_world_model.yaml
```

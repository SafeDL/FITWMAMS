# 世界模型目标

## 当前目标

`world_model/` 的唯一活动实现是 `catk_topk`。它用于从自然驾驶长尾分布构造背景交通行为环境，不负责 ADS 策略、感知、地图或路线规划。

当前不应在未明确下令时启动重新训练或全量评测。

## 测试空间

自然驾驶长尾筛选已经由上游固定。世界模型阶段的测试空间为：

```text
Omega_test = E x Z_flow x Xi_world

E        = (slot_mask, primary_slot)
Z_flow   = 归一化流的连续 latent
Xi_world = CAT-K 每个背景交通 chunk 的候选分支索引
```

Flow 生成完整初始场景：

```text
s0 = Flow(E, Z_flow)
```

世界模型定义背景交通条件动态：

```text
a_bg[t:t+K] = WorldModel(H_t, current_ego_t, current_bg_t, relation_t, Xi_world_t)
x_bg[t+1:t+K] = Integrator(x_bg_t, a_bg[t:t+K])
```

ego 当前/历史状态是外部边界条件。ADS 是被测对象，不属于测试空间，也不作为世界模型输入。世界模型不得读取 ADS 身份、ego future、风险轨迹或 EVT 标签。

## START 与 ROLL

START 读取完整 Flow 样本：

1. 连续 76 维场景特征；
2. `slot_mask`；
3. `primary_slot`；

第一秒背景车动作摘要已包含在 76 维连续特征中，不是独立的第四项输入。

ROLL 只读取已经发生的 ego 历史和已生成的背景车历史。第一秒动作摘要在 ROLL 中恒为零，不得再次注入未来信息。

关系特征是：

```text
g(current ego state, current background state, primary_slot)
```

它是世界模型的固定输入变换，不是 Flow 输出变量或测试空间的新维度。当前实现固定输出 10 维：相对位置、净纵向 gap、相对速度、closing speed、截断 TTC、截断 DRAC、主交互槽位标志和槽位有效标志。

## 运行入口

唯一配置：

```text
world_model/scripts/configs/highd_world_model.yaml
```

```bash
python world_model/scripts/prepare_highd_world_model_dataset.py
python world_model/scripts/train_highd_world_model.py
python world_model/scripts/evaluate_highd_world_model.py
```

## 环境接口

`CATKBackgroundEnvironment` 必须维持以下边界：

```text
reset_from_flow_sample(feature_row, slot_mask, primary_slot_index, world_seed)
start()                                  # 无 ego future
roll(ego_history_states, ego_history_valid)  # 只读过去和当前 ego
```

每个 START/ROLL chunk 返回候选索引与候选概率；环境维护累计 `Xi_world`。调用方应同时保存 `WorldSamplingConfig` 的种子，或保存传给 `reset_from_flow_sample()` 的覆盖种子，以便同一初始场景可在不同 ADS 下复现。

正式环境每次返回 `actions_mps2[25, 6, 2]`、`background_states[25, 6, 6]`、`background_valid[25, 6]`、候选索引和候选概率。`roll()` 的 ego 输入形状固定为 `[25, 6]`，其有效掩码形状固定为 `[25]`。

## 训练与重建验证

训练、open-loop 评测和 logged-ego replay 使用 highD ground-truth ego 状态，以验证背景车重建能力。现有 logged-ego replay 允许使用真实 ego future 构造下一段 ROLL history，但只能留在 `evaluation.py` 作为重建评测，不能进入环境接口。

canonical `risk_trace` 只用于上游自然驾驶长尾筛选，以及未来 rollout 后的统一安全评分；它不是世界模型条件。`high_risk_persistence` 已删除，避免与 canonical 风险轨迹混淆。

# highD 背景交通世界模型

`world_model/` 只保留一个活动实现：

```text
catk_topk
```

它是面向 ADS 规划控制测试的 CAT-K 背景交通行为模型。归一化流生成长尾初始场景；世界模型只生成背景车行为；ego 是外部状态输入，不预测 ego 动作，也不读取 ADS 身份、未来 ego 轨迹或风险标签。

## 测试空间

在已固定自然驾驶长尾筛选口径后，测试空间的随机部分为：

```text
Omega_test = E x Z_flow x Xi_world

E        = (slot_mask, primary_slot)
Z_flow   = 归一化流的连续 latent
Xi_world = CAT-K 每个 START/ROLL chunk 的候选分支索引
```

Flow 先生成完整初始场景：

```text
s0 = Flow(E, Z_flow)
```

随后世界模型在每个 chunk 根据当前/历史场景状态和 `Xi_world` 生成背景车动作。关系特征是当前状态的确定性变换，不属于测试空间的新随机变量。

## 输入与边界

- START：完整 Flow 样本的 76 维连续特征、`slot_mask` 与 `primary_slot_index`。76 维特征内部已经包含每个活跃背景车的第一秒动作摘要，不是额外独立输入。
- ROLL：调用方提供已经发生的 ego 历史 `[25, 6]` 和有效掩码 `[25]`；环境保留已生成的背景车历史，并在内部重算即时关系特征。ROLL 的第一秒动作摘要固定为零。
- 输出：每个 chunk 的 `actions_mps2[25, 6, 2]`、`background_states[25, 6, 6]`、`background_valid[25, 6]`、`candidate_index` 与 `candidate_probabilities[8]`。动作分量为 `[ax_mps2, ay_left_mps2]`，状态分量为 `[x, y, vx, vy, ax, ay]`。
- 禁止输入：ADS 身份或网络特征、ego future、未来风险、`risk_trace`、EVT 标签。

## 数据构造

`process_highD/` 只负责从 highD 原始数据筛选固定的自然驾驶片段，并输出
`results/highd_natural_evt/natural_segments.csv`。当前文件有 161,314 个 6 秒片段，
每个片段只对应一个初始场景，不等于一个世界模型训练样本。

世界模型构造器通过 `process_highD.src.preprocess.prepare_recording()` 读取原始 highD；这与上游 EVT 和归一化流数据构造共用完全相同的方向统一、异常标记和重采样步骤。

世界模型的数据构造器再读取原始 highD 轨迹，将每个片段展开为：

- 1 个 START 样本：锚点时刻预测未来 1 秒；
- 20 个 ROLL 样本：从 1.0 s 到 4.8 s，每 0.2 s 取一个当前时刻，使用此前 1 秒历史预测后续 1 秒。

因此共享缓存共含 `161,314 x (1 + 20) = 3,387,594` 条样本，其中 START 161,314 条、ROLL 3,226,280 条。缓存目录 `results/highd_world_model/shared_dense_start_roll/` 是训练与 logged-ego 重建评测的数据源，不是 ADS 测试时直接输入的场景库。

## 固定仿真语义

- 车辆槽位：每个场景固定六个槽位 `same_front`、`same_rear`、`left_front`、`left_rear`、`right_front`、`right_rear`。`slot_mask` 在 `c0` 决定哪些槽位存在，`primary_slot` 标记主要交互槽位。当前环境不会在 rollout 中重新分配槽位，也不会生成新车或让已有槽位主动消失；有效槽位会持续积分。
- 坐标：`x_m` 为道路纵向，`y_left_m` 为向左横向。START 以 `c0` ego 位置为原点；ROLL 接收与此前背景车输出处于同一持续局部坐标系的 ego 历史，内部仅为模型编码暂时平移到当前 ego 原点，返回状态仍保持该持续坐标系。
- 车辆几何：ego 与背景车均使用长度 4.8 m、宽度 1.9 m，车道宽度为 3.6 m。gap、TTC、DRAC 与碰撞诊断使用这些固定常数。
- 时间：固定 25 Hz；历史长度 25 帧（1 秒），每次 START/ROLL 输出 25 帧（1 秒）；训练片段固定为 6 秒。每个环境 episode 可 ROLL 多次，终止长度由未来 ADS 测试任务设置。

## 目录与入口

配置文件：

```text
world_model/scripts/configs/highd_world_model.yaml
```

数据、训练和重建评测入口与 `normalizing_flow/` 对齐：

```bash
python world_model/scripts/prepare_highd_world_model_dataset.py
python world_model/scripts/train_highd_world_model.py
python world_model/scripts/evaluate_highd_world_model.py
```

当前任务不应在未明确下令时启动训练或全量评测。

## 环境接口

`world_model.src.environment.CATKBackgroundEnvironment` 是后续 ADS 测试使用的正式背景交通接口：

```python
environment.reset_from_flow_sample(
    flow_feature_row,
    slot_mask,
    primary_slot_index=primary_slot,
    world_seed=123,
)
first_second = environment.start()
next_second = environment.roll(ego_history_states, ego_history_valid)
```

`start()` 不接收 ego future。`roll()` 只接收截至当前时刻已经发生的 ego 历史。每个返回 chunk 都记录实际选择的 CAT-K 候选索引；这些索引构成 `Xi_world`。

候选分支由 `WorldSamplingConfig` 控制：默认按 categorical 分布、温度 `1.0`、种子 `123` 采样；也可将 `candidate_selection` 设为 `argmax`。如在 `reset_from_flow_sample()` 传入覆盖的 `world_seed`，调用方应将该种子与 `xi_world_candidate_indices` 一同保存，以复现实例。

## 当前 checkpoint 与历史重建结果

当前 checkpoint：

```text
results/highd_world_model/catk_topk/checkpoints/best_world_model.pt
```

其历史 logged-highD EVT-tail 重建结果为：EVT-tail ADE `0.028524 m`、gap MAE `0.026337 m`、logged-ego START->ROLL ADE `0.055516 m`。这些数值用于验证背景车重建能力；其中 logged-ego replay 会拼接真实 ego future，不能当作 ADS 闭环测试结果。

完整实现边界见 [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)。

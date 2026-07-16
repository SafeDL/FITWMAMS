# 背景交通世界模型

仓库保留两个彼此隔离的实现：

```text
catk_topk                 # 冻结比较基线
semi_markov_relational    # 新的 Semi-Markov Relational Traffic World Model
```

新实现采用可变参与者的动态关系交通图、场景级离散半马尔可夫潜在交互状态、离散 hazard 持续时间（含窗口末端右删失损失），以及“持续意图 + 即时响应”控制分解。它不读取 ADS 身份、ego future、EVT 标签、风险标签或 `risk_trace`。`catk_topk` 的以下说明仅适用于冻结基线，不能用于解释新模型。

## 新模型入口

使用具有 PyTorch 的项目环境执行：

```bash
python world_model/scripts/prepare_highd_semi_markov_relational_dataset.py
python world_model/scripts/train_highd_semi_markov_relational.py
python world_model/scripts/evaluate_highd_semi_markov_relational.py
```

默认配置为 `world_model/scripts/configs/highd_behavior_anchored_semi_markov.yaml`。它使用冻结的 76 维 Flow：后 36 维摘要在 START 模式生成首秒基础控制，并由当前图关系作有界修正；之后切换到 ROLL 模式。训练直接读取每条 150 帧序列的已缓存摘要（1 秒历史 + 5 秒未来），Flow 端到端生成才将一条 76 维样本直接解包为 START 场景。

每个 M1 候选还必须执行 `compare_semi_markov_to_catk.py` 的同序列 1 s 与 5 s
bootstrap 对比。报告会明确标记冻结 CAT-K 在 START 阶段使用真实未来的一秒摘要。

核心闭环接口为：

```python
environment.reset(initial_graph, world_randomness)
result = environment.step(ego_state, ego_valid=True, dt=0.2)
one_second = environment.roll(ego_history_states, ego_history_valid)
```

`roll()` 是兼容包装器，会执行五个 0.2 秒响应更新；它不接收 ego future。环境记录每次状态转移使用的外生状态/持续时间随机数、状态、持续时间及转移时刻，以支持 ADS 无关重放。`snapshot()`/`restore()` 还保存图、历史、latent/duration 与 RNG 状态，可用于 AMS 分支复制后确定性继续。

### 冻结 M0 对照

冻结 M0 checkpoint 位于
`results/highd_world_model/semi_markov_relational_full_tbptt_finetune/checkpoints/`。
它只作为当前 M1 的 cold-start 对照加载；不保留其早期训练配置、实验报告或重训入口。

### 当前行为锚定候选

当前 M1 实现的诊断输出位于：

```text
results/highd_world_model/behavior_anchored_semi_markov_m1_start_roll_v2/
```

在完整的 24,216 条 highD test 上，当前受保护 M1 checkpoint 的 1 s/5 s FDE 为
`0.04162 / 0.77001 m`；同协议冻结 M0 为 `0.04659 / 1.34516 m`。标准 CAT-K
仅在 START 使用真实未来的一秒摘要，ROLL 阶段将该输入置零；它的五秒结果更强
应归因于首秒条件和后续 ROLL 模型本身，而不是每秒重新读取未来摘要。

完整缓存、训练与评测的命令为：

```bash
python world_model/scripts/prepare_highd_semi_markov_relational_dataset.py \
  --config world_model/scripts/configs/highd_behavior_anchored_semi_markov.yaml
python world_model/scripts/train_highd_semi_markov_relational.py \
  --config world_model/scripts/configs/highd_behavior_anchored_semi_markov.yaml
python world_model/scripts/evaluate_highd_semi_markov_relational.py \
  --config world_model/scripts/configs/highd_behavior_anchored_semi_markov.yaml
```

---

## 冻结 CAT-K 基线

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

随后世界模型在每个 chunk 根据当前/历史场景状态和 `Xi_world` 生成背景车动作。关系特征是当前状态的确定性变换，不属于测试空间的新随机变量。未显式传入候选索引时，`world_seed` 只用于生成 `Xi_world`；已记录的候选索引序列可直接传回 `start()` 与 `roll()` 复现对应背景车行为。

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

CAT-K 仅作为冻结比较基线；本仓库保留其配置、checkpoint 兼容代码与
`compare_semi_markov_to_catk.py`，不再提供其数据准备、训练或单独评测入口。

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

当前正式 checkpoint：

```text
results/highd_world_model/catk_topk/checkpoints/best_world_model.pt
```

最终 `catk_topk` 从随机初始化直接训练：`NominalCATKDecoder` 的内部 MAP 动作构成候选 `0`，候选 `1--7` 由场景级残差 token 生成。外层 `nominal_logit_margin` 保证确定性 `argmax` 选择候选 `0`；categorical 采样仍可选择八个联合意图。训练不依赖外部初始化 checkpoint 或冻结基线权重；`resume_from_checkpoint` 仅续训同一输出目录的最新 checkpoint。

保留的历史 checkpoint 相对冻结基线的 test paired-bootstrap（2,000 次）结果为：EVT-tail START 的 ADE、FDE、gap MAE 差值均为 `0`，logged-ego START->ROLL ADE 差值也为 `0`；四项单侧 95% 上界均为 `0`。从零训练得到的新 checkpoint 必须重新通过相同门槛。logged-ego replay 仍只用于世界模型重建验证，不能当作 ADS 闭环测试结果。

完整实现边界见 [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)。

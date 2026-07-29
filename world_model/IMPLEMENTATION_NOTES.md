# 实现说明

本文描述当前保留的 Semi-Markov World Model 与 CAT-TopK 冻结兼容层。Semi-Markov World Model 的训练和测试范围固定为 highD 自然驾驶片段，不包含跨数据集支持。

## Semi-Markov World Model

`world_model/src/semi_markov/model.py` 定义模型；训练、评测和运行时入口分别为：

```text
src/semi_markov/train.py
src/semi_markov/evaluation.py
src/semi_markov/environment.py
```

模型输入是 ego 与背景车的动态关系图。状态张量按 `[x, y, vx, vy, ax, ay]` 表示；关系特征由当前 ego、背景状态和 primary slot 确定性计算，包括相对位置、速度、gap、TTC 和 DRAC。关系特征不是额外随机变量。

模型的潜在变量是场景级离散交互状态及其离散持续时间。每次潜在状态开始时，`WorldRandomness` 可提供一个状态 uniform 和一个持续时间 uniform；未提供的部分由 `seed` 补足。运行环境会记录实际使用的随机数、潜在状态、持续时间和切换时刻，因此同一快照可以在另一分支继续回放。

### START 与 ROLL

- **START**：冻结 Flow 的行为锚定生成首秒基础控制，并以当前关系图作有界响应修正。
- **ROLL**：只接收已发生的 ego 历史、环境内部背景状态和关系图；不接收 ego future。
- **时域计划**：每次响应生成一秒计划，执行前 0.2 秒，并在行为锚定后以受限方式延续计划。

模型使用固定六个背景槽位：`same_front`、`same_rear`、`left_front`、`left_rear`、`right_front`、`right_rear`。当前实现不在 rollout 中做槽位重分配、车辆新增或主动消失。坐标中 `x` 为道路纵向、`y` 向左为正；默认车辆几何为长 4.8 m、宽 1.9 m，默认车道宽 3.6 m。

## 数据、训练与结果工件

highD 顺序缓存由 `src/sequential_dataset.py` 管理，是训练的既有输入；CAT 配对缓存由 `paths.legacy_dataset_dir` 指定。行为锚定依赖冻结的 76 维 Flow checkpoint 与 schema。训练和评测会验证配置中的 SHA-256，并在顺序缓存旁维护与 schema 绑定的行为锚定缓存。

`scripts/train_semi_markov_world_model.py` 将物化配置、命令、git revision、训练历史、checkpoint 和评测摘要写入新的目标输出目录。默认从随机初始化训练；`--initial-checkpoint` 仅用于兼容的续训，且训练输出始终写到新目录。

历史训练与测试工件已清理。新的 Semi-Markov World Model 训练默认输出到 `results/highd_world_model/semi_markov_world_model/`。

## 冻结 CAT-TopK 兼容层

CAT-TopK 是冻结对比基线；其源代码、从零训练脚本和测试脚本均保留。模型层实现位于 `src/cat_topk/`，共享数据、schema 与指标位于 `src/core/`；单模型入口统一为 `scripts/train_*` 与 `scripts/test_*_world_model.py`。跨模型的正式长尾比较由唯一的 `scripts/evaluate_long_tail_reproduction.py` 在统一条件下完成，不保留独立 compare 脚本。

`CATKBackgroundEnvironment` 的初始化输入是完整 Flow 场景样本、slot mask 与 primary slot；`start()` 不接收 ego future，`roll()` 只接收已发生的 ego 历史。其候选索引 `Xi_world` 是显式离散世界随机性。默认按 categorical 分布采样，也可用 `WorldSamplingConfig(candidate_selection="argmax")` 固定为确定性选择。

## 评测解释

highD replay 是背景交通重建评测，不是 ADS 闭环安全结论。CAT-TopK 在 START 使用首秒 Flow 动作摘要，而 Semi-Markov World Model 的 ROLL 没有该未来摘要；因此跨架构报告不能解释为严格同信息下的闭环安全比较。

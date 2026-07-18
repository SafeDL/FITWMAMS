# 实现说明

本文描述当前保留的 Semi-Markov World Model 与 CAT-TopK 冻结兼容层。

## Semi-Markov World Model

`world_model/src/semi_markov_model.py` 定义模型；训练、评测和运行时入口分别为：

```text
src/semi_markov_train.py
src/semi_markov_evaluation.py
src/semi_markov_environment.py
```

模型输入是 ego 与背景车的动态关系图。状态张量按 `[x, y, vx, vy, ax, ay]` 表示；关系特征由当前 ego、背景状态和 primary slot 确定性计算，包括相对位置、速度、gap、TTC 和 DRAC。关系特征不是额外随机变量。

模型的潜在变量是场景级离散交互状态及其离散持续时间。每次潜在状态开始时，`WorldRandomness` 可提供一个状态 uniform 和一个持续时间 uniform；未提供的部分由 `seed` 补足。运行环境会记录实际使用的随机数、潜在状态、持续时间和切换时刻，因此同一快照可以在另一分支继续回放。

### START 与 ROLL

- **START**：冻结 Flow 的行为锚定生成首秒基础控制，并以当前关系图作有界响应修正。
- **ROLL**：只接收已发生的 ego 历史、环境内部背景状态和关系图；不接收 ego future。
- **时域计划**：每次响应生成一秒计划，执行前 0.2 秒，并在行为锚定后以受限方式延续计划。

### 可选的场景级多候选计划

默认及当前最佳 checkpoint 仍为单一名义计划（`plan_num_modes: 1`）。实验配置可启用四个场景级候选：候选 0 与既有名义计划逐元素相同；候选 1--3 由五个有界 jerk 控制点形成整场景联合残差，且不会给单辆背景车分配独立候选。候选 logits 同时条件于场景上下文、持续意图状态和上一个剩余计划，并使用模式转移项保持短时连续性、在意图切换时允许重置。

训练使用候选控制/状态/相对几何/计划重叠能量的混合似然，跨 3--5 秒以软 Viterbi 加入模式切换代价；多样性只作用于尚未执行的计划尾部，并加入概率校准项。`WorldRandomness.plan_mode_uniforms` 控制实际的 categorical 选取，环境 trace 保存 uniform、选中模式和候选概率，快照/恢复会完整保留这些状态。当前实验没有取得验证或测试改进，未替换基线；详见 `baselines/multihypothesis_v1_experiment_log.md`。

模型使用固定六个背景槽位：`same_front`、`same_rear`、`left_front`、`left_rear`、`right_front`、`right_rear`。当前实现不在 rollout 中做槽位重分配、车辆新增或主动消失。坐标中 `x` 为道路纵向、`y` 向左为正；默认车辆几何为长 4.8 m、宽 1.9 m，默认车道宽 3.6 m。

## 数据、训练与结果工件

highD 顺序缓存由 `src/sequential_dataset.py` 管理，是训练的既有输入；CAT 配对缓存由 `paths.legacy_dataset_dir` 指定。行为锚定依赖冻结的 76 维 Flow checkpoint 与 schema。训练和评测会验证配置中的 SHA-256，并在顺序缓存旁维护与 schema 绑定的行为锚定缓存。

`scripts/train_semi_markov_world_model.py` 将物化配置、命令、git revision、训练历史、checkpoint 和评测摘要写入新的目标输出目录。默认从随机初始化训练；`--initial-checkpoint` 仅用于兼容的续训或新增多候选计划头，且训练输出始终写到新目录。

当前最佳模型的历史训练与测试工件保留在 `results/highd_world_model/behavior_anchored_semi_markov_m2_plan_state_v5/`。这是历史结果目录，不能作为新的训练输出目录。

## 冻结 CAT-TopK 兼容层

CAT-TopK 不是当前训练目标；其源代码、训练脚本以及当前最佳训练和测试结果均保留。`src/model.py`、`src/data.py`、`src/evaluation.py`、`src/environment.py` 和 `src/train.py` 用于加载 checkpoint、训练/评测 CAT-TopK，以及执行配对比较。命令行入口为 `scripts/train_cat_topk.py` 与 `scripts/test_cat_topk.py`；它们物化到新输出目录，避免改写冻结结果。

`CATKBackgroundEnvironment` 的初始化输入是完整 Flow 场景样本、slot mask 与 primary slot；`start()` 不接收 ego future，`roll()` 只接收已发生的 ego 历史。其候选索引 `Xi_world` 是显式离散世界随机性。默认按 categorical 分布采样，也可用 `WorldSamplingConfig(candidate_selection="argmax")` 固定为确定性选择。

## 评测解释

highD replay 是背景交通重建评测，不是 ADS 闭环安全结论。CAT-TopK 在 START 使用首秒 Flow 动作摘要，而 Semi-Markov World Model 的 ROLL 没有该未来摘要；因此跨架构报告不能解释为严格同信息下的闭环安全比较。

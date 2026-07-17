# 实现说明

本文描述当前保留代码的运行语义，不记录已清理候选、历史训练过程或废弃指标。

## BARS 主实现

`world_model/src/semi_markov_model.py` 定义 BARS 模型，训练、评测和运行时入口分别位于：

```text
src/semi_markov_train.py
src/semi_markov_evaluation.py
src/semi_markov_environment.py
```

模型输入是 ego 与背景车的动态关系图。状态张量按 `[x, y, vx, vy, ax, ay]` 表示；关系特征由当前 ego、背景状态和 primary slot 确定性计算，包括相对位置、速度、gap、TTC 和 DRAC。关系特征不是额外随机变量。

BARS 的潜在变量是场景级离散交互状态及其离散持续时间。每次潜在状态开始时，`WorldRandomness` 可提供一个状态 uniform 和一个持续时间 uniform；未提供的部分由 `seed` 补足。运行环境会记录实际使用的随机数、潜在状态、持续时间和切换时刻，因此同一快照可以在另一分支继续回放。

### START 与 ROLL

- **START**：BARS-M1 使用冻结 Flow 的行为锚定生成首秒基础控制，并以当前关系图作有界响应修正。
- **ROLL**：只接收已发生的 ego 历史、环境内部背景状态和关系图；不接收 ego future。
- **BARS-M2 v5**：复用 M1 的短程执行路径；计划分支在前三秒后以受限方式延续控制计划。

模型使用固定六个背景槽位：`same_front`、`same_rear`、`left_front`、`left_rear`、`right_front`、`right_rear`。当前实现不在 rollout 中做槽位重分配、车辆新增或主动消失。坐标中 `x` 为道路纵向、`y` 向左为正；默认车辆几何为长 4.8 m、宽 1.9 m，默认车道宽 3.6 m。

## 数据与冻结依赖

highD 顺序缓存由 `src/sequential_dataset.py` 管理，是世界模型训练的既有输入。BARS 的完整缓存路径由配置的 `paths.sequence_cache_dir` 指定，CAT 配对缓存由 `paths.legacy_dataset_dir` 指定；它们由上游数据流程准备，不在本目录重复构建。

行为锚定依赖冻结的 76 维 Flow checkpoint 与 schema。训练和评测会验证配置中的 SHA-256，并在顺序缓存旁维护与 schema 绑定的行为锚定缓存。不要修改该缓存或替换 Flow 工件而不同时更新配置和完整评测证据。

## 训练与结果工件

`train_bars_m2_v5.py` 在目标输出目录写入 effective config、命令、git revision、训练历史、checkpoint 和评测摘要。它先从零训练 M1，再以该 M1 为 M2 的唯一初始化和严格门控参照。

`process_highD/`、`normalizing_flow/` 与既有 BARS 顺序缓存共同提供世界模型训练输入；世界模型复现不重复它们。`scripts/train_bars_m2_v5.py` 直接读取 M1/M2 两份设计配置，覆盖输出路径后将 M1、M2 和评测串成隔离流程。它只允许 M2 从该次运行的 M1 初始化。新训练必须使用新的输出目录，不能覆盖冻结结果目录。

模型选择只保留两个外部比较对象：BARS-M1 与 CAT-TopK。`test_bars_m2_v5_against_m1.py` 在相同 BARS 缓存行、划分和随机种子下报告候选相对 M1 的 1--5 秒及 EVT-tail 指标；`test_bars_m2_v5_against_cat.py` 以同序列方式报告候选相对 CAT-TopK 的 1 秒或 5 秒指标。两者的正式运行都使用完整 test split 与 bootstrap。

## 冻结 CAT-TopK 兼容层

CAT-TopK 不是当前的训练目标。保留 `src/model.py`、`src/data.py`、`src/evaluation.py` 和 `src/environment.py`，是为了加载其冻结 checkpoint、执行 BARS 配对比较，以及在需要时提供其背景交通环境。

`CATKBackgroundEnvironment` 的初始化输入是完整 Flow 场景样本、slot mask 与 primary slot；`start()` 不接收 ego future，`roll()` 只接收已发生的 ego 历史。其候选索引 `Xi_world` 是显式离散世界随机性。默认按 categorical 分布采样，也可用 `WorldSamplingConfig(candidate_selection="argmax")` 固定为确定性选择。复现单个 episode 时，应保存实际候选索引序列或相同随机种子。

CAT 的配置 `scripts/configs/highd_world_model.yaml` 故意只包含配对所需的缓存路径和 START 摘要标记：模型架构与训练状态由冻结 checkpoint 提供。它不能用于重新训练 CAT-TopK。

## 评测解释

highD replay 是背景交通重建评测，不是 ADS 闭环安全结论。CAT-TopK 在 START 使用首秒 Flow 动作摘要，而 BARS 的 ROLL 没有该未来摘要；因此跨架构报告必须保留该信息条件说明，不能将两者解释为严格同信息下的闭环安全比较。

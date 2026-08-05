# highD 背景交通世界模型

本目录包含当前 QR-WM（Query-Refine World Model）、独立演化的 HiQR-WM（Hierarchical Interaction Query-Refine World Model）及四个可复现的对比基线：RAMP-WM、FIRM-WM、Semi-Markov WM 和 CAT-TopK。所有模型以 highD 的固定槽位场景表示为输入：1 辆 ego 加 6 个背景车辆槽位，每辆车状态为 `[x, y, vx, vy, ax, ay]`。

QR-WM 是当前正式模型；RAMP-WM、FIRM-WM、Semi-Markov WM 与 CAT-TopK 是保留的对比基线，不能用 QR 结果替换。CAT-TopK 的条件信息不同，见“完整测试集条件重建”的说明。

## 目录与数据依赖

- `src/core/`：共享的数据加载、批处理、动力学、关系特征、Flow START 解码、指标和 Flow 组合评测。
- `src/qr/`：QR-WM 的模型、训练、离线评测、Flow×QR 评测和单世界/批量在线环境。
- `src/hiqr/`：HiQR-WM 的独立层次交互状态、训练、评测、Flow×HiQR 评测和在线环境；它不会写入 QR 的源码、checkpoint、结果或缓存。
- `src/ramp/`、`src/firm/`、`src/semi_markov/`、`src/cat_topk/`：四个基线的独立实现。
- `scripts/configs/`：各模型的正式配置。
- `tests/`：共享功能及模型行为的回归测试。

RAMP、FIRM、Semi-Markov 和 CAT-TopK 各自使用其正式配置中的数据缓存。QR 使用唯一的
`results/highd_world_model/training_data/qr_sequence_cache`：每个 natural highD 片段保留 150 个记录状态 `S0..S149`，即 149 个转移（5.96 秒）。正式 QR 训练要求完整、不可截断的该缓存，并校验冻结 Flow schema：
`results/highd_tail_flow/dataset_schema.json`。

## QR-WM

### 实现与输入约束

当前 QR-WM 只维护一套实现与一份正式 checkpoint；加载时以 `model_type`、完整模型配置和严格 state-dict 形状校验兼容性，不使用历史版本编号。该实现由 relation-aware scene encoder、persistent scene memory、行为 prior、START 行为锚定控制器和 joint agent-time refiner 组成：编码器读取车辆、地图折线与车道图；行为 prior 在每个 5 Hz 响应采样 16 维、以当前已发生历史和 memory 为条件的背景行为 innovation；refiner 生成并两次细化 25 帧背景动作计划。动作被限制在配置的纵向加速度 `[-8, 4] m/s²` 与横摆角速度 `[-0.6, 0.6] rad/s` 范围内，再由运动学模型积分为背景车辆状态。

默认配置采用 25 帧（1 秒）计划、每次执行 5 帧（0.2 秒）、仿真步长 0.04 秒。每个训练、验证、checkpoint 选择和 held-out 轨迹都从真正的 `encode_start(C0,map)` 开始：`START 25 tick = 1.00 s`，随后在已生成的 25 Hz 联合历史上进入 `ROLL 124 tick = 4.96 s`；最后一个 5 Hz 响应只执行 4 tick，绝不补造 `S150`。训练分为 8 epoch 的 `buffer_warmup`、12 epoch 的 `closed_loop` 和 20 epoch 的 `full_refinement`，共 40 epoch。最佳 checkpoint 只会从完整 5.96 秒阶段按验证 FDE 选出。

`B0` 只在 START 使用一次：它初始化第一个响应的 behavior seed、场景记忆和首段动作计划；后续 ROLL 只依赖已实现的世界状态、计划缓冲区、场景记忆与当前 ego 观测，并在每个响应取新的条件 innovation。主训练不再使用无真实历史的伪 ROLL 半批；若将来加入独立 ROLL 辅助目标，必须从片段内部采样切点并提供其前 25 帧真实历史。这里的 START 严格指“片段起始行为重建”，不等同于 EVT 风险事件的起始时刻。离线重建评测中的 ego 轨迹来自日志回放；在线环境每 0.04 秒将 ADS 动作仅用于 ego 动力学，并每 0.2 秒以已发生的联合历史重规划背景。推理不接受 ADS 动作、未来 ego 控制或交通灯输入。训练时的 posterior 仅用于学习每个响应的 behavior latent；推理与在线环境使用 prior。

### 训练与恢复

```bash
python world_model/scripts/prepare_qr_sequence.py
python world_model/scripts/train_qr_world_model.py
```

前一条命令只准备 QR 数据缓存，不会训练权重。评测与 Flow×QR 只接受按当前 149-transition 协议训练并写入协议字段的 checkpoint。

可通过 `--config`、`--output-dir` 和 `--log-level` 覆盖默认值。训练中断后，恢复文件为 `checkpoints/last_qr_training_state.pt`：

```bash
python world_model/scripts/train_qr_world_model.py --resume
```

成功完成时会删除恢复文件，并在输出目录写入：

- `checkpoints/best_qr_world_model.pt`
- `training_history.csv`
- `training_progress.json`
- `training_summary.json`
- `tensorboard/`（默认启用）

基于完整的 CSV 与 TensorBoard 记录重绘最终训练曲线：

```bash
python world_model/scripts/plot_qr_training_curves.py
```

默认覆盖 `results/highd_world_model/qr_world_model/current_training_curves.png`，图中包含三个课程阶段的训练/验证目标、分阶段验证 FDE、5.96 秒精炼阶段细节和 batch 损失移动平均。

### 离线评测

```bash
python world_model/scripts/test_qr_world_model.py
```

可选参数包括 `--checkpoint`、`--output-dir` 和 `--max-sequences`；后者为 0 时评测完整测试集。评测生成 `qr_world_model_evaluation_summary.json`，包括：

- START 1.00 秒、ROLL 4.96 秒和全 5.96 秒的 ADE/FDE、速度与加速度 MAE；
- 多样本 `minADE`、`minFDE` 与轨迹多样性；
- 碰撞 episode rate、gap、TTC、DRAC；
- 速度、加速度、jerk 分布 KL；
- 相邻规划缓冲区一致性和 refine 前后的计划位置增益；
- EVT-tail 子集指标、checkpoint SHA-256、缓存与 Flow schema 信息。

### Flow 组合评测与在线接口

```bash
python world_model/scripts/evaluate_qr_long_tail.py
```

这是唯一的正式 Flow×QR 长尾评测入口。它先执行非配对的端到端 Flow×QR 分布评测并绘图，再执行成对的 START/ROLL 重建审计。Flow START 条件由 76 维 `C0+B0` 组成，其中 `C0` 为 40 维场景条件，`B0` 为 `6×6` 背景行为锚定。

该端到端研究专用输出为 `results/highd_world_model/long_tail_reproduction/`。冻结 Flow 的事件结构从其训练 split 支持的 slot mask 与主风险槽位中采样；回放参照覆盖 highD 的全部 EVT-tail 序列，并在唯一的直道路型 cohort 内，以相同事件结构和最近的初始 ego 纵向速度进行匹配，再平移到 Flow 起点。回放 ego 的连续 25 Hz 状态用于恢复环境动作，QR 每 0.2 秒只读取已经发生的联合历史。评测使用 149 tick，并将审计字段标为 `25hz_replay_velocity_transition`；ADS 动作绝不进入 QR 网络。

完成训练后，使用同一命令将端到端报告、成对审计和图表写入该目录。二者文件分别保存，避免把分布比较误解释为成对轨迹误差。

在线集成使用单世界的 `QRWorldModelEnvironment`。`metadata` 必须包含 slot mask、地图折线、车道边、主风险槽位与 Flow 对数概率审计字段：

```python
environment.reset_from_flow(C0, B0, metadata, deterministic=True)
observation = environment.observe()
next_observation = environment.step(ads_action)  # ads_action 的形状为 [2]，每 0.04 s 调用
# 以一个完整响应或最后的 4-tick 前缀推进：ads_actions 的形状为 [1..5, 2]
response_observation = environment.advance_response(ads_actions)
```

随机世界不能隐式使用全局随机数，必须显式传入每个世界的控制量：

```python
from world_model.src.qr import WorldRandomness

environment.reset_from_flow(
    C0, B0, metadata,
    deterministic=False,
    world_randomness=WorldRandomness(seed=20260729),
)
```

该 seed 控制可审计的完整响应 innovation 流：START 使用第 0 个扰动，随后每个 ROLL 响应使用按 response index 派生的独立扰动。单体环境 trace 会记录每次实际扰动；`behavior_standard_normal` 可显式控制 START，`innovation_standard_normal` 用于指定一个 AMS 子分支的下一响应扰动。单体和批量环境的 `step()` 都返回一个物理 tick 的联合状态、已执行 ego/背景动作、最新未来背景计划、`planner_updated`、物理 tick 与已完成响应计数；批量输出只是在这些字段前加 batch 维。批量场景使用 `BatchedQRWorldModelEnvironment`，只向量化网络与动力学计算；每行仍是互不共享状态、latent、记忆、计划或 ego 的独立世界。

path-level AMS 在至少完成 START 后的 0.2 秒响应边界复制前缀：

```python
prefix = environment.snapshot()
child = QRWorldModelEnvironment(model)
child.branch_from_snapshot(prefix, WorldRandomness(seed=20260730))
child_observation = child.step(ads_action)
```

`branch_from_snapshot` 只重新采样下一 ROLL innovation；它不会引入 ADS 未来计划或 ego 未来状态。相同分支随机控制严格复现，不同控制给出条件于同一前缀的不同背景未来。

`BatchedQRWorldModelEnvironment` 也提供同名的 snapshot/restore/branch 接口；其 `branch_from_snapshot` 接收每个 batch row 一个 `WorldRandomness`，以便并行执行 AMS 子分支。

### QR 正式产物

完成当前协议训练后，`results/highd_world_model/qr_world_model/` 保存唯一 checkpoint、训练记录和 held-out 评测；`long_tail_reproduction/` 保存 Flow×QR 的分布报告、审计和图表。checkpoint 的 `training_protocol` 记录 QR cache format、149 个总转移、25 个 START 转移、124 个 ROLL 转移和“每 5 Hz 响应的条件 innovation”；评测会核验这些字段，并拒绝旧的仅 START 随机 checkpoint。

## 对比基线

各基线的默认训练和评测入口如下：

| 模型 | 训练 | 评测 | 默认结果目录 |
| --- | --- | --- | --- |
| RAMP-WM | `train_ramp_world_model.py` | `test_ramp_world_model.py` | `ramp_world_model/` |
| FIRM-WM | `train_firm_world_model.py` | `test_firm_world_model.py` | `firm_world_model/` |
| Semi-Markov WM | `train_semi_markov_world_model.py` | `test_semi_markov_world_model.py` | `semi_markov_world_model/` |
| CAT-TopK | `train_cat_topk.py` | `test_cat_topk_world_model.py` | `cat_topk_world_model/` |

上述脚本均位于 `world_model/scripts/`，结果目录均位于 `results/highd_world_model/`。RAMP 和 FIRM 的测试入口支持 `--flow-composition`；CAT-TopK 的 START 使用归档的未来动作摘要，因此在五模型长尾对比中被明确标记为信息条件不对称的参考基线。

## 全 EVT-tail Flow×QR 端到端生成

`results/highd_world_model/long_tail_reproduction/` 不存放上述条件重建基线；它专门用于冻结 Flow + QR 的端到端尾部交通生成分布评测。该目录的正式运行输出 5.96 秒、30 个因果响应（最后一个为 4 tick）的结果。每次 `QRWorldModelEnvironment.reset_from_flow` 表示一个独立世界：`deterministic=True` 在每个响应使用 prior 均值；随机世界必须显式传入 `WorldRandomness(seed=...)`，以控制并审计 START 与全部 ROLL innovation。`BatchedQRWorldModelEnvironment` 的默认执行批次为 96 个 Flow 起点及其 4 条独立世界（384 条）；它要求每行都有自己的随机控制量，绝不共享状态、latent、记忆、计划或 ego。

子集模拟应复用这一批量环境、逐批规约失效指标，并在 audit 中保留 Flow 条件、world seed 和 failure indicator；不要在每个 level 调用完整 `evaluate_qr_long_tail.py`，因为该唯一正式入口还会保存全量轨迹并计算离线 FFD/MMD 分布报告。

## 回归测试

```bash
python -m pytest world_model/tests -q
```

该测试集覆盖共享 Flow 组合报告、QR START/ROLL 与显式随机性、批量独立性，以及各基线的关键形状、历史、候选分支与快照行为。

## IDM 基线

IDM 长尾安全评估是独立的传统 ADS 基线，见 `IDM_subset/README.md`；它不依赖本目录的五模型对比脚本。

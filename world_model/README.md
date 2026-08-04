# highD 背景交通世界模型

本目录包含当前 QR-WM（Query-Refine World Model）及四个可复现的对比基线：RAMP-WM、FIRM-WM、Semi-Markov WM 和 CAT-TopK。所有模型以 highD 的固定槽位场景表示为输入：1 辆 ego 加 6 个背景车辆槽位，每辆车状态为 `[x, y, vx, vy, ax, ay]`。

QR-WM 是当前正式模型；RAMP-WM、FIRM-WM、Semi-Markov WM 与 CAT-TopK 是保留的对比基线，不能用 QR 结果替换。CAT-TopK 的条件信息不同，见“完整测试集条件重建”的说明。

## 目录与数据依赖

- `src/core/`：共享的数据加载、批处理、动力学、关系特征、Flow START 解码、指标和 Flow 组合评测。
- `src/qr/`：QR-WM 的模型、训练、离线评测、Flow×QR 评测和单世界/批量在线环境。
- `src/ramp/`、`src/firm/`、`src/semi_markov/`、`src/cat_topk/`：四个基线的独立实现。
- `scripts/configs/`：各模型的正式配置。
- `tests/`：共享功能及模型行为的回归测试。

QR、RAMP、FIRM 和 Semi-Markov 默认使用
`results/highd_world_model/training_data/semi_markov_sequence_cache`；CAT-TopK 使用
`results/highd_world_model/training_data/cat_topk_start_roll_cache`。这两个缓存是训练、完整测试与 Flow replay 匹配的共同数据依赖，不是可删除的历史产物。正式 QR 训练要求完整、不可截断的序列缓存，并校验冻结 Flow schema：
`results/highd_tail_flow/dataset_schema.json`。

## QR-WM

### 实现与输入约束

当前 QR-WM 只维护一套实现与一份正式 checkpoint；加载时以 `model_type`、完整模型配置和严格 state-dict 形状校验兼容性，不使用历史版本编号。该实现由 relation-aware scene encoder、persistent scene memory、行为 prior、START 行为锚定控制器和 joint agent-time refiner 组成：编码器读取车辆、地图折线与车道图；行为 prior 在 START 采样 16 维背景行为 latent；refiner 生成并两次细化 25 帧背景动作计划。动作被限制在配置的纵向加速度 `[-8, 4] m/s²` 与横摆角速度 `[-0.6, 0.6] rad/s` 范围内，再由运动学模型积分为背景车辆状态。

默认配置采用 25 帧（1 秒）计划、每次执行 5 帧（0.2 秒）、仿真步长 0.04 秒。训练分为 8 epoch 的 `buffer_warmup`、12 epoch 的 `closed_loop` 和 20 epoch 的 `full_refinement`，共 40 epoch。最佳 checkpoint 只会从完整 5 秒阶段按验证 FDE 选出。

`B0` 只在 START 使用一次：它初始化行为 latent 的锚定、场景记忆和首段动作计划；后续 ROLL 只依赖已实现的世界状态、计划缓冲区、场景记忆与当前 ego 观测。离线重建评测中的 ego 轨迹来自日志回放；在线环境每 0.2 秒只接收当前观测到的 ego 状态。推理不接受未来 ego 控制、ADS 动作或交通灯输入。训练时的 posterior 仅用于学习行为 latent；推理与在线环境使用 prior。

### 训练与恢复

```bash
python world_model/scripts/train_qr_world_model.py
```

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

默认覆盖 `results/highd_world_model/qr_world_model/current_training_curves.png`，图中包含三个课程阶段的训练/验证目标、分阶段验证 FDE、5 秒精炼阶段细节和 batch 损失移动平均。

### 离线评测

```bash
python world_model/scripts/test_qr_world_model.py
```

可选参数包括 `--checkpoint`、`--output-dir` 和 `--max-sequences`；后者为 0 时评测完整测试集。评测生成 `qr_world_model_evaluation_summary.json`，包括：

- 1--5 秒 ADE/FDE、速度与加速度 MAE；
- 多样本 `minADE`、`minFDE` 与轨迹多样性；
- 碰撞 episode rate、gap、TTC、DRAC；
- 速度、加速度、jerk 分布 KL；
- 相邻规划缓冲区一致性和 refine 前后的计划位置增益；
- EVT-tail 子集指标、checkpoint SHA-256、缓存与 Flow schema 信息。

### Flow 组合评测与在线接口

```bash
python world_model/scripts/evaluate_qr_flow_tail_composition.py
```

Flow START 条件由 76 维 `C0+B0` 组成，其中 `C0` 为 40 维场景条件，`B0` 为 `6×6` 背景行为锚定。专用脚本读取冻结 Flow、从完整 highD EVT-tail replay 中选取条件兼容的回放，并写入端到端报告与图表。

该端到端研究专用输出为 `results/highd_world_model/long_tail_reproduction/`。冻结 Flow 的事件结构从其训练 split 支持的 slot mask 与主风险槽位中采样；回放参照覆盖 highD 的全部 EVT-tail 序列，并在唯一的直道路型 cohort 内，以相同事件结构和最近的初始 ego 纵向速度进行匹配，再平移到 Flow 起点。QR 每 0.2 秒只接收已实现的 ego 状态。

当前正式结果覆盖 2,209 条 EVT-tail 序列中的 1,761 条受 Flow 事件结构支持的回放：抽取 17,600 个 Flow 起点，每个起点生成 4 条 5 秒世界未来，共 70,400 条。正式运行使用 64 个 Flow 起点 × 4 条独立世界的批次，达到 311.0 条 5 秒未来/s；默认 CLI 批次仍为 96 个 Flow 起点。报告的 traffic-feature Fréchet 为 4.0773、RBF-MMD 为 0.01567；生成 collision episode rate 为 11.19%。这些是生成分布与物理有效性指标，不是逐样本配对重建。完整数值、审计和图表分别见 `flow_composition_evaluation.json`、`flow_start_audit.npz` 与 `figures/`。

在线集成使用单世界的 `QRWorldModelEnvironment`。`metadata` 必须包含 slot mask、地图折线、车道边、主风险槽位与 Flow 对数概率审计字段：

```python
environment.reset_from_flow(C0, B0, metadata, deterministic=True)
observation = environment.observe()
next_observation = environment.step(ego_state)  # ego_state 的形状为 [6]
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

该 seed（或等价的 `behavior_standard_normal`）只控制 START 行为 latent；给定它、已实现 ego 和当前状态，后续响应是确定的。`step()` 返回本响应周期的背景状态、已执行动作、完整未来动作计划和可审计的 Flow/随机性元数据。批量场景使用 `BatchedQRWorldModelEnvironment`，只向量化网络与动力学计算；每行仍是互不共享状态、latent、记忆、计划或 ego 的独立世界。

### 当前正式 QR 产物

`results/highd_world_model/qr_world_model/` 是唯一保留的 QR 训练与原生重建运行：`training_progress.json` 和 `training_summary.json` 记录 40/40 epoch，`qr_world_model_evaluation_summary.json` 记录完整 held-out 测试。Flow×QR 的正式结果位于 `long_tail_reproduction/`，并记录相同最佳 checkpoint 的 SHA-256。使用结果前应以各 JSON 的哈希、协议和测试集规模为准。

## 对比基线

各基线的默认训练和评测入口如下：

| 模型 | 训练 | 评测 | 默认结果目录 |
| --- | --- | --- | --- |
| RAMP-WM | `train_ramp_world_model.py` | `test_ramp_world_model.py` | `ramp_world_model/` |
| FIRM-WM | `train_firm_world_model.py` | `test_firm_world_model.py` | `firm_world_model/` |
| Semi-Markov WM | `train_semi_markov_world_model.py` | `test_semi_markov_world_model.py` | `semi_markov_world_model/` |
| CAT-TopK | `train_cat_topk.py` | `test_cat_topk_world_model.py` | `cat_topk_world_model/` |

上述脚本均位于 `world_model/scripts/`，结果目录均位于 `results/highd_world_model/`。RAMP 和 FIRM 的测试入口支持 `--flow-composition`；CAT-TopK 的 START 使用归档的未来动作摘要，因此在五模型长尾对比中被明确标记为信息条件不对称的参考基线。

## 两类正式评测

### 完整测试集条件重建（不含 Flow）

`results/highd_world_model/test_conditional_reconstruction/` 保存全部 24,216 条 held-out highD 测试序列的模型原生条件重建汇总：每个模型从真实初始交通条件、道路图和逐响应 ego 回放开始，不使用 Flow 采样。因此它衡量的是模型本身的条件闭环背景重建能力，不会将 Flow 初始状态采样误差混入结论。

### 五模型条件重建基线

```bash
python world_model/scripts/test_ramp_world_model.py
python world_model/scripts/test_firm_world_model.py
python world_model/scripts/test_semi_markov_world_model.py
python world_model/scripts/test_cat_topk_world_model.py
python world_model/scripts/test_qr_world_model.py
python world_model/scripts/evaluate_test_conditional_reconstruction.py --mode native
```

五份原生测试报告均覆盖完整 held-out split，并由统一入口以源报告与 checkpoint SHA-256 核验后写入 `results/highd_world_model/test_conditional_reconstruction/`。其中：

- `study_manifest.json`：协议、源报告、checkpoint 路径与 SHA-256；
- `overview/test_conditional_reconstruction_summary.json`：完整测试集的五秒模型原生指标索引。

RAMP、FIRM、Semi-Markov 与 QR 都在闭环中回放已实现的 logged ego；CAT-TopK 使用归档的未来动作摘要，故只能作为信息条件不对称的参考。`evaluate_test_conditional_reconstruction.py --mode diagnostic --output-dir <独立目录>` 可按需运行单模型的 32 分支深入诊断，但不会作为重复子集模拟的全量标准评测。

当前正式五秒指标如下（均越低越好；CAT-TopK 不可与其余四者作严格信息对称排序）：

| 模型 | ADE (m) | FDE (m) | gap MAE (m) | TTC error (s) | DRAC error (m/s²) |
| --- | ---: | ---: | ---: | ---: | ---: |
| RAMP-WM | 0.1282 | 0.5066 | 0.1362 | 0.0551 | 1.1698 |
| FIRM-WM | 0.1924 | 0.6950 | 0.1522 | 0.0573 | 1.2472 |
| Semi-Markov WM | 0.2064 | 0.8329 | 0.1797 | 0.0682 | 1.1960 |
| QR-WM | 0.1473 | 0.5960 | 0.1198 | 0.0288 | 0.8929 |
| CAT-TopK† | 0.0625 | 0.0734 | 0.0531 | 0.0123 | 1.2584 |

图表位于 `overview/01_model_native_comparison.png`；也可用 `python world_model/scripts/plot_reconstruction_result_summaries.py --only test` 从正式汇总重绘。

### 全 EVT-tail Flow×QR 端到端生成

`results/highd_world_model/long_tail_reproduction/` 不存放上述条件重建基线；它专门用于冻结 Flow + QR 的端到端尾部交通生成分布评测。5 秒轨迹含 25 个因果响应更新。每次 `QRWorldModelEnvironment.reset_from_flow` 表示一个独立世界：`deterministic=True` 使用先验均值；随机世界必须显式传入 `WorldRandomness(seed=...)` 或 `behavior_standard_normal`，以控制并审计 START 行为 latent。给定该 latent、已实现 ego 和当前状态后，后续响应是确定的。`BatchedQRWorldModelEnvironment` 的默认执行批次为 96 个 Flow 起点及其 4 条独立世界（384 条）；它要求每行都有自己的随机控制量，绝不共享状态、latent、记忆、计划或 ego。

子集模拟应复用这一批量环境、逐批规约失效指标，并在 audit 中保留 Flow 条件、world seed 和 failure indicator；不要在每个 level 调用完整 `evaluate_qr_flow_tail_composition.py`，因为后者还会保存全量轨迹并计算离线 FFD/MMD 分布报告。

## 回归测试

```bash
python -m pytest world_model/tests -q
```

该测试集覆盖共享 Flow 组合报告、QR START/ROLL 与显式随机性、批量独立性，以及各基线的关键形状、历史、候选分支与快照行为。

## IDM 基线

IDM 长尾安全评估是独立的传统 ADS 基线，见 `IDM_subset/README.md`；它不依赖本目录的五模型对比脚本。

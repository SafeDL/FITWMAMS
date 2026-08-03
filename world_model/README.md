# highD 背景交通世界模型

本目录包含当前 QR-WM（Query-Refine World Model）及四个可复现的对比基线：RAMP-WM、FIRM-WM、Semi-Markov WM 和 CAT-TopK。所有模型以 highD 的固定槽位场景表示为输入：1 辆 ego 加 6 个背景车辆槽位，每辆车状态为 `[x, y, vx, vy, ax, ay]`。

QR-WM 是当前正式模型；基线保留用于同条件下的横向比较，不应删除或替换。

## 目录与数据依赖

- `src/core/`：共享的数据加载、批处理、动力学、关系特征、Flow START 解码、指标和 Flow 组合评测。
- `src/qr/`：QR-WM 的模型、训练、离线评测、Flow 评测和在线环境。
- `src/ramp/`、`src/firm/`、`src/semi_markov/`、`src/cat_topk/`：四个基线的独立实现。
- `scripts/configs/`：各模型的正式配置。
- `tests/`：共享功能及模型行为的回归测试。

QR、RAMP、FIRM 和 Semi-Markov 默认使用
`results/highd_world_model/training_data/semi_markov_sequence_cache`；CAT-TopK 使用
`results/highd_world_model/training_data/cat_topk_start_roll_cache`。正式 QR 训练要求完整、不可截断的序列缓存，并校验 Flow schema：
`results/highd_tail_flow/dataset_schema.json`。

## QR-WM

### 实现与输入约束

QR-WM 的当前架构版本为 5。它用 relation-aware scene encoder 编码车辆、地图折线和车道图；以 persistent scene memory 保存场景记忆；用行为 latent 和联合 agent-time refiner 生成并迭代细化背景车辆的未来动作。动作被限制在配置的纵向加速度和横摆角速度范围内，再由运动学模型积分为背景车辆状态。

默认配置采用 25 帧（1 秒）计划、每次执行 5 帧（0.2 秒）、仿真步长 0.04 秒。训练分为 8 epoch 的 `buffer_warmup`、12 epoch 的 `closed_loop` 和 20 epoch 的 `full_refinement`，共 40 epoch。最佳 checkpoint 只会从完整 5 秒阶段按验证 FDE 选出。

离线重建评测中的 ego 轨迹来自日志回放；在线环境则每 0.2 秒只接收当前观测到的 ego 状态。推理不接受未来 ego 控制、ADS 动作或交通灯输入。训练时的 posterior 仅用于学习行为 latent，推理与在线环境使用 prior。

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
python world_model/scripts/test_qr_world_model.py --flow-composition
```

Flow START 条件由 76 维 `C0+B0` 组成，其中 `C0` 为 40 维场景条件，`B0` 为 `6×6` 背景行为锚定。`B0` 只在 START 阶段用于初始化行为 latent、场景记忆和首段动作计划；之后的 ROLL 只使用已发生状态、维护的计划缓冲区和场景记忆。

该评测对每个支持的 held-out EVT-tail 回放抽取 8 个 Flow 起点，并对每个起点生成 4 条世界未来；默认 5 秒。它写入 `flow_composition_evaluation.json` 和 `flow_start_audit.npz`。这是生成分布与物理有效性评测，不是逐样本配对重建。

在线集成使用 `QRWorldModelEnvironment`：

```python
environment.reset_from_flow(C0, B0, metadata)
observation = environment.observe()
next_observation = environment.step(ego_state)  # ego_state 的形状为 [6]
```

`step()` 返回本响应周期的背景状态、已执行动作、完整未来动作计划和可审计的 Flow 元数据。

### 当前正式 QR 产物

`results/highd_world_model/qr_world_model/` 是唯一保留的 QR 正式运行。其 `training_progress.json` 和 `training_summary.json` 记录 40/40 epoch；重建评测与 Flow 组合评测均记录同一最佳 checkpoint 的 SHA-256。使用结果前应以这些 JSON 中的哈希、协议和测试集规模为准。

## 对比基线

各基线的默认训练和评测入口如下：

| 模型 | 训练 | 评测 | 默认结果目录 |
| --- | --- | --- | --- |
| RAMP-WM | `train_ramp_world_model.py` | `test_ramp_world_model.py` | `ramp_world_model/` |
| FIRM-WM | `train_firm_world_model.py` | `test_firm_world_model.py` | `firm_world_model/` |
| Semi-Markov WM | `train_semi_markov_world_model.py` | `test_semi_markov_world_model.py` | `semi_markov_world_model/` |
| CAT-TopK | `train_cat_topk.py` | `test_cat_topk_world_model.py` | `cat_topk_world_model/` |

上述脚本均位于 `world_model/scripts/`，结果目录均位于 `results/highd_world_model/`。RAMP 和 FIRM 的测试入口支持 `--flow-composition`；CAT-TopK 的 START 使用归档的未来动作摘要，因此在五模型长尾对比中被明确标记为信息条件不对称的参考基线。

## 五模型长尾对比

```bash
python world_model/scripts/evaluate_long_tail_reproduction.py
```

该脚本在 held-out EVT-tail 序列上比较五个 checkpoint。所有模型使用相同的日志一秒历史、冻结 B0、道路图和逐响应 ego 回放；每个条件生成 32 条分支，其中第 1 条为确定性分支。默认输出目录为 `results/highd_world_model/long_tail_reproduction/`，包含：

- `study_manifest.json`：协议、缓存、checkpoint 路径与 SHA-256；
- `selected_events.json`：代表性长尾事件；
- `overview/`：汇总 JSON 和总览图；
- 每个模型的 `metrics.json`、六张图和三段事件 GIF。

`--max-sequences` 仅用于开发运行，并且必须配合非正式输出目录，防止覆盖正式对比结果。

## IDM 基线

IDM 长尾安全评估是独立的传统 ADS 基线，见 `IDM_subset/README.md`；它不依赖本目录的五模型对比脚本。

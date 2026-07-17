# 背景交通世界模型

本目录保留当前可运行的 highD 背景交通世界模型实现，以及后续 BARS 研究所需的最小比较接口。

| 对象 | 状态 | 用途 |
| --- | --- | --- |
| **BARS-M1** | 冻结 incumbent | BARS 系列的正式内部对照与后续尝试的起点。 |
| **BARS-M2 State-Supervised Clean-Plan Carry v5** | 当前保留候选 | 当前最强的已评测 BARS 实现；不增加第三个基线。 |
| **CAT-TopK** | 冻结外部基线 | 仅用于与 BARS 的跨架构配对比较。 |

BARS 是 **Behavior-Anchored Relational Semi-Markov World Model**（行为锚定关系半马尔可夫世界模型）。它以动态交通关系图表示场景，以场景级离散半马尔可夫状态表示交互意图，并以持续意图和即时响应共同生成背景车控制。BARS-M1 在 START 的首秒使用冻结 Flow 提供的行为锚定；之后 ROLL 只使用已发生的 ego 历史和模型状态。

## 当前边界

- 不读取 ADS 身份、ADS 网络特征、ego future、风险标签或 `risk_trace`。
- 背景车使用六个固定槽位；当前 rollout 不做车辆新增、消失或槽位重分配。
- CAT-TopK 的 START 使用冻结的首秒 Flow 动作摘要；这与 BARS 的信息条件不同，所有 CAT 对比报告都会标记该事实。
- CAT-TopK 只保留 checkpoint 兼容、环境调用和配对比较；不提供其数据准备、训练或单独评测流程。

## 从零训练世界模型

世界模型复现不重复任何上游数据工作。`process_highD/`、`normalizing_flow/` 与既有 world-model 缓存负责准备训练输入；`scripts/train_bars_m2_v5.py` 只校验完整 BARS 顺序缓存和冻结 Flow，然后在新的输出目录中训练 BARS-M1、由该次运行的 M1 初始化 BARS-M2 v5，并评测两者。

```bash
python world_model/scripts/train_bars_m2_v5.py
```

默认读取 M1/M2 的两份设计配置，输出写至 `results/highd_world_model/bars_m2_v5_reproduction/`。可用 `--m1-config`、`--m2-config` 和 `--output-dir` 显式覆盖；`--stages validate m1 m2 evaluate` 可分阶段重启，后一阶段只读取准备好的输入或该输出目录中前一阶段产生的工件。

该流程不读取任何历史 BARS checkpoint；冻结 Flow 与顺序缓存是明确的上游训练输入。随机种子、全部物化配置和输入路径会写入 `reproduction_manifest.yaml`。由于浮点硬件、依赖版本和优化的非确定性，从零训练不承诺产生与已发布 checkpoint 字节相同的权重；它复现当前最强世界模型的完整模型设计、训练和测试流程。

## 冻结工件的评测与对比

当前工作区已保留以下运行依赖：

```text
results/highd_world_model/semi_markov_relational_full/                 # BARS 顺序缓存
results/highd_world_model/shared_dense_start_roll/                    # 冻结 CAT 配对缓存
results/highd_tail_flow_best/checkpoints/best_tail_conditional_maf.pt # 冻结 Flow
results/highd_tail_flow_best/dataset_schema.json                      # Flow schema
results/highd_world_model/behavior_anchored_semi_markov_m1_start_roll_v2/checkpoints/best_semi_markov_relational.pt
results/highd_world_model/behavior_anchored_semi_markov_m2_plan_state_v5/checkpoints/best_semi_markov_relational.pt
results/highd_world_model/catk_topk/checkpoints/best_world_model.pt
```

这些冻结工件用于已发布 checkpoint 的加载、推理、highD 评测和配对比较。不要把 M1/M2 的保留结果目录当作新的训练输出目录；新的从零复现与后续实验都应使用独立输出目录。

## 配置与评测

| 模型 | 配置 | checkpoint |
| --- | --- | --- |
| BARS-M1 | `scripts/configs/highd_behavior_anchored_semi_markov.yaml` | `results/highd_world_model/behavior_anchored_semi_markov_m1_start_roll_v2/checkpoints/best_semi_markov_relational.pt` |
| BARS-M2 v5 | `scripts/configs/highd_bars_m2_plan_carry_3s.yaml` | `results/highd_world_model/behavior_anchored_semi_markov_m2_plan_state_v5/checkpoints/best_semi_markov_relational.pt` |
| CAT-TopK | `scripts/configs/highd_world_model.yaml` | `results/highd_world_model/catk_topk/checkpoints/best_world_model.pt` |

上表的 M1/M2 配置同时定义模型设计，并用于加载/评测已发布 checkpoint。复现脚本读取它们后会覆盖输出路径和 M2 的初始化来源，因此不会写入冻结结果目录，也不会读取历史 BARS checkpoint。

脚本目录只保留一个训练入口和两个配对测试入口：

```text
train_bars_m2_v5.py                 # 从零训练 M1，再训练/评测 M2 v5
test_bars_m2_v5_against_m1.py       # M2 v5 与 M1 的配对 test
test_bars_m2_v5_against_cat.py      # M2 v5 与冻结 CAT-TopK 的配对 test
```

使用项目的 PyTorch 环境执行。冻结 M2 v5 的 M1 配对测试为：

```bash
python world_model/scripts/test_bars_m2_v5_against_m1.py \
  --candidate-config world_model/scripts/configs/highd_bars_m2_plan_carry_3s.yaml \
  --candidate-checkpoint results/highd_world_model/behavior_anchored_semi_markov_m2_plan_state_v5/checkpoints/best_semi_markov_relational.pt
```

训练脚本会校验 Flow checkpoint、schema 与完整顺序缓存；配对测试把报告写入候选配置的 `paths.output_dir`。若要保护已发布结果，请为新候选使用独立输出目录。

## 对比协议

后续 BARS 候选只有两个比较对象：冻结 BARS-M1 和冻结 CAT-TopK。机制消融仅用于归因，不构成额外基线。

相对 M1 的完整 highD test 配对比较：

```bash
python world_model/scripts/test_bars_m2_v5_against_m1.py \
  --candidate-config <new-config.yaml> \
  --candidate-checkpoint <new-checkpoint.pt>
```

相对 CAT-TopK 的同序列比较分别运行 1 秒和 5 秒：

```bash
python world_model/scripts/test_bars_m2_v5_against_cat.py \
  --semi-config <new-config.yaml> \
  --semi-checkpoint <new-checkpoint.pt> \
  --horizon-seconds 1

python world_model/scripts/test_bars_m2_v5_against_cat.py \
  --semi-config <new-config.yaml> \
  --semi-checkpoint <new-checkpoint.pt> \
  --horizon-seconds 5
```

这两个脚本默认使用完整 test split、确定性 rollout 和 2,000 次 bootstrap。候选配置的 `paths.output_dir` 用于写入报告；小规模 smoke run 应使用 `--output-suffix`，避免覆盖正式报告。

## BARS 环境接口

`world_model.src.semi_markov_environment.SemiMarkovBackgroundEnvironment` 是 BARS 的运行时环境。它接收调用方构造的 `DynamicTrafficGraph` 和外生 `WorldRandomness`，不接收 ego future：

```python
environment.reset(initial_graph, world_randomness, behavior_anchor=anchor)
result = environment.step(ego_state, ego_valid=True, dt=0.2)
one_second = environment.roll(ego_history_states, ego_history_valid)
snapshot = environment.snapshot()
environment.restore(snapshot)
```

`step()` 是 0.2 秒响应更新；`roll()` 是连续五次响应更新的兼容包装。`snapshot()` 与 `restore()` 保存图、背景状态、历史、潜在状态、持续时间、计划状态和随机数状态，以支持分支后确定性继续。BARS-M1 的 `behavior_anchor` 只在 START 首秒生效；M2 v5 在前三秒保留 M1 执行路径，随后使用受限计划延续。

## CAT-TopK 冻结接口

若需要将冻结 CAT-TopK 用作背景环境，使用 `world_model.src.environment.CATKBackgroundEnvironment.from_checkpoint()` 加载保存的 checkpoint。它以完整 Flow 样本初始化，`start()` 生成首秒，`roll()` 只接受已经发生的 ego 历史。候选索引是其显式世界随机变量；需要复现实例时，应保存 `world_seed` 或每个 chunk 的 `xi_world_candidate_indices`。

更具体的实现约定见 [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)。

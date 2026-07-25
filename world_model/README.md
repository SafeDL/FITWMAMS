# 背景交通世界模型

本目录包含独立训练的 **RAMP-WM**（关系记忆驱动的联合多假设滚动自回归世界模型）、当前 Semi-Markov World Model 和冻结的 CAT-TopK 兼容层。RAMP-WM 面向固定 highD 自然驾驶片段：以动态交通关系图、连续场景记忆和场景级联合候选计划生成 0.2 秒响应的闭环背景车演化；它不使用离散行为状态、边界或持续时间模型。

| 对象 | 状态 | 用途 |
| --- | --- | --- |
| **Semi-Markov World Model** | 当前最佳模型 | 唯一的半马尔可夫世界模型训练、评测与运行时实现。 |
| **CAT-TopK** | 冻结外部基线 | 保留源代码、训练脚本、checkpoint 兼容和配对比较。 |
| **RAMP-WM** | 独立候选方法 | 从随机初始化训练的连续记忆、联合八候选、重叠滚动规划世界模型。 |

## RAMP-WM：从零训练、监控和评测

RAMP-WM 不加载 Semi-Markov 或 CAT-TopK checkpoint。START 仅使用冻结 Flow 的 B0 行为锚定，第一秒后该锚定从运行状态中移除；ROLL 仅接收已发生的 ego 状态和已生成的背景历史。

```bash
python world_model/scripts/train_ramp_world_model.py
python world_model/scripts/test_ramp_world_model.py
python world_model/scripts/compare_ramp_baselines.py --horizon-seconds 5
python world_model/scripts/evaluate_ramp_distribution.py
```

RAMP 工件写入 `results/highd_world_model/ramp_world_model/`。CAT-TopK 的 START 仍使用冻结的未来首秒动作摘要；配对报告会将该条件显式标记为信息不对称，不能作为严格同信息提升声明。

## 当前边界

- 不读取 ADS 身份、ADS 网络特征、ego future、风险标签或 `risk_trace`。
- 背景车使用六个固定槽位；当前 rollout 不做车辆新增、消失或槽位重分配。
- 当前训练和测试范围是固定 highD 自然驾驶数据的未来 0--5 秒闭环片段；不包含跨数据集支持。
- Semi-Markov World Model 的 START 首秒使用冻结 Flow 的行为锚定；ROLL 只使用已发生的 ego 历史和模型状态。
- CAT-TopK 的 START 使用冻结的首秒 Flow 动作摘要；这与 Semi-Markov World Model 的信息条件不同，所有 CAT 对比报告都会标记该事实。

## 从零训练 Semi-Markov World Model

世界模型复现不重复任何上游数据工作。`process_highD/`、`normalizing_flow/` 与既有 world-model 缓存负责准备训练输入；训练入口只校验完整顺序缓存和冻结 Flow，随后从随机初始化训练并评测单一的 Semi-Markov World Model。

```bash
python world_model/scripts/train_semi_markov_world_model.py
```

默认配置为 `scripts/configs/highd_semi_markov_world_model.yaml`，输出写至 `results/highd_world_model/semi_markov_world_model/`。可用 `--config`、`--output-dir` 覆盖，或用 `--stages validate train evaluate` 分阶段执行。该流程不读取历史 Semi-Markov checkpoint；冻结 Flow 与顺序缓存是明确的上游训练输入。

## 冻结工件与评测

历史 checkpoint、训练记录和测试报告已清理；重新训练的 Semi-Markov World Model 默认写入：

```text
results/highd_world_model/semi_markov_world_model/
```

CAT-TopK 的训练 checkpoint、训练记录、测试摘要和损失图统一保存在 `results/highd_world_model/cat_topk_world_model/`；Semi-Markov 对应工件统一保存在 `results/highd_world_model/semi_markov_world_model/`。CAT-TopK 的源代码、训练脚本和配置保留在：

```text
world_model/src/{model,data,evaluation,environment,train}.py
world_model/scripts/configs/highd_cat_topk_world_model.yaml
```

## 单模型测试与 CAT-TopK 配对比较

```bash
python world_model/scripts/test_semi_markov.py
python world_model/scripts/test_cat_topk.py
```

最终横向比较使用配对脚本：

```bash
python world_model/scripts/compare_semi_markov_cat_topk.py \
  --semi-config world_model/scripts/configs/highd_semi_markov_world_model.yaml \
  --semi-checkpoint results/highd_world_model/semi_markov_world_model/checkpoints/best_semi_markov_relational.pt \
  --horizon-seconds 1
```

将 `--horizon-seconds` 改为 `5` 可运行五秒对比。脚本默认使用完整 test split、确定性 rollout 和 2,000 次 bootstrap；小规模 smoke run 应使用 `--output-suffix`，避免覆盖正式报告。

## CAT-TopK 训练与测试

CAT-TopK 的训练与单模型测试入口保留如下；训练默认输出到 `results/highd_world_model/cat_topk_world_model/`：

```bash
python world_model/scripts/train_cat_topk.py
python world_model/scripts/test_cat_topk.py
python world_model/scripts/test_semi_markov.py
```

`scripts/configs/highd_cat_topk_world_model.yaml` 固化当前最佳 CAT-TopK checkpoint 的模型结构与数据缓存路径。若要从头训练，使用新的 `--output-dir`；若要评测现有最优 checkpoint，可直接运行测试脚本。

## 环境接口

`world_model.src.semi_markov_environment.SemiMarkovBackgroundEnvironment` 是 Semi-Markov World Model 的运行时环境。它接收调用方构造的 `DynamicTrafficGraph` 和外生 `WorldRandomness`，不接收 ego future：

```python
environment.reset(initial_graph, world_randomness, behavior_anchor=anchor)
result = environment.step(ego_state, ego_valid=True, dt=0.2)
one_second = environment.roll(ego_history_states, ego_history_valid)
snapshot = environment.snapshot()
environment.restore(snapshot)
```

`step()` 是 0.2 秒响应更新；`roll()` 是连续五次响应更新的兼容包装。`snapshot()` 与 `restore()` 保存图、背景状态、历史、潜在状态、持续时间、计划状态和随机数状态，以支持分支后确定性继续。

## CAT-TopK 冻结接口

若需要将冻结 CAT-TopK 用作背景环境，使用 `world_model.src.environment.CATKBackgroundEnvironment.from_checkpoint()` 加载保存的 checkpoint。它以完整 Flow 场景样本初始化，`start()` 生成首秒，`roll()` 只接受已经发生的 ego 历史。候选索引是其显式世界随机变量；需要复现实例时，应保存 `world_seed` 或每个 chunk 的 `xi_world_candidate_indices`。

更具体的实现约定见 [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)。

# Semi-Markov Relational Traffic World Model：实现与验证状态

更新日期：2026-07-14。本文记录 `world_model_goal_semi_markov_updated.md` 的可复现实验状态；它不把开发缓存结果表述为正式替代冻结基线。

## 已实现并验证的路径

- 可变参与者 `DynamicTrafficGraph`，highD 车道拓扑适配器、rounD 曲线地图适配器、top-R 车道归属、冲突区域节点和动态参与者—冲突区边；冲突区特征同时进入训练、评测和细粒度闭环环境，模型主干不读取 legacy slot 名称。
- 场景级离散 Semi-Markov latent state、离散 hazard 持续时间和窗口末端右删失项。
- `mode + gate * response` 控制解码、可微物理积分、`0.2 s` 的 `step()` 与五次 step 的一秒 `roll()` 兼容包装。可选的 control-curve head 在每个响应点预测五个 25 Hz 控制点、逐点积分当前 0.2 s 区间，并在下一个响应点重新规划。
- 训练时后验仍读取完整六秒序列；闭环生成部分按每 batch 随机 1--5 秒展开，并每 5 个 response steps 做 TBPTT。评测始终使用完整五秒。
- 环境 snapshot/restore 保存图、25 帧历史、latent state、剩余 duration、RNG 状态、未消费 uniforms 和完整 trace，可在新环境实例中确定性继续，供 AMS 分支复制使用。
- `merge`、`diverge`、`cross` 不再被压缩成 adjacent-lane：它们进入动态图边特征和异构注意力；highD→rounD 的可选 conflict-attention 参数是唯一允许的部分 checkpoint 迁移。
- 因果先验 checkpoint 选择：每轮在 validation 集执行固定随机数的五秒自由 rollout，以 `causal_prior_rollout_FDE_m` 选择，而不是以后验重建损失选择。
- 因果先验 rollout 不读取未来背景参与者的有效掩码；背景成员关系从当前生成图因果延续，只有外部已发生的 ego 有效性可在 replay 中前进。
- Flow 使用单独的 `clean_start` schema：40 维 anchor 时刻物理状态；`dataset_schema.json` 显式记录 `initial_observation_only=true`、`future_action_summaries_included=false` 和空 `trajectory_features`。
- `graph_from_clean_start()` 将 clean Flow 样本映射到初始动态图，并拒绝遗留的 76 维、含未来动作摘要的 Flow 向量。
- rounD adapter 可读取标准 `tracks.csv`、JSON/NPZ 向量地图 sidecar，或授权 rounD 包内的 Lanelet2 `.osm` 地图；OSM 路径直接恢复中心线、successor/merge/diverge/adjacent/cross 拓扑并推导冲突区。已具备 150 帧 sequence-cache、独立训练/评估、highD→rounD 迁移，以及磁盘式 highD+rounD 联合缓存入口。实际 rounD 数据尚未提供，因此这些路径目前仅完成 I/O/动态图/联合缓存 smoke，不是数据集实证。

## 可复现实验

| 项目 | 位置 / 配置 | 结果 |
| --- | --- | --- |
| clean Flow 数据 | `results/highd_tail_flow_clean_start/` | 2,209 条 EVT-tail 初始状态；train/val/test = 1,550/330/329；40 features，0 future-action features。 |
| clean Flow 训练 | `normalizing_flow/scripts/configs/highd_tail_flow_clean_start.yaml` | 80 epochs；best val NLL = -28.3204，test NLL = -35.5230。 |
| clean Flow 采样 | `results/highd_tail_flow_clean_start/evaluation_summary.json` | 256 个样本；物理 invalid/overlap/negative-gap/semantic-error rate 均为 0。 |
| highD 世界模型训练 | `world_model/scripts/configs/highd_semi_markov_relational_10k.yaml` | 10,000 条序列开发缓存，train/val/test = 7,026/1,483/1,491。 |
| 因果先验续训 | `world_model/scripts/configs/highd_semi_markov_relational_10k_finetune.yaml` | 从 10k checkpoint 续训 20 epochs；validation FDE 从 1.3205 m 降至 1.3154 m（epoch 17）。 |
| 完整 highD 缓存 | `results/highd_world_model/semi_markov_relational_full/sequence_cache/` | 161,314 条六秒序列；train/val/test = 112,943/24,155/24,216；非有界缓存。 |
| 全量因果微调 | `world_model/scripts/configs/highd_semi_markov_relational_full_finetune.yaml` | 以 10k 续训模型为起点训练 3 epochs；验证因果 5 s FDE 从 1.3240 m 降至 1.2687 m（best=epoch 3）。 |
| 全量随机展开/TBPTT 续训 | `world_model/scripts/configs/highd_semi_markov_relational_full_tbptt_finetune.yaml` | 3 epochs；平均 14.57--14.94 个 0.2 s response steps/batch，TBPTT=5。后三轮验证 FDE 未优于起始模型，竞争式选择保留起始参数。 |
| 长时域端点损失开发检查 | `highd_semi_markov_relational_full_long_horizon_finetune.yaml` | 10,000 train / 2,000 validation 序列、1 epoch；端点权重使固定开发 FDE 从 1.26560 m 变为 1.27191 m，未改善，故未扩大为全量候选。 |
| 完整五秒反传开发检查 | `highd_semi_markov_relational_10k_full_bptt.yaml` | 10k 同一 split、从原 checkpoint 续训 5 epochs；测试 5 s FDE = 1.35170 m（原 10k 模型约 1.37768 m），有小幅改善但仍远高于冻结基线，未扩大。 |
| 物理控制曲线开发检查 | `highd_semi_markov_relational_10k_control_plan_bptt.yaml` | 10k 同一 split、每个 0.2 s 区间实际积分五个 25 Hz 控制点、共 10 轮；测试 5 s FDE = 1.34856 m（无曲线 BPTT 为 1.35170 m），改善不足以扩大。 |
| 相对 ego 位置编码检查 | `highd_semi_markov_relational_10k_ego_relative_position.yaml` | 10k 同一 split、30 epochs；测试 5 s FDE = 1.66040 m，劣化，已拒绝。 |
| 完整 cache BPTT 受限开发 | `highd_semi_markov_relational_full_bptt_dev.yaml` | 明确记录 20,000 train / 5,000 validation 限制；完整 held-out test 5 s FDE = 1.28102 m，较正式 checkpoint 仅低 0.00286 m，非完整训练且改善不足，不能推广。 |
| 完整 held-out 测试 | `results/highd_world_model/semi_markov_relational_full_tbptt_finetune/semi_markov_evaluation_summary.json` | 24,216 条从未参与训练或选择的测试序列，固定种子因果先验 rollout；包含受控响应诊断。 |
| 完整配对基线比较 | `results/highd_world_model/semi_markov_relational_full_tbptt_finetune/paired_semi_markov_vs_catk.json` | 同一 24,216 条序列、坐标一致性误差 0、2,000 次 bootstrap；三项 1 s 主误差门控均通过。 |
| 原型审计 | `results/highd_world_model/semi_markov_relational_full_tbptt_finetune/latent_state_prototypes.json` | 全部 24,155 validation 序列；状态使用率、duration histogram、按状态加权的关系边/车道/主交互对象变化率。 |

最终全量 checkpoint：

```text
results/highd_world_model/semi_markov_relational_full_tbptt_finetune/checkpoints/best_semi_markov_relational.pt
SHA-256: 7cf3733fcb142ef31c1a997f6cbb5164e6ead800e5e98afbdca2de55fa0f7253
```

在完整的 24,216 条 held-out test 序列上，因果先验自由 rollout 为：

| 指标 | 结果 |
| --- | ---: | ---: |
| 1 s ADE / FDE (m) | 0.030372 / 0.041625 |
| 5 s ADE / FDE (m) | 0.379603 / 1.283882 |
| 5 s gap MAE (m) | 0.330469 |
| duration Brier / ECE | 0.057832 / 0.079100 |
| latent duration / episode switches | 7.41 个 0.2 s steps / 4.16 |
| acceleration / jerk out-of-range rate | 0 / 0 |
| physical invalid / negative-gap / overlap rate | 0.000248 / 0.000012 / 0.000012 |

相对进入本次全量微调前的 checkpoint，验证 FDE 改善 4.2%；完整 held-out 测试的 5 s ADE/FDE 也从 0.39444/1.33991 m 降至 0.37960/1.28388 m。TBPTT 续训和额外的端点加权开发检查都没有进一步改善，因此最终参数与这一最佳起始权重相同，但 checkpoint 额外记录了随机展开/TBPTT 协议。该模型不是每个 response step 随机重采样，平均 latent duration 为 7.41 个 response steps。

物理诊断已经改为动态图的几何定义：只有横向车体相交时，纵向投影重叠才算负间距；固定槽位语义不用于可变参与者模型。因此旧报告中的 5.1% “negative gap” 不再是正确的动态模型碰撞率。速度、加速度和 jerk 越界率均为 0。

在固定 256 个 held-out episode 上的受控响应诊断中，同一物理 ego 输入的重复输出误差为 0；对 ego 的 `+1.5 m/s²` 加速、`-2.0 m/s²` 制动、`+0.20 m/s²` 横向扰动，背景控制平均变化范数分别为 0.09497、0.20712、0.03164，且控制时间跳变范数保持 0.01120、0.01922、0.00695。该诊断没有反事实 ground truth，不作因果真实性声明。

冻结 CAT-K 的完整配对 1 s 比较（候选减基线；负值为候选更好）：

| 指标 | 候选 | CAT-K | 差值 | 单侧 95% bootstrap 上界 |
| --- | ---: | ---: | ---: | ---: |
| ADE (m) | 0.031848 | 0.033342 | -0.001494 | -0.001459 |
| FDE (m) | 0.043308 | 0.047644 | -0.004335 | -0.004202 |
| gap MAE (m) | 0.027395 | 0.028928 | -0.001533 | -0.001499 |

三项门控均通过；该历史 CAT-K 基线仍使用含未来动作摘要的旧 Flow，因此报告明确标记 `promotion_information_symmetric=false`，不把这一比较表述为信息完全对称的因果优势。

完整五秒配对也已运行于相同的 24,216 条 held-out 序列（`paired_semi_markov_vs_catk_5s.json`，坐标一致性误差为 0）。它反驳了“当前模型已经在长时域替代 CAT-K”的说法：候选 / CAT-K 的累计 ADE 为 `0.37837 / 0.16348 m`，终点 FDE 为 `1.26276 / 0.66762 m`，gap MAE 为 `0.32781 / 0.13651 m`；关系分布 TV 为 `0.002541 / 0.000376`。因此候选在误差累积和关系分布漂移两项上都没有胜出。评测器已将这一同序列五秒门槛写入 promotion 判定，不能再只因 rounD 缺失而掩盖该 highD 长时域缺口。

共享 10k cache 的核心消融已经完成四种机制且使用完全相同的 7,026/1,483/1,491 train/val/test 划分（`core_ablation_summary.json`）。当前 20-epoch snapshot 的 5 s FDE 分别为 B0 `1.43257`、B1 `1.66249`、B2 `1.69288`、Full `1.64329 m`；训练轮数不足以将该 snapshot 表述为 Full 优于 B0 的收敛结论，它仅证明四个实现分支及其相同数据协议可复现。

## 验收结论

当前 checkpoint 的 `promotion.status` 为 `not_promoted`。highD 全量缓存、duration calibration/persistence、冻结 CAT-K 的完整配对 bootstrap 和 1 s 不劣门控均已通过；但规范的五秒误差累积/关系漂移门槛没有通过。当前用户范围还明确暂缓 rounD 数据下载和处理；这不阻止 highD 开发，但仍保留为正式跨数据集验收缺口：

1. 工作区没有 rounD 原始轨迹与矢量地图文件。因此 rounD adapter 的 CSV/vector-map/Lanelet2-OSM I/O、150 帧可训练缓存、曲线路网、冲突区、动态参与者、highD→rounD conflict-attention 迁移和 highD+rounD 联合缓存 smoke 已通过，但尚未完成 rounD 单数据集、联合训练和迁移实证。必须提供实际数据（或其已授权路径）后才能生成独立 full-rounD summary 并解除晋级门控；门控还会验证该摘要明确来自 rounD cache，而非任意非有界摘要。

核心消融可通过 `world_model/scripts/run_semi_markov_ablation.py --variant b0|b1|b2|full` 在共享缓存上运行：B0=单模式动态图，B1=联合 latent、逐 response 状态更新，B2=learned duration 但禁用即时 response，Full=完整模型。完整摘要由 `summarize_semi_markov_ablations.py` 生成，并把开发 cache 标记为非正式 full-highD 结论。

已运行的回归检查：25 个 Semi-Markov `unittest`（包括 clean Flow→graph、曲线车道、highD 录制地图、step/roll 等价、完整 snapshot/restore、随机 TBPTT、rounD CSV/vector-map/Lanelet2-OSM/cache、joint-cache padding、冲突区端到端环境传递、未来背景 validity 因果隔离、冲突区迁移、响应 probe、动态图物理诊断、完整 BPTT、可选相对位置、25 Hz control-curve checkpoint 迁移/反向传播/环境积分和 modal-duration 无隐藏 uniform smoke）全部通过；2 个 clean Flow schema `pytest` 全部通过；一秒与五秒配对脚本分别在完整 24,216 条正式 run 通过坐标一致性检查；`py_compile` 与 `git diff --check` 通过。

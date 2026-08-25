# 扩散引导 HiQR 分层交通世界模型

本模块实现统一的三级接口：

\[
p(M)p(C_0\mid M)p(K\mid C_0,M)
\rightarrow p_\theta(\tau_{\rm soft}\mid C_0,M,K,z_{\rm diff})
\rightarrow \pi_\phi(a_{\rm bg}\mid H_t,\tau_{\rm soft},z_{\rm HiQR}).
\]

Flow 与冻结扩散共享 `C0(40)+M(6)+K(72)=118` 维物理契约。`K` 是三时刻、六背景槽位的状态结点，不再拆成额外行为符号。扩散生成六车联合 soft plan；HiQR 每 0.04 s 根据已实现历史重对齐预览，并只提交下一帧动作。EVT 只提供外部人类风险标尺，不进入模型损失。

短程层复用 HiQR 的历史—关系查询编码器、车道几何、只更新已观测状态的递归滤波器、1 s 场景潜变量、连续相关的车辆潜变量和联合 jerk-knot 解码。首次调用在 `filter_state` 不存在时初始化观测状态，之后每 0.04 s 只用已实现联合状态递推更新；历史编码器的实际敏感性由单独诊断审计，不将其作用强度作为未经验证的主张。soft-plan 位置、速度、参考控制与当前偏差作为 preview token 同时进入 prior 和 decoder；解码器预测短计划但只提交第一帧，最终控制满足 `j_bg=j_soft+Δj_HiQR`，不存在输出端 gate 或硬轨迹回放。

`scenario_seed` 控制 `z_scenario=(u_M,z_C0,z_K)`；`motion_seed` 独立寻址扩散噪声、场景潜变量和车辆潜变量。snapshot/restore 保存滤波状态、慢潜变量和随机流位置。模型从不接收未来 ego 动作，干预只能从下一次重规划开始生效。

## 当前证据

最终随机—因果模型使用 72,771/13,133 条 recording 隔离训练/验证序列训练，并在全部 10,151 条测试序列上确认：

- 25 Hz 闭环 ADE/FDE/P95 为 `0.03906/0.03306/0.08931 m`，相对 E1 基线没有劣化。
- 固定条件 16 样本的 energy score 相对基线改善 `2.25%`，终点成对距离为基线 `2.41×`。
- 制动/加速方向成功率为 `0.989/1.000`，自然响应 P10–P90 覆盖率为 `0.865/0.685`；横向分离不下降率为 `0.901`，三类干预局部性均低于 `0.15`。

当前证据支持将其视为现行数据驱动范式下的交通世界模型，但不证明任意 ADS 策略下的反事实正确性。唯一维护的模型与最终结果均位于 `results/hierarchical_traffic_world_model/`。

test 窗口上的风险校准诊断使用本地 HighwayEnv 重跑：IDM 是原生 `IDMVehicle`；HiQR
背景车在同一 `Road`、碰撞与 snapshot 机制中执行与训练一致的单轮
`[acceleration, yaw_rate]` 动力学。1,024 个固定 test 场景上，同一记录控制下 HighwayEnv
与离线积分的背景 ADE/FDE 差仅为 `5.66e-5/1.18e-4 m`；确定性 HiQR 的背景重建 ADE/FDE
为 `0.0356/0.0259 m`。记录人类控制重放失效率为 0.586%（highD 观测为 0.684%）；IDM
重放失效率为 4.297%，碰撞率为 2.832%。该诊断不支持“闭环重建系统性放大真实风险”，亦不
构成任意 ADS 策略反事实正确性的证明。其 5 个 IDM 碰撞播放位于
`IDM_subset/results/risk_calibration_diagnostic/`；这些是固定 highD test 窗口的诊断
播放，不是 AMS subset 最终粒子的碰撞案例；正式 AMS 粒子回放位于
`IDM_subset/results/subset/playbacks/`。

### 三目标实验设计（可复现到高可视化）

- 事实/事件保真（`evaluate_world_model`）：基于完整 10,151 条 highD test 做 `ADE/FDE/P95`，并按 EVT 标签与完整语义 cut-in 分层报告；同时给出 open-loop / no-long-horizon / fixed-history 消融，指标保存在 `evaluation.json`。
- 分布随机性（`evaluate_randomness_ablation`）：固定 1,024 条 split，在每种条件下采 `16` 条样本，报告 `energy score`、`mean pairwise trajectory distance`、`terminal pairwise distance`、速度/加速度与 jerk 的分布一致性（对齐 SceneDiffuser 的多样本指标思路）。
- 干预有效性（`evaluate_world_model`）：三档制动/加速/横移，使用共同随机数，报告方向正确率、剂量单调性、局部性比、时滞、分布 Wasserstein 距离与自然响应覆盖率。

评测 schema 保存 149 帧事实误差、highD-adapted 运动/交互直方图、三档干预的
`0.2/0.4/0.8 s` 剂量响应及完整 near/far 时域曲线。

## 运行

```bash
conda run -n tread python hierarchical_traffic_world_model/scripts/train_world_model.py
conda run -n tread python hierarchical_traffic_world_model/scripts/evaluate_world_model.py
conda run -n tread python hierarchical_traffic_world_model/scripts/evaluate_randomness_ablation.py
conda run -n tread python hierarchical_traffic_world_model/scripts/diagnose_risk_calibration.py
```

### 脚本职责与边界

`scripts/` 只放世界模型训练、评测和接口审计；它不实现 AMS/Subset
Simulation，也不保存 IDM 的正式概率结果。正式的 IDM 闭环策略、pCN/AMS、独立
Monte Carlo 和其结果都归 `IDM_subset/` 管理。

| 脚本 | 用途 | 产物/性质 |
| --- | --- | --- |
| `train_world_model.py` | 按唯一正式配置训练 HiQR 世界模型 | 写入 `results/.../checkpoints/`；会更新模型工件 |
| `evaluate_world_model.py` | 完整 highD test 的事实、随机性和干预评测 | 写 `evaluation.json`；离线评测，不是 HighwayEnv 风险估计 |
| `evaluate_randomness_ablation.py` | 固定 1,024 条测试序列的 16 样本随机性消融 | 更新 `randomness_ablation.json`；“cohort”不是 subset simulation |
| `diagnose_risk_calibration.py` | 在本地 HighwayEnv 中比较 highD 记录控制、HiQR 背景和 IDM 反事实 | 写 `risk_calibration_diagnostic.json`；碰撞播放输出到 `IDM_subset/results/risk_calibration_diagnostic/`，不属于 AMS 案例 |
| `evaluate_history_response.py` | 审计已实现历史是否改变当前响应 | 写 `final/history_response_sensitivity.json`；诊断，不是训练 gate |
| `evaluate_ams_readiness.py` | 检查完整世界随机性、快照、CRN、EVT 接口是否可供外部 AMS 调用 | 写 `final/ams_readiness.json`；只做接口 readiness，不运行 AMS |
| `assess_final_acceptance.py` | 汇总正式工件的最终门槛 | 写 `final/acceptance.json`；失败即返回非零状态 |
| `promote_final_checkpoint.py` | 一次性冻结已验证配置并做等价性探针 | 只迁移 checkpoint，不重新训练；已有正式工件时不要重复运行 |

核心模块按数据流分层：

| 模块 | 责任 |
| --- | --- |
| `src/config.py` | 唯一 25 Hz、25 帧历史和六背景槽位的模型契约 |
| `src/data.py`、`src/calibration.py` | highD recording split、响应样本和自然响应校准 |
| `src/planner.py`、`src/composition.py`、`src/world_randomness.py` | Flow → Diffusion → HiQR 场景组合与显式随机性 |
| `src/model.py`、`src/stochastic.py`、`src/reference.py` | HiQR 响应、随机潜变量和 soft-plan 物理特征 |
| `src/environment.py` | 离线可微闭环，仅供训练/单测 |
| `src/highway_env_execution.py`、`src/world_execution.py` | 正式 HighwayEnv 执行、快照、ADS 和 EVT 风险接口 |
| `src/train.py`、`src/losses.py`、`src/evaluation.py` | 训练目标、离线评估和结果汇总 |
| `src/visualization.py` | 评估图表和重建播放，不参与模型推理 |

`src/environment.py` 中的 `ClosedLoopWorld` 是可微离线训练/单元测试环境，保留是为了
模型损失和因果时序契约测试；任何 ADS 风险、IDM 结果或最终 AMS 输入都必须经过
`src/highway_env_execution.py` 的 HighwayEnv backend。`src/world_execution.py` 只负责把
完整世界 rollout 和 EVT 风险接口交给上层 runner，不包含 subset 算法。

推荐执行顺序为：训练（如需）→ `evaluate_world_model.py` →
`evaluate_randomness_ablation.py` → `evaluate_ams_readiness.py` →
`assess_final_acceptance.py`。`diagnose_risk_calibration.py` 和
`evaluate_history_response.py` 是独立诊断，可按需运行。

训练 checkpoint、校准记录、最终全量评价与门槛审计均位于 `results/hierarchical_traffic_world_model/`。正式工件为 `checkpoints/final_world_model.pt`，其完整配置、三层 checkpoint 哈希与数据划分记录在 `checkpoints/final_model_manifest.json`；正式评估不再覆盖 checkpoint 配置。

结果目录只保留一层正式报告和一个 `final/` 门槛包：`evaluation.json`、
`randomness_ablation.json`、`natural_response_calibration.json`、
`risk_calibration_diagnostic.json`、`training_summary.json` 和
`training_history.json` 是主报告；探索性的干预扫描统一收纳在
`intervention_sweeps.json`，不参与 acceptance gate。旧的多级
`stochastic_causal_hiqr_full/`、散落 figures/playbacks 和重复 intervention JSON 已清理。

## AMS 接口

`WorldExogenousState` 显式保存 `scenario_uniform`、Flow 的 C0/K 高斯基变量、Diffusion 初始噪声与逐帧 HiQR innovation。它可保存为 NPZ，且支持对各高斯 block 作 prior-preserving pCN mutation。正式调用链为：

```python
exogenous = sampler.sample_world_exogenous(1, seed=17)
result = rollout_world(sampler, exogenous, ads_policy, evt_model=evt_model)
```

相同 `exogenous` 与 ADS 必须逐点重放；只替换 ADS 即得到 common-random-number 分支。完整接口与数值、风险、重放审计可运行：

```bash
conda run -n tread python hierarchical_traffic_world_model/scripts/promote_final_checkpoint.py
conda run -n tread python hierarchical_traffic_world_model/scripts/evaluate_history_response.py
conda run -n tread python hierarchical_traffic_world_model/scripts/evaluate_ams_readiness.py --worlds 4 --steps 64
conda run -n tread python hierarchical_traffic_world_model/scripts/assess_final_acceptance.py
```

`evaluate_history_response.py` 是诊断而非训练 gate：当前冻结权重若未显示可观测的 history sensitivity，应约束论文表述，而不是仅为通过该诊断扩大模型或噪声。

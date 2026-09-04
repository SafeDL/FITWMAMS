# 扩散引导 HiQR 分层交通世界模型

## 目录约定

```text
hierarchical_world_model/
├── config/release.yaml   # 唯一正式配置
├── src/                   # 模型、组合、执行与协议实现
├── scripts/               # 训练、评测、发布入口
└── tests/                 # 契约与回放测试
```

正式输出统一写入 `results/hierarchical_world_model/`；`IDM_subset/` 保留为独立的
HighwayEnv/AMS 外部评测层，不再复制世界模型代码。

本模块实现统一的三级接口：

\[
p(M)p(C_0\mid M)p(K\mid C_0,M)
\rightarrow p_\theta(\tau_{\rm soft}\mid C_0,M,K,z_{\rm diff})
\rightarrow \pi_\phi(a_{\rm bg}\mid H_t,\tau_{\rm soft},z_{\rm HiQR}).
\]

Flow 与冻结扩散共享 `C0(40)+M(6)+K(72)=118` 维物理契约。`K` 是三时刻、六背景槽位的状态结点，不再拆成额外行为符号。扩散生成六车联合 soft plan；HiQR 每 0.04 s 根据已实现历史重对齐预览，并只提交下一帧动作。EVT 只提供外部人类风险标尺，不进入模型损失。

短程层复用 HiQR 的历史—关系查询编码器、车道几何、只更新已观测状态的递归滤波器、1 s 场景潜变量、连续相关的车辆潜变量和联合 jerk-knot 解码。首次调用在 `filter_state` 不存在时初始化观测状态，之后每 0.04 s 只用已实现联合状态递推更新；历史编码器的实际敏感性由单独诊断审计，不将其作用强度作为未经验证的主张。soft-plan 位置、速度、参考控制与当前偏差作为 preview token 同时进入 prior 和 decoder；解码器预测短计划但只提交第一帧，最终控制满足 `j_bg=j_soft+Δj_HiQR`，不存在输出端 gate 或硬轨迹回放。

`WorldExogenousState` 使用 `world_rng` 的 named block streams：scenario、C0、K、Diffusion、scene 和 agent。scene innovation 只有 `[N,ceil(T/25),16]`（正式 149 steps 为 6 次 refresh），agent innovation 为 `[N,T,7,16]`；响应 horizon 唯一由 agent 数组长度派生。运行时只保存一个 response index，scene 的 `t//25` 索引由其确定性计算。训练和评测分别使用 `training_rng`、`evaluation_rng` namespace。snapshot/restore 保存滤波状态、慢潜变量和该单一响应索引。模型从不接收未来 ego 动作，干预只能从下一次重规划开始生效。

## 正式发布状态

正式工件只能由带 release tag 的干净工作树生成，并且须完成
base → stochastic heads → final → evaluation 全链路。唯一维护的模型与结果位于
`results/hierarchical_world_model/`；风险数值描述测试结果，不代表任意 ADS 策略下的反事实正确性。

### 三目标实验设计（可复现到高可视化）

- 事实/事件保真（`evaluate.py`）：基于完整 highD test 做 `ADE/FDE/P95`，并按 EVT 标签与完整语义 cut-in 分层报告；同时给出 open-loop / no-long-horizon / fixed-history 消融，指标保存在 `evaluation.json`。
- 分布随机性（`randomness_eval.py`）：固定测试 split，在每种条件下采样多条轨迹，报告 `energy score`、轨迹距离与运动学分布一致性。
- 干预有效性（`evaluate.py`）：三档制动/加速/横移，使用共同随机数，报告方向正确率、剂量单调性、局部性比、时滞、分布 Wasserstein 距离与自然响应覆盖率。

评测 schema 保存 149 帧事实误差、highD-adapted 运动/交互直方图、三档干预的
`0.2/0.4/0.8 s` 剂量响应及完整 near/far 时域曲线。

## 正式运行

```bash
git tag <release-tag>
# 从干净 tag 一次性执行正式链路（会写入 release session）
conda run -n tread python hierarchical_world_model/scripts/release.py --release-tag <release-tag>
```

### 脚本职责与边界

`scripts/` 只放世界模型训练、评测和接口审计；它不实现 AMS/Subset
Simulation，也不保存 IDM 的正式概率结果。正式的 IDM 闭环策略、pCN/AMS、独立
Monte Carlo 和其结果都归 `IDM_subset/` 管理。

| 脚本 | 用途 | 产物/性质 |
| --- | --- | --- |
| `train.py` | 运行 `base` 或 `stochastic_heads` stage | 写入明确的 stage checkpoint；两者架构配置完全一致 |
| `sampled_eval.py` | 1,024 sampled Flow→Diffusion→HiQR→HighwayEnv 世界 | 写 `sampled_end_to_end.json`；含 K-adherence、非配对 BG fidelity 与 paired ADS 风险 |
| `evaluate.py` | 完整 highD test 的事实、随机性和干预评测 | 写 `evaluation.json`；离线评测，不是 HighwayEnv 风险估计 |
| `randomness_eval.py` | 固定 1,024 条测试序列的 16 样本随机性消融 | 更新 `randomness_ablation.json`；“cohort”不是 subset simulation |
| `risk_calibration.py` | 在本地 HighwayEnv 中比较 highD 记录控制、HiQR 背景和 IDM 反事实 | 写 `risk_calibration_diagnostic.json`；碰撞播放输出到 `IDM_subset/results/risk_calibration_diagnostic/`，不属于 AMS 案例 |
| `history_eval.py` | 审计已实现历史是否改变当前响应 | 写 `final/history_response_sensitivity.json`；诊断，不是训练 gate |
| `ams_readiness.py` | 检查完整世界随机性、快照、CRN、EVT 接口是否可供外部 AMS 调用 | 写 `final/ams_readiness.json`；只做接口 readiness，不运行 AMS |
| `acceptance.py` | 汇总正式工件的最终门槛 | 写 `final/acceptance.json`；失败即返回非零状态 |
| `promote.py` | 一次性冻结已验证配置并做等价性探针 | 只迁移 checkpoint，不重新训练；已有正式工件时不要重复运行 |

核心模块按数据流分层：

| 模块 | 责任 |
| --- | --- |
| `src/config.py` | 唯一 25 Hz、25 帧历史和六背景槽位的模型契约 |
| `src/data.py`、`src/calibration.py` | highD recording split、响应样本和自然响应校准 |
| `src/planner.py`、`src/composition.py`、`src/randomness.py` | Flow → Diffusion → HiQR 场景组合与显式随机性 |
| `src/model.py`、`src/stochastic.py`、`src/reference.py` | HiQR 响应、随机潜变量和 soft-plan 物理特征 |
| `src/environment.py` | 离线可微闭环，仅供训练/单测 |
| `src/highway.py`、`src/execution.py` | 正式 HighwayEnv 执行、快照、ADS 和 EVT 风险接口 |
| `src/train.py`、`src/losses.py`、`src/evaluation.py` | 训练目标、离线评估和结果汇总 |
| `src/visualization.py` | 评估图表和重建播放，不参与模型推理 |

`src/environment.py` 中的 `ClosedLoopWorld` 是可微离线训练/单元测试环境，保留是为了
模型损失和因果时序契约测试；任何 ADS 风险、IDM 结果或最终 AMS 输入都必须经过
`src/highway.py` 的 HighwayEnv backend。`src/execution.py` 只负责把
完整世界 rollout 和 EVT 风险接口交给上层 runner，不包含 subset 算法。

推荐执行顺序为：训练（如需）→ `evaluate.py` →
`randomness_eval.py` → `ams_readiness.py` →
`acceptance.py`。`risk_calibration.py` 和
`history_eval.py` 是独立诊断，可按需运行。

训练 checkpoint、校准记录、最终全量评价与门槛审计均位于 `results/hierarchical_world_model/`。正式工件为 `world_model/checkpoints/final_world_model.pt`，其完整配置、三层 checkpoint 哈希与数据划分记录在 `world_model/checkpoints/final_model_manifest.json`；正式评估不再覆盖 checkpoint 配置。

结果目录只保留可追溯工件，不持久化可再生的训练 preview cache：

```text
results/hierarchical_world_model/
├── world_model/                   # 主干训练、checkpoint、正式评估、安全审计
│   ├── checkpoints/
│   ├── training/
│   ├── evaluation/
│   ├── causal_diagnostics/
│   └── safety/
└── causal_reaction/               # NPC 因果反应修正项的独立实验
    ├── formal/                     # 冻结 IDM/MOBIL 与已验证 PPO 基线
    └── candidates/                 # 当前动态 PPO、V4 GAIL 与 A3 拒绝证据
```

`train.py` 的 Diffusion preview 计划仅在训练进程的临时目录中保存，退出后自动删除；
其余缓存若服务于独立诊断，必须与对应报告同目录保存。训练、评测、随机性、sampled E2E 与
acceptance 工件均写入上述目录；探索性输出不参与 acceptance gate。

## AMS 接口

`WorldExogenousState` 显式保存 `scenario_uniform`、Flow 的 C0/K 高斯基变量、Diffusion 初始噪声、压缩 scene innovation 与逐 response agent innovation。它可保存为 NPZ，且支持对各高斯 block 作 prior-preserving pCN mutation。正式调用链为：

```python
exogenous = sampler.sample_world_exogenous(1, seed=17)
result = rollout_world(sampler, exogenous, ads_policy, evt_model=evt_model)
```

相同 `exogenous` 与 ADS 必须逐点重放；只替换 ADS 即得到 common-random-number 分支。完整接口与数值、风险、重放审计可运行：

```bash
conda run -n tread python hierarchical_world_model/scripts/promote.py
conda run -n tread python hierarchical_world_model/scripts/history_eval.py
conda run -n tread python hierarchical_world_model/scripts/ams_readiness.py --worlds 4 --steps 64
conda run -n tread python hierarchical_world_model/scripts/acceptance.py
```

`history_eval.py` 是诊断而非训练 gate：当前冻结权重若未显示可观测的 history sensitivity，应约束论文表述，而不是仅为通过该诊断扩大模型或噪声。

## IDM/MOBIL 与 GAIL 自然性扩展

`reaction_naturalistic.yaml` 是唯一维护的四臂消融配置。PPO 的语义是“动态因果后车反应残差”
而非通用驾驶策略：只有已执行的 ADS 制动写入 HighwayEnv 后，动态影响图认定的直接或一跳
次级候选车辆才可输出 `α` 与纵向残差；无控制权的槽位逐元素保留 HiQR 基础动作。图只读取
已实现状态和已提交控制，风险解除并稳定 0.5 s 后以 recovery envelope 释放。先从 training
split 标定规则与人类先验，再训练新增策略：

```bash
conda run -n tread python hierarchical_world_model/scripts/calibrate_reaction_rules.py
conda run -n tread python hierarchical_world_model/scripts/evaluate_reaction_rules.py --split validation
conda run -n tread python hierarchical_world_model/scripts/evaluate_human_driving_prior.py --split validation
conda run -n tread python hierarchical_world_model/scripts/visualize_human_prior_evidence.py
conda run -n tread python hierarchical_world_model/scripts/visualize_a2_a3_fast_evidence.py \
  --artifact-dir results/hierarchical_world_model/causal_reaction/candidates/a3_v4_balanced
conda run -n tread python hierarchical_world_model/scripts/render_reaction_ppo_comparative_playbacks.py \
  --artifact-dir results/hierarchical_world_model/causal_reaction/candidates/ppo_v7_final --row 3456
```

训练新的 prior 或 PPO controller 时必须显式使用 `--output` 或 `--output-dir` 指向新的
candidate 目录，不能覆盖 `formal/`、`ppo_v7_final/`、`gail_v4_temporal/` 或
`a3_v4_balanced/`。完整的保留工件和生命周期说明见
`results/hierarchical_world_model/causal_reaction/README.md`。

消融固定为 `A0_none`、已有纯 `A1_rl_residual`、IDM-referenced
`A2_rl_residual_idm`、以及 GAIL-constrained `A3_rl_residual_gail`。MOBIL 在本轮
只作为从 highD 换道片段标定出的诊断规则；它不输出 yaw-rate，也不修改纵向 controller。
V4 GAIL prior 使用两层128单元的单一 bounded Gaussian actor、独立 critic 和逐 tick
轻量 MLP discriminator。它先经概率行为克隆初始化，再以“prior 控制目标 NPC、其余
车辆保持冻结 HiQR”的连续2秒 HighwayEnv 闭环进行完整 GAE/clipped-PPO 细化；expert
与生成轨迹按同一 highD row/父子关系配对，避免 discriminator 利用 role/TTC 样本构成
区分标签。A3 的自然性奖励对**最终完整纵向动作分布**计算前向 KL，而非对 PPO 残差
计算；V4 对 gate 与 residual 联合数值边缘化，并将安全/jerk 可执行边界纳入同一密度，
不再执行确定性 human-action projection。评估中，动作修正剂量定义为同一 tick 的
`HighwayEnv 执行动作 − HiQR 基础动作`，而不是与已发生状态分叉的 A0 未来动作相减。

最新快速修复工件位于 `causal_reaction/candidates/gail_v4_temporal/` 与
`causal_reaction/candidates/a3_v4_balanced/`。V4 prior 在完整 highD train split 上进行
连续两秒 HighwayEnv GAIL refinement；A3 完成全量动态训练和响应受限 fine-tune。在完整
5,095条动态 test 上，A3 的 0.6/1.0秒 KL 改善约33%/24%，响应保持 A2 的约92%，但
0.2秒 KL 与 jerk W1 未全面改善，因此自动验收仍选择 A2，不覆盖正式 controller 或 GIF。
机器可读结论见 `a3_v4_balanced/evidence/a2_a3_v4_acceptance.json`。

每次四臂评估还会保存逐 tick 的 `counterfactual_telemetry_<split>.npz`：最终/基础动作、
残差、门控、IDM 动作、GAIL KL、gap、closing speed、TTC 与 jerk。随后运行
`visualize_reaction_naturalistic_evidence.py` 会在 `evidence/<split>/` 生成四类证据图：
PPO/GAIL 训练曲线、IDM/MOBIL 标定参数、剂量—响应/时延/局部性/碰撞曲线，以及在
相同 TTC 条件下与 held-out highD 人类动作和 jerk 分布的比较；相应的 Wasserstein-1 与
KS 指标写入 `conditional_distribution_metrics.json`。这些图是实验诊断和论文证据，不接入
正式 release acceptance 或 AMS 概率 gate。

为避免把极端反事实与普通跟驰动作作无条件比较，`evaluate_human_driving_prior.py` 还会
从 held-out highD 挖掘“前车已发生制动 → 同车道后车随后 0.4 s 响应”的参考样本；
`observed_brake_distribution_metrics.json` 只在人类与生成样本均不少于 128 时给出 W1/KS，
否则显式标为 `insufficient_matched_samples`，不得用于自然性结论。

# RAMP-WM：关系记忆驱动的联合多假设滚动自回归交通世界模型目标文档

训练 GPU 环境：

```bash
conda activate tread
```

---

## 0. 方法决策

本轮不再将 Semi-Markov 离散状态、行为阶段边界和持续时间模型作为新世界模型的核心组成。

现有训练结果只能证明：关系图编码、首秒行为锚定、控制计划、动力学积分和闭环自回归路径具有实际价值；不能证明离散 Semi-Markov 状态已经学习到稳定、可解释且持续的交通交互状态。因此，新模型应独立设计，并将现有 Semi-Markov 与 CAT-TopK 作为冻结比较对象。

新模型命名为：

> **RAMP-WM：Relational Autoregressive Memory-and-Planning World Model**

中文名称：

> **关系记忆驱动的联合多假设滚动自回归交通世界模型**

核心运行过程为：

```text
过去一秒交通历史
→ 动态关系图编码
→ 更新连续交通记忆
→ 生成未来一秒场景级联合多假设计划
→ 按显式世界随机数选择一个联合计划
→ 只执行前 0.2 秒
→ 写回生成背景车状态
→ 读取已经发生的自车状态并重新规划
```

RAMP-WM 不识别心理意义上的驾驶意图，不预测稀疏行为边界，也不学习显式持续时间。跨时刻行为持续性由连续交通记忆、上一计划剩余部分和重叠计划一致性共同维持。

---

## 1. 研究目标与核心假设

### 1.1 研究目标

构建一个能够同时满足以下要求的随机闭环交通世界模型：

1. 在相同 highD 测试序列上，确定性 5 秒轨迹重建性能超过现有 Semi-Markov 和 CAT-TopK；
2. 在 EVT-tail 测试子集上保持或改善 1–5 秒 ADE、FDE、gap MAE 和交互安全量；
3. 在相同初始条件下生成多个物理合理、相互区分且概率可解释的联合交通未来；
4. 随机生成轨迹在运动统计、交互关系、时序依赖和长尾风险特征上与自然驾驶长尾分布一致；
5. 支持固定外生随机数下的确定性重放、快照恢复和不同 ADS 之间的公平配对测试。

### 1.2 核心假设

RAMP-WM 建立在以下假设上：

- highD 的短时交通演化具有连续的动力学记忆，但不需要人为划分为离散意图阶段；
- 单一确定性计划容易产生均值回归，多假设联合计划更适合表达同一交通条件下的多种合理未来；
- 多车随机性必须在场景级联合采样，不能为每辆背景车独立选择行为分支；
- 每 0.2 秒重新规划未来 1 秒，并约束相邻计划的 0.8 秒重叠区域，可同时保持反应性和时间连续性；
- 世界模型随机变量必须与 ADS 身份无关、可显式保存并可在不同 ADS 下重放。

---

## 2. 当前证据与设计动机

### 2.1 Semi-Markov 结果不支持“持续意图识别”

现有 Semi-Markov 结果存在以下问题：

- 后验边界频率显著高于自然驾驶行为变化代理率；
- 平均状态持续时间接近 1–2 个响应周期；
- 状态在 5 秒轨迹中频繁切换；
- 状态区分更多反映速度区间，而非稳定的制动、压缩、恢复等交互阶段；
- 5 秒闭环误差仍明显高于 CAT-TopK。

因此，本轮不再继续通过增加状态数量、调整边界类别权重、延长训练 epoch 或改变 teacher forcing 来维持 Semi-Markov 主路径。

### 2.2 CAT-TopK 证明了多假设计划的价值，但仍有改进空间

CAT-TopK 已验证以下设计具有价值：

- 场景级候选编号同时控制全部背景车；
- 多候选责任分配与概率学习；
- 未来一秒平滑动作计划；
- 基于 jerk 控制点的时间曲线；
- 多段闭环训练。

其主要限制为：

- 以 1 秒分块为主，缺少每 0.2 秒重规划时的 0.8 秒重叠约束；
- 没有显式连续场景记忆来维持跨窗口行为惯性；
- 候选计划主要在关系编码后逐车解码，未来计划之间的显式联合协调仍有限；
- 当前 START 信息条件与 Semi-Markov 不完全对称，比较时必须披露。

### 2.3 新模型定位

RAMP-WM 不是 Semi-Markov 的局部补丁，也不是 CAT-TopK 的重命名版本，而是：

> 在动态关系编码和连续交通记忆基础上，对未来一秒的多车联合计划进行概率化生成，并以 0.2 秒执行前缀进行闭环自回归更新。
> 针对确认有效的模块可以参考已有世界模型代码

---

## 3. 固定边界

以下条件在所有训练、验证、测试和基线比较中保持不变：

- 冻结 76 维 Normalizing Flow、checkpoint、schema 和标准化规则；
- START 可使用冻结 Flow 提供的首秒行为锚点；
- 第一秒结束后，任何模块不得继续读取 B0；
- ROLL 只读取：
  - 已发生的自车状态；
  - 模型已生成的背景车历史；
  - 当前道路图和交通关系；
  - 模型内部连续记忆；
  - 上一轮已生成计划及其剩余部分；
- 禁止输入 ego future、未来背景车、ADS 身份、ADS 网络特征、风险标签、EVT 标签或 `risk_trace`；
- 保持 25 Hz 物理更新、0.2 秒响应间隔、六背景车固定槽位和现有车辆动力学；
- 不引入车辆出生、消失或槽位重分配；
- 不加载 Semi-Markov 或 CAT-TopK checkpoint 初始化 RAMP-WM；
- 新模型从随机初始化训练；
- Semi-Markov 和 CAT-TopK 作为冻结基线，不修改其 checkpoint 和正式测试结果。

---

## 4. 直接复用与禁止复用

### 4.1 直接复用的共享基础模块

允许直接复用以下基础代码和数据接口：

```text
world_model/src/relational_encoder.py
world_model/src/dynamics.py
world_model/src/initial_behavior_anchor.py
world_model/src/graph_schema.py
world_model/src/graph_builder.py
world_model/src/sequential_dataset.py
world_model/src/metrics.py
world_model/src/utils.py
```

复用内容包括：

- 动态交通关系图；
- 车辆和场景关系表示；
- 地图编码；
- 冻结 Flow schema 和行为锚点缓存；
- 车辆运动学积分；
- highD 顺序缓存与固定数据划分；
- 物理有效性和交互指标；
- 随机数、日志和结果保存工具。

### 4.2 从 Semi-Markov 路线借鉴的有效机制

只借鉴已经通过代码和实验验证的机制：

- 每 0.2 秒读取当前已发生的自车状态；
- 将生成背景车状态写回历史；
- 未来 1 秒计划、执行前 0.2 秒；
- 相邻计划 0.8 秒重叠；
- START 与 ROLL 的信息边界；
- B0 在第一秒后不可达；
- `snapshot()` / `restore()`；
- 外生随机数记录与确定性重放；
- 完整计划审计输出；
- 同序列配对 bootstrap 比较。

不复用：

```text
posterior_z
posterior_boundary
latent prior
duration hazard
boundary_target
state_bootstrap_target
latent KL
duration NLL
censor NLL
```

### 4.3 从 CAT-TopK 借鉴的有效设计

采用清洁重实现，不直接导入 CAT-TopK 模型类或 checkpoint：

- 场景级联合候选编号；
- 候选概率和软责任分配；
- jerk 控制点生成平滑动作曲线；
- mixture / energy / diversity 训练思想；
- 候选 0 作为名义计划；
- 显式离散世界随机变量选择候选。

RAMP-WM 必须具有独立的模型类、训练入口、环境和 checkpoint。

---

## 5. 总体模型结构

在响应时刻 \(t\)，输入为历史 \(H_t\)、当前交通状态 \(S_t\)、道路图 \(G_t\)、连续记忆 \(h_{t-1}\) 和上一轮计划 \(U_{t-\Delta}\)。

### 5.1 关系编码

```text
agent_context_t, scene_context_t
    = RelationalEncoder(H_t, S_t, G_t)
```

其中：

```text
agent_context_t  [B, agents, hidden]
scene_context_t  [B, hidden]
```

### 5.2 连续交通记忆

连续记忆由场景关系、上一轮计划、上一执行结果和自车已发生变化共同更新：

\[
h_t =
\operatorname{GRU}\left(
h_{t-1},
[
c_t,\,
\operatorname{Pool}(a_t),\,
\operatorname{Summary}(U_{t-\Delta}^{\mathrm{remain}}),\,
\Delta s_t^{ego},\,
\Delta s_t^{bg}
]
\right).
\]

其中：

- \(c_t\)：当前场景级关系表示；
- \(a_t\)：逐车关系表示；
- \(U_{t-\Delta}^{\mathrm{remain}}\)：上一轮未执行的计划部分；
- \(\Delta s_t^{ego}\)：最近 0.2 秒已经发生的自车变化；
- \(\Delta s_t^{bg}\)：最近 0.2 秒生成的背景车变化；
- \(h_t\)：连续交通记忆。

要求：

- 记忆在每个 0.2 秒响应点更新；
- 不定义离散边界；
- 不定义持续时间；
- 记忆更新不得读取 future ego 或未来背景车；
- `snapshot()` / `restore()` 必须保存并恢复 \(h_t\)。

---

## 6. 场景级联合多假设计划

### 6.1 输出定义

模型每次输出 \(M=8\) 个未来一秒场景级联合计划：

\[
\left\{
U_t^{(m)}
\right\}_{m=0}^{7},
\qquad
U_t^{(m)}
\in
\mathbb R^{B\times25\times6\times2},
\]

以及场景级候选概率：

\[
\pi_t
=
\operatorname{softmax}
\left(
f_\pi(c_t,h_t,U_{t-\Delta}^{\mathrm{remain}},m_{t-1})
\right).
\]

一个候选编号同时确定全部六辆背景车的未来计划。禁止逐车独立采样候选。

### 6.2 名义计划

候选 0 为确定性名义计划：

\[
U_t^{(0)}
=
f_{\mathrm{nominal}}(a_t,c_t,h_t).
\]

START：

\[
U_t^{(0)}
=
U_t^{B0}
+
f_{\mathrm{nominal}}(a_t,c_t,h_t).
\]

ROLL：

\[
U_t^{(0)}
=
f_{\mathrm{nominal}}(a_t,c_t,h_t).
\]

候选 0 用于：

- 确定性评测；
- checkpoint 选择；
- 候选退化时的安全回退；
- 与单计划模型的参数受控消融。

### 6.3 残差候选

候选 1–7 在名义计划上加入有界联合 jerk 残差：

\[
U_t^{(m)}
=
U_t^{(0)}
+
\Delta U_t^{(m)},
\qquad m=1,\ldots,7.
\]

每个候选只预测 5 个 jerk 控制点：

```text
jerk_controls [B, 7, 5, 6, 2]
```

通过固定线性样条展开为 25 帧，并积分为动作残差。

约束：

- 纵向和横向 jerk 具有硬边界；
- 候选 0 不增加残差；
- 残差输出层零初始化；
- 候选残差从执行前缀第 2 帧开始渐进生效；
- 执行前 5 帧使用固定渐进系数：

```text
[0.00, 0.25, 0.50, 0.75, 1.00]
```

该设计确保候选真正改变闭环轨迹，同时避免第一个物理帧发生动作突跳。

---

## 7. 计划层多车联合交互

关系编码器只保证当前状态层面的联合编码。RAMP-WM 还需要在未来计划空间中进行一次显式多车协调。

对每个候选 \(m\) 和每个 jerk 控制点 \(c\)，构造逐车计划 token，并执行一层关系条件的跨车辆注意力：

\[
\widetilde J_{i,c}^{(m)}
=
J_{i,c}^{(m)}
+
\sum_{j\neq i}
\alpha_{ij,c}^{(m)}
V(J_{j,c}^{(m)}),
\]

\[
\alpha_{ij,c}^{(m)}
=
\operatorname{softmax}_{j}
\left(
Q_i^\top K_j
+
b(r_{ij})
\right).
\]

关系特征至少包括：

```text
relative_x
relative_y
relative_vx
relative_vy
gap
closing_speed
TTC
DRAC
same_lane / adjacent_lane
```

要求：

- 只使用当前可观测关系；
- 无效车辆严格掩码；
- 所有候选共享注意力参数；
- 注意力输出层零初始化；
- 注意力后再执行 jerk 有界化和时间积分；
- 不增加新的随机变量。

---

## 8. 滚动时域与重叠计划

每 0.2 秒预测未来 1 秒，只执行前 5 帧：

```text
selected_plan       [B, 25, 6, 2]
applied_controls    = selected_plan[:, 0:5]
remaining_plan      = selected_plan[:, 5:25]
```

下一轮计划与上一轮计划的重叠区域为：

```text
old_selected_plan[:, 5:25]
new_candidate_plan[:, 0:20]
```

重叠一致性损失：

\[
L_{\mathrm{overlap}}
=
w_t
\left\|
U_{t-\Delta}^{\mathrm{selected}}[5:25]
-
U_t^{\mathrm{selected}}[0:20]
\right\|_1.
\]

权重 \(w_t\) 只依赖已经观察到的变化：

```text
current relation change
observed ego change
generated background change
previous candidate index
```

关系变化较小时强化连续性；变化明显时允许重新规划。

---

## 9. 自回归闭环与世界随机变量

### 9.1 外生随机变量

RAMP-WM 的世界随机变量定义为：

\[
\Xi_{\mathrm{world}}
=
\left\{
u_t^{plan}
\right\}_{t=0}^{T-1},
\qquad
u_t^{plan}\sim U(0,1).
\]

在每个 0.2 秒响应点，根据候选概率逆 CDF 选择场景级候选：

\[
m_t
=
F_{\pi_t}^{-1}(u_t^{plan}).
\]

确定性评测使用：

\[
m_t=\arg\max_m \pi_t^{(m)}.
\]

### 9.2 自回归过程

```text
预测 8 个一秒联合计划
→ 使用外生 uniform 选择一个候选
→ 执行前 5 帧
→ 通过动力学生成背景车状态
→ 写回 25 帧历史缓存
→ 输入下一时刻已经发生的 ego 状态
→ 更新连续交通记忆
→ 重新预测
```

世界模型不得读取后续 ego 轨迹。

### 9.3 快照与重放

环境必须保存：

```text
agent_states
agent_valid
history_states
history_valid
continuous_memory
previous_selected_plan
previous_candidate_index
previous_candidate_probabilities
previous_relation_summary
plan_uniforms_remaining
rng_state
behavior_anchor_state
response_index
trace
```

恢复后必须逐帧重现：

```text
candidate probabilities
candidate index
controls
background states
continuous memory
trace
```

---

## 10. 正式输入输出

### 10.1 输入

```text
history_states                  [B, 25, 7, 6]
history_valid                   [B, 25, 7]
current_states                  [B, 7, 6]
current_valid                   [B, 7]
map_polylines
map_polyline_valid
lane_graph_edges
continuous_memory               [B, hidden]
previous_selected_plan          [B, 25, 6, 2] 或 None
previous_candidate_index        [B] 或 None
previous_relation_summary
START 时额外输入 B0 raw/std/valid
```

### 10.2 输出

```text
candidate_control_plans         [B, 8, 25, 6, 2]
candidate_probabilities         [B, 8]
selected_candidate_index        [B]
selected_control_plan           [B, 25, 6, 2]
applied_controls                [B, 5, 6, 2]
predicted_candidate_states      [B, 8, 25, 6, 6]
continuous_memory_next          [B, hidden]
overlap_diagnostics
candidate_diversity_diagnostics
```

对外环境每 0.2 秒返回前 5 帧实际背景车状态；完整候选计划和概率作为审计输出。

---

## 11. 损失函数

总损失定义为：

\[
L =
L_{\mathrm{execute}}
+
\lambda_{\mathrm{roll}}L_{\mathrm{roll}}
+
\lambda_{\mathrm{plan}}L_{\mathrm{plan}}
+
\lambda_{\mathrm{mix}}L_{\mathrm{mixture}}
+
\lambda_{\mathrm{prob}}L_{\mathrm{prob}}
+
\lambda_{\mathrm{overlap}}L_{\mathrm{overlap}}
+
\lambda_{\mathrm{joint}}L_{\mathrm{joint}}
+
\lambda_{\mathrm{div}}L_{\mathrm{diversity}}
+
\lambda_{\mathrm{smooth}}L_{\mathrm{smooth}}.
\]

### 11.1 执行前缀损失

对选中计划的前 5 帧积分状态和控制进行监督：

```text
L_execute_state
L_execute_control
```

### 11.2 完整计划损失

对所有候选计算未来一秒物理状态能量：

```text
control error
position / velocity error
ego-background relative-state error
background-background relative-state error
```

真实未来只作为训练标签，不进入解码器输入。

### 11.3 混合候选损失

对候选能量 \(E_m\) 使用：

\[
L_{\mathrm{mixture}}
=
-\tau\log
\sum_m
\pi_m
\exp(-E_m/\tau).
\]

软责任为：

\[
r_m
=
\operatorname{softmax}(-E_m/\tau).
\]

候选概率校准损失：

\[
L_{\mathrm{prob}}
=
-\sum_m r_m\log\pi_m.
\]

### 11.4 状态—关系多样性

候选多样性不得只在控制空间计算。使用候选积分后的：

```text
position
velocity
gap
relative velocity
TTC
DRAC
```

定义候选间距离。

只对具有非忽略责任的候选施加多样性约束，避免无意义扩大尾部候选。

### 11.5 联合物理约束

```text
action bounds
jerk bounds
speed bounds
vehicle overlap
negative body clearance
cross-agent plan consistency
```

这些约束不得使用 EVT 或风险标签。

---

## 12. 从零训练协议

RAMP-WM 不加载 Semi-Markov 或 CAT-TopK checkpoint。

### 12.1 Stage A：确定性名义计划预训练

目标：

- 学习稳定的一秒名义计划；
- 建立连续记忆；
- 保住首秒和执行前缀性能。

设置：

```text
num_candidates = 1
rollout horizon = 1 s
background teacher forcing inside rollout = 0
训练关系编码器、连续记忆和名义计划解码器
```

说明：

- Stage A 使用真实历史作为窗口输入，但一旦开始预测，不允许用真实未来背景车替换生成状态；
- 不使用 scheduled teacher forcing；
- 最多 8 epochs；
- 完整验证集 1 秒 FDE 早停。

### 12.2 Stage B：联合多假设训练

目标：

- 加入 8 个场景级候选；
- 学习候选责任、概率、平滑 jerk 和计划层联合交互。

设置：

```text
rollout curriculum = 1 s → 2 s → 3 s
background teacher forcing = 0
候选 0 从 Stage A 初始化
候选残差和计划注意力零初始化
```

最多 12 epochs。

### 12.3 Stage C：五秒闭环联合微调

目标：

- 优化完整 5 秒自回归；
- 维持首秒和物理质量；
- 校准候选概率与重叠连续性。

设置：

```text
rollout horizon = 3–5 s
完整训练集
完整验证集
低学习率
TBPTT = 5 response steps
```

最多 20 epochs，至少 6 epochs 后允许早停：

```text
selection metric = val deterministic 5 s FDE
patience = 4
min_delta = 1e-4 m
```

---

## 13. 新代码布局

新方法必须使用独立目录，避免继续修改 Semi-Markov 主文件：

```text
world_model/src/ramp/
    __init__.py
    config.py
    memory.py
    joint_plan_decoder.py
    plan_attention.py
    model.py
    environment.py
    losses.py
    train.py
    evaluation.py
    distribution_evaluation.py

world_model/scripts/
    train_ramp_world_model.py
    test_ramp_world_model.py
    compare_ramp_baselines.py
    evaluate_ramp_distribution.py

world_model/scripts/configs/
    highd_ramp_world_model.yaml

world_model/tests/
    test_ramp_shapes.py
    test_ramp_no_future_leak.py
    test_ramp_autoregressive_history.py
    test_ramp_candidate_jointness.py
    test_ramp_candidate_effect.py
    test_ramp_plan_overlap.py
    test_ramp_snapshot_restore.py
    test_ramp_randomness_replay.py
    test_ramp_b0_lifecycle.py
```

结果目录：

```text
results/highd_world_model/ramp_world_model/
```

禁止从新模型代码导入：

```text
semi_markov_state.py
CATKResidualDynamics
NominalCATKDecoder
旧模型 checkpoint
```

---

## 15. 公平比较协议

比较对象：

| 方法 | 角色 |
|---|---|
| 当前 Semi-Markov | 冻结离散持续状态基线 |
| 当前 CAT-TopK | 冻结多假设计划基线 |
| RAMP-WM | 新方法 |

所有方法必须使用：

```text
相同 highD train / val / test split
相同 24,216 条 test sequences
相同 EVT-tail 标记
相同 ego replay
相同物理坐标系
相同有效车辆掩码
相同 1 s / 5 s 评价代码
2,000 次 paired bootstrap
```

START 信息条件必须单独记录。若 CAT-TopK 使用未来首秒动作摘要，应在主表中显式标记，不得将其视为信息完全对称的比较。

确定性比较采用每个方法自己的正式 MAP / nominal 路径；随机分布比较采用各方法自身的正式随机变量和相同数量的随机样本。

---

## 16. 确定性评测指标

完整 test split 和 EVT-tail 子集必须报告：

```text
ADE_1s ... ADE_5s
FDE_1s ... FDE_5s
gap MAE
relative-vx MAE
velocity MAE
acceleration MAE
TTC error
DRAC error
small-TTC rate
risk-relaxation rate
```

计划诊断：

```text
0.2 s execution-prefix control error
1 s full-plan control error
1 s plan-state position / velocity error
pairwise relative-state error
overlap-plan discontinuity
candidate-switch rate
memory update norm
```

物理指标：

```text
invalid rate
vehicle-overlap rate
negative-gap rate
acceleration-out-of-range rate
jerk-out-of-range rate
speed-out-of-range rate
```

统计检验：

```text
RAMP minus baseline point estimate
one-sided 95% paired-bootstrap upper bound
```

---

## 17. 随机长尾分布评测

在完整 EVT-tail test 子集上，每个条件生成 32 条轨迹。

### 17.1 单条件覆盖与校准

```text
50%、80%、90%、95% empirical coverage
rank histogram / PIT
Energy Score
CRPS 或多变量等价指标
minADE_32
minFDE_32
candidate probability ECE
candidate responsibility cross-entropy
```

### 17.2 真实—生成两样本检验

比较自然驾驶尾部与生成轨迹的：

```text
initial / final speed
speed change
mean / minimum / final acceleration
braking start and duration
longitudinal / lateral displacement
minimum gap
minimum TTC
maximum DRAC
```

一维指标：

```text
Wasserstein-1
KS distance
quantile error
```

联合指标：

```text
Energy Distance
MMD
Pearson correlation-matrix error
Spearman correlation-matrix error
```

### 17.3 时序与多车依赖

```text
acceleration autocorrelation
jerk spectrum / temporal smoothness
front-vehicle braking to rear-vehicle response delay
cross-agent acceleration correlation
gap-relative-speed joint distribution
relationship-state occupancy and transition
```

### 17.4 自然数据基准线

将真实 EVT-tail 样本随机分为两组，计算真实—真实距离，并通过 bootstrap 得到 95%区间。

RAMP-WM 的生成—真实距离应：

- 落入真实—真实抽样波动区间；或
- 至少显著优于 Semi-Markov 和 CAT-TopK 的生成—真实距离。

---

## 18. 主要晋升目标

RAMP-WM 只有同时满足以下条件才被认定为超过两个冻结基线。

### 18.1 相对 Semi-Markov

- 1 秒 ADE/FDE/gap 非劣；
- 5 秒 ADE/FDE/gap 全部改善；
- EVT-tail 5 秒 FDE 和 gap MAE 改善；
- 一侧 95% paired-bootstrap 上界小于 0；
- 物理指标不劣化。

### 18.2 相对 CAT-TopK

主要目标：

- 5 秒 FDE 的 paired difference 小于 0；
- 一侧 95% paired-bootstrap 上界小于 0。

同时要求：

- 1 秒性能非劣；
- EVT-tail 5 秒主指标至少非劣；
- 关系分布漂移和物理无效率不高于 CAT-TopK；
- 不依赖比 CAT-TopK 更多的未来信息。

---

最终论文中的方法表述建议为：

> 本文提出一种关系记忆驱动的联合多假设滚动自回归交通世界模型。该模型利用连续交通记忆维持跨窗口交互惯性，在每个闭环响应时刻生成未来一秒的场景级联合计划分布，仅执行其短时前缀，并基于新的自车观测和生成背景状态持续重规划。与依赖离散行为边界和持续时间的半马尔可夫模型不同，所提方法无需稀疏边界代理监督；与分块多候选模型相比，其通过重叠滚动视野、计划层多车交互和显式可重放世界随机性实现更细粒度的闭环交通演化。

# FITWMAMS：保留 A2 有效能力的人类响应校准修改方案

> **状态（暂停）**：adapter 架构已经实现，但本计划假定的 A2 factual baseline 尚未在
> 对齐的 policy-on 协议下测得。不得据此宣称 A2 factual non-inferiority 或启动训练；
> 前置要求见
> [`FITWMAMS_A2_Factual_Protocol_Correction.md`](FITWMAMS_A2_Factual_Protocol_Correction.md)。

> **用途**：作为下一轮代码修改、实验执行与论文方法收敛的统一依据。  
> **当前仓库**：`https://github.com/SafeDL/FITWMAMS`  
> **本文核对的 main 提交**：`aa1042bcf1f517009ea30eda1e624abc9a7998f2`  
> **GPU 环境**：`conda activate tread`  
> **基本原则**：不重新设计 Flow–Diffusion–HiQR 主世界模型，不丢弃已经有效的 A2 模型结构与训练权重；新工作只解决“如何在保持事实精度的同时，将 A2 已有响应能力校准为更可信的人类响应，并在数据外安全关键状态中进行有边界的合理外推”。

---

# 1. 论文 Motivation：建议采用的学术中文表述

## 1.1 高精度轨迹重建并不足以支撑自动驾驶安全评价

基于真实驾驶数据学习交通世界模型，首先需要准确恢复真实交通参与者的运动规律。近年来，数据驱动交通仿真方法已经能够利用真实轨迹学习多智能体运动分布，并在相同场景条件下生成具有较高轨迹保真度的交通演化。然而，对于自动驾驶安全测试而言，仅仅能够准确重建已经发生的交通轨迹仍然是不充分的。其根本原因在于，被测自动驾驶系统一旦采取了不同于真实日志中人类驾驶员的动作，原有交通历史即被改变；此时，周围车辆不能继续机械地沿着原始记录轨迹运行，而必须根据新的间距、相对速度、冲突关系及自身历史行为作出新的闭环响应。日志重放虽然能够取得极低甚至为零的轨迹误差，却无法保证这种交互有效性；已有工作也明确指出，当被测车辆行为偏离日志时，非反应式重放可能产生不真实的碰撞或交互结果（《TrafficSim: Learning to Simulate Realistic Multi-Agent Behaviors》；《Improving Agent Behaviors with RL Fine-tuning for Autonomous Driving》）。

因此，用于自动驾驶验证的交通世界模型需要同时满足两个不同层面的真实性：一方面，在未发生新的强交互时，应最大程度保持真实数据中已经学习到的事实交通行为；另一方面，当被测系统改变交通演化后，NPC 车辆应能够基于新的已实现历史自主调整行为。前者对应**事实保真性（factual fidelity）**，后者对应**交互响应真实性（response realism）**。这两个目标不能被简单等同。

## 1.2 从观察性自然驾驶数据可以学习响应规律，但闭环分布偏移仍然存在

真实自然驾驶数据并非只能用于静态轨迹重建。前车减速、车距压缩、后车制动、制动力建立以及风险解除后的恢复，本身都构成了真实交通中的刺激—响应过程。因此，观察性数据可以用于学习“在什么交通条件下，人类驾驶员通常如何响应”。TrafficSim 通过在训练阶段展开多智能体策略并通过可微仿真进行闭环优化，说明从真实观察数据中学习反应式交通策略是可行的；其结果同时表明，单纯开环行为克隆在长期展开时容易因误差累积而进入训练分布之外的状态（《TrafficSim: Learning to Simulate Realistic Multi-Agent Behaviors》）。

进一步地，RTR 将真实驾驶示范与闭环强化学习结合：真实数据负责约束 human-like 行为，而强化学习负责在长尾场景中提供交通规则与违规相关的闭环学习信号。该研究特别强调，程序化生成的长尾场景虽然能够提供丰富的交互与强化学习信号，但由于缺乏真实人类示范，不能直接作为 imitation target（《Learning Realistic Traffic Agents in Closed-loop》）。Waymo 的相关工作同样证明了“预训练行为模型 + 闭环强化学习微调”能够改善闭环交通代理，并提升仿真器区分不同自动驾驶规划器性能的能力（《Improving Agent Behaviors with RL Fine-tuning for Autonomous Driving》）。

因此，本文并不将“观察性数据”与“闭环学习”视为相互替代的两条路线，而是赋予二者不同职责：真实自然驾驶数据提供人类行为证据，闭环训练负责让策略在自身动作导致的新状态分布上保持合理行为。

## 1.3 一般交通真实性指标仍不足以直接评价安全关键“响应过程”

现有闭环交通仿真研究已经开始从单纯轨迹误差转向分布级真实性。RLFTSim 直接使用多次闭环 rollout 构成的交通真实性指标进行强化学习微调，并通过 leave-one-out 构造低方差的 rollout-level reward，从而避免仅使用 ADE 追踪某一条日志轨迹（《RLFTSim: Realistic and Controllable Multi-Agent Traffic Simulation via Reinforcement Learning Fine-Tuning》）。这一思想说明，随机交通模拟器应该评价生成分布与真实分布之间的一致性，而不是要求所有未来严格返回唯一日志轨迹。

但是，用于安全评价的 NPC 响应还具有更强的时序要求。例如，两种模型可能具有相似的总体加速度分布，但一个模型提前制动，另一个模型严重延迟；或者两者具有相同峰值减速度，但制动力建立速度和风险解除后的恢复过程完全不同。对于安全关键跟驰事件，这些差异会直接影响 TTC、最小间距和最终碰撞结果。因此，本文需要进一步将真实性评价从“一般交通特征分布”收缩到“**给定真实交通刺激以后，完整人类响应序列的条件概率分布**”。

## 1.4 安全关键场景中的“少碰撞”不能直接等同于“更真实”

安全关键场景生成研究表明，困难场景需要同时满足风险有效性和行为真实性。STRIVE 在真实交通先验空间中优化危险场景，并进一步区分可解和不可解场景，说明单纯追求碰撞并不能保证场景具有测试价值（《Generating Useful Accident-Prone Driving Scenarios via a Learned Traffic Prior》）。CCDiff 将闭环安全关键生成明确表述为 controllability 与 realism 之间的约束问题，并通过因果结构减少对无关交通参与者的干预（《Causal Composition Diffusion Model for Closed-loop Traffic Generation》）。CounterScene 进一步强调，真实风险通常来自局部行为变化经过多智能体交互逐步传播，而非对全场景施加无结构扰动；其核心思想是识别关键交通参与者并施加最小局部改变，使危险通过自然交互链产生（《CounterScene: Counterfactual Causal Reasoning in Generative World Models for Safety-Critical Closed-Loop Evaluation》）。

对于自动驾驶安全测试环境，上述问题还存在一个特殊性：**ADS–NPC 碰撞本身不一定意味着 NPC 模型错误**。如果被测 ADS 进行了极端且不可挽救的操作，真实人类驾驶员也可能因为有限反应时间和有限制动能力而发生碰撞。因此，若将“所有碰撞都必须避免”直接写入 NPC 强化学习目标，可能把背景车辆训练成理想化的超级驾驶员，从而替被测 ADS 消除本应暴露的风险。这与一般“infraction-free simulator”的优化目标并不完全相同。

## 1.5 本文需要解决的具体科学问题

综合上述工作，本文关注的不是如何重新建立一个更复杂的交通生成模型，而是如下更具体的问题：

> **在一个已经能够高精度重建自然交通的分层世界模型上，如何保留既有事实行为和已经学到的响应能力；如何利用真实自然驾驶事件，对 NPC 在完整时间维度上的响应概率分布进行闭环校准；以及在自然驾驶数据缺少覆盖的极端状态下，如何只施加具有明确交通机制含义的响应约束，而不是伪造人类反事实标签或把所有风险强行消除？**

由此，本文的方法创新应围绕三个核心设计展开：

1. **事实锚定（Factual Anchoring）**：保留冻结 HiQR 的高精度事实行为，同时将已经有效的 A2 作为冻结响应先验，而不是重新从零学习新的 response policy。
2. **完整响应序列的概率校准（Sequence-Level Human Response Calibration）**：以真实事件的完整 acceleration–jerk 响应序列为概率监督对象，在闭环 rollout 上使用 proper scoring rule 进行分布校准。
3. **数据外极端工况的机制约束（Mechanism-Constrained OOD Response）**：自然数据无充分人类证据时，不构造 synthetic human target，不追求“零碰撞”，而只约束交通条件变化前后 NPC 响应的方向、幅度、连续性和恢复合理性。

这三个部分共同服务于一个目标：构建**高事实保真、具有人类响应证据、同时不人为消除 ADS 风险**的可采样交通世界。

---

# 2. 当前代码与最新结果的重新诊断

## 2.1 当前结果并不是“方向失败”

当前 `human_response_a2` full validation 给出的结果为：

```text
factual:
  ADE delta  = +0.00085 m
  FDE delta  = -0.00476 m
  P95 delta  = +0.00063 m
  passed     = true

raw event Energy Score:
  frozen A2      = 79.2169
  new supervised = 125.2007
  new PPO        = 107.7192
```

因此当前结果说明：

```text
事实锚定结构：有效
闭环 PPO 学习信号：有效（125.20 -> 107.72）
相对于已有 A2 的人类响应：仍然不足（107.72 > 79.22）
```

不能把这次结果解释成“闭环人类响应校准没有价值”。更准确的结论是：

> 当前新候选成功解决了 factual degradation，但由于重新初始化了最关键的 response action head，实际丢掉了 A2 已经学到的响应能力；随后又试图用有限事件 PPO 从头把这一能力学回来，因此最终仍未达到 A2 的 response quality。

## 2.2 当前 `human_response_a2` 实际没有真正继承 A2

现有 `copy_a2_hidden()` 仅复制：

```text
actor.0.weight
actor.0.bias
actor.2.weight
actor.2.bias
```

A2 的：

```text
actor_mean
actor_log_std
动作映射语义
已学会的 response intensity
```

均未继承。

新 controller 的输出从 2 维变成 3 维，并重新初始化输出 head；同时 gate bias 设为 `-4.0`，初始 gate 大约只有 0.018。其本质是：

```text
保留部分 feature representation
+
几乎关闭响应
+
重新学习新的 response head
```

因此，新 supervised checkpoint factual 很好但人类响应 Energy Score 很差，是可以由当前实现直接解释的。

## 2.3 supervised initialization 当前主要学习“事件前不要动”

当前 `teacher_anchor_loss()`：

- 采样 event scene；
- 将其作为 `non_event` reset；
- 只执行 reset 后**一个 transition**；
- 该时刻位于真正 leader braking onset 之前。

所以 400 次 pretrain update 主要监督：

```text
事件发生前，尽量不要破坏 logged/HiQR 行为
```

而没有系统监督：

```text
onset 后 25 帧真实 follower 如何制动与恢复
```

因此 `supervised ES = 125.20` 并不是“监督学习不适合人类响应”，而是“当前 supervised task 没有训练完整响应”。

## 2.4 当前 kNN reference 的职责过重，而且数据结构存在混淆

当前 `human_response_prior.py` 先构建完整 train `library`，但最终 `HumanResponsePrior` 对象保存的是 split `query` 的 `descriptor/response`。结果是当前训练时真正做 neighbor search 的 reference corpus 主要是 replayable ego-leader query 事件，而不是更大的 train human library。

当前 audit 为：

```text
train events: 1841
  empirically_supported: 1246
  weakly_supported:      273
  unsupported:           322

validation events: 265
  empirically_supported: 254
  weakly_supported:      11
```

这说明当前 validation 并不是主要因为 OOD 而失败。更重要的问题是：

> kNN 邻居本身并不等于“这个具体事件条件下真实驾驶员的唯一分布”。

目前训练优化的是：

```text
模型的 8 条响应
vs.
16 个相似事件的人类响应
```

而最终 validation 优化的是：

```text
模型的 32 条响应
vs.
当前事件自己真实观察到的响应
```

存在明确的训练—评价目标错位。

## 2.5 当前训练历史已经显示 factual–response Pareto 冲突

quick validation 中，Energy Score 大致呈现：

```text
122.36 -> 120.15 -> 116.25 -> 112.11 -> 109.65 -> 104.81
```

此时 factual 仍能通过。

继续训练后：

```text
101.44 -> 99.11 -> 99.05 -> 96.99
```

Energy Score 继续改善，但 factual 开始失败并不断恶化。

这说明：

> 问题不是“PPO update 不够多”，而是当前策略必须通过越来越强的 response correction 才能降低 human response loss，从而跨越 factual constraint。

因此下一步不应该继续扩大 500 updates 或重新调大 PPO budget，而应该首先改变**初始化与训练目标**。

## 2.6 当前 formal factual validation 还需要修正范围

现有 `full_validate.py` 先通过：

```python
rows = event_rows(reference)
```

仅构造 validation reaction-event rows，然后在这些 rows 上做 factual check。

因此当前所谓：

```text
full validation factual passed
```

实际上是：

```text
full human-response event subset factual passed
```

而不是：

```text
整个 validation split factual non-inferiority passed
```

下一轮必须将 formal factual gate 改为完整 validation population，同协议测：

```text
frozen HiQR
legacy A2
new supervised
new final
```

否则无法真正证明“高精度世界模型能力被保持”。

---

# 3. 下一轮方法总览：A2 不再只是 baseline，而是冻结响应先验

下一轮最重要的修改是：

> **不再创建一个几乎从零开始的 3D response policy。直接保留完整 A2 已训练 policy，冻结其有效权重，并只学习一个小型的人类响应校准层。**

总体结构建议改为：

```math
a_t^{H}
=
\pi_{\mathrm{HiQR}}(H_t,K,z_H)
```

冻结 A2 输出一个 learned response proposal：

```math
a_t^{A2}
=
\pi_{\mathrm{A2}}^{proposal}(H_t,K,z_H,z_{A2})
```

定义 A2 已经学到的响应修正：

```math
\delta_t^{A2}
=
a_t^{A2}-a_t^{H}
```

新的 calibration policy 只学习：

```math
(s_t,r_t)
\sim
q_\phi(\cdot\mid H_t,\delta_t^{A2},h_t^{A2})
```

最终：

```math
a_t^{pre}
=
a_t^{H}
+
s_t \delta_t^{A2}
+
r_t
```

其中：

```text
s_t：校准 A2 已有 correction 的保留/抑制/有限放大
r_t：只学习 A2 无法表达的小型 signed correction
```

这一结构的关键不是“再叠加一个 residual”，而是：

```text
A2 = 已经学会如何响应的冻结 prior
新策略 = 人类响应 calibration
```

而不是：

```text
新策略重新学习整个 response policy
```

---

# 4. 必须完整保留哪些 A2 内容

## 4.1 保留完整 checkpoint，而不是只复制 hidden layers

新增一个只读组件：

```text
FrozenA2ResponsePrior
```

必须完整加载现有：

```text
results/hierarchical_world_model/ppo_idm_response/checkpoint.pt
```

以下权重全部冻结：

```text
actor.0
actor.2
actor_mean
actor_log_std
```

旧 A2 critic 可以：

```text
保留用于诊断，但不参与新 reward value estimation
```

新 candidate 不允许修改上述 A2 policy 参数。

## 4.2 A2 的旧随机性必须完整保留

旧：

```text
policy_response_innovations
```

继续只供 A2 使用。

禁止因为新 candidate 改变维度而改变 A2 原始随机流。

新增独立随机块：

```text
policy_calibration_innovations
```

建议 2 维：

```text
scale innovation
residual innovation
```

这样：

```text
同 seed 下
legacy A2 的 raw action 完全不变
```

新 calibration stochasticity 与 A2 原 response randomness 分离。

## 4.3 将 A2 “已训练 policy”与“非学习 hard guard”在代码中拆开

当前 A2 forward 同时包含：

1. frozen actor distribution；
2. IDM-bounded action mapping；
3. TTC / unresolved dynamic braking rewrite；
4. jerk rewrite。

为了既保留 A2 learned behavior，又满足 safety-testing world 的风险保留原则，应将 A2 forward 重构成两个内部阶段：

```python
a2_policy_proposal(...)
legacy_a2_execution_guard(...)
```

旧 `IDMResidualReactionController.forward()` 仍然调用两者，保证旧 A2 baseline 的输出语义不变。

新 candidate 只调用：

```text
a2_policy_proposal
```

即保留：

```text
A2 actor
A2 head
A2 log_std
A2 IDM feature
A2 dynamic action mapping
```

但不继承“为了避免风险而强制重写动作”的 legacy behavior guard。

原因：

> guard 并不是训练得到的 A2 权重；将其与 learned response prior 分开，不属于丢弃 A2 已学能力，反而能够明确区分“模型学会了响应”和“规则层替模型刹车”。

旧 A2 仍完整保留作 baseline，不修改历史结论。

---

# 5. 新的 A2 Calibration Adapter

## 5.1 输入

建议直接复用 frozen A2 的 hidden representation：

```text
h_A2
```

并拼接少量在线可观测量：

```text
delta_A2
base HiQR acceleration
IDM - HiQR difference
influence authority
previous adapter correction
```

不要重新训练一个新的 128×128 response trunk。

## 5.2 输出分布

adapter 使用一个小型 stochastic actor：

```text
Linear -> SiLU -> Linear(2)
```

输出：

```math
u_s,u_r
```

并学习自己的：

```text
log_std_calibration
```

推荐初始化：

```text
mean = 0
log_std = -2.0
```

映射：

```math
s_t = 1 + \eta_s \tanh(u_s)
```

```math
r_t = r_{max}\tanh(u_r)
```

首轮：

```yaml
scale_radius: 0.75
residual_max_mps2: 2.0
```

于是：

```text
adapter mean at initialization:
s = 1
r = 0
```

即**均值行为保持 A2 proposal**。

不同于当前 3D policy 的 gate≈0 初始化，新模型不是从“几乎不响应”开始，而是从“已经有效的 A2 response”开始。

## 5.3 为什么 adapter 必须是 stochastic policy，而不是 deterministic head

闭环 HighwayEnv 并不是端到端可微仿真。

若冻结 A2 后只添加 deterministic adapter，则 adapter 不参数化 rollout sampling probability，不能直接通过 PPO score-function 获得梯度。

因此 adapter 必须有自己的随机变量：

```math
q_\phi(u_s,u_r\mid\cdot)
```

PPO 的 log probability 只计算：

```text
calibration policy log_prob
```

冻结 A2 的 stochastic raw action被视为世界中的 exogenous response prior，不参与新的 PPO 更新。

这样既能：

```text
保留 A2 权重
```

又能：

```text
让新 adapter 通过 PPO 真正学习
```

---

# 6. 事实锚定：从“从零关闭响应”改为“选择性校准 A2”

最终 correction：

```math
\delta_t^{new}
=
s_t\delta_t^{A2}+r_t
```

相对 A2 的额外 calibration：

```math
\Delta_t^{cal}
=
(s_t-1)\delta_t^{A2}+r_t
```

执行时只约束：

```text
calibration-induced jerk
```

即：

```math
j_t^{cal}
=
\frac{|\Delta_t^{cal}-\Delta_{t-1}^{cal}|}{\Delta t}
```

而不是要求新 controller 修复 HiQR/A2 历史动作自身的所有 jerk。

初始：

```text
s = 1
r = 0
```

因此：

```text
calibration correction = 0
```

不会一开始破坏 A2。

在普通/事实场景中，训练可以学到：

```text
s -> 0
r -> 0
```

从而逐渐把不必要的 A2 correction 拉回 HiQR。

在真实交互事件中：

```text
s ≈ 1
```

即可保留 A2 已经有效的响应；

只有 A2 与真实人类存在系统差异时才学习：

```text
s != 1
或
r != 0
```

这才是“事实锚定 + 保留已有响应能力”的正确结构。

---

# 7. Supervised Initialization 必须改成“完整事件 teacher forcing”

这是下一轮最重要的训练修改。

## 7.1 当前做法必须停止

禁止继续：

```text
event scene
-> reset 到 onset 前
-> 只监督第一帧
```

因为它只教模型“别乱动”，没有教模型“如何响应”。

## 7.2 新增 logged-context teacher-forced cache

新增：

```text
teacher_forced_response_cache
```

对每个 train natural event 的：

```text
pre-event + onset 后 25 帧
```

逐帧构造 online-safe controller context。

这里不通过 closed-loop HighwayEnv 前推生成背景状态，而是直接使用真实 highD 已经发生的：

```text
H_t^human
```

然后让冻结 HiQR 与冻结 A2 在该真实历史上计算：

```text
a_HiQR(H_t^human)
a_A2_proposal(H_t^human)
h_A2(H_t^human)
```

target 使用同一时刻真实 follower acceleration：

```text
a_human,t
```

这样是合法的 supervised learning：

```text
真实状态 -> 同一真实状态下发生的真实动作
```

不存在“把偏离后的状态强配日志标签”的问题。

## 7.3 cache 生成必须复用 runtime 同一套 context builder

将 `HighwayEnvClosedLoopWorld._controller_context()` 中与：

```text
history/current/influence graph/features
```

相关的纯计算逻辑抽成可复用 helper。

teacher-forced cache 和 runtime 都调用该 helper，避免训练/执行特征语义分叉。

## 7.4 supervised loss

自然 event：

```math
L_{event}
=
Huber(
a_{\phi,\mathrm{mean}}^{exec}(H_t^{human}),
a_t^{human}
)
```

matched non-event：

```math
L_{non}
=
Huber(
a_{\phi,\mathrm{mean}}^{exec},
a_t^{human}
)
+
\lambda_0|\Delta^{cal}|
```

总损失：

```math
L_{sup}
=
L_{event}
+
\lambda_{non}L_{non}
+
\lambda_{prior}\|\mu_\phi\|^2
```

其中 `prior` 项只是防止无证据的大幅偏离 A2，不应过大。

## 7.5 supervised 阶段的验收

进入 PPO 前必须同时满足：

```text
full-validation factual gate 通过
```

并且：

```text
supervised natural-event ES
不能比 legacy A2 差超过 10%
```

若 full-event supervised 仍然从 A2≈79 恶化到 >100：

```text
禁止启动 PPO
```

说明 teacher-forcing / action mapping 仍存在实现问题，应先修代码，不允许再靠 RL 从头补救。

---

# 8. 人类响应主创新：完整响应序列的 Event-Level Energy Score PPO

## 8.1 不再用 kNN human response 作为主要 target

kNN human library 保留，但职责修改为：

```text
empirical support / coverage estimation
```

不再作为主要 human response target。

原因：

> “相似的 gap/TTC/brake dose”并不意味着是同一个条件驾驶分布；邻居驾驶员具有不同风格和未观测因素。将 16 个近邻直接视为当前事件真实条件分布会引入 target bias。

## 8.2 每个真实事件直接使用自己观察到的完整 response realization

对于 event `e`：

真实序列：

```math
Y_e^{obs}
=
[a_{1:25},|j|_{1:25}]
```

模型从同一真实 event prefix 出发生成：

```math
Y_e^{(1)},\ldots,Y_e^{(K)}
```

计算：

```math
ES_e
=
\frac1K\sum_k
\|Y_e^{(k)}-Y_e^{obs}\|
-
\frac1{2K^2}\sum_{k,l}
\|Y_e^{(k)}-Y_e^{(l)}\|
```

训练时 acceleration / jerk 使用 train-only IQR 归一化：

```text
Normalized Event Energy Score
```

最终论文评估同时报告：

```text
normalized ES
raw historical ES
```

## 8.3 为什么“一条真实 future”也能够训练概率分布

对于每个完全相同的交通上下文，确实只有一条观察到的人类未来。

但是 across events，我们拥有大量：

```math
(X_e,Y_e^{obs})
```

样本。

Energy Score 属于 proper scoring rule。最小化跨事件平均 score：

```math
E_{(X,Y)\sim P_{human}}
[
ES(P_\theta(\cdot|X),Y)
]
```

在总体意义上鼓励模型条件分布接近真实条件分布。

因此，无需人为假设：

```text
每个 event 的 16 个 kNN neighbor
就是同一条件下的 16 条 ground truth future
```

这使训练目标与最终验证目标完全一致。

---

# 9. 将 Event Energy Score 转为 PPO reward

## 9.1 Leave-One-Out contribution

对于 K 条 rollout，删除第 k 条后：

```math
ES_{e,-k}
```

由于 Energy Score 越低越好，定义：

```math
R_{e,k}^{human}
=
ES_{e,-k}
-
\frac1K\sum_j ES_{e,-j}
```

解释：

```text
删除 rollout k 后 score 变差
=> rollout k 对解释真实 human response 有价值
=> 正 reward
```

反之：

```text
删除它以后 score 变好
=> 该 rollout 是不自然样本
=> 负 reward
```

PPO 只优化 calibration adapter 的 log probability：

```math
\log q_\phi(u_s,u_r|\cdot)
```

冻结 A2 不更新。

## 9.2 增加 prefix-level temporal credit

当前一个 sequence reward 平均摊到 25 帧，credit 太粗。

建议计算三个 prefix score：

```text
5 frames
10 frames
25 frames
```

分别捕捉：

```text
5 frames:  response onset / latency
10 frames: braking ramp / early intensity
25 frames: sustained response + recovery trend
```

对应权重首轮固定：

```yaml
prefix_weights:
  5:  0.25
  10: 0.35
  25: 0.40
```

每个 prefix 单独计算 LOO contribution，并在对应时间点注入 reward。

这样不需要额外 reward shaping，就能给时序行为更清晰的 credit。

---

# 10. PPO 不再负责“从零学会响应”，只负责 closed-loop calibration

## 10.1 新 PPO 的职责

supervised 已经在真实状态上学到：

```text
人应该怎样响应
```

PPO 只解决：

```text
当自己的动作使状态偏离日志以后，
如何仍保持人类响应分布
```

因此 PPO 应当是小步微调。

## 10.2 建议预算

当前：

```yaml
max_ppo_updates: 500
ppo_epochs_per_update: 8
learning_rate_start: 3e-4
```

对 calibration adapter 过大。

建议改为：

```yaml
minimum_ppo_updates: 40
max_ppo_updates: 160

ppo_epochs_per_update: 4

learning_rate_start: 3e-5
learning_rate_end:   1e-5

target_kl: 0.010
inner_epoch_stop_kl: 0.020
```

quick validation：

```yaml
interval: 10
events: 32
futures: 8
```

连续 4 次无 improvement：

```text
early stop
```

## 10.3 增加对 supervised adapter 的 trust region

冻结一份：

```text
q_ref = supervised calibration adapter
```

PPO loss 增加：

```math
\beta_{KL}
D_{KL}
(
q_\phi
\|
q_{ref}
)
```

目标：

```text
target KL ≈ 0.01
```

这和“冻结 A2”是两个不同层次：

```text
A2：长期 response prior，完全冻结
supervised adapter：本轮人类行为初始化，用 KL 防止 PPO 漂移
```

---

# 11. kNN Human Library 的新职责：只负责 empirical support

## 11.1 修正数据结构

当前 `HumanResponsePrior` 混合了：

```text
train reference library
split query events
```

必须拆成：

```text
TrainHumanResponseLibrary
HumanResponseQuerySet
```

### TrainHumanResponseLibrary

保存所有可用 train human response events：

```text
descriptor
response
recording
leader/follower
metrics
```

### QuerySet

仅保存 train/validation/test 对应 event 的：

```text
descriptor
support label
neighbor distance
recording count
```

query 不应再成为 neighbor reference library。

## 11.2 support 只用于三件事

1. 论文报告：
   ```text
   多少 validation/test event 有充分训练证据
   ```
2. PPO event weighting：
   ```text
   supported = 1.0
   weak      = 0.5
   unsupported=0
   ```
3. claim boundary：
   ```text
   只有 supported 才称为 data-supported human response
   ```

support label：

```text
绝对不能作为 online controller input
```

---

# 12. 数据外极端工况：保留机制差分，但修正 sampler 与作用对象

## 12.1 继续坚持：没有 synthetic human label

synthetic：

```text
-2/-4/-6/-8 m/s²
ramp
pulse
```

不产生：

```text
human target
absolute IDM target
collision-avoidance human reward
```

## 12.2 继续比较“变化”而不是绝对 IDM 动作

baseline 与 intervention 使用：

```text
同 initial state
同 A2 noise
同 calibration noise
同 world noise
```

计算：

```math
\Delta a^{model}
=
a^{int}-a^{base}
```

```math
\Delta a^{IDM}
=
a^{IDM,int}-a^{IDM,base}
```

只约束：

```text
direction
weak magnitude
continuity
recovery
```

不要求：

```text
a_model == a_IDM
```

## 12.3 修复当前 synthetic row sampler

当前实现先取：

```python
np.arange(N)[:pairs]
```

再只平移 `seed % pairs`，导致 `pairs=8` 时长期集中在很少的前部场景。

必须改为真正覆盖整个 train pool：

```python
rows = rng.choice(
    eligible_rows,
    size=pairs,
    replace=len(eligible_rows) < pairs,
)
```

baseline/intervention 共用同一 rows。

## 12.4 mechanism auxiliary 只更新 calibration adapter

冻结 A2。

mechanism loss：

```math
L_{mech}
=
L_{dir}
+
0.1 L_{mag}
```

只对：

```text
calibration adapter
```

回传。

这样 OOD 训练不会破坏 A2 已经学到的 response prior。

---

# 13. 三个创新重点如何在最终算法中对应

## 13.1 创新一：事实锚定

不是：

```text
重新训练一个 response controller，再要求它不要破坏 HiQR
```

而是：

```text
HiQR factual prior
+
Frozen A2 response prior
+
Small stochastic calibration adapter
```

通过结构把：

```text
事实交通
响应能力
人类校准
```

分成三个职责。

## 13.2 创新二：完整人类响应序列概率校准

不是：

```text
frame-wise action regression
```

不是：

```text
总体 acceleration histogram
```

也不是：

```text
GAIL real/fake
```

而是直接优化：

```math
p(
a_{1:T},j_{1:T}
\mid
interaction history
)
```

使用 Event-Level Energy Score 评价整段随机响应，并通过 LOO 将 ensemble proper score 转为 rollout-level PPO signal。

## 13.3 创新三：数据外极端工况的合理响应约束

数据外：

```text
不假装有 human label
不奖励万能避碰
```

只要求：

```text
交互变危险 -> 响应方向不反常
响应幅度有限
动作连续
风险解除后可恢复
```

因此模型保留：

```text
真实不可避免风险发生的可能性
```

而不是把 NPC 变成 safety shield。

---

# 14. Formal factual evaluation 必须重做

## 14.1 当前 event-only factual 不够

正式 validation 必须使用：

```text
完整 validation split
```

而不是：

```text
reaction event rows only
```

## 14.2 必须同时测 4 个模型

统一协议：

| Model | Full Factual | Human ES |
|---|---:|---:|
| Frozen HiQR | required | required |
| Legacy A2 | required | required |
| New Supervised Adapter | required | required |
| New PPO Adapter | required | required |

当前 A2 的历史 `factual_reconstruction.json` 实际记录的是：

```text
controller = A0_none
```

它只能证明 HighwayEnv bridge，不代表 A2 factual non-inferiority。

下一轮开始训练之前，必须先补：

```text
legacy A2 full-validation factual
```

## 14.3 Acceptance 改为“约束下的最优响应”，不是机械支配所有 baseline

正式目标写成：

```math
min HumanResponseScore
```

subject to：

```math
FactualNonInferiority = true
```

如果 legacy A2 自己不能通过 factual gate，则它是：

```text
response reference
```

不是：

```text
必须被新模型同时在 factual 和 ES 上完全支配的 feasible baseline
```

此时 acceptance 要求：

```text
new PPO 在所有 factual-feasible models 中 human ES 最优
```

同时报告：

```text
与 legacy A2 ES 的剩余 gap
```

如果 legacy A2 本身 factual 也通过，则要求 new PPO 对 A2 至少 human-response non-inferior。

---

# 15. 新的 Acceptance Criteria

## 15.1 Protocol

必须全部通过：

```text
train/validation/test recording isolation
no future GT online
no event/source/split/intervention label online
A2 old noise stream unchanged
new calibration RNG independent
```

## 15.2 Full factual

相对 frozen HiQR：

```text
ADE absolute delta <= 0.02 m
FDE absolute delta <= 0.06 m
P95 absolute delta <= 0.10 m
```

并且各自：

```text
relative degradation <= 5%
```

在**整个 validation population**计算。

## 15.3 Supervised warm-start gate

必须：

```text
factual pass
```

且：

```text
supervised event ES
<= 1.10 × legacy A2 event ES
```

若失败：

```text
STOP before PPO
```

## 15.4 PPO increment

final vs own supervised：

```text
Supervised ES - Final ES
```

recording-cluster bootstrap：

```text
LCB95 > 0
```

否则：

```text
PPO contribution not established
```

## 15.5 Human response

若 A2 factual feasible：

```text
Final vs A2:
Energy Score non-inferiority / improvement
```

若 A2 factual infeasible：

```text
Final 必须是 factual-feasible arm 中 ES 最优
```

并报告：

```text
A2 response gap
```

## 15.6 Diagnostics

至少：

```text
latency
peak acceleration
peak abs jerk
gap
closing
TTC
recovery
```

不能出现一个关键维度明显恶化却被综合 ES 掩盖。

## 15.7 Non-event

报告：

```text
mean |new - HiQR|
P95  |new - HiQR|
adapter activation rate
fraction(|calibration| > 0.25 m/s²)
```

禁止 calibration adapter 变成全天候第二控制器。

## 15.8 OOD

要求：

```text
finite = 100%
action physical bound violation = 0
controller-induced jerk violation = 0
mechanism direction agreement >= 95%
```

碰撞：

```text
outcome only
```

但区分：

```text
target ADS–NPC collision
unrelated NPC–NPC artifact collision
response-induced offroad
```

后两者属于 world-model validity failure。

---

# 16. 代码修改清单

## 16.1 `reaction_controller.py`

### Refactor A2

新增内部 API：

```python
IDMResidualReactionController.proposal(...)
IDMResidualReactionController.legacy_execute(...)
```

要求旧 `forward()` 输出不变。

### 新增

```text
FrozenA2ResponsePrior
A2HumanCalibrationAdapter
HumanResponseA2Controller
```

`HumanResponseA2Controller` 内部：

```text
frozen A2 proposal
+
stochastic calibration adapter
+
physical-only execution
```

A2 参数：

```python
requires_grad_(False)
```

## 16.2 `randomness.py`

保留：

```text
policy_response_innovations
```

新增独立：

```text
policy_calibration_innovations
```

禁止改变旧 random block 的 shape/order。

新增测试：

```text
same seed -> legacy A2 old noise bitwise unchanged
```

## 16.3 `human_response_prior.py`

拆为：

```text
TrainHumanResponseLibrary
HumanResponseQuerySet
```

删除：

```text
kNN human response as PPO primary target
```

保留：

```text
descriptor scaling
neighbor distance
recording coverage
support labels
```

## 16.4 `human_response_training.py`

新增：

```python
event_energy_score(...)
loo_event_energy_rewards(...)
prefix_loo_event_energy_rewards(...)
```

保留：

```python
mechanism_auxiliary_loss(...)
controller_induced_jerk(...)
```

## 16.5 `reaction_training.py`

新增：

```text
logged-context / teacher-forced context builder
full-event supervised cache
```

teacher-forced context 和 runtime context 必须共享相同 feature construction。

## 16.6 `scripts/human_response_a2/train.py`

重写流程：

```text
load frozen world
load complete frozen A2
build/load full-event teacher cache
train calibration adapter supervised
full factual + supervised response gate
closed-loop event ES PPO
mechanism auxiliary
teacher anchor auxiliary
early stop
```

禁止：

```text
copy_a2_hidden(...)
```

替换为：

```text
load_full_frozen_a2(...)
```

## 16.7 `full_validate.py`

必须同时计算：

```text
full split factual
event ES
normalized event ES
diagnostics
legacy A2 factual
```

并加入 recording-cluster bootstrap。

---

# 17. 建议的新配置

```yaml
method:
  frozen_response_prior: ppo_idm_response
  controller: human_response_a2
  use_legacy_ttc_guard: false

adapter:
  hidden_dim: 64
  scale_radius: 0.75
  residual_max_mps2: 2.0
  initial_log_std: -2.0
  correction_jerk_limit_mps3: 12.0

human_response:
  channels:
    - acceleration
    - abs_jerk
  prefixes: [5, 10, 25]
  prefix_weights: [0.25, 0.35, 0.40]
  use_knn_as_target: false
  support_weight:
    empirically_supported: 1.0
    weakly_supported: 0.5
    unsupported_by_training_evidence: 0.0

supervised:
  updates: 400
  event_frames: 25
  include_matched_non_event: true
  event_weight: 1.0
  non_event_weight: 1.0
  prior_regularization: 0.01

ppo:
  event_groups_per_update: 8
  futures_per_event: 8
  minimum_updates: 40
  maximum_updates: 160
  epochs_per_update: 4
  learning_rate_start: 3.0e-5
  learning_rate_end: 1.0e-5
  target_kl: 0.01
  inner_stop_kl: 0.02
  entropy_coefficient: 0.001
  reference_kl_target: 0.01
  quick_validation_interval: 10
  early_stopping_patience: 4

synthetic:
  pairs_per_update: 8
  direction_weight: 1.0
  magnitude_weight: 0.1
  human_reward: 0.0
  collision_penalty: 0.0
```

首轮不要为这些数值做大规模 grid search。

只有：

```text
实现 bug
```

或：

```text
明显数值发散
```

才允许调整。

---

# 18. GPU 与运行环境

所有训练前：

```bash
conda activate tread
```

保存：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY

nvidia-smi
git rev-parse HEAD
git status --short
```

当前方案应以单张 RTX 4090 级 GPU 可完成为约束。

不得因为训练慢自动扩成：

```text
大规模自博弈
多模型 ensemble
从零训练 PPO
重新训练 Flow/Diffusion/HiQR
```

---

# 19. 实验矩阵

下一轮只保留：

| Arm | 用途 |
|---|---|
| Frozen HiQR | factual anchor |
| IDM-only | mechanistic reference |
| Legacy A2 | frozen learned response reference |
| New adapter supervised | 检验完整事件监督是否保住 A2 |
| New adapter PPO | 最终候选 |

不再训练：

```text
GAIL
MLOO
CalibratedResidual old branch
NominalPreserving alternative
new diffusion policy
new convex controller
```

历史结果只作为 rejected evidence 保留。

---

# 20. 执行顺序

## Phase 0 — 基线补全

先完成：

```text
Legacy A2 full-validation factual
A2 learned proposal vs legacy guard contribution audit
Frozen HiQR full factual
```

在知道 A2 当前真正的 factual–response 坐标之前，不训练新模型。

## Phase 1 — A2 refactor

拆 proposal / guard。

旧 baseline regression test 必须通过。

## Phase 2 — teacher-forced event cache

构造完整 onset+25-frame cache。

人工检查若干 event，确认：

```text
logged context
base action
A2 proposal
human action
```

时间对齐正确。

## Phase 3 — supervised calibration

只训练 adapter。

若 supervised ES 仍明显差于 A2：

```text
STOP
```

## Phase 4 — event-level ES PPO

只对 adapter PPO。

## Phase 5 — OOD mechanism auxiliary

修复 random sampler，加入 paired mechanism。

## Phase 6 — full validation

一次正式 validation。

## Phase 7 — one-shot test

只有 validation acceptance 全通过才执行。

## Phase 8 — 2,000 paired MC

只做风险敏感性诊断。

本轮仍不自动重新运行 AMS。

---

# 21. 论文贡献建议

最终若实验成功，贡献不建议写成：

```text
我们提出 HiQR + IDM + PPO
```

而建议写成：

### Contribution 1：事实锚定的响应校准架构

在冻结高精度交通行为基座和已训练 A2 响应策略的基础上，仅学习小型随机校准策略，使闭环响应学习不再需要重新学习基础驾驶行为，并将 factual fidelity 明确作为约束。

### Contribution 2：完整人类响应序列的闭环概率校准

针对自然驾驶交互事件，以 acceleration–jerk 完整响应序列构造 event-conditioned Energy Score，并通过 leave-one-out contribution 将 ensemble proper score 转化为闭环策略优化信号，从而直接优化响应时机、强度、持续性及多样性。

### Contribution 3：面向安全评价的数据支持感知机制外推

显式区分有真实人类证据的行为区域与数据外安全关键状态；前者进行人类分布校准，后者只施加 paired mechanism-difference 与物理可执行约束，不构造伪人类标签、不无条件惩罚 ADS–NPC collision，从而避免背景车辆过度避险掩盖被测系统风险。

---

# 22. 论文 claim 边界

若成功，可以写：

> 本文构建了一个以高精度事实行为为锚点、在自然驾驶支持域内经完整响应序列概率校准、并在数据外安全关键状态中进行机制约束外推的闭环交通行为世界模型。

可以写：

```text
human-response calibrated in empirically supported domain
```

可以写：

```text
mechanism-constrained extrapolation outside the supported domain
```

可以写：

```text
risk-preserving background response for ADS evaluation
```

不能写：

```text
恢复了真实驾驶员唯一反事实行为
```

不能写：

```text
synthetic -8 m/s² response 等同于真实人类
```

不能写：

```text
collision 更少就是 world model 更真实
```

不能写：

```text
条件重仿真就是 unconditional real-world prediction
```

---

# 23. 最终判断

当前 `human_response_a2` 不应该继续在现有 3D head 上增加训练预算。

下一轮最关键的改变是：

```text
从“借 A2 hidden representation 重新学响应”
```

改成：

```text
“冻结完整 A2 作为已训练 response prior，
只学习一个小型 stochastic human-response calibration policy”
```

并将：

```text
kNN two-sample Energy Distance
```

从主要 human target 降级为：

```text
empirical support estimator
```

真正的闭环 human objective 改为：

```text
当前事件自身 observed full response
+
Event-Level Leave-One-Out Energy Score
```

这样三个论文核心创新才能真正对齐：

```text
事实锚定
=
HiQR + frozen A2 + small calibration

完整人类响应概率校准
=
full-event teacher forcing + event-level ES PPO

数据外合理响应
=
paired mechanism difference + no fake human target + no universal collision penalty
```

这是目前最有希望既保留已有 A2 成果、又把方法创新做实，同时控制剩余 GPU 与工程成本的一条收敛路线。

---

# 24. 本文主要参考文献

本文 Motivation 与方法设计重点参考了以下用户提供的全文材料：

- 《TrafficSim: Learning to Simulate Realistic Multi-Agent Behaviors》
- 《Learning Realistic Traffic Agents in Closed-loop》
- 《Improving Agent Behaviors with RL Fine-tuning for Autonomous Driving》
- 《RLFTSim: Realistic and Controllable Multi-Agent Traffic Simulation via Reinforcement Learning Fine-Tuning》
- 《Generating Useful Accident-Prone Driving Scenarios via a Learned Traffic Prior》
- 《Causal Composition Diffusion Model for Closed-loop Traffic Generation》
- 《CounterScene: Counterfactual Causal Reasoning in Generative World Models for Safety-Critical Closed-Loop Evaluation》

文中的引用位置已经按要求直接在正文括号中列出相应论文标题，便于后续写作时替换为正式编号引用。

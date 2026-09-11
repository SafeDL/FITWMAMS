# FITWMAMS 下一轮目标实验任务书  
## Human-Response-Calibrated A2：事实锚定、事件条件化的人类响应闭环校准

> **用途**：作为下一轮 Codex 的唯一主线执行目标。  
> **代码仓库**：`https://github.com/SafeDL/FITWMAMS`  
> **冻结基线提交**：`889e57c74f7a7a543ef8ec8bebb59f1e34e22dd0`  
> **GPU/Conda 环境**：`conda activate tread`  
> **目标范围**：高速公路纵向跟驰、前车制动、切入引发的纵向冲突；不扩展为全城市多车博弈，不重训 Flow / Diffusion / HiQR 主干。  
> **本轮核心原则**：只实现一个新候选，不重新开启 GAIL、MLOO、全新 diffusion policy、从零 PPO 等并行路线。

---

# 0. 一句话目标

在现有高精度 `Flow → Diffusion → HiQR → HighwayEnv` 世界模型上，保留 HiQR 作为**事实行为锚点**，升级 A2 的局部响应层，使 NPC：

1. **自然/事实条件下尽量保持现有高精度行为**；
2. **真实 highD 有证据的交互条件下，学习人类响应的条件分布与时间过程**；
3. **观察数据没有覆盖的极端状态下，只学习有明确含义的响应机制和物理可执行性，不伪造“人类真值”**；
4. **不能通过把所有 ADS–NPC 碰撞训练掉来获得虚假的安全性**；
5. 最终只有在**事实精度 + 人类响应 + 闭环外推 + 信息时序**同时通过后，才允许进入 ADS 风险测试。

内部候选名称统一为：

```text
human_response_a2
```

论文最终名称以后再定，本轮不要花时间命名。

---

# 1. 为什么本轮选择这条路线

## 1.1 现有工程不是从零开始

当前仓库已经具备：

- Flow 场景分布；
- Diffusion soft plan；
- HiQR 逐帧背景动作；
- HighwayEnv 25 Hz 闭环执行；
- `CausalInfluenceGraph` 局部影响作用域；
- global IDM/MOBIL 参考；
- A1/A2/A3 历史 PPO 结果；
- highD 自然制动响应事件；
- event-level acceleration + jerk Energy Score；
- matched non-event；
- `-2/-4/-6/-8 m/s²` synthetic brake、ramp、pulse；
- 显式 `WorldExogenousState` 与 common random numbers；
- EVT / MC / AMS 风险接口。

因此，本轮不应该再重新发明一套世界模型。

## 1.2 历史 A2 有价值，不应推翻

仓库历史审计已经表明，A2 (`rl_residual_idm`) 能够产生：

- 局部响应；
- 延迟响应；
- command 结束后仍持续的响应；
- 在旧协议下保留 factual replay。

本轮的任务不是证明“A2 思想错误”，而是解决：

> **A2 会响应，但这种响应是否在正确情境下、以正确强度和正确时间过程接近真实人类？**

## 1.3 文献给出的共同启示

本轮设计直接参考以下论文，不把它们的模块机械拼接：

### TrafficSim, CVPR 2021
关键启示：

- 从观察性真实驾驶数据可以学习反应式交通策略，不仅仅是轨迹重放；
- 训练时闭环展开比单纯 open-loop BC 更能抑制 compounding error；
- 但纯数据模仿在安全关键状态缺少监督，因此需要额外约束。

### Learning Realistic Traffic Agents in Closed-loop (RTR), CoRL 2023
关键启示：

- IL 负责 human-like；
- RL 负责显式规则与长尾能力；
- **procedural long-tail 没有人类专家标签，所以只提供 RL 信号，不能施加 imitation loss**；
- pure RL 容易得到不自然的行为。

### Improving Agent Behaviors with RL Fine-tuning, ECCV 2024
关键启示：

- “预训练行为模型 + 闭环 RL 微调”是成立的；
- 但其 reward 中包含对 logged GT position 的追踪；
- 对本项目的 ADS 干预场景而言，历史发生变化以后不能要求 NPC 追赶原 GT 轨迹。

### RLFTSim, 2026
关键启示：

- ADE 不是安全关键闭环中的理想 reward；
- 分布级 realism metric 更适合随机闭环生成；
- MLOO 的核心思想是把“一组 rollout 才能计算的分布真实性指标”转成每条 rollout 的 dense contribution reward；
- 论文明确承认 RMM 仍只是 realism proxy；
- 本项目不复现 RMM/MLOO，而借鉴“群体分布指标 → leave-one-out rollout reward”的思想。

### STRIVE, CVPR 2022
关键启示：

- 危险行为不能脱离真实交通先验无限优化；
- challenging scenario 还要判断是否“useful / solvable”，避免只制造退化碰撞；
- 对本项目而言，synthetic stress 是训练响应能力的工具，不代表自然发生概率。

### CCDiff, 2025
关键启示：

- realism 与 controllability 冲突时，不应全场景一起修改；
- 应只作用于真正相关的 agent；
- FITWMAMS 已经有 `CausalInfluenceGraph`，本轮直接复用，不重建 causal discovery 网络。

### CounterScene, 2026
关键启示：

- “改谁”与“怎么改”同样重要；
- 最小局部干预后，让风险通过闭环交互自然传播，比粗暴制造碰撞更合理；
- 但 CounterScene 部分离线 conflict mining 使用 future GT，本项目在线 controller 严禁使用类似信息。

---

# 2. 本轮必须回答的四个研究问题

## RQ1：事实精度
**在不改变 Flow、Diffusion、HiQR 主干的前提下，新响应层是否仍能保持现有 factual reconstruction？**

不能只证明 `gate=0` 时数学上透传；必须在完整执行链上测 ADE/FDE/P95。

## RQ2：人类响应
**在 held-out highD 自然事件中，新模型是否比 original A2 更接近真实后车的响应过程？**

这里的“响应过程”至少包括：

- response latency；
- acceleration；
- absolute jerk；
- peak braking；
- gap；
- closing speed；
- TTC；
- recovery。

## RQ3：闭环学习的增量价值
**最终闭环 RL 是否比同结构的 supervised initialization 有明确改进？**

如果 final PPO 不能优于自己的 supervised checkpoint，则不能把“闭环学习”作为论文贡献。

## RQ4：数据外极端工况
**当 ADS 把世界带到 highD 支持域之外时，NPC 是否表现出方向正确、连续、有限、可恢复的响应，同时不把所有真实风险强行消除？**

这里不宣称“human-like”，只宣称：

- mechanism-consistent；
- physically executable；
- risk-preserving；
- support boundary explicit。

---

# 3. 非谈判原则（Codex 必须遵守）

## 3.1 不重训主世界模型

本轮冻结：

```text
Flow
Diffusion
HiQR
```

除非发现确定性的代码 bug，否则禁止修改其 checkpoint、训练协议和网络结构。

## 3.2 不修改正式发布配置

不得覆盖：

```text
hierarchical_world_model/config/release.yaml
```

新实验必须使用独立配置：

```text
hierarchical_world_model/config/human_response_a2.yaml
```

## 3.3 不把 synthetic 变成人类标签

synthetic brake/ramp/pulse：

```text
human_target = NONE
```

禁止：

```text
target_action = min(base, IDM)
```

并把它解释成“human supervision”。

## 3.4 不将 ADS–NPC collision 无条件设成负 reward

对于 safety testing world：

```text
ADS-NPC collision != simulator error by definition
```

如果 ADS 的行为本来就不可挽救，合理人类也可能发生碰撞。

因此：

- synthetic stress 中不能以“消灭所有 ego–NPC collision”为优化目标；
- collision 必须保存为 outcome；
- numerical invalid / NaN / impossible dynamics 可以处罚；
- unrelated NPC–NPC model artifacts 可以单独诊断，但不要把目标冲突全部训练掉。

## 3.5 在线 controller 禁止 future leakage

在线输入只能来自：

- 当前和历史已执行状态；
- previous executed actions；
- 当前 causal influence graph；
- 当前/历史 soft reference；
- frozen exogenous randomness；
- controller internal state。

禁止输入：

- logged/synthetic/event 类型；
- intervention label；
- future GT trajectory；
- future ego action；
- event peak brake（尚未发生时）；
- source file / split label。

必须继续满足：

```text
same (H_t, K_ref, z, controller_state)
=> same p(a_NPC,t)
```

## 3.6 validation 通过前禁止碰 test

开发和超参数决策只使用：

```text
train
validation
```

正式 `test` 仅允许在 validation gate 全通过后运行一次。

---

# 4. 当前代码中本轮必须先修正的问题

以下是基于当前提交的已知问题，Codex 必须先修，不要直接开始 700 PPO updates。

## 4.1 synthetic 仍存在 IDM absolute pseudolabel

当前：

```python
def synthetic_safety_supervision(...):
    target = torch.minimum(base, rule_action)
```

问题：

- 这实际上把 IDM absolute action 当成 synthetic 的行为目标；
- 它与“IDM 只提供机制参考，不是 human truth”的研究语义冲突。

### 必须修改

移除：

```text
synthetic_safety_supervision -> absolute IDM target
```

替换为“paired mechanism difference”监督，见 §11。

## 4.2 supervised pretrain 训练的不是最终执行动作

当前 `pretrain_final_action()`：

1. 从 policy mean 得到 `desired`；
2. 对 `desired` 与 target 做 loss；
3. online rollout 最后却还经过 guard / jerk / execution mapping。

结果：

```text
training object != evaluation object
```

### 必须修改

所有 human supervision 必须作用于：

```text
final executed acceleration
```

若 execution mapping 可微，则统一调用同一个 tensor 函数。

如果某部分 HighwayEnv 本身不可微：

- 把 policy-to-final-action 的动作映射抽成共享 torch 函数；
- 训练和 online execution 都调用它；
- 不需要反向传播整个 HighwayEnv。

## 4.3 `pretrain_epochs` 的语义实际上只是少量 optimizer steps

当前每个 `pretrain_epoch`：

- sample 一批 episode；
- rollout；
- aggregate loss；
- `optimizer.step()` 一次。

因此：

```text
pretrain_epochs = 3
```

本质接近 3 次参数更新，而不是 3 个完整 epoch。

### 必须修改

改名为：

```text
pretrain_updates
```

并使用真正的 minibatch / cached supervision。

首轮固定：

```yaml
pretrain_updates: 400
```

如 VRAM 不足，可用 gradient accumulation，但**有效 update 数不能偷偷变化**。

## 4.4 新候选不能继续依赖行为型 TTC guard

当前多个 controller 后存在 TTC-based safety guard。

对于 safety-testing world，这个 guard 会产生两个问题：

1. policy 没学会，人为 guard 仍替它制动；
2. 会系统性消掉原本应保留的 ADS-induced risk。

### 新候选要求

`human_response_a2` 采用：

```text
physical-only execution guard
```

只保留：

- acceleration bound；
- yaw-rate bound；
- jerk continuity；
- numerical validity。

默认关闭：

```text
TTC-based forced braking rewrite
```

原 A2 baseline 保持旧逻辑不变，不能改旧 baseline 的语义。

评测必须报告：

```text
guard_rewrite_rate
```

新候选理论目标应接近 0（除了 physical clipping/jerk）。

---

# 5. 新候选控制器：Human-Response-Calibrated A2

新增 controller mode：

```text
human_response_a2
```

建议放在：

```text
hierarchical_world_model/src/reaction_controller.py
```

不要删除旧 A2。

## 5.1 动作分解

记：

```math
a^H_{i,t} = HiQR base action
```

```math
a^M_{i,t} = IDM mechanistic reference
```

定义：

```math
d^M_{i,t} = a^M_{i,t} - a^H_{i,t}
```

policy 输出：

```math
g_{i,t} \in [0,1]
```

```math
\lambda_{i,t} \in [0,\lambda_{max}]
```

```math
r_{i,t} \in [-r_{max}, r_{max}]
```

执行前动作：

```math
a^{pre}_{i,t}
=
a^H_{i,t}
+
I_{i,t} g_{i,t}
\left[
\lambda_{i,t} d^M_{i,t}
+
r_{i,t}
\right]
```

最终：

```math
a^{exec}_{i,t} = G_{physical}(a^{pre}_{i,t})
```

其中：

```text
I_i,t = CausalInfluenceGraph 的局部作用域
```

### 解释

- HiQR：事实行为锚点；
- IDM difference：当前交互相对 HiQR 的机制变化方向；
- `lambda`：机制参考的适用强度；
- residual `r`：数据驱动地修正规则模型不能解释的部分；
- `g`：是否启动响应；
- influence graph：决定谁有资格响应。

## 5.2 必须满足的结构性质

当：

```text
I = 0
```

或：

```text
g = 0
```

时：

```text
a_pre == a_HiQR
```

并且 physical-only execution 不应额外改写正常 HiQR。

写单元测试：

```text
test_zero_response_exact_base_passthrough
```

误差标准：

```text
max_abs_error <= 1e-6
```

## 5.3 取消 brake-only 支持限制

新候选不能使用 A2 中：

```python
nominal_upper = min(base, rule_action)
```

作为普通跟驰的硬支持。

原因：

- 人类恢复阶段可能需要正 residual；
- rule model 可能过度制动；
- 如果真实动作不在 action support 中，任何 human loss 都学不出来。

新候选应允许：

```text
signed correction
```

但动作范围仍受 physical guard 限制。

---

# 6. 驾驶员参数化：作为“有门槛的辅助模块”，不是第二条主线

当前 `rule_models.py` 是 single global IDM。

不要一开始就训练大型 personalized driver network。

## 6.1 首轮只考虑两个个体参数

候选参数：

```text
desired_headway_s (T)
comfortable_brake_mps2 (b)
```

其他 IDM 参数保持 global。

原因：

- highD 高速公路跟驰对 T、b 相对更有解释价值；
- 避免把不可辨识的六个参数一起学习；
- 减少 residual 与 rule parameter 的可替代性。

## 6.2 先做 identifiability / utility gate

新增离线脚本：

```text
hierarchical_world_model/scripts/audit_driver_parameterization.py
```

比较：

```text
global IDM
vs.
2-parameter personalized IDM
```

只使用 train 拟合，validation 检验。

### personalization 的实现建议

使用 pre-event / ordinary-following history：

```text
history -> small posterior network -> delta_T, delta_b
```

约束在物理范围内。

禁止使用事件后未来来预测参数。

## 6.3 是否进入正式候选的门槛

只有同时满足：

```text
validation acceleration MAE 改善 >= 5%
```

且：

```text
natural-event initial response Energy Score 不变差
```

才把 personalized IDM 放入最终 `human_response_a2`。

否则：

```text
继续使用 global IDM
```

并记录：

```text
personalization rejected by validation gate
```

本轮不继续调第三个、第四个参数。

---

# 7. 构建 Event-Conditioned Human Response Reference

这是本轮最重要的数据侧新增。

建议新增：

```text
hierarchical_world_model/src/human_response_prior.py
hierarchical_world_model/scripts/prepare_human_response_prior.py
```

## 7.1 扩充现有事件字段

在 `reaction_evidence.py` 的现有 event artifact 上增加离线字段：

### 只用于事件描述/分层，不进入 online controller

```text
leader_peak_brake_mps2
leader_brake_dose_mps
leader_brake_ramp_mps3
leader_brake_duration_s
```

### follower response 描述

```text
response_latency_s
follower_peak_brake_mps2
follower_peak_abs_jerk_mps3
follower_brake_dose_mps
recovery_time_s
```

### 起始/实时条件

```text
gap_m
closing_mps
ttc_s
follower_previous_accel_mps2
follower_speed_mps
leader_current_accel_mps2
```

注意：

```text
peak brake / future duration
```

可以用于离线事件索引和训练 reference construction，

但严禁作为 online controller 输入。

## 7.2 修正当前“brake strength cell”的解释

当前 cell 使用 onset crossing 时的 leader acceleration。

新增：

```text
onset_strength
peak_strength
brake_dose
```

三者必须分别保存。

不要再用 onset cell 推断整个数据集中没有强制动。

## 7.3 建立条件人类参考库

对每个 train event 构建 stimulus descriptor：

```text
x_event = [
  gap,
  closing,
  ttc,
  follower_previous_accel,
  follower_speed,
  leader_onset_accel,
  leader_short_horizon_brake_dose
]
```

其中 `leader_short_horizon_brake_dose` 只能来自已经发生的短刺激窗口，例如 onset 后 0.4 s。

该 descriptor 仅用于**训练/评测 reference selection**，不进入在线 policy。

所有维度使用 train split 的 robust scale：

```text
median + IQR
```

## 7.4 kNN 条件人类响应参考

对于一个 target event / synthetic condition：

1. 在 train events 中计算 standardized distance；
2. 排除同一：
   - recording；
   - leader/follower pair；
   - event key；
3. 取最近：

```yaml
human_reference_neighbors: 16
```

4. 至少来自：

```yaml
human_reference_min_recordings: 5
```

否则标记：

```text
unsupported
```

## 7.5 数据支持度

定义：

```text
rho(x) in [0,1]
```

首版不要训练额外 OOD 网络。

建议根据：

```text
kNN distance + unique recording count
```

构造 deterministic support score。

至少输出：

```text
supported / weak_support / unsupported
```

正式 human reward：

```text
只在 supported 区域启用
```

---

# 8. 人类响应真实性：从“单一轨迹误差”升级到条件时序分布

## 8.1 保留现有 held-out Event Energy Score 作为最终主指标

最终 validation/test 仍然使用真实 event 的实际 observed trajectory：

```text
acceleration + abs jerk sequence
```

计算 event-level Energy Score。

不要修改 test 主指标，以保证与已有实验可比较。

## 8.2 新增训练用 Conditional Human Energy Distance

训练时，对一个 stimulus condition：

模型生成：

```math
Y_1,\dots,Y_K
```

kNN human reference 提供：

```math
H_1,\dots,H_M
```

其中每个序列包含：

```text
acceleration
abs jerk
```

可选在训练中加入：

```text
closing
gap
```

但首版建议仍使用 acceleration + abs jerk，避免换指标太多。

定义 two-sample energy distance：

```math
ED =
\frac{2}{KM}\sum_{k,m}\|Y_k-H_m\|
-
\frac{1}{K^2}\sum_{k,l}\|Y_k-Y_l\|
-
\frac{1}{M^2}\sum_{m,n}\|H_m-H_n\|
```

目标：

```text
minimize ED
```

优势：

- 不把一条 logged future 当作唯一正确答案；
- 同时约束 accuracy 和 diversity；
- 保留整个响应时间过程；
- 比总体 acceleration histogram 更贴近 safety-response 问题。

---

# 9. 将 Energy Distance 变成闭环 PPO rollout reward

借鉴 RLFTSim 的 leave-one-out 思想，但本轮不复制 RMM/MLOO。

训练每个 event group：

```yaml
train_event_groups_per_update: 8
train_futures_per_event: 8
```

即每个 event 生成：

```text
K = 8
```

条 closed-loop stochastic futures。

## 9.1 Leave-One-Out Human Response Reward

定义：

```math
S = -ED
```

越大越好。

去掉第 k 条 rollout 后：

```math
S_{-k}
```

定义：

```math
R^{human}_k
=
\frac{1}{K}\sum_j S_{-j}
-
S_{-k}
```

解释：

- 如果去掉 rollout k 以后 reference alignment 变差，则 k 对人类分布有正贡献；
- 如果去掉它以后 alignment 变好，则 k 是不自然样本，得到负 reward。

## 9.2 计算必须避免 O(K^3)

实现时先计算：

```text
model-human distance matrix [K,M]
model-model distance matrix [K,K]
human-human constant [M,M]
```

通过 sum cache 计算所有 leave-one-out ED。

复杂度：

```text
O(K*M + K^2 + M^2)
```

不是 K 次重算全部距离。

新增测试：

```text
test_loo_energy_reward_matches_naive_implementation
```

## 9.3 PPO credit assignment

event trajectory 的 human reward 是 sequence-level。

首版：

- 只把该 reward 分配给 event follower；
- 只覆盖 event onset 到 `EVALUATION_FRAMES` 的 policy steps；
- 其他 agent reward = 0；
- GAE/PPO 使用已有 factorized per-agent buffer。

不要做 scene-level reward，避免 credit assignment 再次变复杂。

---

# 10. 三流训练协议

最终训练只保留三种 stream。

## 10.1 Stream A：真实自然 event

来源：

```text
highD train event reference
```

目标：

```text
human response distribution calibration
```

训练信号：

```text
Conditional Human Energy Distance LOO reward
+
small execution-action supervised anchor
```

supervised anchor 只在真实 history 对应的时刻使用，并且监督：

```text
final executed action
```

不是 `desired_action`。

## 10.2 Stream B：matched non-event

来源：

```text
与 event 起始 gap/closing/previous acceleration 匹配的普通跟驰
```

目标：

```text
不该响应时不要乱改 HiQR
```

损失：

```math
L_{anchor}
=
|a^{exec} - a^{human}|
```

以及：

```text
small correction preference
```

但不要强制 residual 永远为 0；真实人类动作优先。

## 10.3 Stream C：synthetic stress

来源：

```text
-2, -4, -6, -8 m/s²
ramp
pulse
```

目标：

```text
学习闭环响应能力与数据外机制外推
```

明确：

```text
NO human label
NO imitation loss
NO absolute IDM target
NO universal collision-avoidance reward
```

---

# 11. Synthetic OOD：Paired Mechanism-Difference Training

这是替换当前 `synthetic_safety_supervision()` 的核心。

## 11.1 生成成对世界

对每个 synthetic source scene，生成两个分支：

### Baseline branch

```text
logged/nominal ego action
```

### Intervention branch

```text
same initial state
same world exogenous randomness
same policy innovation stream
+ brake/ramp/pulse intervention
```

必须使用 common random numbers。

## 11.2 比较 IDM 的“变化”，不是 IDM 的绝对值

计算：

```math
\Delta a^{IDM}_t
=
a^{IDM,intervention}_t
-
a^{IDM,baseline}_t
```

模型：

```math
\Delta a^{model}_t
=
a^{exec,intervention}_t
-
a^{exec,baseline}_t
```

## 11.3 Mechanism loss

首版使用两个分量。

### Direction loss

当：

```text
|delta_IDM| > 0.25 m/s²
```

时：

```math
L_{dir}
=
\max(
0,
-\operatorname{sign}(\Delta a^{IDM})
\Delta a^{model}
)
```

### Soft magnitude loss

```math
L_{mag}
=
Huber(
clip(\Delta a^{model}),
clip(\Delta a^{IDM})
)
```

但 magnitude 权重必须小：

```yaml
mechanism_direction_weight: 1.0
mechanism_magnitude_weight: 0.1
```

含义：

> IDM 提供 response structure，而不是人类动作真值。

## 11.4 Synthetic 不奖励“永不碰撞”

删除当前 synthetic 主 reward 中的：

```text
generic collision penalty
generic TTC-to-safe reward
```

替换为：

```text
mechanism response
jerk
action bound
numerical validity
recovery
```

collision：

```text
record only
```

训练可在首次 collision 后结束 rollout 以避免 post-impact dynamics 污染，但：

```text
termination != negative human reward
```

必须在 artifact 中记录：

```text
collision_terminated: true/false
collision_penalty_applied: false
```

---

# 12. Recovery 的定义

不能再用“追上原 schedule”定义恢复。

继续沿用：

```text
longitudinal reference offset weight = 1
lateral = 0
```

即：

- 交互造成的 longitudinal delay 保留；
- 不要求回到原 x(t)；
- 风险解除后允许速度逐渐恢复；
- 不允许出现为追赶 original schedule 的异常加速。

训练/评测新增：

```text
recovery_to_local_speed
schedule_chasing_rate
post_risk_positive_rebound_rate
```

---

# 13. 建议配置

新增：

```text
hierarchical_world_model/config/human_response_a2.yaml
```

首轮固定建议：

```yaml
base_config: hierarchical_world_model/config/release.yaml

method:
  controller: human_response_a2
  physical_only_guard: true
  max_mechanism_scale: 1.5
  max_signed_residual_mps2: 4.0

human_reference:
  neighbors: 16
  minimum_recordings: 5
  support_knn_k: 16

training:
  device: auto
  seed: 20260909

  pretrain_updates: 400
  pretrain_minibatch_size: 2048

  event_groups_per_update: 8
  event_futures: 8
  non_event_episodes: 8
  synthetic_pairs: 8

  rollout_steps: 149

  max_ppo_updates: 120
  validation_interval_updates: 10
  validation_patience: 4

  learning_rate: 0.0003
  clip_ratio: 0.2
  gamma: 0.99
  gae_lambda: 0.95
  entropy_coefficient: 0.005
  value_coefficient: 0.5
  max_grad_norm: 1.0

  mechanism_direction_weight: 1.0
  mechanism_magnitude_weight: 0.1
  non_event_anchor_weight: 1.0

evaluation:
  validation_futures: 32
  factual_relative_tolerance: 0.05
```

如果显存不够：

- 可以减少物理 batch；
- 使用 gradient accumulation；
- **不能增加 max PPO updates 作为补偿**；
- 有效训练样本数必须写入 summary。

---

# 14. GPU / 环境要求

所有训练和 GPU 评测开始前：

```bash
conda activate tread
```

然后执行：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
```

并保存：

```bash
nvidia-smi
```

输出到：

```text
results/.../environment.txt
```

必须同时保存：

```bash
git rev-parse HEAD
git status --short
conda env export --from-history
```

不要因为本机 GPU 型号不同改变算法语义。

---

# 15. Codex 具体代码修改清单

## 15.1 必改文件

### `hierarchical_world_model/src/reaction_controller.py`

新增：

```text
HumanResponseA2Controller
```

实现：

- HiQR anchor；
- IDM difference；
- signed residual；
- gate；
- mechanism scale；
- physical-only execution；
- diagnostics。

新增 diagnostics：

```text
mechanism_delta
residual_delta
gate
mechanism_scale
pre_guard_action
final_action
physical_rewrite
```

旧 A2 不改。

### `hierarchical_world_model/src/reaction_training.py`

修改：

- 删除 synthetic absolute IDM target；
- pretrain 使用 final executed action；
- `pretrain_epochs -> pretrain_updates`；
- event group stochastic rollout；
- LOO Energy Distance reward；
- synthetic paired baseline/intervention；
- synthetic collision 不作为 universal negative reward；
- 保留 PPO infrastructure。

### `hierarchical_world_model/src/reaction_evidence.py`

增加：

- peak brake；
- brake dose；
- ramp；
- latency；
- recovery；
- driver/event descriptor；
- 明确区分 online-safe fields 与 offline-only fields。

### `hierarchical_world_model/src/rule_models.py`

只做：

- 提供参数化 IDM 的接口；
- 不删除 global bundle；
- personalization 受 §6 gate 控制。

### `hierarchical_world_model/scripts/evaluate_reaction_policy.py`

新增：

- `human_response_a2` arm；
- conditional human energy distance；
- support-stratified metrics；
- physical rewrite；
- mechanism direction agreement；
- collision 只作为 outcome；
- original A2 / frozen HiQR / new supervised / new final 的同协议比较。

### `hierarchical_world_model/scripts/validate_reaction_policy.py`

不要覆盖旧 validator。

建议新增：

```text
validate_human_response_a2.py
```

避免改变旧 candidate 的历史语义。

## 15.2 新增文件

```text
hierarchical_world_model/src/human_response_prior.py
hierarchical_world_model/scripts/prepare_human_response_prior.py
hierarchical_world_model/scripts/train_human_response_a2.py
hierarchical_world_model/scripts/evaluate_human_response_a2.py
hierarchical_world_model/scripts/validate_human_response_a2.py
hierarchical_world_model/scripts/audit_driver_parameterization.py
hierarchical_world_model/config/human_response_a2.yaml
```

---

# 16. 必须新增的单元测试

至少：

```text
test_zero_response_exact_base_passthrough
test_physical_guard_does_not_apply_ttc_behavior_rewrite
test_online_policy_has_no_event_or_intervention_label
test_online_policy_has_no_future_gt
test_human_reference_excludes_same_recording
test_human_reference_support_is_train_defined
test_loo_energy_reward_matches_naive
test_loo_energy_reward_prefers_helpful_rollout
test_synthetic_has_no_human_target
test_synthetic_uses_paired_crn
test_synthetic_collision_has_no_human_penalty
test_mechanism_difference_zero_when_no_intervention
test_pretrain_updates_are_real_optimizer_updates
test_training_and_execution_share_final_action_mapping
```

所有旧测试必须继续通过。

---

# 17. 实验矩阵：只保留 5 个 arm

正式 validation/test 统一比较：

| Arm | 含义 |
|---|---|
| `frozen_hiqr` | 高精度事实基座 |
| `idm_only` | 纯规则响应参考 |
| `a2_transfer` | 当前旧 A2 同协议 baseline |
| `human_response_a2_supervised` | 新结构，未做 closed-loop PPO |
| `human_response_a2` | 最终候选 |

不重新加入：

```text
A1
A3
GAIL variants
MLOO variants
convex optimizer
new diffusion controller
```

这些只在论文 related/history 中说明。

---

# 18. Validation 验收标准

## Gate A：数据与协议

必须全部通过：

```text
recording split isolated
train-defined support only
no future GT in online context
no event/synthetic/source labels in policy
same random state -> deterministic reproducibility contract
```

任意失败：

```text
STOP
```

## Gate B：factual non-inferiority

继续使用仓库已经冻结的严格标准，不事后放宽：

候选相对于 `frozen_hiqr`：

```text
ADE absolute degradation <= 0.02 m
FDE absolute degradation <= 0.06 m
P95 absolute degradation <= 0.10 m
```

并同时满足：

```text
relative degradation <= 5%
```

任一失败：

```text
STOP
```

不增加 PPO budget 继续刷。

## Gate C：真实人类事件响应

在 held-out validation 自然事件上，32 futures/event。

主指标：

```text
event-level acceleration + abs-jerk Energy Score
```

要求：

### 新 final vs original A2

定义：

```text
A2 EnergyScore - New EnergyScore
```

recording-cluster bootstrap：

```text
LCB95 > 0
```

### 新 final vs 自己的 supervised initialization

定义：

```text
Supervised EnergyScore - Final EnergyScore
```

要求：

```text
LCB95 > 0
```

否则说明：

```text
closed-loop PPO 没有贡献
```

不得把 PPO 写成主要创新。

## Gate D：响应过程 diagnostics

对以下指标：

```text
latency
peak acceleration
jerk
gap
closing
recovery
```

沿用现有原则：

```text
paired degradation CI95 high
<= 10% of train IQR
```

不能只靠综合 Energy Score 掩盖一个关键行为变差。

## Gate E：conditional human distribution

新增指标：

```text
conditional energy distance
```

新 final vs original A2：

```text
paired improvement LCB95 > 0
```

并按：

```text
supported
weak support
unsupported
```

分层报告。

只对 supported 宣称 human-distribution improvement。

## Gate F：matched non-event

必须满足：

- factual gate 已通过；
- `|final-base|` 的均值和 P95 不高于 supervised initialization 的 10%；
- 不出现大范围无事件 correction。

如果 human response 提高，但 non-event 普遍乱改：

```text
STOP
```

## Gate G：synthetic OOD mechanism

不要求 collision=0。

要求：

```text
finite states/actions = 100%
action bound violation = 0
```

jerk：

```text
no configured physical jerk violation
```

当：

```text
|delta_IDM| > 0.25
```

时，direct same-lane follower 的：

```text
mechanism direction agreement >= 95%
```

同时报告：

```text
collision rate
rear collision rate
minimum gap
TTC
response latency
peak brake
recovery
positive rebound
schedule chasing
```

collision 不作为 human-likeness fail gate。

## Gate H：guard dependence

新候选：

```text
TTC behavior guard must be disabled
```

physical rewrite 只能来自：

```text
action bound
jerk bound
numerical clipping
```

若候选的主要安全表现来自隐藏 guard rewrite：

```text
STOP
```

---

# 19. Training stop rule

不要无限增加训练预算。

最大：

```yaml
max_ppo_updates: 120
```

每：

```text
10 updates
```

跑 validation objective check。

如果连续：

```text
4 次
```

没有 primary human response improvement：

```text
early stop
```

如果 supervised checkpoint 已经失败 factual gate：

```text
不进入 PPO
```

如果 PPO final 未优于 supervised：

```text
保留 supervised 作为研究结果
```

但不能追加新一轮 700/1400 update。

---

# 20. Test 执行规则

只有 validation 全 gate 通过：

```text
A-H
```

才运行一次 test。

test 完成后：

- 不根据 test 修改 threshold；
- 不根据 test 重新训练；
- 不根据 test 选择 checkpoint。

输出：

```text
final_test_report.json
final_acceptance.json
decision.md
```

---

# 21. 风险测试：本轮只做 post-acceptance diagnostic，不立即重跑 AMS

只有最终 candidate 通过 human response acceptance 后：

1. 冻结 candidate；
2. 接入现有 `execution.py`；
3. 使用相同 `WorldExogenousState`；
4. `same_rear` 必须包含；
5. 做小规模 paired MC diagnostic。

首轮：

```yaml
paired_mc_samples: 2000
```

比较：

```text
frozen_hiqr
original A2
human_response_a2
```

目的只回答：

> 背景响应模型变化对 ADS 风险结论有多大敏感性？

本轮不自动启动 AMS。

只有该 diagnostic 合理后，再单独规划最终 MC/AMS。

---

# 22. 结果目录规范

本轮输出新目录，例如：

```text
results/hierarchical_world_model/causal_reaction/
  human_response_a2_round1_20260909/
```

必须包含：

```text
environment.txt
git_state.txt
config.yaml
data_audit.json
human_reference_audit.json
driver_parameterization_audit.json
controllers/
  supervised.pt
  final.pt
training/
  supervised_summary.json
  ppo_history.json
validation/
  report.json
  acceptance.json
test/
  report.json
  acceptance.json
ood/
  paired_mechanism_report.json
risk_diagnostic/
  paired_mc.json
decision.md
```

任何 failure 都必须保留 artifact，不删除失败结果。

---

# 23. 最终论文能够 claim 什么

如果全部通过，可以主张：

> 在冻结的高精度分层交通行为基座上，本文通过事件条件化的人类响应分布对局部 NPC 响应策略进行闭环校准，使模型在自然驾驶支持域内保持事实重建非劣性并改善真实响应过程的统计一致性；对于自然数据未覆盖的安全关键状态，模型采用显式标识的数据支持边界和机制差分约束进行有限外推，而不将合成压力场景伪装成人类反事实标签。

可以主张：

```text
human-response calibrated in supported domain
mechanism-consistent in unsupported stress domain
risk-preserving closed-loop simulator
```

---

# 24. 最终论文不能 claim 什么

即使全部通过，也禁止写：

```text
我们恢复了真实人类在任意 ADS 干预下唯一正确的反事实行为
```

禁止写：

```text
synthetic -8 m/s² response 已经被证明等于真实人类响应
```

禁止写：

```text
碰撞率更低，因此 NPC 更真实
```

禁止写：

```text
AMS sampling CI 缩小意味着 real-world model bias 消失
```

禁止将 conditional K_GT reconstruction 说成 unconditional prediction。

---

# 25. Codex 执行顺序

严格按以下顺序。

## Phase 0 — Environment + freeze

```bash
conda activate tread
git rev-parse HEAD
git status --short
nvidia-smi
pytest ...
```

确认：

```text
HEAD = 889e57c...
```

若不一致，先记录，不擅自 reset 用户工作。

## Phase 1 — 数据审计

先实现：

- peak/dose/ramp；
- human reference；
- support；
- driver parameter audit。

先产出：

```text
data_audit.json
human_reference_audit.json
```

未确认数据支持范围前，不开始 PPO。

## Phase 2 — Controller + execution semantics

实现：

```text
human_response_a2
physical-only guard
exact base passthrough
```

所有结构测试通过后再训练。

## Phase 3 — Supervised initialization

修复：

```text
real pretrain updates
final executed action supervision
```

只训练：

```text
events + non-events
```

先跑 validation。

若 factual 或 basic human response 明显失败：

```text
STOP
```

## Phase 4 — Closed-loop human alignment

实现：

```text
conditional energy distance
LOO rollout reward
```

在 natural events 上训练。

## Phase 5 — Synthetic paired mechanism

实现：

```text
baseline/intervention CRN pair
IDM delta
model delta
mechanism reward
```

确认：

```text
no synthetic human label
no universal collision penalty
```

## Phase 6 — Fixed-budget PPO

最多：

```text
120 updates
```

early stopping。

## Phase 7 — Validation acceptance

只有全 gate 通过才继续。

## Phase 8 — One-shot test

只跑一次。

## Phase 9 — Paired risk diagnostic

只在 candidate 被正式接受后执行。

---

# 26. Codex 完工检查表

完成前逐项确认：

- [ ] `conda activate tread` 环境已使用并保存 GPU 信息
- [ ] 旧 Flow/Diffusion/HiQR checkpoint 未修改
- [ ] `release.yaml` 未修改
- [ ] 旧 A2 controller 语义未修改
- [ ] 新 `human_response_a2` 为独立 controller
- [ ] synthetic absolute IDM pseudolabel 已删除
- [ ] synthetic 不存在 human target
- [ ] synthetic ego–NPC collision 不作为 universal negative reward
- [ ] human supervision 针对 final executed action
- [ ] TTC behavior guard 对新候选关闭
- [ ] event support 只由 train 定义
- [ ] same recording 不进入自己的 kNN human reference
- [ ] no future GT / intervention label 进入 online policy
- [ ] LOO energy reward 与 naive 实现数值一致
- [ ] paired synthetic 使用 CRN
- [ ] factual non-inferiority gate 未放宽
- [ ] PPO 必须与自己的 supervised checkpoint 比较
- [ ] validation 未通过前未运行 test
- [ ] test 后未重新调参
- [ ] AMS 未被自动重跑
- [ ] 所有失败工件保留

---

# 27. 最终的研究判断标准

本轮不是以：

```text
“训练跑完了”
```

作为成功。

而是必须满足：

```math
\boxed{
\text{Factual Fidelity}
\land
\text{Human Response Alignment}
\land
\text{Closed-loop Increment}
\land
\text{OOD Mechanism Validity}
\land
\text{Risk Preservation}
}
```

如果其中任意一项失败：

- 记录 failure；
- 解释 failure；
- 不开启新的大模型搜索。

本轮最重要的科学判断是：

> **A2 是否能够从“一个会响应的 residual controller”，升级成“在有数据处有证据地像人响应、在无数据处有边界地进行机制外推，并且不会替 ADS 人为消除风险”的 traffic behavior world model。**

这就是下一轮实验唯一需要回答的问题。

---

# 28. 给 Codex 的启动提示词

可直接将下面内容交给 Codex：

```text
请在 SafeDL/FITWMAMS 上执行
“Human-Response-Calibrated A2：事实锚定、事件条件化的人类响应闭环校准”
这一轮唯一主实验。

基线提交为：
889e57c74f7a7a543ef8ec8bebb59f1e34e22dd0

GPU 环境必须先：
conda activate tread

严格按照本任务书 Phase 0 -> Phase 9 顺序执行。

硬约束：
1. 不重训 Flow/Diffusion/HiQR；
2. 不修改 release.yaml；
3. 不修改旧 A2 的历史语义；
4. 新建 human_response_a2；
5. 删除 synthetic 的 absolute IDM action pseudolabel；
6. human supervision 必须落在 final executed action；
7. synthetic 不得有 human target；
8. synthetic ADS-NPC collision 不得作为 universal negative reward；
9. 新 candidate 禁止 TTC-based behavior guard，只保留 physical execution constraints；
10. validation 全通过前禁止跑 test；
11. test 只允许一次；
12. 最多 120 PPO updates，不因失败自动扩大预算；
13. 不重新启动 GAIL/MLOO/新 diffusion controller/全新架构搜索；
14. 所有失败结果必须保留。

先完成代码审计和单元测试，再进行任何 GPU 长训练。
每一个 Phase 完成后都写 machine-readable JSON summary。
最终给出 decision.md，明确：
accepted / rejected，
失败 gate，
是否允许进入 paired risk diagnostic。
```

---

# 29. 设计依据文献

本任务书主要依据用户提供的以下全文材料进行设计：

1. **TrafficSim: Learning to Simulate Realistic Multi-Agent Behaviors**, CVPR 2021.
2. **Generating Useful Accident-Prone Driving Scenarios via a Learned Traffic Prior (STRIVE)**, CVPR 2022.
3. **Learning Realistic Traffic Agents in Closed-loop (RTR)**, CoRL 2023.
4. **Improving Agent Behaviors with RL Fine-tuning for Autonomous Driving**, ECCV 2024.
5. **Causal Composition Diffusion Model for Closed-loop Traffic Generation (CCDiff)**, 2025.
6. **RLFTSim: Realistic and Controllable Multi-Agent Traffic Simulation via Reinforcement Learning Fine-Tuning**, 2026.
7. **CounterScene: Counterfactual Causal Reasoning in Generative World Models for Safety-Critical Closed-Loop Evaluation**, 2026.

本任务书没有把上述论文中的“安全奖励”“adversarial generation”或“realism metric”直接照搬到 FITWMAMS，而是根据**自动驾驶安全测试环境必须保留真实风险**这一不同目标重新划分它们的职责。

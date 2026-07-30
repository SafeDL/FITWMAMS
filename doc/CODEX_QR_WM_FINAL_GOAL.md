# Codex Goal：Query-Refine World Model（QR-WM）

> 适用模型：`query_refine_world_model`  
> 核心原则：QR-WM 只根据已经发生、可观测的交通状态生成背景交通行为，不直接读取 ADS 动作、ADS 内部规划、未来 ego 控制序列或未来 ego 状态。QR-WM 是独立于 RAMP/FIRM 的高维背景交通世界模型。它以固定六辆背景车槽位建模并输出多模态、闭环、受已观测 ego 状态和历史条件化的未来交通演化；不输入 traffic light、ADS 动作或 ego 未来序列，也不修改基线实现。

---

## 1. Codex 的总任务

在不改变 RAMP-WM、FIRM-WM 等基线语义的前提下，完成并验证 QR-WM，使其成为 FITWMAMS 中负责**背景交通动态演化**的状态空间交通世界模型。

最终系统必须能够：

1. 从冻结的 Normalizing Flow 接收带显式概率密度的长尾驾驶事件初始条件；
2. 正确恢复场景初始状态 `C0`、首秒背景动作摘要 `B0`、有效槽位、事件结构和概率元数据；
3. 仅在 START 阶段使用 `B0`，初始化背景行为潜变量、场景记忆和首个背景控制缓存；
4. 在 ROLL 阶段仅依据已经发生的交通历史、道路信息、场景记忆和背景控制缓存生成后续背景交通行为；
5. 与任意 ADS 在环境层闭环交互，但不把 ADS 动作或未来规划直接暴露给 QR-WM 网络；
6. 联合生成六辆背景车辆的未来控制，并保持多车辆交互一致性；
7. 通过行为潜变量支持确定性和可复现的随机多模态 rollout；
8. 支持中间状态快照、恢复和 path-level 分支；
9. 保留从 Flow 初始采样到世界模型轨迹的完整概率与运行审计信息；
10. 通过 Flow START、长尾事件保真、闭环响应、随机分支和高保真复核等协议验收。

系统职责链如下：
```text
START: C0 + B0 + map; ROLL: observed history + map
              |
relation-aware multi-head scene encoder
              |
persistent scene memory + CVAE behavior prior
              |
joint agent-time background future-action refiner
              |
背景车辆未来动作序列 [H, 6, 2]
              |
differentiable closed-loop traffic dynamics
```

---

## 2. 研究定位与因果边界

### 2.1 QR-WM 建模什么

QR-WM 建模背景交通的条件演化：

\[
p_{\theta}\!\left(
U_{t:t+H}^{bg},m_{t+\Delta},U_{t+\Delta}^{buf}
\mid
H_t,S_t,M,m_t,U_t^{buf},z
\right).
\]

其中：

- \(H_t\)：截至当前时刻已经发生的交通历史；
- \(S_t\)：当前所有车辆的可观测状态；
- \(M\)：道路 polyline 和 lane topology；
- \(m_t\)：唯一的 persistent scene memory；
- \(U_t^{buf}\)：尚未执行的背景控制缓存；
- \(z\)：背景交通行为潜变量。

QR-WM 不直接建模 ADS 的决策过程，也不预测或修改 ego 动作。它负责回答：

> 在当前已观测交通状态和既往交互下，背景交通参与者接下来可能如何联合行动？

### 2.2 QR-WM 明确禁止读取的信息

网络前向过程中禁止输入：

- 当前 ADS 动作；
- ADS 的未来 ego 控制序列；
- ADS 的内部轨迹计划；
- 由 ADS 计划提前积分得到的未来 ego 状态；
- 真实未来 ego 轨迹；
- 真实未来背景车辆状态或控制。

即禁止使用：

\[
a_t^{ADS},\quad
U_{t:t+H}^{ego},\quad
S_{t:t+H}^{ego}.
\]

原因是这些信息在现实交互中对背景车辆并不可直接获得。将其输入网络会造成 ADS 私有意图泄漏、提前反应和不真实的因果关系。

### 2.3 正确的 ADS—背景交通闭环

在每个响应周期，ADS 与 QR-WM 同时读取当前观测：

\[
H_t\rightarrow
\begin{cases}
a_t^{ego}=\pi_{ADS}(H_t),\\
U_t^{bg}\sim p_{\theta}(\cdot\mid H_t,S_t,M,m_t,U_t^{buf},z).
\end{cases}
\]

随后环境同步执行两类控制：

\[
S_{t+\Delta}=f_{env}(S_t,a_t^{ego},U_t^{bg}).
\]

ADS 动作对背景交通的影响只能通过已经实现并进入下一历史窗口的 ego 运动体现：

\[
S_{t+\Delta}^{ego}\rightarrow H_{t+\Delta}\rightarrow U_{t+\Delta}^{bg}.
\]

因此，背景车辆不会在 ADS 行为尚未表现为可观测运动之前提前响应。这种有限反应延迟是正确的因果约束。

### 2.4 Normalizing Flow 的职责

冻结的 Flow 给出：

\[
(C_0,B_0,e)\sim q_{\phi}(C_0,B_0,e\mid\text{tail}),
\]

其中：

- \(C_0\)：场景初始条件；
- \(B_0\)：六辆背景车辆的首秒动作摘要；
- \(e\)：事件结构、有效槽位和主风险车辆等离散语义；
- \(\log q_{\phi}\)：初始测试条件的可审计对数密度。

Flow 只负责生成带显式密度的长尾事件起点；QR-WM 只负责从该起点继续生成背景交通动态。QR-WM 不重新估计或修改 Flow 初始密度。

### 2.5 “任意 ADS”的准确含义

QR-WM 网络与 ADS 动作接口解耦，因此环境层可以接入不同 ADS。正式支持范围为：

- ADS 根据状态、轨迹、地图或经过渲染的传感器观测独立决策；
- ADS 输出可被环境动力学执行的 ego 动作；
- QR-WM 只接收执行后形成的交通状态历史；
- 动作和运行时域位于已验证的环境动力学范围内。

对于视觉端到端 ADS，需要额外的传感器或图像渲染层：

```text
QR-WM state → sensor renderer → visual ADS → ego action → environment
```

不得把状态世界模型本身直接表述为完整视觉 ADS 测试平台。

---

## 3. 数据与 Flow START 合同

### 3.1 场景结构

固定车辆顺序：

```text
[ego, background slot 0, ..., background slot 5]
```

单车状态：

\[
[x,y,v_x,v_y,a_x,a_y].
\]

每个训练样本至少提供：

- `agent_states`；
- `agent_valid`；
- map polylines；
- map validity；
- lane graph edges；
- ego index；
- 与冻结 Flow schema 对齐的原始 `B0` sidecar；
- 长尾事件标签或 EVT-tail 标识。

不输入 traffic light。

### 3.2 Flow START schema

Flow 条件为：

\[
X_{Flow}=[C_0,B_0],
\]

其中：

- \(C_0\in\mathbb{R}^{40}\)；
- \(B_0\in\mathbb{R}^{6\times6}\)；
- 总维度为 76。

所有 QR-WM 解码必须复用项目共享 Flow START adapter，禁止在 QR 模块内维护第二套字段解释。

强制要求：

- 背景位置和速度按 Flow schema 的相对语义还原；
- 背景绝对速度使用
  \[
  v^{bg}=v^{ego}+\Delta v^{bg};
  \]
- `slot_valid[6]` 必填，不允许默认全部有效；
- primary slot 必须对应有效槽位；
- `mask_pattern` 必须与 `slot_valid` 一致；
- `log_prob = event_structure_log_prob + conditional_log_prob`；
- Flow schema hash 必须与 QR-WM checkpoint 中记录值一致；
- `C0`、`B0` 和所有概率字段必须为有限数值。

### 3.3 Flow 元数据对象

不得只传递匿名 76 维 tensor。运行对象必须同步保存：

- 原始 `C0+B0`；
- `slot_valid`；
- primary slot；
- event structure / event structure id；
- `event_structure_log_prob`；
- `conditional_log_prob`；
- `log_prob`；
- Flow schema hash；
- Flow checkpoint hash；
- Flow sampling seed；
- sampling temperature；
- rejection configuration；
- audit id。

这些信息必须伴随 rollout、snapshot、restore 和结果导出。

---

## 4. 必须实现的模型架构

### 4.1 START 与 ROLL 严格分离

#### START

START 仅执行一次：

\[
(C_0,B_0,M)\rightarrow(S_0,m_0,z,U_0^{buf}).
\]

START 必须：

- 使用共享 Flow adapter；
- 使用专用 current-state / START 编码路径；
- 不重复 \(C_0\) 构造伪历史；
- 使用 `B0` 初始化首秒背景行为锚点、behavior seed 和 scene memory；
- 完成后不再把原始 `B0` 作为 ROLL 网络输入。

#### ROLL

ROLL 只依赖已经发生的信息：

\[
(S_t,H_t,M,m_t,U_t^{buf},z)
\rightarrow
(U_t^{bg},m_{t+\Delta},U_{t+\Delta}^{buf}).
\]

ROLL 禁止：

- 读取真实未来背景状态或控制；
- 再次读取原始 `B0`；
- 读取 ADS 动作、ADS 未来规划或未来 ego 状态；
- 在 ADS 测试中读取真实 future ego trajectory。

### 4.2 Safety-aware relation/query scene encoder

ROLL 使用 temporal history encoder；START 使用专用 current-state encoder。两条入口共享：

- relation-aware multi-head agent attention；
- agent/map cross-attention；
- scene pooling。

关系特征至少包括：

- 相对位置；
- 相对速度；
- lane relation；
- closing rate；
- TTC；
- DRAC。

#### 无效槽位强制不变量

对车辆对 \((i,j)\)：

\[
M_{ij}=v_i\land v_j.
\]

关系聚合必须为：

\[
h_i^{rel}=
\frac{\sum_j M_{ij}f(r_{ij})}
{\max(1,\sum_j M_{ij})}.
\]

禁止对全部固定槽位直接求均值。改变任一无效槽位的填充值，不得改变：

- 有效车辆 token；
- scene embedding；
- behavior prior；
- background controls；
- rollout states。

### 4.3 单一 persistent scene memory

仅维护一个持续记忆：

\[
m_t=f(m_{t-1},H_t,S_t,U_{t-1}^{buf}).
\]

它保存：

- 已经发生的多车交互；
- 已执行和未执行的背景计划信息；
- 已经表现为可观测运动的 ego 行为影响；
- 未完全观测的背景行为状态。

不得重新增加语义重叠的 world memory 或 continuous memory。memory 更新不得接收 ADS 当前动作或未来计划。

### 4.4 CVAE behavior prior

训练期：

- prior 只读取当前可用条件；
- posterior 可以读取背景未来监督；
- 记录 KL、future-feature reconstruction、prior/posterior scale 和 collapse 指标。

推理期必须支持：

```python
deterministic=True          # 使用 prior mean
world_seed=...              # 可复现随机采样
behavior_latent=...         # 直接注入指定分支 latent
```

START 中，`B0` 只作为 prior mean 的一次性 behavior seed。

必须记录：

- behavior latent；
- prior mean/log-scale；
- latent 条件概率或可审计 prior 参数；
- RNG 状态。

QR-WM 的多模态随机性只由行为潜变量及显式环境随机变量提供，不通过额外扩散噪声制造。

### 4.5 Marginal proposal + deterministic joint agent-time refiner

首先构造背景边缘控制提议：

\[
U^{fresh}\in\mathbb{R}^{H\times6\times2}.
\]

随后联合精炼整个背景控制张量：

\[
U^{bg}\in\mathbb{R}^{H\times6\times2}.
\]

joint refiner 必须同时包含：

1. 同一车辆沿时间维的 attention；
2. 同一时刻不同背景车辆之间的 attention；
3. scene / memory / map cross-attention；
4. behavior latent conditioning；
5. validity、carried、appended 和 refinable masks。

refiner 不接收：

- ADS 动作；
- ego future controls；
- ego future states；
- diffusion timestep 或 noise-level embedding。

联合精炼采用确定性残差修正：

\[
U_t^{refined}
=
U_t^{pre}
+
R_{\theta}
\left(
U_t^{pre},H_t,S_t,M,m_t,z,\text{masks}
\right).
\]

允许执行 1–2 次共享参数的 refinement，但不得加入随机加噪、多步反向去噪或长扩散采样链。

### 4.6 不采用扩散式加噪去噪

从核心架构中删除：

- Gaussian corruption；
- noise schedule；
- diffusion timestep；
- noise-level embedding；
- denoising loss；
- 多步随机去噪推理；
- `noise-conditioned amortized refinement` 或完整 diffusion 的表述。

原因：

- CVAE behavior prior 已承担多模态生成；
- future buffer 已承担跨周期计划连续性；
- joint refiner 已承担多车协调和计划修正；
- 人工噪声没有明确的交通行为语义；
- 多步去噪增加长尾测试和 path-level 分支的计算成本；
- 当前没有证据证明去噪改善长尾事件覆盖或安全验证有效性。

训练阶段允许加入小幅控制扰动作为普通数据增强或鲁棒性正则化，但它不得成为生成过程、不得在推理阶段运行，也不得使用扩散术语。

### 4.7 `B0` 首秒背景行为锚定

`B0` 通过共享摘要投影逻辑得到：

\[
U^{B_0}\in\mathbb{R}^{H\times6\times2}.
\]

START 初始计划采用随时间衰减的凸组合：

\[
U_0^{pre}(h)=
(1-\alpha_h)U_0^{fresh}(h)
+\alpha_h U^{B_0}(h).
\]

禁止使用：

\[
U^{fresh}+\alpha U^{B_0}.
\]

\(\alpha_h\) 只能依赖 `C0`、`B0`、地图、背景行为 latent 和 START scene context，不得依赖 ADS 动作或未来 ego 计划。

生成首秒状态后重新计算：

\[
\widehat B_0=\Phi(S_{0:1s}^{generated}),
\]

并加入：

\[
\mathcal{L}_{B_0}=\|\widehat B_0-B_0\|_1.
\]

`B0` 定义的是初始长尾背景行为趋势，不是整个一秒内不可修改的刚性脚本。每次环境响应后，模型可依据新形成的交通历史修正尚未执行的背景计划。

### 4.8 Future background control buffer

背景 future buffer 表示尚未执行的背景控制：

\[
U_t^{buf}\in\mathbb{R}^{25\times6\times2}.
\]

它不表示未来状态或轨迹。

每个 0.2 s response：

1. 执行前五帧背景控制；
2. 左移其余未执行控制；
3. 追加五帧新背景控制；
4. 对有效、可精炼背景区域执行确定性联合 refinement；
5. 更新 scene memory 和交通 history。

必须维护：

- `executed`；
- `carried`；
- `appended`；
- `refinable`；
- `valid`。

历史观测和已执行控制不可修改。

### 4.9 动力学与同步执行

控制定义：

\[
U=[a,\dot\psi].
\]

环境在每个物理步同步执行：

\[
S_{t+1}^{ego}=f_{ego}(S_t^{ego},a_t^{ADS}),
\]

\[
S_{t+1}^{bg}=f_{bg}(S_t^{bg},U_t^{bg}).
\]

ADS 动作只进入环境动力学，不进入 QR-WM 网络。下一周期 QR-WM 通过更新后的 \(S_{t+1}\) 和 \(H_{t+1}\) 感知 ego 行为结果。

---

## 5. 在线环境合同

正式环境接口：

```python
env.reset_from_flow(
    C0,
    B0,
    metadata,
    world_seed=None,
    behavior_latent=None,
    deterministic=False,
)
observation = env.observe()
observation, reward_or_risk, terminated, truncated, info = env.step(ego_action)
snapshot = env.snapshot()
env.restore(snapshot)
```

关键边界：

> `env.step(ego_action)` 可以接收 ADS 动作，但 `ego_action` 只能用于环境推进 ego；调用 QR-WM 网络时必须从输入结构中剔除该动作以及由该动作产生的未来计划。

### 5.1 `reset_from_flow`

必须初始化并保存：

- `C0`；
- `B0`；
- decoded scene state；
- `slot_valid`；
- primary slot；
- event structure；
- Flow log probabilities；
- map；
- behavior latent；
- scene memory；
- initial background future buffer；
- Flow/QR checkpoint hash；
- RNG state；
- audit id。

### 5.2 `observe`

返回至少包括：

- 当前所有有效车辆状态；
- validity mask；
- map 或稳定 map reference；
- 当前仿真时间；
- terminated/truncated 状态；
- audit id。

ADS 可以使用这些观测独立决策。QR-WM 使用相同的已观测状态历史，但不得读取 ADS 内部输出。

### 5.3 `step`

每次接收 ego 动作，并完成：

1. 检查 ADS 动作 shape、NaN/Inf 和物理范围；
2. 在调用 QR-WM 前，根据当前历史独立生成背景控制；
3. 将 ADS ego 动作和 QR-WM 背景控制同步送入环境动力学；
4. 执行一个响应周期；
5. 更新状态、history、scene memory 和 background future buffer；
6. 计算风险、碰撞、越界和终止信息；
7. 返回实际执行的 ego 动作和 audit 信息。

禁止在第 2 步把当前 ADS 动作、未来 ego controls 或未来 ego states传入 QR-WM。

### 5.4 `snapshot/restore`

快照必须完整保存：

\[
\mathcal{X}_t=
\{S_t,H_t,m_t,z,U_t^{buf},M,RNG,metadata\}.
\]

至少包括：

- 当前状态；
- 历史窗口；
- scene memory；
- behavior latent；
- prior 参数；
- 未执行背景 control buffer；
- map；
- Flow metadata；
- world-model RNG state；
- response index 和 elapsed time；
- audit id。

ADS 内部状态由 ADS 测试框架单独保存，不属于 QR-WM snapshot。系统级 path branch 必须同时复制 ADS snapshot 与 QR-WM environment snapshot。

restore 后，在相同 ADS 行为和 RNG 下，轨迹必须逐位一致。

### 5.5 时域与终止

环境必须设置：

- `max_response_steps` 或最大已验证时域；
- collision termination；
- off-road termination；
- invalid-state termination；
- horizon truncation。

超过训练或验证时域时必须终止或显式标记为外推，不能静默无限滚动。

---

## 6. 训练协议

### 6.1 两种训练入口

#### START batch

输入：

\[
(C_0,B_0,M).
\]

START 不得使用真实历史，也不得伪造静态历史。监督包括：

- 首秒背景状态与控制；
- 完整后续背景轨迹；
- `B0` 摘要一致性；
- initial background buffer；
- 行为潜变量和联合背景计划。

#### ROLL batch

输入：

\[
(H_t,S_t,M,m_{t-1},U_{t-1}^{buf},z).
\]

只允许读取已经发生的信息。真实未来背景数据只作为 loss target。

默认 batch 组成：

- 50% START；
- 50% ROLL。

该比例可配置，但 checkpoint 选择必须优先服务 Flow START 部署模式。

### 6.2 Logged reconstruction 与闭环 ADS 测试分离

#### Logged reconstruction

- ego 可沿 highD logged trajectory 由环境重放；
- QR-WM 仍只读取已经发生的历史；
- 用于基础背景交通重建；
- 不得把该结果解释为反事实 ADS 响应。

#### Closed-loop ADS testing

- ego 由 ADS 和环境动力学推进；
- QR-WM 不读取 ADS 动作；
- 背景交通只根据后续观察到的 ego 运动进行响应；
- 禁止 highD future ego 覆盖；
- 评价反应方向、反应延迟、风险演化和长期稳定性。

### 6.3 关于干预数据

若要证明 QR-WM 在不同 ADS 引起的状态分布下仍然可靠，可加入多策略或干预轨迹数据。但训练样本仍应表示为：

\[
H_t\rightarrow Y_{t:t+H}^{bg},
\]

而不是把干预动作作为网络条件。

干预数据的作用是扩展已观测历史分布，覆盖：

- ego 急制动已经表现后的背景响应；
- ego 急加速已经表现后的间隙变化；
- ego 横向侵入已经表现后的避让或竞争；
- ego 取消操作后背景重新规划；
- 多次连续状态变化下的响应。

数据可来自高保真仿真、受控交通模型或真实干预数据。即使使用这些数据，也禁止把施加干预的动作标签直接输入 QR-WM。

### 6.4 总损失

\[
\mathcal{L}=
\lambda_{pos}\mathcal{L}_{pos}
+\lambda_{vel}\mathcal{L}_{vel}
+\lambda_{ctrl}\mathcal{L}_{ctrl}
+\lambda_{plan}\mathcal{L}_{plan}
+\lambda_{ref}\mathcal{L}_{ref}
+\lambda_{overlap}\mathcal{L}_{overlap}
+\lambda_{int}\mathcal{L}_{interaction}
+\lambda_{phy}\mathcal{L}_{physical}
+\lambda_{KL}\mathcal{L}_{KL}
+\lambda_{rec}\mathcal{L}_{behavior-rec}
+\lambda_{div}\mathcal{L}_{diversity}
+\lambda_B\mathcal{L}_{B_0}.
\]

各项语义：

- `L_pos`：背景位置误差；
- `L_vel`：背景速度误差；
- `L_ctrl`：背景控制误差；
- `L_plan`：完整背景控制计划监督；
- `L_ref`：确定性 joint refinement 改善约束；
- `L_overlap`：连续 response 的 buffer 一致性；
- `L_interaction`：多车碰撞、间距和关系一致性；
- `L_physical`：加速度、横摆率、速度和 jerk 约束；
- `L_KL`：prior/posterior 对齐；
- `L_behavior-rec`：潜变量未来特征重建；
- `L_diversity`：防止 latent collapse；
- `L_B0`：首秒动作摘要一致性。

不得包含 denoising loss、noise-level loss 或 diffusion objective。

所有损失必须使用 agent、time 和 pairwise validity masks。无效车辆、无效时间和无效车辆对不得进入平均值。

### 6.5 `B0` 投影专项验证

在真实 Flow tail 样本上执行：

\[
B_0\rightarrow U^{B_0}\rightarrow S_{0:1s}\rightarrow\widehat B_0.
\]

逐维报告：

- MAE；
- p50；
- p90；
- p95；
- p99；
- 极端样本误差；
- 不可同时满足摘要的样本比例。

若尾部投影误差较大，应优先修正投影和 START 组合，不得依赖后续 refiner 掩盖固定先验错误。

### 6.6 训练课程

默认预算：

1. 8 epoch：START 与短期 buffer warm-up；
2. 12 epoch：3 s closed-loop rollout；
3. 20 epoch：5 s full joint refinement。

总计 40 epoch。每阶段同时记录 START 与 ROLL 指标。epoch 数本身不是能力成立的证据。

---

## 7. Checkpoint 与模型选择

### 7.1 Checkpoint 必须保存

- `model_type`；
- `architecture_version = 4`；
- 完整 model config；
- `direct_ads_conditioning = false`；
- `diffusion_refinement = false`；
- Flow schema hash；
- Flow START / `B0` lifecycle contract；
- model state dict；
- optimizer/scheduler state；
- training seed；
- dataset/cache manifest；
- intervention/history-expansion dataset version；
- loss weights；
- 选择指标及分项；
- source commit hash。

旧架构 checkpoint 必须严格拒绝加载，不允许静默部分加载后作正式比较。

### 7.2 部署一致的 checkpoint 选择

至少计算：

- \(\mathrm{FDE}_{START}\)：无历史 Flow START；
- \(E_{B_0}\)：首秒摘要误差；
- \(\mathrm{FDE}_{ROLL}\)：有历史 ROLL；
- \(E_{risk}\)：TTC/DRAC/gap 等风险特征误差；
- \(E_{closed-loop}\)：闭环状态分布和背景响应误差。

建议复合指标：

\[
J_{select}=
\mathrm{FDE}_{START}
+\lambda_B E_{B_0}
+\lambda_R\mathrm{FDE}_{ROLL}
+\lambda_{risk}E_{risk}
+\lambda_C E_{closed-loop}.
\]

Flow 初始化测试是主要用途时，`FDE_START`、`E_B0` 和 `E_risk` 的权重必须高于普通 reconstruction FDE。

---

## 8. 正式评测协议

### Protocol A：Logged-ego reconstruction

目的：验证基础背景运动建模。

报告：

- 1–5 s ADE/FDE；
- velocity/acceleration error；
- joint-plan refinement gain；
- buffer overlap；
- EVT-tail 子集指标。

不得解释为反事实 ADS 验证。

### Protocol B：Flow START fidelity

目的：验证从 `C0+B0` 无历史启动。

报告：

- C0 解码一致性；
- START ADE/FDE；
- 首秒 `B0` 摘要 MAE；
- START 后长期漂移；
- slot mask 与事件结构一致性；
- Flow audit 文件和 hash。

### Protocol C：Long-tail event and distribution fidelity

目的：验证是否保持长尾事件演化与测试空间。

对真实与生成分布报告：

- TTC；
- DRAC；
- minimum gap；
- relative speed；
- acceleration；
- jerk；
- collision/near-miss；
- 高风险区域覆盖；
- 事件发生频率；
- 风险排序一致性。

使用 Wasserstein、KS、tail-CDF、联合密度或合适的 feature-distribution distance。不得只报告平均轨迹误差。

### Protocol D：Causal closed-loop response

固定相同 `C0+B0` 和 behavior latent，接入不同 ADS。QR-WM 不读取 ADS 动作，仅观察执行后形成的不同交通历史。

报告：

- 在 ego 行为可观测后，背景响应方向是否合理；
- reaction delay；
- gap/TTC 的后续变化；
- 多车联合让行、竞争和连锁制动；
- 与高保真仿真或参考数据的差异；
- 已验证状态分布内与分布外结果。

必须验证：在 ADS 实际状态历史尚未发生差异之前，QR-WM 输出不应因 ADS 内部动作不同而提前发生差异。

### Protocol E：Stochasticity and branching

固定 Flow START，采样多个 behavior latent。报告：

- minADE/minFDE；
- pairwise trajectory diversity；
- 有效行为模式覆盖；
- collision/risk 分支率；
- seed 重复性；
- latent/prior 审计信息。

### Protocol F：Path-level branch reproducibility

执行：

```text
snapshot → branch → restore
```

必须验证：

- restore 后无干预轨迹逐位一致；
- 固定 QR-WM RNG 时，仅 ADS 的后续实际状态变化导致背景分支差异；
- memory、buffer、latent、metadata 完整恢复；
- 失效轨迹可由 audit 数据复现。

### Protocol G：Ablation of diffusion removal

为确认删除扩散式去噪不会损害核心能力，至少比较：

1. CVAE + deterministic joint refiner；
2. CVAE + 原噪声条件 refiner（仅作为消融基线）。

比较：

- START/ROLL ADE/FDE；
- 长尾事件特征分布；
- 有效模式覆盖；
- collision/near-miss 分支率；
- 单条 rollout 时延；
- path-level 测试吞吐量。

除非噪声去噪在长尾保真上产生稳定、显著且可重复的收益，否则正式模型保持无扩散设计。

### Protocol H：ADS testing value

在相同测试预算下比较：

- failure discovery rate；
- time-to-first-failure；
- failure mode diversity；
- high-risk region coverage；
- 风险排序一致性；
- 高保真复核成功率。

QR-WM 的价值不能只由 ADE/FDE 决定。

---

## 9. 强制不变量与禁止事项

### 9.1 强制不变量

- 无效槽位不能影响有效车辆输出；
- START 不构造伪历史；
- ROLL 不读取真实未来背景信息；
- 原始 `B0` 仅在 START 消费一次；
- QR-WM 网络不读取 ADS 动作、未来 ego 控制或未来 ego 状态；
- ADS 动作只由环境用于推进 ego；
- 背景模型不能修改 ego 行为；
- Flow 概率与事件元数据不得丢失；
- logged reconstruction 与 closed-loop ADS testing 必须分开报告；
- stochastic rollout 必须可由 seed/latent 精确复现；
- snapshot/restore 必须完整恢复隐藏状态和 RNG；
- 超过验证时域必须终止或显式标记为外推；
- 多模态来自 behavior latent，不来自隐式未审计噪声。

### 9.2 禁止事项

- 禁止将相对速度直接当作背景绝对速度；
- 禁止将同一 `C0` 重复 25 帧伪造历史；
- 禁止对全部固定车辆槽位无 mask 求均值；
- 禁止使用 `fresh + alpha * B0_anchor` 叠加控制；
- 禁止把 ADS 动作或未来 ego 序列送入 QR-WM；
- 禁止根据 ADS 内部计划让背景车辆提前响应；
- 禁止在 ADS 模式用 highD future ego 覆盖；
- 禁止加入无必要性的扩散噪声、多步去噪和 diffusion loss；
- 禁止仅依据 history-aware FDE 选择部署模型；
- 禁止丢弃 Flow log probability 后继续声称概率可审计；
- 禁止使用旧 architecture checkpoint 代表当前版本；
- 禁止仅以低 ADE/FDE 替代长尾事件保真和闭环测试价值验证。

---

## 10. Codex 实施优先级

### P0：语义与代码清理

1. 删除 joint refiner、scene memory 和其他网络中的 ADS action/future ego conditioning；
2. 删除 ego future state token 及相关 mask、embedding 和梯度测试；
3. 删除 Gaussian corruption、noise schedule、noise embedding、denoising loss 和多步去噪逻辑；
4. 将模型版本升级为 `architecture_version = 4`；
5. 严格拒绝加载包含旧输入或旧 refiner 参数的 checkpoint；
6. 更新模型配置、文档和 TensorBoard 标签，删除过时字段；
7. 保证 environment 接收 ADS 动作，但 QR-WM forward 输入中不存在该动作。

### P1：阻断性正确性

1. 验证 Flow 相对速度正确还原；
2. 强制 `slot_valid` 和 Flow metadata consistency；
3. 验证无效槽位不影响有效车辆；
4. 验证 START/ROLL 无未来泄漏；
5. 验证 `B0` 仅在 START 使用；
6. 验证所有损失正确应用 agent/time/pairwise masks；
7. 验证确定性 joint refiner 的跨车辆依赖；
8. 验证 ADS 动作不会出现在 QR-WM 调用图中。

### P2：在线闭环与可复现分支

1. 完成 deterministic/stochastic reset；
2. 支持显式 `world_seed` 和 `behavior_latent`；
3. 实现完整 `snapshot/restore`；
4. 增加 map observation、termination 和最大验证时域；
5. 记录 Flow/QR checkpoint hash、latent、RNG 和 audit id；
6. 验证 restore 后逐位一致；
7. 系统级分支同时保存 ADS 内部状态和 QR-WM 环境状态。

### P3：训练—部署一致性

1. 训练 batch 同时包含 START 与 ROLL；
2. 将 START 指标纳入主 checkpoint 选择；
3. 独立评估 `B0` 投影尾部误差；
4. 重新训练 architecture v4；
5. 禁止使用旧 v1/v2/v3 artifact 代表当前架构性能。

### P4：闭环状态分布与长尾保真

1. 通过多策略、高保真或干预轨迹扩展已观测 history 分布；
2. 不把干预动作标签作为网络输入；
3. 评价可观测 ego 行为变化后的背景反应方向和延迟；
4. 完成长尾事件和风险特征分布协议；
5. 明确状态分布内和分布外能力边界。

### P5：长尾测试与高保真复核

1. 完成 stochastic branching 和 path-level reproducibility；
2. 与 RAMP/FIRM 进行统一预算比较；
3. 连接 failure discovery / path-level rare-event analysis；
4. 将高风险结果送入 CARLA/HIL 复核；
5. 评估单位测试预算下的失效发现效率。

---

---

## 12. Definition of Done

只有同时满足以下条件，QR-WM architecture v4 才可作为 FITWMAMS 的正式世界模型：

### 架构成立

- Flow START 和 history ROLL 明确分离；
- safety-aware query encoder、单一 scene memory、CVAE behavior prior、deterministic joint agent-time refiner 和 background future buffer 均已实现；
- 网络中不存在 ADS action/future ego conditioning；
- 网络中不存在扩散式加噪去噪模块；
- 无效槽位不影响有效车辆；
- 不读取任何真实未来背景信息。

### Flow 组合成立

- `C0` 数值和相对语义正确还原；
- `B0` 只在 START 使用；
- `B0` 投影误差在尾部样本上完成报告；
- Flow 密度、事件结构和 schema 信息完整保留。

### 在线测试成立

- ADS 与 QR-WM 只通过环境状态闭环耦合；
- environment 可接入不同 ADS 并同步执行 ego/background 控制；
- QR-WM 不读取 ADS 私有意图；
- deterministic/stochastic rollout 均可复现；
- snapshot/restore 支持 path-level 分支；
- 环境具有明确的动作合同、终止条件和验证时域。

### 训练与评测成立

- architecture v4 已从头训练；
- checkpoint 选择以 Flow START 和长尾风险保真为主；
- 完成 logged reconstruction、Flow START、长尾事件保真、因果闭环响应、stochastic branching 和 ADS testing value 协议；
- 完成删除 diffusion 的对照消融；
- 任何“支持任意 ADS”或“安全验证有效”的结论均受已验证观测域、环境动力学和高保真复核结果约束。

---

## 13. 最终核心定义

QR-WM 的最终设计应概括为：

\[
\boxed{
\text{Density-preserving Flow START}
+
\text{Safety-aware Query Encoding}
+
\text{CVAE Behavior Prior}
+
\text{Deterministic Joint Agent-Time Refinement}
+
\text{Closed-loop State Feedback}
}
\]

其最关键的因果原则是：

> QR-WM 不知道 ADS 接下来想做什么；它只观察 ADS 已经做了什么，并据此生成背景交通的后续联合行为。

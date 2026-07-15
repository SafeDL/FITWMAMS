# 世界模型下一阶段目标：初始行为锚定的半马尔可夫关系交通世界模型

> 文档状态：下一阶段目标规范，尚不代表当前代码已经完成。  
> 本文档在现有 `semi_markov_relational` 实现与完整 highD 验证结果的基础上，重新确定下一阶段唯一主线。现有模型、checkpoint、评测结果和失败实验全部保留为历史证据；不得把目标方案写成已有结果。

---

## 1. 决策摘要

下一阶段不再继续尝试更多零散特征、辅助损失或控制头，也不再把 Normalizing Flow 简化为只含单帧物理状态的 40 维 `clean_start` 版本作为正式主线。

新的统一决定是：

1. **恢复并冻结原有 76 维长尾 Normalizing Flow 作为正式初始化模型**；
2. 将其中的未来一秒动作摘要明确解释为**初始短时行为条件**，它是测试样本的一部分，而不是世界模型偷看的真实答案；
3. 该行为条件只允许作用于仿真开始后的第一秒；
4. 第一秒之后，世界模型不得再次读取动作摘要，只根据已经发生的 ego 状态、已生成的背景交通状态、动态关系图和自身随机状态继续滚动；
5. 核心修改集中在世界模型：使其能够把 Flow 给出的初始行为条件实现为平滑、可响应的第一秒交通动作，并把该信息正确地传递给后续半马尔可夫滚动状态；
6. 不恢复固定候选 0、`nominal_logit_margin`、嵌套 8 分支或其他基线保护 trick；
7. 不改变上游 EVT 长尾筛选，不把 EVT 风险、ADS 身份或未来 ego 输入世界模型。

下一阶段模型命名为：

> **Behavior-Anchored Semi-Markov Relational Traffic World Model**  
> **初始行为锚定的半马尔可夫关系交通世界模型**

其中“行为锚定”只描述第一秒初始化条件，不表示第一秒动作被完全锁死，也不表示后续交通行为由 Flow 预先给定。

---

## 2. 为什么调整目标

### 2.1 当前半马尔可夫模型的基础能力已经成立

当前 `semi_markov_relational` 已经完成：

- 可变参与者动态关系图；
- 0.2 秒响应更新和 25 Hz 动力学积分；
- 半马尔可夫潜在交互状态及持续时间；
- `mode + gate * response` 控制解码；
- 1--5 秒随机闭环展开；
- 环境 `snapshot()` / `restore()`；
- 完整 highD 六秒序列训练和 held-out 测试。

这些模块继续保留，不从头推翻。

### 2.2 当前主要缺口集中在 START 冷启动

当前 clean START 只提供 anchor 时刻单帧物理状态，而模型训练时通常能看到过去一秒真实历史。正式环境为了构造历史，会把同一个初始状态重复填充为 25 帧。这会丢失：

- 车辆是否正在减速或加速；
- 车距是在压缩还是扩大；
- 横向运动是否正在发展；
- 当前交互阶段已经进行到何种程度。

因此，问题不应继续通过随机增加历史位移特征、端点权重或控制曲线盲目修补，而应直接修正 START 信息合同。

### 2.3 原有 76 维 Flow 已经提供了合适的初始行为信息

原有 Normalizing Flow 的 76 维连续变量可拆为：

\[
Y_0=(S_0,B_0),
\]

其中：

- \(S_0\) 是 anchor 时刻的初始物理交通状态；
- \(B_0\) 是六个背景车在未来一秒内的紧凑动作摘要。

每个有效背景车的 \(B_0\) 包含：

```text
delta_vx_1s_mps
delta_vy_left_1s_mps
mean_ax_1s_mps2
min_ax_1s_mps2
final_ax_1s_mps2
mean_ay_left_1s_mps2
```

这些量不规定 25 帧逐帧动作，只给出第一秒的总体趋势。它们适合作为测试开始前采样的**初始短时行为条件**。

---

## 3. 固定不变的研究边界

### 3.1 继续使用当前六秒片段

不重新采集数据，不改变自然驾驶片段长度：

```text
segment length: 6.0 s
sampling rate: 25 Hz
frames: 150
```

每条序列继续组织为：

- 第 0--1 秒：世界模型 START 条件实现区间；
- 第 1--6 秒：世界模型自回归闭环滚动区间；
- 完整六秒真实轨迹：训练和 held-out 重建目标；
- recording/ego 级别划分保持不变。

### 3.2 原有 Normalizing Flow 保持不变

正式 Flow 继续使用：

```text
normalizing_flow/scripts/configs/highd_tail_flow_best.yaml
results/highd_tail_flow_best/checkpoints/best_tail_conditional_maf.pt
```

继续保留：

- 76 维连续联合建模；
- 条件 RQ-spline MAF；
- 稀有槽位加权；
- 分阶段训练；
- 离散结构配额采样；
- 既有采样温度与物理拒绝规则。

`clean_start` 40 维 Flow 已随其结果和适配代码移除，不再是正式或可选链路。

### 3.3 世界模型禁止读取的信息

任何 START 或 ROLL 路径都禁止输入：

- ADS 身份、算法类别、网络参数或内部表示；
- ego future；
- EVT 标签、事件风险、`risk_trace`；
- 第一秒之后的任何未来背景行为摘要；
- 真实未来背景有效掩码。

第一秒动作摘要 \(B_0\) 的合法性来自：它由冻结 Flow 在测试开始前采样，是外生测试条件，而不是运行时从真实未来读取。

---

## 4. 测试分布与测试空间

### 4.1 离散事件结构

离散结构仍为：

\[
E=(\mathrm{slot\_mask},\mathrm{primary\_slot}),
\]

并按 EVT-tail 数据中的经验分布 \(\widehat p_E(E)\) 采样。Normalizing Flow 不直接学习离散结构概率。

### 4.2 Flow 初始化分布

原有 Flow 学习：

\[
p_\phi(S_0,B_0\mid E).
\]

采样过程为：

\[
z_{\mathrm{flow}}\sim\mathcal N(0,I),
\]

\[
(S_0,B_0)=F_\phi(E,z_{\mathrm{flow}}).
\]

其中：

- \(S_0\) 决定初始位置、速度和加速度；
- \(B_0\) 决定第一秒背景交通的大致行为趋势；
- 同一个 \(B_0\) 仍可由世界模型实现为多条细节不同但摘要一致的动作轨迹。

### 4.3 世界模型随机过程

半马尔可夫世界模型继续使用外生状态随机数和持续时间随机数：

\[
U_n^z,U_n^d\sim\mathrm{Uniform}(0,1).
\]

它们通过模型预测分布得到潜在交互状态和持续时间：

\[
z_n=Q_{p_\theta(z\mid C_{t_n},z_{n-1})}(U_n^z),
\]

\[
d_n=Q_{p_\theta(d\mid z_n,C_{t_n})}(U_n^d).
\]

实现层测试空间可写为：

\[
\boxed{
\Omega_{\mathrm{tail}}
=
\mathcal E
\times
\mathcal Z_{\mathrm{flow}}
\times
\mathcal U_z
\times
\mathcal U_d
}
\]

其中 \(B_0\) 不需要额外再列一个随机空间，因为它已经由 \((E,z_{\mathrm{flow}})\) 唯一生成。

在审计和回放层，也可以记录实现后的：

\[
\Xi_{\mathrm{world}}=igl\{(z_n,d_n)\bigr\}_{n=1}^{N}.
\]

### 4.4 ADS 的位置

ADS 不属于测试空间。给定同一组：

```text
E
z_flow
state uniforms
duration uniforms
world-model checkpoint
map adapter
```

不同 ADS 接受相同外生测试条件。它们产生不同的已发生 ego 运动，因此背景交通的具体响应可以不同，但环境模型和基础随机性不变。

---

## 5. 新世界模型的核心设计

下一阶段只引入一个集中、可解释的结构变化：

> **以 Flow 的第一秒行为摘要初始化半马尔可夫交互状态，并用“行为锚点 + 实时响应修正”生成第一秒动作；第一秒后自动移除行为锚点，切换到纯闭环自回归滚动。**

### 5.1 输入拆分

从原 76 维 Flow 样本中确定性拆出：

```text
initial_physical_state S0
initial_behavior_anchor B0
slot_mask
primary_slot
```

建立：

\[
G_0=\mathrm{GraphAdapter}(S_0,E,\mathrm{map}).
\]

动态关系图、地图适配器和参与者表示继续沿用当前半马尔可夫实现。

### 5.2 行为锚点编码器

新增一个紧凑编码器：

\[
b_0=E_B(B_0,\mathrm{slot\_mask}).
\]

它只负责把每个有效背景车的第一秒摘要编码为：

- 每车短时行为表示；
- 场景级初始行为表示。

该编码器不读取逐帧未来动作，也不生成后续五秒行为。

### 5.3 初始潜在状态与持续时间

第一段潜在交互状态不再仅由静态初始图猜测，而是由初始图与行为锚点共同确定：

\[
p_\theta(z_0,d_0\mid G_0,B_0).
\]

直观含义是：

- 初始物理状态告诉模型车辆在哪里、速度是多少；
- 初始行为锚点告诉模型第一秒大致在加速、减速还是横向移动；
- 半马尔可夫状态将两者组合成可持续到后续时间的交互状态。

这样可以减少第一秒潜在状态误判及过早切换。

### 5.4 第一秒动作：锚点实现与实时响应并存

在 \(0\le t<1\text{ s}\) 内，背景车控制写为：

\[
u_{i,t}
=
D_A(h_{i,t},b_{0,i},z_0)
+
 g_{i,t}D_R(h_{i,t},C_t).
\]

其中：

- \(D_A\) 将 Flow 采样的行为锚点实现为具体动作趋势；
- \(D_R\) 根据当前动态图和已经发生的 ego 状态产生响应修正；
- \(g_{i,t}\in[0,1]\) 决定实时响应修正的强度。

因此，第一秒不是固定轨迹回放：

- Flow 决定趋势；
- 世界模型决定逐步动作；
- 背景车仍能对最新物理关系作出修正。

### 5.5 一秒后的硬切断

在 \(t\ge1\text{ s}\) 后：

\[
b_0\equiv0.
\]

后续控制只使用：

\[
u_{i,t}
=
D_M(h_{i,t},z_n,d_n)
+
 g_{i,t}D_R(h_{i,t},C_t).
\]

也就是说：

- 第一秒行为锚点不循环注入；
- 不从 Flow 取得第二秒、第三秒行为信息；
- 后续行为完全由闭环历史、动态关系图、潜在状态和外生世界随机数决定。

该硬切断必须在代码和单元测试中显式保证。

### 5.6 更新周期

保持当前通用研究版：

```text
physics integration: 0.04 s / 25 Hz
world-model response update: 0.2 s
compatibility roll(): five 0.2 s steps = 1.0 s
```

第一秒五个 response step 使用 \(B_0\)；第一秒结束后所有 step 都禁用 \(B_0\)。

---

## 6. 训练数据组织

### 6.1 继续使用完整六秒序列缓存

每条样本仍为：

```text
1 s observed history + 5 s future = 150 frames
```

在每条真实六秒序列中，从第一秒真实背景动作计算训练用 \(B_0^{\mathrm{log}}\)。该摘要与原 76 维 Flow 的特征定义完全一致。

### 6.2 训练时的两种 START 来源

训练分两个互补路径：

#### 路径 A：真实条件重建

输入：

\[
(S_0^{\mathrm{log}},B_0^{\mathrm{log}}).
\]

用途：

- 学习如何准确实现一个已知初始行为条件；
- 训练第一秒摘要一致性；
- 训练第一秒到后续状态的衔接。

该路径的第一秒指标必须称为**条件重建**，不能称为仅根据状态的无条件预测。

#### 路径 B：Flow 采样一致性

输入：

\[
(S_0,B_0)\sim p_\phi(S_0,B_0\mid E).
\]

用途：

- 检查模型在真正测试空间样本上的物理有效性；
- 检查生成动作重新计算后的摘要是否匹配 \(B_0\)；
- 检查后续闭环滚动是否稳定。

由于 Flow 样本没有唯一对应的真实未来，路径 B 不做逐轨迹 ADE/FDE 监督，只做分布和一致性评测。

### 6.3 不改变后续滚动训练

第一秒结束后继续使用当前模型状态闭环训练：

- 背景历史由模型生成；
- ego 使用该时刻已经发生的真实历史或 ADS 外部状态；
- 训练随机展开 1--5 秒；
- 保留 TBPTT（Truncated Backpropagation Through Time，截断时间反向传播）；
- 验证与测试始终评估完整五秒后续时域。

---

## 7. 紧凑训练目标

不再扩展新的大批辅助损失。总目标只保留四组：

\[
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{start}}
+
\lambda_B\mathcal L_{\mathrm{anchor}}
+
\lambda_R\mathcal L_{\mathrm{roll}}
+
\beta\mathcal L_{\mathrm{latent}}
}
\]

### 7.1 第一秒条件重建

\[
\mathcal L_{\mathrm{start}}
=
 d(\hat X_{0:1},X_{0:1})
+
\lambda_u d(\hat U_{0:1},U_{0:1}).
\]

保证第一秒轨迹和动作准确。

### 7.2 行为锚点一致性

从生成动作重新计算摘要：

\[
\widehat B_0=h(\hat U_{0:1}).
\]

损失为：

\[
\mathcal L_{\mathrm{anchor}}
=
\|\widehat B_0-B_0\|_1.
\]

这保证 Flow 采样的第一秒行为条件被真正实现，而不是被模型忽略。

### 7.3 后续 ROLL 模式

\[
\mathcal L_{\mathrm{roll}}
=
 d(\hat X_{1:6},X_{1:6}).
\]

该项使用模型自己生成的背景历史，重点衡量第一秒条件是否正确地传递到后续状态。

### 7.4 半马尔可夫状态与持续时间

继续使用当前的：

- 先验—后验一致性；
- 持续时间 likelihood；
- 右删失项。

但不得再新增与 EVT 风险相关的监督。

---

## 8. 推理与正式环境

### 8.1 重置

```python
flow_sample = sample_tail_c0(...)
states, valid, behavior_anchor, anchor_valid = start_state_from_flow_feature(
    flow_sample.features, flow_sample.slot_mask,
)
initial_graph = graph_builder.graph_at(..., states=states, valid=valid)
environment.reset(
    initial_graph,
    behavior_anchor=behavior_anchor,
    behavior_anchor_valid=anchor_valid,
    world_randomness=world_randomness,
)
```

环境必须记录：

```text
event_structure
z_flow / flow base sample
initial_physical_state
initial_behavior_anchor
state uniforms
duration uniforms
model checkpoint hash
map adapter version
```

### 8.2 第一秒

每 0.2 秒：

1. 读取最新动态图和已发生 ego 状态；
2. 使用 \(B_0\) 条件化初始状态与动作；
3. 生成响应修正；
4. 积分 0.2 秒；
5. 更新图和历史。

### 8.3 一秒之后

环境自动删除或屏蔽 \(B_0\)，所有后续 step 只沿当前半马尔可夫闭环路径运行。

### 8.4 快照和 AMS

`snapshot()` / `restore()` 必须额外保存：

```text
behavior_anchor
behavior_anchor_active
elapsed_since_reset
```

AMS（Adaptive Multilevel Splitting，自适应多层分裂）在第一秒内复制路径时需要保留锚点；在第一秒后复制时，锚点必须已经失效，不能重新激活。

---

## 9. 不再进行的盲目尝试

下一阶段禁止把以下内容作为主线继续排列组合：

- 继续增加 hidden dimension、Transformer 层数或 attention head 数；
- 重复尝试 ego 相对位置、历史相对位移等已失败特征；
- 再增加 endpoint、tail acceleration、jerk、energy、diversity 等独立损失；
- 恢复 CAT-K 固定候选 0 和 logit margin；
- 同时融合扩散模型；
- 改变六秒片段长度；
- 在没有 rounD 实际数据时宣称跨数据集有效。

仅允许围绕“第一秒行为锚点如何被世界模型实现和传递”进行有限、预先定义的比较。

---

## 10. 最小实验设计

只做三组核心模型，不进行大规模结构搜索。

| 模型 | Flow 条件 | 初始 latent 条件化 | 第一秒锚点实现 | 后续滚动 |
|---|---|---:|---:|---:|
| M0：当前半马尔可夫基线 | 40 维 clean state | 仅初始图 | 无 | 当前实现 |
| M1：行为锚定解码 | 原 76 维 Flow | 仅初始图 | 有 | 当前实现 |
| M2：完整目标模型 | 原 76 维 Flow | 初始图 + \(B_0\) | 有 | 当前实现 |

三个比较分别回答：

1. 恢复第一秒行为条件能否解决冷启动；
2. 只在动作解码中使用 \(B_0\) 是否足够；
3. 用 \(B_0\) 同时初始化潜在状态和持续时间，是否进一步降低后续漂移。

不要求对每个网络层和每个损失权重单独做消融。

---

## 11. 分阶段开发门槛

### Gate 0：信息与接口检查

必须先通过：

- 76 维 Flow 样本拆分可逆；
- \(B_0\) 只在前五个 0.2 秒 step 可见；
- 第六个 step 起 \(B_0\) 梯度和数值影响均为零；
- START 不读取 ego future；
- snapshot/restore 可逐位复现；
- 同一外生随机数对不同 ADS 不包含身份信息。

### Gate 1：第一秒条件重建

在固定 held-out highD 测试集上报告：

- 1 s ADE/FDE；
- action MAE；
- gap MAE；
- \(B_0\) 摘要重建误差。

M1 或 M2 必须显著优于 M0，才进入全量五秒训练。

### Gate 2：五秒后续滚动

固定同一测试集、坐标、积分器和 logged ego 条件，报告：

- 2--5 s ADE/FDE；
- 5 s gap MAE；
- 关系分布漂移；
- 物理无效率；
- 潜在状态切换率和持续时间校准。

目标不是仅改善第一秒，而是验证第一秒行为锚点能否降低后续累计漂移。

当前正式半马尔可夫模型的参考值为：

```text
1 s ADE/FDE: 0.03037 / 0.04163 m
5 s ADE/FDE: 0.37960 / 1.28388 m
5 s gap MAE: 0.33047 m
```

开发晋级至少要求：

- 5 s FDE 相对当前模型下降不少于 15%；
- 5 s gap MAE 相对当前模型下降不少于 15%；
- 第一秒性能不退化；
- 物理无效率不增加。

若 M1 和 M2 均不能满足该门槛，应停止继续增加复杂度，并重新审查半马尔可夫状态本身，而不是继续添加新模块。

### Gate 3：Flow→世界模型端到端分布验证

对 Flow 采样样本，不做一对一轨迹误差，报告：

- 生成第一秒摘要与 Flow \(B_0\) 的一致性；
- 速度、加速度、横向运动和 gap 分布；
- 动态关系类型分布；
- 潜在状态和持续时间分布；
- 物理有效率；
- 多随机种子稳定性。

### Gate 4：真实 ADS 闭环与 AMS 接口

验证：

- 第一秒背景车可对已发生 ego 状态作出有限响应；
- 第一秒后不再依赖 \(B_0\)；
- AMS 分支复制后可确定性继续；
- 相同外生测试条件下可配对比较多个 ADS。

---

## 12. 评测口径

必须同时保留两类结果，不能混写。

### 12.1 条件重建

使用真实片段提取的 \(B_0^{\mathrm{log}}\)：

\[
\hat\tau\sim p_\theta(\tau\mid S_0,B_0^{\mathrm{log}}).
\]

该结果回答：

> 给定初始行为条件后，世界模型能否准确实现第一秒并继续闭环滚动？

它不回答仅从单帧状态预测未来的能力。

### 12.2 端到端生成

使用 Flow 采样的：

\[
(S_0,B_0)\sim p_\phi(S_0,B_0\mid E).
\]

该结果回答：

> Flow 与世界模型共同诱导的长尾交通过程，是否保持自然驾驶尾部事件的统计特征和物理合理性？

此时评价分布，不要求与某条真实未来逐一对应。

---

## 13. 文件级实施建议

### 13.1 保留

```text
world_model/src/graph_schema.py
world_model/src/relational_encoder.py
world_model/src/semi_markov_state.py
world_model/src/dynamics.py
world_model/src/semi_markov_environment.py
world_model/src/semi_markov_train.py
world_model/src/semi_markov_evaluation.py
```

### 13.2 新增

```text
world_model/src/initial_behavior_anchor.py
```

职责：

- 从 76 维 Flow feature 中拆分 \(S_0,B_0\)；
- 对 \(B_0\) 做 mask-aware 编码；
- 提供摘要重建函数 \(h(U_{0:1})\)；
- 管理第一秒有效期。

### 13.3 修改

```text
world_model/src/semi_markov_model.py
```

- `prior_logits()` 的初始调用可接收行为锚点上下文；
- 初始 duration hazard 接收行为锚点；
- 第一秒 decoder 接收行为锚点，后续自动屏蔽。

```text
world_model/src/intent_response_decoder.py
```

- 增加一个紧凑 anchor realization 分支；
- 保留现有 response 分支；
- 不增加新的候选头和扩散模块。

```text
world_model/src/semi_markov_environment.py
```

- 支持从原 76 维 Flow 构造初始图与行为锚点；
- 记录 anchor 是否仍有效；
- 第五个 response step 后强制失效；
- snapshot/restore 保存相关状态。

```text
world_model/src/semi_markov_data.py
```

- 从每条六秒真实序列提取与原 Flow 完全一致的 \(B_0^{\mathrm{log}}\)；
- 不改变 150 帧长度与划分。

### 13.4 配置

新增唯一正式配置：

```text
world_model/scripts/configs/highd_behavior_anchored_semi_markov.yaml
```

不得为每个小特征建立大量配置分支。M0、M1、M2 通过一个明确的 `variant` 字段控制。

---

## 14. 论文中的方法表述

建议将核心方法写成：

> 原始长尾 Normalizing Flow 联合采样初始物理状态和第一秒短时行为条件。世界模型将该条件实现为可响应的第一秒背景交通动作，并以其形成的状态和半马尔可夫交互状态为起点，在不再访问未来行为摘要的条件下自回归生成后续交通演化。

英文可表述为：

> The tail normalizing flow jointly samples the initial physical configuration and a short-horizon behavior anchor. The traffic world model realizes this anchor into a reactive first-second rollout, initializes its latent interaction state from the resulting condition, and subsequently evolves the background traffic autoregressively without further access to future behavior summaries.

该设计的模型价值不在于“重新使用未来信息”，而在于：

1. 明确区分**长尾事件初始化**和**后续闭环动力学**；
2. 解决无历史 Flow 样本启动一个历史条件世界模型时的冷启动；
3. 使第一秒行为趋势成为可采样、可审计的测试变量；
4. 保留一秒后对未知 ADS 的响应式自回归滚动。

---

## 15. 最终目标

下一阶段不追求继续堆叠模型组件，而只验证一个清晰假设：

> **将第一秒短时行为条件显式纳入长尾初始化分布，并由半马尔可夫关系世界模型在第一秒内实现、随后切断，能否同时恢复高精度冷启动和可信的长期闭环滚动。**

若该假设在固定 highD 全量测试、Flow 端到端采样和 ADS 闭环中成立，则它成为下一篇论文世界模型部分的核心设计；若不成立，则依据 M0/M1/M2 的固定结果定位问题，而不是继续盲目增加网络和损失。

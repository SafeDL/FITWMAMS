# 世界模型目标

> 文档状态：新的世界模型设计规范。  
> 该文档用于替换现有 `world_model_goal.md` 中以 `catk_topk`、固定八候选、名义候选 0 和 `nominal_logit_margin` 为核心的目标描述。现有 `catk_topk` 及其 checkpoint 继续作为冻结基线保留；本文档描述的新模型尚需按下述规范重新实现、训练和评测，不能将其表述为当前代码已经完成的功能。

---

## 1. 研究目标与模型定位

下一代世界模型命名为：

> **Semi-Markov Relational Traffic World Model**  
> **半马尔可夫关系交通世界模型**

其核心潜在变量命名为：

> **Semi-Markov Latent Interaction State**  
> **半马尔可夫潜在交互状态**

模型面向自动驾驶系统（Automated Driving System，ADS）的闭环测试，只学习背景交通参与者的条件动态。模型不预测被测 ADS 的策略，不读取 ADS 身份、网络参数、未来动作、未来轨迹、风险标签、极值理论（Extreme Value Theory，EVT）标签或 `risk_trace`。

模型需要解决三个核心问题：

1. **跨数据集的交通关系表示**：摆脱 highD 固定六槽位对网络结构的绑定，统一表示 highD、rounD 及后续匝道、交叉口等不同道路拓扑；
2. **交互模式的持续性**：避免每个固定一秒窗口重新选择匿名候选，显式学习一个联合交互状态持续多久以及何时切换；
3. **意图持续与行为响应并存**：背景车辆保持自身交互状态，同时根据最新 ego 和其他交通参与者的已发生状态持续调整动作。

总体模型为：

\[
\mathcal G_{t-H:t}
\xrightarrow{E_\theta}
C_t
\xrightarrow{p_\theta(z_n,d_n\mid C_t,z_{n-1})}
(z_n,d_n)
\xrightarrow{D_\theta}
u_{1:N,t}
\xrightarrow{F_{\mathrm{dyn}}}
X_{1:N,t+1}.
\]

其中：

- \(\mathcal G_t\) 是动态关系交通图；
- \(C_t\) 是当前交通上下文；
- \(z_n\) 是半马尔可夫潜在交互状态；
- \(d_n\) 是该状态的持续时间；
- \(u_{i,t}\) 是第 \(i\) 个背景参与者的控制量；
- \(F_{\mathrm{dyn}}\) 是可微分交通参与者动力学。

---

## 2. 当前基线与迁移原则

当前仓库保留以下冻结基线：

```text
results/highd_world_model/catk_topk/checkpoints/best_world_model.pt
results/highd_world_model/catk_topk_baseline/checkpoints/best_world_model.pt
```

现有 `catk_topk` 的主要特点为：

- 固定八个候选；
- 候选 `0` 来自名义解码器；
- 候选 `1--7` 来自场景级意图 token；
- 通过 `nominal_logit_margin` 强制候选 `0` 成为确定性最大概率分支；
- 每个一秒 chunk 重新选择候选；
- 以 jerk 控制点生成一秒动作；
- 最多进行五个 chunk 的模型状态训练。

这些机制继续作为比较基线，但不再作为下一代模型的目标结构。新模型原则上删除：

- 嵌套的名义解码器内部候选；
- 固定候选 `0`；
- 人为覆盖候选 `0` logit；
- 将“八个分支”解释为世界中固有的八种行为；
- 仅由固定一秒 chunk 边界决定潜在行为切换。

下一代模型的随机性来自可学习的半马尔可夫潜在交互状态及其持续时间，而不是固定八个输出头。

---

## 3. 固定数据边界：继续使用六秒自然驾驶片段

### 3.1 六秒片段保持不变

上游 `process_highD/` 继续输出固定六秒自然驾驶片段。不得为了学习半马尔可夫状态而重新定义自然驾驶事件长度，也不要求重新采集更长轨迹。

每个六秒片段作为一个完整的序列训练样本：

```text
segment length: 6.0 s
sampling rate: 25 Hz
frames per segment: 150
```

推荐的序列组织为：

- 前 1.0 秒：可观测历史；
- 后 5.0 秒：世界模型闭环重建目标；
- 完整六秒序列：仅供训练期后验网络识别潜在交互状态及持续时间；
- 推理期先验网络只能读取当前和过去信息。

允许在同一六秒片段内部进行随机时间裁剪或随机起始点增强，但不得把裁剪结果当作新的独立自然驾驶事件，也不得改变 recording/ego 级别的数据划分。

### 3.2 序列级缓存

现有 START/ROLL 行缓存应升级为序列级缓存。统一缓存至少包含：

```text
sequence_id
recording_id
ego_id
timestamps                    [T]
agent_ids                     [N]
agent_states                  [T, N, D_agent]
agent_valid                   [T, N]
ego_index                     scalar
primary_agent_index           scalar or -1
map_polylines                 [M, P, D_map]
map_polyline_valid            [M, P]
lane_graph_edges              variable length
agent_lane_candidates         [T, N, R]
split                         train / val / test
is_evt_tail                   metadata only
```

其中 \(T=150\)，\(N\) 和 \(M\) 允许随数据集和场景变化。

`is_evt_tail`、事件风险值和 EVT 诊断只允许作为评测切片 metadata，禁止作为世界模型输入或训练监督。

### 3.3 六秒窗口末端的持续时间截断

若最后一个潜在交互状态在六秒窗口结束时仍未终止，则该持续时间是右删失观测。持续时间损失需要使用生存概率：

\[
\mathcal L_{\mathrm{censor}}
=-\log P_\theta
\left(
D_n\ge d_n^{\mathrm{obs}}
\mid z_n,C_{t_n}
\right),
\]

而不能错误地把窗口结束当作真实交互状态结束。

---

## 4. Normalizing Flow 的干净 START 模型

Normalizing Flow（归一化流，NF）与世界模型的职责必须严格分离。

### 4.1 离散结构与连续初始化分布

长尾初始条件采用：

\[
p_{\mathrm{tail}}(E,S_0)
=
\widehat p_E(E)
p_\phi(S_0\mid E),
\]

其中：

- \(E\) 是离散事件结构；
- \(\widehat p_E(E)\) 是由 EVT-tail 自然驾驶样本统计得到的经验分布；
- \(S_0\) 是连续初始物理状态；
- \(p_\phi(S_0\mid E)\) 由条件 Normalizing Flow 建模。

### 4.2 Flow 不再建模未来动作摘要

下一代论文模型中，Flow 的连续目标只包含仿真初始时刻可观测的物理状态、车辆几何和必要的静态场景属性，不包含：

- 未来一秒平均加速度；
- 未来一秒最小加速度；
- 终端加速度；
- 未来速度变化；
- 未来横向动作摘要；
- 任何由未来轨迹计算的行为语义变量。

第一项半马尔可夫潜在交互状态由世界模型产生：

\[
(z_1,d_1)
\sim
p_\theta(z,d\mid \mathcal G_0,E),
\]

而不是由 Flow 的未来动作摘要隐式指定。

### 4.3 START 条件

干净的 START 过程为：

```text
E ~ empirical event-structure distribution
z_flow ~ N(0, I)
S0 = Flow(E, z_flow)
G0 = DatasetAdapter(S0, map, E)
(z1, d1) ~ WorldModel(G0)
```

因此，Flow 只决定“初始交通状态是什么”，世界模型决定“从该状态开始背景交通如何演化”。

---

## 5. 数据集无关的动态关系交通图

### 5.1 统一图定义

定义动态关系交通图：

\[
\mathcal G_t=
\left(
V_t^A,
V^L,
V^C,
E_t^{AA},
E_t^{AL},
E^{LL},
E_t^{AC}
\right).
\]

其中：

- \(V_t^A\)：交通参与者节点；
- \(V^L\)：车道或道路折线节点；
- \(V^C\)：冲突区域节点；
- \(E_t^{AA}\)：参与者—参与者动态关系边；
- \(E_t^{AL}\)：参与者—车道动态归属边；
- \(E^{LL}\)：车道—车道静态拓扑边；
- \(E_t^{AC}\)：参与者—冲突区域动态关系边。

“动态”表示参与者集合、车辆关系、车道归属和冲突区域接近关系可以随时间改变；地图折线及车道拓扑通常静态并可缓存。

### 5.2 参与者节点

统一参与者节点采用局部车体坐标或道路局部坐标，推荐特征为：

\[
x_i^A=
[
 v_x^{\mathrm{local}},
 v_y^{\mathrm{local}},
 a_x^{\mathrm{local}},
 a_y^{\mathrm{local}},
 \sin\psi,
 \cos\psi,
 l,
 w,
 \mathrm{type},
 \mathrm{valid}
].
\]

绝对位置主要通过相对边特征表达，避免模型绑定某个 recording 的全局坐标。

### 5.3 车道折线节点

每条车道折线包含：

\[
x_l^L=
[
\mathrm{centerline},
\mathrm{tangent},
\mathrm{curvature},
\mathrm{width},
\mathrm{lane\ type},
\mathrm{priority}
].
\]

highD 的曲率可为零；rounD 的环岛弧段、入口和出口使用非零曲率和拓扑关系。

### 5.4 冲突区域节点

冲突区域用于表示：

- 环岛入口合流区；
- 匝道汇入区；
- 交叉口冲突区；
- 让行控制区；
- 多流向轨迹的潜在交叉区域。

highD 可以不构造冲突区域节点，rounD 和城市道路适配器按地图拓扑生成。

### 5.5 参与者—参与者边

推荐边特征：

\[
e_{ij}^{AA}=
[
\Delta x_{ij}^{(i)},
\Delta y_{ij}^{(i)},
\Delta v_{x,ij}^{(i)},
\Delta v_{y,ij}^{(i)},
\sin\Delta\psi_{ij},
\cos\Delta\psi_{ij},
r_{ij}^{\mathrm{topo}},
\mathrm{valid}
].
\]

其中拓扑关系类型至少包括：

```text
same_lane
adjacent_lane
merge
diverge
cross
unrelated
```

TTC、DRAC 等派生安全量可以作为评测指标或可选辅助特征，但不应成为跨数据集动态交通图的必要定义。

### 5.6 参与者—车道边

推荐特征为：

\[
e_{il}^{AL}=
[
 d_{\mathrm{lateral}},
 \Delta\psi_{\mathrm{lane}},
 s_{\mathrm{progress}},
 p(l\mid x_i)
].
\]

每辆车保留 top-\(R\) 个候选车道归属，而不是强制唯一 lane ID。换道、环岛入口和分流区域可同时关联多条车道。

### 5.7 数据集适配器

#### highD 适配器

- 从车道线和 recording metadata 构造直线 lane polylines；
- 建立前驱、后继、左右相邻关系；
- 将车辆投影到候选车道；
- 根据车道拓扑和局部纵向距离构造稀疏参与者关系边；
- 现有 `same_front`、`same_rear` 等六槽位只作为后处理解释标签和旧基线接口，不再是新模型输入结构。

#### rounD 适配器

- 将环岛圆环、入口、出口切分为 lane polylines；
- 建立环形 successor、入口 merge、出口 diverge 关系；
- 根据几何和优先权生成冲突区域；
- 使用局部切向坐标和相对位姿；
- 允许参与者动态进入、退出及更换主要交互对象。

所有数据集适配器最终输出统一的 `DynamicTrafficGraph` schema，世界模型主干不包含任何 highD 或 rounD 专有槽位名称。

---

## 6. 动态关系图编码器

推荐采用：

> **Lane-Topology-Gated Pairwise-Relative Heterogeneous Attention**  
> **车道拓扑门控的成对相对异构注意力**

### 6.1 静态地图编码

车道折线先由 polyline encoder 编码，车道—车道关系通过静态图注意力更新：

\[
H^L=E_{\mathrm{map}}(V^L,E^{LL}).
\]

静态地图特征在 episode 开始前编码一次并缓存，不在每个仿真步重复计算。

### 6.2 参与者时间编码

每个参与者过去 \(H\) 帧的状态独立编码：

\[
h_{i,t}^{\mathrm{temp}}
=E_{\mathrm{temp}}
(x_{i,t-H:t}).
\]

时间编码器可使用轻量 Transformer、时序卷积网络或状态空间序列模型。该模块属于基础实现，不作为核心创新。

### 6.3 参与者—地图交叉注意力

参与者 token 作为 query，候选车道及冲突区域 token 作为 key/value：

\[
h_{i,t}^{AM}
=
\mathrm{CrossAttn}
\left(
q_i,
\{k_l,v_l,e_{il}^{AL}\}
\right).
\]

### 6.4 参与者—参与者稀疏相对注意力

先使用车道拓扑和冲突区域关系筛选邻居：

\[
\mathcal N_i(t)
=
\left\{
j:
 r_{ij}^{\mathrm{topo}}
\in
\{\mathrm{same},\mathrm{adjacent},\mathrm{merge},\mathrm{cross}\}
\right\}.
\]

再进行成对相对注意力：

\[
\alpha_{ij}
=
\operatorname{softmax}_{j\in\mathcal N_i}
\left[
\frac{q_i^\top(k_j+\phi_k(e_{ij}^{AA}))}{\sqrt d}
\right],
\]

\[
h_{i,t}'
=
\sum_{j\in\mathcal N_i}
\alpha_{ij}
\left(v_j+\phi_v(e_{ij}^{AA})\right).
\]

该表示同时满足：

- 参与者数量可变；
- 对参与者顺序置换等变；
- 对全局平移和旋转更稳健；
- 地图简单时计算开销低；
- 可自然扩展到环岛、匝道和交叉口。

---

## 7. Semi-Markov Latent Interaction State

### 7.1 潜在状态定义

第 \(n\) 个潜在交互阶段表示为：

\[
(z_n,d_n),
\]

其中：

- \(z_n\in\{1,\ldots,K_z\}\) 是离散半马尔可夫潜在交互状态；
- \(d_n\) 是该状态持续时间；
- \(z_n\) 是场景级状态，一个状态联合调制当前所有背景参与者；
- \(z_n\) 不预设为“跟驰”“让行”或“切入”，其语义通过训练后统计解释。

### 7.2 先验与训练后验

ROLL 模式先验只读取过去和当前：

\[
p_\theta(z_n,d_n\mid C_{t_n},z_{n-1}).
\]

训练后验可以读取完整六秒真实序列：

\[
q_\varphi(z_n,d_n\mid \mathcal G_{1:T}).
\]

训练时由后验帮助识别潜在交互阶段，ROLL 模式只使用先验；这表示不读取未来背景状态，不构成因果效应识别。

### 7.3 持续时间分布

推荐使用离散 hazard 参数化。在响应更新步 \(r\) 上定义终止概率：

\[
h_{n,r}
=
P(D_n=r\mid D_n\ge r,z_n,C_{t_n}).
\]

则：

\[
P(D_n=r)
=
h_{n,r}
\prod_{s<r}(1-h_{n,s}).
\]

该形式自然支持六秒窗口末端的右删失持续时间。

### 7.4 状态语义与可审计性

每个训练 checkpoint 必须输出潜在状态原型统计，包括：

- 状态使用频率；
- 平均持续时间和持续时间分布；
- 参与者速度和加速度变化；
- 关系边变化；
- 车道归属变化；
- 主交互对象变化；
- highD 和 rounD 中的跨数据集对应关系。

潜在状态编号存在标签置换，不得宣称 `z=3` 在不同 checkpoint 间天然具有相同语义。

---

## 8. 意图持续—动作响应的多速率动力学

### 8.1 控制分解

每个背景参与者的控制量分解为：

\[
u_{i,t}
=
u_{i,t}^{\mathrm{mode}}
+
g_{i,t}
u_{i,t}^{\mathrm{response}}.
\]

其中：

\[
u_{i,t}^{\mathrm{mode}}
=D_M(h_{i,t},z_n,d_n)
\]

描述当前半马尔可夫潜在交互状态下的基本运动趋势；

\[
u_{i,t}^{\mathrm{response}}
=D_R(h_{i,t},C_t)
\]

描述车辆对最新 ego、其他参与者和道路关系的响应；

\[
g_{i,t}
=
\sigma(D_G(h_{i,t},C_t,z_n))
\]

决定响应修正强度。

模型不读取 ADS 身份，只读取已经发生的 ego 物理状态。因此环境规律对 ADS 身份无关，但对 ADS 已发生行为有响应。

### 8.2 通用控制空间

通用研究模型内部采用：

\[
u=[a,\dot\psi],
\]

其中：

- \(a\) 是纵向加速度；
- \(\dot\psi\) 是横摆角速度。

通过可微单轨或自行车动力学得到：

\[
\begin{aligned}
x_{t+1}&=x_t+v_t\cos\psi_t\Delta t,\\
y_{t+1}&=y_t+v_t\sin\psi_t\Delta t,\\
v_{t+1}&=v_t+a_t\Delta t,\\
\psi_{t+1}&=\psi_t+\dot\psi_t\Delta t.
\end{aligned}
\]

highD 兼容层可将 \([a,\dot\psi]\) 转换为现有 \([a_x,a_y]\) 输出；新模型主干不再绑定直线道路笛卡尔加速度。

### 8.3 控制曲线参数化

控制解码器可以继续使用低维控制点或样条参数化，以保证动作连续和降低输出自由度。该机制属于物理解码实现，不作为独立核心创新。

---

## 9. 更新周期与通用研究版接口

### 9.1 三个时间尺度

#### 物理积分周期

\[
\Delta t_{\mathrm{sim}}=0.04\ \mathrm{s}
\]

highD 原型保持 25 Hz。其他数据集通过 adapter 重采样到统一内部频率或使用可配置积分步长。

#### 交互响应更新周期

\[
\Delta t_{\mathrm{resp}}
\in\{0.1,0.2\}\ \mathrm{s}
\]

默认 highD 原型采用 0.2 秒。每个响应步执行：

1. 更新动态参与者集合；
2. 更新车道归属和关系边；
3. 重新编码当前图；
4. 保持或切换潜在交互状态；
5. 重新计算响应控制；
6. 积分到下一响应时刻。

#### 潜在交互状态更新周期

潜在状态不按固定时间重采样，而在持续时间结束时切换：

\[
t_{n+1}=t_n+d_n.
\]

在状态持续期间，\(z_n\) 保持不变，但响应控制会按 \(\Delta t_{\mathrm{resp}}\) 根据最新交通图更新。

### 9.2 通用研究版

核心环境增加细粒度接口：

```python
environment.reset(initial_graph, world_randomness)
environment.step(ego_state, ego_valid, dt=0.1_or_0.2)
```

当前一秒 `roll()` 接口保留为兼容包装器，而不是方法约束：

```python
def roll(ego_history_states, ego_history_valid):
    # 内部执行 5 个 0.2 s 响应更新，或 10 个 0.1 s 响应更新
    # 最后拼接并返回 1.0 s 输出
```

因此：

- 对现有调用方仍可返回一秒背景交通；
- 核心研究模型可以在一秒内部响应 ADS；
- 一秒不再是潜在交互状态边界；
- 同一潜在状态可以跨越多个 `roll()` 调用；
- 一个 `roll()` 内也可以在持续时间结束后切换状态。

---

## 10. 训练目标

训练目标保持紧凑，仅保留三个损失组：

\[
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{recon}}
+
\beta\mathcal L_{\mathrm{latent}}
+
\lambda\mathcal L_{\mathrm{roll}}
}
\]

### 10.1 重建损失

重建未来五秒背景参与者状态和控制：

\[
\mathcal L_{\mathrm{recon}}
=
\sum_t
\left[
\lambda_p\|\hat p_t-p_t\|_1
+
\lambda_v\|\hat v_t-v_t\|_1
+
\lambda_\psi d_\mathrm{angle}(\hat\psi_t,\psi_t)
+
\lambda_u\|\hat u_t-u_t\|_1
\right].
\]

所有项按 agent/time validity mask 计算。

### 10.2 潜在状态损失

包含：

- 先验—后验 Kullback-Leibler 散度或离散交叉熵；
- 持续时间负对数似然；
- 六秒窗口末端右删失生存损失；
- 必要的 codebook commitment，但不再堆叠多个相似的 diversity/entropy trick。

可写为：

\[
\mathcal L_{\mathrm{latent}}
=
D_{\mathrm{KL}}
\left[
q_\varphi(z,d\mid\mathcal G_{1:T})
\|\
p_\theta(z,d\mid\mathcal G_{\le t})
\right]
+
\mathcal L_{\mathrm{duration}}
+
\mathcal L_{\mathrm{censor}}.
\]

### 10.3 闭环滚动损失

从第一秒历史开始，背景交通后续输入使用模型自己的生成状态，ego 使用该时刻已经发生的 logged 状态：

\[
\mathcal L_{\mathrm{roll}}
=
\sum_{t=1}^{5\mathrm{s}}
 d(\hat X_t,X_t).
\]

训练采用随机展开长度和截断时间反向传播（Truncated Backpropagation Through Time，TBPTT）：

- 每个 batch 从 1--5 秒中随机选择展开长度；
- 长序列保留生成状态但对较早计算图停止梯度；
- 不再把 `max_chunks=5` 解释为模型理论限制；
- 五秒上限仅来自固定六秒片段中“一秒历史 + 五秒未来”的数据边界。

---

## 11. 训练组织

### 11.1 数据划分

继续按 `recording_ego` 分组：

```text
train / validation / test = 0.70 / 0.15 / 0.15
```

同一 recording/ego 的相邻片段不得跨划分。

### 11.2 两阶段训练

#### 阶段一：后验辅助的序列重建

- 后验网络读取完整六秒序列；
- 先验网络读取过去和当前；
- 学习潜在交互状态、持续时间和物理动作重建；
- 初期可使用较强 teacher forcing 稳定训练。

#### 阶段二：模型状态闭环训练

- 后续背景输入逐步替换为模型生成状态；
- ego 仍使用截至当前时刻已经发生的真实状态；
- 状态替换比例逐步提高；
- 训练到未来五秒，但使用 TBPTT 控制显存。

### 11.3 EVT 独立性

训练 loader 不向模型提供：

```text
is_evt_tail
event_risk
risk_trace
EVT threshold
failure label
ADS identity
```

EVT-tail 仅用于独立评测切片，不能用于风险引导或尾部保持训练。

---

## 12. 长尾测试空间

### 12.1 初始条件空间

离散事件结构与连续初始条件定义为：

\[
E\sim\widehat p_E(E),
\qquad
z_{\mathrm{flow}}\sim\mathcal N(0,I),
\qquad
S_0=F_\phi(E,z_{\mathrm{flow}}).
\]

highD 中可取：

\[
E=(\mathrm{slot\_mask},\mathrm{primary\_slot}),
\]

但世界模型内部会把该结构转换成统一动态关系图。rounD 等数据集可使用对应的离散结构描述和主要交互关系，只要最终映射到相同图 schema。

### 12.2 世界模型路径变量

实现和回放层可记录：

\[
\Xi_{\mathrm{world}}
=
\{(z_1,d_1),\ldots,(z_N,d_N)\}.
\]

固定：

- Flow checkpoint；
- 世界模型 checkpoint；
- 动态图构造规则；
- 动力学；
- 响应更新频率；
- 潜在状态与持续时间采样结果；
- episode 边界；

即可复现一条背景交通随机过程。

### 12.3 ADS 无关的基础随机空间

为了比较不同 ADS，基础随机变量使用外生均匀随机数：

\[
U_n^z,U_n^d
\overset{\mathrm{i.i.d.}}{\sim}
\mathrm{Uniform}(0,1),
\]

并通过逆累积分布得到：

\[
z_n
=F^{-1}_{p_\theta(z\mid C_{t_n},z_{n-1})}(U_n^z),
\]

\[
d_n
=F^{-1}_{p_\theta(d\mid z_n,C_{t_n})}(U_n^d).
\]

基础测试空间为：

\[
\boxed{
\Omega_0^{(T)}
=
\mathcal E
\times
\mathcal Z_{\mathrm{flow}}
\times
[0,1]^{\infty}
\times
[0,1]^{\infty}
}
\]

有限 episode 只消耗有限个随机数。不同 ADS 使用相同的：

\[
(E,z_{\mathrm{flow}},U^z,U^d),
\]

环境模型参数保持不变，但由于各 ADS 已发生的 ego 状态不同，背景车可以产生不同的物理响应。ADS 不属于测试空间，ADS 只决定同一个外生测试点如何映射为闭环轨迹。

### 12.4 与 EVT 和 AMS 的边界

- EVT 负责从自然驾驶数据中识别长尾事件、拟合风险尾部和确定安全关键阈值；
- 经验离散结构分布和 Flow 负责初始条件；
- 半马尔可夫关系交通世界模型负责背景交通条件动态；
- Adaptive Multilevel Splitting（自适应多层分裂，AMS）负责在该目标概率测度上高效估计稀有安全事件概率。

世界模型本身不读取 EVT 或 AMS 信息。

---

## 13. 推理与复现

每个 episode 至少保存：

```text
event_structure
z_flow or flow base sample
initial_physical_state
world_random_seed
uniform_state_random_numbers
uniform_duration_random_numbers
realized_latent_states
realized_durations
latent_transition_times
response_update_period
model_checkpoint_hash
map_adapter_version
dynamics_version
```

一秒兼容包装器还应保存每个 `roll()` 内部发生的潜在状态边界，而不能只保存一秒级候选索引。

---

## 14. 评测体系

### 14.1 单步条件重建

报告：

- 控制误差；
- 速度和加速度误差；
- Average Displacement Error（平均位移误差，ADE）；
- Final Displacement Error（终点位移误差，FDE）；
- 相对位置和相对速度误差；
- 车道归属与关系类型重建准确率。

### 14.2 六秒序列重建

以第一秒为历史，自由滚动未来五秒，报告：

- 1、2、3、4、5 秒误差；
- 车辆关系分布；
- 交互阶段持续时间分布；
- 潜在状态切换频率；
- 物理违规、道路偏离和碰撞诊断；
- 生成状态对 logged 状态的分布一致性。

### 14.3 潜在交互状态评测

报告：

- prior/posterior 一致性；
- 持续时间校准；
- 状态使用率；
- 有效状态数量；
- 状态原型可重复性；
- 不同随机种子下的标签对齐后稳定性；
- highD 与 rounD 中共享状态和数据集特异状态。

### 14.4 响应性评测

对 ego 的已发生动作施加受控扰动，检查：

- 背景车响应连续性；
- 相同物理 ego 行为在不同 ADS 身份下是否产生一致响应；
- 不同 ego 加减速和横向运动下的背景动作变化；
- 意图状态是否保持，响应项是否合理调整；
- 是否出现明显因果混淆和历史复制。

该实验用于验证响应合理性，不能宣称获得了真实反事实 ground truth。

### 14.5 EVT-tail 独立切片

在 held-out EVT-tail 六秒片段上重复上述重建和分布评测，但 EVT 标签不参与训练。目标是验证世界模型在自然驾驶长尾条件下仍能复现真实行为，而不是让世界模型学习风险排序。

### 14.6 跨数据集评测

至少包括：

1. highD 单数据集训练和测试；
2. rounD 单数据集训练和测试；
3. highD + rounD 联合训练；
4. 共享图编码器、数据集特异输入 adapter；
5. 从 highD 预训练到 rounD 微调；
6. 去除地图关系或改回固定槽位的对照。

---

## 15. 核心消融

主论文只需要以下核心消融：

| 模型 | 动态关系图 | 半马尔可夫潜在状态 | 学习持续时间 | 意图—响应分解 |
|---|---:|---:|---:|---:|
| B0：单模式动态图预测器 | ✓ |  |  |  |
| B1：联合潜在交互状态 | ✓ | ✓ |  |  |
| B2：持续时间感知模型 | ✓ | ✓ | ✓ |  |
| Full：完整模型 | ✓ | ✓ | ✓ | ✓ |

附加敏感性实验只需：

- 潜在状态词表大小 \(K_z\)；
- 响应更新周期 0.1 秒与 0.2 秒；
- 随机 rollout 训练长度；
- 是否使用多候选车道归属。

控制点数量、网络层数、损失权重等常规超参数放入附录，不逐项作为方法消融。

---

## 16. 文件级实现建议

建议新增：

```text
world_model/src/graph_schema.py
world_model/src/graph_builder.py
world_model/src/adapters/highd_adapter.py
world_model/src/adapters/round_adapter.py
world_model/src/relational_encoder.py
world_model/src/semi_markov_state.py
world_model/src/intent_response_decoder.py
world_model/src/dynamics.py
world_model/src/sequential_dataset.py
world_model/src/semi_markov_train.py
world_model/src/semi_markov_environment.py
world_model/scripts/configs/highd_semi_markov_relational.yaml
world_model/scripts/configs/round_semi_markov_relational.yaml
```

现有文件处理原则：

- `catk_topk` 代码和 checkpoint 只作为冻结基线；
- 新模型类型建议为 `semi_markov_relational`；
- 不复用 `nominal_logit_margin`；
- 不保留固定候选 `0`；
- 不在新 Flow schema 中保留未来动作摘要；
- 现有一秒 `roll()` 作为兼容 wrapper，核心环境使用细粒度 `step()`；
- 数据缓存从 338 万条 START/ROLL 行样本迁移为 161,314 条六秒序列及其动态图片段表示。

---

## 17. 实施阶段

### 阶段 A：highD 动态图与干净 Flow

1. 保留原 161,314 个六秒片段和 recording/ego 划分；
2. 重建不含未来动作摘要的 Flow 数据；
3. 构造 highD 轻量 lane graph；
4. 将六槽位转换为可变参与者图；
5. 验证图适配后的单步重建不弱于旧基线。

### 阶段 B：Semi-Markov Latent Interaction State

1. 实现训练后验和 ROLL 模式先验；
2. 实现持续时间 hazard 与右删失损失；
3. 实现意图—响应分解解码器；
4. 使用六秒序列训练；
5. 完成未来五秒自由滚动评测。

### 阶段 C：通用研究版闭环环境

1. 增加 0.1--0.2 秒 `step()`；
2. 一秒 `roll()` 内执行多次响应更新；
3. 保存潜在状态边界、持续时间和外生随机数；
4. 接入 ADS 闭环仿真；
5. 为 AMS 提供状态快照、复制和继续运行能力。

### 阶段 D：rounD 扩展

1. 构造环岛 lane graph 和 conflict-zone graph；
2. 支持动态 agent 生命周期；
3. 支持 top-\(R\) 车道归属；
4. highD/rounD 联合训练；
5. 完成跨道路拓扑泛化实验。

---

## 18. 验收标准

新模型只有同时满足以下条件，才能替代 `catk_topk` 成为正式世界模型：

1. 在固定 highD test 划分上的一秒 ADE、FDE 和交互误差不弱于冻结基线；
2. 在未来五秒模型状态自由滚动中显著降低误差累积或关系分布漂移；
3. 潜在状态持续时间具有可校准性，且状态切换不退化为每个响应步随机跳变；
4. 不依赖 ADS 身份、ego future、EVT 标签或风险标签；
5. 干净 Flow 初始化不使用任何未来动作摘要；
6. 一秒兼容 `roll()` 与细粒度研究版 `step()` 在相同随机数和更新时间设置下结果一致；
7. 至少完成 highD 与 rounD 两种道路拓扑的数据适配验证；
8. checkpoint、随机变量、地图适配器和动力学版本可完整审计与复现。

---

## 19. 非目标与限制

本模型不负责：

- 视觉、LiDAR 或占用场生成；
- ADS 感知误差；
- 被测 ADS 的规划和控制；
- EVT 风险建模；
- AMS 稀有事件概率估计算法；
- 通过风险损失主动生成更危险行为；
- 证明未观测 ego 干预下的真实反事实响应。

六秒片段限制意味着潜在状态持续时间超过六秒时只能得到右删失信息；该限制必须在持续时间估计和论文讨论中明确说明。

---

## 20. 最终方法概括

下一代世界模型不再将未来交通限制为固定八个候选，也不使用名义候选 0 和 logit margin。其核心定义为：

\[
\boxed{
\text{统一动态关系交通图}
+
\text{Semi-Markov Latent Interaction State}
+
\text{意图持续—动作响应的多速率动力学}
}
\]

在整个长尾测试框架中：

\[
\boxed{
\text{经验离散事件结构}
+
\text{Normalizing Flow 连续初始物理状态}
+
\text{Semi-Markov Relational Traffic World Model 条件动态}
}
\]

共同定义可采样、可闭环执行、可概率解释的长尾测试过程；EVT 负责长尾筛选和风险阈值，AMS 负责稀有概率估计，二者均不进入世界模型训练。

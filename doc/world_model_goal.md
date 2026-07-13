# 世界模型目标

## 当前基线与结论

`world_model/` 当前唯一的活动模型名为 `catk_topk`。冻结 v1 基线位于：

```text
results/highd_world_model/catk_topk_baseline/checkpoints/best_world_model.pt
```

正式 v4 checkpoint 位于：

```text
results/highd_world_model/catk_topk_anchored/checkpoints/best_world_model.pt
```

v4 将 v1 名义自然驾驶动力学封装为候选 `0`，并将完成多 chunk 训练的 CAT-K 联合残差行为放在候选 `1--7`。这是同一个 checkpoint 内的八分类 \(\Xi_{world}\)，没有增加外部输入或测试空间自由度。确定性重建选择候选 `0`；categorical 环境采样仍可选择八个候选。

完整 test paired-bootstrap（2,000 次）表明 EVT-tail START 的 ADE、FDE、gap MAE，以及 logged-ego START->ROLL ADE 相对冻结基线的点差和单侧 95% 上界均为 `0`。因此 v4 通过“不弱于冻结基线”的晋升条件；冻结基线继续保留用于后续审计。

代码核查表明，当前实现已经用同一个离散索引联合选择六个背景车 slot 的动作候选；但候选来自逐 slot 的输出通道，而不是显式的场景级意图 token。训练损失是 hard best-of-K：

\[
k^\star=\arg\min_k D_k,
\]

只对获胜候选施加主要动作梯度，并用动态的 `best_idx` 监督候选概率。这会造成三个问题：候选编号没有稳定的行为语义、未获胜候选的训练信号稀疏、八个候选可能塌缩为相近轨迹。当前的 `closed_loop_adaptation` 只是对 logged history 加扰动，不是在模型生成的背景车历史上展开多个连续 chunk，因此不能称为真正的模型状态闭环训练。

已完成的实现目标是：

> **概率校准的离散意图—物理残差闭环世界模型**

该目标在不改变外部接口的前提下，用软多模态联合残差候选、时空交互和模型状态多 chunk 闭环训练扩展名义自然驾驶锚点。候选 `0` 是性能约束锚点；候选 `1--7` 是可采样的联合残差意图。由于锚点没有额外 token 参数，文档中“八个可学习 token”的表述应理解为七个残差 token 加一个固定的名义意图，不应误称八个候选都独立学习。

## 长尾测试空间不变性

上游 `process_highD/` 固定长尾筛选口径；归一化流负责初始条件。测试空间仍严格定义为：

\[
\Omega_{\mathrm{tail}}=E\times Z_{\mathrm{flow}}\times\Xi_{\mathrm{world}},
\]

其中：

- \(E=(\mathrm{slot\_mask},\mathrm{primary\_slot})\)：离散交通结构；
- \(z_{\mathrm{flow}}\in Z_{\mathrm{flow}}\)：归一化流连续隐变量；
- \(s_0=\mathrm{Flow}(E,z_{\mathrm{flow}})\)：完整 76 维 START 场景条件；
- \(\Xi_{\mathrm{world}}=(\xi_0,\ldots,\xi_{J-1})\)，且 \(\xi_j\in\{0,\ldots,7\}\)：每个 START/ROLL chunk 的背景交通离散意图。

固定 Flow checkpoint、世界模型 checkpoint、候选采样温度、积分器、坐标规则、槽位规则、车辆几何常数和 episode 时间边界后，\((E,z_{\mathrm{flow}},\Xi_{\mathrm{world}})\) 唯一确定背景交通随机过程。

关系特征不是新的测试空间变量：

\[
r_j=g\bigl(x^{\mathrm{ego}}_j,x^{\mathrm{bg}}_j,\mathrm{primary\_slot}\bigr).
\]

它由当前已发生的 ego 状态和已生成背景状态确定性计算。不同 ADS 由于实际 ego 历史不同，会得到不同的背景车响应；这是同一个条件环境动力学对不同外部干预的响应，不是测试空间被 ADS 改写。

正式测试环境不得引入连续分支噪声、额外 latent、ADS 身份、ego future、风险标签或 `risk_trace` 作为世界模型随机变量或输入。归一化流中的第一秒动作摘要仍只属于 START 初始条件；ROLL 中该张量保持零值。

## 不变的外部边界

以下内容全部保持不变：

- `reset_from_flow_sample(feature_row, slot_mask, primary_slot_index, world_seed)`、`start()`、`roll(ego_history_states, ego_history_valid)` 的调用方式；
- START/ROLL 的现有输入张量及其语义；
- `actions_mps2[25, 6, 2]`、`background_states[25, 6, 6]`、`background_valid[25, 6]`；
- `candidate_index` 和 `candidate_probabilities[8]`；
- 每个 chunk 只使用一个八分类离散索引 \(\xi_j\)；
- 六个固定 slot、静态 `slot_mask`、持续局部坐标系、25 Hz、每 chunk 25 帧、固定积分器和车辆几何常数；
- ego 仅作为已经发生的外部状态历史，绝不把 ADS 身份、ego future、风险标签或 EVT 标签输入模型。

因此，改进只能替换 `catk_topk` 的内部编码器、候选解码器和训练目标。它不能改变 Flow 到环境的三元数组接口，也不能增加测试空间自由度。

## 必要的四项核心创新

### 1. 场景级离散意图与名义锚点

候选 `0` 固定为冻结的自然驾驶名义动力学 \(A_j^{(0)}\)。候选 `1--7` 使用七个可学习 token：

\[
e_k\in\mathbb{R}^d,\qquad k\in\{1,\ldots,7\},
\]

并以完整场景上下文 \(C_j\) 和同一个 token 解码全部六辆背景车的联合残差：

\[
\hat A_j^{(k)}=A_j^{(0)}+D_\theta(C_j,e_k),\qquad k\in\{1,\ldots,7\}.
\]

候选概率应由场景级池化表示产生：

\[
\pi_j=\operatorname{softmax}\bigl(l_\theta(C_j)\bigr),\qquad \pi_j\in\mathbb{R}^8,
\]

而不是先为每个 slot 输出 logits 再平均。`candidate_index=0` 表示名义自然驾驶模式，`candidate_index=k>0` 表示采用 token \(e_k\) 对应的联合背景交通残差行为；它不再只是动作通道编号。

残差 token 的可解释性必须如实限定：在同一固定 checkpoint 内，\(k>0\) 与 token、概率和轨迹是一一对应且可复现；不同独立训练运行之间存在标签置换对称性，不能宣称 `k=3` 天然具有跨 checkpoint 的固定语义。每个 checkpoint 必须附带七个残差 token 与一个名义锚点的条件原型统计，例如纵横向加速度、间隙变化、变道趋势和责任质量，以供审计与比较。

### 2. 概率校准的软多模态学习

对每个候选计算 mask 后的动作负对数似然或等价距离 \(D_k\)，并使用温度化软责任：

\[
r_k=
\frac{\pi_k\exp(-D_k/\tau_r)}
{\sum_j\pi_j\exp(-D_j/\tau_r)}.
\]

主损失采用温度化 mixture likelihood：

\[
\mathcal{L}_{\mathrm{mix}}
=-\tau_r\log\sum_k\pi_k\exp(-D_k/\tau_r).
\]

训练初期可使用较高 \(\tau_r\)，但校准评估和最终概率语义必须在 \(\tau_r=1\) 下报告。名义锚点保持冻结，七个残差候选通过 mixture 目标获得梯度；`candidate_probabilities[8]` 仍对应场景级意图发生概率。

为避免候选塌缩，在不牺牲物理可行性的前提下增加轻量正则：

\[
\mathcal{L}=
\mathcal{L}_{\mathrm{mix}}
+\lambda_{\mathrm{es}}\mathcal{L}_{\mathrm{energy}}
+\lambda_{\mathrm{div}}\mathcal{L}_{\mathrm{diversity}}
+\lambda_{\mathrm{smooth}}\mathcal{L}_{\mathrm{smooth}}.
\]

其中 energy score 使用候选间的概率加权距离，diversity 使用带 margin 的候选对距离；它们只鼓励具有责任质量的候选分开，不能通过制造不现实动作获得收益。验证时必须报告 mixture NLL、责任质量、候选熵、有效候选数、成对轨迹距离和概率校准误差。采样温度 `candidate_temperature=1` 时返回的概率才可解释为校准概率；非 1 温度仅是人为改变后的采样分布。

### 3. 时空交互编码器与物理残差动作解码器

内部编码器替换为：

\[
\text{逐 agent 时间编码}
\;\longrightarrow\;
\text{相对交互图注意力}
\;\longrightarrow\;
\text{场景级池化与意图解码}.
\]

时间编码器读取既有 history/current 张量。图注意力中的 ego—背景车关系保留现有关系特征；背景车—背景车的相对位置、相对速度、gap 和有效性由当前状态在模型内部确定性计算。若需物理量，模型应使用固定的反标准化常数恢复物理坐标后再构图。不得新增外部输入字段，也不得把这些边特征视为测试空间变量。

对候选 \(k\)，解码器不再自由输出 25 帧独立加速度，而输出每辆车、每个动作维度的低维 jerk 样条控制点。以当前加速度保持的零 jerk 运动学为参考：

\[
a_t^{\mathrm{physics}}=a_0,
\qquad
j_t^{(k)}=j_{\max}\tanh\bigl(B_t c^{(k)}\bigr),
\]

\[
\Delta a_{t+1}^{(k)}=\Delta a_t^{(k)}+\Delta t\,j_t^{(k)},
\qquad
\hat a_t^{(k)}=
\operatorname{clip}\bigl(a_t^{\mathrm{physics}}+\Delta a_t^{(k)}\bigr).
\]

\(B\) 是固定的 25 帧低阶样条基，\(c^{(k)}\) 是由 \((C_j,e_k)\) 生成的控制点。残差的正负号只是参数化约定，等价于题述的 \(a^{\mathrm{physics}}-\Delta a\)。动作范围、jerk 范围与现有积分器范围一致。该设计保持输出 `actions_mps2[25,6,2]` 不变，同时让加速度和 jerk 在时间上连续。

### 4. 真正的多 chunk 模型状态闭环训练

训练序列由现有缓存的同一 `(segment_id, anchor_frame)` 分组，不增加数据字段。使用对齐的一秒 offset：START 的 0 帧，以及 ROLL 的 25、50、75、100 帧；因此一个 6 秒片段最多可监督五个连续一秒 chunk。

第 \(j\) 个 chunk 的流程是：

\[
\hat A_j=D_\theta(C_j,e_{\xi_j}),
\qquad
\hat S_{j+1}=F_{\mathrm{integrator}}(\hat S_j,\hat A_j).
\]

下一 chunk 的背景车 history 必须由 \(\hat S_{j+1}\) 构造，关系特征也必须从该预测背景状态与当前 ego 状态重新计算。训练前向传播仍只选择一个离散候选；可用基于 \(r_k\) 的 straight-through 分类采样，让前向状态转移与正式环境一致，同时将软责任梯度传回候选概率和 token。

离线训练在推进到下一 chunk 后可使用该时刻已经发生的 logged ego 历史作为外部条件，但当前 chunk 的模型输入不得包含该 chunk 之后的 ego 状态。也就是说，logged ego future 只在时间推进后成为下一步的历史，不能泄漏为当前预测条件。部署到 ADS 测试时，`roll()` 始终接收 ADS 实际执行后的过去一秒 ego 历史。

训练应采用从 1 到 5 chunk 的课程展开，并在每个 chunk 累积动作 mixture loss、状态一致性和物理平滑损失。现有 history 扰动可以保留为轻量数据增强，但不能再作为闭环训练的替代品。

## 实施边界与代码映射

实现时只允许以下内部调整：

- `world_model/src/model.py`：以场景 token、图注意力和物理残差 decoder 替换当前 `candidate_head` 与逐 slot logits；公共采样函数仍返回动作、候选索引和八维概率。
- `world_model/src/train.py`：以软 mixture 目标和多 chunk 展开替换 hard `best_idx` 主监督；按现有缓存的 segment/offset 分组构造序列。
- `world_model/src/data.py`：只增加内部序列索引或读取逻辑，不改变 START/ROLL 样本字段、Flow 条件字段和缓存语义。
- `world_model/src/evaluation.py`：保留现有 open-loop、EVT-tail 和 logged-ego 重建口径，并增加 2--5 chunk 的模型状态重建诊断。logged-ego replay 始终只作重建评测。
- `world_model/src/environment.py` 与 `world_model/src/rollout.py`：不得改变对外接口、张量形状、积分规则或测试空间变量；仅接入新的内部模型输出。

新旧架构的 checkpoint 必须通过显式 `architecture_version` 区分，以便冻结的当前 checkpoint 可继续严格加载并参与对比。`catk_topk` 是对外模型名；旧版本加载仅作为性能基线，不作为新的活动方法或隐藏兼容分支。

不把增大 Transformer、增加候选数、引入高维 normalizing flow 或输入 ADS 身份列为当前优先级。因果性体现在 ego 作为外部干预条件和模型状态闭环响应学习；归一化流继续只承担长尾初始条件分布，未来最多作为低维意图残差的可选增强。

## 性能与验收门槛

新模型需要努力提升表现,直到超过或者接近现有方法 checkpoint。比较必须固定：数据缓存、train/val/test 分组、评测脚本、候选数 8、采样温度、随机种子、积分器和 logged-ego 重建口径。

当前必须保住的历史基线为：

| 指标 | 冻结基线 |
|---|---:|
| EVT-tail ADE | 0.028524 m |
| EVT-tail FDE | 0.041313 m |
| EVT-tail gap MAE | 0.026337 m |
| logged-ego START->ROLL ADE | 0.055516 m |

候选模型的晋升条件：

1. 所有上述误差型主指标的点估计均不得高于冻结基线；使用相同样本的 paired bootstrap 后，新增模型相对基线的单侧 95% 置信上界也不得为正。
2. 新增的 2--5 chunk 模型状态重建指标必须同时报告，并与冻结基线在同一协议下比较；不能用单步指标掩盖长 horizon 退化。
3. `candidate_temperature=1` 下必须报告概率校准、责任质量、候选熵、有效候选数和成对轨迹距离。若八个候选退化为近似相同轨迹，或概率与软责任显著失配，即使单步 ADE 更低也不得晋升。
4. 保持物理边界、槽位语义、无未来信息和正式环境接口不变；任何违反该边界的结果均无效。

在所有门槛通过前，当前 `catk_topk` checkpoint 继续是唯一有效基线。本文件描述的是下一阶段必须满足的实现与验收规范，不表示四项创新已经完成，也不授权启动训练或全量评测。

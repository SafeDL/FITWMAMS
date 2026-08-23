# Stochastic Causal HiQR：分布随机性与干预有效性增强目标文档

gpu的系统环境在:
conda activate tread

## 1. 文档目的

本文档定义 `FITWMAMS` 当前分层交通世界模型下一阶段的核心研发目标。

当前系统已经形成三层生成与执行框架：

\[
p(M)p(C_0\mid M)p(K\mid C_0,M)

ightarrow
p_	heta(	au_{\mathrm{soft}}\mid C_0,M,K,z_{\mathrm{diff}})

ightarrow
\pi_\phi(a^{bg}_t\mid H_t,	au_{\mathrm{soft}},z_{\mathrm{HiQR}})
\]

其中：

- **Normalizing Flow**：生成场景结构 \(M\)、初始状态 \(C_0\) 与长时状态结点 \(K\)；
- **Diffusion**：根据 \((C_0,M,K)\) 生成 5.96 s 六车联合 soft plan；
- **HiQR**：以 25 Hz 读取已实现联合历史，对 soft plan 重对齐，并只提交下一帧背景动作。

当前轨迹事实重建已经达到较高精度。下一阶段不再把降低 ADE/FDE 作为首要目标，而是重点解决：

1. 固定场景条件下的**交通行为随机性不足**；
2. 背景车辆对 ADS 干预的**响应分布、响应幅值、时延和局部性仍不充分**；
3. 随机性目前仍较接近“小幅连续噪声”，尚未充分形成可解释的多模态驾驶行为分支；
4. observational reconstruction 与 counterfactual closed-loop response 之间仍缺少更强的训练约束。

---

# 2. 总体目标

下一阶段将当前 HiQR 响应层升级为：

\[
oxed{	ext{Stochastic Causal HiQR}}
\]

完整体系定义为：

\[
oxed{
	ext{Flow}

ightarrow
	ext{Diffusion Soft Plan}

ightarrow
	ext{Stochastic Causal HiQR}
}
\]

目标不是增加更多生成模块，而是将短时响应层升级为：

> **结构化随机行为状态 + 交互耦合随机演化 + 因果响应场 + 共同随机数反事实分支训练。**

最终希望得到：

\[
egin{aligned}
z_t^{scene},z_t^{agent}
&\sim
p_\psi(z_t\mid z_{t-1},H_t,G_t),\
R_t
&=
\mathcal R_\phi(H_t,G_t,z_t,	au_{\mathrm{soft}}),\
a_t^{bg}
&=
a_t^{soft}
+
\Delta a_t^{interaction}
+
\epsilon_t^{micro}.
\end{aligned}
\]

其中：

- \(H_t\)：截至当前时刻已经发生的联合交通历史；
- \(G_t\)：当前车辆交互图；
- \(z_t^{scene}\)：慢时间尺度场景行为状态；
- \(z_t^{agent}\)：每辆背景车的持续行为潜变量；
- \(\mathcal R_\phi\)：因果交互响应场；
- \(\epsilon_t^{micro}\)：仅承担微观运动扰动的小尺度噪声。

---

# 3. 当前架构中必须保留的原则

## 3.1 保留 Flow 的职责

Flow 继续负责：

\[
p(M)p(C_0\mid M)p(K\mid C_0,M)
\]

不得将以下职责重新塞入 Flow：

- 25 Hz 背景响应；
- ADS 动作条件；
- 逐帧控制生成；
- 人为行为类别；
- 未来 ego 轨迹。

Flow 只描述：

> **场景级概率、场景结构、初始条件和长时间尺度行为约束。**

## 3.2 保留 Diffusion 的职责

Diffusion 继续负责：

\[
p(	au_{\mathrm{soft}}\mid C_0,M,K)
\]

其 soft plan：

- 是长时运动先验；
- 是可修改的参考；
- 不是必须执行的硬轨迹；
- 不能直接决定 ADS 干预后的最终背景运动；
- 不读取未来 ego 行为。

不得退回到“Diffusion 直接承担完整闭环背景仿真”的设计。

## 3.3 保留 25 Hz 闭环执行

正式仿真继续采用：

\[
\Delta t=0.04s
\]

每个物理 tick：

1. HiQR 根据当前已实现历史生成下一帧背景动作；
2. ADS 生成当前 ego 动作；
3. ego 与背景车辆同步进入动力学；
4. 获得新的联合状态；
5. 新状态进入下一时刻历史；
6. 再次规划。

必须继续满足：

\[
a_t^{ADS}

ightarrow
S_{t+1}^{ego}

ightarrow
a_{t+1}^{bg}
\]

而不是：

\[
a_t^{ADS}

ightarrow
a_t^{bg}.
\]

即未来或尚未执行的 ADS 动作不得泄漏到背景模型。

## 3.4 保留随机性可重放

必须继续显式管理：

- `scenario_seed`
- `motion_seed`
- diffusion noise
- response random stream
- snapshot / restore 状态

相同：

\[
(C_0,M,K,	au_{\mathrm{soft}},\Xi)
\]

与相同 ADS 策略应得到可复现世界。

对比不同 ADS 时，必须允许固定同一个世界随机性 \(\Xi\)。

---

# 4. 核心创新一：持续随机行为潜变量过程

## 4.1 当前问题

当前 agent stochasticity 主要通过高相关连续噪声实现。

该机制更接近：

\[
	ext{deterministic behavior}
+
	ext{small continuous perturbation}
\]

而不是：

\[
	ext{multiple plausible driver response modes}.
\]

固定 \(C_0,M,K\) 后，轨迹分支间差异仍较小，说明随机性主要来源于 Flow 对 \(K\) 的变化，而非短时响应层自身。

## 4.2 新目标

显式建模：

\[
z_i(t)=	ext{driver behavior state}
\]

该潜变量不需要人工驾驶风格标签，但应能够隐式表示：

- 激进 / 保守；
- 强跟驰 / 弱跟驰；
- 快速反应 / 延迟反应；
- 主动让行 / 保持行为；
- 横向稳定 / 横向活跃；
- 对风险敏感程度差异。

推荐层次：

\[
z^{scene}

ightarrow
z_i^{intent}(t)

ightarrow
\epsilon_i^{micro}(t).
\]

### 慢场景潜变量

\[
z^{scene}
\]

用于表达整个交通世界共享的慢变化行为环境。

### Agent 行为状态

\[
z_i^{intent}(t)
\]

应持续多个物理 tick，不允许每 0.04 s 独立重采样。

### 微观随机扰动

\[
\epsilon_i^{micro}(t)
\]

仅保留为低幅度局部运动随机性。

## 4.3 推荐实现

优先实现条件随机状态转移：

\[
z_{t+\Delta t}
\sim
p_\psi
\left(
z_{t+\Delta t}
\mid
z_t,H_t,G_t,	au_{\mathrm{soft}}

ight).
\]

推荐优先级：

1. **Conditional Normalizing Flow latent transition**
2. Mixture Density latent transition
3. Latent Neural SDE
4. 简单 Gaussian AR transition（只可作为基线）

首选 Conditional Flow 的原因：

- 可显式计算 log likelihood；
- 与当前场景 Flow 方法逻辑一致；
- 可直接评价行为 latent 的概率校准；
- 能表达非 Gaussian、多峰行为转移；
- 更容易控制随机性与可重放性。

---

# 5. 核心创新二：交互耦合随机行为演化

## 5.1 当前问题

背景车辆的随机行为不能视为互相独立。

真实交通中存在：

- 制动传播；
- 间隙竞争；
- 让行协同；
- 相邻车道补偿行为；
- 多车连锁反应。

因此随机潜变量必须由车辆交互图共同决定。

## 5.2 目标模型

构建动态交互图：

\[
G_t=(V_t,E_t)
\]

边特征至少包括：

- longitudinal gap；
- lateral gap；
- relative velocity；
- relative acceleration；
- same-lane / adjacent-lane relation；
- closing rate；
- TTC；
- DRAC；
- soft-plan conflict；
- 当前响应 relevance。

联合随机转移：

\[
z_{t+1}^{1:N}
\sim
p_\psi
\left(
z_{t+1}^{1:N}
\mid
G_t,z_t^{1:N},H_t,	au_{\mathrm{soft}}

ight).
\]

要求不同车辆的行为 latent 不是独立噪声，而能够产生：

\[
	ext{driver-1 random response}

ightarrow
	ext{driver-2 response propagation}.
\]

---

# 6. 核心创新三：Causal Interaction Response Field

## 6.1 当前问题

当前纵向干预响应具有较强显式结构，容易学会：

\[
	ext{ego braking}
\Rightarrow
	ext{rear vehicle braking}
\]

但仍难表达：

- 非线性响应；
- 驾驶员异质性；
- 响应时延；
- 风险状态依赖；
- 横向与纵向耦合；
- 多车传播；
- 强干预和弱干预的不同机制。

## 6.2 新目标

引入：

\[
oxed{\mathcal R_\phi=	ext{Causal Interaction Response Field}}
\]

定义：

\[
\Delta \mathbf a_i^{bg}
=
\mathcal R_\phi
\left(
\Delta S_t^{ego},
R_{ego,i},
z_i,
H_t,
G_t,
	au_{\mathrm{soft}}

ight).
\]

其中：

- \(\Delta S_t^{ego}\)：已经实现的 ego 状态变化；
- \(R_{ego,i}\)：ego 与第 \(i\) 辆背景车的关系特征；
- \(z_i\)：该车行为潜变量。

## 6.3 可选 Jacobian 表达

可增加显式局部因果灵敏度：

\[
J_{i,t}
=
rac{
\partial a_{i,t+\delta}^{bg}
}{
\partial S_t^{ego}
}.
\]

局部近似：

\[
\Delta a_i^{bg}
pprox
J_{i,t}\Delta S_t^{ego}.
\]

期望自动形成：

\[
\|J_{
m near}\|
>
\|J_{
m far}\|
\]

以及：

\[
J_{
m same\ lane}

eq
J_{
m adjacent\ lane}.
\]

该 Jacobian 不应人为指定，而应由关系编码器、行为 latent、风险状态和 soft plan 联合预测。

---

# 7. 核心创新四：共同随机数 Counterfactual Twin-Branch Training

这是下一阶段最重要的训练创新。

## 7.1 基本思想

利用已有：

- world seed；
- snapshot；
- restore；
- 显式 latent；
- 随机流状态；

从同一个世界状态建立两个或多个闭环分支。

例如：

```text
                    ┌── ego nominal action ──→ branch A
same snapshot  ─────┤
                    └── ego braking action ──→ branch B
```

固定：

\[
\Xi_A=\Xi_B
\]

其中 \(\Xi\) 包括所有外生随机变量。

于是：

\[
\Delta S^{bg}
=
S^{bg,B}-S^{bg,A}
\]

主要来自 ego intervention，而不是随机噪声变化。

## 7.2 干预类型

至少实现三档剂量：

### 纵向制动

\[
a^{ego}\in\{0,-2,-4,-6\}\;m/s^2
\]

### 纵向加速

\[
a^{ego}\in\{0,+1,+2,+3\}\;m/s^2
\]

### 横向干预

通过 yaw-rate 或横向轨迹扰动构造低、中、高三档干预。

---

# 8. 反事实训练损失

最终建议：

\[
L=
L_{
m factual}
+
\lambda_d L_{
m distribution}
+
\lambda_c L_{
m counterfactual}
+
\lambda_l L_{
m locality}
+
\lambda_m L_{
m monotonicity}
+
\lambda_r L_{
m recovery}
+
\lambda_p L_{
m physical}.
\]

## 8.1 Factual Loss

继续保持：

- action loss；
- position loss；
- velocity loss；
- likelihood / NLL；
- jerk regularization；
- soft-plan consistency；
- interaction loss。

要求新增创新不能明显破坏当前高精度事实重建。

## 8.2 Distribution Loss

用于防止模型退化成确定性单模态：

推荐使用：

- Energy Score；
- CRPS；
- multi-sample NLL；
- MMD；
- Wasserstein；
- behavior-mode coverage；
- pairwise trajectory diversity。

必须同时约束：

\[
	ext{diversity}
\]

与：

\[
	ext{realism}.
\]

禁止仅通过放大 noise 获得更高轨迹距离。

## 8.3 Direction Consistency Loss

对于与 ego 存在明确跟驰关系的车辆，ego 更强制动时，背景响应不应系统性朝反方向变化。

形式：

\[
L_{
m direction}
=
\operatorname{ReLU}
\left(
-s_{
m expected}\Delta a^{bg}

ight).
\]

仅作用于高 relevance 车辆。

## 8.4 Dose Monotonicity Loss

若：

\[
|u_1|<|u_2|<|u_3|
\]

则期望响应幅值：

\[
R_1\le R_2\le R_3.
\]

损失：

\[
L_{
m monotonicity}
=
\sum_k
\operatorname{ReLU}(R_k-R_{k+1}).
\]

## 8.5 Locality Loss

定义 near 与 far 两组车辆：

\[
L_{
m locality}
=
rac{
\|\Delta a_{
m far}\|
}{
\|\Delta a_{
m near}\|+\epsilon
}.
\]

也可针对图距离使用指数衰减：

\[
R_i
\propto
e^{-lpha d_G(ego,i)}.
\]

不得强制远端车辆完全不响应，而应约束响应随物理和交互相关性合理衰减。

## 8.6 Response Delay Loss

要求：

- 干预前无提前响应；
- 干预后在合理时延内出现响应；
- 延迟分布与自然数据统计相符。

重点评价：

\[
T_{
m first\ response},
\quad
T_{
m peak},
\quad
T_{
m recovery}.
\]

## 8.7 Recovery Loss

干预结束后，背景车辆应逐步重新接近自然行为轨迹，而不是永久保留无意义偏差。

定义：

\[
L_{
m recovery}
=
\|\Delta S^{bg}_{t+T_{
m recovery}}\|.
\]

但对于已经改变交通拓扑的干预，不应强制回归原轨迹，只约束恢复到新的合理状态分布。

---

# 9. 可选增强：Latent Response-Style Mixture of Experts

该部分为二级优先级，不作为第一轮必须实现内容。

建模：

\[
p(a^{bg}\mid H)
=
\sum_{k=1}^{K}
\pi_k(H,z)
p_k(a^{bg}\mid H,z).
\]

Expert 不人工命名，但可能自动形成：

- fast response；
- delayed response；
- defensive；
- aggressive；
- gap-maintaining；
- lateral-yielding。

只有在 latent flow 仍无法形成明显行为模式时，再考虑引入 MoE。

禁止单纯为了增加参数量使用 MoE。

---

# 10. 随机性评估目标

## 10.1 固定 \(C_0,M,K\)

保持场景和长程目标完全相同，只改变短时 response latent。

评价：

- Energy Score；
- CRPS；
- pairwise ADE；
- terminal pairwise distance；
- acceleration diversity；
- jerk diversity；
- response-mode count；
- support precision；
- support recall。

目标：

> 固定长时目标条件下产生多个合理、物理合法、交互机制不同但仍满足整体意图的局部交通行为分支。

## 10.2 固定 \(C_0,M\)，改变 \(K\)

评价 Flow 场景级行为多样性：

- unique joint mode；
- K-space coverage；
- closed-loop terminal diversity；
- event-frequency coverage；
- risk-variable coverage。

## 10.3 多层随机性消融

至少包含：

1. deterministic Flow + deterministic diffusion + deterministic response；
2. stochastic K only；
3. stochastic K + diffusion；
4. stochastic K + diffusion + response latent；
5. full model + correlated latent transition。

---

# 11. 干预有效性评估目标

每个干预均使用：

\[
	ext{same snapshot}
+
	ext{same random seed}
\]

进行 paired evaluation。

必须报告：

- 方向正确率；
- 剂量单调性；
- far/near 局部性；
- 首次响应延迟；
- 响应幅值分布；
- 自然响应覆盖率；
- recovery time；
- post-intervention gap；
- TTC recovery；
- speed recovery。

---

# 12. 必须维持的安全与因果约束

## 12.1 禁止 future ego leakage

HiQR 不得读取：

- future ego control；
- future ego trajectory；
- future ego state；
- ADS planned path。

## 12.2 已实现动作可以进入历史

允许使用：

- 已执行 ego 控制；
- 已实现 ego 状态；
- 当前和过去联合状态。

但优先目标是逐渐减少对显式 ADS 控制接口的依赖，使世界模型主要依靠已实现交通状态判断干预。

如果后续能够稳定从 ego 状态历史恢复干预强度，可移除 `committed_ego_controls` 网络输入。

## 12.3 soft plan 不能硬执行

无论 Diffusion soft plan 多精确，都必须允许因 ADS 干预发生偏移。

最终背景控制必须来自：

\[
a_t^{bg}
=
	ext{soft prior}
+
	ext{reactive residual}.
\]

## 12.4 不允许通过扩大随机噪声伪造多样性

任何随机性增强必须同时满足：

- factual ADE/FDE 不显著恶化；
- action bounds 合法；
- jerk 合理；
- collision rate 合理；
- TTC/DRAC/gap 分布合理；
- Energy Score 改善；
- support precision 不显著下降。

---

# 13. 目标验收门槛

以下门槛用于指导研发，不代表最终论文必须逐项使用完全相同数值。

## 13.1 Factual fidelity

相对当前正式模型：

- ADE 恶化不超过 15%；
- FDE 恶化不超过 20%；
- P95 displacement error 恶化不超过 20%；
- 物理合法率保持 100%。

## 13.2 固定条件随机性

相对当前模型：

- Energy Score 明显改善；
- fixed-\(K\) terminal pairwise distance 至少提升 2–3 倍；
- 不通过扩大物理噪声实现；
- acceleration/jerk KS 不显著恶化；
- 有效 behavior mode 数增加。

## 13.3 干预方向

目标：

- braking direction success ≥ 0.95；
- acceleration direction success ≥ 0.95；
- lateral intervention direction success ≥ 0.90。

## 13.4 剂量单调性

目标：

- braking monotonicity ≥ 0.95；
- acceleration monotonicity ≥ 0.95；
- lateral monotonicity ≥ 0.90。

## 13.5 局部性

要求 near 车辆响应显著大于 far 车辆，同时避免完全无传播。

建议：

\[
R_{
m far}/R_{
m near}<0.15
\]

并报告不同图距离下的响应衰减曲线。

## 13.6 自然响应覆盖率

这是当前最需要改善的指标。

目标：

- braking natural P10–P90 coverage ≥ 0.6；
- acceleration natural P10–P90 coverage ≥ 0.5；
- 最终争取 ≥ 0.7。

不得通过简单 clipping 强行提高覆盖率。

---

# 14. 推荐代码实现顺序

## Phase 1：Counterfactual Twin-Branch Training

优先实现。

任务：

- 在训练中支持 world state branch；
- 固定随机流；
- 构造 nominal / brake / accelerate / lateral 分支；
- 加入 direction / monotonicity / locality / recovery losses；
- 保持当前 stochastic architecture 不变。

目的：

> 先验证干预训练机制本身能否提高 counterfactual response。

## Phase 2：Persistent Agent Behavior Latent

将当前高相关 Gaussian agent noise 升级为显式持续行为状态。

任务：

- 新增 `AgentBehaviorState`；
- 每辆车保存 latent memory；
- 设计 transition prior；
- snapshot / restore 保存 latent state；
- 支持固定 latent 精确重放。

## Phase 3：Conditional Flow Latent Transition

将：

\[
z_{t+1}
\]

改为条件 Flow 转移。

输入：

- previous latent；
- relational scene embedding；
- soft-plan embedding；
- interaction graph；
- current risk relation。

输出：

- latent sample；
- conditional log probability。

## Phase 4：Causal Interaction Response Field

将当前简单 response gain 升级为关系条件响应场。

支持：

- longitudinal response；
- lateral response；
- multi-agent propagation；
- dynamic locality；
- latent-dependent response delay / strength。

## Phase 5：可选 Response-Style MoE

仅在前四阶段后行为模式仍不足时实施。

---

# 15. 必须进行的消融实验

至少包含：

### A. Full model

Flow + Diffusion + Stochastic Causal HiQR

### B. No response latent

删除 persistent behavior latent。

### C. Independent agent latent

保留 latent，但删除 cross-agent coupling。

### D. No counterfactual branch loss

只做自然数据事实训练。

### E. No causal response field

退回当前 scalar response gain。

### F. Deterministic response

关闭短程随机性。

### G. No diffusion soft plan

验证长时先验的重要性。

---

# 16. 论文最终应证明的四级证据链

## Level 1：事实重建

\[
	ext{Does the model reconstruct natural traffic?}
\]

指标：

- ADE/FDE；
- velocity；
- acceleration；
- jerk；
- TTC；
- DRAC；
- gap。

## Level 2：分布随机性

\[
	ext{Does the model reproduce plausible multi-modal futures?}
\]

指标：

- Energy Score；
- CRPS；
- diversity；
- mode coverage；
- support precision / recall。

## Level 3：因果响应

\[
	ext{Does the model react appropriately to realised ADS interventions?}
\]

指标：

- direction；
- monotonicity；
- locality；
- response delay；
- natural response coverage；
- recovery。

## Level 4：ADS 测试有效性

\[
	ext{Do generated worlds preserve ADS risk ranking and failure mechanisms?}
\]

最终应验证：

- 不同 ADS 的风险排序；
- failure reproduction；
- same-world A/B comparison；
- rare-event search transfer；
- 高保真仿真复验；
- Top-k dangerous scenario recall。

---

# 17. 最终方法定位

下一版本不再定位为：

> “轨迹预测精度更高的世界模型”。

而应定位为：

> **A probabilistic, stochastic, and causally reactive traffic world model for closed-loop ADS testing.**

核心贡献概括为：

### Contribution 1 — Probabilistic scenario hierarchy

Flow 显式建模场景级概率、初始状态与长时间尺度行为约束。

### Contribution 2 — Generative soft motion prior

Diffusion 提供可修改的联合长时运动先验，而不是硬执行未来。

### Contribution 3 — Stochastic causal reactive execution

HiQR 利用持续行为潜变量、交互耦合随机状态转移和因果响应场，以 25 Hz 根据已实现历史生成下一帧背景动作。

### Contribution 4 — Common-randomness counterfactual testing

通过相同场景、相同随机性和不同 ADS 干预的 twin-branch rollout，实现反事实响应训练、公平 ADS 对比与精确失效重放。

---

# 18. 最终研发原则

下一阶段必须坚持：

\[
oxed{
	ext{不要继续以降低 ADE/FDE 为主要优化目标}
}
\]

而应重点优化：

\[
oxed{
	ext{distribution fidelity}
+
	ext{behavior multimodality}
+
	ext{causal intervention response}
}
\]

最终模型应做到：

> **同一交通世界可以产生多个真实、可审计、可重放的背景行为模式；当 ADS 改变已经实现的 ego 行为后，背景车辆能够以具有驾驶员异质性、局部性、时延和剂量一致性的方式产生合理反事实响应。**

这应成为下一阶段架构与实验设计的唯一主线。

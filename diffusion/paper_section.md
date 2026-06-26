# 基于扩散模型的自然驾驶行为先验

## 1. 问题形式化

### 1.1 行为先验学习

给定 anchor-frame 场景条件 $\mathbf{c}$（包含初始交互关系和参考窗口的压缩动作/轨迹摘要），目标是学习目标车（前车/切入车）未来动作 $\mathbf{a}$ 的条件分布：

```math
p(\mathbf{a} \mid \mathbf{c}) = \int p(\mathbf{a} \mid \mathbf{c}, \mathbf{z}) \, p(\mathbf{z}) \, d\mathbf{z}
```

其中 $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$ 为潜在变量。这一形式化允许通过扩散模型进行灵活的条件生成。

### 1.2 场景类型

模型支持两类驾驶场景：
- **跟车（following）**：目标车为前车，运动为纯纵向
- **切入（cut-in）**：目标车从相邻车道切入，运动为二维

---

## 2. 数据集构建

实现中，数据读取后的跨模块公共逻辑统一由根目录 `tools/` 提供，包括归一化适配、
NPZ/JSON IO、风险配置和论文图样式。`diffusion/` 仅保留数据集封装、模型、训练、
采样和 prior 评估代码，不维护旧 `utils/` 兼容入口。

### 2.1 Anchor-frame 窗口采样

从 highD 交互事件中，以锚定帧构建训练样本。每个样本包含锚定帧初始状态和未来
$H$ 帧：

```math
\mathcal{W}_{t_a} = (\mathbf{S}_{t_a}, \mathcal{F}_{t_a})
```

其中 $\mathbf{S}_{t_a}$ 为锚定帧两车初始状态，$\mathcal{F}_{t_a} = \{t_a + 1, \dots, t_a + H\}$。当前 following 使用 $H=125$（25 Hz 下 5 秒），cut-in 使用 $H=100$（25 Hz 下 4 秒）。

窗口采样策略：
- **跟车**：以固定步长 $s = 5$ 帧滑动锚定帧
- **切入**：默认使用等长 100 步窗口，cross 前偏移为 15/20/25/30/35/45/50 帧
- 每事件最多采样 12 个窗口
- `train_val_test` 模式按记录 ID 划分训练/验证/测试；following 默认比例为 70/15/15，cut-in 默认比例为 75/15/10
- 当前 following 和 cut-in 均只维护 `train_val_test` prior；`IDM_subset/` 长尾闭环测试使用同一套已验证权重

following 的世界状态直接来自 `process_highD` 写出的
`following_event_segments.npz`。该 cache 缺失、字段不全或 `target_fps` 与配置不一致时，数据构建直接失败，
不再从 raw highD 走兼容重建分支；cut-in 仍按 scored semantic event 表回读对应 recording 和 cross-frame
窗口。

### 2.2 世界状态提取

对每个窗口，提取两车（本车和目标车）在世界坐标系下的状态矩阵：

```math
\mathbf{S}^{\text{world}} \in \mathbb{R}^{(H+1) \times 2 \times 6}, \quad \text{特征: } (x, y, v_x, v_y, a_x, a_y)
```

速度可通过原始加速度或 Savitzky-Golay 平滑后差分获取。Savitzky-Golay 滤波器的实现直接构造 Vandermonde 矩阵并通过伪逆求解：

```math
\mathbf{c} = (\mathbf{A}^T \mathbf{A})^{-1} \mathbf{A}^T \mathbf{y}, \quad A_{ij} = (i - m)^j
```

其中窗口大小为 9，多项式阶数为 2。

### 2.3 本车坐标系变换

将所有世界状态变换至当前本车坐标系（本车当前位置为原点，$x$ 轴对齐道路方向）：

```math
\begin{bmatrix} x^{\text{ego}} \\ y^{\text{ego}} \end{bmatrix} = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x^{\text{world}} - x_0 \\ y^{\text{world}} - y_0 \end{bmatrix}
```

由于 highD 数据已对齐道路方向，旋转矩阵退化为单位阵 $(\cos\theta=1, \sin\theta=0)$。

### 2.4 动作目标表示

**跟车动作表示**：
- `acceleration`：直接加速度 $a_x$（从未来状态提取）
- `jerk`：加加速度 $j_x$，由加速度后向差分计算：

```math
j_{x,t} = \frac{a_{x,t} - a_{x,t-1}}{\Delta t}
```

- 可通过 $j_x \to a_x \to v_x \to x$ 积分恢复轨迹

**切入动作表示**：
- `ax_ay`：目标车二维加速度 $[a_x, a_y]$

### 2.5 场景条件特征

模型唯一条件输入为 `scenario_conditions`。这些条件包含初始关系、对抗车动作趋势和关键轨迹形状摘要；细节变化由 diffusion latent 表达。

**跟车条件**（7 维）：

```math
\mathbf{c}_{\text{follow}} = \begin{bmatrix}
v_{x,\text{ego},0} & g_0 & \Delta v_0 & a_{x,\text{lead},0} &
\Delta v_{\text{lead},0:H} & \min_t a_{x,\text{lead},t} &
\sum_t \mathbb{1}[a_{x,\text{lead},t}<0]\Delta t
\end{bmatrix}
```

其中 $\Delta v_{\text{lead},0:H}=v_{x,\text{lead},H}-v_{x,\text{lead},0}$。

**切入条件**（10 维）：

```math
\mathbf{c}_{\text{cutin}} = \begin{bmatrix}
v_{x,\text{ego},0} & g_0 & \Delta y_0 & \Delta v_{x,0} &
v_{y,\text{target},0} & a_{y,\text{target},0} &
y_{\text{target},H}-y_{\text{ego},0} & t_{\text{cross}}-t_0 &
\Delta v_{\text{target},0:H} & \left.\frac{dy}{dx}\right|_{\text{cross}}
\end{bmatrix}
```

### 2.6 数据标准化

所有数组采用 Z-score 标准化，统计量仅从训练集计算：

```math
\hat{x} = \frac{x - \bar{x}_{\text{train}}}{\sigma_{\text{train}}}
```

标准化对象只包括 `scenario_conditions` 和 `actions`。

---

## 3. 扩散模型架构

### 3.1 去噪扩散概率模型（DDPM）

扩散模型通过前向加噪过程将数据分布逐渐变换为标准正态分布，再学习反向去噪过程。

**前向过程**（固定马尔可夫链）：

```math
q(\mathbf{a}_k \mid \mathbf{a}_{k-1}) = \mathcal{N}\left(\mathbf{a}_k; \sqrt{1 - \beta_k}\,\mathbf{a}_{k-1}, \beta_k \mathbf{I}\right)
```

其中 $\{\beta_k\}_{k=1}^{K}$ 为余弦噪声调度（$K=100$）。定义 $\alpha_k = 1 - \beta_k$，$\bar{\alpha}_k = \prod_{s=1}^{k} \alpha_s$，则任意第 $k$ 步可通过闭式采样：

```math
\mathbf{a}_k = \sqrt{\bar{\alpha}_k}\,\mathbf{a}_0 + \sqrt{1 - \bar{\alpha}_k}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})
```

**反向过程**（参数化马尔可夫链）：

```math
p_\theta(\mathbf{a}_{k-1} \mid \mathbf{a}_k, \mathbf{c}) = \mathcal{N}\left(\mathbf{a}_{k-1}; \boldsymbol{\mu}_\theta(\mathbf{a}_k, k, \mathbf{c}), \tilde{\beta}_k \mathbf{I}\right)
```

其中后验方差固定为 $\tilde{\beta}_k = \frac{1 - \bar{\alpha}_{k-1}}{1 - \bar{\alpha}_k} \beta_k$。

### 3.2 条件编码器

场景条件编码器为 MLP：

```math
\mathbf{h}_{\text{cond}} = \text{MLP}_{\text{scenario}}(\mathbf{c}), \quad \text{激活: SiLU}
```

### 3.3 FiLM 变换器去噪器

去噪网络采用 FiLM（Feature-wise Linear Modulation）条件变换器架构。

输入的噪声动作 $\mathbf{a}_k \in \mathbb{R}^{T \times d_a}$ 首先通过线性投影映射到隐藏维度，并加入可学习的时间位置编码：

```math
\mathbf{h}^{(0)} = \mathbf{a}_k \mathbf{W}_{\text{proj}} + \mathbf{E}_{\text{pos}} \in \mathbb{R}^{T \times d_{\text{hidden}}}
```

扩散时间步 $k$ 通过正弦位置编码嵌入，经 MLP 后与条件向量融合：

```math
\mathbf{e}_k = \text{MLP}_{\text{time}}(\text{SinusoidalEmbedding}(k)), \quad \tilde{\mathbf{h}}_{\text{cond}} = \mathbf{h}_{\text{cond}} + \mathbf{e}_k
```

对每层 FiLM 变换器块（$N = 3\sim4$ 层）：

```math
\begin{aligned}
\mathbf{h}^{(l)}_{\text{attn}} &= \text{MultiheadSelfAttention}(\mathbf{h}^{(l-1)}) \\
\boldsymbol{\gamma}^{(l)}, \boldsymbol{\beta}^{(l)} &= \text{MLP}_{\text{FiLM}}^{(l)}(\tilde{\mathbf{h}}_{\text{cond}}) \\
\mathbf{h}^{(l)} &= \text{LayerNorm}\left(\mathbf{h}^{(l)}_{\text{attn}} \odot (1 + \boldsymbol{\gamma}^{(l)}) + \boldsymbol{\beta}^{(l)}\right)
\end{aligned}
```

最终输出为预测噪声：

```math
\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{a}_k, k, \mathbf{c}) = \text{LayerNorm}\left(\mathbf{h}^{(N)}\right) \mathbf{W}_{\text{out}} \in \mathbb{R}^{T \times d_a}
```

### 3.4 训练目标

**噪声预测损失**（MSE）：

```math
\mathcal{L}_{\text{noise}} = \mathbb{E}_{k, \mathbf{a}_0, \boldsymbol{\epsilon}}\left[ \|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{a}_k, k, \mathbf{c})\|_2^2 \right]
```

**辅助损失**（可选）：

$x_0$ 预测的 L1 损失：

```math
\mathcal{L}_{x_0} = \|\mathbf{a}_0 - \hat{\mathbf{a}}_0\|_1
```

其中 $\hat{\mathbf{a}}_0 = \frac{1}{\sqrt{\bar{\alpha}_k}} (\mathbf{a}_k - \sqrt{1 - \bar{\alpha}_k}\,\hat{\boldsymbol{\epsilon}}_\theta)$。

平滑正则化：

```math
\mathcal{L}_{\text{smooth}} = \frac{1}{T-1}\sum_{t=1}^{T-1} \|\hat{\mathbf{a}}_{0,t+1} - \hat{\mathbf{a}}_{0,t}\|_2^2
```

轨迹运动学一致性损失（切入模式）：

```math
\mathcal{L}_{\text{traj}} = \|\mathbf{x}^{\text{pred}} - \mathbf{x}^{\text{true}}\|_1 + \|\mathbf{v}^{\text{pred}} - \mathbf{v}^{\text{true}}\|_1 + \mathcal{L}_{\text{cross}} + \mathcal{L}_{\text{end}}
```

其中 $\mathcal{L}_{\text{cross}}$ 和 $\mathcal{L}_{\text{end}}$ 分别为跨线时刻和切入结束时刻的横向误差。

总损失：

```math
\mathcal{L} = \mathcal{L}_{\text{noise}} + w_{x_0}\mathcal{L}_{x_0} + w_{\text{smooth}}\mathcal{L}_{\text{smooth}} + w_{\text{traj}}\mathcal{L}_{\text{traj}}
```

默认权重：$w_{x_0} = 0.10$，$w_{\text{smooth}} = 0.01$。

---

## 4. 确定性与随机采样

### 4.1 DDIM 采样

为保证潜在子集模拟的可重复性，推理采用确定性 DDIM（Denoising Diffusion Implicit Models）采样。DDIM 反向步为：

```math
\mathbf{a}_{k-1} = \sqrt{\bar{\alpha}_{k-1}} \cdot \hat{\mathbf{a}}_0 + \sqrt{1 - \bar{\alpha}_{k-1} - \sigma_k^2} \cdot \hat{\boldsymbol{\epsilon}}_\theta + \sigma_k \boldsymbol{\epsilon}
```

其中确定性模式 $\sigma_k = 0$，消去随机项。$\hat{\mathbf{a}}_0$ 由噪声预测导出：

```math
\hat{\mathbf{a}}_0 = \frac{\mathbf{a}_k - \sqrt{1 - \bar{\alpha}_k}\,\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{a}_k, k, \mathbf{c})}{\sqrt{\bar{\alpha}_k}}
```

可选的 $\hat{\mathbf{a}}_0$ 裁剪（`maybe_clip_x0`）将预测值约束在 $[-1, 1]$ 归一化范围内。

推理步数可通过子采样减少（默认 50 步）：从 $\{0, \dots, K-1\}$ 中均匀抽取 $\tau$ 个时间步，按降序排列执行去噪。

**关键性质**：给定相同背景 $\mathbf{c}$ 和初始噪声 $\mathbf{z} = \mathbf{a}_K \sim \mathcal{N}(0, \mathbf{I})$，DDIM 确定性采样始终产生相同的动作序列：

```math
\mathbf{a} = \text{DDIM}(\mathbf{c}, \mathbf{z}; \theta)
```

### 4.2 DDPM 随机采样

如需多样本生成，可采用完整 DDPM 采样（$\sigma_k = \sqrt{\tilde{\beta}_k}$），每次从同一背景生成不同动作。

---

## 5. 动作到轨迹的运动学积分

### 5.1 跟车积分

从加加速度 $j_x$（或加速度 $a_x$）积分得到轨迹：

```math
\begin{aligned}
a_{x,t} &\leftarrow \text{clip}\left(a_{x,t-1} + j_{x,t} \cdot \Delta t, a_{x,\min}, a_{x,\max}\right) \\
v_{x,t} &\leftarrow \max\left(v_{x,t-1} + a_{x,t} \cdot \Delta t, 0\right) \\
x_t &\leftarrow x_{t-1} + v_{x,t-1} \cdot \Delta t + \frac{1}{2} a_{x,t} \cdot \Delta t^2
\end{aligned}
```

运动学约束：$a_{x} \in [-8.0, 4.0]\ \text{m/s}^2$，$|j_x| \leq 12.0\ \text{m/s}^3$。

### 5.2 切入加速度积分

从二维加速度 $[a_x, a_y]$ 积分：

```math
\begin{aligned}
a_{x,t} &\leftarrow \text{clip}(a_{x,t}, a_{x,\min}, a_{x,\max}) \\
a_{y,t} &\leftarrow \text{clip}(a_{y,t}, -a_{y,\text{abs}}, a_{y,\text{abs}}) \\
v_{x,t} &\leftarrow \text{clip}\left(v_{x,t-1} + a_{x,t} \cdot \Delta t, v_{\min}, v_{\max}\right) \\
v_{y,t} &\leftarrow v_{y,t-1} + a_{y,t} \cdot \Delta t \\
x_t &\leftarrow x_{t-1} + v_{x,t-1} \cdot \Delta t + \frac{1}{2} a_{x,t} \cdot \Delta t^2 \\
y_t &\leftarrow y_{t-1} + v_{y,t-1} \cdot \Delta t + \frac{1}{2} a_{y,t} \cdot \Delta t^2
\end{aligned}
```

约束：$|a_y| \leq 4.0\ \text{m/s}^2$，$|j_y| \leq 8.0\ \text{m/s}^3$，$v \in [0, 50]\ \text{m/s}$。

## 6. 训练流程

采用 AdamW 优化器（$\beta_1=0.9$, $\beta_2=0.999$，权重衰减 $=10^{-4}$），余弦学习率衰减从 $3\times10^{-4}$ 至 $5\times10^{-5}$。当前 following 默认训练 250 轮，cut-in 默认训练 1000 轮。

每轮执行：
- 训练：随机时间步 $k \sim \mathcal{U}(0, K-1)$，计算 $\mathcal{L}$，反向传播，梯度裁剪（$\|\nabla_\theta\| \leq 1.0$）
- 验证（`train_val_test`）：与训练相同但无梯度，监控验证噪声 MSE
- 固定噪声评估：固定噪声种子和时间步 $\{0, 25, 50, 75, 99\}$，计算确定性损失以追踪采样质量

实现中固定使用 `split.mode: train_val_test`。checkpoint 文件名保留该训练方式标签：

```text
best_noise_mse_train_val_test.pt
final_train_val_test.pt
```

后续 `IDM_subset/` 默认使用 `best_noise_mse_train_val_test.pt`，保证长尾闭环测试使用经过 held-out
评估的扩散先验。

---

## 7. 评估体系

### 7.1 分布指标

- **Wasserstein 距离**：$\mathcal{W}_1(P, Q) = \int |F_P^{-1}(u) - F_Q^{-1}(u)| \, du$
- **Kolmogorov-Smirnov 统计量**：$D_{KS} = \sup_x |F_P(x) - F_Q(x)|$
- **直方图 L1 距离**：对各动作维度比较生成与真实分布的离散直方图

### 7.2 物理可行性

- 动作裁剪率：超出物理边界的动作分数
- 速度负值率：$v < 0$ 的比例
- 加加速度违规率：$|j| > j_{\max}$ 的比例
- 横向加速度违规率
- 横向位移违规率（切入模式）

### 7.3 交互自然度

从生成轨迹计算交互指标（车间距、TTC、THW、相对速度）并与真实数据对比分布距离。

### 7.4 条件生成质量

- 高斯负对数似然（Gaussian NLL）
- 连续分级概率评分（CRPS）
- 集成的均方误差和 L1 误差
- 覆盖区间

### 7.5 轨迹重构精度

- 位置和速度的均方误差
- 动态时间规整（DTW）距离
- 终点误差
- 跨线时刻和切入结束时刻的横向位移误差

---

## 8. 可视化输出

生成以下诊断图：
- 加速度、加加速度、速度、横向偏移的分布直方图（真实 vs 生成）
- 相空间散点图（$v_x$-$a_x$，$\Delta v$-$\text{gap}$，$v_y$-$a_y$，$\Delta y$-$v_y$）
- 示例轨迹展开时间序列图
- 轨迹重构误差直方图

---

## 9. 实现边界

`diffusion/` 只包含数据集构建接口、条件扩散模型、训练流程和 prior 评估。跨模块 IO、归一化适配、
论文图样式和 frozen prior 加载统一复用 `tools/`，不维护旧 `utils/` 兼容入口。模型类的
`forward` 方法是 PyTorch 运行接口；训练脚本和评估脚本是公开入口。checkpoint、训练数据 NPZ、
归一化 NPZ 和生成样本 NPZ 均可由脚本重建，不作为核心方法代码保存。

# QR-WM：目标、训练与架构

## 0. 模型定位

QR-WM（Query-Refine World Model）是当前正式的 highD 背景交通世界模型。它独立于 RAMP-WM、FIRM-WM、Semi-Markov WM 与 CAT-TopK：不加载这些基线的 checkpoint，只生成六个背景车辆槽位的后续动作与状态演化。四个基线仍保留用于统一条件下的横向比较。

模型的直接输出是未来背景动作序列 `background_future_actions_before_refinement` 与 `background_future_actions`、精炼前后背景状态、动作来源/有效性 mask 和下一响应周期的 `scene_memory`。完整重建 rollout 另返回 `predicted_states`、`target_states`、`target_valid` 及当前响应实际执行的 `executed_background_action_masks`；后者不是下一轮计划缓冲区的 mask。

## 0.1 训练/ADS 的统一时间协议

highD 的“6 秒”自然片段实际保存 150 个 25 Hz 状态点 `S0..S149`，因此可观测的状态转移只有 149 个、总跨度为 5.96 s。QR 缓存保留全部这些原始点：

\[
\underbrace{S_0\to\cdots\to S_{25}}_{25\ \text{START transitions}=1.00\ \mathrm{s}}
\quad+\quad
\underbrace{S_{25}\to\cdots\to S_{149}}_{124\ \text{ROLL transitions}=4.96\ \mathrm{s}}.
\]

训练的完整阶段、held-out 重建和 Flow×QR/ADS 评测都采用这一相同定义，并且每条轨迹均由 `encode_start(C0,map)` 启动。`B0` 只直接约束 START；ROLL 不再读取原始 `B0`。没有虚构 `S150`，因而不能把它宣称为完整 5.00 s 的自由 ROLL；若需要该定义，原始窗口必须至少提供 151 个状态点。

`anchor_frame` 是自然驾驶片段的起点。因此 START 的严格含义是 **segment-start behavior reconstruction（片段起始行为重建）**，并不自动表示 EVT 风险事件起始；风险峰值可能在窗口更晚的时刻发生。所有论文、报告与图表均应使用此前者表述，除非另行按风险事件起始重锚定数据。

该缓存位于 `results/highd_world_model/training_data/qr_sequence_cache`。

## 1. 运行边界与因果约束

QR-WM 使用固定 highD 场景布局 `[ego, six background slots]`。单车状态为 `[x, y, vx, vy, ax, ay]`；单个背景动作是 `[longitudinal_acceleration, yaw_rate]`。模型只预测背景车辆动作。

响应时刻 `t` 的推理分布为：

\[
p_\theta(A_t^{bg}\mid H_t,S_t,M,m_t,A_{t-1}^{bg},z),
\qquad A_t^{bg}\in\mathbb{R}^{H\times6\times2}.
\]

其中 `H_t` 是已发生历史，`S_t` 是当前状态，`M` 是地图，`m_t` 是唯一场景记忆，`A_{t-1}^{bg}` 是未执行背景动作序列，`z` 是行为 latent。推理和动作生成路径不接收 ADS 当前动作、ADS 内部计划、ego 未来动作、ego 未来状态、traffic light 或未来背景标签，也不加载 RAMP/FIRM checkpoint。ADS 的影响只能在其动作已形成新的 ego 观测状态，并进入 `H_{t+1}` 后被模型感知。

代码有两条互补运行路径：

- `rollout_reconstruction(...)` 是 highD 监督重建路径。它先生成背景动作，再把对应的日志 ego 状态写入下一已发生历史帧。
- `QRWorldModelEnvironment` 是在线路径。外部 ADS 每 0.04 秒调用 `step(ads_action)`，环境以 `f_ego` 推进 ego，并与背景车辆同步积分；QR-WM 只在每 0.2 秒规划边界读取已经发生的联合历史。`ads_action` 绝不进入 QR-WM 网络。

## 2. 场景编码器

`QueryRelationalSceneEncoder` 的输入是历史、当前状态、有效 mask、ego mask、map polyline、地图点有效 mask 与 lane graph edges。

ROLL 中，历史最多为 25 帧，经 `temporal_layers` 个 temporal Transformer 编码（正式配置为 1 层）。每车状态特征包括归一化的位置、速度、加速度、航向角正余弦、ego 标记和有效标记；最后一个有效 temporal token 再与当前状态 token 相加。

当前 agent token 随后经过 `attention_layers` 个 relation-aware 多头注意力层（正式配置为 2 层，`num_heads=4`）。注意力偏置来自相对位置/速度、航向和车道关系、closing speed、TTC、DRAC。编码器还会构造 map polyline token、用车道拓扑更新它们，并执行 agent/map 多头交叉注意力，输出：

\[
(E_t,s_t,M_t,V_t^{map}),
\]

其中 `E_t` 含 7 个 agent token，`s_t` 是池化 scene token，`M_t` 是 map token 集合。

START 使用专用 `encode_start(C0, map)` 路径，复用当前状态、关系和地图编码，但 temporal 分量为零；它不会复制当前状态来伪造 25 帧历史。

## 3. START：Flow C0+B0

冻结 Flow 条件有 76 维：

\[
[C0,B0]=[\text{ego}(4),\text{six relative background states}(36),\text{six action summaries}(36)].
\]

共享 adapter `start_state_from_flow_tensor` 是 QR 唯一的 Flow 张量解码器。它把 Flow 坐标变为 `[B,7,6]` 场景状态和 `[B,6,6]` 的原始 B0 摘要。Flow 的背景速度是相对 ego 的量，因此进入编码器前必须还原：

\[
v_x^{bg}=v_x^{ego}+\Delta v_x,\qquad
v_y^{bg}=v_y^{ego}+\Delta v_y.
\]

无效槽位会归零并保持无效。每个背景槽位的 B0 含六个首秒摘要：纵/横速度变化、纵向平均/最小/末端加速度和横向平均加速度。

`initialize_start(...)` 只在 episode 开始时执行一次：

1. 用 `encode_start` 编码 C0 和地图；
2. 将 B0 投影为平滑的 25 帧背景动作锚点；
3. 将 B0 投影为逐车 behavior seed；
4. 用动作锚点和零状态变化初始化唯一 `PersistentSceneMemory`；
5. 从条件行为先验取样或选取均值；
6. 返回 `scene_memory`、`behavior_latent` 和 `start_anchor_actions`。

第一次 `plan_step` 中，fresh 动作与 B0 锚点采用随时间衰减的凸组合：

\[
A_0^{pre}(\tau)=(1-\alpha_\tau)A_0^{fresh}(\tau)+\alpha_\tau A_0^{B0}(\tau),
\]

其中 `alpha` 在 25 帧内从 `start_anchor_mix=0.75` 线性下降到 0。原始 B0 不作为 ROLL 参数，也不会在 START 后再次读取。

`FlowStartMetadata` 另行保存 slot mask、地图张量、primary-risk slot、event structure、mask pattern、event-structure log probability、conditional log probability 与联合 `log_prob`。它会校验

\[
\mathrm{log\_prob}=\mathrm{event\_structure\_log\_prob}+\mathrm{conditional\_log\_prob},
\]

并将这些字段保留在环境 trace 与 Flow 组合审计中。

## 4. ROLL：memory、行为与动作序列

### Persistent scene memory

模型只维护一个 memory：

\[
m_t=f(m_{t-1},s_t,\operatorname{pool}(E_t),A_{t-1}^{bg},\Delta S_t).
\]

实现将上一背景动作序列的均值和绝对均值、当前 agent token 的池化结果以及全场景状态变化的均值送入 GRU cell。不存在 `ContinuousTrafficMemory`、`world_memory`、`world_initializer` 或 `world_update`；memory 不接收 ADS 动作。

### Behavior prior 与训练 posterior

`BehaviorPrior` 为每个 agent 提供以 agent token、scene token 和 memory 为条件的 Gaussian prior。START 时 B0 behavior seed 会平移 prior 均值。推理中，`deterministic=True` 取 prior 均值；否则从该 Gaussian 取样，它是 rollout 的唯一随机来源。

训练期额外使用 posterior 及 KL/重建损失。posterior 的特征只从未来背景状态监督中提取：实现先将 ego 未来状态替换为当前 ego 状态，再在所有 behavior loss 中排除 ego 槽位。因此，未来 ego 标签不会条件化背景动作；未来背景监督只用于训练期 latent 后验，不进入推理接口。

### 滚动背景车辆未来动作序列

`plan_step(...)` 先生成 `fresh = fresh_plan(E_t,m_t,z)`，其形状为 `[B,25,6,2]`。后续响应中，上一序列前 5 帧已执行，剩余 20 帧与当前 fresh 序列末 5 帧连接为 carried candidate；之后它与完整 fresh 序列线性混合：

\[
A_t^{pre}=(1-\lambda)A_t^{fresh}+\lambda\,[A_{t-1}^{bg}[5:],A_t^{fresh}[-5:]],
\]

正式配置中 `lambda=buffer_carry_mix=0.35`。这是连续性偏置，不是不可修改的硬 carry 区。

`background_future_action_masks` 含四个 `[B,25,6]` mask：

- `carried`：来源于上一序列剩余区的动作；
- `appended`：新生成的尾部区；
- `refinable`：仍未执行且有效、可精炼的背景动作；
- `valid`：预测时域内的有效背景槽位。

完整 rollout 额外给出 `executed_background_action_masks`，表示本响应实际送入动力学的前 5 帧有效性；它不属于下一动作序列。

## 5. 确定性联合残差精炼

`JointAgentTimeRefiner` 先由背景 agent token、memory 和背景 behavior latent 生成 fresh 联合动作序列。残差精炼器接收当前动作序列、由动作积分的背景计划状态、观测 agent token、scene token、memory、behavior latent、map token 和有效 mask。

每个 `_AgentTimeBlock` 依次执行：

1. 每辆车的时间维多头自注意力；
2. 每个未来时刻跨全部 7 个 agent token 的多头自注意力；
3. agent-time token 对 scene、memory、map token 的多头交叉注意力；
4. 前馈更新。

ego token 是当前/历史场景编码得到的 token 在时域上的重复。它不包含 ego 动作或 ego 未来状态，且不会输出 ego 动作残差。

残差输出只作用于有效背景动作，内部固定动作尺度为 `(1.5, 0.15)`。精炼采用减法和动作裁剪：

\[
A^{(i+1)}=\operatorname{clip}\left(A^{(i)}-R_\theta(A^{(i)},\widehat S^{bg},E_t,s_t,m_t,z,M_t)\right).
\]

正式配置为 `refinement_iterations=2`，两次调用共享 refiner 参数。模型没有 noise schedule、noise-level embedding、corruption、denoising loss、diffusion objective 或反向扩散采样。

## 6. 动力学与状态推进

`KinematicTrafficDynamics` 是可微的 unicycle/single-track 兼容动力学。它将 `[a,yaw_rate]` 积分为下一六维状态；训练监督则先把 highD 笛卡尔加速度标签投影到相同动作坐标。

重建过程中，背景动作按帧积分。每个响应的动作计划确定后，日志 ego 状态才逐帧写入生成历史。在线环境中，`QRWorldModelEnvironment.step(ads_action, ego_valid=True)` 是一个 0.04 秒物理 tick：在 0.2 秒边界先以已发生联合历史规划背景动作，然后将 ADS 的 `[longitudinal_acceleration, yaw_rate]` 与当前背景计划帧组成联合控制，并同步积分 ego 和背景车辆。`advance_response(ads_actions)` 是严格等价的 1--5 tick 便捷接口；最后一个响应自然只有 4 tick。ADS 动作只属于环境 `f_ego`，不会输入编码器、memory、prior 或 refiner；其影响只能通过下一规划边界已发生的 ego 状态/历史被模型感知。

## 7. 训练目标与记录

正式配置使用完整 immutable cache、40 个 epoch（`8 + 12 + 20`）：前两个阶段的 train/validation batch size 为 96，5.96 秒精炼阶段为 64；每 5 个 response 进行一次 truncated backpropagation。完整阶段有 30 个 5 Hz 响应，其中最后一个只监督 4 个物理 tick。`forward_training` 对每个自然片段执行一次完整 START→ROLL rollout：第一个响应走 `encode_start(C0,map)`，其后响应使用此前已生成的 25 Hz 历史自然进入 temporal ROLL。训练、验证、checkpoint FDE 选择和正式 held-out 评测均使用这个相同路径。

不存在 50/50 的伪历史 ROLL 半批。独立的 history-conditioned ROLL 辅助训练当前未启用；若将来启用，必须从片段内部的真实切点（例如 1/2/3 秒）取样，并提供切点前真实或明确生成的 25 帧历史。

完整 START→ROLL rollout 的目标包含：

- 已执行响应段的背景位置、速度、动作误差；
- 完整 1 秒背景状态和动作误差；
- 鼓励精炼后位置误差低于精炼前的 hinge 项；
- 相邻动作序列重叠、交互和物理损失；
- behavior KL、behavior reconstruction、diversity floor；
- `start_summary_weight=0.10` 乘以有效槽位上的 `L1(\widehat B0,B0)` 摘要损失。

TensorBoard 每个优化 batch 写入 `batch/train/loss`；每个 epoch 写入全部有限的 `train_*`、`val_*` 标量、rollout 时长与 `selection/validation_fde_m`。最优 checkpoint 只在完整 5.96 秒 stage 中按真正 START 初始化的验证 FDE 选择，并写入 cache format、149/25/124 帧、START 初始化和无独立 ROLL 辅助训练的协议字段。

## 8. 评测、Flow 组合与 checkpoint

`evaluate_qr_world_model` 在 held-out 重建集上使用确定性和采样 behavior latent；两者都明确以 `start_mode=True` 进入 `encode_start`，随后才使用 temporal ROLL。它报告轨迹误差、minADE/minFDE、多样性、collision/gap/TTC/DRAC、速度/加速度/jerk 分布 KL、精炼位置增益及 `background_future_action_overlap_l1`。

`evaluate_flow_composition` 从全部 highD EVT-tail replay 中匹配固定 Flow tail starts。冻结 Flow 按 slot mask 和主风险槽位采样；在高D唯一的直道路型 cohort 内，以 Flow 初始 ego 纵向速度最近邻匹配 replay，并将其平移到 Flow 起点。评测从相邻 25 Hz replay 速度状态恢复 ego 的 `[a,yaw_rate]`，仅由环境动力学应用，绝不输入 QR-WM；它按 149 个 tick 输出 `1.00 s START + 4.96 s ROLL` 的生成分布，不是任意 ADS 的配对重建。

一个 `QRWorldModelEnvironment` 只维护一个世界。该世界的随机变量是 START 行为 latent 的标准正态扰动：用 `WorldRandomness(seed=...)` 可重现地生成，或直接以 `behavior_standard_normal` 注入；给定它以后，后续响应没有隐式随机数。`BatchedQRWorldModelEnvironment` 一次推进 96 个 Flow 起点的 4 条独立世界（384 条），只共享张量计算；每行有独立 seed/latent，绝不共享场景状态、latent、记忆、计划或 ego。它的 `step`/`advance_response` 与单世界接口使用同名审计字段并只增加 batch 维，因此不会再读取私有计划缓存。它写出 `flow_start_audit.npz` 和 `flow_composition_evaluation.json`，保留 Flow 元数据、速度匹配误差、world seed 与哈希。

checkpoint 保存模型名 `query_refine_world_model`、model config、state dict 和 Flow schema hash。

## 9. 当前正式产物与复现入口

训练完成后，正式 QR 运行位于 `results/highd_world_model/qr_world_model/`，并使用本文件定义的唯一训练和评测协议。

- `training_summary.json`、`training_progress.json` 和 `training_history.csv` 记录完整训练；`current_training_curves.png` 由 `world_model/scripts/plot_qr_training_curves.py` 从 CSV 与 TensorBoard 记录重绘。
- `qr_world_model_evaluation_summary.json` 是完整 held-out 重建评测（24,216 条测试序列）。
- `results/highd_world_model/long_tail_reproduction/` 专用于 Flow × QR-WM 端到端分布评测，保存 `flow_composition_evaluation.json`、`flow_start_audit.npz` 和协议清单。

常用入口如下：

```bash
python world_model/scripts/train_qr_world_model.py
python world_model/scripts/train_qr_world_model.py --resume
python world_model/scripts/test_qr_world_model.py
python world_model/scripts/evaluate_qr_long_tail.py
python world_model/scripts/plot_qr_training_curves.py
```

`evaluate_qr_long_tail.py` 是唯一的正式长尾入口：它先执行非配对 Flow×QR 分布评测并生成图表，再执行成对 START/ROLL 重建审计。两类结果仍分别写入 `flow_composition_evaluation.json` 与 `reconstruction_validation_audit.json`，不可互相替代。

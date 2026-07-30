# Query-Refine World Model（QR-WM）

## 定位

QR-WM 是独立于 RAMP/FIRM 的高维背景交通世界模型。它以固定六辆背景车槽位建模并输出多模态、闭环、受 ADS ego 行为条件化的未来交通演化；不输入 traffic light，也不修改基线实现。

## 设计目标

QR-WM 的核心不是“共享 scene 条件下逐车预测”，而是联合建模整个未来控制张量：

\[
U\in\mathbb{R}^{H\times N\times2},\quad U=[a,\dot\psi].
\]

模型应同时学习同车连续行为、车辆间竞争/让行/连锁制动、道路约束以及多个背景车辆对 ego 干预的协同响应。

## 实际架构

```text
START: C0 + B0 + map; ROLL: observed history + map + ADS controls
              |
relation-aware multi-head scene encoder
              |
persistent scene memory + CVAE behavior prior
              |
joint agent-time control planner/refiner
              |
background future control buffer [H, 6, 2]
              |
differentiable closed-loop traffic dynamics
```

START encoder 只编码 C0 与地图，不复制状态伪造 25 帧历史；ROLL encoder 才编码真实已发生历史。唯一的 persistent scene memory 保存交通交互、已执行计划和 ego 响应；不维护重复的 continuous/world memory。joint refiner 使用时间维 attention、车辆维 attention、scene/map cross-attention，并把由 ADS 控制积分得到的 ego future states 作为不可修改 agent token。

## Flow 组合

冻结 Flow 生成 `(C0,B0,map)` 时统一经项目 START adapter 解码：背景速度以 `v_bg=v_ego+Δv_bg` 还原为绝对速度。**START** 中 B0 仅初始化背景行为 latent、scene memory 和首个 25-frame background control buffer；该 buffer 以随时间衰减的凸组合融合 B0 控制先验与新计划。**ROLL** 只读取已发生状态、scene memory、移位后的背景 buffer 和 ADS 控制。

`rollout_reconstruction` 是唯一允许 highD ego replay 的重建接口。`rollout(..., ego_future_controls=...)` 和 `rollout_from_flow` 都以动力学推进 ego，不会用真实未来轨迹覆盖 ADS 状态。在线测试使用 `QRWorldModelEnvironment.reset_from_flow(C0,B0,metadata)`、`observe()`、`step(ego_action)`；每次 `step` 只接收下一个 0.2 s 的五帧控制，并以末控制保持补全内部一秒 planning horizon。

高D 训练、验证和测试均使用冻结 Flow schema 派生的 B0 sidecar，并记录 schema hash。每个训练 batch 平衡包含 50% 无历史 START 样本和 50% 有历史 ROLL 样本；首秒生成状态重新汇总为 `B̂0` 并以 `L1(B̂0,B0)` 约束。Flow 组合运行还保留 slot mask、primary risk vehicle、事件结构、条件概率及 `log q_Flow`，以支持后续重要性加权与分布审计。

## 精炼与评测

future buffer 严格表示未执行背景控制，并提供 carried、appended、refinable、valid 和 executed masks。训练时采用 noise-conditioned amortized denoising；推理只用 1--2 次零噪声精炼，不宣称完整 diffusion chain。

正式训练预算为 40 epoch，TensorBoard 记录训练/验证损失、FDE、denoising 和 buffer 指标。评测覆盖 ADE/FDE/minADE/minFDE、diversity、collision、TTC/DRAC、运动分布和长期 buffer consistency。当前没有可加载的 QR 训练 artifact；完成全量训练和测试前不作正式横向结论。

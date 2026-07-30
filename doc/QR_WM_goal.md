# Query-Refine World Model（QR-WM）

## 定位

QR-WM 是独立于 RAMP/FIRM 的高维背景交通世界模型。它以固定六辆背景车槽位建模并输出多模态、闭环、受已观测 ego 状态和历史条件化的未来交通演化；不输入 traffic light、ADS 动作或 ego 未来序列，也不修改基线实现。

## 设计目标

QR-WM 的核心不是“共享 scene 条件下逐车预测”，而是联合建模整个背景车辆未来动作张量：

\[
U\in\mathbb{R}^{H\times N\times2},\quad U=[a,\dot\psi].
\]

模型应同时学习同车连续行为、车辆间竞争/让行/连锁制动、道路约束以及多个背景车辆对 ego 干预的协同响应。

## 实际架构

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

START encoder 只编码 C0 与地图，不复制状态伪造 25 帧历史；ROLL encoder 才编码真实已发生历史。唯一的 persistent scene memory 保存交通交互、既有背景动作和观测到的状态变化；不维护重复的 continuous/world memory。joint refiner 使用时间维 attention、车辆维 attention 与 scene/map cross-attention；ego 仅作为由当前/历史状态编码得到的不可修改 agent token，不输入动作或未来状态。

## Flow 组合

冻结 Flow 生成 `(C0,B0,map)` 时统一经项目 START adapter 解码：背景速度以 `v_bg=v_ego+Δv_bg` 还原为绝对速度。**START** 中 B0 仅初始化背景行为 latent、scene memory 和首段 25-frame 背景车辆未来动作序列；该序列以随时间衰减的凸组合融合 B0 动作先验与新生成动作。**ROLL** 只读取已发生状态、scene memory 和移位后的背景车辆未来动作序列。

`rollout_reconstruction` 用于日志 ego state replay：每段背景动作生成后才写入下一帧已发生的 ego 状态，未来 ego 不参与当前规划。在线测试使用 `QRWorldModelEnvironment.reset_from_flow(C0,B0,metadata)`、`observe()`、`step(ego_state)`；ADS 在环境外推进自身动力学，并每 0.2 s 提供当前观测到的 `[6]` ego 状态。

高D 训练、验证和测试均使用冻结 Flow schema 派生的 B0 sidecar，并记录 schema hash。每个训练 batch 平衡包含 50% 无历史 START 样本和 50% 有历史 ROLL 样本；首秒生成状态重新汇总为 `B̂0` 并以 `L1(B̂0,B0)` 约束。Flow 组合运行还保留 slot mask、primary risk vehicle、事件结构、条件概率及 `log q_Flow`，以支持后续重要性加权与分布审计。

## 精炼与评测

背景车辆未来动作序列严格表示未执行的背景动作，并提供 carried、appended、refinable、valid 和 executed masks。预动作序列由 carried 与 appended 区组成，再经两次共享参数的确定性联合残差精炼；多模态随机性仅来自 CVAE behavior latent。

正式训练预算为 40 epoch，TensorBoard 记录训练/验证损失、FDE、背景车辆未来动作序列和 mask 指标。评测覆盖 ADE/FDE/minADE/minFDE、diversity、collision、TTC/DRAC、运动分布和长期动作序列一致性。当前没有可加载的 QR 训练 artifact；完成全量训练和测试前不作正式横向结论。

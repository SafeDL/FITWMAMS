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
history + map + ADS ego controls
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

scene encoder 使用 temporal、多头 relation-aware agent attention 和 map cross-attention。唯一的 persistent scene memory 保存交通交互、已执行计划和 ego 响应；不维护重复的 continuous/world memory。joint refiner 使用时间维 attention、车辆维 attention、scene/map cross-attention 与 ego-control conditioning。

## Flow 组合

冻结 Flow 生成 `(C0,B0,map)`。在 **START**，B0 仅用于初始化背景行为 latent、scene memory 和首个 25-frame background control buffer；在 **ROLL**，模型只读取已发生状态、scene memory、移位后的背景 buffer 和 ADS ego controls。公开接口 `rollout_from_flow` 接收 `[B,T,2]` ego controls，明确禁止背景模型改写 ego 行为。

高D 训练、验证和测试均使用冻结 Flow schema 派生的 B0 sidecar，并记录 schema hash。这样 Flow 负责带显式密度的初始条件，QR-WM 负责条件于 ADS 行为的动态交通演化。

## 精炼与评测

future buffer 严格表示未执行背景控制，并提供 carried、appended、refinable、valid 和 executed masks。训练时采用 noise-conditioned amortized denoising；推理只用 1--2 次零噪声精炼，不宣称完整 diffusion chain。

正式训练预算为 40 epoch，TensorBoard 记录训练/验证损失、FDE、denoising 和 buffer 指标。评测覆盖 ADE/FDE/minADE/minFDE、diversity、collision、TTC/DRAC、运动分布和长期 buffer consistency。现有 4-epoch 历史 artifact 与当前架构不兼容，不能加载或用于正式横向结论。

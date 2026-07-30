# QR-WM Training Architecture：Query-Refine World Model 模块设计说明

## 1. 训练定位

QR-WM 基于 FITWMAMS 中 RAMP-WM/FIRM-WM 作为 baseline，但采用独立实现。

当前训练阶段保持与 RAMP/FIRM 一致的数据形式：

-   输入：历史交通状态 $S_{history}$
-   监督：未来真实轨迹 $S_{future}$

Normalizing Flow 不参与 QR-WM
主训练，仅作为未来推理阶段的初始场景采样模块。

当前不引入 EVT，但模型预留未来风险条件接口。

------------------------------------------------------------------------

## 2. 总体训练架构

    Historical Traffic Scene
              |
              v
    Query-centric Relational Scene Encoder
              |
              +----------------+
              |                |
    Agent-specific       Persistent World
    Scene Latent          Memory
              |                |
              +----------------+
                       |
                       v
              Behavior Prior Module
                       |
                       v
           Multimodal Future Planner
                       |
                       v
            Future Trajectory Buffer
                       |
                       v
           Trajectory Refinement Module
                       |
                       v
              Future Trajectory Output

------------------------------------------------------------------------

# 3. Query-centric Relational Scene Encoder

## 输入

保持 RAMP/FIRM 数据接口：

-   agent history states
-   agent-agent relational features
-   road/map topology

不编码 traffic light。

## Query机制

每个 agent 生成独立 query：

$q_i=W_qh_i$

通过 cross attention 查询场景：

$z_i=Attention(q_i,C,C)$

其中 $C$ 包含 agent context 和 map context。

## Attention作用

Cross Attention：

-   agent 查询地图和周围交通信息；
-   形成 agent-specific scene representation。

Self Attention：

-   建模车辆之间交互；
-   学习跟驰、避让、合流行为。

Temporal Attention：

-   建模轨迹时间演化；
-   保持速度和行为连续性。

------------------------------------------------------------------------

# 4. Persistent World Memory Module

继承 FIRM-WM 的思想，引入持续世界状态：

$z_W$

用于表示瞬时状态之外的隐藏交通状态。

更新：

$z_W^t=f(z_W^{t-1},Z_{scene})$

作用：

-   保留长期交通演化信息；
-   避免仅依赖当前frame。

------------------------------------------------------------------------

# 5. Behavior Prior Module

## 目的

解决单一轨迹预测导致的 mode averaging。

学习：

$p(z_B|scene)$

其中 $z_B$ 表示潜在行为模式。

## 训练方式

利用真实未来轨迹：

$Y^{gt}$

编码行为latent，并学习：

$p(trajectory|scene,z_B)$

支持：

-   keep lane
-   braking
-   merging
-   lane change

等多模态行为。

------------------------------------------------------------------------

# 6. Multimodal Future Planner

输入：

-   agent scene latent
-   world latent
-   behavior latent

输出未来轨迹窗口：

$Y=[S_{t+1},...,S_{t+H}]$

区别于传统单步 autoregressive prediction。

------------------------------------------------------------------------

# 7. Future Trajectory Buffer

借鉴 SceneDiffuser。

维护未来轨迹缓存：

$B_t=[S_{t+1},...,S_{t+H}]$

目的：

-   共享未来约束；
-   降低 rollout drift；
-   保持时间一致性。

每一步：

1.  执行 buffer 第一时间步；
2.  更新历史状态；
3.  refine 剩余未来轨迹；
4.  补充新的未来预测。

------------------------------------------------------------------------

# 8. Trajectory Refinement Module

将：

$S_t ightarrow S_{t+1}$

转变为：

未来轨迹持续优化。

初始预测：

$Y^0$

精炼：

$Y^{k+1}=R(Y^k,S)$

监督目标：

$Y^{k+1}ightarrow Y^{gt}$

------------------------------------------------------------------------

# 9. Normalizing Flow扩展接口

训练阶段：

    Real traffic data
            |
            v
           QR-WM

未来推理阶段：

    Normalizing Flow
            |
            v
    Initial Scene S0
            |
            v
    QR-WM rollout

Flow 输出必须转换为与训练数据一致的 scene tensor 格式。

------------------------------------------------------------------------

# 10. 总结

QR-WM 保持 RAMP/FIRM 的训练范式：

$History ightarrow Future$

但引入：

1.  Query-centric scene encoder；
2.  Persistent world memory；
3.  Behavior prior；
4.  Future trajectory buffer；
5.  Trajectory refinement。

目标是构建稳定、多模态、可持续精炼，并支持未来可控初始化扩展的交通世界模型。

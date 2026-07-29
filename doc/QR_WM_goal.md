# Query-Refine World Model（QR-WM）：基于 RAMP/FIRM 基线的多模态轨迹精炼式交通世界模型

训练环境：`conda activate tread`

## 1. 研究定位与代码边界

QR-WM 是基于 FITWMAMS 中 RAMP-WM 和 FIRM-WM 的全新独立实现方法。现有
RAMP/FIRM 作为 baseline，不直接修改其代码。

新方法需要独立建立模型目录、配置文件、训练脚本、checkpoint
和结果目录，并保持工程现有编码风格与 snake_case 命名规范。

当前阶段不引入 EVT 和风险条件生成，但模型结构需要预留未来扩展接口：

\[ p(Y\|S,c\_{risk}) \]

未来可用于风险感知长尾交通世界建模。

## 2. 设计目标

当前 autoregressive world model 存在长期 rollout drift：

\[ S\_{t+1}=f(S_t) \]

误差会随时间累积。

QR-WM 通过三个模块改进：

1.  Query-centric relational scene encoder；
2.  Behavior prior module；
3.  Trajectory refinement buffer。

## 3. Query-centric Relational Scene Encoder

Scene encoder 仅编码：

-   agent history states；
-   agent-agent relational features；
-   road/map topology。

不编码 traffic light。

每个 agent 生成独立 query：

\[ q_i=W_qh_i \]

通过 cross attention 查询：

\[ z_i=Attention(q_i,C,C) \]

其中 C 包含 agent context 和 map context。

Self attention 用于建模 agent-agent interaction，Temporal attention
用于保持轨迹时间一致性。

## 4. Behavior Prior Module

学习潜在行为模式：

\[ z_i \]

并建模：

\[ p(z_i\|scene) \]

生成：

\[ p( au_i\|scene,z_i) \]

保持交通行为多模态。

未来可以扩展为：

\[ p(z_i\|scene,c\_{risk}) \]

但当前不实现 EVT。

## 5. Trajectory Refinement Mechanism

由单步 autoregressive transition：

\[ S_tightarrow S\_{t+1} \]

转变为未来轨迹缓存：

\[ Y_t=\[S\_{t+1},...,S\_{t+H}\] \]

维护 future trajectory buffer，并进行持续 refinement：

\[ Y_t^{k+1}=Refiner(Y_t^k,S_t) \]

每个 physical timestep：

1.  执行 buffer 第一时间步；
2.  更新历史状态；
3.  修正剩余未来轨迹；
4.  补充新的未来预测。

## 6. 代码结构规划

    FITWMAMS
     |
     +-- baselines
     |      +-- ramp
     |      +-- firm
     |
     +-- qr_wm
            +-- scene_encoder
            +-- behavior_prior
            +-- trajectory_refiner
            +-- world_memory

禁止覆盖 baseline 实现。

## 7. 实验目标

验证：

-   Query-centric encoder 是否增强场景理解；
-   Behavior prior 是否提升多模态行为生成；
-   Trajectory refinement 是否降低长期 rollout drift。

指标：

-   ADE/FDE/minADE/minFDE；
-   trajectory diversity；
-   collision rate；
-   TTC/DRAC；
-   velocity/acceleration/jerk distribution；
-   long horizon consistency。

最终目标：构建稳定、多模态、可持续精炼的交通世界模型，为后续风险感知长尾测试扩展提供基础。

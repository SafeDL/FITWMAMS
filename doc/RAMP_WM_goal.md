# FIRM-WM：RAMP-WM 内部的关系状态转移创新

训练环境：`conda activate tread`

## 1. 研究边界

FIRM-WM（Feedback-Integrated Relational-Memory World Model）是在 RAMP-WM 闭环交通模型内部
改进状态转移和背景控制分布的候选方法。EVT 只用于离线筛选自然驾驶长尾事件；冻结的外层 Flow 只在
reset 时采样场景条件 \((C_0,B_0)\)。二者都不是 FIRM-WM 的方法贡献，也不能掩盖世界模型的不足。

FIRM-WM 从随机初始化训练；不读取未来 ego/background，不使用学习式 highD 地图编码器，也不伪造
历史。START 只使用当前 C0、B0 和 Flow 条件，随后只输入已发生的 ego 状态与模型已生成的背景状态。

## 2. 当前正式模型

FIRM-WM 保留 RAMP 的连续交通记忆、25 Hz jerk 动力学、0.2 s 滚动执行和随机数回放，并以：

- 当前状态的动态图关系编码替代地图输入；
- 逐车、逐帧关系计划场生成 1 s 联合 jerk 中心；
- 持续世界潜变量 \(z_W\) 与实际执行的 0.2 s 联合 action-flow 残差表示随机性；
- 完整 1 s 计划的监督约束计划重叠部分。

一次测试 episode 的随机变量为 \(\Xi=(z_F,z_W,\epsilon_{1:K})\)。固定 \(\Xi\) 时，不同 ADS
策略共享相同世界随机性；背景只能因已发生的 ego 状态而改变。改变 \(z_W\) 或 \(\epsilon\) 才代表同一
条件下的另一条世界轨迹。

## 3. 正式评价协议

主指标是 full held-out highD 的背景车 1--5 s ADE/FDE、速度、加速度、gap、相对速度及 TTC/DRAC。
外部写回的 ego 不计入重建误差。RAMP-WM 与 Semi-Markov 使用重新匹配的背景车评测；CAT-TopK 的
START 使用未来动作摘要，只能作为信息不对称参考。

Flow × FIRM 固定为 held-out EVT 条件上的 8 个外层 Flow 起点 × 每个起点 4 个内层世界，报告 B0
执行误差、风险分布、随机 jerk、无效率和重叠率。进入 ADS 测试的前提是无效轨迹率和重叠率均低于 1%，
且 q90 风险超越率处于真实 highD bootstrap 区间；不得以拒绝、重采样或后处理过滤达成。

## 4. 当前正式证据

唯一正式结果位于：

    results/highd_world_model/firm_world_model/
    results/highd_world_model/paper_experiments/firm_world_model/

| 证据 | 结果 | 结论 |
|---|---:|---|
| held-out 5 s FDE，FIRM / RAMP / Semi-Markov | 0.695 / 0.679 / 0.833 m | 优于 Semi-Markov，未超过 RAMP |
| EVT-tail 5 s FDE，FIRM / RAMP / Semi-Markov | 0.721 / 0.750 / 0.898 m | FIRM 终点误差较好，但 tail ADE 不优于 RAMP |
| 固定条件 invalid / overlap | 0.0165% / 0.00127% | 固定条件物理门槛通过 |
| Flow × FIRM q90 exceedance | 7.83%，真实 10.12%，bootstrap [6.75%, 13.50%] | 风险尾部位置通过 |
| Flow × FIRM invalid / overlap | 33.57% / 3.04% | 严重失败，不得进入 ADS 测试 |
| 随机 longitudinal jerk q90 / q99 | 1.45 / 4.26；highD 为 0.25 / 0.75 m/s³ | 概率控制尾部失真 |

因此，当前可主张 FIRM-WM 是可回放、会响应已发生 ego 行为的闭环背景模型，并在该协议下优于
Semi-Markov。不能主张它已超过 RAMP，或已能与 Flow 一起构成物理有效的长尾测试环境。

## 5. 结果工件与实现纪律

论文图只读取正式评测文件，位于
`results/highd_world_model/paper_experiments/firm_world_model/`；其 README 与 manifest 记录输入、哈希、
固定扫描顺序和 skipped 工件。图表不得重新训练、采样、拟合 EVT 或隐藏无效轨迹。

- 只保留 `firm_world_model` 的 checkpoint、训练记录、评测、Flow 组合和论文图；候选、smoke 与中间结果不保留。
- 配置、脚本与结果目录统一使用完整的 `snake_case` 名称。
- 不保留废弃 wrapper、旧模型分支或未使用字段；共享组件放在 `world_model/src/core/`。
- 论文文字不得将 CAT-TopK 写成同信息比较，也不得声称 FIRM 已满足 ADS 环境要求。

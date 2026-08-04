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

标准测试报告各模型自身的完整 held-out highD 条件重建。RAMP、FIRM、Semi-Markov 与 QR 在每个响应段只回放已实现的
logged ego，不读取未来背景；完整报告由五个 `test_*_world_model.py` 入口生成，再由
`evaluate_test_conditional_reconstruction.py --mode native` 以源报告和 checkpoint SHA-256 汇总。CAT-TopK 使用归档未来动作摘要，
因此只保留为信息条件不对称的参考。

全测试集汇总保留五秒 ADE/FDE、gap、TTC 与 DRAC 等模型原生指标。单模型的 32 分支脚本只用于深入诊断，
不作为重复子集模拟的全量标准评测。highD 直路数据没有可靠让行/交叉口语义，因此社会行为结论仅限可测的跟驰、制动响应和安全间距。

Flow × QR 组合另存于 `results/highd_world_model/long_tail_reproduction/`：它从全部 EVT-tail 样本的事件结构中采样
Flow START，再在匹配、平移后的 ego replay 下生成背景交通分布。这是生成分布审核，不是 paired reconstruction，
不能报告或推断 donor ADE/FDE。

## 4. 正式结果工件与实现纪律

每个模型保留 checkpoint、训练记录和标准测试；完整测试集汇总位于：

    results/highd_world_model/test_conditional_reconstruction/

根目录的 `study_manifest.json` 固定五份源报告及 checkpoint 哈希，
`overview/test_conditional_reconstruction_summary.json` 给出可追溯的五秒指标索引。Flow × QR 的全 EVT-tail 产物只保留在
`results/highd_world_model/long_tail_reproduction/`。

- 配置、脚本与结果目录统一使用完整的 `snake_case` 名称。
- 不保留废弃 wrapper、旧模型分支或未使用字段；共享组件放在 `world_model/src/core/`。
- CAT-TopK 必须标注 START 使用归档未来动作摘要；不能作为同信息提升结论。
- FIRM 是否能超过 RAMP、是否能构成物理有效长尾环境，只能依据 hash 固定的完整测试与 Flow 结果判断，不延用已删除图表的结论。

## 5. 当前已验证结论（2026-08-04）

五个 checkpoint 的完整 held-out 测试均为 24,216 条序列。信息对称的五秒条件重建中，RAMP 的
ADE/FDE 为 **0.1282/0.5066 m**，QR 为 **0.1473/0.5960 m**，FIRM 为 **0.1924/0.6950 m**，
Semi-Markov 为 **0.2064/0.8329 m**。所以按当前轨迹重建指标，RAMP 优于 FIRM；不能沿用“FIRM 最优”的旧结论。

QR 的交互误差最好：DRAC **0.8929 m/s²**、TTC **0.02884 s**、gap **0.1198 m**，低于 RAMP 的
1.1698 m/s²、0.05511 s、0.1362 m。CAT-TopK 的 FDE 为 0.0734 m，但因其未来动作摘要信息不能进入上述同信息排序。

全部 2,209 条 EVT-tail 中有 1,761 条具有冻结 Flow 的事件结构支持；8 个 Flow START 和 4 条 QR 世界未来
共产生 70,400 条五秒合成未来。每条未来的 START 行为 latent 由独立、可审计的 `world_seed` 控制。Flow × QR 的交通特征 Fréchet 距离为 **4.0773**、RBF-MMD 为 **0.01567**，
碰撞 episode rate 为 **11.19%**。这仍说明组合的长尾安全与交互分布需要进一步校准，不能以单条 replay 重建误差代替该分布级审核。

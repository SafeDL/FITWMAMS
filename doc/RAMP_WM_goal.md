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

标准测试只报告各模型自身的 full held-out highD 重建。正式长尾评测由
`evaluate_long_tail_reproduction.py` 在同一 EVT-tail 条件下同时运行 RAMP、FIRM、Semi-Markov 和
CAT-TopK：每个条件有一条确定性路径与 32 条随机闭环未来。外部 ego 不计入背景车 ADE/FDE。

评测分为四层：轨迹 ADE/FDE/minADE/minFDE；加速度、jerk、曲率和速度 KL；全车辆对碰撞、
TTC/DRAC/gap/相对速度和制动响应；以及 Wasserstein、KS、RBF-MMD 与交通行为特征 Fréchet 距离。
highD 直路数据没有可靠让行/交叉口语义，因此社会行为结论仅限可测的跟驰、制动响应和安全间距。

RAMP/FIRM 的 Flow 组合通过各自 `test_* --flow-composition` 执行固定 8×4 测试。这是生成分布审核，
不是 paired reconstruction，不能报告或推断 donor ADE/FDE。

## 4. 正式结果工件与实现纪律

每个模型只保留 checkpoint、训练记录、标准测试和一个最终 Flow 组合 JSON。长尾正式结果位于：

    results/highd_world_model/long_tail_reproduction/

其中每个模型都有独立子目录，包含 `metrics.json`、六张静态图和三段固定真实事件 GIF；根目录保留协议、
checkpoint 哈希、固定场景和四模型总览。旧论文图、独立 compare 脚本和中间评测不保留。

- 配置、脚本与结果目录统一使用完整的 `snake_case` 名称。
- 不保留废弃 wrapper、旧模型分支或未使用字段；共享组件放在 `world_model/src/core/`。
- CAT-TopK 必须标注 START 使用归档未来动作摘要；不能作为同信息提升结论。
- FIRM 是否能超过 RAMP、是否能构成物理有效长尾环境，只能依据新的正式长尾与 Flow 结果判断，不延用已删除图表的结论。

## 5. 当前正式运行结论（2026-07-29）

已使用四个最终 checkpoint 完成标准 held-out 测试和固定种子 `20260729` 的 EVT-tail 长尾重建：328 个条件，
每个条件一条确定性未来及 32 条随机未来。正式文件为
`results/highd_world_model/long_tail_reproduction/<model>/metrics.json`，其中 checkpoint SHA-256、事件清单和
协议由根目录 `study_manifest.json` 固定。

在同一条件下，FIRM-WM 的确定性 FDE 为 **0.710 m**，`minFDE@32` 为 **0.473 m**，优于 RAMP-WM 的
0.752 m / 0.746 m 和 Semi-Markov 的 0.928 m / 0.881 m；因此 FIRM 是当前 RAMP 内部状态转移改进中最好的
轨迹重建版本。Semi-Markov 的交通行为特征 Fréchet 距离为 **0.720**，低于 FIRM 的 4.159 与 RAMP 的 15.145，
但这不能抵消其较差的逐轨迹重建。CAT-TopK 的结果保留为参考，不参与同信息优劣判断，因为其 START 使用归档
未来动作摘要。

目前不能把任一模型表述为已完成的长尾闭环测试环境：在独立的 Flow 8×4 组合审核中，RAMP/FIRM 的碰撞
episode rate 分别为 **0.348**/**0.374**。这说明初始 Flow 条件与世界模型状态转移之间仍缺少足够的安全和分布
校准；后续改进应直接降低该分布级碰撞率并改善 DRAC 高分位，而不是增加历史/地图初始化模块或使用 donor
轨迹重建指标掩盖该问题。

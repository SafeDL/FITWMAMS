# 交互行为世界模型

本目录维护项目自研的自主 NPC 行为候选及其统一评测。它不是单独的“交互指标”，也不是
论文方法复现目录。模型只读取 1 s 历史和当前闭环状态，以 5 Hz 生成 NPC 动作，并在
25 Hz 交通动力学中执行；ADS 改变行为后，NPC 会基于新的状态重新决策。

当前冻结主候选为 B2-RL（`rolling_action_b2rl_s2`），横向响应扩展为 B2-RLP-e3
（`rolling_action_b2rlp_e3`）。二者均已完成三训练种子的 validation 和 test，但只能视为
交互研究候选，不能替代当前高精度事实模型：

| 候选 | 自然测试 ADE/FDE | 纵向响应概率 | 横向响应概率 | 主结论 |
|---|---:|---:|---:|---|
| B2-RL | 0.625/2.158 m | 0.348 | 0.134 | 有闭环纵向响应，事实精度不足 |
| B2-RLP-e3 | 0.624/2.155 m | 0.352 | 0.161 | 横向响应增加，但主动横向避让率仅 0.00091 |

自然全场景 cruise-PNC 下，两者的 NPC 碰撞率约 0.87%，越界率约 8.8%。这些结果明显
优于把条件尾部压力测试比例误当成自然碰撞率，但仍不足以作为 ADS 正式交互环境。

统一协议包含四类证据：自然轨迹分布重建、真实事件重建、配对纵向/横向干预，以及固定
ADS 控制器闭环。完整指标位于
`results/interactive_behavior_world_model/benchmark_v1/aggregate/test_seed_summary.csv`。
配置中的 `npc_interaction_*` 字符串是已生成工件的历史 schema/version 标识，为保证结果
可追溯而保留；新 Python 包、命令和输出路径统一使用 `interactive_behavior_world_model`。

项目级模型取舍与下一步统一架构见
[`doc/ADS_Behavior_World_Model_Assessment.md`](../doc/ADS_Behavior_World_Model_Assessment.md)。


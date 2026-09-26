# NPC 行为候选与交互评测基准

本目录维护项目自研的自主 NPC 行为候选及其统一评测协议，不是项目当前主世界模型的
实现目录，也不是论文方法复现目录。这里的候选只读取 1 s 历史和当前闭环状态，以
5 Hz 生成 NPC 动作，并在 25 Hz 交通动力学中执行；ADS 改变行为后，NPC 会基于
新的状态重新决策。当前以高精度名义轨迹为锚的场景条件化 MA-IDM 方案位于
[`hierarchical_world_model/`](../hierarchical_world_model/)。

本基准保留旧版 prefix-only 研究候选的训练和评测代码，但这些候选未进入当前
Flow→Diffusion→HiQR→在线 MA-IDM 主方案，也不是 ADS 测试环境运行依赖。
对应聚合实验、逐种子输出与检查点已从活动结果目录移除；它们的自主预测协议
与当前条件事实重建协议不同，不应直接排名。方法源码和配置目前仍可用于重训。

本目录是冻结的历史评测工具，不属于当前技术方案，也不参与
`hierarchical_world_model/` 的在线 ADS/NPC 主运行链。它保留旧候选的统一 T1/T2/T3
评测协议，并提供 TrafficBots posterior-resimulation 的可选诊断入口。旧候选的输出树已清理；
TrafficBots 诊断位于
`results/baselines/trafficbots_highd/`。需要重跑旧评测时，可用原始 highD 数据重新生成缓存。
统一协议包含四类
证据：自然轨迹分布重建、真实事件重建、配对
纵向/横向干预，以及固定 ADS 控制器闭环。历史指标位于上述聚合表。
配置中的 `npc_interaction_*` 字符串是已生成工件的历史 schema/version 标识，为保证结果
可追溯而保留；Python 包、命令和新输出路径统一使用 `npc_behavior_benchmark`。
显式重跑基准时，配置中的输出路径会重新生成临时评测目录。

已退役的 B3 位置块扩散＋固定跟踪器也没有进入主方案；其专用训练、评测、跟踪器
代码及权重此前已清理。

因果缓存生成器仍可为本基准及 TrafficBots 的可选严格因果训练/评测重建数据；未使用的
`lane_graph_edges` 缓存字段已移除。活动结果树不再保存生成后的大体积缓存数组。
需要时运行 `conda run -n tread python -m npc_behavior_benchmark.scripts.prepare_cache` 重建。

目录分工：`data/` 维护样本、事件与干预缓存；`policies/` 放 NPC 候选策略；
`evaluation/` 放执行器、指标与统计；`configs/` 放冻结评测协议；`scripts/` 包含
`prepare*` 数据准备、`train*` 训练、`evaluate*` 评测、`aggregate.py` 汇总和
`benchmark*` 延迟测试。`tests/` 验证这些公共组件和协议。

项目级模型取舍与下一步统一架构见
[`doc/ADS_Behavior_World_Model_Assessment.md`](../doc/ADS_Behavior_World_Model_Assessment.md)。

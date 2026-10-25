# 风险分层测试与能力保持修复

当前正式结果为 **Diffusion Planner 三方法、两轮搜索修复及三组最终留出评价**，已于 2026-10-25 完成。CaRL 两轮先导没有接纳增强，保留官方参数。研究设置和实现见 [技术方案](../../doc/risk_stratified_repair.md)及 [ADS 入口](../../ads/README.md)。

| 阅读入口 | 内容 |
| --- | --- |
| [正式分析](reports/diffusion_planner/analysis.md) | 研究问题、等成本发现、学习材料、最终改善与能力保持 |
| [全部最终结果](reports/diffusion_planner/final_outcomes.csv) | 三组集合、初始与三方法最终模型的碰撞、驶出道路、停车与进度 |
| [配对比较](reports/diffusion_planner/repair_comparisons.csv) | 相对初始及方法之间的碰撞率差和 95% 区间 |
| [首轮发现曲线](reports/diffusion_planner/first_round_discovery/discovery.pdf) | 相同初始策略、实际执行成本下的事故发现 |
| [完整执行成本](reports/diffusion_planner/complete_execution_costs.csv) | 搜索、专家、参考、开发、标签诊断及最终评价分项 |
| [完成核查](reports/diffusion_planner/checks/completed_delivery_acceptance.json) | 24,000 次最终日志、策略和实例配对、原生交付、源码及历史证据哈希 |
| [CaRL 先导](reports/carl/pilot_summary.json) | 未获得接纳增强的结果，区别于 DP 正式实验 |
| [nuPlan 手动交接](reports/manual_nuplan/README.md) | 已交付原始／修复权重、共同场景、现有 tread 下的运行命令 |
| [报告索引](reports/index.json) | 每份快照的原件路径、大小及 SHA256 |

本文分支在未见长尾实例上的碰撞由 61 降为 40／2,000；经验自然和自然生成集合分别为 76→80、41→37，详见正式分析。最终权重在创建测试实例前冻结，不依据这些结果重选模型。主要结论是长尾安全改善与已观测到的自然交通保持范围，不宣称所有场景均无回退或多失效域充分覆盖。

## 存储与版本管理

本根目录是工程内的真实目录，`README.md` 和精选 `reports/` 纳入 Git；下列四个子目录链接到 `/media/hp/DATA/nuplan_workspace/FITWMAMS_results/risk_stratified_repair/`，保留完整运行、权重、实例和日志：

| 子目录 | 用途 |
| --- | --- |
| `formal/diffusion_planner/` | 冻结协议、源码包、两轮任务、最终实例、配对日志、原生权重和完整报告 |
| `development/` | 独立先导、工程联调及其结果；不作为正式实验额外预训练 |
| `materials/` | 保持与开发实例、学习缓存和用途账本 |
| `preparation/` | 数据准备、实现集成、清理与布局记录 |

完整模型和原始数据没有随目录整理搬动或改写。原生交接另存 `results/risk_stratified_repair_manual_nuplan/`，城市指标由用户手动生成，独立于 highD 统计。历史实验与仍被引用的初始权重见 [risk_stratified_repair_history](../risk_stratified_repair_history/README.md)。

DP 正式实验保存实际使用的 424 个 Python 文件源码包；结束后合入已验收的 CaRL 学习器，当前 434 个 Python 文件与其审核清单一致。相关 101 项回归检查及 Ruff 通过，源码集成未增加训练或场景执行。完整复现应使用相应实验自己的 `source_archive.json` 和配置。

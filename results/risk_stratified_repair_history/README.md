# 风险分层测试与策略修复：历史实验

本目录保存新版能力保持修复之前的实验、有效初始权重与诊断。原 `results/method_c/` 已按 [index.json](index.json) 的路径映射迁移，历史实验保持暂停。新版方案见[技术方案](../../doc/risk_stratified_repair.md)，新版结果位于 `results/risk_stratified_repair/`。

| 路径 | 内容 |
| --- | --- |
| `formal/` | 较早的四策略、三种子实验；当前 DQN、PPO、PlanT 仍从这里读取初始权重 |
| `branching/` | DATA 盘上的八策略、单种子历史实验；三组最终评价未完成 |
| `runtime_reference/`、`runtime_check.json` | 当前执行器仍使用的回放参考与检查结果 |
| `maintenance/validation_reports.tar.gz` | 58 份已结束的接口验收、诊断和图表，包内保留原 `validation/` 路径及逐文件哈希 |
| `maintenance/branching_interfaces/` | 保持可直接核对的分支搜索接口验收 |
| `maintenance/processes/` | 历史 formal 启动及进程记录 |
| `maintenance/historical_run_logs.tar.gz` | 历史 formal 的 181 份运行日志，包内保留原路径及逐文件哈希 |
| `maintenance/development_history.tar.gz` | 已结束的环境安装记录、维护脚本、源码快照与加载诊断，共 52 个原始文件；内含逐文件哈希清单 `ARCHIVE_INDEX.json` |
| `maintenance/retired_plant_prefetch.tar.gz` | 已退出当前流程的 PlanT 专用预取工具源码 |
| `maintenance/validation_tools.tar.gz` | 11 个退役验收脚本及旧源码副本；逐文件哈希见包内 `ARCHIVE_INDEX.json` |
| `manual_nuplan/` | 单独保存的历史 nuPlan 人工评价 |
| [reports/](reports/) | 纳入 Git 的 17 份关键协议、分析、计费和回归诊断快照，以及[历史配置](reports/original_config.yaml) |

旧维护文件、退役源码、验收报告和运行日志分别归档，逐文件核对 SHA256 后移除散文件。其中 181 份旧日志压缩为约 52 KB，58 份验收报告保留原目录结构。17 份精选报告及其原件、3 份仍引用的初始权重、实际轨迹、专家缓存和运行参考保持不变；当前代码不依赖已归档的散文件。

在本目录恢复验收报告执行 `tar -xzf maintenance/validation_reports.tar.gz validation/`，恢复历史运行日志执行 `tar -xzf maintenance/historical_run_logs.tar.gz formal/`。只读查看清单可用 `tar -xOf <归档路径> ARCHIVE_INDEX.json`。

`branching`、`manual_nuplan` 是有效的 DATA 链接，物理根目录为 `/media/hp/DATA/nuplan_workspace/FITWMAMS_results/risk_stratified_repair_history/`。索引保存迁移时的清单，后续删除空目录、归档工具等变化记录在 `cleanup`，不改写原迁移证据。

`formal/` 中的模型权重和配套学习状态已逐文件迁至 `/media/hp/DATA/nuplan_workspace/FITWMAMS_models/results/risk_stratified_repair_history/formal/`，原文件路径保留有效符号链接。所有内容经 SHA256 核对，初始策略、历史修复模型和报告身份保持不变；实例与报告仍在原目录。路径对应关系及验证见[模型存储记录](../model_storage_20261025.json)。

Git 跟踪本说明、索引与精选报告；完整运行目录、归档、权重及 DATA 链接留在本地。原始报告中的旧路径、协议标识与统计保持原样，按索引定位。档案诊断不作为最终留出评价；暂停的旧进程不直接恢复为新版实验。

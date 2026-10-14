# 世界模型结果

保存当前世界模型权重、NPC 参数及两类独立评测：完整 Test 的事实执行，以及 NPC 自主交通的分布一致性。

| 目录 | 内容 |
| --- | --- |
| `hiqr_training/` | HiQR 训练记录与权重 |
| `model_assets/` | 正式 IDM/MOBIL、换道模型、可选门控权重与发布清单 |
| `factual_test/` | 完整 Test 的逐场景指标、误差样本、执行记录与精度图 |
| `diffusion_plans/` | 当前 Diffusion 的 Val／Test 软计划缓存 |
| `npc_response/policy_calibration/` | 选定 NPC 策略、来源及标定记录 |
| `npc_response/conditional_behavior/` | 给定交通条件下的行为统计 |
| `npc_response/autonomous_traffic/` | NPC 自主道路运行、事件及可复算分布统计 |
| `npc_response/figures/` | 当前模型的六张论文 PNG |
| `npc_response/tables/` | 条件行为与闭环执行两张 CSV |
| `npc_response/animations/` | 四段自主交通 GIF 与来源清单，25 fps 原速 |

## 查看结果

- [事实报告](factual_test/report.json)：10,151 个 Test 场景，真实结点经当前 Diffusion 形成软计划，再由世界模型执行。
- [NPC 报告](npc_response/report.json)：Train 拟合、Val 选择的当前策略，覆盖全部 60 段录像；120 个道路方向 × 16 次重复 × 两个时域，共 3,840 次自主运行。
- [论文图](npc_response/figures/)与[原速动画](npc_response/animations/)：动画内所有车辆均由 NPC 策略驾驶，没有轨迹回放自车或 ADS。

`short_horizon/` 为 5.96 秒，`continuous_runs/` 为 120 秒。各自的 `statistics.npz`、`runs.json`、`events.csv` 和 `protocol.json` 保存可复算统计及执行记录，不再保留逐路段分片。

事实评测的 `scene_metrics.csv`、`error_samples.npz` 保存逐场景指标与误差样本，`execution.json` 保存分批执行记录。完整指标与核验状态以报告为准，不在 README 重复抄写。

训练与标定输入见[共享数据](../highd_shared_training_data/README.md)，入口见[代码说明](../../hierarchical_world_model/README.md)。历史试验和清理备份位于 [archive/hierarchical_world_model/](../../archive/hierarchical_world_model/)。

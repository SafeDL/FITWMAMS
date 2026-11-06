# CaRL 先导与共享材料

[CaRL 先导报告](reports/carl/pilot_summary.json)保留未接纳增强的结论；当前 CaRL 使用官方 `nuplan_51892_1B/model_best.pth`。

`materials/carl/` 和 `materials/diffusion_planner/` 保存通用材料准备接口仍使用的实例、轨迹和索引；`preparation/carl_repair_review/` 只保留先导结论、原生输入和参数一致性的验收记录。旧启动脚本、补丁、源码副本、失败试跑输入和测试日志已删除。这些目录通过链接访问 DATA，不作为当前 DP 搜索实验的成绩。

当前 DP 交付及完整结果见[安全增强版](../../README.md)，nuPlan 命令见[当前人工评价入口](../../runs/safety_enhanced/manual_nuplan/README.md)。旧两轮 DP 模型和测试记录已删除。

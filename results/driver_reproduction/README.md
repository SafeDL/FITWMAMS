# 驾驶人模型对照

保存 B-IDM、MA-IDM、Dynamic-AR 和 Multi-regime 等跟驰模型的独立复现证据。

- `matched_metrics.json`：218 个跟驰事件中，182 个训练、36 个测试；使用 5 秒历史、3 秒预测和 64 次随机采样。
- `driver_model_scorecard.json`：各论文复现状态与比较口径。

这两份小型报告仍由外部模型证据汇总和测试读取，与当前 NPC 自主交通评测分别维护。生成入口见[评测代码](../../external_model_baselines/evaluation/driver_reproduction/README.md)。

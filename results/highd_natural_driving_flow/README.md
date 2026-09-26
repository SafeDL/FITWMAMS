# highD 场景条件 Flow

`dataset.npz` 与 `dataset_schema.json` 定义当前场景条件数据，`manifest.json`
记录模型结构和协议；`evaluation_summary.json` 是正式测试结果。
训练摘要和历史保存在本目录。`checkpoints/best_scenario_condition_flow.pt` 是推理权重；
`figures/` 保存分布及训练诊断图；`samples/generated_samples.npz` 是验证脚本读取的生成样本。
验证入口见
[`normalizing_flow/README.md`](../../normalizing_flow/README.md)。

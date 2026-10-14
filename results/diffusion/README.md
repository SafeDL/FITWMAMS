# 背景车轨迹生成结果

保存 PCHIP 参考轨迹加条件残差扩散模型的权重与评测。输入为 68 维 `C0/M/K`，输出背景车位置软计划。

完整 Test 含 10,151 个场景，其中 9,841 个有有效背景车目标，每个条件生成 16 次。完整指标见 [evaluation_summary.json](evaluation_summary.json)。

评测给定真实位置结点，衡量结点之间的轨迹重建；固定结点的端点误差不代表未来预测精度。使用当前 Diffusion 的完整 Test 动作执行评测已完成，单独保存在[世界模型结果](../hierarchical_world_model/README.md)。

文件用途：

- `dataset_contract.json`：数据格式与仅在 Train 拟合的标准化参数。
- `training.json`：训练摘要和逐轮记录。
- `evaluation_summary.json` / `evaluation_per_sequence.npz`：汇总指标、逐场景误差和绘图数据。
- `checkpoints/`：唯一最佳 EMA 权重。
- `cohorts/`：切入场景索引及来源说明。
- [figures/](figures/)：训练曲线和评测诊断两张 PNG。

运行方法见[代码说明](../../diffusion/README.md)。

# 分层世界模型结果

当前仅保留两个运行资产/结果目录：

- `factual_hiqr/`：`checkpoints/final_world_model.pt` 是在线闭环使用的冻结 HiQR 权重；同目录的 manifest 记录来源，`natural_response_calibration.json` 保存响应校准参考。
- [`evaluation/`](evaluation/README.md)：唯一维护的评测包，包含全 10,151 条 Test 的事实重建、
  ADS/NPC 响应评估、必要的因果归因审计、GIF，以及可重放的 Diffusion 计划缓存。

旧截速事实报告与已否决 CIH 分支不再放在此目录。指标定义、限制和方法解释集中记录于
[`doc/Traffic_World_Model_Convergence.md`](../../doc/Traffic_World_Model_Convergence.md)。

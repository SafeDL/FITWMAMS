# 全 highD EVT-tail 的 Flow × QR-WM 端到端结果

本评测覆盖缓存中的全部 2,209 条 EVT-tail 序列；其中 1,761 条具有冻结 Flow 训练 split 支持的 slot mask 与主风险槽位。对每条受支持 replay 抽取 8 个 Flow 初始条件，并从每个条件生成 4 条世界未来。

## 规模与因果协议

- 17,600 个 Flow 初始条件，70,400 条 5 秒合成未来。
- Flow 起点与 replay 在 slot mask、主风险槽位上严格一致；所有样本属于 `highd_straight_lane` cohort。
- replay 按初始 ego 纵向速度最近邻匹配；绝对速度误差中位数为 0.0114 m/s，95% 分位为 0.5646 m/s。
- QR 每 0.2 秒只接收已实现的、平移后的 ego 状态。每条未来有唯一 `world_seed`，只控制 START 行为 latent；给定该 latent 后的响应演化是确定的。独立世界仅在网络前向时批处理，不共享状态、latent、记忆或计划缓存。
- 使用 64 个 Flow 起点 × 4 个独立世界的批次时，Flow 采样与 replay 匹配耗时 41.48 s，QR 演化耗时 226.36 s（311 条 5 秒世界未来/s）。

## 结果

- 交通交互特征 Fréchet 距离：4.0773；RBF-MMD：0.01567。
- 合成样本 collision episode rate：11.19%；collision pair-point rate：0.2905%。

该目录测量的是 Flow + QR 的端到端尾部交通生成质量，不是逐样本轨迹重建。尽管 QR 在真实初始条件下的重建性能可单独评价，当前组合仍需进一步降低碰撞率并校准高风险交互分布。

完整数值见 `flow_composition_evaluation.json`；每条生成未来的 Flow 条件、密度、匹配记录与 `world_seed` 见 `flow_start_audit.npz`。

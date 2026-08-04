# 全 highD EVT-tail 的 Flow × QR-WM 端到端评测

本目录只保存完整 highD 长尾驾驶数据（2,209 条 EVT-tail 序列）的端到端 Flow + QR 评测。

冻结 normalizing flow 按 slot mask 和主风险槽位采样 `C0+B0`；在高D直道路型 cohort 内，按初始 ego 纵向速度匹配 EVT-tail ego replay，并将 replay 平移到 Flow 起点。每条 QR 世界未来都有独立、可复现的 `world_seed`，它只控制 START 行为 latent；QR-WM 仍然每 0.2 秒只接收已实现的 ego 状态。

独立世界分支在每个响应时刻批量推进；默认一次处理 96 个 Flow 起点，即 384 条 QR 独立世界。该批量已在当前 24GB GPU 上完成 5 秒演化验证；批处理只共享网络计算，不共享任何场景状态、latent、计划缓存或 ego 信息，因此不改变因果协议。

正式命令为：

```bash
python world_model/scripts/evaluate_qr_flow_tail_composition.py
```

该目录中的 `flow_composition_evaluation.json` 覆盖全部 EVT-tail 的正式结果，`flow_start_audit.npz` 记录每条未来的 Flow 条件、匹配 replay 与 `world_seed`；旧的 328 条 held-out EVT-tail 组合结果不会被混入此目录。

`figures/01_tail_interaction_distribution.png` 展示真实 EVT-tail 与 Flow × QR-WM 合成未来的交互特征上分位数、KS 与 Wasserstein-1 差异；`figures/02_flow_sampling_and_runtime.png` 展示 replay 速度匹配、主风险槽位、碰撞诊断、运行时间和 `world_seed` 审计。图表完全由正式 JSON 与 NPZ 审计文件生成；如需重绘：

```bash
python world_model/scripts/plot_reconstruction_result_summaries.py --only tail
```

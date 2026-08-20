# 设计审计

本目录只保留最终 118 维条件扩散模型所依赖的两项设计证据，不保存被正式全量结果替代的 pilot checkpoint 或测试汇总。

- `constraint_ceiling.json/png`：在完整 10,151 条 recording-level 测试序列上，比较稀疏位置/速度结点的插值重建上限。2 s、4 s、5.96 s 三个结点的 Hermite 参考 ADE 为 0.02605 m，证明该条件具备达到历史精度参照的表达能力。
- `motion_basis.json`：在固定 1,024 条设计审计序列上比较 Savitzky–Golay 窗口。最终使用三阶、41 帧窗口，在保留厘米级位置残差的同时消除预设加速度和 jerk 阈值越界。

正式训练、测试及限制以父目录的 `manifest.json` 和 `evaluation_summary.json` 为准。

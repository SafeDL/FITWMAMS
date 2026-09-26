# highD 自然驾驶数据与 EVT

`natural_segments.csv` 是清洗后的 96,055 条自然驾驶序列索引；
`natural_segments_summary.json` 记录筛选结果，`natural_risk_traces.npz` 保存逐窗风险轨迹。
`evt/` 保存尾部分布模型、标定摘要、阈值敏感性和诊断图；`playbacks/random_audit/`
保存六段随机筛选场景的语义审计 GIF，`playbacks/evt_tail/` 保存五段高风险真实轨迹 GIF。
动画均非生成式仿真结果。预处理协议见
[`process_highD/README.md`](../../process_highD/README.md)。

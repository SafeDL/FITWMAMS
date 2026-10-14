# highD 共享训练数据

本目录保存训练与标定输入，不保存模型权重或评估结果。

| 目录 | 内容 |
| --- | --- |
| `scenario_conditions/` | 车辆存在标记 `M`、26 维初始状态 `C0`、36 维位置结点 `K`、尺寸 `D` 及片段索引 |
| `highd_sequence_cache/sequence_cache/` | 对齐片段的 150 个真实状态点、149 个转移、地图、尺寸和动作 |
| `driver_behavior/` | 全部 60 段录像提取的驾驶状态、完整换道、道路初态与固定统计协议 |
| `response_gate_training/` | 可选门控的原始采样轨迹、监督样本和事件索引，不参与默认 NPC/HiQR 评测 |

场景条件和时序缓存均包含 96,055 个片段，按 `segment_id` 对齐。42 / 9 / 9 段录像对应 72,771 / 13,133 / 10,151 个 Train / Val / Test 片段，其中 EVT 尾部共 2,964 条。

有效车辆保留实际长宽；缺失槽位的尺寸在条件数据中为零，在时序缓存中为 4.8 m × 1.8 m 占位，均须结合车辆存在标记使用。

前两类来自 `results/evt/natural_segments.csv`，生成顺序见 [process_highD](../../process_highD/README.md)。`driver_behavior/` 直接从完整原始录像提取，由世界模型的 `calibrate_drivers.py` 准备，沿用相同录像划分，不只使用七车短片段。门控训练输入与权重分开保存，权重位于世界模型结果目录。

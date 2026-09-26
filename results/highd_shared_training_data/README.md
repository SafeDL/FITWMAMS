# Shared highD sequence data

`highd_sequence_cache/sequence_cache/` 中保存 96,055 条 5.96 秒规范自然驾驶序列的
manifest 与内存映射数组；`cohorts/cutin_crossing_4s/` 中的 `index.npz` 和
`manifest.json` 保存用于扩散评测的完整语义 cut-in 队列。
它是 HiQR 与扩散模型共享的数据表示，不属于某个模型，
也不包含模型权重或评估结果。

当前 manifest 直接绑定：

- `results/highd_natural_driving_evt/natural_segments.csv`；
- `results/highd_natural_driving_flow/dataset.npz`；
- `results/highd_natural_driving_flow/dataset_schema.json`。

split 与 Flow 完全一致：train/val/test 为 72,771 / 13,133 / 10,151；EVT 标签
为当前阈值 `event_risk > u` 的 2,964 条。数据在自然驾驶清洗阶段已经满足横向事件
完整性和 C0 有效背景槽位全窗稳定性，模型不得再实施第二次 stable-slot 筛选。

该 cohort 的 manifest 绑定上述规范序列。旧模型样本缓存和旧扩散反应参考缓存已经删除；其他专题 cohort 应从
当前规范序列重新生成。

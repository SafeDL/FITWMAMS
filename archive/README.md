# 已退出主线的代码与结果

这里保存从活动目录移出的旧实现和历史结果，不是另一套可直接运行的方法。

- `hierarchical_world_model/`：旧 NPC 基准、驾驶响应候选、重复审计、旧动画入口及相应测试。
- `process_highD/data/`：旧论文事件库、旧 Train-only EVT 拟合和旧 ADS 专用清单。
- `results/`：旧 HiQR 权重、旧学习型选择器诊断、旧版逐事件结果、`paper_benchmark`、已退出主链的直接位置 Diffusion 实验和历史开发检查。
- `code_before_cleanup/`：被精简文件的原始版本，便于核对或恢复嵌入其中的旧分支。
- `hierarchical_world_model/main_chain_cleanup/`：旧横向续接审计、无消费者 RL 控制器、失效行号探针及整理前元数据。

当前唯一入口与组件见 [世界模型 README](../hierarchical_world_model/README.md)。
当前结果位置见 [results README](../results/README.md)。
最近一次世界模型更新前的权重、参考和闭环记录见
[`upstream_refresh/`](results/hierarchical_world_model/upstream_refresh/README.md)。
原详细[清理清单](results/hierarchical_world_model/cleanup_manifest.json) 已移入本目录。

所有归档均保留文件内容，没有清空回收站或永久删除原始数据。
恢复时只移动所需条目，并先检查原位置是否已有新文件，不要覆盖当前主线。
归档代码仍可能引用旧模块或旧结果，不能加入默认导入路径，也不参与默认 pytest。
模型与数据目录改名只调整存储位置，旧报告仍代表原来的实验，不视为重新评测。

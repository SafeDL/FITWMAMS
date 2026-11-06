# 历史模型与结论

本目录保留历史模型的唯一参数、必要来源和结论汇总。已删除废弃实现、重复源码备份，以及被当前结果取代的调参验证、旧 Test 和条件诊断记录。

- `hierarchical_world_model/`：旧 NPC 参数、拟合来源和必要报告；字节相同的模型只保留一份。
- `results/`：历史 HiQR、直接位置 Diffusion 权重及评价摘要；旧 DQN 选策结果已随算法退役删除。
- `process_highD/data/`：旧 Train-only EVT 参数与来源汇总，不能替代当前全量尾部资产。
- `retained_models/`：从旧备份包提取的独有权重；[来源与哈希](retained_models/provenance.json)记录提取来源，以及重复文件的原路径、当前保留位置和 SHA256。
- [retirement_summary.json](retirement_summary.json)：已退役世界模型参数搜索的选择结论与原记录哈希；不作为当前方法的实验成绩。

旧运行的完整中间过程已清理，不能再将本目录视为可整体恢复的实验副本。当前运行入口见[世界模型](../hierarchical_world_model/README.md)，论文证据见[结果目录](../results/README.md)。

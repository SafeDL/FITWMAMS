# 共享交通基础组件

`traffic_components/` 是当前交通世界模型主链使用的底层 Python 包，不是一条独立的模型候选。
正式模型、训练和闭环验收位于 [`hierarchical_world_model/`](../hierarchical_world_model/)。

- `src/hiqr/`：当前 HiQR 的配置、关系编码器和观测滤波器，由分层世界模型直接导入。
- `src/core/`：highD 规范序列读写、统一动力学、评估口径及工具函数，由数据预处理、
  Diffusion、分层世界模型和对照评测共用。
- `src/traffic_graph/`：构建规范 highD 序列缓存时使用的车道与车辆关系图。

共享序列缓存位于
[`results/highd_shared_training_data/`](../results/highd_shared_training_data/README.md)。
源数据或预处理规则变化后，使用
`conda run -n tread python process_highD/scripts/prepare_highd_sequences.py --rebuild`
重建。

当前在线结果的运行指纹包含 `src/core/dynamics.py` 的文件路径和内容，见
[`hierarchical_world_model/src/provenance.py`](../hierarchical_world_model/src/provenance.py)。
本次包更名仅迁移代码命名空间和路径，原运行哈希记录在
[`package_rename_manifest.json`](../results/hierarchical_world_model/evaluation/package_rename_manifest.json)；
指标未重算，现有报告按新路径重新核验。
TrafficBots 自身实现位于
[`external_model_baselines/models/trafficbots/`](../external_model_baselines/models/trafficbots/)。

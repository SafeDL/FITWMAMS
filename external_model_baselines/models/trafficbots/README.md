# TrafficBots V1.5 的 highD 适配

保留 TrafficBots 的相对位姿交互编码、CVAE、目的地条件和背景车动力学，适配本工程的共享 highD 缓存。六个背景车槽位均参与推理与统计，包括同车道后车。

| 文件或目录 | 用途 |
| --- | --- |
| `data.py`、`module.py` | 共享缓存转换和训练模块。 |
| `rollout.py`、`evaluation.py` | 因果滚动、随机预测和配对干预评价。 |
| `upstream/` | 实际使用的上游网络与工具，来源见 [NOTICE.md](NOTICE.md)。 |
| `config/highd.yaml` | 数据、训练与评价参数。 |
| `scripts/` | 训练、评价和已完成结果核查入口。 |

使用现有 `tread` 环境，依赖版本见 `requirements-trafficbots-highd.txt`。

```bash
conda activate tread
python -m external_model_baselines.models.trafficbots.scripts.audit
python -m external_model_baselines.models.trafficbots.scripts.verify_full
```

训练与评价分别使用 `scripts.train`、`scripts.evaluate`，后者通过 `--checkpoint` 指定权重。主要生成评价只读取当前状态，以因果先验采样；真实未来仅用于离线评价。完整场景后验重模拟单独报告。

既有权重与报告位于 [results/baselines/trafficbots_highd](../../../results/baselines/trafficbots_highd/README.md)。旧评测的缓存顺序与当前共享数据不同，保留原实验身份，不直接将它当作当前场景顺序下的新评测。上游来源和改动范围见 NOTICE；本目录是 highD 适配版本。

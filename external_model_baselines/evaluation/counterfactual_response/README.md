# 随机驾驶人的配对响应检查

在相同领车制动条件下，比较 B-IDM、MA-IDM、Dynamic-AR(5) 和 Multi-regime B-IDM 的跟驰响应。每次参数与外生随机量固定，分别执行原动作计划和根据当前交通重新计算的动作，观察闭环响应带来的间距变化。

使用 25 Hz 物理执行、5 Hz 驾驶决策；领车在第 1—2 秒额外制动，幅度为 1.5、3.0、5.0 m/s²，每模型每幅度保存 256 对轨迹。场景在干预前选定，其真实输入保存在 `artifacts/response_ensembles.npz`。旧记录中的缓存行号不用于索引当前共享数据。

| 文件或目录 | 用途 |
| --- | --- |
| `experiment.py` | 配对滚动和响应统计。 |
| `visualize.py` | 响应图及 GIF。 |
| `evidence/reaction_events/` | 自然响应事件与来源。 |
| `artifacts/response_metrics.json` | 协议、场景身份、响应结果。 |
| `artifacts/response_ensembles.npz` | 原始输入及各分支轨迹。 |

```bash
conda activate tread
python -m external_model_baselines.evaluation.counterfactual_response.scripts.run
```

该检查验证模型内部的响应机制。干预后不存在对应的真人反事实轨迹，不能将响应幅度解释为真人行为精度；当前世界模型的 NPC 验证见 `results/hierarchical_world_model/`。共同评测口径见[驾驶人评测](../driver_reproduction/README.md)。

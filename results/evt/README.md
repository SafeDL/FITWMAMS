# 自然驾驶风险与 EVT

EVT 在全部 96,055 个自然驾驶片段上统一拟合和筛选，得到 2,964 条尾部事件，供 Flow 建模测试场景分布。

## 文件用途

- `natural_segments.csv`、`natural_risk_traces.npz`：96,055 个清洗片段及逐帧风险；
- [evt_model.json](evt_model.json)、[evt_summary.json](evt_summary.json)：全量拟合模型与摘要；
- `threshold_sensitivity.csv`、`figures/`：阈值扫描、六联总图和六张独立诊断图；
- `event_bank/`：统一事件表、逐录像选择阈值和摘要；
- `manifests/`：完整尾部清单，以及普通、经验极值、EVT 三组固定实验清单，不替代下游完整 Test；
- [playbacks/](playbacks/README.md)：25 fps 原速回放原始 highD 轨迹，不是生成式仿真。

后续 Flow 读取同一尾部标签，闭环风险评价读取同一 EVT 模型。阈值及拟合诊断以报告为准，生成顺序见 [process_highD](../../process_highD/README.md)。

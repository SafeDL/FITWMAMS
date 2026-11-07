# Active Inference 驾驶人模型

本目录保留两条明确分开的实现：`official_wrapper.py` 逐步调用外部锁定的官方 v1.0.0 模型；`model.py` 是基于论文机制、针对 highD 纵向跟驰任务编写的适配。官方受限源码不复制进本仓库。

适配保留信念粒子、CEM 动作规划、视觉逼近观测、意外证据累积和重规划机制。外部车辆以 25 Hz 推进，驾驶动作以 5 Hz 更新并保持五帧；真实领车未来不进入规划器。具体预算见 `config.py`。

| 位置 | 用途 |
| --- | --- |
| `official_wrapper.py` | 官方模型的逐步调用和 25 Hz 执行桥接。 |
| `model.py`、`evaluate.py` | highD 纵向适配及闭环评测。 |
| `scripts/` | 原生后追尾复现、官方调用一致性与桥接时序检查。 |
| `artifacts/native_reproduction/` | 官方原生复现、来源数据和参数／轨迹核查。 |
| `artifacts/full_highd/` | 纵向适配的完整 highD 结果与诊断图。 |

官方包装器的原生逐步一致性和五个制动条件的执行时序已有记录，见 `official_step_wrapper_audit.json`、`official_25hz_bridge.json`。highD 的 304 个领车制动事件由纵向适配完成，协议见 `artifacts/full_highd/highd_protocol_audit.json`；原生核查与 highD 适配成绩分别报告。

```bash
conda activate tread
python -m external_model_baselines.models.active_inference_driver.run --max-events 304 --output external_model_baselines/models/active_inference_driver/artifacts/full_highd
pytest -q external_model_baselines/models/active_inference_driver/tests
```

调用 `OfficialPOMDP25Hz` 时需提供锁定的外部源码及其 `Results_following` 数据路径。原始逐实验轨迹保留在磁盘，不纳入 Git；复现摘要、来源与验收报告可随代码核查。模型登记见[注册表](../../registry.json)。

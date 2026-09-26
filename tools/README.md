# 跨模块工具

这里只保留被当前主链实际复用的实现：

| 文件 | 职责 | 使用方 |
| --- | --- | --- |
| `evt.py` | POT/GPD 尾部模型和 EVT 标定 | highD 预处理、世界模型、IDM–AMS |
| `idm_ego.py`、`idm_ego.yaml` | 固定 ADS IDM 参数和辅助函数 | 世界模型、IDM–AMS |
| `plot_style.py` | 论文图与 GIF 的统一绘图样式 | Flow、Diffusion、世界模型等 |

旧的 longitudinal/cut-in 风险、暴露量、context NPZ、frozen diffusion adapter、
通用 I/O 和归一化模块没有现行调用方，已移除。当前自然驾驶 EVT 的风险定义位于
[`process_highD/src/safety_envelope_risk.py`](../process_highD/src/safety_envelope_risk.py)，
配置位于 [`process_highD/configs/highd_natural_evt.yaml`](../process_highD/configs/highd_natural_evt.yaml)。
不要在此添加仅转发 import 的兼容入口；只服务单个模块的实现应留在该模块内部。

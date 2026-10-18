# 共享工具

供数据处理、场景生成和 ADS 评测直接导入，本目录不单独训练或保存结果。

| 文件 | 用途 |
| --- | --- |
| [evt.py](evt.py) | POT–GPD 尾部拟合与风险映射 |
| [idm_ego.py](idm_ego.py)、[idm_ego.yaml](idm_ego.yaml) | 固定 IDM 自车的参数名和配置读取 |
| [plot_style.py](plot_style.py) | 图表与 GIF 共用的字体、配色和坐标样式 |

IDM 自车配置与 NPC 驾驶参数分别维护。自然驾驶风险定义在 [safety_envelope_risk.py](../process_highD/src/safety_envelope_risk.py)；只服务单个模块的实现放回所属目录。

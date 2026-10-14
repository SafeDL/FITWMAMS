# 共享工具

供多个模块直接导入，不单独训练或保存实验结果。

| 文件 | 用途 |
| --- | --- |
| [evt.py](evt.py) | POT–GPD 尾部拟合与风险映射，供 highD 处理、世界模型和 IDM–AMS 使用 |
| [idm_ego.py](idm_ego.py)、[idm_ego.yaml](idm_ego.yaml) | 被测 IDM 自车的参数名与配置读取；不同于 NPC 驾驶参数分布 |
| [plot_style.py](plot_style.py) | 论文图与动画共用的字体、配色和坐标样式 |

自然驾驶风险定义位于 [process_highD/src/safety_envelope_risk.py](../process_highD/src/safety_envelope_risk.py)。只服务单个模块的实现留在所属模块，不在此增加转发入口。

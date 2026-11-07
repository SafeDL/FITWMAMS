# highD 原始数据与工具

`Matlab/data/` 保存取得授权的原始录像表及配套文件，正式 Python 数据流水线从这里读取。原始数据不纳入 Git，也不在工程清理中删除。

| 目录 | 内容 |
| --- | --- |
| [Matlab/](Matlab/README.md) | highD 数据读取、交互可视化及已有 MATLAB 工具。 |
| [Python/](Python/README.md) | 原始轨迹读取与逐帧可视化工具。 |

论文使用的片段提取、安全评分、数据划分和 EVT 入口统一在 [process_highD](../process_highD/README.md)。本目录的工具用于检查原始数据，不替代正式预处理协议。

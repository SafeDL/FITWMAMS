# highD MATLAB 工具

- `data/`：原始录像表及配套文件，Python 正式流水线也从这里读取。
- `initialize.m`：添加工具路径。
- `utils/`：读取录像与车辆表、绘制道路和轨迹，以及已有片段筛选工具。
- `visualization/`：交互轨迹查看器。

在 MATLAB 中运行 `initialize.m` 后使用读取或可视化函数。论文的统一筛选、评分与数据划分由项目根目录 `process_highD/` 实现，不能用本目录的独立筛选脚本替换其协议。

# highD 预处理与风险事件

读取原始 highD 录像，提取最多七车的自然驾驶片段，计算安全包络侵入风险 `R_SEI`，再用 EVT 筛选长尾事件。Flow、Diffusion 和世界模型共用这里的片段及划分。配置统一在 [config.yaml](config.yaml)。

## 数据口径

- 原始数据放在 `highD_dataset/Matlab/data/`。统一车辆中心坐标和行驶方向，过滤异常帧，保留原生 25 Hz。
- 每片段包含 150 个状态点、149 次转移，跨度 5.96 秒。主车为小汽车，六个背景车槽位在起点确定，有效车辆覆盖完整窗口。
- 换道保留完整过程及至少 1 秒前、2 秒后轨迹；保留实际车身尺寸。
- Train / Val / Test 按录像划分。EVT 在全部自然片段上拟合，`R_SEI` 表示风险排序，不是碰撞概率。具体数量与拟合结果见输出报告。

## 主要入口

在项目根目录依次运行：

```bash
conda activate tread
python process_highD/scripts/extract_highd_natural_segments.py
python process_highD/scripts/fit_evt.py
python normalizing_flow/scripts/prepare.py
python process_highD/scripts/prepare_highd_sequences.py
python process_highD/scripts/build_event_bank.py
python process_highD/scripts/build_manifests.py
```

依次完成片段抽取、EVT 拟合、共享场景条件、时序缓存、事件库及实验清单。Flow 的 `prepare.py` 只准备条件数据；时序缓存依赖它。已有缓存直接复用，需要重建时为 `prepare_highd_sequences.py` 添加 `--rebuild`。

生成原始轨迹 GIF：`python process_highD/scripts/play_highd_natural_events.py`。这类回放展示测量轨迹，桥接仿真回放由世界模型入口生成。

## 输出

- [results/evt/](../results/evt/README.md)：片段与风险数据、EVT 模型、事件清单、诊断图及原始轨迹 GIF。
- [results/highd_shared_training_data/](../results/highd_shared_training_data/README.md)：场景条件、真实时序缓存及 NPC 训练数据。前两项按 `segment_id` 对齐。

`src/` 保存预处理、风险、EVT 和绘制实现，`scripts/` 提供运行入口。下游模型由各自模块训练；事件库的小样本清单不替代完整 Test 评测。验证代码：`pytest -q process_highD/tests`。

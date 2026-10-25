# 专利工程实例附图

绘制日期：2026年10月24日；交付格式更新：2026年10月25日。图片由已有数值结果整理生成，未新增模型采样、仿真、训练或重放。图内保留中文坐标、必要图例及物理时刻，不嵌入解释段落；用途与来源在本文件说明。

每图提供300 dpi PNG。

## 已绘图片

### E01_natural_event：自然驾驶事件与关键风险时刻

[PNG](E01_natural_event.png)

从存在严格切入、至少三个有效背景位置且未跟踪车辆未超过已跟踪峰值风险的片段中，按事件风险降序选取首例；道路视图按各时刻主车纵向位置平移，保持实际空间比例。risk_trace是截至当前时刻的峰值加暴露项，不是瞬时风险；虚线时刻取原始事件表的瞬时风险峰值。

数据身份：`{"event_id": "nat_25_001243_012870", "row": 31209, "focus_slot": 1, "frames": [0, 13, 75, 149], "event_risk": 1.0443952083587646, "instant_peak_frame": 13}`。

### E02_reference_execution：稀疏结点、参考生成与实际执行

[PNG](E02_reference_execution.png)

案例0，背景位置1。参考来自既有真实结点条件下的确定性扩散计划；下排为各轨迹相对插值基准的位置差，不是预测误差。

数据身份：`{"case": 0, "event_id": "nat_04_000417_009582", "actor": 1, "seed": 20261015, "knots_s": [0.0, 2.0, 4.0, 5.96]}`。

### E03_natural_replay：自然回放的轨迹保持

[PNG](E03_natural_replay.png)

左图为案例0的背景车逐时平均位移误差，主车不参与；右图为有背景车场景的场景平均误差经验分布，响应臂先在同场景内平均随机实现。自然报告包含原全量运行和后续回归复用，不属于重新盲测。

数据身份：`{"scene_count": 9841, "example_case": 0}`。

### E04_reference_retention：无干预与自由行驶下的参考保持

[PNG](E04_reference_retention.png)

左列为无干预案例0，右列为自由行驶案例50；每列分别比较同一刺激下的参考跟踪与响应执行，所选关注车辆均未接管。两列不是相同初态的配对样本。

数据身份：`{"cases": [0, 50], "seed": 20261015}`。

### E05_braking_response：主车制动后的跟驰响应

[PNG](E05_braking_response.png)

道路视图为响应臂实际状态，虚线包围框和轨迹为同一刺激下参考跟踪臂的关注车辆；下方为真实执行曲线。主车受脚本控制。首次碰撞时刻按碰撞后状态索引对应的物理时刻标记。

数据身份：`{"case": 1, "event_id": "nat_04_000417_009582", "actor": 1, "seed": 20261015, "handover_s": 3.84, "onset_s": 1.0, "any_collision": false, "ego_collision": false, "first_collision_s": null, "snapshot_frames": [86, 114]}`。

### E06_leader_release：前车离开原车道后的背景响应

[PNG](E06_leader_release.png)

道路视图为响应臂实际状态，虚线包围框和轨迹为同一刺激下参考跟踪臂的关注车辆；下方为真实执行曲线。主车受脚本控制。首次碰撞时刻按碰撞后状态索引对应的物理时刻标记。

数据身份：`{"case": 3, "event_id": "nat_04_000417_009582", "actor": 1, "seed": 20261015, "handover_s": 2.2800000000000002, "onset_s": 1.0, "any_collision": true, "ego_collision": true, "first_collision_s": 2.48, "snapshot_frames": [47, 75]}`。

### E07_lateral_response：目标车道车辆的交互响应

[PNG](E07_lateral_response.png)

道路视图为响应臂实际状态，虚线包围框和轨迹为同一刺激下参考跟踪臂的关注车辆；下方为真实执行曲线。主车受脚本控制。首次碰撞时刻按碰撞后状态索引对应的物理时刻标记。

数据身份：`{"case": 3, "event_id": "nat_04_000417_009582", "actor": 5, "seed": 20261015, "handover_s": 1.2, "onset_s": 1.0, "any_collision": true, "ego_collision": true, "first_collision_s": 2.48, "snapshot_frames": [20, 48]}`。

### E08_maneuver_handover：已开始换道的接管与机动承接

[PNG](E08_maneuver_handover.png)

案例28，关注背景位置1；接管记录在2.60秒记载换道已开始，进度0.854。曲线显示实际位置、速度及控制动作，保留该回合后续碰撞时刻；未补画未保存的连续进度或重置换道对照。

数据身份：`{"case": 28, "seed": 20261015, "event_id": "nat_21_000186_003789", "actor": 1, "handover_s": 2.6, "logged_progress": 0.8540358543395996, "first_collision_s": 5.5200000000000005}`。

### E09_handover_timing：接管状态与实际运动改变的逐步记录

[PNG](E09_handover_timing.png)

案例1，关注背景位置1；横轴为控制器调用时标。运动改变标记检查上一物理步已执行动作，两个标记的首次记录索引差不解释为物理响应延迟。

数据身份：`{"case": 1, "actor": 1, "seed": 20261015, "first_owned_index": 96, "first_motion_changed_index": 97}`。

### E11_interaction_sources：实际交互类别与抽样来源

[PNG](E11_interaction_sources.png)

使用已完成开发搜索的5000条执行记录，按实例哈希去重；左图为全部去重实例类别，右图按实例数选取前12个来源，同行来源跨类别统一编号。类别中的“接管”是控制权状态。此图为观察计数，不是联合名额分配或完整失效域识别结果。

数据身份：`{"raw_records": 5000, "unique_instances": 5000, "source_ranking": "instance_count_desc_then_root_id", "root_count": 3000}`。

### E14_runtime：不同背景车辆数量下的执行耗时

[PNG](E14_runtime.png)

分别绘制桥接与完整世界更新分位统计。各配置2831步，共8493步；桥接计时包含于世界更新，不相加。不包含ADS计算，40毫秒线为物理步长。

数据身份：`{"physical_steps": 8493, "ADS_compute_included": false, "physical_step_ms": 40}`。

## 跳过项目

- **E10**：已有一般生成搜索父子记录，但固定自然事件及非选中输入一致性证据尚未完成，不绘制固定事件风险配对图。

- **E12**：所核对搜索档案保留输入及执行摘要，缺少用于该图的原始完整逐帧闭环轨迹；按用户要求跳过，不启动重放。

- **E13**：缺少原运行与重放的成对轨迹证据；跳过，不启动重放。

- **E15**：跨类别联合名额分配尚无完整实际分配与后续调用日志。

- **E16**：尚无本案完整联合流程的同预算比较和消融结果。


## 来源与复现

完整来源、数据身份及取材范围见[manifest.json](manifest.json)，成图数据见[data/](data/)，绘图程序见[scripts/draw_engineering_patent.py](scripts/draw_engineering_patent.py)。程序只读取既有数组与报告，不调用模型或仿真器。

对应根目录[图片需求](../../../专利工程实例附图需求.md)。未将一般生成搜索的组件结果改标为本案联合流程的已完成效果。

# highD 共享训练数据

本目录只保留当前方法仍使用的场景条件、真实时序和 NPC 驾驶数据。它们用于训练、标定与自然场景评测；在线桥接读取已拟合的经验轨迹库，不读取全量误差窗口。

| 保留内容 | 当前用途 |
| --- | --- |
| `scenario_conditions/` | Flow 训练条件，以及与 Diffusion／世界模型对齐的场景身份、初始状态、车辆尺寸和位置结点。 |
| `highd_sequence_cache/sequence_cache/` | Diffusion 训练、世界模型自然 Test、IDM／DQN 与 TrafficBots 使用的状态、有效性、动作、地图和车辆尺寸。 |
| `driver_behavior/recordings/NN.npz` | 全部 60 段录像提取的跟驰、自由驾驶和换道机会；用于 IDM 人群、MOBIL 与条件行为评测。 |
| `driver_behavior/interaction_response/` | 当前正净间距领车、侧向交互和观测横向趋势协议下的纵向响应输入。 |
| `driver_behavior/` 顶层文件 | Train／Val／Test 汇总事件与完整换道轨迹、道路初态、数据清单、覆盖及标签协议。 |

场景条件与时序包含 96,055 个片段，按 `segment_id` 一一对应。42／9／9 段录像对应 72,771／13,133／10,151 个 Train／Val／Test 片段，其中 EVT 尾部共 2,964 条。时序中的历史前缀属于当前数组索引契约，不能直接裁剪。

有效车辆保留实际长宽；缺失槽位尺寸在条件数据中为零，在时序缓存中为 4.8 m × 1.8 m 占位，读取时必须结合有效性标记。`driver_behavior/manifest.json` 与 `interaction_response/manifest.json` 还用于冻结模型的数据身份校验，应保留原文件及时间戳。

场景条件与时序来自 `results/evt/natural_segments.csv`，生成入口见 [process_highD](../../process_highD/README.md)。NPC 数据直接来自完整 highD 原录像，不局限于七车短片段，由 `hierarchical_world_model.src.stochastic_drivers.raw_data.prepare_raw_data()` 与当前响应数据提取器生成。每段录像的事件与轨迹临时分片在三个 split 完整汇总并保存报告后删除，保留 `NN.npz`。

已删除旧预测包络 `plan_forecast_errors/`、旧横向匀速误差 `empirical_pose_errors/`、旧 NPC 输入 `driver_behavior/physical_response/`，以及已经完成标定的 `interaction_pose_errors/` 全量中间窗口。当前 256 条代表轨迹、样本权重、来源索引及全量统计仍保存在[世界模型结果](../hierarchical_world_model/README.md)中。删除前已逐条核对代表轨迹与原窗口完全一致。

重新运行 `hierarchical_world_model/scripts/calibrate_error_trajectories.py` 会从原始录像和冻结 Train 计划重建误差窗口，临时放在 `results/hierarchical_world_model/bridge/.cache/error_trajectories/`；模型与报告成功保存后自动清理，失败时保留中间数据供续跑。完整重标定仍会暂时需要约 34 GB 磁盘空间，日常仿真与评测不需要保留这部分。

清理范围、体积及验证记录见[整理记录中的 shared_training_data_cleanup](../hierarchical_world_model/organization.json)。本次不改变数据划分、提取规则、模型权重或已完成的评测结果。

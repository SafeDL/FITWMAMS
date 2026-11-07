# highD 共享训练数据

本目录只保留当前方法仍使用的场景条件、真实时序和 NPC 驾驶数据。它们用于训练、标定与自然场景评测；在线桥接读取已拟合的经验轨迹库，不读取全量误差窗口。

| 保留内容 | 当前用途 |
| --- | --- |
| `scenario_conditions/` | Flow 训练条件，以及与 Diffusion／世界模型对齐的场景身份、初始状态、车辆尺寸和位置结点。 |
| `highd_sequence_cache/sequence_cache/` | Diffusion 训练、世界模型自然 Test、IDM 诊断、ADS 与 TrafficBots 使用的状态、有效性、动作、地图和车辆尺寸。 |
| `driver_behavior/recordings/NN.npz` | 全部 60 段录像提取的跟驰、自由驾驶和换道机会；用于 IDM 人群、MOBIL 与条件行为评测。 |
| `driver_behavior/interaction_response/` | 当前正净间距领车、侧向交互和观测横向趋势协议下的纵向响应输入。 |
| `driver_behavior/` 顶层文件 | Train／Val／Test 汇总事件与完整换道轨迹、道路初态、数据清单、覆盖及标签协议。 |

场景条件与时序包含 96,055 个片段，按 `segment_id` 一一对应。42／9／9 段录像对应 72,771／13,133／10,151 个 Train／Val／Test 片段，其中 EVT 尾部共 2,964 条。时序中的历史前缀属于当前数组索引契约，不能直接裁剪。

有效车辆保留实际长宽；缺失槽位尺寸在条件数据中为零，在时序缓存中为 4.8 m × 1.8 m 占位，读取时必须结合有效性标记。`driver_behavior/manifest.json` 与 `interaction_response/manifest.json` 还用于冻结模型的数据身份校验，应保留原文件及时间戳。

场景条件与时序来自 `results/evt/natural_segments.csv`，生成入口见 [process_highD](../../process_highD/README.md)。NPC 数据直接来自完整 highD 原录像，不局限于七车短片段，由 `hierarchical_world_model.src.stochastic_drivers.raw_data.prepare_raw_data()` 与当前响应数据提取器生成。每段录像的事件与轨迹临时分片在三个 split 完整汇总并保存报告后删除，保留 `NN.npz`。

误差库重标定会从原始录像和冻结 Train 计划重建中间窗口，临时保存到 `results/hierarchical_world_model/bridge/.cache/error_trajectories/`，模型和报告成功写入后清理。日常仿真使用发布的经验误差库，不需要这些全量临时窗口。
